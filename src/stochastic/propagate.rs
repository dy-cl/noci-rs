// stochastic/propagate.rs
use rand::rngs::SmallRng;
use rand::SeedableRng;
use mpi::topology::Communicator;
use mpi::collective::SystemOperation;
use mpi::traits::*;
use mpi::datatype::{Partition, PartitionMut};
use rayon::prelude::*;

use crate::input::{Input, Propagator};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::noci::{DetPair, NOCIData};
use crate::time_call;
use super::state::{MCState, PopulationUpdate, ExcitationHist, ProjectedEnergyUpdate, PropagationState, MPIScratch, 
                   QMCRunInfo, ScratchSize, Shifts, PropagationResult, ThreadPropagation, PopulationStats};

use crate::noci::{calculate_s_pair, calculate_hs_pair};
use super::report::{print_row, print_header, print_cached_row, print_initial_row, check_stop};
use super::init::{initialise_qmc_state, max_scratch_sizes};
use super::state::{owner};

/// Find overlap matrix element S_{ij}.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `i`: Index of state `i`.
/// - `j`: Index of state `j`.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element `S_{ij}`.
pub(in crate::stochastic) fn find_s(data: &NOCIData<'_>, i: usize, j: usize, scratch: &mut WickScratchSpin) -> f64 {
    // Get the sorted pair of indices 
    let (a, b) = if i <= j {(i, j)} else {(j, i)};
    calculate_s_pair(data, DetPair::new(&data.basis[a], &data.basis[b]), Some(scratch))
}

/// Find Hamiltonian and overlap matrix elements H_{ij} and S_{ij}.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `i`: Index of state `i`.
/// - `j`: Index of state `j`.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `(f64, f64)`: Hamiltonian and overlap matrix elements `H_{ij}` and `S_{ij}`.
pub(in crate::stochastic) fn find_hs(data: &NOCIData<'_>, i: usize, j: usize, scratch: &mut WickScratchSpin) -> (f64, f64) {
    // Get the sorted pair of indices 
    let (a, b) = if i <= j {(i, j)} else {(j, i)};
    calculate_hs_pair(data, DetPair::new(&data.basis[a], &data.basis[b]), Some(scratch))
}

/// Accumulate the population change dn for a determinant i into the per-iteration delta vector.
/// Note that the actual populations stored in mc.walkers are not yet changed here.
/// # Arguments:
/// - `mc`: Contains information about the current Monte Carlo state.
/// - `i`: Index of determinant i to be updated.
/// - `dn`: Population change on determinant i. 
/// # Returns
/// - `()`: Updates the accumulated population changes in place.
fn add_delta(mc: &mut MCState, i: usize, dn: i64) {
    if dn == 0 {return;}
    // If current delta for this determinant is zero this function being called is its first
    // modification for this iteration and so we record this fact in changed.
    if mc.delta[i] == 0 {
        mc.changed.push(i);
    }
    // Add the population change to delta.
    mc.delta[i] += dn;
}

/// Apply accumulated population changes from the mc.delta vector to the actual populations stored
/// in mc.walkers.
/// # Arguments:
/// - `mc`: Contains information about the current Monte Carlo state.
/// # Returns
/// - `Vec<PopulationUpdate>`: List of applied population updates for this iteration.
fn apply_delta(mc: &mut MCState) -> Vec<PopulationUpdate> {
    time_call!(crate::timers::stochastic::add_apply_delta, {
        // Fast exit path for no changes.
        if mc.changed.is_empty() {return Vec::new();}

        // Iterate over only determinants which have a changed population, read in the change
        // (`delta`), zero the `delta` and apply it to the walker population.
        let mut applied = Vec::with_capacity(mc.changed.len());
        for &i in &mc.changed {
            let dn = mc.delta[i];
            mc.delta[i] = 0;
            mc.walkers.add(i, dn);
            applied.push(PopulationUpdate {det: i as u64, dn});
        }

        mc.changed.clear();
        applied
    })
}

/// Communicate spawned walker updates between MPI ranks.
/// Remote spawn updates are stored locally as one `Vec<PopulationUpdate>` per destination rank.
/// This routine packs those per-destination buffers into one contiguous send buffer, exchanges the
/// number of updates each rank will send/receive, then performs one `MPI_Alltoallv`-style exchange
/// of the packed payloads.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `scratch`: Reusable MPI scratch space for counts, displacements, and contiguous send/recv buffers.
/// # Returns
/// - `&[PopulationUpdate]`: Flat buffer containing all spawned walker updates received from other ranks.
pub(crate) fn communicate_spawn_updates<'a>(world: &impl CommunicatorCollectives, scratch: &'a mut MPIScratch) -> &'a [PopulationUpdate] {
    time_call!(crate::timers::stochastic::add_communicate_spawn_updates, {
        let nsend = scratch.send_contig.len();
        let nranks = world.size() as usize;

        // Fast-path, if no rank has any remote updates to send, there is no need
        // to do the count exchange or the payload exchange this iteration.
        let localany = if nsend > 0 {1i32} else {0i32};
        let mut globalany = 0i32;
        world.all_reduce_into(&localany, &mut globalany, mpi::collective::SystemOperation::sum());
        if globalany == 0 {
            scratch.recv_contig.clear();
            return &scratch.recv_contig[..];
        }

        // Exchange only the per-rank message sizes.
        // After this, `recv_counts[peer]` contains how many updates rank `peer` will send to the
        // current rank.
        world.all_to_all_into(&scratch.send_counts[..], &mut scratch.recv_counts[..]);

        // Build the incoming MPI metadata.
        // `recv_displacements[peer]` is the starting offset of rank `peer`'s block inside the packed
        // contiguous receive buffer `recv_contig`.
        // `nrecv` is the total number of remote updates this rank will receive.
        let mut nrecv = 0usize;
        for peer in 0..nranks {
            scratch.recv_displacements[peer] = nrecv as i32;
            nrecv += scratch.recv_counts[peer] as usize;
        }

        // Reuse one contiguous send buffer large enough for all remote updates.
        scratch.recv_contig.clear();
        scratch.recv_contig.resize(nrecv, PopulationUpdate {det: 0, dn: 0});

        let send_part = Partition::new(&scratch.send_contig[..], &scratch.send_counts[..], &scratch.send_displacements[..]);
        let mut recv_part = PartitionMut::new(&mut scratch.recv_contig[..], &scratch.recv_counts[..], &scratch.recv_displacements[..]);
        world.all_to_all_varcount_into(&send_part, &mut recv_part);

        // Return the receive buffer for later accumulation.
        &scratch.recv_contig[..]
    })
}

/// Gather variable-length walker updates from all ranks into a reusable receive buffer.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `send`: Local walker updates to gather.
/// - `scratch`: Reusable MPI scratch space.
/// # Returns
/// - `&[PopulationUpdate]`: Global gathered walker updates.
pub(crate) fn gather_all_walkers<'a>(world: &impl Communicator, send: &[PopulationUpdate], scratch: &'a mut MPIScratch) -> &'a [PopulationUpdate] {
    time_call!(crate::timers::stochastic::add_gather_all_walkers, {
        let nsend = send.len() as i32;
        world.all_gather_into(&nsend, &mut scratch.gather_counts[..]);

        let mut ntot = 0usize;
        for (i, &n) in scratch.gather_counts.iter().enumerate() {
            scratch.gather_displs[i] = ntot as i32;
            ntot += n as usize;
        }

        if ntot == 0 {
            scratch.gather_recv.clear();
            return &scratch.gather_recv[..];
        }

        scratch.gather_recv.resize(ntot, PopulationUpdate {det: 0, dn: 0});
        let mut recv = PartitionMut::new(&mut scratch.gather_recv[..], &scratch.gather_counts[..], &scratch.gather_displs[..]);
        world.all_gather_varcount_into(send, &mut recv);
        &scratch.gather_recv[..]
    })
}

/// Compute the off-diagonal coupling between determinants depending on which propagator is used.
/// # Arguments:
/// - `hlg`: Matrix element H_{\Lambda\Gamma}
/// - `slg`: Matrix element S_{\Lambda\Gamma}
/// - `es_s`: E_s^S(\tau) shift energy.
/// - `es`: E_s(\tau) shift energy.
/// - `prop`: Chosen propagator.
/// # Returns
/// - `f64`: Off-diagonal coupling used by the propagator.
pub(in crate::stochastic) fn coupling(hlg: f64, slg: f64, es_s: f64, es: f64, prop: &Propagator) -> f64 {
    match prop {
        Propagator::Unshifted => hlg - es_s * slg,
        Propagator::Shifted => hlg - es_s * slg,
        Propagator::DoublyShifted => hlg - es_s * slg,
        Propagator::DifferenceDoublyShiftedU1 => hlg - 0.5 * (es + es_s) * slg,
        Propagator::DifferenceDoublyShiftedU2 => hlg - es_s * slg,
    }
}

/// Compute the local projected-energy numerator and denominator increments.
/// # Arguments:
/// - `d`: Applied determinant population updates for this iteration.
/// - `iref`: Reference determinant index.
/// - `data`: Immutable stochastic propagation data.
/// # Returns
/// - `(f64, f64)`: Local projected-energy numerator and denominator increments.
fn projected_energy_local(d: &[PopulationUpdate], iref: usize, data: &NOCIData<'_>) -> (f64, f64) {
    // For each updated determinant `\Gamma` accumulate `dN_\Gamma H_{\text{ref}, \Gamma}` into the
    // numerator and `dN_\Gamma S_{\text{ref}, \Gamma}` into the denominator. 
    d.par_iter().map_init(WickScratchSpin::new, |scratch, up| {
        let gamma = up.det as usize;
        let dn = up.dn as f64;
        (dn * find_hs(data, iref, gamma, scratch).0, dn * find_s(data, iref, gamma, scratch))
    }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
}

/// Add the contribution from a sparse set of determinant updates to the local `p_{\Gamma}` block.
/// # Arguments:
/// - `plocal`: Local portion of `p_{\Gamma}` on this rank.
/// - `updates`: Sparse determinant population updates to apply.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// # Returns
/// - `()`: Adds the contribution from `updates` into `plocal`.
fn add_plocal(plocal: &mut [f64], updates: &[PopulationUpdate], data: &NOCIData<'_>, run: &QMCRunInfo) {
    // Fast exit.
    if updates.is_empty() {return;}
    
    // Parallel iteration over rank local elements of `p_\Gamma`, for each local owned determinant
    // `\Gamma` we accumulate `\sum_\Omega S_{\Gamma, \Omega} dN_\Omega` over the updates.
    plocal.par_iter_mut().enumerate().for_each_init(WickScratchSpin::new, |scratch, (k, pgamma)| {
        let gamma = run.start + k;
        let mut dp = 0.0;
        for up in updates {
            let omega = up.det as usize;
            dp += find_s(data, gamma, omega, scratch) * up.dn as f64;
        }
        *pgamma += dp;
    });
}

/// Update the vector `p_{\Gamma} = \sum_\Omega S_{\Gamma,\Omega} N_{\Omega}`.
/// # Arguments:
/// - `plocal`: Local portion of `p_{\Gamma}` on this rank.
/// - `dlocal`: Local determinant population updates.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// - `mpi`: Reusable MPI scratch space.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `bool`: `true` if any determinant populations changed globally in this iteration.
fn update_p(plocal: &mut [f64], dlocal: &[PopulationUpdate], data: &NOCIData<'_>, run: &QMCRunInfo, mpi: &mut MPIScratch, world: &impl CommunicatorCollectives) -> bool {
    time_call!(crate::timers::stochastic::add_update_p, {

        // Gather number of updates that each rank will send to this rank.
        let nsend = dlocal.len() as i32;
        time_call!(crate::timers::stochastic::add_update_p_gather_counts, {
            world.all_gather_into(&nsend, &mut mpi.gather_counts[..]);
        });
        
        // Calculate displacements for the recieve buffer and the total number of updates across
        // all ranks.
        let mut ntot = 0usize;
        for (i, &n) in mpi.gather_counts.iter().enumerate() {
            mpi.gather_displs[i] = ntot as i32;
            ntot += n as usize;
        }
        if ntot == 0 {return false;}

        // Size recieve buffer to hold all updates from all ranks and calculate slice boundaries
        // that correspond to this rank's own constribution in the gathered buffer. These will
        // later be skipped during the apply step to avoid double counting.
        mpi.gather_recv.resize(ntot, PopulationUpdate {det: 0, dn: 0});
        let locallow = mpi.gather_displs[run.irank] as usize;
        let localhigh = locallow + mpi.gather_counts[run.irank] as usize;
        
        let mut recv = PartitionMut::new(&mut mpi.gather_recv[..], &mpi.gather_counts[..], &mpi.gather_displs[..]);
        mpi::request::scope(|scope| {
            // Perform non-blocking all-gather such that the local computation can be overlapped.
            let req = world.immediate_all_gather_varcount_into(scope, dlocal, &mut recv);
            
            // Apply this ranks own updates to `plocal` whilst the all-gather is occuring.
            time_call!(crate::timers::stochastic::add_update_p_local_overlap, {
                add_plocal(plocal, dlocal, data, run);
            });
            
            // Now we must wait for the all gather to finish before we read the remote updates.
            time_call!(crate::timers::stochastic::add_update_p_wait, {
                req.wait();
            })
        });
        
        // Apply remote updates from all other ranks to `plocal`. We avoid double counting by
        // excluding the local slice.
        time_call!(crate::timers::stochastic::add_update_p_apply, {
            add_plocal(plocal, &mpi.gather_recv[..locallow], data, run);
            add_plocal(plocal, &mpi.gather_recv[localhigh..], data, run);
        });
        true
    })
}

/// Accumulate all local population updates into the global delta vector, update the local
/// excitation histogram if requested, and pack remote updates into per-rank MPI send buffers.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `prop`: Population updates generated in the spawning and death/cloning step.
/// - `input`: User specified input options.
/// - `nranks`: Total number of MPI ranks.
/// - `ndets`: Total number of determinants.
/// - `scratch`: Reusable MPI communication scratch.
/// # Returns
/// - `()`: Updates the local delta vector, optional excitation histogram, and MPI send buffers.
fn acc_pack_updates(mc: &mut MCState, prop: PropagationResult, input: &Input, ndets: usize, nranks: usize, scratch: &mut MPIScratch) {
    time_call!(crate::timers::stochastic::add_acc_pack_updates, {
        // Add the `delta` for local determinants.
        for (det, dn) in prop.local {
            add_delta(mc, det, dn);
        }
        
        // Optionally record excitation generation data.
        if input.write.write_excitation_hist && let Some(hist) = mc.excitation_hist.as_mut() {
            for p in prop.samples {
                hist.add(p);
            }
        }

        pack_spawn_updates(&prop.remote, ndets, nranks, scratch);
    })
}

/// Accumulate thread-local updates into the global delta vector and exchange any remote
/// updates across MPI ranks.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `prop`: Population updates generated in the spawning and death/cloning step.
/// - `input`: User specified input options.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `nranks`: Total number of MPI ranks.
/// - `ndets`: Total number of determinants.
/// - `mpi`: Reusable MPI communication scratch.
/// # Returns
/// - `()`: Updates `mc.delta` in place.
fn accumulate_updates(mc: &mut MCState, prop: PropagationResult, input: &Input, mpi: &mut MPIScratch, world: &impl CommunicatorCollectives, ndets: usize, nranks: usize) {
    // Sort updates into local and remote events based on determinant ownership.
    acc_pack_updates(mc, prop, input, ndets, nranks, mpi);
    
    // Exchange updates with all other ranks.
    if nranks > 1 {
        let recv = communicate_spawn_updates(world, mpi);
        for &up in recv {
            add_delta(mc, up.det as usize, up.dn);
        }
    }
}

/// Sort and combine repeated remote population updates in place.
/// # Arguments:
/// - `updates`: Remote population updates for a single destination rank.
/// # Returns
/// - `()`: Compresses repeated determinant updates in place.
fn compress_updates(updates: &mut Vec<PopulationUpdate>) {
    // Fast return.
    if updates.is_empty() {return;}

    // Sort by determinant index so duplicate entries are consecutive.
    updates.sort_unstable_by_key(|up| up.det);
    
    // For each element if the previous entry is the same determinant accumulate 
    // its `dn` into that entry, otherwise write it as a new entry.
    let mut out = 0usize;
    for i in 0..updates.len() {
        if out > 0 && updates[out - 1].det == updates[i].det {
            updates[out - 1].dn += updates[i].dn;
        } else {
            updates[out] = updates[i];
            out += 1;
        }
    }
    
    // Remove empty and zero elements and return.
    updates.truncate(out);
    updates.retain(|up| up.dn != 0);
}

/// Pack remote spawned walker updates into a contiguous buffer grouped by destination rank.
/// # Arguments:
/// - `remote`: Remote spawned walker updates.
/// - `ndets`: Number of determinants in the stochastic basis.
/// - `nranks`: Number of MPI ranks.
/// - `scratch`: Reusable MPI scratch space.
/// # Returns
/// - `()`: Fills `send_counts`, `send_displacements`, and `send_contig` in `scratch`.
fn pack_spawn_updates(remote: &[PopulationUpdate], ndets: usize, nranks: usize, scratch: &mut MPIScratch) {
    // Count how many updates will be sent to each destination rank.
    for x in scratch.send_counts.iter_mut() {*x = 0;}
    for up in remote {
        let peer = owner(up.det as usize, ndets, nranks);
        scratch.send_counts[peer] += 1;
    }

    // Build the starting offset of each rank's block in the contiguous send buffer.
    let mut nsend = 0usize;
    for peer in 0..nranks {
        scratch.send_displacements[peer] = nsend as i32;
        nsend += scratch.send_counts[peer] as usize;
    }

    // Reuse a contiguous send buffer large enough for all remote updates.
    scratch.send_contig.clear();
    scratch.send_contig.resize(nsend, PopulationUpdate {det: 0, dn: 0});

    // Initialise the current write positions for each destination rank.
    scratch.send_write_pos.clone_from(&scratch.send_displacements);

    // Pack updates into `send_contig` so that each destination rank occupies one contiguous block.
    for &up in remote {
        let peer = owner(up.det as usize, ndets, nranks);
        let pos = scratch.send_write_pos[peer] as usize;
        scratch.send_contig[pos] = up;
        scratch.send_write_pos[peer] += 1;
    }
}

/// Perform spawning and death/cloning steps over the currently occupied determinants.
/// # Arguments:
/// - `it`: Current iteration number.
/// - `mc`: Contains the current Monte Carlo state.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// - `scratchsize`: Maximum sizes required for per-thread Wick scratch space.
/// - `shifts`: Current non-overlap and overlap-transformed shifts.
/// # Returns
/// - `PropagationResult`: Local and remote population updates together with spawning probability samples.
fn propagate_iteration(it: usize, mc: &MCState, data: &NOCIData<'_>, run: &QMCRunInfo, scratchsize: &ScratchSize, shifts: Shifts) -> PropagationResult {
    time_call!(crate::timers::stochastic::add_propagate_iteration, {
        
        // Closure to initialise per Rayon thread state. Each thread gets individual RNG seed and
        // scratch space for non-orthogonal Wick's theorem calculations.
        let initialise = || -> ThreadPropagation {
            let tid = rayon::current_thread_index().unwrap_or(0) as u64;
            ThreadPropagation{
                local: Vec::new(),
                remote: Vec::new(),
                samples: Vec::new(),
                rng: SmallRng::seed_from_u64(run.rank_seed ^ tid ^ ((it as u64).wrapping_mul(0x9E3779B97F4A7C15))),
                scratch: WickScratchSpin::with_sizes(scratchsize.maxsame, scratchsize.maxla, scratchsize.maxlb),
            }
        };
        
        // Closure to carry out the spawning, death and cloning propagation steps. Exits early if a
        // determinant has zero population. 
        let propagate = |mut acc: ThreadPropagation, &gamma: &usize| -> ThreadPropagation {
            let ngamma = mc.walkers.get(gamma);
            if ngamma == 0 {
                return acc;
            }

            acc.death_cloning(gamma, ngamma, shifts, data);
            acc.spawning(gamma, ngamma, shifts, data, run);
            acc
        };
        
        // Closure to combine two thread local accumulators by appending update vectors.
        let merge = |mut a: ThreadPropagation, mut b: ThreadPropagation| -> ThreadPropagation {
            a.local.append(&mut b.local);
            a.remote.append(&mut b.remote);
            a.samples.append(&mut b.samples);
            a
        };
    
        // Carry out the above closures via parallel fold over occupied determinants.
        let acc = mc.walkers.occ().par_iter().fold(initialise, propagate).reduce(
            || ThreadPropagation {
                local: Vec::new(),
                remote: Vec::new(),
                samples: Vec::new(),
                rng: SmallRng::seed_from_u64(0),
                scratch: WickScratchSpin::new(),
            },
            merge,
        );

        PropagationResult {
            local: acc.local,
            remote: acc.remote,
            samples: acc.samples,
        }
    })
}

/// Compute the local non-overlap and overlap-transformed population statistics.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `isref`: Boolean mask specifying which determinants are reference determinants.
/// - `run`: Rank-local run metadata.
/// # Returns
/// - `([i64; 3], [f64; 2])`: Local integer and floating-point population statistics.
fn population_local(mc: &MCState, isref: &[bool], run: &QMCRunInfo) -> ([i64; 3], [f64; 2]) {

    // Calculate non-overlap transformed population on the reference determinants.
    // Iterates over occupied determinants keeping those flagged as a reference and summing their populations.
    let nrefclocal: i64 = mc.walkers.occ().iter()
        .filter(|&&det| isref[det])
        .map(|&det| mc.walkers.get(det).abs())
        .sum();
    
    // Calculate overlap transformed population on the reference determinants.
    // `mc.pg` is the vector p_\Gamma = \sum_\Omega S_{\Gamma, \Omega} N_\Omega, so we can get this
    // population by iterating over p_\Gamma and summing entries which are marked as corresponding
    // to a reference determinant.
    let nrefsclocal: f64 = mc.pg.iter().enumerate()
        .filter(|(k, _)| isref[run.start + *k])
        .map(|(_, x)| x.abs())
        .sum();
    
    // Pack and return local quantities.
    let intslocal = [mc.walkers.norm(), nrefclocal, mc.walkers.occ().len() as i64];
    let floatslocal = [mc.pg.iter().map(|x| x.abs()).sum::<f64>(), nrefsclocal];
    (intslocal, floatslocal)
}

/// Update projected-energy accumulators and compute current population statistics.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `d`: Applied determinant population updates for this iteration.
/// - `pe`: Current projected-energy accumulators.
/// - `isref`: Boolean mask specifying which determinants are reference determinants.
/// - `run`: Rank-local run metadata.
/// - `data`: Immutable stochastic propagation data.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `PopulationStats`: Current total and reference populations in both representations.
fn update_observables(mc: &MCState, d: &[PopulationUpdate], pe: &mut ProjectedEnergyUpdate, isref: &[bool], 
                      run: &QMCRunInfo, data: &NOCIData<'_>, world: &impl Communicator) -> PopulationStats {
    // Compute rank local number and denominator of the projected energy.
    let (dnumlocal, ddenlocal) = time_call!(crate::timers::stochastic::add_update_projected_energy, {
        projected_energy_local(d, pe.iref, data)
    });
    // Compute rank local population data.
    let (intslocal, popfloatslocal) = time_call!(crate::timers::stochastic::add_compute_populations, {
        population_local(mc, isref, run)
    });
    
    // Pack all seven of the rank local observables into an array for a single all reduce.
    let obslocal = [dnumlocal, ddenlocal, popfloatslocal[0], popfloatslocal[1], intslocal[0] as f64, intslocal[1] as f64, intslocal[2] as f64];
    let mut obsglobal = [0.0_f64; 7];
    // Perform the all reduce.
    time_call!(crate::timers::stochastic::add_observables_allreduce, {
        world.all_reduce_into(&obslocal[..], &mut obsglobal[..], SystemOperation::sum());
    });

    // Accumlate the changes in projected energy numerator and denominator.
    pe.num += obsglobal[0];
    pe.den += obsglobal[1];

    PopulationStats {
        nwc: obsglobal[4] as i64,
        nrefc: obsglobal[5] as i64,
        noccdets: obsglobal[6] as i64,
        nwsc: obsglobal[2],
        nrefsc: obsglobal[3],
    }
}

/// Cache the latest population statistics inside the propagation state for later
/// printing and possible early exiting.
/// # Arguments:
/// - `state`: Propagation state containing QMC stats.
/// - `stats`: Population statistics computed for the current iteration.
/// # Returns
/// - `()`: Updates cached population values in `state` in place.
fn cache_population_stats(state: &mut PropagationState, stats: &PopulationStats) {
    state.cur_pop.nwc = stats.nwc;
    state.cur_pop.nrefc = stats.nrefc;
    state.cur_pop.nwsc = stats.nwsc;
    state.cur_pop.nrefsc = stats.nrefsc;
    state.cur_pop.noccdets = stats.noccdets;
}

/// Update the shift energies according to the current walker populations and shift.
/// # Arguments:
/// - `stats`: Population statistics computed for the current iteration.
/// - `state`: Propagation state containing QMC stats.
/// - `es`: Non-overlap transformed shift energy.
/// - `input`: User specified input options.
/// # Returns
/// - `()`: Updates the shift energies and associated state variables in place.
fn update_shifts(stats: &PopulationStats, state: &mut PropagationState, es: &mut f64, input: &Input) {
    let qmc = input.qmc.as_ref().unwrap();
    let dteff = input.prop_ref().dt * (qmc.ncycles as f64);
    
    // Once non-overlap transformed population reaches the target set flag.
    if !state.reached_c && stats.nwc >= qmc.target_population {
        state.reached_c = true;
    }
    // Once overlap transformed population reaches the target set flag.
    if !state.reached_sc && stats.nwsc >= qmc.target_population as f64 {
        state.reached_sc = true;
    }
    
    // Update non-overlap transformed shift.
    if state.reached_c {
        *es -= (qmc.shift_damping / dteff) * (stats.nwc as f64 / state.prev_pop.nwc as f64).ln();
    }
    // Update overlap transformed shift.
    if state.reached_sc {
        state.es_s -= (qmc.shift_damping / dteff) * (stats.nwsc / state.prev_pop.nwsc).ln();
    }

    state.prev_pop = *stats;
}

/// Propagate according to the stochastic update equations with a two-level loop. Iterations over
/// over `nreports` outer report loops each containing `ncycles` Monte Carlo iterations. Updates of
/// the `p_{\Gamma}` vector, projected energy, populations and shifts occur once report report loop.  
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: Initial determinant coefficient vector to be translated into walker populations.
/// - `es`: Non-overlap transformed shift energy.
/// - `ref_indices`: Indices of the reference determinants in the full basis.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `(f64, Option<ExcitationHist>)`: Final projected energy estimate and optional excitation histogram.
pub fn qmc_step(data: &NOCIData<'_>, c0: &[f64], es: &mut f64, ref_indices: &[usize], world: &impl Communicator) -> (f64, Option<ExcitationHist>) {
    let qmc = data.input.qmc.as_ref().unwrap();

    // Local MPI rank metadata.
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let ndets = data.basis.len();
    let start = (ndets * irank) / nranks;
    let end = (ndets * (irank + 1)) / nranks;

    // Mark reference determinants for projected-energy calculations.
    let mut isref = vec![false; ndets];
    for &i in ref_indices {
        isref[i] = true;
    }
    
    // Each MPI rank gets a unique RNG seed.
    let base_seed = qmc.seed.unwrap_or_else(rand::random);
    let rank_seed = base_seed.wrapping_add((irank as u64).wrapping_mul(0x9E3779B9));

    let run = QMCRunInfo {irank, nranks, ndets, start, end, iref: 0, base_seed, rank_seed};
    
    // Precompute largest possible size needed for the non-orthogonal Wick's theorem scratch space.
    let scratchsize = {
        let (maxsame, maxla, maxlb) = max_scratch_sizes(data.basis);
        ScratchSize {maxsame, maxla, maxlb}
    };

    // Thread local scratch for Wick's theorem and for MPI communicattion.
    let mut scratch = WickScratchSpin::new();
    let mut mpiscratch = MPIScratch::new(run.nranks);
    
    // Initialise walker populations, projected-energy accumulators and shifts.
    let mut state = initialise_qmc_state(c0, es, data, &run, &isref, world, &mut scratch, &mut mpiscratch);
    
    if irank == 0 {
        println!(
            "Size of Wick's Scratch (MiB): {}", 
            std::mem::size_of::<WickScratchSpin>() as f64 / (1024.0 * 1024.0)
        );
        type ThreadState = (Vec<(usize, i64)>, Vec<PopulationUpdate>, Vec<f64>, SmallRng, WickScratchSpin);
        println!(
            "Size of per thread state (MiB): {}", 
            std::mem::size_of::<ThreadState>() as f64 / (1024.0 * 1024.0)
        );
    }
    
    // Accumulate applied population changes over a report block.
    let mut pendingd: Vec<PopulationUpdate> = Vec::new();
    
    print_header(irank);
    print_initial_row(irank, &state, data.basis[0].e);
    
    // Outer loop over reports.
    for report in state.start_report..qmc.nreports {
        // Clear the determinant population changes vector for this report. 
        pendingd.clear();
        // Inner loop over cycles.
        for cycle in 0..qmc.ncycles {
            // Global iteration count.
            let iter = report * qmc.ncycles + cycle;
            
            // Current shifts used for all cycles in this report.
            let shifts = Shifts {es: *es, es_s: state.es_s};
            
            // Perform spawning, death, cloning and annhilation.
            let prop = propagate_iteration(iter, &state.mc, data, &run, &scratchsize, shifts);
            
            // Accumulate local updates, assemble the remote updates and send with MPI.
            accumulate_updates(&mut state.mc, prop, data.input, &mut mpiscratch, world, ndets, nranks);
            // Apply the changes in walker populations to each determinant.
            let d = apply_delta(&mut state.mc);
            // Store this cycles population changes.
            pendingd.extend_from_slice(&d);
        }
        
        // Last global iteration count for this report.
        let end = (report + 1) * qmc.ncycles - 1;

        compress_updates(&mut pendingd);

        // Update the vector `p_{\Gamma}`. 
        let changedglobal = update_p(&mut state.mc.pg, &pendingd, data, &run, &mut mpiscratch, world);
        
        // If populations haven't changed we can end the report here early.
        if !changedglobal {
            let stopshifts = Shifts {es: *es, es_s: state.es_s};
            if let Some(ret) = check_stop(report, &mut state, stopshifts, &run, world) {
                return ret;
            }
            print_cached_row(irank, end + 1, &state, data.basis[0].e, *es);
            continue;
        }
        
        // Update incremental projected-energy, populations and shifts.
        let stats = update_observables(&state.mc, &pendingd, &mut state.pe, &isref, &run, data, world);
        cache_population_stats(&mut state, &stats);
        update_shifts(&stats, &mut state, es, data.input);
        
        // Calculate projected energy.
        state.eprojcur = state.pe.num / state.pe.den;
        
        // Check if the calculation has been requested to stop.
        let stopshifts = Shifts {es: *es, es_s: state.es_s};
        if let Some(ret) = check_stop(report, &mut state, stopshifts, &run, world) {
            return ret;
        }

        print_row(irank, end + 1, &state, &stats, data.basis[0].e, *es);
    }
    (state.eprojcur, state.mc.excitation_hist)
}

