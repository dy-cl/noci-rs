// stochastic/propagate.rs
use rand::rngs::SmallRng;
use rand::SeedableRng;
use mpi::topology::Communicator;
use mpi::collective::SystemOperation;
use mpi::traits::*;
use mpi::datatype::PartitionMut;
use rayon::prelude::*;

use crate::input::{Input, Propagator};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::noci::{DetPair, NOCIData};
use crate::time_call;
use super::state::{MCState, PopulationUpdate, ExcitationHist, ProjectedEnergyUpdate, PropagationState, MPIScratch, 
                   QMCRunInfo, ScratchSize, Shifts, PropagationResult, ThreadPropagation, PopulationStats};

use crate::noci::{calculate_s_pair, calculate_hs_pair};
use crate::mpiutils::{owner, communicate_spawn_updates};
use super::report::{print_row, print_header, print_cached_row, print_initial_row, check_stop};
use super::init::{initialise_qmc_state, max_scratch_sizes};

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
        if mc.changed.is_empty() {return Vec::new();}
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
    if updates.is_empty() {return;}

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
        let nsend = dlocal.len() as i32;
        time_call!(crate::timers::stochastic::add_update_p_gather_counts, {
            world.all_gather_into(&nsend, &mut mpi.gather_counts[..]);
        });

        let mut ntot = 0usize;
        for (i, &n) in mpi.gather_counts.iter().enumerate() {
            mpi.gather_displs[i] = ntot as i32;
            ntot += n as usize;
        }

        if ntot == 0 {return false;}

        mpi.gather_recv.resize(ntot, PopulationUpdate {det: 0, dn: 0});

        let locallow = mpi.gather_displs[run.irank] as usize;
        let localhigh = locallow + mpi.gather_counts[run.irank] as usize;
        
        let mut recv = PartitionMut::new(&mut mpi.gather_recv[..], &mpi.gather_counts[..], &mpi.gather_displs[..]);
        mpi::request::scope(|scope| {
            let req = world.immediate_all_gather_varcount_into(scope, dlocal, &mut recv);
            
            time_call!(crate::timers::stochastic::add_update_p_local_overlap, {
                add_plocal(plocal, dlocal, data, run);
            });

            time_call!(crate::timers::stochastic::add_update_p_wait , {
                req.wait();
            })
        });

        time_call!(crate::timers::stochastic::add_update_p_apply, {
            add_plocal(plocal, &mpi.gather_recv[..locallow], data, run);
            add_plocal(plocal, &mpi.gather_recv[localhigh..], data, run);
        });
        true
    })
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

        let propagate = |mut acc: ThreadPropagation, &gamma: &usize| -> ThreadPropagation {
            let ngamma = mc.walkers.get(gamma);
            if ngamma == 0 {
                return acc;
            }

            acc.death_cloning(gamma, ngamma, shifts, data);
            acc.spawning(gamma, ngamma, shifts, data, run);
            acc
        };

        let merge = |mut a: ThreadPropagation, mut b: ThreadPropagation| -> ThreadPropagation {
            a.local.append(&mut b.local);
            a.remote.append(&mut b.remote);
            a.samples.append(&mut b.samples);
            a
        };

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

/// Accumulate all local population updates into the global delta vector, update the local
/// excitation histogram if requested, and pack remote updates into per-rank MPI send buffers.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `send`: Per-rank MPI send buffers.
/// - `prop`: Population updates generated in the spawning and death/cloning step.
/// - `input`: User specified input options.
/// - `nranks`: Total number of MPI ranks.
/// # Returns
/// - `()`: Updates the local delta vector, optional excitation histogram, and MPI send buffers.
fn acc_pack_updates(mc: &mut MCState, send: &mut [Vec<PopulationUpdate>], prop: PropagationResult, input: &Input, nranks: usize) {
    time_call!(crate::timers::stochastic::add_acc_pack_updates, {
        for (det, dn) in prop.local {
            add_delta(mc, det, dn);
        }

        if input.write.write_excitation_hist && let Some(hist) = mc.excitation_hist.as_mut() {
            for p in prop.samples {
                hist.add(p);
            }
        }

        if nranks > 1 {
            for up in prop.remote {
                let dest = owner(up.det as usize, mc.walkers.len(), nranks);
                send[dest].push(up);
            }
        }
    })
}

/// Exchange remote spawned walker updates between MPI ranks.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `send`: Per-rank MPI send buffers.
/// - `mpi`: Reusable MPI scratch space.
/// # Returns
/// - `&[PopulationUpdate]`: Walker updates received from remote ranks.
fn exchange_updates<'a>(world: &impl Communicator, send: &[Vec<PopulationUpdate>], mpi: &'a mut MPIScratch) -> &'a [PopulationUpdate] {
    time_call!(crate::timers::stochastic::add_exchange_updates, {
        communicate_spawn_updates(world, send, mpi)
    })
}

/// Unpack received walker updates into the global delta vector.
/// # Arguments:
/// - `mc`: Contains information about the current Monte Carlo state.
/// - `received`: Walker updates received from remote MPI ranks.
/// # Returns
/// - `()`: Adds received walker updates into `mc.delta`.
fn unpack_received_updates(mc: &mut MCState, received: &[PopulationUpdate]) {
    time_call!(crate::timers::stochastic::add_unpack_received_updates, {
        for up in received {
            add_delta(mc, up.det as usize, up.dn);
        }
    })
}

/// Accumulate thread-local updates into the global delta vector and exchange any remote
/// updates across MPI ranks.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `send`: Per-rank MPI send buffers.
/// - `prop`: Population updates generated in the spawning and death/cloning step.
/// - `input`: User specified input options.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `nranks`: Total number of MPI ranks.
/// # Returns
/// - `()`: Updates `mc.delta` in place.
fn accumulate_updates(mc: &mut MCState, send: &mut [Vec<PopulationUpdate>], prop: PropagationResult, 
                      input: &Input, mpi: &mut MPIScratch, world: &impl Communicator, nranks: usize) {
    
    for buf in send.iter_mut() {
        buf.clear();
    }

    acc_pack_updates(mc, send, prop, input, nranks);

    if nranks > 1 {
        for buf in send.iter_mut() {
            if buf.len() > 1 {
                compress_updates(buf);
            }
        }

        let received = exchange_updates(world, send, mpi);
        unpack_received_updates(mc, received);
    }
}

/// Sort and combine repeated remote population updates in place.
/// # Arguments:
/// - `updates`: Remote population updates for a single destination rank.
/// # Returns
/// - `()`: Compresses repeated determinant updates in place.
fn compress_updates(updates: &mut Vec<PopulationUpdate>) {
    if updates.is_empty() {return;}
    updates.sort_unstable_by_key(|up| up.det);

    let mut out = 0usize;
    for i in 0..updates.len() {
        if out > 0 && updates[out - 1].det == updates[i].det {
            updates[out - 1].dn += updates[i].dn;
        } else {
            updates[out] = updates[i];
            out += 1;
        }
    }

    updates.truncate(out);
    updates.retain(|up| up.dn != 0);
}

/// Compute the local non-overlap and overlap-transformed population statistics.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `isref`: Boolean mask specifying which determinants are reference determinants.
/// - `run`: Rank-local run metadata.
/// # Returns
/// - `([i64; 3], [f64; 2])`: Local integer and floating-point population statistics.
fn population_local(mc: &MCState, isref: &[bool], run: &QMCRunInfo) -> ([i64; 3], [f64; 2]) {
    let nrefclocal: i64 = mc.walkers.occ().iter().filter(|&&det| isref[det]).map(|&det| mc.walkers.get(det).abs()).sum();
    let nrefsclocal: f64 = mc.pg.iter().enumerate().filter(|(k, _)| isref[run.start + *k]).map(|(_, x)| x.abs()).sum();

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
    let (dnumlocal, ddenlocal) = time_call!(crate::timers::stochastic::add_update_projected_energy, {
        projected_energy_local(d, pe.iref, data)
    });

    let (intslocal, popfloatslocal) = time_call!(crate::timers::stochastic::add_compute_populations, {
        population_local(mc, isref, run)
    });
    
    let obslocal = [dnumlocal, ddenlocal, popfloatslocal[0], popfloatslocal[1], intslocal[0] as f64, intslocal[1] as f64, intslocal[2] as f64];
    let mut obsglobal = [0.0_f64; 7];

    time_call!(crate::timers::stochastic::add_observables_allreduce, {
        world.all_reduce_into(&obslocal[..], &mut obsglobal[..], SystemOperation::sum());
    });

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
    let dt_eff = input.prop_ref().dt * (qmc.ncycles as f64);

    if !state.reached_c && stats.nwc >= qmc.target_population {
        state.reached_c = true;
    }
    if !state.reached_sc && stats.nwsc >= qmc.target_population as f64 {
        state.reached_sc = true;
    }

    if state.reached_c {
        *es -= (qmc.shift_damping / dt_eff) * (stats.nwc as f64 / state.prev_pop.nwc as f64).ln();
    }

    if state.reached_sc {
        state.es_s -= (qmc.shift_damping / dt_eff) * (stats.nwsc / state.prev_pop.nwsc).ln();
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
    
    // Per destination send buffers for remote walker updates over MPI.
    let mut send: Vec<Vec<PopulationUpdate>> = (0..nranks).map(|_| Vec::new()).collect();
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
            accumulate_updates(&mut state.mc, &mut send, prop, data.input, &mut mpiscratch, world, nranks);
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

