// stochastic/propagate.rs
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use mpi::collective::SystemOperation;
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;

use super::state::{
    ExcitationHist, MCState, MPIScratch, PopulationStats, PopulationUpdate, ProjectedEnergyUpdate,
    PropagationResult, PropagationState, QMCRunInfo, ScratchSize, SparsePopulations,
    ThreadPropagation,
};
use crate::input::Input;
use crate::noci::{DetPair, NOCIData, OverlapFactor, OverlapFactorScratch};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::time_call;

use super::init::{initialise_qmc_state, max_scratch_sizes};
use super::report::{check_stop, print_header, print_initial_row, print_row};
use super::state::{owned, owner};
use crate::noci::{calculate_hs_pair, calculate_s_pair};

/// Find overlap matrix element S_{ij}.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `i`: Index of state `i`.
/// - `j`: Index of state `j`.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element `S_{ij}`.
pub(in crate::stochastic) fn find_s(
    data: &NOCIData<'_, f64>,
    i: usize,
    j: usize,
    scratch: &mut WickScratchSpin<f64>,
) -> f64 {
    // Get the sorted pair of indices
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    let ldet = &data.basis[a];
    let gdet = &data.basis[b];

    // If the determinants share the same parent take an orthogonal early exit.
    if ldet.parent == gdet.parent
        && let Some(mocache) = data.mocache
        && mocache[ldet.parent].orthogonal_slater_condon
    {
        if ldet.oa == gdet.oa && ldet.ob == gdet.ob {
            return (ldet.pha * gdet.pha) * (ldet.phb * gdet.phb);
        }
        return 0.0;
    }

    // Otherwise calculate normally.
    calculate_s_pair(data, DetPair::new(ldet, gdet), Some(scratch))
}

/// Find Hamiltonian and overlap matrix elements H_{ij} and S_{ij}.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `i`: Index of state `i`.
/// - `j`: Index of state `j`.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `(f64, f64)`: Hamiltonian and overlap matrix elements `H_{ij}` and `S_{ij}`.
pub(in crate::stochastic) fn find_hs(
    data: &NOCIData<'_, f64>,
    i: usize,
    j: usize,
    scratch: &mut WickScratchSpin<f64>,
) -> (f64, f64) {
    // Get the sorted pair of indices
    let (a, b) = if i <= j { (i, j) } else { (j, i) };

    // Calculate the matrix element.
    calculate_hs_pair(
        data,
        DetPair::new(&data.basis[a], &data.basis[b]),
        Some(scratch),
    )
}

/// Accumulate a real population change on determinant `i`.
/// # Arguments:
/// - `mc`: Current Monte Carlo state.
/// - `i`: Determinant index.
/// - `dn`: Signed real population change.
/// # Returns:
/// - `()`: Updates the population-change accumulator.
fn add_delta(
    mc: &mut MCState,
    i: usize,
    dn: f64,
) {
    // Take an early exit if the population-change is nill.
    if dn == 0.0 {
        return;
    }

    // If delta is zero it is already dealt with.
    if mc.delta[i] == 0.0 {
        mc.changed.push(i);
    }

    // Add population change to the delta.
    mc.delta[i] += dn;
}

/// Convert the population-change accumulator `delta` into a sparse
/// list of any non-zero changes `changes`.
/// # Arguments:
/// - `mc`: Current Monte Carlo state.
/// # Returns:
/// - `Vec<PopulationUpdate>`: Sparse real population changes.
fn take_population_changes(mc: &mut MCState) -> Vec<PopulationUpdate> {
    time_call!(crate::timers::stochastic::add_take_population_changes, {
        if mc.changed.is_empty() {
            Vec::new()
        } else {
            let mut changes = Vec::with_capacity(mc.changed.len());

            for &det in &mc.changed {
                let dn = mc.delta[det];
                mc.delta[det] = 0.0;

                if dn != 0.0 {
                    changes.push(PopulationUpdate {
                        det: det as u64,
                        dn,
                    });
                }
            }

            mc.changed.clear();
            changes
        }
    })
}

/// Communicate spawned population updates between MPI ranks.
/// Remote spawn updates are stored locally as one `Vec<PopulationUpdate>` per destination rank.
/// This routine packs those per-destination buffers into one contiguous send buffer, exchanges the
/// number of updates each rank will send/receive, then performs one `MPI_Alltoallv`-style exchange
/// of the packed payloads.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `scratch`: Reusable MPI scratch space for counts, displacements, and contiguous send/recv buffers.
/// # Returns
/// - `&[PopulationUpdate]`: Flat buffer containing all spawned population updates received from other ranks.
pub(crate) fn exchange_population_changes<'a>(
    world: &impl CommunicatorCollectives,
    scratch: &'a mut MPIScratch,
) -> &'a [PopulationUpdate] {
    time_call!(
        crate::timers::stochastic::add_exchange_population_changes,
        {
            let nranks = world.size() as usize;

            // Exchange only the per-rank message sizes.
            // After this, `recv_counts[peer]` contains how many updates rank `peer` will send to the
            // current rank.
            time_call!(
                crate::timers::stochastic::add_exchange_population_change_counts,
                {
                    world.all_to_all_into(&scratch.send_counts[..], &mut scratch.recv_counts[..]);
                }
            );

            // Build the incoming MPI metadata.
            // `recv_displacements[peer]` is the starting offset of rank `peer`'s block inside the packed
            // contiguous receive buffer `recv_contig`.
            // `nrecv` is the total number of remote updates this rank will receive.
            let mut nrecv = 0usize;
            for peer in 0..nranks {
                scratch.recv_displacements[peer] = nrecv as i32;
                nrecv += scratch.recv_counts[peer] as usize;
            }

            // Reuse one contiguous recieve buffer large enough for all remote updates.
            scratch.recv_contig.clear();
            scratch
                .recv_contig
                .resize(nrecv, PopulationUpdate { det: 0, dn: 0.0 });

            let send_part = Partition::new(
                &scratch.send_contig[..],
                &scratch.send_counts[..],
                &scratch.send_displacements[..],
            );
            let mut recv_part = PartitionMut::new(
                &mut scratch.recv_contig[..],
                &scratch.recv_counts[..],
                &scratch.recv_displacements[..],
            );
            time_call!(
                crate::timers::stochastic::add_exchange_population_change_payload,
                {
                    world.all_to_all_varcount_into(&send_part, &mut recv_part);
                }
            );

            // Return the receive buffer for later accumulation.
            &scratch.recv_contig[..]
        }
    )
}

/// Gather variable-length population updates from all ranks into a reusable receive buffer.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `send`: Local population updates to gather.
/// - `scratch`: Reusable MPI scratch space.
/// # Returns
/// - `&[PopulationUpdate]`: Global gathered population updates.
pub(crate) fn gather_all_populations<'a>(
    world: &impl Communicator,
    send: &[PopulationUpdate],
    scratch: &'a mut MPIScratch,
) -> &'a [PopulationUpdate] {
    time_call!(crate::timers::stochastic::add_gather_all_populations, {
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

        scratch
            .gather_recv
            .resize(ntot, PopulationUpdate { det: 0, dn: 0.0 });
        let mut recv = PartitionMut::new(
            &mut scratch.gather_recv[..],
            &scratch.gather_counts[..],
            &scratch.gather_displs[..],
        );
        world.all_gather_varcount_into(send, &mut recv);
        &scratch.gather_recv[..]
    })
}

/// Apply \delta N_\Gamma = \sum_\Omega S_{\Gamma\Omega}\Delta_\Omega.
/// # Arguments:
/// - `populations`: Rank-local persistent populations N_\Gamma.
/// - `updates`: Sparse pre-overlap changes \Omega, \Delta_\Omega.
/// - `data`: Immutable NOCI data.
/// - `overlap_factor`: Precomputed determinant and spin-component mappings.
/// - `targets`: Global determinant indices for rank-local rows.
/// - `scratch`: Reusable allocation storage for one application of S\Delta.
/// # Returns:
/// - `()`: Applies N_\Gamma \leftarrow N_\Gamma + \delta N_\Gamma.
fn apply_population_changes_local<I>(
    populations: &mut [f64],
    updates: I,
    data: &NOCIData<'_, f64>,
    overlap_factor: &OverlapFactor,
    targets: &[usize],
    scratch: &mut OverlapFactorScratch,
) where
    I: IntoIterator<Item = (usize, f64)>,
{
    overlap_factor.apply(populations, targets, updates, data, scratch);
}

/// Apply the global overlap-transformed population change.
/// Each rank initially owns a subset of the accumulated changes
/// \(\Delta_\Omega\). The changes are gathered across MPI ranks and each
/// rank updates its locally owned persistent populations according to
/// \(N_\Gamma \leftarrow N_\Gamma + \sum_\Omega S_{\Gamma\Omega}\Delta_\Omega\).
/// Since every change has the form \(S\Delta\), the population vector remains in
/// \(\operatorname{range}(S)\) provided the initial vector is in
/// \(\operatorname{range}(S)\), therefore avoiding population growth in the null space.
/// # Arguments:
/// - `populations`: Rank-local persistent populations.
/// - `dlocal`: Local determinant population changes.
/// - `data`: Immutable stochastic propagation data.
/// - `overlap_factor`: Reusable spin overlap factors.
/// - `run`: Rank-local run metadata.
/// - `mpi`: Reusable MPI scratch space.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `scratch`: Reusable overlap allocation storage for grouped S\Delta application.
/// # Returns
/// - `()`: Applies the global overlap-transformed population change.
fn apply_overlap_population_changes(
    populations: &mut [f64],
    dlocal: &[PopulationUpdate],
    data: &NOCIData<'_, f64>,
    overlap_factor: &OverlapFactor,
    run: &QMCRunInfo,
    mpi: &mut MPIScratch,
    world: &impl CommunicatorCollectives,
    scratch: &mut OverlapFactorScratch,
) {
    time_call!(crate::timers::stochastic::add_apply_overlap_changes, {
        // Gather number of updates that each rank will send to this rank.
        let nsend = dlocal.len() as i32;
        time_call!(
            crate::timers::stochastic::add_overlap_change_gather_counts,
            {
                world.all_gather_into(&nsend, &mut mpi.gather_counts[..]);
            }
        );

        // Calculate displacements for the recieve buffer and the total number of updates across
        // all ranks.
        let mut ntot = 0usize;
        for (i, &n) in mpi.gather_counts.iter().enumerate() {
            mpi.gather_displs[i] = ntot as i32;
            ntot += n as usize;
        }
        if ntot == 0 {
            return;
        }

        // Size recieve buffer to hold all updates from all ranks and calculate slice boundaries
        // that correspond to this rank's own constribution in the gathered buffer. These will
        // later be skipped during the apply step to avoid double counting.
        mpi.gather_recv
            .resize(ntot, PopulationUpdate { det: 0, dn: 0.0 });
        let locallow = mpi.gather_displs[run.irank] as usize;
        let localhigh = locallow + mpi.gather_counts[run.irank] as usize;

        let mut recv = PartitionMut::new(
            &mut mpi.gather_recv[..],
            &mpi.gather_counts[..],
            &mpi.gather_displs[..],
        );
        mpi::request::scope(|scope| {
            // Perform non-blocking all-gather such that the local computation can be overlapped.
            let req = world.immediate_all_gather_varcount_into(scope, dlocal, &mut recv);

            // Apply this ranks own updates to `populations` whilst the all-gather is occuring.
            time_call!(
                crate::timers::stochastic::add_apply_local_overlap_changes,
                {
                    apply_population_changes_local(
                        populations,
                        dlocal.iter().map(|up| (up.det as usize, up.dn)),
                        data,
                        overlap_factor,
                        &run.owned,
                        scratch,
                    );
                }
            );

            // Now we must wait for the all gather to finish before we read the remote updates.
            time_call!(crate::timers::stochastic::add_wait_overlap_change_gather, {
                req.wait();
            })
        });

        // Apply remote updates from all other ranks to `populations`. We avoid double counting by
        // excluding the local slice.
        time_call!(
            crate::timers::stochastic::add_apply_remote_overlap_changes,
            {
                let (local_and_left, right) = mpi.gather_recv.split_at_mut(localhigh);
                let (left, _) = local_and_left.split_at_mut(locallow);

                left.sort_unstable_by_key(|update| update.det);
                right.sort_unstable_by_key(|update| update.det);

                apply_population_changes_local(
                    populations,
                    left.iter().map(|up| (up.det as usize, up.dn)),
                    data,
                    overlap_factor,
                    &run.owned,
                    scratch,
                );
                apply_population_changes_local(
                    populations,
                    right.iter().map(|up| (up.det as usize, up.dn)),
                    data,
                    overlap_factor,
                    &run.owned,
                    scratch,
                );
            }
        );
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
fn acc_pack_updates(
    mc: &mut MCState,
    prop: &mut PropagationResult,
    input: &Input,
    ndets: usize,
    nranks: usize,
    scratch: &mut MPIScratch,
) {
    time_call!(crate::timers::stochastic::add_acc_pack_updates, {
        // Add the `delta` for local determinants.
        for (det, dn) in prop.local.drain(..) {
            add_delta(mc, det, dn);
        }

        // Optionally record excitation generation data.
        if input.write.write_excitation_hist
            && let Some(hist) = mc.excitation_hist.as_mut()
        {
            for p in prop.samples.drain(..) {
                hist.add(p);
            }
        } else {
            prop.samples.clear();
        }

        pack_spawn_updates(&prop.remote, ndets, nranks, scratch);
        prop.remote.clear();
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
fn accumulate_updates(
    mc: &mut MCState,
    prop: &mut PropagationResult,
    input: &Input,
    mpi: &mut MPIScratch,
    world: &impl CommunicatorCollectives,
    ndets: usize,
    nranks: usize,
) {
    // Sort updates into local and remote events based on determinant ownership.
    acc_pack_updates(mc, prop, input, ndets, nranks, mpi);

    // Exchange updates with all other ranks.
    if nranks > 1 {
        let received = exchange_population_changes(world, mpi);

        time_call!(crate::timers::stochastic::add_unpack_population_changes, {
            for &update in received {
                add_delta(mc, update.det as usize, update.dn);
            }
        });
    }
}

/// Combine repeated consecutive determinant updates in place.
/// # Arguments:
/// - `updates`: Remote population updates for a single destination rank.
/// # Returns
/// - `()`: Compresses repeated determinant updates in place.
fn coalesce_population_updates(updates: &mut Vec<PopulationUpdate>) {
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
    updates.retain(|up| up.dn != 0.0);
}

/// Pack remote spawned population updates into a contiguous buffer grouped by destination rank.
/// # Arguments:
/// - `remote`: Remote spawned population updates.
/// - `ndets`: Number of determinants in the stochastic basis.
/// - `nranks`: Number of MPI ranks.
/// - `scratch`: Reusable MPI scratch space.
/// # Returns
/// - `()`: Fills `send_counts`, `send_displacements`, and `send_contig` in `scratch`.
fn pack_spawn_updates(
    remote: &[PopulationUpdate],
    ndets: usize,
    nranks: usize,
    scratch: &mut MPIScratch,
) {
    for x in scratch.send_counts.iter_mut() {
        *x = 0;
    }
    for x in scratch.send_displacements.iter_mut() {
        *x = 0;
    }

    // Copy remote updates into the reusable contiguous buffer.
    scratch.send_contig.clear();
    scratch.send_contig.extend(remote.iter().cloned());

    // Sort by (destination rank, determinant) so duplicates for the same peer
    // are adjacent and each peer still occupies one contiguous block.
    scratch
        .send_contig
        .sort_unstable_by_key(|up| (owner(up.det as usize, ndets, nranks), up.det));

    // Merge repeated updates to the same determinant.
    coalesce_population_updates(&mut scratch.send_contig);

    // Recount after compression.
    for up in &scratch.send_contig {
        let peer = owner(up.det as usize, ndets, nranks);
        scratch.send_counts[peer] += 1;
    }

    // Build displacements for the already-packed contiguous buffer.
    let mut nsend = 0usize;
    for peer in 0..nranks {
        scratch.send_displacements[peer] = nsend as i32;
        nsend += scratch.send_counts[peer] as usize;
    }
}

/// Generate one stochastic estimate of the pre-overlap population change.
/// Given the sampled populations \tilde N, this function
/// estimates \Delta = -\Delta\tau(H - E_s S)\tilde N.
/// # Arguments:
/// - `it`: Global stochastic-cycle index.
/// - `sampled`: Sparse sampled populations \tilde N.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local propagation metadata.
/// - `shift`: Current population-control shift E_s(\Delta \tau).
/// - `workers`: Persistent thread-local propagation storage.
/// - `result`: Reusable storage for generated population changes.
/// # Returns:
/// - `()`: Fills `result` with an estimate of -\Delta\tau(H - E_s S)\tilde N.
fn propagate_iteration(
    it: usize,
    sampled: &SparsePopulations,
    data: &NOCIData<'_, f64>,
    run: &QMCRunInfo,
    shift: f64,
    workers: &mut [Mutex<ThreadPropagation>],
    result: &mut PropagationResult,
) {
    time_call!(
        crate::timers::stochastic::add_generate_population_changes,
        {
            result.clear();

            let occ = sampled.occ();

            if !occ.is_empty() {
                let next = AtomicUsize::new(0);
                let workers_shared: &[Mutex<ThreadPropagation>] = workers;

                rayon::broadcast(|context| {
                    let tid = context.index();
                    let mut worker = workers_shared[tid].lock().unwrap();

                    worker.clear();
                    worker.rng = SmallRng::seed_from_u64(
                        run.rank_seed ^ tid as u64 ^ (it as u64).wrapping_mul(0x9E3779B97F4A7C15),
                    );

                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);

                        if i >= occ.len() {
                            break;
                        }

                        let gamma = occ[i];
                        let population = sampled.get(gamma);

                        if population == 0.0 {
                            continue;
                        }

                        worker.diagonal_population_change(
                            gamma,
                            population,
                            shift,
                            data,
                            &run.diagonal_hs,
                        );

                        worker.spawning(gamma, population, shift, data, run);
                    }
                });
            }

            for worker in workers.iter_mut() {
                let worker = worker.get_mut().unwrap();
                result.local.append(&mut worker.local);
                result.remote.append(&mut worker.remote);
                result.samples.append(&mut worker.samples);
            }
        }
    );
}

/// Compute the projected energy from the persistent real populations.
/// # Arguments:
/// - `populations`: Persistent rank-local populations.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// # Returns:
/// - `ProjectedEnergyUpdate`: Global projected-energy numerator and denominator.
pub(in crate::stochastic) fn projected_energy(
    populations: &[f64],
    run: &QMCRunInfo,
    world: &impl Communicator,
) -> ProjectedEnergyUpdate {
    time_call!(crate::timers::stochastic::add_compute_projected_energy, {
        let (num_local, den_local) = populations
            .par_iter()
            .zip(run.reference_hs.par_iter())
            .map(|(&population, &(h, s))| (population * h, population * s))
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        let local = [num_local, den_local];
        let mut global = [0.0; 2];

        world.all_reduce_into(&local, &mut global, SystemOperation::sum());

        ProjectedEnergyUpdate {
            num: global[0],
            den: global[1],
        }
    })
}

/// Construct an FRI-style sparse unbiased stochastic sample of the populations.
/// For a population x and some cutoff c > 0, \Phi_c(x) = x if |x| \geq c. Otherwise,
/// Phi_c(x) = \text{sign}(x)c with probability |x| / c, \Phi_c(x) = 0 with probability
/// 1 - |x| / c. Therefore, \mathbb E[\Phi_c(x)] = x, and hence \mathbb E[\tilde N \mid N] = N.
/// # Arguments:
/// - `populations`: Persistent rank-local population vector N.
/// - `sampled`: Temporary sparse sampled vector \tilde N.
/// - `cutoff`: Stochastic sampling cutoff c.
/// - `run`: Rank-local determinant ownership information.
/// - `rng`: Random-number generator.
/// # Returns:
/// - `()`: Replaces `sampled` with a sparse unbiased sample of `populations`.
fn sample_populations(
    populations: &[f64],
    sampled: &mut SparsePopulations,
    cutoff: f64,
    run: &QMCRunInfo,
    rng: &mut SmallRng,
) {
    time_call!(crate::timers::stochastic::add_sample_populations, {
        sampled.clear();

        for (k, &population) in populations.iter().enumerate() {
            let population = stochastic_population_cutoff(population, cutoff, rng);

            if population != 0.0 {
                sampled.add(run.owned[k], population);
            }
        }
    });
}

/// Apply unbiased stochastic rounding to a signed real value.
/// For some cutoff c > 0, \mathcal \Phi_c(x) = x when x = 0,
/// c \leq 0, or |x| \geq c. For 0 < |x| < c, \Phi_c(x) = \text{sign}(x)c
/// with probability |x| / c, and zero otherwise. The rounding is conditionally unbiased
/// as \mathbb E[\Phi_c(x) \mid x] = x.
/// # Arguments:
/// - `value`: Signed real value x.
/// - `cutoff`: Minimum nonzero retained magnitude c.
/// - `rng`: Random-number generator.
/// # Returns:
/// - `f64`: Unbiased stochastically rounded value.
pub(in crate::stochastic) fn stochastic_population_cutoff(
    value: f64,
    cutoff: f64,
    rng: &mut SmallRng,
) -> f64 {
    if value == 0.0 || cutoff <= 0.0 || value.abs() >= cutoff {
        return value;
    }

    if rng.gen_range(0.0..1.0) < value.abs() / cutoff {
        value.signum() * cutoff
    } else {
        0.0
    }
}

/// Compute global statistics for the persistent and sampled populations.
/// # Arguments:
/// - `mc`: Current Monte Carlo state.
/// - `isref`: Reference-determinant mask.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// # Returns:
/// - `PopulationStats`: Global population statistics.
fn population_stats(
    mc: &MCState,
    isref: &[bool],
    run: &QMCRunInfo,
    world: &impl Communicator,
) -> PopulationStats {
    time_call!(crate::timers::stochastic::add_compute_population_stats, {
        let nw_local = mc
            .populations
            .iter()
            .map(|population| population.abs())
            .sum::<f64>();

        let nref_local = mc
            .populations
            .iter()
            .enumerate()
            .filter(|(k, _)| isref[run.owned[*k]])
            .map(|(_, population)| population.abs())
            .sum::<f64>();

        let local = [
            nw_local,
            nref_local,
            mc.sampled.norm(),
            mc.sampled.occ().len() as f64,
        ];

        let mut global = [0.0; 4];

        world.all_reduce_into(&local, &mut global, SystemOperation::sum());

        PopulationStats::new(global[0], global[1], global[2], global[3] as i64)
    })
}

/// Update the single population-control shift.
/// # Arguments:
/// - `stats`: Current population statistics.
/// - `state`: Current propagation state.
/// - `shift`: Population-control shift.
/// - `input`: User input options.
/// # Returns:
/// - `()`: Updates the shift and cached previous population.
fn update_shift(
    stats: &PopulationStats,
    state: &mut PropagationState,
    shift: &mut f64,
    input: &Input,
) {
    let qmc = input.qmc.as_ref().unwrap();
    let dteff = input.prop_ref().dt * qmc.ncycles as f64;

    if !state.reached && stats.nw >= qmc.target_population {
        state.reached = true;
    }

    if state.reached {
        *shift -= (qmc.shift_damping / dteff) * (stats.nw / state.prev_pop.nw).ln();
    }

    state.prev_pop = *stats;
}

/// Perform range-preserving null-space avoidant stochastic NOCI propagation.
/// The initial population is N_0 = S c_0, rescaled to the requested  population 1-norm.
/// Within each report block, the population vector is held fixed while `ncycles` independent
/// samples \tilde N^{(a)} = \Phi_c(N) generate pre-overlap changes
/// \Delta^{(a)} \approx -\Delta\tau(H - E_s S)\tilde N^{(a)}.
/// At the end of the report block, the accumulated change is applied as
/// N'= N + S\sum_{a = 1}^{n_{\text{cycles}}}\Delta^{(a)}.
/// This update preserves N \in \range(S) and removes null-space components.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: Initial determinant coefficient vector c_0.
/// - `es`: Population-control shift E_s.
/// - `ref_indices`: Determinants included in the reference-population norm.
/// - `world`: MPI communicator.
/// # Returns:
/// - `(f64, Option<ExcitationHist>)`: Final projected energy and optional
///   spawning-magnitude histogram.
pub fn qmc_step(
    data: &NOCIData<'_, f64>,
    c0: &[f64],
    es: &mut f64,
    ref_indices: &[usize],
    world: &impl Communicator,
) -> (f64, Option<ExcitationHist>) {
    let qmc = data.input.qmc.as_ref().unwrap();

    // Local MPI rank metadata.
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let ndets = data.basis.len();

    // Mark reference determinants for projected-energy calculations.
    let mut isref = vec![false; ndets];
    for &i in ref_indices {
        isref[i] = true;
    }

    // Each MPI rank gets a unique RNG seed.
    let base_seed = qmc.seed.unwrap_or_else(rand::random);
    let rank_seed = base_seed.wrapping_add((irank as u64).wrapping_mul(0x9E3779B9));

    // Precompute largest possible size needed for the non-orthogonal Wick's theorem scratch space.
    let scratchsize = {
        let (maxsame, maxla, maxlb) = max_scratch_sizes(data.basis);
        ScratchSize {
            maxsame,
            maxla,
            maxlb,
        }
    };

    let owned = owned(irank, ndets, nranks);

    let reference = ref_indices
        .iter()
        .filter_map(|&i| {
            let coefficient = c0[i];

            if coefficient == 0.0 {
                None
            } else {
                Some((i, coefficient))
            }
        })
        .collect::<Vec<_>>();

    let diagonal_hs: Vec<(f64, f64)> = (0..ndets)
        .into_par_iter()
        .map_init(
            || {
                WickScratchSpin::with_sizes(
                    scratchsize.maxsame,
                    scratchsize.maxla,
                    scratchsize.maxlb,
                )
            },
            |scratch, i| find_hs(data, i, i, scratch),
        )
        .collect();

    let reference_hs = owned
        .par_iter()
        .map_init(
            || {
                WickScratchSpin::with_sizes(
                    scratchsize.maxsame,
                    scratchsize.maxla,
                    scratchsize.maxlb,
                )
            },
            |scratch, &gamma| {
                let mut h = 0.0;
                let mut s = 0.0;

                for &(i, coefficient) in &reference {
                    let (hig, sig) = find_hs(data, i, gamma, scratch);

                    h += coefficient * hig;
                    s += coefficient * sig;
                }

                (h, s)
            },
        )
        .collect::<Vec<_>>();

    let overlap_factor = OverlapFactor::new(data);
    let run = QMCRunInfo {
        irank,
        nranks,
        ndets,
        owned,
        base_seed,
        rank_seed,
        reference_hs,
        diagonal_hs,
    };

    let mut workers = (0..rayon::current_num_threads())
        .map(|tid| {
            Mutex::new(ThreadPropagation::with_sizes(
                run.rank_seed ^ tid as u64,
                scratchsize.maxsame,
                scratchsize.maxla,
                scratchsize.maxlb,
            ))
        })
        .collect::<Vec<_>>();
    let mut propagation_result = PropagationResult::new();
    let mut overlap_scratch = overlap_factor.scratch();

    // Thread local scratch for Wick's theorem and for MPI communicattion.
    let mut scratch = WickScratchSpin::new();
    let mut mpiscratch = MPIScratch::new(run.nranks);

    // Initialise populations, projected-energy accumulators and shift.
    let mut state = initialise_qmc_state(
        c0,
        es,
        data,
        &run,
        &isref,
        &mut scratch,
        (world, &mut mpiscratch),
    );

    if irank == 0 {
        println!(
            "Size of Wick's Scratch (MiB): {}",
            std::mem::size_of::<WickScratchSpin<f64>>() as f64 / (1024.0 * 1024.0)
        );
        type ThreadState = (
            Vec<(usize, f64)>,
            Vec<PopulationUpdate>,
            Vec<f64>,
            SmallRng,
            WickScratchSpin<f64>,
        );
        println!(
            "Size of per thread state (MiB): {}",
            std::mem::size_of::<ThreadState>() as f64 / (1024.0 * 1024.0)
        );
    }

    print_header(irank);
    print_initial_row(irank, &state, data.basis[0].e);

    for report in state.start_report..qmc.nreports {
        for cycle in 0..qmc.ncycles {
            let iter = report * qmc.ncycles + cycle;

            let mut rng = SmallRng::seed_from_u64(
                run.rank_seed ^ 0xD1B54A32D192ED03 ^ (iter as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );

            sample_populations(
                &state.mc.populations,
                &mut state.mc.sampled,
                qmc.sampling_cutoff,
                &run,
                &mut rng,
            );

            propagate_iteration(
                iter,
                &state.mc.sampled,
                data,
                &run,
                *es,
                &mut workers,
                &mut propagation_result,
            );

            accumulate_updates(
                &mut state.mc,
                &mut propagation_result,
                data.input,
                &mut mpiscratch,
                world,
                ndets,
                nranks,
            );
        }

        let mut population_changes = take_population_changes(&mut state.mc);

        population_changes.sort_unstable_by_key(|update| update.det);

        coalesce_population_updates(&mut population_changes);

        apply_overlap_population_changes(
            &mut state.mc.populations,
            &population_changes,
            data,
            &overlap_factor,
            &run,
            &mut mpiscratch,
            world,
            &mut overlap_scratch,
        );

        let end = (report + 1) * qmc.ncycles;

        let stats = population_stats(&state.mc, &isref, &run, world);

        state.pe = projected_energy(&state.mc.populations, &run, world);

        state.eprojcur = state.pe.num / state.pe.den;

        state.cur_pop = stats;

        update_shift(&stats, &mut state, es, data.input);

        if let Some(ret) = check_stop(
            report,
            &mut state,
            *es,
            &run,
            world,
            data.input.write.write_restart.as_ref(),
        ) {
            return ret;
        }

        print_row(irank, end, &state, &stats, data.basis[0].e, *es);
    }

    (state.eprojcur, state.mc.excitation_hist)
}
