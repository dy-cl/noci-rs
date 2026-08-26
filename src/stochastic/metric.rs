// stochastic/metric.rs

// Standard library imports.
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// External crate imports.
use mpi::collective::SystemOperation;
use mpi::datatype::PartitionMut;
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

// Crate-root imports.
use crate::ReducedTwoSpinDetState;
use crate::input::{ExcitationGen, Input};
use crate::noci::{NOCIData, OverlapFactors, OverlapScratch, SpinFactorisation};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::time_call;

// Parent/sibling imports.
use super::common::{
    coalesce_population_updates, exchange_population_changes, find_hs, max_scratch_sizes,
};
use super::excit::update_overlap_weight;
use super::init::initialise_qmc_state;
use super::overlapweighted::OverlapWeightedGenerator;
use super::report::{check_stop, print_header, print_initial_row, print_row, write_restart};
use super::restart::basis_hash;
use super::state::owner;
use super::state::{
    ExcitationHist, MCState, MPIScratch, OverlapDerivativeSums, PopulationStats, PopulationUpdate,
    ProjectedEnergyUpdate, PropagationResult, PropagationState, QMCRunInfo, QmcRng, ScratchSize,
    ShiftSpec, SparsePopulations, ThreadPropagation,
};

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

/// Drain the population-change accumulator `delta` into a sparse list of non-zero changes.
/// # Arguments:
/// - `mc`: Current Monte Carlo state.
/// - `changes`: Reusable output buffer receiving sparse real population changes.
/// # Returns:
/// - `()`: Replaces `changes` with the current sparse population changes.
pub(in crate::stochastic) fn take_population_changes(
    mc: &mut MCState,
    changes: &mut Vec<PopulationUpdate>,
) {
    time_call!(crate::timers::stochastic::add_take_population_changes, {
        changes.clear();

        if !mc.changed.is_empty() {
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
        }
    })
}

/// `Apply \delta N_w = \sum_\Omega S_{w\Omega}\Delta_\Omega.`
/// # Arguments:
/// - `populations`: `Rank-local persistent populations N_w.`
/// - `updates`: `Sparse pre-overlap changes \Omega, \Delta_\Omega.`
/// - `data`: Immutable NOCI data.
/// - `overlap_factor`: Precomputed determinant and spin-component mappings.
/// - `overlap_factors`: Persistent cross-parent overlap factors.
/// - `targets`: Global determinant indices for rank-local rows.
/// - `scratch`: `Reusable allocation storage for one application of S\Delta.`
/// # Returns:
/// - `()`: `Applies N_w \leftarrow N_w + \delta N_w.`
fn apply_population_changes_local<I>(
    populations: &mut [f64],
    updates: I,
    data: &NOCIData<'_, f64>,
    overlap_factor: &SpinFactorisation,
    overlap_factors: &OverlapFactors,
    targets: &[usize],
    scratch: &mut OverlapScratch,
) where
    I: IntoIterator<Item = (usize, f64)>,
{
    overlap_factor.apply_overlap_sparse(
        populations,
        targets,
        updates,
        data,
        overlap_factors,
        scratch,
    );
}

/// Apply the global overlap-transformed population change.
/// Each rank initially owns a subset of the accumulated changes
/// `\Delta_\Omega`. The changes are gathered across MPI ranks and each
/// rank updates its locally owned persistent populations according to
/// `N_w \leftarrow N_w + \sum_\Omega S_{w\Omega}\Delta_\Omega`.
/// Since every change has the form `S\Delta`, the population vector remains in
/// `\operatorname{range}(S)` provided the initial vector is in
/// `\operatorname{range}(S)`, therefore avoiding population growth in the null space.
/// # Arguments:
/// - `populations`: Rank-local persistent populations.
/// - `dlocal`: Local determinant population changes.
/// - `data`: Immutable stochastic propagation data.
/// - `overlap_factor`: Reusable spin overlap factors.
/// - `overlap_factors`: Persistent cross-parent overlap factors.
/// - `run`: Rank-local run metadata.
/// - `mpi`: MPI communicator and reusable MPI scratch storage.
/// - `scratch`: `Reusable overlap allocation storage for grouped S\Delta application.`
/// # Returns
/// - `()`: Applies the global overlap-transformed population change.
fn apply_overlap_population_changes(
    changes: (&mut [f64], &[PopulationUpdate]),
    data: &NOCIData<'_, f64>,
    overlap: (&SpinFactorisation, &OverlapFactors),
    run: &QMCRunInfo,
    mpi: (&impl CommunicatorCollectives, &mut MPIScratch),
    scratch: &mut OverlapScratch,
) {
    let (populations, dlocal) = changes;
    let (overlap_factor, overlap_factors) = overlap;
    let (world, mpi) = mpi;

    time_call!(crate::timers::stochastic::add_apply_overlap_changes, {
        if run.nranks == 1 {
            time_call!(
                crate::timers::stochastic::add_apply_local_overlap_changes,
                {
                    apply_population_changes_local(
                        populations,
                        dlocal.iter().map(|up| (up.det as usize, up.dn)),
                        data,
                        overlap_factor,
                        overlap_factors,
                        &run.owned,
                        scratch,
                    );
                }
            );
            return;
        }

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

        // Size recieve buffer to hold all updates from all ranks.
        mpi.gather_recv
            .resize(ntot, PopulationUpdate { det: 0, dn: 0.0 });

        let mut recv = PartitionMut::new(
            &mut mpi.gather_recv[..],
            &mpi.gather_counts[..],
            &mpi.gather_displs[..],
        );
        time_call!(crate::timers::stochastic::add_wait_overlap_change_gather, {
            world.all_gather_varcount_into(dlocal, &mut recv);
        });

        // Apply all report updates in one pass. This avoids regrouping local and remote updates
        // separately, which is more expensive than the small all-gather wait on these workloads.
        time_call!(
            crate::timers::stochastic::add_apply_remote_overlap_changes,
            {
                mpi.gather_recv.sort_unstable_by_key(|update| update.det);
                coalesce_population_updates(&mut mpi.gather_recv);

                apply_population_changes_local(
                    populations,
                    mpi.gather_recv.iter().map(|up| (up.det as usize, up.dn)),
                    data,
                    overlap_factor,
                    overlap_factors,
                    &run.owned,
                    scratch,
                );
            }
        );
    })
}

/// Accumulate all local population updates into the global delta vector, update the local
/// excitation histogram if requested, and retain remote updates for report-end exchange.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `prop`: Population updates generated in the spawning and death/cloning step.
/// - `input`: User specified input options.
/// - `scratch`: Reusable MPI communication scratch.
/// # Returns
/// - `()`: Updates the local delta vector, optional excitation histogram, and remote update buffer.
pub(in crate::stochastic) fn accumulate_generated_updates(
    mc: &mut MCState,
    prop: &mut PropagationResult,
    input: &Input,
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

        scratch.send_ranked.append(&mut prop.remote);
        prop.remote.clear();
    })
}

/// Prepare accumulated remote spawned population updates for exchange.
/// # Arguments:
/// - `nranks`: Number of MPI ranks.
/// - `det_owner`: MPI owner rank for each global determinant.
/// - `scratch`: Reusable MPI scratch space.
/// # Returns
/// - `()`: Fills `send_counts`, `send_displacements`, and `send_contig` in `scratch`.
fn prepare_spawn_update_exchange(
    nranks: usize,
    scratch: &mut MPIScratch,
) {
    for x in scratch.send_counts.iter_mut() {
        *x = 0;
    }
    for x in scratch.send_displacements.iter_mut() {
        *x = 0;
    }

    // Sort by (destination rank, determinant) so duplicates for the same peer
    // are adjacent and each peer still occupies one contiguous block.
    scratch
        .send_ranked
        .sort_unstable_by_key(|&(peer, up)| (peer, up.det));

    let mut out = 0usize;
    for i in 0..scratch.send_ranked.len() {
        let (peer, up) = scratch.send_ranked[i];

        if out > 0
            && scratch.send_ranked[out - 1].0 == peer
            && scratch.send_ranked[out - 1].1.det == up.det
        {
            scratch.send_ranked[out - 1].1.dn += up.dn;
        } else {
            scratch.send_ranked[out] = (peer, up);
            out += 1;
        }
    }
    scratch.send_ranked.truncate(out);

    scratch.send_contig.clear();
    for &(peer, up) in &scratch.send_ranked {
        if up.dn != 0.0 {
            scratch.send_counts[peer] += 1;
            scratch.send_contig.push(up);
        }
    }

    // Build displacements for the already-packed contiguous buffer.
    let mut nsend = 0usize;
    for peer in 0..nranks {
        scratch.send_displacements[peer] = nsend as i32;
        nsend += scratch.send_counts[peer] as usize;
    }
}

/// Exchange report-accumulated remote spawn updates and add received updates to `delta`.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `mpi`: Reusable MPI communication scratch.
/// - `world`: `MPI communicator object (MPI_COMM_WORLD).`
/// - `ndets`: Total number of determinants.
/// - `nranks`: Total number of MPI ranks.
/// # Returns
/// - `()`: Adds received remote updates to `mc.delta`.
pub(in crate::stochastic) fn exchange_accumulated_updates(
    mc: &mut MCState,
    mpi: &mut MPIScratch,
    world: &impl CommunicatorCollectives,
    run: &QMCRunInfo,
) {
    if run.nranks <= 1 {
        mpi.send_contig.clear();
        mpi.send_ranked.clear();
        return;
    }

    prepare_spawn_update_exchange(run.nranks, mpi);
    let received = exchange_population_changes(world, mpi);

    time_call!(crate::timers::stochastic::add_unpack_population_changes, {
        for &update in received {
            add_delta(mc, update.det as usize, update.dn);
        }
    });

    mpi.send_contig.clear();
    mpi.send_ranked.clear();
}

/// Generate one stochastic estimate of the pre-overlap population change.
/// `Given the sampled populations \tilde N, this function`
/// `estimates \Delta = -\Delta\tau(H - E_s S)\tilde N.`
/// # Arguments:
/// - `it`: Global stochastic-cycle index.
/// - `sampled`: `Sparse sampled populations \tilde N.`
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local propagation metadata.
/// - `shift`: `Current population-control shift E_s(\Delta \tau).`
/// - `overlap_factors`: Persistent cross-parent overlap factors.
/// - `overlap_generator`: Optional overlap-weighted excitation generator.
/// - `overlap_weight`: Current report overlap mixture probability.
/// - `optimise_overlap_weight`: Whether to accumulate adaptive derivative sums.
/// - `workers`: Persistent thread-local propagation storage.
/// - `result`: Reusable storage for generated population changes.
/// # Returns:
/// - `()`: `Fills result with an estimate of -\Delta\tau(H - E_s S)\tilde N.`
pub(in crate::stochastic) fn propagate_iteration(
    sample: (usize, &SparsePopulations),
    data: &NOCIData<'_, f64>,
    run: &QMCRunInfo,
    shift: ShiftSpec,
    overlap: (
        Option<&OverlapFactors>,
        Option<&OverlapWeightedGenerator>,
        f64,
        bool,
    ),
    workers: &mut [Mutex<ThreadPropagation>],
    result: &mut PropagationResult,
) {
    let (it, sampled) = sample;
    let (overlap_factors, overlap_generator, overlap_weight, optimise_overlap_weight) = overlap;

    time_call!(
        crate::timers::stochastic::add_generate_population_changes,
        {
            result.clear();

            let occ = sampled.occ();

            if !occ.is_empty() {
                let next = AtomicUsize::new(0);
                let chunk_size = 8usize;
                let workers_shared: &[Mutex<ThreadPropagation>] = workers;

                rayon::broadcast(|context| {
                    let tid = context.index();
                    let mut worker = workers_shared[tid].lock().unwrap();

                    worker.clear();
                    worker.rng = QmcRng::seed_from_u64(
                        run.rank_seed ^ tid as u64 ^ (it as u64).wrapping_mul(0x9E3779B97F4A7C15),
                    );

                    loop {
                        let start = next.fetch_add(chunk_size, Ordering::Relaxed);

                        if start >= occ.len() {
                            break;
                        }

                        let end = (start + chunk_size).min(occ.len());
                        for &gamma in &occ[start..end] {
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

                            worker.spawning(
                                gamma,
                                population,
                                shift,
                                data,
                                run,
                                (overlap_factors, overlap_generator, overlap_weight),
                            );
                        }
                    }

                    worker.resolve_batched_spawning(
                        shift,
                        data,
                        run,
                        (
                            overlap_factors,
                            overlap_generator,
                            overlap_weight,
                            optimise_overlap_weight,
                        ),
                    );
                });
            }

            for worker in workers.iter_mut() {
                let worker = worker.get_mut().unwrap();
                result.local.append(&mut worker.local);
                result.remote.append(&mut worker.remote);
                result.samples.append(&mut worker.samples);
                result.overlap_derivatives.add(&worker.overlap_derivatives);
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

        if run.nranks == 1 {
            global = local;
        } else {
            world.all_reduce_into(&local, &mut global, SystemOperation::sum());
        }

        ProjectedEnergyUpdate {
            num: global[0],
            den: global[1],
        }
    })
}

/// Compute population statistics and projected energy with one MPI reduction.
/// # Arguments:
/// - `mc`: Current Monte Carlo state.
/// - `isref`: Reference-determinant mask.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// # Returns:
/// - `(PopulationStats, ProjectedEnergyUpdate)`: Global population statistics and projected energy.
pub(in crate::stochastic) fn population_stats_projected_energy(
    mc: &MCState,
    isref: &[bool],
    run: &QMCRunInfo,
    world: &impl Communicator,
) -> (PopulationStats, ProjectedEnergyUpdate) {
    time_call!(crate::timers::stochastic::add_compute_population_stats, {
        let (nw_local, nref_local, num_local, den_local) = mc
            .populations
            .par_iter()
            .enumerate()
            .zip(run.reference_hs.par_iter())
            .map(|((k, &population), &(h, s))| {
                let abs = population.abs();
                let nref = if isref[run.owned[k]] { abs } else { 0.0 };

                (abs, nref, population * h, population * s)
            })
            .reduce(
                || (0.0, 0.0, 0.0, 0.0),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
            );

        let local = [
            nw_local,
            nref_local,
            mc.sampled.norm(),
            mc.sampled.occ().len() as f64,
            num_local,
            den_local,
        ];
        let mut global = [0.0; 6];

        if run.nranks == 1 {
            global = local;
        } else {
            world.all_reduce_into(&local, &mut global, SystemOperation::sum());
        }

        (
            PopulationStats::new(global[0], global[1], global[2], global[3] as i64),
            ProjectedEnergyUpdate {
                num: global[4],
                den: global[5],
            },
        )
    })
}

/// Construct an FRI-style sparse unbiased stochastic sample of the populations.
/// `For a population x and some cutoff c > 0, \Phi_c(x) = x if |x| \geq c. Otherwise,`
/// `Phi_c(x) = \text{sign}(x)c with probability |x| / c, \Phi_c(x) = 0 with probability`
/// `1 - |x| / c. Therefore, \mathbb E[\Phi_c(x)] = x, and hence \mathbb E[\tilde N \mid N] = N.`
/// # Arguments:
/// - `populations`: Persistent rank-local population vector N.
/// - `sampled`: `Temporary sparse sampled vector \tilde N.`
/// - `cutoff`: Stochastic sampling cutoff c.
/// - `run`: Rank-local determinant ownership information.
/// - `rng`: Random-number generator.
/// # Returns:
/// - `()`: Replaces `sampled` with a sparse unbiased sample of `populations`.
pub(in crate::stochastic) fn sample_populations(
    populations: &[f64],
    sampled: &mut SparsePopulations,
    cutoff: f64,
    run: &QMCRunInfo,
    rng: &mut QmcRng,
    chunks: &mut Vec<Vec<(usize, f64)>>,
) {
    time_call!(crate::timers::stochastic::add_sample_populations, {
        if cutoff <= 0.0 {
            sampled.clear();

            for (k, &population) in populations.iter().enumerate() {
                if population != 0.0 {
                    sampled.insert_nonzero(run.owned[k], population);
                }
            }
            return;
        }

        if populations.len() < 8192 {
            sampled.clear();

            for (&det, &population) in run.owned.iter().zip(populations.iter()) {
                if population == 0.0 {
                    continue;
                }

                let abs_population = population.abs();

                if abs_population >= cutoff {
                    sampled.insert_nonzero(det, population);
                } else if rng.r#gen::<f64>() < abs_population / cutoff {
                    sampled.insert_nonzero(det, cutoff.copysign(population));
                }
            }

            return;
        }

        let nthreads = rayon::current_num_threads().max(1);
        let chunk_size = populations.len().div_ceil(nthreads).max(1024);
        let nchunks = populations.len().div_ceil(chunk_size);
        let seed = rng.r#gen::<u64>();

        chunks.resize_with(nchunks, Vec::new);

        chunks[..nchunks]
            .par_iter_mut()
            .enumerate()
            .for_each(|(chunk, entries)| {
                let mut rng =
                    QmcRng::seed_from_u64(seed ^ (chunk as u64).wrapping_mul(0x9E3779B97F4A7C15));
                let start = chunk * chunk_size;
                let end = (start + chunk_size).min(populations.len());
                let populations = &populations[start..end];
                let owned = &run.owned[start..end];

                entries.clear();

                for (&det, &population) in owned.iter().zip(populations.iter()) {
                    if population == 0.0 {
                        continue;
                    }

                    let abs_population = population.abs();

                    if abs_population >= cutoff {
                        entries.push((det, population));
                    } else if rng.r#gen::<f64>() < abs_population / cutoff {
                        entries.push((det, cutoff.copysign(population)));
                    }
                }
            });

        sampled.clear();

        for entries in chunks.iter().take(nchunks) {
            for &(det, population) in entries {
                sampled.insert_nonzero(det, population);
            }
        }
    });
}

/// Apply unbiased FRI stochastic rounding to a signed real value.
/// `For some cutoff c > 0, \mathcal \Phi_c(x) = x when x = 0,`
/// `c \leq 0, or |x| \geq c. For 0 < |x| < c, \Phi_c(x) = \text{sign}(x)c`
/// with probability |x| / c, and zero otherwise. The rounding is conditionally unbiased
/// `as \mathbb E[\Phi_c(x) \mid x] = x.`
/// # Arguments:
/// - `value`: Signed real value x.
/// - `cutoff`: Minimum nonzero retained magnitude c.
/// - `rng`: Random-number generator.
/// # Returns:
/// - `f64`: Unbiased stochastically rounded value.
pub(in crate::stochastic) fn fri(
    value: f64,
    cutoff: f64,
    rng: &mut QmcRng,
) -> f64 {
    if value == 0.0 || cutoff <= 0.0 || value.abs() >= cutoff {
        return value;
    }

    if rng.r#gen::<f64>() < value.abs() / cutoff {
        cutoff.copysign(value)
    } else {
        0.0
    }
}

/// Apply FRI compression to sparse population updates in place.
/// `Each update \Delta b_x is replaced by \Phi_c(\Delta b_x), so the compressed`
/// pre-overlap vector remains conditionally unbiased:
/// `\mathbb E[\Phi_c(\Delta b_x) \mid \Delta b_x] = \Delta b_x.`
/// # Arguments:
/// - `updates`: Sparse population changes.
/// - `cutoff`: Minimum nonzero retained magnitude c.
/// - `rng`: Random-number generator.
/// # Returns:
/// - `()`: Replaces `updates` with its sparse FRI-compressed form.
pub(in crate::stochastic) fn fri_population_updates(
    updates: &mut Vec<PopulationUpdate>,
    cutoff: f64,
    rng: &mut QmcRng,
) {
    let mut out = 0usize;

    for i in 0..updates.len() {
        let mut update = updates[i];
        update.dn = fri(update.dn, cutoff, rng);

        if update.dn != 0.0 {
            updates[out] = update;
            out += 1;
        }
    }

    updates.truncate(out);
}

/// Update the single population-control shift.
/// # Arguments:
/// - `stats`: Current population statistics.
/// - `state`: Current propagation state.
/// - `shift`: Population-control shift.
/// - `input`: User input options.
/// # Returns:
/// - `()`: Updates the shift and cached previous population.
pub(in crate::stochastic) fn update_shift(
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
/// `The initial population is N_0 = S c_0, rescaled to the requested  population 1-norm.`
/// Within each report block, the population vector is held fixed while `ncycles` independent
/// `samples \tilde N^{(a)} = \Phi_c(N) generate pre-overlap changes`
/// `\Delta^{(a)} \approx -\Delta\tau(H - E_s S)\tilde N^{(a)}.`
/// At the end of the report block, the accumulated change is applied as
/// `N'= N + S\sum_{a = 1}^{n_{\text{cycles}}}\Delta^{(a)}.`
/// `This update preserves N \in \range(S) and removes null-space components.`
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: `Initial determinant coefficient vector c_0.`
/// - `es`: `Population-control shift E_s.`
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

    if let Some(write_restart_interval) = data.input.write.write_restart_interval
        && (write_restart_interval == 0 || write_restart_interval % qmc.ncycles != 0)
    {
        println!("write_restart_interval must be divisible by qmc.ncycles");
        std::process::exit(1);
    }

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

    let det_owner = if nranks == 1 {
        vec![0; ndets]
    } else {
        (0..ndets)
            .map(|det| owner(det, ndets, nranks))
            .collect::<Vec<_>>()
    };
    let owned = if nranks == 1 {
        (0..ndets).collect::<Vec<_>>()
    } else {
        det_owner
            .iter()
            .enumerate()
            .filter_map(|(det, &owner)| if owner == irank { Some(det) } else { None })
            .collect::<Vec<_>>()
    };

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

    let local_diagonal_hs: Vec<(f64, f64)> = owned
        .par_iter()
        .map_init(
            || {
                WickScratchSpin::with_sizes(
                    scratchsize.maxsame,
                    scratchsize.maxla,
                    scratchsize.maxlb,
                )
            },
            |scratch, &gamma| find_hs(data, gamma, gamma, scratch),
        )
        .collect();
    let mut diagonal_hs = vec![(0.0, 0.0); ndets];
    for (&gamma, hs) in owned.iter().zip(local_diagonal_hs) {
        diagonal_hs[gamma] = hs;
    }

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

    let reduced_basis = data
        .basis
        .iter()
        .map(ReducedTwoSpinDetState::from_state)
        .collect::<Vec<_>>();

    let overlap_factor = SpinFactorisation::new(data);
    let build_overlap_cdfs = matches!(qmc.excitation_gen, ExcitationGen::OverlapWeighted);
    let factor_cache = data.input.wicks.cachedir.as_deref().unwrap_or(".");
    let overlap_factors = overlap_factor.build_overlap_factors(
        data,
        Path::new(factor_cache),
        world.rank(),
        qmc.factor_tables,
        build_overlap_cdfs,
    );
    let overlap_generator = if let ExcitationGen::OverlapWeighted = qmc.excitation_gen {
        Some(OverlapWeightedGenerator::new(
            data,
            &overlap_factor,
            &overlap_factors,
        ))
    } else {
        None
    };
    let run = QMCRunInfo {
        irank,
        nranks,
        ndets,
        basis_hash: basis_hash(data.basis),
        reduced_basis,
        det_owner,
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
    let mut overlap_scratch = overlap_factor.overlap_scratch();

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
            QmcRng,
            WickScratchSpin<f64>,
        );
        println!(
            "Size of per thread state (MiB): {}",
            std::mem::size_of::<ThreadState>() as f64 / (1024.0 * 1024.0)
        );
    }

    let propagator = data.input.prop_ref().propagator;
    print_header(irank, propagator);
    print_initial_row(
        irank,
        state.start_report * qmc.ncycles,
        &state,
        data.basis[0].e,
        *es,
        propagator,
    );

    let mut population_changes = Vec::new();
    let mut sample_chunks = Vec::new();
    let mut overlap_derivatives = OverlapDerivativeSums::default();

    for report in state.start_report..qmc.nreports {
        for cycle in 0..qmc.ncycles {
            let iter = report * qmc.ncycles + cycle;

            let mut rng = QmcRng::seed_from_u64(
                run.rank_seed ^ 0xD1B54A32D192ED03 ^ (iter as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );

            sample_populations(
                &state.mc.populations,
                &mut state.mc.sampled,
                qmc.sampling_cutoff1,
                &run,
                &mut rng,
                &mut sample_chunks,
            );

            propagate_iteration(
                (iter, &state.mc.sampled),
                data,
                &run,
                ShiftSpec::direct_overlap(*es),
                (
                    Some(&overlap_factors),
                    overlap_generator.as_ref(),
                    state.overlap_weight,
                    qmc.optimise_overlap_weight,
                ),
                &mut workers,
                &mut propagation_result,
            );
            overlap_derivatives.add(&propagation_result.overlap_derivatives);

            accumulate_generated_updates(
                &mut state.mc,
                &mut propagation_result,
                data.input,
                &mut mpiscratch,
            );
        }

        exchange_accumulated_updates(&mut state.mc, &mut mpiscratch, world, &run);

        take_population_changes(&mut state.mc, &mut population_changes);

        population_changes.sort_unstable_by_key(|update| update.det);

        let mut fri_rng = QmcRng::seed_from_u64(
            run.rank_seed ^ 0xA0761D6478BD642F ^ (report as u64).wrapping_mul(0xE7037ED1A0B428DB),
        );
        fri_population_updates(&mut population_changes, qmc.sampling_cutoff2, &mut fri_rng);

        apply_overlap_population_changes(
            (&mut state.mc.populations, &population_changes),
            data,
            (&overlap_factor, &overlap_factors),
            &run,
            (world, &mut mpiscratch),
            &mut overlap_scratch,
        );

        let end = (report + 1) * qmc.ncycles;

        let (stats, pe) = population_stats_projected_energy(&state.mc, &isref, &run, world);
        state.pe = pe;

        state.eprojcur = state.pe.num / state.pe.den;

        state.cur_pop = stats;

        update_shift(&stats, &mut state, es, data.input);
        update_overlap_weight(
            &mut state,
            &mut overlap_derivatives,
            data.input,
            &run,
            world,
        );

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

        if let Some(write_restart_interval) = data.input.write.write_restart_interval
            && end.is_multiple_of(write_restart_interval)
        {
            write_restart(
                report,
                &state,
                *es,
                &run,
                world,
                data.input.write.write_restart.as_ref(),
            );
        }

        print_row(irank, end, &state, &stats, data.basis[0].e, *es, propagator);
    }

    (state.eprojcur, state.mc.excitation_hist)
}
