// stochastic/common.rs
// Standard library imports.
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// External crate imports.
use mpi::collective::SystemOperation;
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::Communicator;
use mpi::traits::*;
use ndarray::Array2;
use rand::SeedableRng;
use rayon::prelude::*;

// Crate-root imports.
use crate::input::{Input, Propagator};
use crate::maths::general_evp;
use crate::mpiutils::broadcast;
use crate::noci::{
    AuxiliarySpace, DetPair, NOCIData, OverlapFactors, calculate_h_pairs_orthogonal_batched,
    calculate_hs_pair, calculate_hs_pairs_wicks_batched, calculate_s_pair,
};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::time_call;

// Parent/sibling imports.
use super::excit::OrthogonalUniformGenerator;
use super::overlapweighted::OverlapWeightedGenerator;
use super::restart::basis_hash;
use super::state::{
    AuxiliaryPropagationResult, AuxiliaryThreadPropagation, MCState, NOCIMPIScratch,
    NOCIPopulationUpdate, NOCIPropagationResult, NOCIThreadPropagation, PopulationStats,
    ProjectedEnergyUpdate, QMCRunInfo, QmcRng, ScratchSize, ShiftSpec, SparsePopulations, owner,
};

/// Accumulate one signed population change in dense/sparse Monte Carlo storage.
/// # Arguments:
/// - `mc`: Current Monte Carlo state.
/// - `det`: Global determinant index.
/// - `dn`: Signed population change.
/// # Returns:
/// - `()`: Adds `dn` and records the determinant on first touch.
fn add_delta(
    mc: &mut MCState,
    det: usize,
    dn: f64,
) {
    if dn == 0.0 {
        return;
    }
    if mc.delta[det] == 0.0 {
        mc.changed.push(det);
    }
    mc.delta[det] += dn;
}

/// Drain the dense population-change accumulator into sparse updates.
/// # Arguments:
/// - `mc`: Current Monte Carlo state.
/// - `changes`: Reusable sparse output buffer.
/// # Returns:
/// - `()`: Clears touched dense entries and fills `changes`.
pub(in crate::stochastic) fn take_population_changes(
    mc: &mut MCState,
    changes: &mut Vec<NOCIPopulationUpdate>,
) {
    changes.clear();
    for det in mc.changed.drain(..) {
        let dn = mc.delta[det];
        mc.delta[det] = 0.0;
        if dn != 0.0 {
            changes.push(NOCIPopulationUpdate {
                det: det as u64,
                dn,
            });
        }
    }
}

/// Accumulate worker-generated local updates and retain remote updates for report exchange.
/// # Arguments:
/// - `mc`: Current Monte Carlo state.
/// - `propagation`: Worker propagation results.
/// - `input`: User input controlling optional histograms.
/// - `scratch`: Reusable MPI update storage.
/// # Returns:
/// - `()`: Updates local delta, histogram, and remote buffers.
pub(in crate::stochastic) fn accumulate_generated_updates(
    mc: &mut MCState,
    propagation: &mut NOCIPropagationResult,
    input: &Input,
    scratch: &mut NOCIMPIScratch,
) {
    for (det, dn) in propagation.local.drain(..) {
        add_delta(mc, det, dn);
    }
    if input.write.write_excitation_hist
        && let Some(histogram) = mc.excitation_hist.as_mut()
    {
        for sample in propagation.samples.drain(..) {
            histogram.add(sample);
        }
    } else {
        propagation.samples.clear();
    }
    scratch.send_ranked.append(&mut propagation.remote);
}

/// Pack remote population updates into deterministic destination blocks.
/// # Arguments:
/// - `nranks`: Number of MPI ranks.
/// - `scratch`: Reusable MPI update storage.
/// # Returns:
/// - `()`: Fills contiguous send data, counts, and displacements.
pub(in crate::stochastic) fn prepare_spawn_update_exchange(
    nranks: usize,
    scratch: &mut NOCIMPIScratch,
) {
    scratch.send_counts.fill(0);
    scratch.send_displacements.fill(0);
    scratch
        .send_ranked
        .sort_unstable_by_key(|&(peer, update)| (peer, update.det));
    let mut out = 0usize;
    for i in 0..scratch.send_ranked.len() {
        let (peer, update) = scratch.send_ranked[i];
        if out != 0
            && scratch.send_ranked[out - 1].0 == peer
            && scratch.send_ranked[out - 1].1.det == update.det
        {
            scratch.send_ranked[out - 1].1.dn += update.dn;
        } else {
            scratch.send_ranked[out] = (peer, update);
            out += 1;
        }
    }
    scratch.send_ranked.truncate(out);
    scratch.send_contig.clear();
    for &(peer, update) in &scratch.send_ranked {
        if update.dn != 0.0 {
            scratch.send_counts[peer] += 1;
            scratch.send_contig.push(update);
        }
    }
    let mut sent = 0usize;
    for peer in 0..nranks {
        scratch.send_displacements[peer] = sent as i32;
        sent += scratch.send_counts[peer] as usize;
    }
}

/// Exchange report-accumulated remote spawn updates and add owner-local results.
/// # Arguments:
/// - `mc`: Current Monte Carlo state.
/// - `scratch`: Reusable MPI update storage.
/// - `world`: MPI communicator.
/// - `run`: Rank-local ownership metadata.
/// # Returns:
/// - `()`: Adds received updates to `mc.delta`.
pub(in crate::stochastic) fn exchange_accumulated_updates(
    mc: &mut MCState,
    scratch: &mut NOCIMPIScratch,
    world: &impl CommunicatorCollectives,
    run: &QMCRunInfo,
) {
    if run.nranks <= 1 {
        scratch.send_contig.clear();
        scratch.send_ranked.clear();
        return;
    }
    prepare_spawn_update_exchange(run.nranks, scratch);
    let received = exchange_population_changes(world, scratch);
    for &update in received {
        add_delta(mc, update.det as usize, update.dn);
    }
    scratch.send_contig.clear();
    scratch.send_ranked.clear();
}

/// Generate one stochastic estimate of the ordinary or SApply NOCI residual.
/// SApply additionally accumulates `B = \partial\Delta/\partial E_s = dt S\tilde N` from the
/// same sampled paths; ordinary walker propagators omit this tangent.
/// # Arguments:
/// - `sample`: Global cycle index and sparse sampled populations.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local propagation metadata.
/// - `shift`: Current shifted residual specification.
/// - `overlap`: Persistent factors, generator, mixture weight, and optimisation flag.
/// - `workers`: Persistent thread-local propagation storage.
/// - `result`: Reusable generated-update storage.
/// # Returns:
/// - `()`: Fills physical updates and, for SApply, worker shift tangents.
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
    workers: &mut [Mutex<NOCIThreadPropagation>],
    result: &mut NOCIPropagationResult,
) {
    let (iteration, sampled) = sample;
    let (overlap_factors, overlap_generator, overlap_weight, optimise_overlap_weight) = overlap;
    let accumulate_tangent = matches!(shift.propagator, Propagator::SApply);
    let dt = data.input.prop_ref().dt;
    if accumulate_tangent {
        for worker in workers.iter_mut() {
            worker.get_mut().unwrap().shift_tangent.prepare(run.ndets);
        }
    }
    result.clear();
    let occupied = sampled.occ();
    if !occupied.is_empty() {
        let next = AtomicUsize::new(0);
        let workers_shared: &[Mutex<NOCIThreadPropagation>] = workers;
        rayon::broadcast(|context| {
            let tid = context.index();
            let mut worker = workers_shared[tid].lock().unwrap();
            worker.clear();
            worker.rng = QmcRng::seed_from_u64(
                run.rank_seed ^ tid as u64 ^ (iteration as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
            loop {
                let start = next.fetch_add(8, Ordering::Relaxed);
                if start >= occupied.len() {
                    break;
                }
                for &source in &occupied[start..(start + 8).min(occupied.len())] {
                    let population = sampled.get(source);
                    if population == 0.0 {
                        continue;
                    }
                    if accumulate_tangent {
                        // For `d\Delta_x = -dt(H_xx-E_sS_xx)\tilde N_x`,
                        // `\partial d\Delta_x/\partial E_s = dt S_xx\tilde N_x`.
                        let dn = dt * run.diagonal_hs[source].1 * population;
                        worker.shift_tangent.add(source, dn, run.nranks > 1);
                    }
                    worker.diagonal_population_change(
                        source,
                        population,
                        shift,
                        data,
                        &run.diagonal_hs,
                    );
                    worker.spawning(
                        source,
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
                accumulate_tangent,
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

/// Generate one cycle of the stochastic auxiliary-space residual
/// `\chi_D=-dt\sum_{a,x}[H_{Dx}-E_s\delta_{Dx}]\tilde N_x^{(a)}`.
/// # Arguments:
/// - `sample`: Global cycle index and sparse sampled populations.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local ownership and cached diagonal metadata.
/// - `shift`: Current physical shift `E_s`.
/// - `auxiliary`: Uniform connection generator and canonical auxiliary-space topology.
/// - `workers`: Persistent thread-local auxiliary propagation storage.
/// - `result`: Reusable local, remote, and generation-sample results.
/// # Returns
/// - `()`: Fills one cycle's realised auxiliary updates and worker shift tangents.
pub(in crate::stochastic) fn propagate_iteration_auxiliary(
    sample: (usize, &SparsePopulations),
    data: &NOCIData<'_, f64>,
    run: &QMCRunInfo,
    shift: f64,
    auxiliary: (&OrthogonalUniformGenerator, &AuxiliarySpace),
    workers: &mut [Mutex<AuxiliaryThreadPropagation>],
    result: &mut AuxiliaryPropagationResult,
) {
    let (iteration, sampled) = sample;
    let (generator, auxiliary_space) = auxiliary;
    let dt = data.input.prop_ref().dt;
    for worker in workers.iter_mut() {
        worker.get_mut().unwrap().shift_tangent.prepare(run.ndets);
    }
    result.clear();
    let occupied = sampled.occ();
    if !occupied.is_empty() {
        let next = AtomicUsize::new(0);
        let workers_shared: &[Mutex<AuxiliaryThreadPropagation>] = workers;
        rayon::broadcast(|context| {
            let tid = context.index();
            let mut worker = workers_shared[tid].lock().unwrap();
            worker.clear();
            worker.rng = QmcRng::seed_from_u64(
                run.rank_seed ^ tid as u64 ^ (iteration as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );

            loop {
                let start = next.fetch_add(8, Ordering::Relaxed);
                if start >= occupied.len() {
                    break;
                }
                for &source in &occupied[start..(start + 8).min(occupied.len())] {
                    let population = sampled.get(source);
                    if population == 0.0 {
                        continue;
                    }
                    worker
                        .shift_tangent
                        .add(source, dt * population, run.nranks > 1);
                    worker.diagonal_population_change(
                        source,
                        population,
                        shift,
                        data,
                        auxiliary_space,
                        run,
                    );
                    worker.spawning(source, population, generator);
                }
            }

            worker.resolve_batched_spawning(data, generator, auxiliary_space, run);
        });
    }
    for worker in workers.iter_mut() {
        let worker = worker.get_mut().unwrap();
        result.local.append(&mut worker.local);
        result.remote.append(&mut worker.remote);
        result.samples.append(&mut worker.samples);
    }
}

/// Contract a fixed projected-energy trial state against every locally owned determinant.
/// For `|Psi_P> = \sum_i c_i^P |Phi_i>`, constructs
/// `H_w^P = \sum_i c_i^P H_{iw}` and `S_w^P = \sum_i c_i^P S_{iw}`.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `projection`: Global determinant indices and coefficients defining `|Psi_P>`.
/// - `owned`: Global determinant indices owned by this MPI rank.
/// - `cached_hs`: Optional precomputed contractions indexed by global determinant.
/// - `scratch_size`: Maximum same-spin, alpha-spin and beta-spin Wick scratch dimensions.
/// # Returns:
/// - `Vec<(f64, f64)>`: Trial Hamiltonian and overlap contractions aligned with `owned`.
fn build_projection_hs(
    data: &NOCIData<'_, f64>,
    projection: &[(usize, f64)],
    owned: &[usize],
    cached_hs: &[Option<(f64, f64)>],
    scratch_size: (usize, usize, usize),
) -> Vec<(f64, f64)> {
    let (maxsame, maxla, maxlb) = scratch_size;

    owned
        .par_iter()
        .map_init(
            || WickScratchSpin::with_sizes(maxsame, maxla, maxlb),
            |scratch, &w| {
                // Reuse selected-space contractions whenever this column is already known.
                if let Some(hs) = cached_hs[w] {
                    return hs;
                }

                // Contract the trial coefficients against column `w` of H and S.
                projection
                    .iter()
                    .fold((0.0, 0.0), |(h, s), &(i, coefficient)| {
                        let (hiw, siw) = find_hs(data, i, w, scratch);

                        (h + coefficient * hiw, s + coefficient * siw)
                    })
            },
        )
        .collect()
}

/// Recover the reference-space Rayleigh quotient from the existing trial contractions.
/// For the reference vector `c_0`, evaluates `E_0 = (c_0^T H c_0)/(c_0^T S c_0)`.
/// # Arguments:
/// - `c0`: Reference coefficients embedded in the full stochastic determinant basis.
/// - `projection_hs`: Original reference-state `H` and `S` contractions aligned with `owned`.
/// - `owned`: Global determinant indices owned by this MPI rank.
/// - `world`: MPI communicator.
/// # Returns:
/// - `f64`: Reference-space NOCI energy `E_0`.
fn reference_projected_energy(
    c0: &[f64],
    projection_hs: &[(f64, f64)],
    owned: &[usize],
    world: &impl Communicator,
) -> f64 {
    // Only determinants carrying reference coefficients contribute to this Rayleigh quotient.
    let (num_local, den_local) =
        owned
            .iter()
            .zip(projection_hs.iter())
            .fold((0.0, 0.0), |(num, den), (&w, &(h, s))| {
                let coefficient = c0[w];

                (num + coefficient * h, den + coefficient * s)
            });

    let local = [num_local, den_local];
    let mut global = [0.0; 2];

    // Combine the distributed reference contributions before forming the energy.
    if world.size() == 1 {
        global = local;
    } else {
        world.all_reduce_into(&local, &mut global, SystemOperation::sum());
    }

    let energy = global[0] / global[1];

    // A non-finite result means the supplied reference state is not a valid projector.
    if !energy.is_finite() {
        panic!(
            "reference projected energy is not finite: numerator {}, denominator {}",
            global[0], global[1]
        );
    }

    energy
}

/// Select the largest external components of the reference-state Schrödinger residual.
/// For `R_w = H_w^0 - E_0 S_w^0`, retains the largest external values of `|R_w|`.
/// # Arguments:
/// - `projection_hs`: Original reference-state `H` and `S` contractions aligned with `owned`.
/// - `reference_energy`: Reference-space NOCI energy `E_0`.
/// - `ref_indices`: Original reference determinant indices.
/// - `isref`: Boolean mask identifying original reference determinants.
/// - `owned`: Global determinant indices owned by this MPI rank.
/// - `n_projected`: Total requested projection-space dimension including original references.
/// - `world`: MPI communicator.
/// # Returns:
/// - `Vec<usize>`: Ordered projection-space determinant indices with references first.
fn select_projected_determinants(
    projection_hs: &[(f64, f64)],
    reference_energy: f64,
    ref_indices: &[usize],
    isref: &[bool],
    owned: &[usize],
    n_projected: usize,
    world: &impl Communicator,
) -> Vec<usize> {
    let n_external = n_projected - ref_indices.len();

    // The reference-only projector requires no residual ranking.
    if n_external == 0 {
        return ref_indices.to_vec();
    }

    let mut local_residuals = vec![0.0; isref.len()];

    // Ownership is disjoint, so each residual is written on exactly one rank.
    for (&w, &(h, s)) in owned.iter().zip(projection_hs.iter()) {
        if !isref[w] {
            local_residuals[w] = (h - reference_energy * s).abs();
        }
    }

    let mut residuals = vec![0.0; isref.len()];

    // Reconstruct the complete residual vector once during deterministic setup.
    if world.size() == 1 {
        residuals = local_residuals;
    } else {
        world.all_reduce_into(
            &local_residuals[..],
            &mut residuals[..],
            SystemOperation::sum(),
        );
    }

    // Exclude references because they are retained unconditionally.
    let mut candidates = residuals
        .into_iter()
        .enumerate()
        .filter_map(|(w, residual)| (!isref[w]).then_some((w, residual)))
        .collect::<Vec<_>>();

    // Use the determinant index as a stable tie-break for equal residuals.
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Keep only the external determinants required to reach `n_projected`.
    candidates.truncate(n_external);

    // The earlier dimension validation should make this impossible.
    if candidates.len() != n_external {
        panic!(
            "requested {} external projected-energy determinants but found only {}",
            n_external,
            candidates.len()
        );
    }

    let mut projected = Vec::with_capacity(n_projected);

    // Preserve the original ordering of the reference NOCI basis.
    projected.extend_from_slice(ref_indices);

    // Append external determinants in decreasing residual importance.
    projected.extend(candidates.into_iter().map(|(w, _)| w));

    projected
}

/// Solve the NOCI generalised eigenproblem in the residual-selected projection space.
/// Constructs `H_PP c_P = E_P S_PP c_P` and trial contractions for selected determinants.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `projected`: Global determinant indices defining the selected projection space.
/// - `scratch_size`: Maximum same-spin, alpha-spin and beta-spin Wick scratch dimensions.
/// - `world`: MPI communicator.
/// # Returns:
/// - `(Vec<f64>, Vec<(f64, f64)>)`: Ground-state coefficients and selected-column contractions.
fn solve_projected_state(
    data: &NOCIData<'_, f64>,
    projected: &[usize],
    scratch_size: (usize, usize, usize),
    world: &impl Communicator,
) -> (Vec<f64>, Vec<(f64, f64)>) {
    let mut coefficients = Vec::new();
    let mut selected_hs = Vec::new();

    // Construct the deterministic selected-space problem only once on rank zero.
    if world.rank() == 0 {
        let n = projected.len();
        let (maxsame, maxla, maxlb) = scratch_size;

        // Enumerate the independent upper-triangle matrix elements.
        let pairs = (0..n)
            .flat_map(|i| (i..n).map(move |j| (i, j)))
            .collect::<Vec<_>>();

        // Evaluate independent H and S elements in parallel with thread-local Wick scratch.
        let values = pairs
            .par_iter()
            .map_init(
                || WickScratchSpin::with_sizes(maxsame, maxla, maxlb),
                |scratch, &(i, j)| {
                    let hs = find_hs(data, projected[i], projected[j], scratch);

                    (i, j, hs)
                },
            )
            .collect::<Vec<_>>();

        let mut h = Array2::<f64>::zeros((n, n));
        let mut s = Array2::<f64>::zeros((n, n));

        // Scatter the evaluated upper triangle into the real-symmetric H and S matrices.
        for (i, j, (hij, sij)) in values {
            h[(i, j)] = hij;
            h[(j, i)] = hij;

            s[(i, j)] = sij;
            s[(j, i)] = sij;
        }

        // Use the established positive-overlap projection used by the NOCI solver.
        let (energies, vectors) = general_evp(&h, &s, true, data.tol);

        // The first generalized eigenvector is the selected-space ground state.
        coefficients = vectors.column(0).iter().copied().collect();

        // Reuse the selected H and S block instead of recalculating selected columns.
        selected_hs = (0..n)
            .into_par_iter()
            .map(|j| {
                let mut hp = 0.0;
                let mut sp = 0.0;

                for i in 0..n {
                    hp += coefficients[i] * h[(i, j)];
                    sp += coefficients[i] * s[(i, j)];
                }

                (hp, sp)
            })
            .collect();

        println!("Projected-energy trial determinants: {n}");
        println!("Projected-energy trial energy: {:.12}", energies[0]);
    }

    // Every rank requires the optimized coefficients for its owned determinant columns.
    broadcast(world, &mut coefficients);

    // Every rank also receives the contractions already known inside the selected space.
    broadcast(world, &mut selected_hs);

    (coefficients, selected_hs)
}

/// Construct shared rank-local metadata for every stochastic propagation family.
/// The projected-energy state retains references and the largest components of
/// `R_w = <Phi_w|(H-E_0)|Psi_0>` until its requested dimension is reached.
/// # Arguments:
/// - `data`: Shared NOCI data.
/// - `c0`: Initial reference coefficient vector embedded in the stochastic determinant basis.
/// - `ref_indices`: Original reference determinant indices.
/// - `world`: MPI communicator.
/// # Returns:
/// - `(Vec<bool>, ScratchSize, QMCRunInfo)`: Reference mask, Wick scratch bounds, and run metadata.
pub(in crate::stochastic) fn construct_qmc_run(
    data: &NOCIData<'_, f64>,
    c0: &[f64],
    ref_indices: &[usize],
    world: &impl Communicator,
) -> (Vec<bool>, ScratchSize, QMCRunInfo) {
    let qmc = data.input.qmc.as_ref().unwrap();
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let ndets = data.space.len();

    // An omitted `n_projected` means that only the original references are used.
    let n_projected = qmc.n_projected.unwrap_or(ref_indices.len());

    // The projection state must contain every original NOCI reference.
    if n_projected < ref_indices.len() {
        panic!(
            "qmc.n_projected = {} is smaller than the number of NOCI references {}",
            n_projected,
            ref_indices.len()
        );
    }

    // The projection state cannot exceed the stochastic determinant basis.
    if n_projected > ndets {
        panic!(
            "qmc.n_projected = {} exceeds the stochastic determinant-space size {}",
            n_projected, ndets
        );
    }

    // Keep the reference-population mask independent of the enlarged energy projector.
    let mut isref = vec![false; ndets];

    for &i in ref_indices {
        isref[i] = true;
    }

    // Preserve the established deterministic rank-dependent RNG construction.
    let base_seed = qmc.seed.unwrap_or_else(rand::random);
    let rank_seed = base_seed.wrapping_add((irank as u64).wrapping_mul(0x9E3779B9));

    // Determine the largest Wick scratch dimensions required by the stochastic basis.
    let (maxsame, maxla, maxlb) = max_scratch_sizes(data.space);

    let scratchsize = ScratchSize {
        maxsame,
        maxla,
        maxlb,
    };

    // Assign every stochastic determinant to one deterministic MPI owner.
    let det_owner = if nranks == 1 {
        vec![0; ndets]
    } else {
        (0..ndets).map(|det| owner(det, nranks)).collect::<Vec<_>>()
    };

    // Store only the persistent population coordinates owned by this rank.
    let owned = if nranks == 1 {
        (0..ndets).collect::<Vec<_>>()
    } else {
        det_owner
            .iter()
            .enumerate()
            .filter_map(|(det, &owner)| if owner == irank { Some(det) } else { None })
            .collect::<Vec<_>>()
    };

    // Embed the original reference NOCI state in the full stochastic basis.
    let reference = ref_indices
        .iter()
        .filter_map(|&i| {
            let coefficient = c0[i];

            (coefficient != 0.0).then_some((i, coefficient))
        })
        .collect::<Vec<_>>();

    // Cache diagonal matrix elements because stochastic propagation reuses them repeatedly.
    let local_diagonal_hs = owned
        .par_iter()
        .map_init(
            || WickScratchSpin::with_sizes(maxsame, maxla, maxlb),
            |scratch, &w| find_hs(data, w, w, scratch),
        )
        .collect::<Vec<_>>();

    // Retain the existing global-indexed diagonal layout used throughout propagation.
    let mut diagonal_hs = vec![(0.0, 0.0); ndets];

    for (&w, hs) in owned.iter().zip(local_diagonal_hs) {
        diagonal_hs[w] = hs;
    }

    // No selected-space contractions are known before constructing the reference projector.
    let empty_cache = vec![None; ndets];

    // Build the original reference contractions needed both for EProj and for the residual.
    let mut projection_hs = build_projection_hs(
        data,
        &reference,
        &owned,
        &empty_cache,
        (maxsame, maxla, maxlb),
    );

    // Preserve the exact old setup path when only the original references are requested.
    if n_projected > ref_indices.len() {
        // Recover E_0 without evaluating any additional H or S matrix elements.
        let reference_energy = reference_projected_energy(c0, &projection_hs, &owned, world);

        // Select external determinants where the reference wavefunction violates H Psi = E S Psi most.
        let projected = select_projected_determinants(
            &projection_hs,
            reference_energy,
            ref_indices,
            &isref,
            &owned,
            n_projected,
            world,
        );

        // Optimize the coefficients by rediagonalizing H and S in the selected space.
        let (projected_coefficients, selected_hs) =
            solve_projected_state(data, &projected, (maxsame, maxla, maxlb), world);

        // Pair each selected determinant with its optimized NOCI coefficient.
        let projection = projected
            .iter()
            .copied()
            .zip(projected_coefficients)
            .collect::<Vec<_>>();

        // Cache projected contractions already available from H_PP and S_PP.
        let mut cached_hs = vec![None; ndets];

        for (&w, hs) in projected.iter().zip(selected_hs) {
            cached_hs[w] = Some(hs);
        }

        // Evaluate only projection-to-basis columns not already available from the selected solve.
        projection_hs = build_projection_hs(
            data,
            &projection,
            &owned,
            &cached_hs,
            (maxsame, maxla, maxlb),
        );
    }

    // Propagation, shift control and NRangeRef remain independent of projector enrichment.
    (
        isref,
        scratchsize,
        QMCRunInfo {
            irank,
            nranks,
            ndets,
            basis_hash: basis_hash(data.space),
            det_owner,
            owned,
            base_seed,
            rank_seed,
            projection_hs,
            diagonal_hs,
        },
    )
}

/// Compute the projected energy from persistent real populations.
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
        // Projector enrichment still leaves exactly one stored H and S contraction per determinant.
        let (num_local, den_local) = populations
            .par_iter()
            .zip(run.projection_hs.par_iter())
            .map(|(&population, &(h, s))| (population * h, population * s))
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        // Reduce numerator and denominator together exactly as in the existing estimator.
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

/// Compute range-population statistics and projected energy with one MPI reduction.
/// # Arguments:
/// - `mc`: Current Monte Carlo state.
/// - `isref`: Original reference-determinant mask.
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
        // Accumulate population diagnostics and the enriched estimator in one local pass.
        let (nw_local, nref_local, num_local, den_local) = mc
            .populations
            .par_iter()
            .enumerate()
            .zip(run.projection_hs.par_iter())
            .map(|((k, &population), &(h, s))| {
                let abs = population.abs();

                // NRangeRef remains tied only to the original NOCI references.
                let nref = if isref[run.owned[k]] { abs } else { 0.0 };

                (abs, nref, population * h, population * s)
            })
            .reduce(
                || (0.0, 0.0, 0.0, 0.0),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
            );

        // Preserve the existing single collective for all report-level scalar observables.
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

/// `Find overlap matrix element S_{ij}.`
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
    let ldet = data.space.state(crate::noci::NOCIIndex(a));
    let gdet = data.space.state(crate::noci::NOCIIndex(b));

    // If the determinants share the same parent take an orthogonal early exit.
    if ldet.parent == gdet.parent
        && let Some(mocache) = data.mocache
        && mocache[ldet.parent].orthogonal_slater_condon
    {
        if data.space.occupations(crate::noci::NOCIIndex(a))
            == data.space.occupations(crate::noci::NOCIIndex(b))
        {
            return data.space.phase(crate::noci::NOCIIndex(a))
                * data.space.phase(crate::noci::NOCIIndex(b));
        }
        return 0.0;
    }

    // Otherwise calculate normally.
    calculate_s_pair(
        data,
        DetPair::new(crate::noci::NOCIIndex(a), crate::noci::NOCIIndex(b)),
        Some(scratch),
    )
}

/// `Find Hamiltonian and overlap matrix elements H_{ij} and S_{ij}.`
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
        DetPair::new(crate::noci::NOCIIndex(a), crate::noci::NOCIIndex(b)),
        Some(scratch),
    )
}

/// Find batched Hamiltonian and overlap matrix elements for canonically ordered determinant pairs.
/// Extended nonorthogonal Wick evaluation uses the batched NOCI path. Without Wick evaluation,
/// requests fall back to the existing scalar matrix-element evaluator.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `pairs`: Canonically ordered determinant-index pairs `(a, b)` with `a <= b`.
/// - `scratch`: Reusable Wick scratch space for scalar and generic-rank evaluation.
/// - `out`: Hamiltonian and overlap results in the same order as `pairs`.
/// # Returns:
/// - `()`: Writes every requested `(H, S)` pair into `out`.
pub(in crate::stochastic) fn find_hs_batched(
    data: &NOCIData<'_, f64>,
    pairs: &[(usize, usize)],
    scratch: &mut WickScratchSpin<f64>,
    out: &mut [(f64, f64)],
) {
    if data.input.wicks.enabled && data.wicks.is_some() {
        calculate_hs_pairs_wicks_batched(data, pairs, scratch, out);
        return;
    }

    for (i, &(a, b)) in pairs.iter().enumerate() {
        let ldet = data.space.state(crate::noci::NOCIIndex(a));
        let gdet = data.space.state(crate::noci::NOCIIndex(b));
        let loa = data.space.occupations(crate::noci::NOCIIndex(a));
        let goa = data.space.occupations(crate::noci::NOCIIndex(b));

        if ldet.parent == gdet.parent
            && (loa.0 ^ goa.0).count_ones() + (loa.1 ^ goa.1).count_ones() > 4
        {
            out[i] = (0.0, 0.0);
        } else {
            out[i] = find_hs(data, a, b, scratch);
        }
    }
}

/// Evaluate batched orthogonal Hamiltonian elements
/// `H_{D_kx_k}=\langle D_k^{P_k}|\hat H|\Phi_{x_k}^{P_k}\rangle`.
/// # Arguments:
/// - `data`: Shared stochastic propagation data and parent MO caches.
/// - `generator`: Uniform parent-orthogonal connection topology.
/// - `factorisation`: Canonical parent-local source component IDs.
/// - `components`: Prepared occupied and virtual labels for those canonical components.
/// - `pairs`: Compact retained-source and relative-connection indices.
/// - `scratch`: Reusable numerical parent-and-sector request groups.
/// - `out`: Hamiltonian results in request order.
/// # Returns
/// - `()`: Writes all requested orthogonal Hamiltonian elements into `out`.
pub(in crate::stochastic) fn find_h_orthogonal_batched(
    data: &NOCIData<'_, f64>,
    generator: &OrthogonalUniformGenerator,
    pairs: &[(crate::noci::NOCIIndex, usize)],
    scratch: &mut crate::noci::OrthogonalHamiltonianScratch,
    out: &mut [f64],
) {
    calculate_h_pairs_orthogonal_batched(data, generator.connections(), pairs, scratch, out);
}

/// Determine the maximum scratch sizes required for computation of matrix elements using extended
/// non-orthogonal Wick's theorem depending on the maximum excitation rank present in the basis.
/// # Arguments:
/// - `basis`: Full list of the NOCI-QMC basis.
/// # Returns
/// - `(usize, usize, usize)`: Maximum same-spin scratch size, alpha excitation size, and beta
///   excitation size.
pub(in crate::stochastic) fn max_scratch_sizes(
    space: &crate::noci::NOCISpace<f64>
) -> (usize, usize, usize) {
    let maxexa = space
        .components
        .iter()
        .flat_map(|parent| &parent.alpha)
        .map(|st| st.excitation.holes.count_ones() as usize)
        .max()
        .unwrap_or(0);
    let maxexb = space
        .components
        .iter()
        .flat_map(|parent| &parent.beta)
        .map(|st| st.excitation.holes.count_ones() as usize)
        .max()
        .unwrap_or(0);
    let maxsame = 2 * maxexa.max(maxexb);
    let maxla = 2 * maxexa;
    let maxlb = 2 * maxexb;
    (maxsame, maxla, maxlb)
}

/// Communicate spawned population updates between MPI ranks.
/// Remote spawn updates are stored locally as one `Vec<NOCIPopulationUpdate>` per destination rank.
/// This routine packs those per-destination buffers into one contiguous send buffer, exchanges the
/// number of updates each rank will send/receive, then performs one `MPI_Alltoallv`-style exchange
/// of the packed payloads.
/// # Arguments:
/// - `world`: `MPI communicator object (MPI_COMM_WORLD).`
/// - `scratch`: Reusable MPI scratch space for counts, displacements, and contiguous send/recv buffers.
/// # Returns
/// - `&[NOCIPopulationUpdate]`: Flat buffer containing all spawned population updates received from other ranks.
pub(crate) fn exchange_population_changes<'a>(
    world: &impl CommunicatorCollectives,
    scratch: &'a mut NOCIMPIScratch,
) -> &'a [NOCIPopulationUpdate] {
    time_call!(
        crate::timers::stochastic::add_exchange_population_changes,
        {
            let nranks = world.size() as usize;

            // Gather every rank's per-destination message sizes.
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
                .resize(nrecv, NOCIPopulationUpdate { det: 0, dn: 0.0 });

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
/// - `world`: `MPI communicator object (MPI_COMM_WORLD).`
/// - `send`: Local population updates to gather.
/// - `scratch`: Reusable MPI scratch space.
/// # Returns
/// - `&[NOCIPopulationUpdate]`: Global gathered population updates.
pub(crate) fn gather_all_populations<'a>(
    world: &impl Communicator,
    send: &[NOCIPopulationUpdate],
    scratch: &'a mut NOCIMPIScratch,
) -> &'a [NOCIPopulationUpdate] {
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
            .resize(ntot, NOCIPopulationUpdate { det: 0, dn: 0.0 });
        let mut recv = PartitionMut::new(
            &mut scratch.gather_recv[..],
            &scratch.gather_counts[..],
            &scratch.gather_displs[..],
        );
        world.all_gather_varcount_into(send, &mut recv);
        &scratch.gather_recv[..]
    })
}

/// Combine repeated consecutive determinant updates in place.
/// # Arguments:
/// - `updates`: Population updates sorted by determinant index.
/// # Returns
/// - `()`: Compresses repeated determinant updates in place.
pub(in crate::stochastic) fn coalesce_population_updates(updates: &mut Vec<NOCIPopulationUpdate>) {
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
