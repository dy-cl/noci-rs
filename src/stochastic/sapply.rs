// stochastic/sapply.rs

// Standard library imports.
use std::path::Path;
use std::sync::Mutex;

// External crate imports.
use mpi::datatype::PartitionMut;
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::SeedableRng;
use rayon::prelude::*;

// Crate-root imports.
use crate::input::ExcitationGen;
use crate::noci::{NOCIData, OverlapFactors, OverlapScratch, SpinFactorisation};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::time_call;

// Parent/sibling imports.
use super::common::{
    accumulate_generated_updates, coalesce_population_updates, exchange_accumulated_updates,
    exchange_population_changes, population_stats_projected_energy, prepare_spawn_update_exchange,
    propagate_iteration, take_population_changes,
};
use super::excit::update_overlap_weight;
use super::fri::{compress_dense_to_sparse, compress_sparse, sample_populations, target_cutoff};
use super::init::initialise_qmc_state;
use super::overlapweighted::OverlapWeightedGenerator;
use super::report::{check_stop, print_header, print_initial_row, print_row, write_restart};
use super::shift::update_shift_tangent;
use super::state::{
    ExcitationHist, NOCIMPIScratch, NOCIPopulationUpdate, NOCIPropagationResult,
    NOCIThreadPropagation, OverlapDerivativeSums, QMCRunInfo, QmcRng, ShiftSpec, ShiftTangent,
    TangentWorker,
};

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
/// - `changes`: Rank-local persistent populations and local determinant population changes.
/// - `data`: Immutable stochastic propagation data.
/// - `overlap`: Spin factorisation and persistent cross-parent overlap factors.
/// - `run`: Rank-local run metadata.
/// - `mpi`: MPI communicator and reusable MPI scratch storage.
/// - `scratch`: `Reusable overlap allocation storage for grouped S\Delta application.`
/// # Returns
/// - `()`: Applies the global overlap-transformed population change.
pub(in crate::stochastic) fn apply_overlap_population_changes(
    changes: (&mut [f64], &[NOCIPopulationUpdate]),
    data: &NOCIData<'_, f64>,
    overlap: (&SpinFactorisation, &OverlapFactors),
    run: &QMCRunInfo,
    mpi: (&impl CommunicatorCollectives, &mut NOCIMPIScratch),
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
            .resize(ntot, NOCIPopulationUpdate { det: 0, dn: 0.0 });

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

/// Exchange remote SApply tangent contributions and add them to owner-local storage.
/// # Arguments:
/// - `tangent`: Report-level shift tangent.
/// - `mpi`: Reusable MPI communication scratch.
/// - `world`: MPI communicator.
/// - `run`: Rank-local determinant ownership metadata.
/// # Returns:
/// - `()`: Adds received tangent updates to owner-local dense storage.
pub(in crate::stochastic) fn exchange_shift_tangent(
    tangent: &mut ShiftTangent,
    mpi: &mut NOCIMPIScratch,
    world: &impl CommunicatorCollectives,
    run: &QMCRunInfo,
) {
    if run.nranks <= 1 {
        tangent.remote.clear();
        return;
    }
    // Each rank holds partial contributions to `B_w = \sum_a dt \sum_x S_{wx}\tilde N_x`.
    // Redistribute them by determinant ownership before report-level FRI.
    mpi.send_ranked.append(&mut tangent.remote);
    prepare_spawn_update_exchange(run.nranks, mpi);
    let received = exchange_population_changes(world, mpi);
    for &update in received {
        tangent.add(update.det as usize, update.dn, true);
    }
    mpi.send_contig.clear();
    mpi.send_ranked.clear();
}

/// Reduce thread-local dense retained-space tangents into one report-level vector.
/// Computes `B_w = \sum_t B_w^{(t)}` in determinant-major order for either range propagator.
/// # Arguments:
/// - `workers`: Per-thread dense tangent accumulators.
/// - `tangent`: Dense report-level tangent receiving thread sum.
/// # Returns:
/// - `()`: Replaces `tangent` with report tangent and clears thread-local tangents.
pub(in crate::stochastic) fn reduce_shift_tangent<T: TangentWorker + Send>(
    workers: &mut [Mutex<T>],
    tangent: &mut [f64],
) {
    // Single-rank B is nearly dense. Reduce determinant-major to avoid touched-index bookkeeping
    // and sparse materialisation before target-NNZ compression.
    let sources = workers
        .iter_mut()
        .map(|worker| worker.get_mut().unwrap().tangent().values.as_slice())
        .collect::<Vec<_>>();

    tangent.par_iter_mut().enumerate().for_each(|(det, value)| {
        *value = sources.iter().map(|source| source[det]).sum();
    });

    drop(sources);
    workers.par_iter_mut().for_each(|worker| {
        worker.get_mut().unwrap().tangent().values.fill(0.0);
    });
}

/// Drain sparse worker shift tangents into the report accumulator for MPI exchange.
/// # Arguments:
/// - `workers`: Persistent SApply or BApply propagation workers.
/// - `tangent`: Owner-local report tangent and remote contributions.
/// # Returns:
/// - `()`: Adds touched worker values and drains remote updates.
pub(in crate::stochastic) fn collect_shift_tangent<T: TangentWorker>(
    workers: &mut [Mutex<T>],
    tangent: &mut ShiftTangent,
) {
    for worker in workers.iter_mut() {
        let source = worker.get_mut().unwrap().tangent();
        for det in source.changed.drain(..) {
            let dn = source.values[det];
            source.values[det] = 0.0;
            tangent.add(det, dn, true);
        }
        tangent.remote.append(&mut source.remote);
    }
}

/// Perform SApply range-preserving stochastic NOCI propagation.
/// `The initial population is N_0 = S c_0, rescaled to the requested  population 1-norm.`
/// Within each report block, the population vector is held fixed while `ncycles` independent
/// `samples \tilde N^{(a)} = \Phi_c(N) generate pre-overlap changes`
/// `\Delta^{(a)} \approx -\Delta\tau(H - E_s S)\tilde N^{(a)}.`
/// At the end of the report block, the accumulated change is applied as
/// `N'= N + S\sum_{a = 1}^{n_{\text{cycles}}}\Delta^{(a)}.`
/// The same sampled paths produce `B = \partial\Delta/\partial E_s`, and the damped Newton
/// controller uses `d||N'||_1/dE_s = sign(N')^T SB`.
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

    let (isref, scratchsize, run) = super::common::construct_qmc_run(data, c0, ref_indices, world);

    let overlap_factor = SpinFactorisation::new(data);
    let build_overlap_cdfs = matches!(qmc.excitation_gen, ExcitationGen::OverlapWeighted);
    let factor_cache = data.input.wicks.cachedir.as_deref().unwrap_or(".");
    let overlap_factors = overlap_factor.build_overlap_factors(
        data,
        Path::new(factor_cache),
        world.rank(),
        qmc.sapply_factor_tables,
        build_overlap_cdfs,
    );
    if run.irank == 0 {
        let (tables, cdfs) = overlap_factors.storage_bytes();
        let mib = 1024.0 * 1024.0;
        println!(
            "SApply factor storage: {}",
            qmc.sapply_factor_tables.as_str()
        );
        println!("SApply factor tables: {:.3} MiB", tables as f64 / mib);
        println!("SApply proposal CDFs: {:.3} MiB", cdfs as f64 / mib);
        println!(
            "SApply total factor storage: {:.3} MiB",
            (tables + cdfs) as f64 / mib
        );
    }
    let overlap_generator = if let ExcitationGen::OverlapWeighted = qmc.excitation_gen {
        Some(OverlapWeightedGenerator::new(
            data,
            &overlap_factor,
            &overlap_factors,
        ))
    } else {
        None
    };
    let mut workers = (0..rayon::current_num_threads())
        .map(|tid| {
            Mutex::new(NOCIThreadPropagation::with_sizes(
                run.rank_seed ^ tid as u64,
                scratchsize.maxsame,
                scratchsize.maxla,
                scratchsize.maxlb,
            ))
        })
        .collect::<Vec<_>>();
    let mut propagation_result = NOCIPropagationResult::new();
    let mut overlap_scratch = overlap_factor.overlap_scratch();

    // Thread local scratch for Wick's theorem and for MPI communicattion.
    let mut scratch = WickScratchSpin::new();
    let mut mpiscratch = NOCIMPIScratch::new(run.nranks);

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

    if run.irank == 0 {
        println!(
            "Size of Wick's Scratch (MiB): {}",
            std::mem::size_of::<WickScratchSpin<f64>>() as f64 / (1024.0 * 1024.0)
        );
        type ThreadState = (
            Vec<(usize, f64)>,
            Vec<NOCIPopulationUpdate>,
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
    print_header(run.irank, propagator);
    print_initial_row(
        run.irank,
        state.start_report * qmc.ncycles,
        &state,
        data.space.parents[0].e,
        *es,
        propagator,
    );

    let mut population_changes = Vec::new();
    let mut shift_tangent = ShiftTangent::new(run.ndets);
    let mut shift_tangent_changes = Vec::new();
    let mut propagated_shift_tangent = vec![0.0; run.owned.len()];
    let mut pre_overlap_cutoff_hint = 0.0;
    let mut shift_tangent_cutoff_hint = 0.0;
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
                qmc.fri.population_cutoff,
                &run,
                &mut rng,
                &mut sample_chunks,
            );

            propagate_iteration(
                (iter, &state.mc.sampled),
                data,
                &run,
                ShiftSpec::s_apply(*es),
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

        // Form `B = \sum_a dB^{(a)}` once per report. The single-rank path deliberately reduces
        // dense thread vectors because B is almost fully occupied before FRI.
        if run.nranks == 1 {
            reduce_shift_tangent(&mut workers, &mut shift_tangent.values);
        } else {
            collect_shift_tangent(&mut workers, &mut shift_tangent);
        }

        exchange_accumulated_updates(&mut state.mc, &mut mpiscratch, world, &run);
        if run.nranks > 1 {
            exchange_shift_tangent(&mut shift_tangent, &mut mpiscratch, world, &run);
        }

        take_population_changes(&mut state.mc, &mut population_changes);
        if run.nranks > 1 {
            shift_tangent.take_sparse(&mut shift_tangent_changes);
        }

        population_changes.sort_unstable_by_key(|update| update.det);

        // Both report vectors are now coalesced by determinant ownership. Select target-NNZ
        // cutoffs only from these complete realised vectors, before any report-level FRI draw.
        // Choose c_Delta from `E[||Phi_c(Delta)||_0] = \sum_i min(1,|Delta_i|/c)` and compress
        // the complete physical report change before the outer overlap action.
        let mut fri_rng = QmcRng::seed_from_u64(
            run.rank_seed ^ 0xA0761D6478BD642F ^ (report as u64).wrapping_mul(0xE7037ED1A0B428DB),
        );
        let pre_overlap_cutoff = target_cutoff(
            &population_changes,
            qmc.fri.pre_overlap_target_nnz,
            pre_overlap_cutoff_hint,
            |update| update.dn.abs(),
        );
        pre_overlap_cutoff_hint = pre_overlap_cutoff;
        compress_sparse(&mut population_changes, pre_overlap_cutoff, &mut fri_rng);

        // Compress B independently. `E[Phi(B)|B] = B`, hence linearity of S gives
        // `E[S Phi(B)|B] = SB`.
        let mut tangent_rng = QmcRng::seed_from_u64(
            run.rank_seed ^ 0x8EBC6AF09C88C6E3 ^ (report as u64).wrapping_mul(0x589965CC75374CC3),
        );
        let shift_tangent_cutoff = if run.nranks == 1 {
            target_cutoff(
                &shift_tangent.values,
                qmc.fri.shift_tangent_target_nnz,
                shift_tangent_cutoff_hint,
                |value| value.abs(),
            )
        } else {
            target_cutoff(
                &shift_tangent_changes,
                qmc.fri.shift_tangent_target_nnz,
                shift_tangent_cutoff_hint,
                |update| update.dn.abs(),
            )
        };
        shift_tangent_cutoff_hint = shift_tangent_cutoff;
        if run.nranks == 1 {
            compress_dense_to_sparse(
                &mut shift_tangent.values,
                shift_tangent_cutoff,
                &mut tangent_rng,
                &mut shift_tangent_changes,
            );
        } else {
            compress_sparse(
                &mut shift_tangent_changes,
                shift_tangent_cutoff,
                &mut tangent_rng,
            );
        }

        // Apply the physical SApply report update `N' = N + S Delta` using the unchanged
        // single-RHS overlap action from main.
        apply_overlap_population_changes(
            (&mut state.mc.populations, &population_changes),
            data,
            (&overlap_factor, &overlap_factors),
            &run,
            (world, &mut mpiscratch),
            &mut overlap_scratch,
        );

        // Propagate the tangent through the same outer overlap,
        // `\partial N'/\partial E_s = SB`.
        propagated_shift_tangent.fill(0.0);
        apply_overlap_population_changes(
            (&mut propagated_shift_tangent, &shift_tangent_changes),
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

        // `dN_\mathrm{range}/dE_s = sign(N')^T SB` supplies the physical Jacobian for the damped
        // Newton
        // population-control step. EProj is deliberately absent.
        update_shift_tangent(
            &stats,
            &mut state,
            es,
            &propagated_shift_tangent,
            &run,
            world,
            data.input,
        );
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

        print_row(
            run.irank,
            end,
            &state,
            &stats,
            data.space.parents[0].e,
            *es,
            propagator,
        );
    }

    (state.eprojcur, state.mc.excitation_hist)
}
