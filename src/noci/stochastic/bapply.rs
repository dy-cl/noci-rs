// noci/stochastic/bapply.rs

// Standard library imports.
use std::path::Path;
use std::sync::Mutex;

// External crate imports.
use mpi::topology::Communicator;
use rand::SeedableRng;

// Crate-root imports.
use crate::determinant::{AuxiliarySpace, NOCIIndex};
use crate::elements::NOCIData;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::input::ExcitationGen;
use crate::noci::{OverlapFactors, SpinFactorisation};
use crate::qmc::excitgen::{ExcitationHist, OrthogonalUniformGenerator};
use crate::qmc::{
    QmcRng, coalesce, compress_sparse, exchange_updates, gather_updates, pack_updates, stream,
    target_cutoff,
};

// Parent/sibling imports.
use super::common::{
    construct_qmc_run, population_stats_projected_energy, propagate_iteration_auxiliary,
};
use super::excit::{AuxiliaryGenerator, pchb_generator};
use super::fri::{compress_dense_to_sparse, sample_populations};
use super::init::initialise_qmc_state;
use super::report::{check_stop, print_header, print_initial_row, print_row, write_restart};
use super::shift::update_shift_tangent;
use super::state::{
    AuxiliaryMPIScratch, AuxiliaryPopulationUpdate, AuxiliaryPropagationResult,
    AuxiliaryThreadPropagation, NOCIMPIScratch, NOCIPopulationUpdate, ShiftTangent,
};

/// Accumulate one cycle's `\chi_D^P` events and retain remote events for report exchange.
/// # Arguments:
/// - `result`: Ownership-separated worker propagation results.
/// - `updates`: Rank-local report residual accumulator.
/// - `scratch`: Persistent MPI exchange scratch for remote events.
/// - `histogram`: Optional excitation-magnitude histogram.
/// # Returns
/// - `()`: Drains cycle-local auxiliary results into report accumulators.
fn accumulate_generated_updates_auxiliary(
    result: &mut AuxiliaryPropagationResult,
    updates: &mut Vec<AuxiliaryPopulationUpdate>,
    scratch: &mut AuxiliaryMPIScratch,
    histogram: &mut Option<ExcitationHist>,
) {
    updates.append(&mut result.local);
    scratch.send_ranked.append(&mut result.remote);
    if let Some(histogram) = histogram.as_mut() {
        for sample in result.samples.drain(..) {
            histogram.add(sample);
        }
    } else {
        result.samples.clear();
    }
}

/// Exchange pre-routed auxiliary events and combine them with local report updates.
/// # Arguments:
/// - `local`: Rank-owned report residual updates.
/// - `scratch`: Reusable auxiliary MPI storage.
/// - `world`: MPI communicator.
/// # Returns
/// - `()`: Places complete owner-local contributions in `scratch.recv_contig`.
fn redistribute_population_updates_auxiliary(
    local: &mut Vec<AuxiliaryPopulationUpdate>,
    scratch: &mut AuxiliaryMPIScratch,
    world: &impl Communicator,
) {
    // Send every routed event, unmerged, to its owner and append the already-owned local
    // events to the same owner-local buffer.
    pack_updates(scratch.send_counts.len(), scratch, false);
    exchange_updates(world, scratch);
    scratch.recv_contig.append(local);
    scratch.send_ranked.clear();
}

/// Report persistent BApply factor backing and lazily discovered physical components.
/// # Arguments:
/// - `rank`: Current MPI rank; only rank zero writes storage output.
/// - `mode`: Requested shared `SNOCIStorage` backend.
/// - `initial`: Initial factor-table backing bytes.
/// - `factors`: Current persistent cross-parent factor tables.
/// # Returns
/// - `()`: Writes actual initial, final, and peak backing and added component counts.
fn print_bapply_storage(
    rank: usize,
    mode: crate::input::SNOCIStorage,
    initial: usize,
    factors: &OverlapFactors,
) {
    if rank != 0 {
        return;
    }
    let (final_bytes, _) = factors.storage_bytes();
    let (alpha, beta) = factors.added_components();
    let mib = 1024.0 * 1024.0;
    println!("BApply factor storage: {}", mode.as_str());
    println!(
        "BApply initial factor tables: {:.3} MiB",
        initial as f64 / mib
    );
    println!(
        "BApply final factor tables: {:.3} MiB",
        final_bytes as f64 / mib
    );
    println!("BApply added alpha factor columns: {alpha}");
    println!("BApply added beta factor columns: {beta}");
    println!(
        "BApply peak factor storage: {:.3} MiB",
        final_bytes as f64 / mib
    );
}

/// Perform BApply stochastic range propagation.
/// `\chi \simeq -dt(\hat H-E_s)B N` is sampled in parent-orthogonal determinant spaces and the
/// persistent real range population is updated as `N_{r+1} = N_r + B^\dagger\chi_r`, or with
/// report-level heavy-ball momentum as `V_{r+1} = \beta V_r + B^\dagger\chi_r` and
/// `N_{r+1} = N_r + V_{r+1}`.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: Initial determinant coefficient vector.
/// - `es`: Population-control shift energy.
/// - `ref_indices`: Determinants included in the reference-population norm.
/// - `world`: MPI communicator.
/// # Returns
/// - `(f64, Option<ExcitationHist>)`: Final projected energy and optional spawning histogram.
/// # References
/// - Polyak, *USSR Comput. Math. Math. Phys.* **4**, 1–17 (1964),
///   [doi:10.1016/0041-5553(64)90137-5](https://doi.org/10.1016/0041-5553(64)90137-5),
///   for the optional heavy-ball momentum update.
/// # Panics
/// - Panics for unsupported excitation generation, invalid restart intervals, nonorthogonal
///   parent orbitals, missing MO caches or Wick data, or an uninitialised momentum state.
pub fn qmc_step(
    data: &NOCIData<'_, f64>,
    c0: &[f64],
    es: &mut f64,
    ref_indices: &[usize],
    world: &impl Communicator,
) -> (f64, Option<ExcitationHist>) {
    // Validate BApply-specific generator, checkpoint, and parent-orbital requirements.
    let qmc = data.input.qmc.as_ref().unwrap();
    if !matches!(
        qmc.excitation_gen,
        ExcitationGen::Uniform | ExcitationGen::Pchb
    ) {
        panic!("BApply supports excitation_gen = \"uniform\" or \"pchb\"");
    }
    if let Some(interval) = data.input.write.write_restart_interval
        && (interval == 0 || interval % qmc.ncycles != 0)
    {
        panic!("write_restart_interval must be divisible by qmc.ncycles");
    }
    let mocache = data
        .mocache
        .expect("BApply requires parent MO integral caches");
    if mocache.iter().any(|cache| !cache.orthogonal_slater_condon) {
        panic!("BApply requires orthonormal orbitals within every parent MO basis");
    }
    // Build retained/auxiliary spin spaces and reusable overlap-factor storage.
    let (isref, _, run) = construct_qmc_run(data, c0, ref_indices, world);
    let factorisation = SpinFactorisation::new(data);
    let auxiliary = AuxiliarySpace::new(data.space);
    if factorisation.nparents() > 1 && data.wicks.is_none() {
        panic!("BApply cross-parent B^dagger requires Wick intermediates");
    }
    // Select the parent-orthogonal proposal; PCHB tables are built once per parent.
    let generator = if qmc.excitation_gen == ExcitationGen::Pchb {
        let pchb = pchb_generator(data, &mocache[..factorisation.nparents()]);
        if run.irank == 0 {
            println!(
                "PCHB excitation tables: {:.3} MiB",
                pchb.bytes() as f64 / (1024.0 * 1024.0)
            );
        }
        AuxiliaryGenerator::Pchb(pchb)
    } else {
        AuxiliaryGenerator::Uniform(OrthogonalUniformGenerator::new(
            data.space.occupations(NOCIIndex(0)),
            &mocache[data.space.state(NOCIIndex(0)).parent],
        ))
    };
    let factor_cache = data.input.wicks.cachedir.as_deref().unwrap_or(".");
    let mut overlap_factors = factorisation.build_overlap_factors(
        data,
        Path::new(factor_cache),
        world.rank(),
        qmc.bapply_factor_tables,
        false,
    );
    let initial_factor_bytes = overlap_factors.storage_bytes().0;
    // Allocate persistent contraction, communication, and propagation state.
    let mut overlap_scratch = factorisation.overlap_scratch();
    let mut auxiliary_scratch = factorisation.auxiliary_overlap_scratch();
    let mut mpi = NOCIMPIScratch::new(run.nranks);
    let mut auxiliary_mpi = AuxiliaryMPIScratch::new(run.nranks);
    let mut wick = WickScratchSpin::new();
    let mut state = initialise_qmc_state(c0, es, data, &run, &isref, &mut wick, (world, &mut mpi));
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

    // Give each Rayon worker independent RNG and auxiliary propagation scratch.
    let mut workers = (0..rayon::current_num_threads())
        .map(|tid| {
            Mutex::new(AuxiliaryThreadPropagation::new(
                run.rank_seed ^ tid as u64,
                factorisation.nparents(),
            ))
        })
        .collect::<Vec<_>>();
    let mut propagation_result = AuxiliaryPropagationResult::new();
    let mut local_updates = Vec::new();
    let mut shift_tangent = ShiftTangent::new(run.ndets);
    let mut tangent_updates = Vec::<NOCIPopulationUpdate>::new();
    let mut propagated_tangent = vec![0.0; run.owned.len()];
    let mut sample_chunks = Vec::new();
    let mut chi_cutoff_hint = 0.0;
    let mut tangent_cutoff_hint = 0.0;

    // Advance one report block at a time so communication and compression remain amortised.
    for report in state.start_report..qmc.nreports {
        local_updates.clear();
        auxiliary_mpi.send_ranked.clear();
        for cycle in 0..qmc.ncycles {
            let iteration = report * qmc.ncycles + cycle;
            let mut rng = stream(run.rank_seed, 0xD1B54A32D192ED03, iteration as u64);
            sample_populations(
                &state.mc.populations,
                &mut state.mc.sampled,
                qmc.fri.population_cutoff,
                &run,
                &mut rng,
                &mut sample_chunks,
            );

            propagate_iteration_auxiliary(
                (iteration, &state.mc.sampled),
                data,
                &run,
                *es,
                (&generator, &auxiliary),
                &mut workers,
                &mut propagation_result,
            );
            accumulate_generated_updates_auxiliary(
                &mut propagation_result,
                &mut local_updates,
                &mut auxiliary_mpi,
                &mut state.mc.excitation_hist,
            );
        }
        // Coalesce generated auxiliary amplitudes on their owning MPI ranks.
        local_updates.sort_unstable_by_key(|update| update.det);
        coalesce(&mut local_updates);
        let mut owner_updates = if run.nranks == 1 {
            // Preserve allocation across reports while avoiding MPI scratch and a second coalesce.
            std::mem::take(&mut local_updates)
        } else {
            redistribute_population_updates_auxiliary(
                &mut local_updates,
                &mut auxiliary_mpi,
                world,
            );

            let mut updates = std::mem::take(&mut auxiliary_mpi.recv_contig);
            updates.sort_unstable_by_key(|update| update.det);
            coalesce(&mut updates);
            updates
        };
        // Compress the auxiliary vector before applying `B^dagger` to physical populations.
        let chi_cutoff = target_cutoff(
            &owner_updates,
            qmc.fri.pre_overlap_target_nnz,
            chi_cutoff_hint,
            |update| update.dn.abs(),
        );
        chi_cutoff_hint = chi_cutoff;
        let mut fri_rng = QmcRng::seed_from_u64(run.rank_seed ^ 0xA0761D6478BD642F ^ report as u64);
        compress_sparse(&mut owner_updates, chi_cutoff, &mut fri_rng);

        let physical_updates: &[AuxiliaryPopulationUpdate] = if run.nranks == 1 {
            &owner_updates
        } else {
            gather_updates(world, &owner_updates, &mut auxiliary_mpi)
        };

        if qmc.momentum_beta == 0.0 {
            factorisation.apply_auxiliary_overlap_sparse(
                &mut state.mc.populations,
                &run.owned,
                physical_updates
                    .iter()
                    .map(|update| (update.index(), update.dn)),
                (data, &auxiliary),
                &mut overlap_factors,
                &mut auxiliary_scratch,
            );
        } else {
            let Some(momentum) = state.momentum.as_mut() else {
                panic!("BApply heavy-ball momentum state is not initialised");
            };

            // Heavy-ball update: `V_{r+1} = \beta V_r + \Delta_r`, then
            // `N_{r+1} = N_r + V_{r+1}`.
            for velocity in momentum.iter_mut() {
                *velocity *= qmc.momentum_beta;
            }

            factorisation.apply_auxiliary_overlap_sparse(
                momentum.as_mut_slice(),
                &run.owned,
                physical_updates
                    .iter()
                    .map(|update| (update.index(), update.dn)),
                (data, &auxiliary),
                &mut overlap_factors,
                &mut auxiliary_scratch,
            );

            for (population, velocity) in state.mc.populations.iter_mut().zip(momentum.iter()) {
                *population += *velocity;
            }
        }

        // Collect and compress `d chi / d E_s` in the representation local to this MPI mode.
        if run.nranks == 1 {
            local_updates = owner_updates;
        } else {
            auxiliary_mpi.recv_contig = owner_updates;
        }

        if run.nranks == 1 {
            super::sapply::reduce_shift_tangent(&mut workers, &mut shift_tangent.values);
        } else {
            super::sapply::collect_shift_tangent(&mut workers, &mut shift_tangent);
            super::sapply::exchange_shift_tangent(&mut shift_tangent, &mut mpi, world, &run);
            shift_tangent.take_sparse(&mut tangent_updates);
        }
        let tangent_cutoff = if run.nranks == 1 {
            target_cutoff(
                &shift_tangent.values,
                qmc.fri.shift_tangent_target_nnz,
                tangent_cutoff_hint,
                |value| value.abs(),
            )
        } else {
            target_cutoff(
                &tangent_updates,
                qmc.fri.shift_tangent_target_nnz,
                tangent_cutoff_hint,
                |update| update.dn.abs(),
            )
        };
        tangent_cutoff_hint = tangent_cutoff;
        if run.nranks == 1 {
            compress_dense_to_sparse(
                &mut shift_tangent.values,
                tangent_cutoff,
                &mut fri_rng,
                &mut tangent_updates,
            );
        } else {
            compress_sparse(&mut tangent_updates, tangent_cutoff, &mut fri_rng);
        }
        propagated_tangent.fill(0.0);
        // `T = \partial N'/\partial E_s = dt S \sum_a\tilde N^{(a)}` for BApply.
        // The shared tangent controller therefore sees the complete physical population derivative.
        super::sapply::apply_overlap_population_changes(
            (&mut propagated_tangent, &tangent_updates),
            data,
            (&factorisation, &overlap_factors),
            &run,
            (world, &mut mpi),
            &mut overlap_scratch,
        );

        // Update projected-energy statistics and the tangent-aware population-control shift.
        let end = (report + 1) * qmc.ncycles;
        let (stats, pe) = population_stats_projected_energy(&state.mc, &isref, &run, world);
        state.pe = pe;
        state.eprojcur = state.pe.num / state.pe.den;
        state.cur_pop = stats;
        update_shift_tangent(
            &stats,
            &mut state,
            es,
            &propagated_tangent,
            &run,
            world,
            data.input,
        );
        // Honour convergence stops and periodic restart checkpoints before reporting.
        if let Some(result) = check_stop(
            report,
            &mut state,
            *es,
            &run,
            world,
            data.input.write.write_restart.as_ref(),
        ) {
            print_bapply_storage(
                run.irank,
                qmc.bapply_factor_tables,
                initial_factor_bytes,
                &overlap_factors,
            );
            return result;
        }
        if let Some(interval) = data.input.write.write_restart_interval
            && end.is_multiple_of(interval)
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

    // Report final factor-table growth and return the last projected estimate.
    print_bapply_storage(
        run.irank,
        qmc.bapply_factor_tables,
        initial_factor_bytes,
        &overlap_factors,
    );
    (state.eprojcur, state.mc.excitation_hist)
}
