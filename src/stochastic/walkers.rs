// stochastic/walkers.rs

// Standard library imports.
use std::path::Path;
use std::sync::Mutex;

// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::SeedableRng;

// Crate-root imports.
use crate::input::ExcitationGen;
use crate::noci::{NOCIData, SpinFactorisation};

// Parent/sibling imports.
use super::common::{
    accumulate_generated_updates, coalesce_population_updates, exchange_accumulated_updates,
    population_stats_projected_energy, propagate_iteration, take_population_changes,
};
use super::excit::update_overlap_weight;
use super::fri::sample_populations;
use super::overlapweighted::OverlapWeightedGenerator;
use super::report::{check_stop, print_header, print_initial_row, print_row, write_restart};
use super::restart::{population_representation, read_restart_hdf5};
use super::shift::update_shift;
use super::state::{
    ExcitationHist, MCState, NOCIMPIScratch, NOCIPropagationResult, NOCIThreadPropagation,
    OverlapDerivativeSums, PopulationStats, ProjectedEnergyUpdate, PropagationState, QMCRunInfo,
    QmcRng, ShiftSpec, SparsePopulations,
};

/// Initialise rank-local walker populations from the initial coefficient vector.
/// # Arguments:
/// - `c0`: Initial determinant coefficient vector.
/// - `initial_population`: Requested initial population 1-norm.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// # Returns:
/// - `Vec<f64>`: Rank-local walker populations proportional to `c_0`.
fn initialise_populations(
    c0: &[f64],
    initial_population: f64,
    run: &QMCRunInfo,
    world: &impl Communicator,
) -> Vec<f64> {
    let local_norm = run.owned.iter().map(|&i| c0[i].abs()).sum::<f64>();
    let mut global_norm = 0.0;

    if run.nranks == 1 {
        global_norm = local_norm;
    } else {
        world.all_reduce_into(&local_norm, &mut global_norm, SystemOperation::sum());
    }

    let scale = if global_norm == 0.0 {
        0.0
    } else {
        initial_population / global_norm
    };

    run.owned.iter().map(|&i| scale * c0[i]).collect()
}

/// Apply direct walker-population changes to rank-local populations.
/// # Arguments:
/// - `populations`: Rank-local walker population vector.
/// - `changes`: Coalesced sparse global population changes.
/// - `local_pos`: Global determinant to rank-local position map.
/// # Returns:
/// - `()`: Updates rank-local populations in place.
fn apply_population_changes(
    populations: &mut [f64],
    changes: &[super::state::NOCIPopulationUpdate],
    local_pos: &[usize],
) {
    for update in changes {
        let det = update.det as usize;
        let pos = local_pos[det];

        if pos != usize::MAX {
            populations[pos] += update.dn;
        }
    }
}

/// Perform stochastic NOCI propagation in the walker-population representation.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: Initial determinant coefficient vector.
/// - `es`: Population-control shift energy.
/// - `ref_indices`: Determinants included in the reference-population norm.
/// - `world`: MPI communicator.
/// # Returns:
/// - `(f64, Option<ExcitationHist>)`: Final projected energy and optional
///   spawning-magnitude histogram.
/// # Panics
/// - Panics if restart populations do not match this rank's determinant layout.
pub fn qmc_step(
    data: &NOCIData<'_, f64>,
    c0: &[f64],
    es: &mut f64,
    ref_indices: &[usize],
    world: &impl Communicator,
) -> (f64, Option<ExcitationHist>) {
    // Validate report-aligned checkpointing and construct the rank-local run topology.
    let qmc = data.input.qmc.as_ref().unwrap();

    if let Some(write_restart_interval) = data.input.write.write_restart_interval
        && (write_restart_interval == 0 || write_restart_interval % qmc.ncycles != 0)
    {
        println!("write_restart_interval must be divisible by qmc.ncycles");
        std::process::exit(1);
    }

    let (isref, scratchsize, run) = super::common::construct_qmc_run(data, c0, ref_indices, world);

    // Build overlap-weighted proposal factors only for the corresponding generator.
    let overlap_generation = if let ExcitationGen::OverlapWeighted = qmc.excitation_gen {
        let overlap_factor = SpinFactorisation::new(data);
        let factor_cache = data.input.wicks.cachedir.as_deref().unwrap_or(".");
        let overlap_factors = overlap_factor.build_overlap_factors(
            data,
            Path::new(factor_cache),
            world.rank(),
            qmc.factor_tables,
            true,
        );
        let overlap_generator =
            OverlapWeightedGenerator::new(data, &overlap_factor, &overlap_factors);
        Some((overlap_factors, overlap_generator))
    } else {
        None
    };

    // Map global determinant indices to this rank's compact population vector.
    let mut local_pos = vec![usize::MAX; run.ndets];
    for (k, &det) in run.owned.iter().enumerate() {
        local_pos[det] = k;
    }

    // Allocate thread-local propagation and rank-level communication scratch.
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
    let mut mpiscratch = NOCIMPIScratch::new(run.nranks);

    // Restore a compatible checkpoint or initialise a fresh stochastic population.
    let mut state = if let Some(path) = data.input.write.read_restart.as_deref() {
        if run.irank == 0 {
            println!("Reading restart from {path}");
        }

        let restart = read_restart_hdf5(
            path,
            world,
            run.ndets,
            run.basis_hash,
            population_representation(data.input.prop_ref().propagator),
        )
        .unwrap();
        if restart.populations.len() != run.owned.len() {
            panic!(
                "Restart population length mismatch on rank {}: saved {}, current {}.",
                run.irank,
                restart.populations.len(),
                run.owned.len()
            );
        }
        *es = restart.shift;

        let excitation_hist =
            if data.input.write.write_excitation_hist && restart.excitation_hist.is_none() {
                Some(ExcitationHist::new(-60.0, 1e-12, 100))
            } else {
                restart.excitation_hist
            };

        let mc = MCState {
            populations: restart.populations,
            sampled: SparsePopulations::new(run.ndets),
            delta: vec![0.0; run.ndets],
            changed: Vec::new(),
            excitation_hist,
        };
        let (_, pe) = population_stats_projected_energy(&mc, &isref, &run, world);
        let prev_pop = PopulationStats::new(
            restart.nwprev,
            restart.nrefprev,
            restart.nsampledprev,
            restart.nsampledoprev,
        );
        let overlap_weight = restart.overlap_weight.unwrap_or(qmc.overlap_weight);

        PropagationState::new(
            mc,
            None,
            pe,
            restart.report + 1,
            restart.reached.unwrap_or_else(|| {
                if run.irank == 0 {
                    println!("Warning: legacy restart lacks population-control activation state; inferring from saved population.");
                }
                restart.nwprev >= qmc.target_population
            }),
            prev_pop,
            overlap_weight,
        )
    } else {
        let populations = initialise_populations(c0, qmc.initial_population, &run, world);
        let excitation_hist = if data.input.write.write_excitation_hist {
            Some(ExcitationHist::new(-60.0, 1e-12, 100))
        } else {
            None
        };

        let mut state = PropagationState::new(
            MCState {
                populations,
                sampled: SparsePopulations::new(run.ndets),
                delta: vec![0.0; run.ndets],
                changed: Vec::new(),
                excitation_hist,
            },
            None,
            ProjectedEnergyUpdate { num: 0.0, den: 1.0 },
            0,
            false,
            PopulationStats::new(qmc.initial_population, 0.0, 0.0, 0),
            qmc.overlap_weight,
        );

        let (stats, pe) = population_stats_projected_energy(&state.mc, &isref, &run, world);
        state.pe = pe;
        state.eprojcur = state.pe.num / state.pe.den;
        state.prev_pop = stats;
        state.cur_pop = stats;
        state
    };

    // Print the initial observable row from the restored or newly constructed state.
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
    let mut sample_chunks = Vec::new();
    let mut overlap_derivatives = OverlapDerivativeSums::default();

    // Propagate in report blocks, accumulating spawned changes before MPI exchange.
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
                ShiftSpec {
                    es: *es,
                    es_s: *es,
                    propagator: data.input.prop_ref().propagator,
                },
                (
                    overlap_generation.as_ref().map(|(factors, _)| factors),
                    overlap_generation.as_ref().map(|(_, generator)| generator),
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

        // Route spawned amplitudes to owners, coalesce duplicates, and update populations.
        exchange_accumulated_updates(&mut state.mc, &mut mpiscratch, world, &run);

        take_population_changes(&mut state.mc, &mut population_changes);
        population_changes.sort_unstable_by_key(|update| update.det);
        coalesce_population_updates(&mut population_changes);
        apply_population_changes(&mut state.mc.populations, &population_changes, &local_pos);

        // Recompute projected observables and adapt shift and overlap-generator weight.
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

        // Honour convergence stops and periodic restart checkpoints before reporting.
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
