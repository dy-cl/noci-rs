// stochastic/walkers.rs
use std::sync::Mutex;

use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;

use super::common::{coalesce_population_updates, find_hs, max_scratch_sizes};
use super::metric::{
    accumulate_generated_updates, exchange_accumulated_updates, population_stats_projected_energy,
    sample_populations, take_population_changes, update_shift,
};
use super::report::{check_stop, print_header, print_initial_row, print_row};
use super::state::{
    ExcitationHist, MCState, MPIScratch, PopulationStats, ProjectedEnergyUpdate, PropagationResult,
    PropagationState, QMCRunInfo, ScratchSize, ShiftSpec, SparsePopulations, ThreadPropagation,
    owner,
};
use crate::noci::NOCIData;
use crate::nonorthogonalwicks::WickScratchSpin;

/// Initialise rank-local walker populations from the initial coefficient vector.
/// # Arguments:
/// - `c0`: Initial determinant coefficient vector.
/// - `initial_population`: Requested initial population 1-norm.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// # Returns:
/// - `Vec<f64>`: Rank-local walker populations proportional to \(c_0\).
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
    changes: &[super::state::PopulationUpdate],
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
pub fn qmc_step(
    data: &NOCIData<'_, f64>,
    c0: &[f64],
    es: &mut f64,
    ref_indices: &[usize],
    world: &impl Communicator,
) -> (f64, Option<ExcitationHist>) {
    let qmc = data.input.qmc.as_ref().unwrap();

    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let ndets = data.basis.len();

    let mut isref = vec![false; ndets];
    for &i in ref_indices {
        isref[i] = true;
    }

    let base_seed = qmc.seed.unwrap_or_else(rand::random);
    let rank_seed = base_seed.wrapping_add((irank as u64).wrapping_mul(0x9E3779B9));

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

    let run = QMCRunInfo {
        irank,
        nranks,
        ndets,
        det_owner,
        owned,
        base_seed,
        rank_seed,
        reference_hs,
        diagonal_hs,
    };

    let mut local_pos = vec![usize::MAX; ndets];
    for (k, &det) in run.owned.iter().enumerate() {
        local_pos[det] = k;
    }

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
    let mut mpiscratch = MPIScratch::new(run.nranks);

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
        ProjectedEnergyUpdate { num: 0.0, den: 1.0 },
        0,
        false,
        PopulationStats::new(qmc.initial_population, 0.0, 0.0, 0),
    );

    let (stats, pe) = population_stats_projected_energy(&state.mc, &isref, &run, world);
    state.pe = pe;
    state.eprojcur = state.pe.num / state.pe.den;
    state.prev_pop = stats;
    state.cur_pop = stats;

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

            super::metric::propagate_iteration(
                iter,
                &state.mc.sampled,
                data,
                &run,
                ShiftSpec {
                    es: *es,
                    es_s: *es,
                    propagator: data.input.prop_ref().propagator,
                },
                &mut workers,
                &mut propagation_result,
            );

            accumulate_generated_updates(
                &mut state.mc,
                &mut propagation_result,
                data.input,
                &mut mpiscratch,
            );
        }

        exchange_accumulated_updates(&mut state.mc, &mut mpiscratch, world, &run);

        let mut population_changes = take_population_changes(&mut state.mc);
        population_changes.sort_unstable_by_key(|update| update.det);
        coalesce_population_updates(&mut population_changes);
        apply_population_changes(&mut state.mc.populations, &population_changes, &local_pos);

        let end = (report + 1) * qmc.ncycles;
        let (stats, pe) = population_stats_projected_energy(&state.mc, &isref, &run, world);
        state.pe = pe;
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
