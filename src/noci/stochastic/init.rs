// noci/stochastic/init.rs

// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;

// Crate-root imports.
use crate::elements::NOCIData;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::qmc::excitgen::ExcitationHist;
use crate::qmc::{PopulationUpdate, gather_updates, read_restart};
use crate::time_call;

// Parent/sibling imports.
use super::common::{find_s, population_stats_projected_energy, projected_energy};
use super::state::{
    MCState, NOCIMPIScratch, PopulationRepresentation, PopulationStats, PropagationState,
    QMCRunInfo, SparsePopulations,
};

/// Initialise the persistent range-safe population vector as `S c0`, the starting populations of
/// the range propagators.
/// # Arguments:
/// - `c0`: Initial determinant coefficient vector.
/// - `initial_population`: Requested initial population 1-norm.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// - `scratch`: Wick scratch storage.
/// - `mpiscratch`: Reusable MPI scratch storage.
/// # Returns:
/// - `Vec<f64>`: Rank-local persistent real populations.
fn initialise_range_populations(
    c0: &[f64],
    initial_population: f64,
    data: &NOCIData<'_, f64>,
    run: &QMCRunInfo,
    world: &impl Communicator,
    scratch: &mut WickScratchSpin<f64>,
    mpiscratch: &mut NOCIMPIScratch,
) -> Vec<f64> {
    time_call!(crate::timers::stochastic::add_initialise_populations, {
        // Each rank submits only its owned nonzero coefficients; gather them
        // before applying the NOCI overlap to the initial state.
        let local = c0
            .iter()
            .enumerate()
            .filter(|(i, population)| **population != 0.0 && run.det_owner[*i] == run.irank)
            .map(|(i, &population)| PopulationUpdate {
                index: i as u64,
                dn: population,
            })
            .collect::<Vec<_>>();

        let global = gather_updates(world, &local, mpiscratch);

        let mut populations = vec![0.0; run.owned.len()];

        // Persistent populations are `N_\gamma = \sum_j S_{\gamma j} c_j` for each
        // determinant owned by this rank.
        for (k, &gamma) in run.owned.iter().enumerate() {
            let mut population = 0.0;

            for update in global {
                population += find_s(data, gamma, update.index as usize, scratch) * update.dn;
            }

            populations[k] = population;
        }

        let local_norm = populations
            .iter()
            .map(|population| population.abs())
            .sum::<f64>();

        let mut global_norm = 0.0;

        world.all_reduce_into(&local_norm, &mut global_norm, SystemOperation::sum());

        // Set the global `L1` population to the requested initial value.
        let scale = initial_population / global_norm;

        for population in &mut populations {
            *population *= scale;
        }

        populations
    })
}

/// Initialise rank-local coefficient populations proportional to `c_0`, the starting populations of
/// the unshifted propagator.
/// # Arguments:
/// - `c0`: Initial determinant coefficient vector.
/// - `initial_population`: Requested initial population 1-norm.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// # Returns:
/// - `Vec<f64>`: Rank-local coefficient populations proportional to `c_0`.
fn initialise_coefficient_populations(
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

/// Initialise projected energy, populations, and population totals across ranks.
/// # Arguments:
/// - `c0`: Initial determinant coefficient vector.
/// - `es`: Population-control shift.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local propagation metadata.
/// - `isref`: Boolean mask specifying reference determinants.
/// - `scratch`: Scratch space for Wick quantities.
/// - `mpi`: MPI communicator and reusable MPI scratch storage.
/// # Returns:
/// - `PropagationState`: Initialised stochastic propagation state.
/// # Panics
/// - Panics if restart loading fails or saved populations do not match this rank's determinant
///   layout.
pub(in crate::noci::stochastic) fn initialise_qmc_state(
    c0: &[f64],
    es: &mut f64,
    data: &NOCIData<'_, f64>,
    run: &QMCRunInfo,
    isref: &[bool],
    scratch: &mut WickScratchSpin<f64>,
    mpi: (&impl Communicator, &mut NOCIMPIScratch),
) -> PropagationState {
    // Resolve shared MPI scratch and stochastic controls once for both startup paths.
    let (world, mpiscratch) = mpi;
    let qmc = data.input.qmc.as_ref().unwrap();

    // Restore a checkpoint when requested, preserving its report and controller history.
    if let Some(path) = data.input.write.read_restart.as_deref() {
        if run.irank == 0 {
            println!("Reading restart from {path}");
        }

        // Keep the saved populations of the determinants this rank owns, in its owned order.
        let identity = (run.representation.as_str(), run.basis_hash);
        let owns = |det: u64, _| run.det_owner[det as usize] == run.irank;
        let restart = read_restart(path, identity, owns, world).unwrap();

        let mut populations = vec![0.0; run.owned.len()];

        for update in &restart.populations {
            let position = run.owned.binary_search(&(update.index as usize)).unwrap();
            populations[position] = update.dn;
        }

        // Restore the shift, the controller history and the histogram, creating a requested one.
        *es = restart.shift;
        let excitation_hist = restart.histogram.or_else(|| {
            data.input
                .write
                .write_excitation_hist
                .then(|| ExcitationHist::new(-60.0, 1e-12, 100))
        });
        let mc = MCState {
            populations,
            sampled: SparsePopulations::new(run.ndets),
            delta: vec![0.0; run.ndets],
            changed: Vec::new(),
            excitation_hist,
        };
        let previous = &restart.values["previous"];
        let prev_pop =
            PopulationStats::new(previous[0], previous[1], previous[2], previous[3] as i64);

        // Resume at the report following the restart, with the projected energy recomputed.
        let pe = projected_energy(&mc.populations, run, world);
        return PropagationState::new(
            mc,
            pe,
            restart.report + 1,
            restart.reached,
            prev_pop,
            restart.values["overlap_weight"][0],
        );
    }

    // Without a checkpoint, start from the populations of the propagator's representation.
    if run.irank == 0 {
        println!("Initialising populations.....");
    }
    let populations = match run.representation {
        PopulationRepresentation::Range => initialise_range_populations(
            c0,
            qmc.initial_population,
            data,
            run,
            world,
            scratch,
            mpiscratch,
        ),
        PopulationRepresentation::Coefficient => {
            initialise_coefficient_populations(c0, qmc.initial_population, run, world)
        }
    };

    // Allocate optional diagnostics and empty per-iteration population buffers.
    let excitation_hist = if data.input.write.write_excitation_hist {
        Some(ExcitationHist::new(-60.0, 1e-12, 100))
    } else {
        None
    };

    let mc = MCState {
        populations,
        sampled: SparsePopulations::new(run.ndets),
        delta: vec![0.0; run.ndets],
        changed: Vec::new(),
        excitation_hist,
    };

    // The initial statistics start the shift controller, with population control inactive.
    let (stats, pe) = population_stats_projected_energy(&mc, isref, run, world);
    PropagationState::new(mc, pe, 0, false, stats, qmc.overlap_weight)
}
