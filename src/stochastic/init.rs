// stochastic/init.rs
// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;

// Crate-root imports.
use crate::noci::NOCIData;
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::time_call;

// Parent/sibling imports.
use super::common::{find_s, gather_all_populations, projected_energy};
use super::restart::{population_representation, read_restart_hdf5};
use super::state::{
    ExcitationHist, MCState, NOCIMPIScratch, NOCIPopulationUpdate, PopulationStats,
    PropagationState, QMCRunInfo, SparsePopulations,
};

/// Initialise the persistent range-safe population vector as `S c0`.
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
pub(in crate::stochastic) fn initialise_populations(
    c0: &[f64],
    initial_population: f64,
    data: &NOCIData<'_, f64>,
    run: &QMCRunInfo,
    world: &impl Communicator,
    scratch: &mut WickScratchSpin<f64>,
    mpiscratch: &mut NOCIMPIScratch,
) -> Vec<f64> {
    time_call!(crate::timers::stochastic::add_initialise_populations, {
        let local = c0
            .iter()
            .enumerate()
            .filter(|(i, population)| **population != 0.0 && run.det_owner[*i] == run.irank)
            .map(|(i, &population)| NOCIPopulationUpdate {
                det: i as u64,
                dn: population,
            })
            .collect::<Vec<_>>();

        let global = gather_all_populations(world, &local, mpiscratch);

        let mut populations = vec![0.0; run.owned.len()];

        for (k, &gamma) in run.owned.iter().enumerate() {
            let mut population = 0.0;

            for update in global {
                population += find_s(data, gamma, update.det as usize, scratch) * update.dn;
            }

            populations[k] = population;
        }

        let local_norm = populations
            .iter()
            .map(|population| population.abs())
            .sum::<f64>();

        let mut global_norm = 0.0;

        world.all_reduce_into(&local_norm, &mut global_norm, SystemOperation::sum());

        let scale = initial_population / global_norm;

        for population in &mut populations {
            *population *= scale;
        }

        populations
    })
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
pub(in crate::stochastic) fn initialise_qmc_state(
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

        let restart = read_restart_hdf5(
            path,
            world,
            run.ndets,
            run.basis_hash,
            population_representation(data.input.prop_ref().propagator),
        )
        .unwrap();
        // The saved rank-local shard must match the current MPI determinant partition.
        if restart.populations.len() != run.owned.len() {
            panic!(
                "Restart population length mismatch on rank {}: saved {}, current {}.",
                run.irank,
                restart.populations.len(),
                run.owned.len()
            );
        }

        // Restore shift and histogram state, creating a requested legacy-missing histogram.
        *es = restart.shift;

        let excitation_hist =
            if data.input.write.write_excitation_hist && restart.excitation_hist.is_none() {
                Some(ExcitationHist::new(-60.0, 1e-12, 100))
            } else {
                restart.excitation_hist
            };

        // Reconstruct transient sampling buffers around the persistent population vector.
        let mc = MCState {
            populations: restart.populations,
            sampled: SparsePopulations::new(run.ndets),
            delta: vec![0.0; run.ndets],
            changed: Vec::new(),
            excitation_hist,
        };

        // Recompute projected energy but retain saved population-controller observables.
        let pe = projected_energy(&mc.populations, run, world);

        let prev_pop = PopulationStats::new(
            restart.nwprev,
            restart.nrefprev,
            restart.nsampledprev,
            restart.nsampledoprev,
        );
        let overlap_weight = restart.overlap_weight.unwrap_or(qmc.overlap_weight);

        // Resume at the report following the checkpoint, inferring legacy activation if needed.
        return PropagationState::new(
            mc,
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
        );
    }

    // Without a checkpoint, construct and globally normalise fresh initial populations.
    if run.irank == 0 {
        println!("Initialising populations.....");
    }

    let populations = initialise_populations(
        c0,
        qmc.initial_population,
        data,
        run,
        world,
        scratch,
        mpiscratch,
    );

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

    // Evaluate the initial projected energy and rank-local total/reference norms.
    let pe = projected_energy(&mc.populations, run, world);

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

    let local = [nw_local, nref_local];
    let mut global = [0.0; 2];

    // Sum population norms across ranks for the initial shift-controller state.
    world.all_reduce_into(&local, &mut global, SystemOperation::sum());

    let stats = PopulationStats::new(global[0], global[1], 0.0, 0);

    // Start a fresh run at report zero with population control inactive.
    PropagationState::new(mc, pe, 0, false, stats, qmc.overlap_weight)
}
