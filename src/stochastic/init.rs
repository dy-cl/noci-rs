// stochastic/init.rs
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;

use super::common::{find_s, gather_all_populations};
use super::metric::projected_energy;
use super::restart::read_restart_hdf5;
use super::state::{
    ExcitationHist, MCState, MPIScratch, PopulationStats, PopulationUpdate, PropagationState,
    QMCRunInfo, SparsePopulations,
};
use crate::noci::NOCIData;
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::time_call;

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
    mpiscratch: &mut MPIScratch,
) -> Vec<f64> {
    time_call!(crate::timers::stochastic::add_initialise_populations, {
        let local = c0
            .iter()
            .enumerate()
            .filter(|(i, population)| **population != 0.0 && run.det_owner[*i] == run.irank)
            .map(|(i, &population)| PopulationUpdate {
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
    mpi: (&impl Communicator, &mut MPIScratch),
) -> PropagationState {
    let (world, mpiscratch) = mpi;
    let qmc = data.input.qmc.as_ref().unwrap();

    if let Some(path) = data.input.write.read_restart.as_deref() {
        if run.irank == 0 {
            println!("Reading restart from {path}");
        }

        let restart = read_restart_hdf5(path, world).unwrap();

        *es = restart.shift;

        let mc = MCState {
            populations: restart.populations,
            sampled: SparsePopulations::new(run.ndets),
            delta: vec![0.0; run.ndets],
            changed: Vec::new(),
            excitation_hist: restart.excitation_hist,
        };

        let pe = projected_energy(&mc.populations, run, world);

        let prev_pop = PopulationStats::new(restart.nwprev, restart.nrefprev, 0.0, 0);

        return PropagationState::new(
            mc,
            pe,
            restart.report + 1,
            restart.nwprev >= qmc.target_population,
            prev_pop,
        );
    }

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

    world.all_reduce_into(&local, &mut global, SystemOperation::sum());

    let stats = PopulationStats::new(global[0], global[1], 0.0, 0);

    PropagationState::new(mc, pe, 0, false, stats)
}
