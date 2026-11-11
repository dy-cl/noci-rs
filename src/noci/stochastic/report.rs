// noci/stochastic/report.rs

// Standard library imports.
use std::collections::BTreeMap;

// External crate imports.
use mpi::topology::Communicator;

// Crate-root imports.
use crate::input::Propagator;
use crate::qmc::{
    PopulationUpdate, Restart, print_table_header, print_table_row,
    write_restart as qmc_write_restart,
};

// Parent/sibling imports.
use super::state::{PopulationStats, PropagationState, QMCRunInfo};

/// Print the iteration table header on rank zero.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `propagator`: Propagator used by the stochastic calculation.
/// # Returns:
/// - `()`: Writes the table header to stdout on rank zero.
pub(in crate::noci::stochastic) fn print_header(
    irank: usize,
    propagator: Propagator,
) {
    let populations = match propagator {
        Propagator::SApply | Propagator::BApply => ["NRange", "NRangeRef", "NSample", "NSampleOcc"],
        Propagator::Unshifted => ["NWalk", "NRef", "-", "-"],
    };
    print_table_header(irank, populations);
}

/// Print an iteration line from given population statistics.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `iter`: Iteration number to print.
/// - `state`: Propagation state containing QMC statistics.
/// - `stats`: Population statistics to print.
/// - `energies`: Energy of the first basis determinant and the population-control shift.
/// - `propagator`: Propagator used by the stochastic calculation.
/// # Returns:
/// - `()`: Writes the iteration line to stdout on rank zero.
fn print_stats_row(
    irank: usize,
    iter: usize,
    state: &PropagationState,
    stats: &PopulationStats,
    energies: (f64, f64),
    propagator: Propagator,
) {
    let (e0, shift) = energies;
    let shift = if state.reached { shift } else { 0.0 };
    let sampled = match propagator {
        Propagator::SApply | Propagator::BApply => Some((stats.nsampled, stats.nsampledo)),
        Propagator::Unshifted => None,
    };
    print_table_row(
        irank,
        iter,
        (state.pe.num, state.pe.den),
        (e0, shift),
        (
            stats.nw,
            stats.nref,
            sampled.map(|(n, _)| n),
            sampled.map(|(_, o)| o),
        ),
    );
}

/// Print the initial iteration line on rank zero.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `iter`: Iteration number to print.
/// - `state`: Propagation state containing QMC statistics.
/// - `e0`: Energy of the first basis determinant.
/// - `shift`: Population-control shift.
/// - `propagator`: Propagator used by the stochastic calculation.
/// # Returns:
/// - `()`: Writes the initial iteration line to stdout on rank zero.
pub(in crate::noci::stochastic) fn print_initial_row(
    irank: usize,
    iter: usize,
    state: &PropagationState,
    e0: f64,
    shift: f64,
    propagator: Propagator,
) {
    print_stats_row(irank, iter, state, &state.prev_pop, (e0, shift), propagator);
}

/// Print an iteration line using the current population statistics.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `iter`: Iteration number to print.
/// - `state`: Propagation state containing QMC statistics.
/// - `stats`: Population statistics computed for the current iteration.
/// - `e0`: Energy of the first basis determinant.
/// - `shift`: Population-control shift.
/// - `propagator`: Propagator used by the stochastic calculation.
/// # Returns:
/// - `()`: Writes the current iteration line to stdout on rank zero.
pub(in crate::noci::stochastic) fn print_row(
    irank: usize,
    iter: usize,
    state: &PropagationState,
    stats: &PopulationStats,
    e0: f64,
    shift: f64,
    propagator: Propagator,
) {
    print_stats_row(irank, iter, state, stats, (e0, shift), propagator);
}

/// Write the current stochastic propagation state to a restart file.
/// # Arguments:
/// - `report`: Current report number.
/// - `state`: Propagation state containing QMC bookkeeping data.
/// - `shift`: Current population-control shift.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// - `restart_path`: Optional restart file path.
/// # Returns:
/// - `()`: Writes the restart file.
pub(in crate::noci::stochastic) fn write_restart(
    report: usize,
    state: &PropagationState,
    shift: f64,
    run: &QMCRunInfo,
    world: &impl Communicator,
    restart_path: Option<&String>,
) {
    // Owned populations keyed by determinant, and the controller history the propagators need.
    let populations = run
        .owned
        .iter()
        .zip(&state.mc.populations)
        .filter(|&(_, &n)| n != 0.0)
        .map(|(&det, &n)| PopulationUpdate {
            index: det as u64,
            dn: n,
        })
        .collect();
    let previous = &state.prev_pop;
    let values = BTreeMap::from([
        (
            "previous".to_string(),
            vec![
                previous.nw,
                previous.nref,
                previous.nsampled,
                previous.nsampledo as f64,
            ],
        ),
        ("overlap_weight".to_string(), vec![state.overlap_weight]),
    ]);
    let restart = Restart {
        report,
        shift,
        reached: state.reached,
        seed: run.base_seed,
        populations,
        values,
        histogram: state.mc.excitation_hist.clone(),
    };

    let restart_path = restart_path.map(String::as_str).unwrap_or("RESTART.H5");
    let identity = (run.representation.as_str(), run.basis_hash);
    qmc_write_restart(restart_path, identity, &restart, world).unwrap();
}
