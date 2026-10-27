// noci/stochastic/report.rs

// External crate imports.
use mpi::topology::Communicator;

// Crate-root imports.
use crate::input::Propagator;
use crate::qmc::excitgen::ExcitationHist;
use crate::qmc::{print_table_header, print_table_row, stop_requested};

// Parent/sibling imports.
use super::restart::{RestartState, write_restart_hdf5};
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
        _ => ["NWalk", "NRef", "-", "-"],
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
        _ => None,
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
    let restart = RestartState {
        report,
        shift,
        nwprev: state.prev_pop.nw,
        nrefprev: state.prev_pop.nref,
        nsampledprev: state.prev_pop.nsampled,
        nsampledoprev: state.prev_pop.nsampledo,
        populations: state.mc.populations.clone(),
        momentum: state.momentum.clone(),
        excitation_hist: state.mc.excitation_hist.clone(),
        base_seed: Some(run.base_seed),
        overlap_weight: Some(state.overlap_weight),
        ndets: run.ndets,
        basis_hash: run.basis_hash,
        representation: Some(run.representation),
        reached: Some(state.reached),
        target_population: Some(run.target_population),
    };

    let restart_path = restart_path.map(String::as_str).unwrap_or("RESTART.H5");

    write_restart_hdf5(restart_path, world, &restart).unwrap();
}

/// Check for a `STOP` file, write a restart if required, and return early.
/// # Arguments:
/// - `report`: Current report number.
/// - `state`: Propagation state containing QMC bookkeeping data.
/// - `shift`: Current population-control shift.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// - `restart_path`: Optional restart file path.
/// # Returns:
/// - `Option<(f64, Option<ExcitationHist>)>`: Final result if stopping.
pub(in crate::noci::stochastic) fn check_stop(
    report: usize,
    state: &mut PropagationState,
    shift: f64,
    run: &QMCRunInfo,
    world: &impl Communicator,
    restart_path: Option<&String>,
) -> Option<(f64, Option<ExcitationHist>)> {
    // Every rank receives one decision so all ranks enter the collective restart write together.
    if !stop_requested(run.irank, run.nranks, world) {
        return None;
    }

    // Transfer owned population and history buffers into the restart record.
    let restart = RestartState {
        report,
        shift,
        nwprev: state.prev_pop.nw,
        nrefprev: state.prev_pop.nref,
        nsampledprev: state.prev_pop.nsampled,
        nsampledoprev: state.prev_pop.nsampledo,
        populations: std::mem::take(&mut state.mc.populations),
        momentum: state.momentum.take(),
        excitation_hist: state.mc.excitation_hist.take(),
        base_seed: Some(run.base_seed),
        overlap_weight: Some(state.overlap_weight),
        ndets: run.ndets,
        basis_hash: run.basis_hash,
        representation: Some(run.representation),
        reached: Some(state.reached),
        target_population: Some(run.target_population),
    };

    let restart_path = restart_path.map(String::as_str).unwrap_or("RESTART.H5");

    write_restart_hdf5(restart_path, world, &restart).unwrap();

    if run.irank == 0 {
        let _ = std::fs::remove_file("STOP");
        println!("STOP detected, wrote {restart_path} and exiting");
    }

    Some((state.eprojcur, restart.excitation_hist))
}
