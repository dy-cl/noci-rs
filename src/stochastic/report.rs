// stochastic/report.rs
use std::path::Path;

use mpi::topology::Communicator;
use mpi::traits::*;

use super::restart::{RestartState, write_restart_hdf5};
use super::state::{ExcitationHist, PopulationStats, PropagationState, QMCRunInfo};

/// Print the iteration table header on rank zero.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// # Returns:
/// - `()`: Writes the table header to stdout on rank zero.
pub(in crate::stochastic) fn print_header(irank: usize) {
    if irank == 0 {
        println!("{}", "=".repeat(132));
        println!(
            "{:<8} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
            "iter", "E", "Ecorr", "Shift", "Nw", "Nref", "Nsampled", "NsampledOcc",
        );
    }
}

/// Print the initial iteration line on rank zero.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `state`: Propagation state containing QMC statistics.
/// - `e0`: Energy of the first basis determinant.
/// # Returns:
/// - `()`: Writes the initial iteration line to stdout on rank zero.
pub(in crate::stochastic) fn print_initial_row(
    irank: usize,
    state: &PropagationState,
    e0: f64,
) {
    if irank == 0 {
        println!(
            "{:<8} {:>16.12} {:>16.12} {:>16.12} {:>16.6} {:>16.6} {:>16.6} {:>16}",
            0,
            state.eprojcur,
            state.eprojcur - e0,
            0.0,
            state.prev_pop.nw,
            state.prev_pop.nref,
            state.prev_pop.nsampled,
            state.prev_pop.nsampledo,
        );
    }
}

/// Print an iteration line using the current population statistics.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `iter`: Iteration number to print.
/// - `state`: Propagation state containing QMC statistics.
/// - `stats`: Population statistics computed for the current iteration.
/// - `e0`: Energy of the first basis determinant.
/// - `shift`: Population-control shift.
/// # Returns:
/// - `()`: Writes the current iteration line to stdout on rank zero.
pub(in crate::stochastic) fn print_row(
    irank: usize,
    iter: usize,
    state: &PropagationState,
    stats: &PopulationStats,
    e0: f64,
    shift: f64,
) {
    if irank == 0 {
        println!(
            "{:<8} {:>16.12} {:>16.12} {:>16.12} {:>16.6} {:>16.6} {:>16.6} {:>16}",
            iter,
            state.eprojcur,
            state.eprojcur - e0,
            if state.reached { shift } else { 0.0 },
            stats.nw,
            stats.nref,
            stats.nsampled,
            stats.nsampledo,
        );
    }
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
pub(in crate::stochastic) fn check_stop(
    report: usize,
    state: &mut PropagationState,
    shift: f64,
    run: &QMCRunInfo,
    world: &impl Communicator,
    restart_path: Option<&String>,
) -> Option<(f64, Option<ExcitationHist>)> {
    let mut stop = 0;

    if run.irank == 0 && Path::new("STOP").exists() {
        stop = 1;
    }

    world.process_at_rank(0).broadcast_into(&mut stop);

    if stop == 0 {
        return None;
    }

    let restart = RestartState {
        report,
        shift,
        nwprev: state.prev_pop.nw,
        nrefprev: state.prev_pop.nref,
        populations: std::mem::take(&mut state.mc.populations),
        excitation_hist: state.mc.excitation_hist.take(),
        base_seed: Some(run.base_seed),
    };

    let restart_path = restart_path.map(String::as_str).unwrap_or("RESTART.H5");

    write_restart_hdf5(restart_path, world, &restart).unwrap();

    if run.irank == 0 {
        let _ = std::fs::remove_file("STOP");
        println!("STOP detected, wrote {restart_path} and exiting");
    }

    Some((state.eprojcur, restart.excitation_hist))
}
