// stochastic/report.rs
use std::path::Path;

use mpi::topology::Communicator;
use mpi::traits::*;

use super::state::{PropagationState, Walkers, ExcitationHist};
use super::restart::RestartState;

use super::restart::write_restart_hdf5;
use super::state::{QMCRunInfo, PopulationStats, Shifts};

/// Print the iteration table header on rank zero.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// # Returns
/// - `()`: Writes the table header to stdout on rank zero.
pub(in crate::stochastic) fn print_header(irank: usize) {
    if irank == 0 {
        println!("{}", "=".repeat(100));
        println!("{:<6} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
                 "iter", "E", "Ecorr", "Shift (Es)", "Shift (EsS)", "Nw (||C||)", "Nref (||C||)", "Nw (||SC||)", "Nref (||SC||)", "Nocc");
    }
}

/// Print the initial iteration line on rank zero.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `state`: Propagation state containing QMC stats.
/// - `e0`: Energy of the first basis determinant.
/// # Returns
/// - `()`: Writes the initial iteration line to stdout on rank zero.
pub(in crate::stochastic) fn print_initial_row(irank: usize, state: &PropagationState, e0: f64) {
    if irank == 0 {
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
                 0, state.eprojcur, state.eprojcur - e0, 0.0, 0.0, state.prev_pop.nwc as f64, state.prev_pop.nrefc as f64, 
                 state.prev_pop.nwsc, state.prev_pop.nrefsc, state.prev_pop.noccdets);
    }
}

/// Print an iteration line using the cached population statistics stored in the run state.
/// This is used when no population changes occurred during an iteration and so we exit early.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `iter`: Iteration number to print.
/// - `state`: Propagation state containing QMC stats.
/// - `e0`: Energy of the first basis determinant.
/// - `es`: Non-overlap transformed shift energy.
/// # Returns
/// - `()`: Writes the cached iteration line to stdout on rank zero.
pub(in crate::stochastic) fn print_cached_row(irank: usize, iter: usize, state: &PropagationState, e0: f64, es: f64) {
    let es_corr = if state.reached_c {es - e0} else {0.0};
    let es_s_corr = if state.reached_sc {state.es_s - e0} else {0.0};

    if irank == 0 {
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
                 iter, state.eprojcur, state.eprojcur - e0, es_corr, es_s_corr, state.cur_pop.nwc as f64, 
                 state.cur_pop.nrefc as f64, state.cur_pop.nwsc, state.cur_pop.nrefsc, state.cur_pop.noccdets);
    }
}

/// Print an iteration line using the current population statistics.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `iter`: Iteration number to print.
/// - `state`: Propagation state containing QMC stats.
/// - `stats`: Population statistics computed for the current iteration.
/// - `e0`: Energy of the first basis determinant.
/// - `es`: Non-overlap transformed shift energy.
/// # Returns
/// - `()`: Writes the current iteration line to stdout on rank zero.
pub(in crate::stochastic) fn print_row(irank: usize, iter: usize, state: &PropagationState, stats: &PopulationStats, e0: f64, es: f64) {
    let es_corr = if state.reached_c {es - e0} else {0.0};
    let es_s_corr = if state.reached_sc {state.es_s - e0} else {0.0};

    if irank == 0 {
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
                 iter, state.eprojcur, state.eprojcur - e0, es_corr, es_s_corr, stats.nwc as f64, stats.nrefc as f64, 
                 stats.nwsc, stats.nrefsc, stats.noccdets);
    }
}

/// Check for a `STOP` file, write a restart if required, and return early from the
/// stochastic propagation.
/// # Arguments:
/// - `it`: Current report number.
/// - `state`: Propagation state containing QMC bookkeeping data.
/// - `shifts`: Current non-overlap and overlap-transformed shifts.
/// - `run`: Rank-local run metadata.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `timings`: Accumulated stochastic propagation timings.
/// # Returns
/// - `Option<(f64, Option<ExcitationHist>, QMCTimings)>`: Final return values if a
///   stop was requested, otherwise `None`.
pub(in crate::stochastic) fn check_stop(report: usize, state: &mut PropagationState, shifts: Shifts, run: &QMCRunInfo, 
                                        world: &impl Communicator) -> Option<(f64, Option<ExcitationHist>)> {
    let mut stop = 0;
    if run.irank == 0 && Path::new("STOP").exists() {
        stop = 1;
    }
    world.process_at_rank(0).broadcast_into(&mut stop);

    if stop == 0 {
        return None;
    }

    let rs = RestartState {
        report,
        es: shifts.es,
        es_s: shifts.es_s,
        nwprevc: state.prev_pop.nwc,
        nrefprevc: state.prev_pop.nrefc,
        nwprevsc: state.prev_pop.nwsc,
        nrefprevsc: state.prev_pop.nrefsc,
        walkers: std::mem::replace(&mut state.mc.walkers, Walkers::new(run.ndets)),
        noccdets: state.prev_pop.noccdets,
        pg: std::mem::take(&mut state.mc.pg),
        excitation_hist: state.mc.excitation_hist.take(),
        base_seed: Some(run.base_seed),
    };
    
    write_restart_hdf5("RESTART.H5", world, &rs, run.ndets).unwrap();

    if run.irank == 0 {
        let _ = std::fs::remove_file("STOP");
        println!("STOP detected, Wrote RESTART.H5 and exiting");
    }

    Some((state.eprojcur, rs.excitation_hist))
}

