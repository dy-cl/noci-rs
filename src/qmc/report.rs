// qmc/report.rs

// Standard library imports.
use std::path::Path;

// External crate imports.
use mpi::topology::Communicator;
use mpi::traits::*;

/// Decide on every rank whether a `STOP` file requests the run to end.
/// Rank zero observes the file and broadcasts one decision so every rank leaves the propagation
/// loop, and enters any collective restart write, together.
/// # Arguments:
/// - `irank`: MPI rank.
/// - `nranks`: Number of MPI ranks.
/// - `world`: MPI communicator.
/// # Returns:
/// - `bool`: Whether the run should stop.
pub(crate) fn stop_requested(
    irank: usize,
    nranks: usize,
    world: &impl Communicator,
) -> bool {
    let mut stop = 0;

    if irank == 0 && Path::new("STOP").exists() {
        stop = 1;
    }

    if nranks > 1 {
        world.process_at_rank(0).broadcast_into(&mut stop);
    }

    stop != 0
}
