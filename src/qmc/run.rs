// qmc/run.rs
//! The report loop shared by every stochastic propagator. Each report samples `n_\text{cyc}` cycles
//! from the frozen populations and accumulates their spawns, then takes one step of
//! `n_\text{cyc}\Delta\tau`; a propagator supplies its cycle, its step, its restart and its report
//! row, and this loop owns their order, the stop request and the restart interval. Cycles share
//! the per-thread worker scheduler below: one persistent worker and random stream per Rayon thread,
//! with matrix-element queues resolved by their owning workers.

// Standard library imports.
use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// External crate imports.
use mpi::topology::Communicator;

// Parent/sibling imports.
use super::report::stop_requested;
use super::rng::{QmcRng, stream};

/// A stochastic propagator driven by the shared report loop.
pub(crate) trait Propagator {
    /// Spawn one cycle from the frozen populations and accumulate its change.
    /// # Arguments:
    /// - `self`: Propagator.
    /// - `iteration`: Global iteration of the cycle.
    /// # Returns
    /// - `()`: Accumulates the cycle's spawns.
    fn cycle(
        &mut self,
        iteration: usize,
    );

    /// Take the report's step: exchange, compress and apply the accumulated change, then update the
    /// observables and the shift.
    /// # Arguments:
    /// - `self`: Propagator.
    /// - `report`: Report ending with this step.
    /// # Returns
    /// - `()`: Advances the populations by one step.
    fn step(
        &mut self,
        report: usize,
    );

    /// Write a restart of the state after a report.
    /// # Arguments:
    /// - `self`: Propagator.
    /// - `report`: Last completed report.
    /// # Returns
    /// - `()`: Writes the restart file.
    fn write_restart(
        &mut self,
        report: usize,
    );

    /// Print the report row after a report.
    /// # Arguments:
    /// - `self`: Propagator.
    /// - `end`: Iteration count at the end of the report.
    /// # Returns
    /// - `()`: Prints one row on the root rank.
    fn print_row(
        &self,
        end: usize,
    );

    /// Finish the run after its last report or a stop request, by default with nothing to do.
    /// # Arguments:
    /// - `self`: Propagator.
    /// # Returns
    /// - `()`: Prints any final diagnostics.
    fn finish(&mut self) {}
}

/// A persistent per-thread worker of a stochastic cycle.
pub(crate) trait Worker {
    /// Clear the worker for a new cycle and give it the cycle's random stream.
    /// # Arguments:
    /// - `self`: Persistent worker.
    /// - `rng`: Random stream of this worker's thread and the cycle.
    /// # Returns:
    /// - `()`: Clears the worker's per-cycle buffers.
    fn reset(
        &mut self,
        rng: QmcRng,
    );
}

/// Run every report of a stochastic propagation. After each step, a `STOP` file writes a restart
/// and ends the run before the report's row, a report ending on the restart interval writes a
/// restart, and every other report prints its row.
/// # Arguments:
/// - `propagator`: Propagator to drive.
/// - `schedule`: Reports to propagate, from the first after any restart, and cycles per report.
/// - `restart`: Restart interval in iterations, if any, and the restart file path.
/// - `world`: MPI communicator.
/// # Returns
/// - `()`: Propagates until the last report or a stop request.
pub(crate) fn run(
    propagator: &mut impl Propagator,
    schedule: (Range<usize>, usize),
    restart: (Option<usize>, &str),
    world: &impl Communicator,
) {
    let (reports, ncycles) = schedule;
    let (interval, path) = restart;
    let (irank, nranks) = (world.rank() as usize, world.size() as usize);

    for report in reports {
        for cycle in 0..ncycles {
            propagator.cycle(report * ncycles + cycle);
        }

        // The populations remain frozen throughout sampling; apply their accumulated change once.
        propagator.step(report);

        // Every rank receives one stop decision, so all ranks enter the restart write together.
        if stop_requested(irank, nranks, world) {
            propagator.write_restart(report);
            if irank == 0 {
                let _ = std::fs::remove_file("STOP");
                println!("STOP detected, wrote {path} and exiting");
            }
            break;
        }

        let end = (report + 1) * ncycles;
        if interval.is_some_and(|interval| end.is_multiple_of(interval)) {
            propagator.write_restart(report);
        }

        propagator.print_row(end);
    }

    propagator.finish();
}

/// Run one cycle's work items on the persistent per-thread workers. Each worker is reset with the
/// stream of its thread and the iteration, takes item indices in chunks as it becomes free, and
/// finally resolves its own spawns. Items are passed by index, so no list of them is built.
/// # Arguments:
/// - `nitems`: Number of work items of the cycle.
/// - `seed`: Rank seed and iteration of the cycle's streams.
/// - `workers`: Persistent per-thread workers, one per Rayon thread.
/// - `steps`: Work on the item of one index, and the resolution of a worker's spawns.
/// # Returns
/// - `()`: Leaves every worker's spawns in the worker.
pub(crate) fn spawn_on_workers<W: Worker + Send>(
    nitems: usize,
    seed: (u64, usize),
    workers: &mut [Mutex<W>],
    steps: (impl Fn(&mut W, usize) + Sync, impl Fn(&mut W) + Sync),
) {
    // Eight items amortise the atomic claim while leaving short batches available for balancing.
    // Keep the scheduling grain stable: with several threads it affects which stream draws an item.
    const CHUNK: usize = 8;

    let (rank_seed, iteration) = seed;
    let (item_step, resolve) = steps;

    if nitems == 0 {
        return;
    }

    let next = AtomicUsize::new(0);
    let workers: &[Mutex<W>] = workers;
    rayon::broadcast(|context| {
        let tid = context.index();
        let mut worker = workers[tid].lock().unwrap();
        worker.reset(stream(rank_seed, tid as u64, iteration as u64));

        // Take chunks of items until every item has been assigned.
        loop {
            let start = next.fetch_add(CHUNK, Ordering::Relaxed);
            if start >= nitems {
                break;
            }
            for item in start..(start + CHUNK).min(nitems) {
                item_step(&mut worker, item);
            }
        }

        // Flush each worker's final partial matrix-element batch on its own thread.
        resolve(&mut worker);
    });
}
