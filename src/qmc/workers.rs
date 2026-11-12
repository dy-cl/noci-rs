// qmc/workers.rs
//! Persistent per-thread workers of a stochastic cycle. Every propagator spawns from its work
//! items, sampled sources or cluster attempts, on one worker per Rayon thread. Each worker draws
//! from the stream of its thread and the iteration, takes items in small chunks as it becomes free,
//! and finally resolves what it has queued, so matrix elements are evaluated in batches rather than
//! one at a time. A worker may also resolve during the cycle to bound the size of its batches.

// Standard library imports.
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// Parent/sibling imports.
use super::rng::{QmcRng, stream};

/// Number of work items a worker takes at a time.
const CHUNK: usize = 8;

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
        resolve(&mut worker);
    });
}
