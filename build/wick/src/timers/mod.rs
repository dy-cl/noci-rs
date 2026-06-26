// timers/mod.rs

pub mod canonical;
mod print;
pub mod wick;

use std::cell::RefCell;
use std::time::Duration;

/// Container for one timing counter storing total elapsed time and number of calls.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counter {
    /// Total accumulated time in nanoseconds.
    pub ns: u64,
    /// Number of timed calls contributing to `ns`.
    pub calls: u64,
}

impl Counter {
    /// Add one timed call of duration `ns` nanoseconds to this counter.
    /// # Arguments:
    /// - `ns`: Elapsed time in nanoseconds.
    /// # Returns:
    /// - `()`: Updates the accumulated time and call count.
    #[inline(always)]
    pub fn add_ns(
        &mut self,
        ns: u64,
    ) {
        self.ns += ns;
        self.calls += 1;
    }

    /// Convert the accumulated nanoseconds into a duration.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Duration`: Accumulated elapsed time.
    #[inline(always)]
    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.ns)
    }

    /// Add another timing counter into this one.
    /// # Arguments:
    /// - `other`: Counter to accumulate.
    /// # Returns:
    /// - `()`: Updates this counter in place.
    #[inline(always)]
    pub fn merge_from(
        &mut self,
        other: &Counter,
    ) {
        self.ns += other.ns;
        self.calls += other.calls;
    }
}

/// Top-level timing totals grouped by Wick-tool subsystem.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Canonicalisation timings.
    pub canonical: canonical::Totals,
    /// Wick contraction timings.
    pub wick: wick::Totals,
}

impl Totals {
    /// Add another set of Wick-tool timings into this one.
    /// # Arguments:
    /// - `other`: Timing totals to accumulate.
    /// # Returns:
    /// - `()`: Updates this timing collection in place.
    #[inline(always)]
    pub fn merge_from(
        &mut self,
        other: &Totals,
    ) {
        self.canonical.merge_from(&other.canonical);
        self.wick.merge_from(&other.wick);
    }
}

thread_local! {
    /// Thread-local timing storage used to avoid contention inside Rayon workers.
    static TOTALS: RefCell<Totals> = RefCell::new(Totals::default());
}

/// Borrow the current thread-local timing totals mutably and apply a closure.
/// # Arguments:
/// - `f`: Closure receiving the current thread's timing totals.
/// # Returns:
/// - `R`: Return value of `f`.
#[inline(always)]
pub fn with_totals<R>(f: impl FnOnce(&mut Totals) -> R) -> R {
    TOTALS.with(|cell| {
        let mut totals = cell.borrow_mut();
        f(&mut totals)
    })
}

/// Take a copy of the current thread-local timing totals.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Totals`: Current thread-local timing totals.
pub fn snapshot() -> Totals {
    TOTALS.with(|cell| *cell.borrow())
}

/// Reset the current thread-local timing totals.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Clears the current thread's timing totals.
pub fn reset() {
    TOTALS.with(|cell| *cell.borrow_mut() = Totals::default());
}

/// Sum timing totals across the current thread and all Rayon worker threads.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Totals`: Sum of all timing counters.
pub fn snapshot_all() -> Totals {
    let mut total = snapshot();

    for timings in rayon::broadcast(|_| snapshot()) {
        total.merge_from(&timings);
    }

    total
}

/// Reset timing totals on the current thread and all Rayon worker threads.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Clears all timing counters.
pub fn reset_all() {
    reset();
    rayon::broadcast(|_| reset());
}

/// Print timing totals across all threads.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Prints the timing report.
pub fn print_all() {
    print::print(snapshot_all());
}

/// Time a block and record its elapsed nanoseconds using the supplied callback.
/// This macro is only active when the `timings` feature is enabled.
/// # Arguments:
/// - `$path`: Function taking an elapsed nanosecond count.
/// - `$body`: Block of code to execute and time.
/// # Returns:
/// - Value returned by `$body`.
#[macro_export]
macro_rules! time_call {
    ($path:path, $body:block) => {{
        #[cfg(feature = "timings")]
        {
            struct __TimeCallGuard {
                t0: std::time::Instant,
                add: fn(u64),
            }

            impl Drop for __TimeCallGuard {
                fn drop(&mut self) {
                    (self.add)(self.t0.elapsed().as_nanos() as u64);
                }
            }

            let __timings_guard = __TimeCallGuard {
                t0: std::time::Instant::now(),
                add: $path,
            };

            let __timings_ret = { $body };
            std::mem::drop(__timings_guard);
            __timings_ret
        }

        #[cfg(not(feature = "timings"))]
        {
            $body
        }
    }};
}
