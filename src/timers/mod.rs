// timers/mod.rs

pub mod general;
pub mod deterministic;
pub mod stochastic;
pub mod snoci;
pub mod noci;
pub mod nonorthogonalwicks;

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
    /// - `ns`: Elapsed time in nanoseconds for the timed region.
    /// # Returns:
    /// - `()`: Updates the accumulated nanoseconds and increments the call count.
    #[inline(always)]
    pub fn add_ns(&mut self, ns: u64) {
        self.ns += ns;
        self.calls += 1;
    }

    /// Convert the accumulated nanoseconds in this counter into a `Duration`.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Duration`: Total elapsed time represented by this counter.
    #[inline(always)]
    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.ns)
    }

    /// Add the contents of another timing counter into this one.
    /// # Arguments:
    /// - `other`: Counter whose accumulated nanoseconds and call count are to be added.
    /// # Returns:
    /// - `()`: Updates this counter in place by summing nanoseconds and call counts.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Counter) {
        self.ns += other.ns;
        self.calls += other.calls;
    }
}

/// Top level collection of timing totals grouped by subsystem.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Timing counters for general high level workflow stages.
    pub general: general::Totals,
    /// Timing counters for deterministic NOCI-QMC stages.
    pub deterministic: deterministic::Totals,
    /// Timing counters for stochastic NOCI-QMC stages.
    pub stochastic: stochastic::Totals,
    /// Timing counters for SNOCI stages.
    pub snoci: snoci::Totals,
    /// Timing counters for routines in the `noci` module.
    pub noci: noci::Totals,
    /// Timing counters for routines in the `nonorthogonalwicks` module.
    pub nonorthogonalwicks: nonorthogonalwicks::Totals,
}

impl Totals {
    /// Add the contents of another top level timing collection into this one.
    /// # Arguments:
    /// - `other`: Top level timing totals whose subsystem counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this top level timing collection in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Totals) {
        self.general.merge_from(&other.general);
        self.deterministic.merge_from(&other.deterministic);
        self.stochastic.merge_from(&other.stochastic);
        self.snoci.merge_from(&other.snoci);
        self.noci.merge_from(&other.noci);
        self.nonorthogonalwicks.merge_from(&other.nonorthogonalwicks);
    }
}

thread_local! {
    /// Thread local store for timing counters. Each thread accumulates into its own
    /// `Totals` instance to avoid contention during timed regions.
    static TOTALS: RefCell<Totals> = RefCell::new(Totals::default());
}

/// Borrow the current thread local timing totals mutably and apply a closure to them.
/// # Arguments:
/// - `f`: Closure receiving a mutable reference to the current thread's `Totals`.
/// # Returns:
/// - `R`: Return value of the closure `f`.
#[inline(always)]
pub fn with_totals<R>(f: impl FnOnce(&mut Totals) -> R) -> R {
    TOTALS.with(|cell| {
        let mut totals = cell.borrow_mut();
        f(&mut totals)
    })
}

/// Take a copy of the current thread local timing totals.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Totals`: Copy of the current thread's accumulated timing totals.
pub fn snapshot() -> Totals {
    TOTALS.with(|cell| *cell.borrow())
}

/// Reset the current thread local timing totals back to their default zero state.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Clears all timing counters for the current thread.
pub fn reset() {
    TOTALS.with(|cell| *cell.borrow_mut() = Totals::default());
}

/// Take a copy of the timing totals accumulated on the current thread together with all Rayon
/// worker threads and return their sum.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Totals`: Sum of timing counters across the current thread and all Rayon worker threads.
pub fn snapshot_all() -> Totals {
    let mut total = snapshot();
    for t in rayon::broadcast(|_| snapshot()) {
        total.merge_from(&t);
    }
    total
}

/// Reset the timing totals on the current thread and all Rayon worker threads back to zero.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Clears all timing counters on the current thread and all Rayon worker threads.
pub fn reset_all() {
    reset();
    rayon::broadcast(|_| reset());
}

/// Time a block of code and record the elapsed nanoseconds using the supplied callback.
/// This macro is only active when the `timings` feature is enabled.
/// # Arguments:
/// - `$path`: Function taking a `u64` nanosecond count and updating the chosen counter.
/// - `$body`: Block of code to execute and time.
/// # Returns:
/// - Value returned by `$body`.
#[cfg(feature = "timings")]
#[macro_export]
macro_rules! time_call {
    ($path:path, $body:block) => {{
        let __t0 = ::std::time::Instant::now();

        #[allow(clippy::redundant_closure_call)]
        let __out = (|| $body)();

        $path(__t0.elapsed().as_nanos() as u64);
        __out
    }};
}

/// Execute a block of code without timing when the `timings` feature is disabled.
/// # Arguments:
/// - `$path`: Ignored timing callback path kept for interface compatibility.
/// - `$body`: Block of code to execute.
/// # Returns:
/// - Value returned by `$body`.
#[cfg(not(feature = "timings"))]
#[macro_export]
macro_rules! time_call {
    ($path:path, $body:block) => {{$body}};
}
