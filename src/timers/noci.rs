// timers/noci.rs

use super::{with_totals, Counter};

/// Timing counters for routines in the `noci` module.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Total time spent in `calculate_hs_pair`.
    pub calculate_hs_pair: Counter,
    /// Total time spent in `calculate_hs_pair_wicks`.
    pub calculate_hs_pair_wicks: Counter,
    /// Total time spent in `calculate_hs_pair_naive`.
    pub calculate_hs_pair_naive: Counter,
    /// Total time spent in `calculate_hs_pair_orthogonal`.
    pub calculate_hs_pair_orthogonal: Counter,
}

impl Totals {
    /// Add the contents of another set of NOCI matrix-element timings into this one.
    /// # Arguments:
    /// - `other`: NOCI timing totals whose counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this set of NOCI timing totals in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Totals) {
        self.calculate_hs_pair.merge_from(&other.calculate_hs_pair);
        self.calculate_hs_pair_wicks.merge_from(&other.calculate_hs_pair_wicks);
        self.calculate_hs_pair_naive.merge_from(&other.calculate_hs_pair_naive);
        self.calculate_hs_pair_orthogonal.merge_from(&other.calculate_hs_pair_orthogonal);
    }
}

/// Add one timed call to the `calculate_hs_pair` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_hs_pair`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_hs_pair` counter.
#[inline(always)]
pub fn add_calculate_hs_pair(ns: u64) {
    with_totals(|t| t.noci.calculate_hs_pair.add_ns(ns));
}

/// Add one timed call to the `calculate_hs_pair_wicks` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_hs_pair_wicks`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_hs_pair_wicks` counter.
#[inline(always)]
pub fn add_calculate_hs_pair_wicks(ns: u64) {
    with_totals(|t| t.noci.calculate_hs_pair_wicks.add_ns(ns));
}

/// Add one timed call to the `calculate_hs_pair_naive` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_hs_pair_naive`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_hs_pair_naive` counter.
#[inline(always)]
pub fn add_calculate_hs_pair_naive(ns: u64) {
    with_totals(|t| t.noci.calculate_hs_pair_naive.add_ns(ns));
}

/// Add one timed call to the `calculate_hs_pair_orthogonal` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_hs_pair_orthogonal`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_hs_pair_orthogonal` counter.
#[inline(always)]
pub fn add_calculate_hs_pair_orthogonal(ns: u64) {
    with_totals(|t| t.noci.calculate_hs_pair_orthogonal.add_ns(ns));
}
