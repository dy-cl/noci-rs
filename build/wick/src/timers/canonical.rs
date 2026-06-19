// timers/canonical.rs

use super::{Counter, with_totals};

/// Canonicalisation timing totals.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Totals`: Canonicalisation timings.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Time spent adding terms to canonical accumulators.
    pub accumulate: Counter,
    /// Time spent merging canonical accumulators.
    pub merge: Counter,
    /// Time spent finishing canonical accumulators.
    pub finish: Counter,
    /// Time spent converting accumulators into expressions.
    pub intoexpr: Counter,
    /// Time spent sparsifying high-rank cumulant orbits.
    pub spar: Counter,
    /// Time spent in the final canonical sum.
    pub final_sum: Counter,
}

impl Totals {
    /// Add another set of canonicalisation timings into this one.
    /// # Arguments:
    /// - `other`: Canonicalisation timings to accumulate.
    /// # Returns:
    /// - `()`: Updates this timing collection in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Totals) {
        self.accumulate.merge_from(&other.accumulate);
        self.merge.merge_from(&other.merge);
        self.finish.merge_from(&other.finish);
        self.intoexpr.merge_from(&other.intoexpr);
        self.spar.merge_from(&other.spar);
        self.final_sum.merge_from(&other.final_sum);
    }
}

/// Add one canonical accumulation timing.
/// # Arguments:
/// - `ns`: Elapsed nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local counter.
#[inline(always)]
pub fn add_accumulate(ns: u64) {
    with_totals(|totals| totals.canonical.accumulate.add_ns(ns));
}

/// Add one canonical accumulator merge timing.
/// # Arguments:
/// - `ns`: Elapsed nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local counter.
#[inline(always)]
pub fn add_merge(ns: u64) {
    with_totals(|totals| totals.canonical.merge.add_ns(ns));
}

/// Add one canonical finish timing.
/// # Arguments:
/// - `ns`: Elapsed nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local counter.
#[inline(always)]
pub fn add_finish(ns: u64) {
    with_totals(|totals| totals.canonical.finish.add_ns(ns));
}

/// Add one expression-reconstruction timing.
/// # Arguments:
/// - `ns`: Elapsed nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local counter.
#[inline(always)]
pub fn add_intoexpr(ns: u64) {
    with_totals(|totals| totals.canonical.intoexpr.add_ns(ns));
}

/// Add one cumulant-sparsification timing.
/// # Arguments:
/// - `ns`: Elapsed nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local counter.
#[inline(always)]
pub fn add_spar(ns: u64) {
    with_totals(|totals| totals.canonical.spar.add_ns(ns));
}

/// Add one final canonical-sum timing.
/// # Arguments:
/// - `ns`: Elapsed nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local counter.
#[inline(always)]
pub fn add_final_sum(ns: u64) {
    with_totals(|totals| totals.canonical.final_sum.add_ns(ns));
}
