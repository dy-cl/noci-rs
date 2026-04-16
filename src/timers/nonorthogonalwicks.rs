// timers/nonorthogonalwicks.rs

use super::{with_totals, Counter};

/// Timing counters for routines in the `nonorthogonalwicks` module.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Total time spent in `prepare_same`.
    pub prepare_same: Counter,
    /// Total time spent in `get_det_adjt_same`.
    pub get_det_adjt_same: Counter,
    /// Total time spent in `get_det_adjt_diff`.
    pub get_det_adjt_diff: Counter,
    /// Total time spent in `construct_determinant_indices`
    pub construct_determinant_indices: Counter,
    /// Total time spent in `lg_overlap`.
    pub lg_overlap: Counter,
    /// Total time spent in `lg_h1`.
    pub lg_h1: Counter,
    /// Total time spent in `lg_h2_same`.
    pub lg_h2_same: Counter,
    /// Total time spent in `lg_h2_diff`.
    pub lg_h2_diff: Counter,
}

impl Totals {
    /// Add the contents of another set of nonorthogonal Wick timing counters into this one.
    /// # Arguments:
    /// - `other`: Nonorthogonal Wick timing totals whose counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this set of nonorthogonal Wick timing totals in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Totals) {
        self.prepare_same.merge_from(&other.prepare_same);
        self.get_det_adjt_same.merge_from(&other.get_det_adjt_same);
        self.get_det_adjt_diff.merge_from(&other.get_det_adjt_diff);
        self.lg_overlap.merge_from(&other.lg_overlap);
        self.lg_h1.merge_from(&other.lg_h1);
        self.lg_h2_same.merge_from(&other.lg_h2_same);
        self.lg_h2_diff.merge_from(&other.lg_h2_diff);
        self.construct_determinant_indices.merge_from(&other.construct_determinant_indices);
    }
}

/// Add one timed call to the `prepare_same` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `prepare_same`.
/// # Returns:
/// - `()`: Updates the current thread local `prepare_same` counter.
#[inline(always)]
pub fn add_prepare_same(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.prepare_same.add_ns(ns));
}

/// Add one timed call to the `get_det_adjt_same` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `get_det_adjt_same`.
/// # Returns:
/// - `()`: Updates the current thread local `get_det_adjt_same` counter.
#[inline(always)]
pub fn add_get_det_adjt_same(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.get_det_adjt_same.add_ns(ns));
}

/// Add one timed call to the `get_det_adjt_diff` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `get_det_adjt_diff`.
/// # Returns:
/// - `()`: Updates the current thread local `get_det_adjt_diff` counter.
#[inline(always)]
pub fn add_get_det_adjt_diff(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.get_det_adjt_diff.add_ns(ns));
}

/// Add one timed call to the `construct_determinant_indices` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `construct_determinant_indices`.
/// # Returns:
/// - `()`: Updates the current thread local `construct_determinant_indices` counter.
#[inline(always)]
pub fn add_construct_determinant_indices(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.construct_determinant_indices.add_ns(ns));
}

/// Add one timed call to the `lg_overlap` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `lg_overlap`.
/// # Returns:
/// - `()`: Updates the current thread local `lg_overlap` counter.
#[inline(always)]
pub fn add_lg_overlap(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.lg_overlap.add_ns(ns));
}

/// Add one timed call to the `lg_h1` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `lg_h1`.
/// # Returns:
/// - `()`: Updates the current thread local `lg_h1` counter.
#[inline(always)]
pub fn add_lg_h1(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.lg_h1.add_ns(ns));
}

/// Add one timed call to the `lg_h2_same` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `lg_h2_same`.
/// # Returns:
/// - `()`: Updates the current thread local `lg_h2_same` counter.
#[inline(always)]
pub fn add_lg_h2_same(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.lg_h2_same.add_ns(ns));
}

/// Add one timed call to the `lg_h2_diff` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `lg_h2_diff`.
/// # Returns:
/// - `()`: Updates the current thread local `lg_h2_diff` counter.
#[inline(always)]
pub fn add_lg_h2_diff(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.lg_h2_diff.add_ns(ns));
}
