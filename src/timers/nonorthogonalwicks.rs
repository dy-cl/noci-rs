// timers/nonorthogonalwicks.rs
use super::{Counter, with_totals};

/// Timing counters for routines in the `nonorthogonalwicks` module.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Total time spent in `prepare_same`.
    pub prepare_same: Counter,
    /// Total time spent in `prepare_same_gen`.
    pub prepare_same_gen: Counter,
    /// Total time spent in `prepare_same_m0`.
    pub prepare_same_m0: Counter,
    /// Total time spent in `get_det_adjt_same`.
    pub get_det_adjt_same: Counter,
    /// Total time spent in `get_det_adjt_diff`.
    pub get_det_adjt_diff: Counter,
    /// Total time spent in `construct_determinant_indices`.
    pub construct_determinant_indices: Counter,
    /// Total time spent in `construct_determinant_indices_l1`.
    pub construct_determinant_indices_l1: Counter,
    /// Total time spent in `construct_determinant_indices_l2`.
    pub construct_determinant_indices_l2: Counter,
    /// Total time spent in `construct_determinant_indices_l3`.
    pub construct_determinant_indices_l3: Counter,
    /// Total time spent in `construct_determinant_indices_l4`.
    pub construct_determinant_indices_l4: Counter,
    /// Total time spent in `construct_determinant_indices_gen`.
    pub construct_determinant_indices_gen: Counter,
    /// Total time spent in `xw_overlap`.
    pub xw_overlap: Counter,
    /// Total time spent in `xw_overlap_m0`.
    pub xw_overlap_m0: Counter,
    /// Total time spent in `xw_overlap_m0_l1`.
    pub xw_overlap_m0_l1: Counter,
    /// Total time spent in `xw_overlap_m0_l2`.
    pub xw_overlap_m0_l2: Counter,
    /// Total time spent in `xw_overlap_m0_l3`.
    pub xw_overlap_m0_l3: Counter,
    /// Total time spent in `xw_overlap_m0_l4`.
    pub xw_overlap_m0_l4: Counter,
    /// Total time spent in `xw_overlap_ml`.
    pub xw_overlap_ml: Counter,
    /// Total time spent in `xw_overlap_ml_l1`.
    pub xw_overlap_ml_l1: Counter,
    /// Total time spent in `xw_overlap_ml_l2`.
    pub xw_overlap_ml_l2: Counter,
    /// Total time spent in `xw_overlap_ml_l3`.
    pub xw_overlap_ml_l3: Counter,
    /// Total time spent in `xw_overlap_gen`.
    pub xw_overlap_gen: Counter,
    /// Total time spent in `xw_h1`.
    pub xw_h1: Counter,
    /// Total time spent in `xw_f`.
    pub xw_f: Counter,
    /// Total time spent in `xw_rdm1`.
    pub xw_rdm1: Counter,
    /// Total time spent in `xw_rdm1_m0`.
    pub xw_rdm1_m0: Counter,
    /// Total time spent in `xw_rdm1_gen`.
    pub xw_rdm1_gen: Counter,
    /// Total time spent in `xw_one_body_gen`.
    pub xw_one_body_gen: Counter,
    /// Total time spent in `xw_one_body_m0`.
    pub xw_one_body_m0: Counter,
    /// Total time spent in `xw_h2_same`.
    pub xw_h2_same: Counter,
    /// Total time spent in `xw_h2_same_gen`.
    pub xw_h2_same_gen: Counter,
    /// Total time spent in `xw_h2_same_m0`.
    pub xw_h2_same_m0: Counter,
    /// Total time spent in `xw_h2_diff`.
    pub xw_h2_diff: Counter,
    /// Total time spent in `xw_h2_diff_gen`.
    pub xw_h2_diff_gen: Counter,
    /// Total time spent in `xw_h2_diff_m0`.
    pub xw_h2_diff_m0: Counter,
    /// Total time spent in `xw_rdm2_same`.
    pub xw_rdm2_same: Counter,
    /// Total time spent in `xw_rdm2_same_gen`.
    pub xw_rdm2_same_gen: Counter,
    /// Total time spent in `xw_rdm2_same_m0`.
    pub xw_rdm2_same_m0: Counter,
    /// Total time spent in `xw_rdm2_diff`.
    pub xw_rdm2_diff: Counter,
    /// Total time spent in `prepare_same_m0_l1`.
    pub prepare_same_m0_l1: Counter,
    /// Total time spent in `prepare_same_m0_l2`.
    pub prepare_same_m0_l2: Counter,
    /// Total time spent in `prepare_same_m0_l3`.
    pub prepare_same_m0_l3: Counter,
    /// Total time spent in `prepare_same_m0_l4`.
    pub prepare_same_m0_l4: Counter,
    /// Total time spent in `xw_one_body_m0_gen`.
    pub xw_one_body_m0_gen: Counter,
    /// Total time spent in `xw_one_body_m0_l1`.
    pub xw_one_body_m0_l1: Counter,
    /// Total time spent in `xw_one_body_m0_l2`.
    pub xw_one_body_m0_l2: Counter,
    /// Total time spent in `xw_one_body_m0_l3`.
    pub xw_one_body_m0_l3: Counter,
    /// Total time spent in `xw_h2_same_m0_gen`.
    pub xw_h2_same_m0_gen: Counter,
    /// Total time spent in `xw_h2_same_m0_l1`.
    pub xw_h2_same_m0_l1: Counter,
    /// Total time spent in `xw_h2_same_m0_l2`.
    pub xw_h2_same_m0_l2: Counter,
    /// Total time spent in `xw_h2_diff_m0_gen`.
    pub xw_h2_diff_m0_gen: Counter,
    /// Total time spent in `xw_h2_diff_m0_11`.
    pub xw_h2_diff_m0_11: Counter,
    /// Total time spent in `xw_h2_diff_m0_12`.
    pub xw_h2_diff_m0_12: Counter,
    /// Total time spent in `xw_h2_diff_m0_21`.
    pub xw_h2_diff_m0_21: Counter,
    /// Total time spent in `xw_h2_diff_m0_22`.
    pub xw_h2_diff_m0_22: Counter,
    /// Total time spent in `xw_h2_same_m0_l3`.
    pub xw_h2_same_m0_l3: Counter,
    /// Total time spent in `xw_h2_same_m0_l4`.
    pub xw_h2_same_m0_l4: Counter,
    /// Total time spent in `xw_h2_diff_m0_13`.
    pub xw_h2_diff_m0_13: Counter,
    /// Total time spent in `xw_h2_diff_m0_31`.
    pub xw_h2_diff_m0_31: Counter,
}

impl Totals {
    /// Add the contents of another set of nonorthogonal Wick timing counters into this one.
    /// # Arguments:
    /// - `other`: Nonorthogonal Wick timing totals whose counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this set of nonorthogonal Wick timing totals in place.
    #[inline(always)]
    pub fn merge_from(
        &mut self,
        other: &Totals,
    ) {
        self.prepare_same.merge_from(&other.prepare_same);
        self.prepare_same_gen.merge_from(&other.prepare_same_gen);
        self.prepare_same_m0.merge_from(&other.prepare_same_m0);
        self.get_det_adjt_same.merge_from(&other.get_det_adjt_same);
        self.get_det_adjt_diff.merge_from(&other.get_det_adjt_diff);
        self.construct_determinant_indices
            .merge_from(&other.construct_determinant_indices);
        self.construct_determinant_indices_l1
            .merge_from(&other.construct_determinant_indices_l1);
        self.construct_determinant_indices_l2
            .merge_from(&other.construct_determinant_indices_l2);
        self.construct_determinant_indices_l3
            .merge_from(&other.construct_determinant_indices_l3);
        self.construct_determinant_indices_gen
            .merge_from(&other.construct_determinant_indices_gen);
        self.xw_overlap.merge_from(&other.xw_overlap);
        self.xw_overlap_m0.merge_from(&other.xw_overlap_m0);
        self.xw_overlap_m0_l1.merge_from(&other.xw_overlap_m0_l1);
        self.xw_overlap_m0_l2.merge_from(&other.xw_overlap_m0_l2);
        self.xw_overlap_m0_l3.merge_from(&other.xw_overlap_m0_l3);
        self.xw_overlap_m0_l4.merge_from(&other.xw_overlap_m0_l4);
        self.xw_overlap_ml.merge_from(&other.xw_overlap_ml);
        self.xw_overlap_ml_l1.merge_from(&other.xw_overlap_ml_l1);
        self.xw_overlap_ml_l2.merge_from(&other.xw_overlap_ml_l2);
        self.xw_overlap_ml_l3.merge_from(&other.xw_overlap_ml_l3);
        self.xw_overlap_gen.merge_from(&other.xw_overlap_gen);
        self.xw_h1.merge_from(&other.xw_h1);
        self.xw_f.merge_from(&other.xw_f);
        self.xw_rdm1.merge_from(&other.xw_rdm1);
        self.xw_rdm1_m0.merge_from(&other.xw_rdm1_m0);
        self.xw_rdm1_gen.merge_from(&other.xw_rdm1_gen);
        self.xw_one_body_gen.merge_from(&other.xw_one_body_gen);
        self.xw_one_body_m0.merge_from(&other.xw_one_body_m0);
        self.xw_h2_same.merge_from(&other.xw_h2_same);
        self.xw_h2_same_gen.merge_from(&other.xw_h2_same_gen);
        self.xw_h2_same_m0.merge_from(&other.xw_h2_same_m0);
        self.xw_h2_diff.merge_from(&other.xw_h2_diff);
        self.xw_h2_diff_gen.merge_from(&other.xw_h2_diff_gen);
        self.xw_h2_diff_m0.merge_from(&other.xw_h2_diff_m0);
        self.xw_rdm2_same.merge_from(&other.xw_rdm2_same);
        self.xw_rdm2_same_gen.merge_from(&other.xw_rdm2_same_gen);
        self.xw_rdm2_same_m0.merge_from(&other.xw_rdm2_same_m0);
        self.xw_rdm2_diff.merge_from(&other.xw_rdm2_diff);
        self.prepare_same_m0_l1
            .merge_from(&other.prepare_same_m0_l1);
        self.prepare_same_m0_l2
            .merge_from(&other.prepare_same_m0_l2);
        self.prepare_same_m0_l3
            .merge_from(&other.prepare_same_m0_l3);
        self.xw_one_body_m0_gen
            .merge_from(&other.xw_one_body_m0_gen);
        self.xw_one_body_m0_l1.merge_from(&other.xw_one_body_m0_l1);
        self.xw_one_body_m0_l2.merge_from(&other.xw_one_body_m0_l2);
        self.xw_one_body_m0_l3.merge_from(&other.xw_one_body_m0_l3);
        self.xw_h2_same_m0_gen.merge_from(&other.xw_h2_same_m0_gen);
        self.xw_h2_same_m0_l1.merge_from(&other.xw_h2_same_m0_l1);
        self.xw_h2_same_m0_l2.merge_from(&other.xw_h2_same_m0_l2);
        self.xw_h2_diff_m0_gen.merge_from(&other.xw_h2_diff_m0_gen);
        self.xw_h2_diff_m0_11.merge_from(&other.xw_h2_diff_m0_11);
        self.xw_h2_diff_m0_12.merge_from(&other.xw_h2_diff_m0_12);
        self.xw_h2_diff_m0_21.merge_from(&other.xw_h2_diff_m0_21);
        self.xw_h2_diff_m0_22.merge_from(&other.xw_h2_diff_m0_22);
        self.xw_h2_same_m0_l3.merge_from(&other.xw_h2_same_m0_l3);
        self.xw_h2_same_m0_l4.merge_from(&other.xw_h2_same_m0_l4);
        self.xw_h2_diff_m0_13.merge_from(&other.xw_h2_diff_m0_13);
        self.xw_h2_diff_m0_31.merge_from(&other.xw_h2_diff_m0_31);
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

/// Add one timed call to the `prepare_same_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `prepare_same_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `prepare_same_gen` counter.
#[inline(always)]
pub fn add_prepare_same_gen(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.prepare_same_gen.add_ns(ns));
}

/// Add one timed call to the `prepare_same_m0` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `prepare_same_m0`.
/// # Returns:
/// - `()`: Updates the current thread local `prepare_same_m0` counter.
#[inline(always)]
pub fn add_prepare_same_m0(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.prepare_same_m0.add_ns(ns));
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
    with_totals(|t| {
        t.nonorthogonalwicks
            .construct_determinant_indices
            .add_ns(ns)
    });
}

/// Add one timed call to the `construct_determinant_indices_l1` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `construct_determinant_indices_l1`.
/// # Returns:
/// - `()`: Updates the current thread local `construct_determinant_indices_l1` counter.
#[inline(always)]
pub fn add_construct_determinant_indices_l1(ns: u64) {
    with_totals(|t| {
        t.nonorthogonalwicks
            .construct_determinant_indices_l1
            .add_ns(ns)
    });
}

/// Add one timed call to the `construct_determinant_indices_l2` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `construct_determinant_indices_l2`.
/// # Returns:
/// - `()`: Updates the current thread local `construct_determinant_indices_l2` counter.
#[inline(always)]
pub fn add_construct_determinant_indices_l2(ns: u64) {
    with_totals(|t| {
        t.nonorthogonalwicks
            .construct_determinant_indices_l2
            .add_ns(ns)
    });
}

/// Add one timed call to the `construct_determinant_indices_l3` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `construct_determinant_indices_l3`.
/// # Returns:
/// - `()`: Updates the current thread local `construct_determinant_indices_l3` counter.
#[inline(always)]
pub fn add_construct_determinant_indices_l3(ns: u64) {
    with_totals(|t| {
        t.nonorthogonalwicks
            .construct_determinant_indices_l3
            .add_ns(ns)
    });
}

/// Add one timed call to the `construct_determinant_indices_l4` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `construct_determinant_indices_l4`.
/// # Returns:
/// - `()`: Updates the current thread local `construct_determinant_indices_l4` counter.
#[inline(always)]
pub fn add_construct_determinant_indices_l4(ns: u64) {
    with_totals(|t| {
        t.nonorthogonalwicks
            .construct_determinant_indices_l4
            .add_ns(ns)
    });
}

/// Add one timed call to the `construct_determinant_indices_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `construct_determinant_indices_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `construct_determinant_indices_gen` counter.
#[inline(always)]
pub fn add_construct_determinant_indices_gen(ns: u64) {
    with_totals(|t| {
        t.nonorthogonalwicks
            .construct_determinant_indices_gen
            .add_ns(ns)
    });
}

/// Add one timed call to the `xw_overlap` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap` counter.
#[inline(always)]
pub fn add_xw_overlap(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap.add_ns(ns));
}

/// Add one timed call to the `xw_overlap_m0` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap_m0`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap_m0` counter.
#[inline(always)]
pub fn add_xw_overlap_m0(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap_m0.add_ns(ns));
}

/// Add one timed call to the `xw_overlap_m0_l1` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap_m0_l1`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap_m0_l1` counter.
#[inline(always)]
pub fn add_xw_overlap_m0_l1(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap_m0_l1.add_ns(ns));
}

/// Add one timed call to the `xw_overlap_m0_l2` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap_m0_l2`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap_m0_l2` counter.
#[inline(always)]
pub fn add_xw_overlap_m0_l2(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap_m0_l2.add_ns(ns));
}

/// Add one timed call to the `xw_overlap_m0_l3` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap_m0_l3`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap_m0_l3` counter.
#[inline(always)]
pub fn add_xw_overlap_m0_l3(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap_m0_l3.add_ns(ns));
}

/// Add one timed call to the `xw_overlap_m0_l4` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap_m0_l4`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap_m0_l4` counter.
#[inline(always)]
pub fn add_xw_overlap_m0_l4(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap_m0_l4.add_ns(ns));
}

/// Add one timed call to the `xw_overlap_ml` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap_ml`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap_ml` counter.
#[inline(always)]
pub fn add_xw_overlap_ml(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap_ml.add_ns(ns));
}

/// Add one timed call to the `xw_overlap_ml_l1` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap_ml_l1`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap_ml_l1` counter.
#[inline(always)]
pub fn add_xw_overlap_ml_l1(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap_ml_l1.add_ns(ns));
}

/// Add one timed call to the `xw_overlap_ml_l2` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap_ml_l2`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap_ml_l2` counter.
#[inline(always)]
pub fn add_xw_overlap_ml_l2(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap_ml_l2.add_ns(ns));
}

/// Add one timed call to the `xw_overlap_ml_l3` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap_ml_l3`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap_ml_l3` counter.
#[inline(always)]
pub fn add_xw_overlap_ml_l3(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap_ml_l3.add_ns(ns));
}

/// Add one timed call to the `xw_overlap_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_overlap_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_overlap_gen` counter.
#[inline(always)]
pub fn add_xw_overlap_gen(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_overlap_gen.add_ns(ns));
}

/// Add one timed call to the `xw_h1` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h1`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h1` counter.
#[inline(always)]
pub fn add_xw_h1(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h1.add_ns(ns));
}

/// Add one timed call to the `xw_f` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_f`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_f` counter.
#[inline(always)]
pub fn add_xw_f(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_f.add_ns(ns));
}

/// Add one timed call to the `xw_rdm1` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_rdm1`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_rdm1` counter.
#[inline(always)]
pub fn add_xw_rdm1(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_rdm1.add_ns(ns));
}

/// Add one timed call to the `xw_rdm1_m0` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_rdm1_m0`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_rdm1_m0` counter.
#[inline(always)]
pub fn add_xw_rdm1_m0(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_rdm1_m0.add_ns(ns));
}

/// Add one timed call to the `xw_rdm1_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_rdm1_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_rdm1_gen` counter.
#[inline(always)]
pub fn add_xw_rdm1_gen(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_rdm1_gen.add_ns(ns));
}

/// Add one timed call to the `xw_one_body_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_one_body_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_one_body_gen` counter.
#[inline(always)]
pub fn add_xw_one_body_gen(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_one_body_gen.add_ns(ns));
}

/// Add one timed call to the `xw_one_body_m0` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_one_body_m0`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_one_body_m0` counter.
#[inline(always)]
pub fn add_xw_one_body_m0(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_one_body_m0.add_ns(ns));
}

/// Add one timed call to the `xw_h2_same` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_same`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_same` counter.
#[inline(always)]
pub fn add_xw_h2_same(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_same.add_ns(ns));
}

/// Add one timed call to the `xw_h2_same_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_same_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_same_gen` counter.
#[inline(always)]
pub fn add_xw_h2_same_gen(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_same_gen.add_ns(ns));
}

/// Add one timed call to the `xw_h2_same_m0` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_same_m0`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_same_m0` counter.
#[inline(always)]
pub fn add_xw_h2_same_m0(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_same_m0.add_ns(ns));
}

/// Add one timed call to the `xw_h2_diff` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_diff`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_diff` counter.
#[inline(always)]
pub fn add_xw_h2_diff(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_diff.add_ns(ns));
}

/// Add one timed call to the `xw_h2_diff_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_diff_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_diff_gen` counter.
#[inline(always)]
pub fn add_xw_h2_diff_gen(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_diff_gen.add_ns(ns));
}

/// Add one timed call to the `xw_h2_diff_m0` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_diff_m0`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_diff_m0` counter.
#[inline(always)]
pub fn add_xw_h2_diff_m0(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_diff_m0.add_ns(ns));
}

/// Add one timed call to the `xw_rdm2_same` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_rdm2_same`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_rdm2_same` counter.
#[inline(always)]
pub fn add_xw_rdm2_same(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_rdm2_same.add_ns(ns));
}

/// Add one timed call to the `xw_rdm2_same_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_rdm2_same_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_rdm2_same_gen` counter.
#[inline(always)]
pub fn add_xw_rdm2_same_gen(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_rdm2_same_gen.add_ns(ns));
}

/// Add one timed call to the `xw_rdm2_same_m0` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_rdm2_same_m0`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_rdm2_same_m0` counter.
#[inline(always)]
pub fn add_xw_rdm2_same_m0(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_rdm2_same_m0.add_ns(ns));
}

/// Add one timed call to the `xw_rdm2_diff` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_rdm2_diff`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_rdm2_diff` counter.
#[inline(always)]
pub fn add_xw_rdm2_diff(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_rdm2_diff.add_ns(ns));
}

/// Add one timed call to the `prepare_same_m0_l1` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `prepare_same_m0_l1`.
/// # Returns:
/// - `()`: Updates the current thread local `prepare_same_m0_l1` counter.
#[inline(always)]
pub fn add_prepare_same_m0_l1(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.prepare_same_m0_l1.add_ns(ns));
}

/// Add one timed call to the `prepare_same_m0_l2` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `prepare_same_m0_l2`.
/// # Returns:
/// - `()`: Updates the current thread local `prepare_same_m0_l2` counter.
#[inline(always)]
pub fn add_prepare_same_m0_l2(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.prepare_same_m0_l2.add_ns(ns));
}

/// Add one timed call to the `prepare_same_m0_l3` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `prepare_same_m0_l3`.
/// # Returns:
/// - `()`: Updates the current thread local `prepare_same_m0_l3` counter.
#[inline(always)]
pub fn add_prepare_same_m0_l3(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.prepare_same_m0_l3.add_ns(ns));
}

/// Add one timed call to the `prepare_same_m0_l4` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `prepare_same_m0_l4`.
/// # Returns:
/// - `()`: Updates the current thread local `prepare_same_m0_l4` counter.
#[inline(always)]
pub fn add_prepare_same_m0_l4(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.prepare_same_m0_l4.add_ns(ns));
}

/// Add one timed call to the `xw_one_body_m0_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_one_body_m0_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_one_body_m0_gen` counter.
#[inline(always)]
pub fn add_xw_one_body_m0_gen(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_one_body_m0_gen.add_ns(ns));
}

/// Add one timed call to the `xw_one_body_m0_l1` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_one_body_m0_l1`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_one_body_m0_l1` counter.
#[inline(always)]
pub fn add_xw_one_body_m0_l1(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_one_body_m0_l1.add_ns(ns));
}

/// Add one timed call to the `xw_one_body_m0_l2` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_one_body_m0_l2`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_one_body_m0_l2` counter.
#[inline(always)]
pub fn add_xw_one_body_m0_l2(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_one_body_m0_l2.add_ns(ns));
}

/// Add one timed call to the `xw_one_body_m0_l3` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_one_body_m0_l3`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_one_body_m0_l3` counter.
#[inline(always)]
pub fn add_xw_one_body_m0_l3(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_one_body_m0_l3.add_ns(ns));
}

/// Add one timed call to the `xw_h2_same_m0_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_same_m0_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_same_m0_gen` counter.
#[inline(always)]
pub fn add_xw_h2_same_m0_gen(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_same_m0_gen.add_ns(ns));
}

/// Add one timed call to the `xw_h2_same_m0_l1` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_same_m0_l1`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_same_m0_l1` counter.
#[inline(always)]
pub fn add_xw_h2_same_m0_l1(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_same_m0_l1.add_ns(ns));
}

/// Add one timed call to the `xw_h2_same_m0_l2` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_same_m0_l2`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_same_m0_l2` counter.
#[inline(always)]
pub fn add_xw_h2_same_m0_l2(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_same_m0_l2.add_ns(ns));
}

/// Add one timed call to the `xw_h2_diff_m0_gen` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_diff_m0_gen`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_diff_m0_gen` counter.
#[inline(always)]
pub fn add_xw_h2_diff_m0_gen(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_diff_m0_gen.add_ns(ns));
}

/// Add one timed call to the `xw_h2_diff_m0_11` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_diff_m0_11`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_diff_m0_11` counter.
#[inline(always)]
pub fn add_xw_h2_diff_m0_11(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_diff_m0_11.add_ns(ns));
}

/// Add one timed call to the `xw_h2_diff_m0_12` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_diff_m0_12`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_diff_m0_12` counter.
#[inline(always)]
pub fn add_xw_h2_diff_m0_12(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_diff_m0_12.add_ns(ns));
}

/// Add one timed call to the `xw_h2_diff_m0_21` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_diff_m0_21`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_diff_m0_21` counter.
#[inline(always)]
pub fn add_xw_h2_diff_m0_21(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_diff_m0_21.add_ns(ns));
}

/// Add one timed call to the `xw_h2_diff_m0_22` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_diff_m0_22`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_diff_m0_22` counter.
#[inline(always)]
pub fn add_xw_h2_diff_m0_22(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_diff_m0_22.add_ns(ns));
}

/// Add one timed call to the `xw_h2_same_m0_l3` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_same_m0_l3`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_same_m0_l3` counter.
#[inline(always)]
pub fn add_xw_h2_same_m0_l3(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_same_m0_l3.add_ns(ns));
}

/// Add one timed call to the `xw_h2_same_m0_l4` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_same_m0_l4`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_same_m0_l4` counter.
#[inline(always)]
pub fn add_xw_h2_same_m0_l4(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_same_m0_l4.add_ns(ns));
}

/// Add one timed call to the `xw_h2_diff_m0_13` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_diff_m0_13`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_diff_m0_13` counter.
#[inline(always)]
pub fn add_xw_h2_diff_m0_13(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_diff_m0_13.add_ns(ns));
}

/// Add one timed call to the `xw_h2_diff_m0_31` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `xw_h2_diff_m0_31`.
/// # Returns:
/// - `()`: Updates the current thread local `xw_h2_diff_m0_31` counter.
#[inline(always)]
pub fn add_xw_h2_diff_m0_31(ns: u64) {
    with_totals(|t| t.nonorthogonalwicks.xw_h2_diff_m0_31.add_ns(ns));
}
