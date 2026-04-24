// timers/noci.rs
use super::{with_totals, Counter};

/// Timing counters for routines in the `noci` module.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Total time spent in `build_mo_cache`.
    pub build_mo_cache: Counter,
    /// Total time spent in `build_fock_mo_cache`.
    pub build_fock_mo_cache: Counter,
    /// Total time spent in `calculate_s_pair`.
    pub calculate_s_pair: Counter,
    /// Total time spent in `calculate_s_pair_wicks`.
    pub calculate_s_pair_wicks: Counter,
    /// Total time spent in `calculate_s_pair_naive`.
    pub calculate_s_pair_naive: Counter,
    /// Total time spent in `calculate_s_pair_orthogonal`.
    pub calculate_s_pair_orthogonal: Counter,
    /// Total time spent in `calculate_f_pair`.
    pub calculate_f_pair: Counter,
    /// Total time spent in `calculate_f_pair_wicks`.
    pub calculate_f_pair_wicks: Counter,
    /// Total time spent in `calculate_f_pair_naive`.
    pub calculate_f_pair_naive: Counter,
    /// Total time spent in `calculate_f_pair_orthogonal`.
    pub calculate_f_pair_orthogonal: Counter,
    /// Total time spent in `calculate_hs_pair`.
    pub calculate_hs_pair: Counter,
    /// Total time spent in `calculate_hs_pair_wicks`.
    pub calculate_hs_pair_wicks: Counter,
    /// Total time spent in `calculate_hs_pair_naive`.
    pub calculate_hs_pair_naive: Counter,
    /// Total time spent in `calculate_hs_pair_orthogonal`.
    pub calculate_hs_pair_orthogonal: Counter,
    /// Total time spent building full NOCI Fock matrices.
    pub build_full_fock: Counter,
    /// Total time spent building full NOCI overlap matrices.
    pub build_full_overlap: Counter,
    /// Total time spent building full NOCI Hamiltonian and overlap matrices.
    pub build_full_hs: Counter,
}

impl Totals {
    /// Add the contents of another set of NOCI matrix-element timings into this one.
    /// # Arguments:
    /// - `other`: NOCI timing totals whose counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this set of NOCI timing totals in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Totals) {
        self.build_mo_cache.merge_from(&other.build_mo_cache);
        self.build_fock_mo_cache.merge_from(&other.build_fock_mo_cache);
        self.calculate_s_pair.merge_from(&other.calculate_s_pair);
        self.calculate_s_pair_wicks.merge_from(&other.calculate_s_pair_wicks);
        self.calculate_s_pair_naive.merge_from(&other.calculate_s_pair_naive);
        self.calculate_s_pair_orthogonal.merge_from(&other.calculate_s_pair_orthogonal);
        self.calculate_f_pair.merge_from(&other.calculate_f_pair);
        self.calculate_f_pair_wicks.merge_from(&other.calculate_f_pair_wicks);
        self.calculate_f_pair_naive.merge_from(&other.calculate_f_pair_naive);
        self.calculate_f_pair_orthogonal.merge_from(&other.calculate_f_pair_orthogonal);
        self.calculate_hs_pair.merge_from(&other.calculate_hs_pair);
        self.calculate_hs_pair_wicks.merge_from(&other.calculate_hs_pair_wicks);
        self.calculate_hs_pair_naive.merge_from(&other.calculate_hs_pair_naive);
        self.calculate_hs_pair_orthogonal.merge_from(&other.calculate_hs_pair_orthogonal);
        self.build_full_fock.merge_from(&other.build_full_fock);
        self.build_full_overlap.merge_from(&other.build_full_overlap);
        self.build_full_hs.merge_from(&other.build_full_hs);
    }
}

/// Add one timed call to the `build_mo_cache` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `build_mo_cache`.
/// # Returns:
/// - `()`: Updates the current thread local `build_mo_cache` counter.
#[inline(always)]
pub fn add_build_mo_cache(ns: u64) {
    with_totals(|t| t.noci.build_mo_cache.add_ns(ns));
}

/// Add one timed call to the `build_fock_mo_cache` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `build_fock_mo_cache`.
/// # Returns:
/// - `()`: Updates the current thread local `build_fock_mo_cache` counter.
#[inline(always)]
pub fn add_build_fock_mo_cache(ns: u64) {
    with_totals(|t| t.noci.build_fock_mo_cache.add_ns(ns));
}

/// Add one timed call to the `calculate_s_pair` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_s_pair`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_s_pair` counter.
#[inline(always)]
pub fn add_calculate_s_pair(ns: u64) {
    with_totals(|t| t.noci.calculate_s_pair.add_ns(ns));
}

/// Add one timed call to the `calculate_s_pair_wicks` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_s_pair_wicks`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_s_pair_wicks` counter.
#[inline(always)]
pub fn add_calculate_s_pair_wicks(ns: u64) {
    with_totals(|t| t.noci.calculate_s_pair_wicks.add_ns(ns));
}

/// Add one timed call to the `calculate_s_pair_naive` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_s_pair_naive`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_s_pair_naive` counter.
#[inline(always)]
pub fn add_calculate_s_pair_naive(ns: u64) {
    with_totals(|t| t.noci.calculate_s_pair_naive.add_ns(ns));
}

/// Add one timed call to the `calculate_s_pair_orthogonal` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_s_pair_orthogonal`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_s_pair_orthogonal` counter.
#[inline(always)]
pub fn add_calculate_s_pair_orthogonal(ns: u64) {
    with_totals(|t| t.noci.calculate_s_pair_orthogonal.add_ns(ns));
}

/// Add one timed call to the `calculate_f_pair` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_f_pair`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_f_pair` counter.
#[inline(always)]
pub fn add_calculate_f_pair(ns: u64) {
    with_totals(|t| t.noci.calculate_f_pair.add_ns(ns));
}

/// Add one timed call to the `calculate_f_pair_wicks` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_f_pair_wicks`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_f_pair_wicks` counter.
#[inline(always)]
pub fn add_calculate_f_pair_wicks(ns: u64) {
    with_totals(|t| t.noci.calculate_f_pair_wicks.add_ns(ns));
}

/// Add one timed call to the `calculate_f_pair_naive` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_f_pair_naive`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_f_pair_naive` counter.
#[inline(always)]
pub fn add_calculate_f_pair_naive(ns: u64) {
    with_totals(|t| t.noci.calculate_f_pair_naive.add_ns(ns));
}

/// Add one timed call to the `calculate_f_pair_orthogonal` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_f_pair_orthogonal`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_f_pair_orthogonal` counter.
#[inline(always)]
pub fn add_calculate_f_pair_orthogonal(ns: u64) {
    with_totals(|t| t.noci.calculate_f_pair_orthogonal.add_ns(ns));
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

/// Add one timed call to the `build_full_fock` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `build_full_fock`.
/// # Returns:
/// - `()`: Updates the current thread local `build_full_fock` counter.
#[inline(always)]
pub fn add_build_full_fock(ns: u64) {
    with_totals(|t| t.noci.build_full_fock.add_ns(ns));
}

/// Add one timed call to the `build_full_overlap` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `build_full_overlap`.
/// # Returns:
/// - `()`: Updates the current thread local `build_full_overlap` counter.
#[inline(always)]
pub fn add_build_full_overlap(ns: u64) {
    with_totals(|t| t.noci.build_full_overlap.add_ns(ns));
}

/// Add one timed call to the `build_full_hs` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `build_full_hs`.
/// # Returns:
/// - `()`: Updates the current thread local `build_full_hs` counter.
#[inline(always)]
pub fn add_build_full_hs(ns: u64) {
    with_totals(|t| t.noci.build_full_hs.add_ns(ns));
}
