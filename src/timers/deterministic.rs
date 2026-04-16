// timers/deterministic.rs

use super::{with_totals, Counter};

/// Timing counters for deterministic NOCI-QMC stages.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Total time spent in `run_qmc_deterministic_noci`.
    pub run_qmc_deterministic_noci: Counter,
    /// Total time spent in `generate_excited_basis` during deterministic NOCI-QMC.
    pub generate_excited_basis: Counter,
    /// Total time spent in `build_noci_hs` during deterministic NOCI-QMC.
    pub build_noci_hs: Counter,
    /// Total time spent in `propagate`.
    pub propagate: Counter,
}

impl Totals {
    /// Add the contents of another set of deterministic NOCI-QMC timings into this one.
    /// # Arguments:
    /// - `other`: Deterministic NOCI-QMC timing totals whose counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this set of deterministic timing totals in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Totals) {
        self.run_qmc_deterministic_noci.merge_from(&other.run_qmc_deterministic_noci);
        self.generate_excited_basis.merge_from(&other.generate_excited_basis);
        self.build_noci_hs.merge_from(&other.build_noci_hs);
        self.propagate.merge_from(&other.propagate);
    }
}

/// Add one timed call to the `run_qmc_deterministic_noci` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `run_qmc_deterministic_noci`.
/// # Returns:
/// - `()`: Updates the current thread local `run_qmc_deterministic_noci` counter.
#[inline(always)]
pub fn add_run_qmc_deterministic_noci(ns: u64) {
    with_totals(|t| t.deterministic.run_qmc_deterministic_noci.add_ns(ns));
}

/// Add one timed call to the `generate_excited_basis` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one deterministic call to `generate_excited_basis`.
/// # Returns:
/// - `()`: Updates the current thread local `generate_excited_basis` counter.
#[inline(always)]
pub fn add_generate_excited_basis(ns: u64) {
    with_totals(|t| t.deterministic.generate_excited_basis.add_ns(ns));
}

/// Add one timed call to the `build_noci_hs` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one deterministic call to `build_noci_hs`.
/// # Returns:
/// - `()`: Updates the current thread local `build_noci_hs` counter.
#[inline(always)]
pub fn add_build_noci_hs(ns: u64) {
    with_totals(|t| t.deterministic.build_noci_hs.add_ns(ns));
}

/// Add one timed call to the `propagate` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `propagate`.
/// # Returns:
/// - `()`: Updates the current thread local `propagate` counter.
#[inline(always)]
pub fn add_propagate(ns: u64) {
    with_totals(|t| t.deterministic.propagate.add_ns(ns));
}
