// timers/general.rs

use super::{with_totals, Counter};

/// Timing counters for general high level workflow stages.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Total time spent in `run_pyscf`.
    pub run_pyscf: Counter,
    /// Total time spent in `run_scf`.
    pub run_scf: Counter,
    /// Total time spent in `run_reference_noci`.
    pub run_reference_noci: Counter,
    /// Total time spent in `calculate_noci_energy`.
    pub calculate_noci_energy: Counter,
    /// Total time spent in `build_wicks_shared`.
    pub build_wicks_shared: Counter,
}

impl Totals {
    /// Add the contents of another set of general workflow timings into this one.
    /// # Arguments:
    /// - `other`: General workflow timing totals whose counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this set of general workflow timing totals in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Totals) {
        self.run_pyscf.merge_from(&other.run_pyscf);
        self.run_scf.merge_from(&other.run_scf);
        self.run_reference_noci.merge_from(&other.run_reference_noci);
        self.calculate_noci_energy.merge_from(&other.calculate_noci_energy);
        self.build_wicks_shared.merge_from(&other.build_wicks_shared);
    }
}

/// Add one timed call to the `run_pyscf` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `run_pyscf`.
/// # Returns:
/// - `()`: Updates the current thread local `run_pyscf` counter.
#[inline(always)]
pub fn add_run_pyscf(ns: u64) {
    with_totals(|t| t.general.run_pyscf.add_ns(ns));
}

/// Add one timed call to the `run_scf` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `run_scf`.
/// # Returns:
/// - `()`: Updates the current thread local `run_scf` counter.
#[inline(always)]
pub fn add_run_scf(ns: u64) {
    with_totals(|t| t.general.run_scf.add_ns(ns));
}

/// Add one timed call to the `run_reference_noci` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `run_reference_noci`.
/// # Returns:
/// - `()`: Updates the current thread local `run_reference_noci` counter.
#[inline(always)]
pub fn add_run_reference_noci(ns: u64) {
    with_totals(|t| t.general.run_reference_noci.add_ns(ns));
}

/// Add one timed call to the `calculate_noci_energy` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `calculate_noci_energy`.
/// # Returns:
/// - `()`: Updates the current thread local `calculate_noci_energy` counter.
#[inline(always)]
pub fn add_calculate_noci_energy(ns: u64) {
    with_totals(|t| t.general.calculate_noci_energy.add_ns(ns));
}

/// Add one timed call to the `build_wicks_shared` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `build_wicks_shared`.
/// # Returns:
/// - `()`: Updates the current thread local `build_wicks_shared` counter.
#[inline(always)]
pub fn add_build_wicks_shared(ns: u64) {
    with_totals(|t| t.general.build_wicks_shared.add_ns(ns));
}
