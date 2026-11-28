// timers/selected.rs

// Parent/sibling imports.
use super::{Counter, with_totals};

/// Timing counters for SNOCI stages.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Total time spent in `run_snoci`.
    pub run_snoci: Counter,
    /// Total time spent in `snoci_step`.
    pub snoci_step: Counter,
    /// Total time spent in `CandidatePool::new`.
    pub candidate_pool_new: Counter,
    /// Total time spent in `CandidatePool::update`.
    pub candidate_pool_update: Counter,
    /// Total time spent building the generalised Fock matrices.
    pub build_generalised_fock: Counter,
    /// Total time spent in `gmres`.
    pub gmres: Counter,
    /// Total time spent in `build_snoci_couplings`.
    pub build_snoci_couplings: Counter,
    /// Total time spent in `build_candidate_m`.
    pub build_candidate_m: Counter,
    /// Total time spent in `build_candidate_m_diag`.
    pub build_candidate_m_diag: Counter,
    /// Total time spent in `apply_candidate_m`.
    pub apply_candidate_m: Counter,
    /// Total time spent in `apply_omega_m`.
    pub apply_omega_m: Counter,
    /// Total time spent in `build_omega_m_diag`.
    pub build_omega_m_diag: Counter,
}

impl Totals {
    /// Add the contents of another set of SNOCI timings into this one.
    /// # Arguments:
    /// - `other`: SNOCI timing totals whose counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this set of SNOCI timing totals in place.
    #[inline(always)]
    pub fn merge_from(
        &mut self,
        other: &Totals,
    ) {
        self.run_snoci.merge_from(&other.run_snoci);
        self.snoci_step.merge_from(&other.snoci_step);
        self.candidate_pool_new
            .merge_from(&other.candidate_pool_new);
        self.candidate_pool_update
            .merge_from(&other.candidate_pool_update);
        self.build_generalised_fock
            .merge_from(&other.build_generalised_fock);
        self.gmres.merge_from(&other.gmres);
        self.build_snoci_couplings
            .merge_from(&other.build_snoci_couplings);
        self.build_candidate_m.merge_from(&other.build_candidate_m);
        self.build_candidate_m_diag
            .merge_from(&other.build_candidate_m_diag);
        self.apply_candidate_m.merge_from(&other.apply_candidate_m);
        self.apply_omega_m.merge_from(&other.apply_omega_m);
        self.build_omega_m_diag
            .merge_from(&other.build_omega_m_diag);
    }
}

/// Add one timed call to the `run_snoci` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `run_snoci`.
/// # Returns:
/// - `()`: Updates the current thread local `run_snoci` counter.
#[inline(always)]
pub fn add_run_snoci(ns: u64) {
    with_totals(|t| t.selected.run_snoci.add_ns(ns));
}

/// Add one timed call to the `snoci_step` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `snoci_step`.
/// # Returns:
/// - `()`: Updates the current thread local `snoci_step` counter.
#[inline(always)]
pub fn add_snoci_step(ns: u64) {
    with_totals(|t| t.selected.snoci_step.add_ns(ns));
}

/// Add one timed call to the `candidate_pool_new` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `CandidatePool::new`.
/// # Returns:
/// - `()`: Updates the current thread local `candidate_pool_new` counter.
#[inline(always)]
pub fn add_candidate_pool_new(ns: u64) {
    with_totals(|t| t.selected.candidate_pool_new.add_ns(ns));
}

/// Add one timed call to the `candidate_pool_update` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `CandidatePool::update`.
/// # Returns:
/// - `()`: Updates the current thread local `candidate_pool_update` counter.
#[inline(always)]
pub fn add_candidate_pool_update(ns: u64) {
    with_totals(|t| t.selected.candidate_pool_update.add_ns(ns));
}

/// Add one timed call to the `build_generalised_fock` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one generalised Fock build.
/// # Returns:
/// - `()`: Updates the current thread local `build_generalised_fock` counter.
#[inline(always)]
pub fn add_build_generalised_fock(ns: u64) {
    with_totals(|t| t.selected.build_generalised_fock.add_ns(ns));
}

/// Add one timed call to the `gmres` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `gmres`.
/// # Returns:
/// - `()`: Updates the current thread local `gmres` counter.
#[inline(always)]
pub fn add_gmres(ns: u64) {
    with_totals(|t| t.selected.gmres.add_ns(ns));
}

/// Add one timed call to the `build_snoci_couplings` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `build_snoci_couplings`.
/// # Returns:
/// - `()`: Updates the current thread local `build_snoci_couplings` counter.
#[inline(always)]
pub fn add_build_snoci_couplings(ns: u64) {
    with_totals(|t| t.selected.build_snoci_couplings.add_ns(ns));
}

/// Add one timed call to the `apply_candidate_m` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `apply_candidate_m`.
/// # Returns:
/// - `()`: Updates the current thread local `apply_candidate_m` counter.
#[inline(always)]
pub fn add_apply_candidate_m(ns: u64) {
    with_totals(|t| t.selected.apply_candidate_m.add_ns(ns));
}

/// Add one timed call to the `build_candidate_m` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `build_candidate_m`.
/// # Returns:
/// - `()`: Updates the current thread local `build_candidate_m` counter.
#[inline(always)]
pub fn add_build_candidate_m(ns: u64) {
    with_totals(|t| t.selected.build_candidate_m.add_ns(ns));
}

/// Add one timed call to the `build_candidate_m_diag` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `build_candidate_m_diag`.
/// # Returns:
/// - `()`: Updates the current thread local `build_candidate_m_diag` counter.
#[inline(always)]
pub fn add_build_candidate_m_diag(ns: u64) {
    with_totals(|t| t.selected.build_candidate_m_diag.add_ns(ns));
}

/// Add one timed call to the `apply_omega_m` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `apply_omega_m`.
/// # Returns:
/// - `()`: Updates the current thread local `apply_omega_m` counter.
#[inline(always)]
pub fn add_apply_omega_m(ns: u64) {
    with_totals(|t| t.selected.apply_omega_m.add_ns(ns));
}

/// Add one timed call to the `build_omega_m_diag` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `build_omega_m_diag`.
/// # Returns:
/// - `()`: Updates the current thread local `build_omega_m_diag` counter.
#[inline(always)]
pub fn add_build_omega_m_diag(ns: u64) {
    with_totals(|t| t.selected.build_omega_m_diag.add_ns(ns));
}
