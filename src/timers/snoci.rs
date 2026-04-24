// timers/snoci.rs

use super::{with_totals, Counter};

/// Timing counters for SNOCI stages.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Total time spent in `run_snoci`.
    pub run_snoci: Counter,
    /// Total time spent in `snoci_step`.
    pub snoci_step: Counter,
    /// Total time spent in `solve_current_space`.
    pub solve_current_space: Counter,
    /// Total time spent in `CandidatePool::new`.
    pub candidate_pool_new: Counter,
    /// Total time spent in `CandidatePool::update`.
    pub candidate_pool_update: Counter,
    /// Total time spent in `CandidatePool::filter_candidates`
    pub candidate_pool_filter_candidates: Counter,
    /// Total time spent building the pseudoinverse.
    pub build_pseudoinverse: Counter,
    /// Total time spent in `project_candidate_space`.
    pub project_candidate_space: Counter,
    /// Total time spent building candidate-current `H_ai`.
    pub build_candidate_h_ai: Counter,
    /// Total time spent building the generalised Fock matrices.
    pub build_generalised_fock: Counter,
    /// Total time spent in `gmres`.
    pub gmres: Counter,
}

impl Totals {
    /// Add the contents of another set of SNOCI timings into this one.
    /// # Arguments:
    /// - `other`: SNOCI timing totals whose counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this set of SNOCI timing totals in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Totals) {
        self.run_snoci.merge_from(&other.run_snoci);
        self.snoci_step.merge_from(&other.snoci_step);
        self.solve_current_space.merge_from(&other.solve_current_space);
        self.candidate_pool_new.merge_from(&other.candidate_pool_new);
        self.candidate_pool_update.merge_from(&other.candidate_pool_update);
        self.candidate_pool_filter_candidates.merge_from(&other.candidate_pool_filter_candidates);
        self.build_pseudoinverse.merge_from(&other.build_pseudoinverse);
        self.project_candidate_space.merge_from(&other.project_candidate_space);
        self.build_candidate_h_ai.merge_from(&other.build_candidate_h_ai);
        self.build_generalised_fock.merge_from(&other.build_generalised_fock);
        self.gmres.merge_from(&other.gmres);
    }
}

/// Add one timed call to the `run_snoci` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `run_snoci`.
/// # Returns:
/// - `()`: Updates the current thread local `run_snoci` counter.
#[inline(always)]
pub fn add_run_snoci(ns: u64) {
    with_totals(|t| t.snoci.run_snoci.add_ns(ns));
}

/// Add one timed call to the `snoci_step` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `snoci_step`.
/// # Returns:
/// - `()`: Updates the current thread local `snoci_step` counter.
#[inline(always)]
pub fn add_snoci_step(ns: u64) {
    with_totals(|t| t.snoci.snoci_step.add_ns(ns));
}

/// Add one timed call to the `solve_current_space` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `solve_current_space`.
/// # Returns:
/// - `()`: Updates the current thread local `solve_current_space` counter.
#[inline(always)]
pub fn add_solve_current_space(ns: u64) {
    with_totals(|t| t.snoci.solve_current_space.add_ns(ns));
}

/// Add one timed call to the `candidate_pool_new` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `CandidatePool::new`.
/// # Returns:
/// - `()`: Updates the current thread local `candidate_pool_new` counter.
#[inline(always)]
pub fn add_candidate_pool_new(ns: u64) {
    with_totals(|t| t.snoci.candidate_pool_new.add_ns(ns));
}

/// Add one timed call to the `candidate_pool_update` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `CandidatePool::update`.
/// # Returns:
/// - `()`: Updates the current thread local `candidate_pool_update` counter.
#[inline(always)]
pub fn add_candidate_pool_update(ns: u64) {
    with_totals(|t| t.snoci.candidate_pool_update.add_ns(ns));
}

/// Add one timed call to the `candidate_pool_filter_candidates` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `CandidatePool::filter_candidates`.
/// # Returns:
/// - `()`: Updates the current thread local `candidate_pool_filter_candidates` counter.
#[inline(always)]
pub fn add_candidate_pool_filter_candidates(ns: u64) {
    with_totals(|t| t.snoci.candidate_pool_filter_candidates.add_ns(ns));
}

/// Add one timed call to the `build_pseudoinverse` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one pseudoinverse build.
/// # Returns:
/// - `()`: Updates the current thread local `build_pseudoinverse` counter.
#[inline(always)]
pub fn add_build_pseudoinverse(ns: u64) {
    with_totals(|t| t.snoci.build_pseudoinverse.add_ns(ns));
}

/// Add one timed call to the `project_candidate_space` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `project_candidate_space`.
/// # Returns:
/// - `()`: Updates the current thread local `project_candidate_space` counter.
#[inline(always)]
pub fn add_project_candidate_space(ns: u64) {
    with_totals(|t| t.snoci.project_candidate_space.add_ns(ns));
}

/// Add one timed call to the `build_candidate_h_ai` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one candidate-current Hamiltonian build.
/// # Returns:
/// - `()`: Updates the current thread local `build_candidate_h_ai` counter.
#[inline(always)]
pub fn add_build_candidate_h_ai(ns: u64) {
    with_totals(|t| t.snoci.build_candidate_h_ai.add_ns(ns));
}

/// Add one timed call to the `build_generalised_fock` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one generalised Fock build.
/// # Returns:
/// - `()`: Updates the current thread local `build_generalised_fock` counter.
#[inline(always)]
pub fn add_build_generalised_fock(ns: u64) {
    with_totals(|t| t.snoci.build_generalised_fock.add_ns(ns));
}

/// Add one timed call to the `gmres` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `gmres`.
/// # Returns:
/// - `()`: Updates the current thread local `gmres` counter.
#[inline(always)]
pub fn add_gmres(ns: u64) {
    with_totals(|t| t.snoci.gmres.add_ns(ns));
}
