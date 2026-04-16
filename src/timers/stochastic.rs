// timers/stochastic.rs

use super::{with_totals, Counter};

/// Timing counters for individual stochastic propagation substeps.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepTotals {
    /// Total time spent in `initialise_walkers`.
    pub initialise_walkers: Counter,
    /// Total time spent in `propagate_iteration`.
    pub propagate_iteration: Counter,
    /// Total time spent in `acc_pack_updates`.
    pub acc_pack_updates: Counter,
    /// Total time spent in `exchange_updates`.
    pub exchange_updates: Counter,
    /// Total time spent in `unpack_received_updates`.
    pub unpack_received_updates: Counter,
    /// Total time spent in `compute_populations`
    pub compute_populations: Counter,
    /// Total time spent in `apply_delta`.
    pub apply_delta: Counter,
    /// Total time spent in `update_p`.
    pub update_p: Counter,
    /// Total time spent in `update_projected_energy`.
    pub update_projected_energy: Counter,
}

impl StepTotals {
    /// Add the contents of another set of stochastic propagation step timings into this one.
    /// # Arguments:
    /// - `other`: Stochastic propagation step timing totals whose counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this set of stochastic propagation step timings in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &StepTotals) {
        self.initialise_walkers.merge_from(&other.initialise_walkers);
        self.propagate_iteration.merge_from(&other.propagate_iteration);
        self.acc_pack_updates.merge_from(&other.acc_pack_updates);
        self.exchange_updates.merge_from(&other.exchange_updates);
        self.unpack_received_updates.merge_from(&other.unpack_received_updates);
        self.compute_populations.merge_from(&other.compute_populations);
        self.apply_delta.merge_from(&other.apply_delta);
        self.update_p.merge_from(&other.update_p);
        self.update_projected_energy.merge_from(&other.update_projected_energy);
    }
}

impl Totals {
    /// Add the contents of another set of stochastic NOCI-QMC timings into this one.
    /// # Arguments:
    /// - `other`: Stochastic NOCI-QMC timing totals whose counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this set of stochastic timing totals in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Totals) {
        self.run_qmc_stochastic_noci.merge_from(&other.run_qmc_stochastic_noci);
        self.generate_excited_basis.merge_from(&other.generate_excited_basis);
        self.qmc_step.merge_from(&other.qmc_step);
        self.step.merge_from(&other.step);
    }
}

/// Timing counters for stochastic NOCI-QMC stages.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Total time spent in `run_qmc_stochastic_noci`.
    pub run_qmc_stochastic_noci: Counter,
    /// Total time spent in `generate_excited_basis` during stochastic NOCI-QMC.
    pub generate_excited_basis: Counter,
    /// Total time spent in `qmc_step`.
    pub qmc_step: Counter,
    /// Timings for individual stochastic propagation substeps.
    pub step: StepTotals,
}

/// Add one timed call to the `run_qmc_stochastic_noci` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `run_qmc_stochastic_noci`.
/// # Returns:
/// - `()`: Updates the current thread local `run_qmc_stochastic_noci` counter.
#[inline(always)]
pub fn add_run_qmc_stochastic_noci(ns: u64) {
    with_totals(|t| t.stochastic.run_qmc_stochastic_noci.add_ns(ns));
}

/// Add one timed call to the `generate_excited_basis` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one stochastic call to `generate_excited_basis`.
/// # Returns:
/// - `()`: Updates the current thread local `generate_excited_basis` counter.
#[inline(always)]
pub fn add_generate_excited_basis(ns: u64) {
    with_totals(|t| t.stochastic.generate_excited_basis.add_ns(ns));
}

/// Add one timed call to the `qmc_step` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `qmc_step`.
/// # Returns:
/// - `()`: Updates the current thread local `qmc_step` counter.
#[inline(always)]
pub fn add_qmc_step(ns: u64) {
    with_totals(|t| t.stochastic.qmc_step.add_ns(ns));
}

/// Add one timed call to the `initialise_walkers` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `initialise_qmc_state`.
/// # Returns:
/// - `()`: Updates the current thread local `initialise_qmc_state` counter.
#[inline(always)]
pub fn add_initialise_walkers(ns: u64) {
    with_totals(|t| t.stochastic.step.initialise_walkers.add_ns(ns));
}

/// Add one timed call to the `propagate_iteration` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `propagate_iteration`.
/// # Returns:
/// - `()`: Updates the current thread local `propagate_iteration` counter.
#[inline(always)]
pub fn add_propagate_iteration(ns: u64) {
    with_totals(|t| t.stochastic.step.propagate_iteration.add_ns(ns));
}

/// Add one timed call to the `acc_pack_updates` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `acc_pack_updates`.
/// # Returns:
/// - `()`: Updates the current thread local `acc_pack_updates` counter.
#[inline(always)]
pub fn add_acc_pack_updates(ns: u64) {
    with_totals(|t| t.stochastic.step.acc_pack_updates.add_ns(ns));
}

/// Add one timed call to the `exchange_updates` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `exchange_updates`.
/// # Returns:
/// - `()`: Updates the current thread local `exchange_updates` counter.
#[inline(always)]
pub fn add_exchange_updates(ns: u64) {
    with_totals(|t| t.stochastic.step.exchange_updates.add_ns(ns));
}

/// Add one timed call to the `unpack_received_updates` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `unpack_received_updates`.
/// # Returns:
/// - `()`: Updates the current thread local `unpack_received_updates` counter.
#[inline(always)]
pub fn add_unpack_received_updates(ns: u64) {
    with_totals(|t| t.stochastic.step.unpack_received_updates.add_ns(ns));
}

/// Add one timed call to the `compute_populations` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `compute_populations`.
/// # Returns:
/// - `()`: Updates the current thread local `compute_populations` counter.
#[inline(always)]
pub fn add_compute_populations(ns: u64) {
    with_totals(|t| t.stochastic.step.compute_populations.add_ns(ns));
}

/// Add one timed call to the `apply_delta` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `apply_delta`.
/// # Returns:
/// - `()`: Updates the current thread local `apply_delta` counter.
#[inline(always)]
pub fn add_apply_delta(ns: u64) {
    with_totals(|t| t.stochastic.step.apply_delta.add_ns(ns));
}

/// Add one timed call to the `update_p` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `update_p`.
/// # Returns:
/// - `()`: Updates the current thread local `update_p` counter.
#[inline(always)]
pub fn add_update_p(ns: u64) {
    with_totals(|t| t.stochastic.step.update_p.add_ns(ns));
}

/// Add one timed call to the `update_projected_energy` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds for one call to `update_projected_energy`.
/// # Returns:
/// - `()`: Updates the current thread local `update_projected_energy` counter.
#[inline(always)]
pub fn add_update_projected_energy(ns: u64) {
    with_totals(|t| t.stochastic.step.update_projected_energy.add_ns(ns));
}
