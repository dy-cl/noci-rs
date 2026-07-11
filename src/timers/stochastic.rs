use super::{Counter, with_totals};

/// Timing counters for individual stochastic propagation substeps.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepTotals {
    /// Total time spent constructing the initial persistent populations.
    pub initialise_populations: Counter,
    /// Total time spent constructing sparse unbiased population samples.
    pub sample_populations: Counter,
    /// Total time spent generating pre-overlap population changes.
    pub generate_population_changes: Counter,
    /// Total time spent accumulating local changes and packing remote changes.
    pub acc_pack_updates: Counter,
    /// Total time spent exchanging population changes between MPI ranks.
    pub exchange_population_changes: Counter,
    /// Total time spent exchanging population-change counts between MPI ranks.
    pub exchange_population_change_counts: Counter,
    /// Total time spent exchanging population-change payloads between MPI ranks.
    pub exchange_population_change_payload: Counter,
    /// Total time spent unpacking received population changes.
    pub unpack_population_changes: Counter,
    /// Total time spent taking accumulated changes into sparse storage.
    pub take_population_changes: Counter,
    /// Total time spent applying overlap-transformed population changes.
    pub apply_overlap_changes: Counter,
    /// Total time spent gathering overlap-change counts between MPI ranks.
    pub overlap_change_gather_counts: Counter,
    /// Total time spent applying locally generated overlap changes.
    pub apply_local_overlap_changes: Counter,
    /// Total time spent waiting for gathered overlap changes.
    pub wait_overlap_change_gather: Counter,
    /// Total time spent applying remotely generated overlap changes.
    pub apply_remote_overlap_changes: Counter,
    /// Total time spent gathering sparse populations across MPI ranks.
    pub gather_all_populations: Counter,
    /// Total time spent computing persistent and sampled population statistics.
    pub compute_population_stats: Counter,
    /// Total time spent computing the projected-energy estimator.
    pub compute_projected_energy: Counter,
}

impl StepTotals {
    /// Add another set of stochastic propagation timings into this one.
    /// # Arguments:
    /// - `other`: Stochastic propagation timings to accumulate.
    /// # Returns:
    /// - `()`: Updates this timing collection in place.
    #[inline(always)]
    pub fn merge_from(
        &mut self,
        other: &StepTotals,
    ) {
        self.initialise_populations
            .merge_from(&other.initialise_populations);
        self.sample_populations
            .merge_from(&other.sample_populations);
        self.generate_population_changes
            .merge_from(&other.generate_population_changes);
        self.acc_pack_updates.merge_from(&other.acc_pack_updates);
        self.exchange_population_changes
            .merge_from(&other.exchange_population_changes);
        self.exchange_population_change_counts
            .merge_from(&other.exchange_population_change_counts);
        self.exchange_population_change_payload
            .merge_from(&other.exchange_population_change_payload);
        self.unpack_population_changes
            .merge_from(&other.unpack_population_changes);
        self.take_population_changes
            .merge_from(&other.take_population_changes);
        self.apply_overlap_changes
            .merge_from(&other.apply_overlap_changes);
        self.overlap_change_gather_counts
            .merge_from(&other.overlap_change_gather_counts);
        self.apply_local_overlap_changes
            .merge_from(&other.apply_local_overlap_changes);
        self.wait_overlap_change_gather
            .merge_from(&other.wait_overlap_change_gather);
        self.apply_remote_overlap_changes
            .merge_from(&other.apply_remote_overlap_changes);
        self.gather_all_populations
            .merge_from(&other.gather_all_populations);
        self.compute_population_stats
            .merge_from(&other.compute_population_stats);
        self.compute_projected_energy
            .merge_from(&other.compute_projected_energy);
    }
}

/// Timing counters for stochastic NOCI-QMC stages.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Total time spent in `run_qmc_stochastic_noci`.
    pub run_qmc_stochastic_noci: Counter,
    /// Total time spent in `generate_excited_basis`.
    pub generate_excited_basis: Counter,
    /// Total time spent in `qmc_step`.
    pub qmc_step: Counter,
    /// Timings for individual stochastic propagation substeps.
    pub step: StepTotals,
}

impl Totals {
    /// Add another set of stochastic NOCI-QMC timings into this one.
    /// # Arguments:
    /// - `other`: Stochastic NOCI-QMC timings to accumulate.
    /// # Returns:
    /// - `()`: Updates this timing collection in place.
    #[inline(always)]
    pub fn merge_from(
        &mut self,
        other: &Totals,
    ) {
        self.run_qmc_stochastic_noci
            .merge_from(&other.run_qmc_stochastic_noci);
        self.generate_excited_basis
            .merge_from(&other.generate_excited_basis);
        self.qmc_step.merge_from(&other.qmc_step);
        self.step.merge_from(&other.step);
    }
}

/// Add one timed call to the `run_qmc_stochastic_noci` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_run_qmc_stochastic_noci(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.run_qmc_stochastic_noci.add_ns(ns);
    });
}

/// Add one timed call to the `generate_excited_basis` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_generate_excited_basis(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.generate_excited_basis.add_ns(ns);
    });
}

/// Add one timed call to the `qmc_step` counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_qmc_step(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.qmc_step.add_ns(ns);
    });
}

/// Add one timed call to the initial-population construction counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_initialise_populations(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.step.initialise_populations.add_ns(ns);
    });
}

/// Add one timed call to the population-sampling counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_sample_populations(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.step.sample_populations.add_ns(ns);
    });
}

/// Add one timed call to the population-change generation counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_generate_population_changes(ns: u64) {
    with_totals(|totals| {
        totals
            .stochastic
            .step
            .generate_population_changes
            .add_ns(ns);
    });
}

/// Add one timed call to the local accumulation and packing counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_acc_pack_updates(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.step.acc_pack_updates.add_ns(ns);
    });
}

/// Add one timed call to the population-change exchange counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_exchange_population_changes(ns: u64) {
    with_totals(|totals| {
        totals
            .stochastic
            .step
            .exchange_population_changes
            .add_ns(ns);
    });
}

/// Add one timed call to the population-change count exchange counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_exchange_population_change_counts(ns: u64) {
    with_totals(|totals| {
        totals
            .stochastic
            .step
            .exchange_population_change_counts
            .add_ns(ns);
    });
}

/// Add one timed call to the population-change payload exchange counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_exchange_population_change_payload(ns: u64) {
    with_totals(|totals| {
        totals
            .stochastic
            .step
            .exchange_population_change_payload
            .add_ns(ns);
    });
}

/// Add one timed call to the received population-change unpacking counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_unpack_population_changes(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.step.unpack_population_changes.add_ns(ns);
    });
}

/// Add one timed call to the accumulated population-change extraction counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_take_population_changes(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.step.take_population_changes.add_ns(ns);
    });
}

/// Add one timed call to the global overlap-change application counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_apply_overlap_changes(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.step.apply_overlap_changes.add_ns(ns);
    });
}

/// Add one timed call to the overlap-change count-gather counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_overlap_change_gather_counts(ns: u64) {
    with_totals(|totals| {
        totals
            .stochastic
            .step
            .overlap_change_gather_counts
            .add_ns(ns);
    });
}

/// Add one timed call to the local overlap-change application counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_apply_local_overlap_changes(ns: u64) {
    with_totals(|totals| {
        totals
            .stochastic
            .step
            .apply_local_overlap_changes
            .add_ns(ns);
    });
}

/// Add one timed call to the overlap-change communication wait counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_wait_overlap_change_gather(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.step.wait_overlap_change_gather.add_ns(ns);
    });
}

/// Add one timed call to the remote overlap-change application counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_apply_remote_overlap_changes(ns: u64) {
    with_totals(|totals| {
        totals
            .stochastic
            .step
            .apply_remote_overlap_changes
            .add_ns(ns);
    });
}

/// Add one timed call to the sparse-population all-gather counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_gather_all_populations(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.step.gather_all_populations.add_ns(ns);
    });
}

/// Add one timed call to the population-statistics counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_compute_population_stats(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.step.compute_population_stats.add_ns(ns);
    });
}

/// Add one timed call to the projected-energy counter.
/// # Arguments:
/// - `ns`: Elapsed time in nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local timing counter.
#[inline(always)]
pub fn add_compute_projected_energy(ns: u64) {
    with_totals(|totals| {
        totals.stochastic.step.compute_projected_energy.add_ns(ns);
    });
}
