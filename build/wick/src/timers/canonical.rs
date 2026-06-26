// timers/canonical.rs

use super::{Counter, with_totals};

/// Maximum histogram bucket used by cumulant sparsifier timings.
const HIST: usize = 25;

/// Cumulant sparsifier search totals.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Best`: Search statistics.
#[derive(Clone, Copy, Debug, Default)]
pub struct Best {
    /// Sparsifier calls by cumulant rank.
    pub calls_by_rank: [u64; 5],
    /// Input support histogram.
    pub input_support: [u64; HIST],
    /// Gram-rank histogram.
    pub gram_rank: [u64; HIST],
    /// Basis fallback support histogram.
    pub basis_support: [u64; HIST],
    /// Effective search limit histogram.
    pub limit: [u64; HIST],
    /// Candidate supports visited by candidate size.
    pub visited: [u64; HIST],
    /// Candidate supports rejected before exact solve.
    pub rejected: u64,
    /// Exact integer span checks performed.
    pub checks: u64,
    /// Exact rational solves performed.
    pub solves: u64,
    /// Successful support size histogram.
    pub success: [u64; HIST],
    /// Known fallback returns.
    pub fallback_returns: u64,
    /// Time spent enumerating supports.
    pub enumerate_ns: u64,
    /// Time spent in cheap rejection.
    pub reject_ns: u64,
    /// Time spent in exact solve.
    pub solve_ns: u64,
    /// Maximum single-call elapsed time.
    pub max_ns: u64,
}

impl Best {
    /// Add another set of sparsifier stats into this one.
    /// # Arguments:
    /// - `other`: Sparsifier stats to add.
    /// # Returns:
    /// - `()`: Updates this stats collection in place.
    #[inline(always)]
    pub fn merge_from(
        &mut self,
        other: &Best,
    ) {
        for i in 0..self.calls_by_rank.len() {
            self.calls_by_rank[i] += other.calls_by_rank[i];
        }

        for i in 0..HIST {
            self.input_support[i] += other.input_support[i];
            self.gram_rank[i] += other.gram_rank[i];
            self.basis_support[i] += other.basis_support[i];
            self.limit[i] += other.limit[i];
            self.visited[i] += other.visited[i];
            self.success[i] += other.success[i];
        }

        self.rejected += other.rejected;
        self.checks += other.checks;
        self.solves += other.solves;
        self.fallback_returns += other.fallback_returns;
        self.enumerate_ns += other.enumerate_ns;
        self.reject_ns += other.reject_ns;
        self.solve_ns += other.solve_ns;
        self.max_ns = self.max_ns.max(other.max_ns);
    }
}

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
    /// Cumulant sparsifier search statistics.
    pub best: Best,
}

impl Totals {
    /// Add another set of canonicalisation timings into this one.
    /// # Arguments:
    /// - `other`: Canonicalisation timings to accumulate.
    /// # Returns:
    /// - `()`: Updates this timing collection in place.
    #[inline(always)]
    pub fn merge_from(
        &mut self,
        other: &Totals,
    ) {
        self.accumulate.merge_from(&other.accumulate);
        self.merge.merge_from(&other.merge);
        self.finish.merge_from(&other.finish);
        self.intoexpr.merge_from(&other.intoexpr);
        self.spar.merge_from(&other.spar);
        self.final_sum.merge_from(&other.final_sum);
        self.best.merge_from(&other.best);
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

/// Add one histogram count.
/// # Arguments:
/// - `xs`: Histogram buckets.
/// - `i`: Bucket index.
/// - `n`: Count to add.
/// # Returns:
/// - `()`: Updates the histogram.
#[inline(always)]
#[cfg(feature = "timings")]
fn add_hist(
    xs: &mut [u64],
    i: usize,
    n: u64,
) {
    if i < xs.len() {
        xs[i] += n;
    }
}

/// Add one sparsifier rank call.
/// # Arguments:
/// - `rank`: Cumulant rank.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_rank(rank: usize) {
    with_totals(|totals| add_hist(&mut totals.canonical.best.calls_by_rank, rank, 1));
}

/// Add one sparsifier input-support count.
/// # Arguments:
/// - `support`: Input support size.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_input(support: usize) {
    with_totals(|totals| add_hist(&mut totals.canonical.best.input_support, support, 1));
}

/// Add one sparsifier Gram-rank count.
/// # Arguments:
/// - `rank`: Gram rank.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_gram_rank(rank: usize) {
    with_totals(|totals| add_hist(&mut totals.canonical.best.gram_rank, rank, 1));
}

/// Add one sparsifier fallback-support count.
/// # Arguments:
/// - `support`: Basis fallback support size.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_basis_support(support: usize) {
    with_totals(|totals| add_hist(&mut totals.canonical.best.basis_support, support, 1));
}

/// Add one sparsifier search-limit count.
/// # Arguments:
/// - `limit`: Effective search limit.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_limit(limit: usize) {
    with_totals(|totals| add_hist(&mut totals.canonical.best.limit, limit, 1));
}

/// Add visited sparsifier candidates.
/// # Arguments:
/// - `size`: Candidate support size.
/// - `count`: Candidate count.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_visited(
    size: usize,
    count: u64,
) {
    with_totals(|totals| add_hist(&mut totals.canonical.best.visited, size, count));
}

/// Add sparsifier pre-solve rejections.
/// # Arguments:
/// - `count`: Rejection count.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_rejected(count: u64) {
    with_totals(|totals| totals.canonical.best.rejected += count);
}

/// Add sparsifier exact integer checks.
/// # Arguments:
/// - `count`: Check count.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_checks(count: u64) {
    with_totals(|totals| totals.canonical.best.checks += count);
}

/// Add sparsifier exact rational solves.
/// # Arguments:
/// - `count`: Solve count.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_solves(count: u64) {
    with_totals(|totals| totals.canonical.best.solves += count);
}

/// Add one sparsifier success-support count.
/// # Arguments:
/// - `support`: Successful support size.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_success(support: usize) {
    with_totals(|totals| add_hist(&mut totals.canonical.best.success, support, 1));
}

/// Add one sparsifier fallback return.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_fallback_return() {
    with_totals(|totals| totals.canonical.best.fallback_returns += 1);
}

/// Add sparsifier enumeration time.
/// # Arguments:
/// - `ns`: Elapsed nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_enumerate(ns: u64) {
    with_totals(|totals| totals.canonical.best.enumerate_ns += ns);
}

/// Add sparsifier pre-solve rejection time.
/// # Arguments:
/// - `ns`: Elapsed nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_reject_time(ns: u64) {
    with_totals(|totals| totals.canonical.best.reject_ns += ns);
}

/// Add sparsifier exact solve time.
/// # Arguments:
/// - `ns`: Elapsed nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_solve_time(ns: u64) {
    with_totals(|totals| totals.canonical.best.solve_ns += ns);
}

/// Add sparsifier maximum call time.
/// # Arguments:
/// - `ns`: Elapsed nanoseconds.
/// # Returns:
/// - `()`: Updates the current thread-local counters.
#[inline(always)]
#[cfg(feature = "timings")]
pub(crate) fn add_best_max(ns: u64) {
    with_totals(|totals| {
        totals.canonical.best.max_ns = totals.canonical.best.max_ns.max(ns);
    });
}
