// timers/mod.rs

pub mod general;
pub mod deterministic;
pub mod stochastic;
pub mod snoci;
pub mod noci;
pub mod nonorthogonalwicks;

use std::cell::RefCell;
use std::time::Duration;

use mpi::traits::*;

/// Apply a helper macro to every timing counter field in `Totals`.
/// This macro centralises the list of counters used for packing, unpacking, and size counting.
/// # Arguments:
/// - `$tot`: Identifier naming the `Totals` value whose counter fields will be visited.
/// - `$f`: Identifier naming a macro that is invoked once for each counter field.
/// # Returns:
/// - Expands to a sequence of `$f!(...)` invocations, one for each timing counter field.
#[allow(unused_macros)]
macro_rules! for_each_counter {
    ($tot:ident, $f:ident) => {
        $f!($tot.general.run_pyscf);
        $f!($tot.general.run_scf);
        $f!($tot.general.run_reference_noci);
        $f!($tot.general.calculate_noci_energy);
        $f!($tot.general.build_wicks_shared);

        $f!($tot.deterministic.run_qmc_deterministic_noci);
        $f!($tot.deterministic.generate_excited_basis);
        $f!($tot.deterministic.build_noci_hs);
        $f!($tot.deterministic.propagate);

        $f!($tot.stochastic.run_qmc_stochastic_noci);
        $f!($tot.stochastic.generate_excited_basis);
        $f!($tot.stochastic.qmc_step);
        $f!($tot.stochastic.step.initialise_walkers);
        $f!($tot.stochastic.step.propagate_iteration);
        $f!($tot.stochastic.step.acc_pack_updates);
        $f!($tot.stochastic.step.compute_populations);
        $f!($tot.stochastic.step.apply_delta);
        $f!($tot.stochastic.step.update_p);
        $f!($tot.stochastic.step.update_p_gather_counts);
        $f!($tot.stochastic.step.update_p_local_overlap);
        $f!($tot.stochastic.step.update_p_wait);
        $f!($tot.stochastic.step.update_p_apply);
        $f!($tot.stochastic.step.update_projected_energy);
        $f!($tot.stochastic.step.communicate_spawn_updates);
        $f!($tot.stochastic.step.gather_all_walkers);
        $f!($tot.stochastic.step.observables_allreduce);

        $f!($tot.snoci.run_snoci);
        $f!($tot.snoci.snoci_step);
        $f!($tot.snoci.solve_current_space);
        $f!($tot.snoci.candidate_pool_new);
        $f!($tot.snoci.candidate_pool_update);
        $f!($tot.snoci.candidate_pool_filter_candidates);
        $f!($tot.snoci.build_pseudoinverse);
        $f!($tot.snoci.project_candidate_space);
        $f!($tot.snoci.build_candidate_h_ai);
        $f!($tot.snoci.build_generalised_fock);
        $f!($tot.snoci.build_fock_mo_cache);
        $f!($tot.snoci.update_wicks_fock);
        $f!($tot.snoci.build_focks);
        $f!($tot.snoci.build_omega_fock);
        $f!($tot.snoci.gmres);
        $f!($tot.snoci.select_candidates);

        $f!($tot.noci.calculate_hs_pair);
        $f!($tot.noci.calculate_hs_pair_wicks);
        $f!($tot.noci.calculate_hs_pair_naive);
        $f!($tot.noci.calculate_hs_pair_orthogonal);

        $f!($tot.nonorthogonalwicks.prepare_same);
        $f!($tot.nonorthogonalwicks.prepare_same_gen);
        $f!($tot.nonorthogonalwicks.prepare_same_m0);
        $f!($tot.nonorthogonalwicks.get_det_adjt_same);
        $f!($tot.nonorthogonalwicks.get_det_adjt_diff);
        $f!($tot.nonorthogonalwicks.construct_determinant_indices);
        $f!($tot.nonorthogonalwicks.lg_overlap);
        $f!($tot.nonorthogonalwicks.lg_h1);
        $f!($tot.nonorthogonalwicks.lg_one_body_gen);
        $f!($tot.nonorthogonalwicks.lg_one_body_m0);
        $f!($tot.nonorthogonalwicks.lg_h2_same);
        $f!($tot.nonorthogonalwicks.lg_h2_same_gen);
        $f!($tot.nonorthogonalwicks.lg_h2_same_m0);
        $f!($tot.nonorthogonalwicks.lg_h2_diff);
        $f!($tot.nonorthogonalwicks.lg_h2_diff_gen);
        $f!($tot.nonorthogonalwicks.lg_h2_diff_m0);
        $f!($tot.nonorthogonalwicks.prepare_same_m0_l1);
        $f!($tot.nonorthogonalwicks.prepare_same_m0_l2);
        $f!($tot.nonorthogonalwicks.lg_one_body_m0_gen);
        $f!($tot.nonorthogonalwicks.lg_one_body_m0_l1);
        $f!($tot.nonorthogonalwicks.lg_one_body_m0_l2);
        $f!($tot.nonorthogonalwicks.lg_h2_same_m0_gen);
        $f!($tot.nonorthogonalwicks.lg_h2_same_m0_l1);
        $f!($tot.nonorthogonalwicks.lg_h2_same_m0_l2);
        $f!($tot.nonorthogonalwicks.lg_h2_diff_m0_gen);
        $f!($tot.nonorthogonalwicks.lg_h2_diff_m0_11);
        $f!($tot.nonorthogonalwicks.lg_h2_diff_m0_22);
        $f!($tot.nonorthogonalwicks.lg_h2_same_m0_l3);
        $f!($tot.nonorthogonalwicks.lg_h2_diff_m0_13);
        $f!($tot.nonorthogonalwicks.lg_h2_diff_m0_31);
    };
}

/// Container for one timing counter storing total elapsed time and number of calls.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counter {
    /// Total accumulated time in nanoseconds.
    pub ns: u64,
    /// Number of timed calls contributing to `ns`.
    pub calls: u64,
}

impl Counter {
    /// Add one timed call of duration `ns` nanoseconds to this counter.
    /// # Arguments:
    /// - `ns`: Elapsed time in nanoseconds for the timed region.
    /// # Returns:
    /// - `()`: Updates the accumulated nanoseconds and increments the call count.
    #[inline(always)]
    pub fn add_ns(&mut self, ns: u64) {
        self.ns += ns;
        self.calls += 1;
    }

    /// Convert the accumulated nanoseconds in this counter into a `Duration`.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Duration`: Total elapsed time represented by this counter.
    #[inline(always)]
    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.ns)
    }

    /// Add the contents of another timing counter into this one.
    /// # Arguments:
    /// - `other`: Counter whose accumulated nanoseconds and call count are to be added.
    /// # Returns:
    /// - `()`: Updates this counter in place by summing nanoseconds and call counts.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Counter) {
        self.ns += other.ns;
        self.calls += other.calls;
    }
}

/// Top level collection of timing totals grouped by subsystem.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Timing counters for general high level workflow stages.
    pub general: general::Totals,
    /// Timing counters for deterministic NOCI-QMC stages.
    pub deterministic: deterministic::Totals,
    /// Timing counters for stochastic NOCI-QMC stages.
    pub stochastic: stochastic::Totals,
    /// Timing counters for SNOCI stages.
    pub snoci: snoci::Totals,
    /// Timing counters for routines in the `noci` module.
    pub noci: noci::Totals,
    /// Timing counters for routines in the `nonorthogonalwicks` module.
    pub nonorthogonalwicks: nonorthogonalwicks::Totals,
}

impl Totals {
    /// Add the contents of another top level timing collection into this one.
    /// # Arguments:
    /// - `other`: Top level timing totals whose subsystem counters are to be accumulated.
    /// # Returns:
    /// - `()`: Updates this top level timing collection in place.
    #[inline(always)]
    pub fn merge_from(&mut self, other: &Totals) {
        self.general.merge_from(&other.general);
        self.deterministic.merge_from(&other.deterministic);
        self.stochastic.merge_from(&other.stochastic);
        self.snoci.merge_from(&other.snoci);
        self.noci.merge_from(&other.noci);
        self.nonorthogonalwicks.merge_from(&other.nonorthogonalwicks);
    }
    
    /// Number of `u64` entries needed to pack this timing structure.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `usize`: Number of packed `u64` entries.
    pub fn flat_len() -> usize {
        let mut n = 0usize;
        macro_rules! count { ($x:expr) => {{ let _ = &$x; n += 2; }}; }
        let t = Totals::default();
        for_each_counter!(t, count);
        n
    }

    /// Pack all timing counters into a flat `u64` vector as `(ns, calls)` pairs.
    /// # Arguments:
    /// - `out`: Output buffer to append packed timing data to.
    /// # Returns:
    /// - `()`: Appends packed timing data to `out`.
    pub fn pack(&self, out: &mut Vec<u64>) {
        macro_rules! push_pair { ($x:expr) => {{ out.push($x.ns); out.push($x.calls); }}; }
        for_each_counter!(self, push_pair);
    }

    /// Unpack a flat `(ns, calls)` buffer into a timing structure.
    /// # Arguments:
    /// - `buf`: Flat packed timing buffer.
    /// # Returns:
    /// - `Totals`: Unpacked timing totals.
    pub fn unpack(buf: &[u64]) -> Self {
        let mut t = Totals::default();
        let mut i = 0usize;
        macro_rules! pull_pair {
            ($x:expr) => {{
                $x.ns = buf[i];
                $x.calls = buf[i + 1];
                i += 2;
            }};
        }
        for_each_counter!(t, pull_pair);
        debug_assert_eq!(i, buf.len());
        t
    }
}

thread_local! {
    /// Thread local store for timing counters. Each thread accumulates into its own
    /// `Totals` instance to avoid contention during timed regions.
    static TOTALS: RefCell<Totals> = RefCell::new(Totals::default());
}

/// Take a copy of the timing totals accumulated on the current thread and all Rayon worker
/// threads on this rank, then reduce across MPI ranks using a max reduction.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns:
/// - `Totals`: Max timing counters across all MPI ranks after summing all Rayon threads on each rank.
pub fn snapshot_all_mpi(world: &impl CommunicatorCollectives) -> Totals {
    let local = snapshot_all();
    let mut send = Vec::with_capacity(Totals::flat_len());
    local.pack(&mut send);

    let mut recv = vec![0_u64; send.len()];
    world.all_reduce_into(&send[..], &mut recv[..], mpi::collective::SystemOperation::max());

    Totals::unpack(&recv)
}

/// Borrow the current thread local timing totals mutably and apply a closure to them.
/// # Arguments:
/// - `f`: Closure receiving a mutable reference to the current thread's `Totals`.
/// # Returns:
/// - `R`: Return value of the closure `f`.
#[inline(always)]
pub fn with_totals<R>(f: impl FnOnce(&mut Totals) -> R) -> R {
    TOTALS.with(|cell| {
        let mut totals = cell.borrow_mut();
        f(&mut totals)
    })
}

/// Take a copy of the current thread local timing totals.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Totals`: Copy of the current thread's accumulated timing totals.
pub fn snapshot() -> Totals {
    TOTALS.with(|cell| *cell.borrow())
}

/// Reset the current thread local timing totals back to their default zero state.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Clears all timing counters for the current thread.
pub fn reset() {
    TOTALS.with(|cell| *cell.borrow_mut() = Totals::default());
}

/// Take a copy of the timing totals accumulated on the current thread together with all Rayon
/// worker threads and return their sum.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Totals`: Sum of timing counters across the current thread and all Rayon worker threads.
pub fn snapshot_all() -> Totals {
    let mut total = snapshot();
    for t in rayon::broadcast(|_| snapshot()) {
        total.merge_from(&t);
    }
    total
}

/// Reset the timing totals on the current thread and all Rayon worker threads back to zero.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Clears all timing counters on the current thread and all Rayon worker threads.
pub fn reset_all() {
    reset();
    rayon::broadcast(|_| reset());
}

/// Time a block of code and record the elapsed nanoseconds using the supplied callback.
/// This macro is only active when the `timings` feature is enabled.
/// # Arguments:
/// - `$path`: Function taking a `u64` nanosecond count and updating the chosen counter.
/// - `$body`: Block of code to execute and time.
/// # Returns:
/// - Value returned by `$body`.
#[macro_export]
macro_rules! time_call {
    ($add:path, $body:block) => {{
        #[cfg(feature = "timings")]
        {
            struct __TimeCallGuard {
                t0: std::time::Instant,
                add: fn(u64),
            }

            impl Drop for __TimeCallGuard {
                fn drop(&mut self) {
                    (self.add)(self.t0.elapsed().as_nanos() as u64);
                }
            }

            let __timings_guard = __TimeCallGuard {
                t0: std::time::Instant::now(),
                add: $add,
            };

            let __timings_ret = { $body };
            std::mem::drop(__timings_guard);
            __timings_ret
        }

        #[cfg(not(feature = "timings"))]
        {
            $body
        }
    }};
}
