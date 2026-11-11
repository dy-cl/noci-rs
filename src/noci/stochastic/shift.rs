// noci/stochastic/shift.rs

// Standard library imports.
use std::sync::Mutex;

// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;
use rayon::prelude::*;

// Crate-root imports.
use crate::noci::{OverlapFactors, OverlapScratch, SpinFactorisation};
use crate::qmc::{PopulationUpdate, QmcRng, compress_sparse, exchange_to_owners, target_cutoff};

// Parent/sibling imports.
use super::fri::compress_dense_to_sparse;
use super::propagate::NOCIRun;
use super::sapply::apply_overlap_population_changes;
use super::state::{NOCIMPIScratch, PropagationWorker, QMCRunInfo, ShiftTangent};

/// The shift tangent of one report of a range propagator, `B = \partial\Delta/\partial E_s`:
/// gathered from the workers, compressed, and propagated through the overlap to
/// `\partial N'/\partial E_s = S\Phi(B)`, the Jacobian of the tangent shift controller.
pub(in crate::noci::stochastic) struct ReportTangent {
    /// Report tangent gathered from the workers.
    gathered: ShiftTangent,
    /// Sparse tangent after compression.
    sparse: Vec<PopulationUpdate>,
    /// Propagated tangent on the owned determinants.
    propagated: Vec<f64>,
    /// Last FRI cutoff, which seeds the next search.
    hint: f64,
}

impl ReportTangent {
    /// Allocate the report tangent of a run.
    /// # Arguments:
    /// - `run`: Rank-local topology of the run.
    /// # Returns
    /// - `Self`: Empty report tangent.
    pub(in crate::noci::stochastic) fn new(run: &QMCRunInfo) -> Self {
        Self {
            gathered: ShiftTangent::new(run.ndets),
            sparse: Vec::new(),
            propagated: vec![0.0; run.owned.len()],
            hint: 0.0,
        }
    }

    /// Gather the report tangent from the workers, compress it to its target NNZ and propagate it
    /// through the overlap. `E[\Phi(B)|B] = B`, so linearity of `S` gives `E[S\Phi(B)|B] = SB`.
    /// # Arguments:
    /// - `self`: Report tangent.
    /// - `workers`: Per-thread workers holding their tangent contributions.
    /// - `run`: Shared run state.
    /// - `rng`: Random stream of the tangent compression.
    /// - `overlap`: Factorisation and factor tables of the overlap.
    /// - `scratch`: Reusable storage of one application of the overlap.
    /// # Returns
    /// - `&[f64]`: Propagated tangent `\partial N'/\partial E_s` on the owned determinants.
    pub(in crate::noci::stochastic) fn propagate<T: PropagationWorker + Send, C: Communicator>(
        &mut self,
        workers: &mut [Mutex<T>],
        run: &mut NOCIRun<'_, C>,
        rng: &mut QmcRng,
        overlap: (&SpinFactorisation, &OverlapFactors),
        scratch: &mut OverlapScratch,
    ) -> &[f64] {
        let qmc = run.data.input.qmc.as_ref().unwrap();
        let target = qmc.fri.shift_tangent_target_nnz;

        // One rank reduces the nearly dense thread tangents; several ranks route them to owners.
        if run.run.nranks == 1 {
            reduce_shift_tangent(workers, &mut self.gathered.values);
            self.hint = target_cutoff(&self.gathered.values, target, self.hint, |v| v.abs());
            compress_dense_to_sparse(&mut self.gathered.values, self.hint, rng, &mut self.sparse);
        } else {
            collect_shift_tangent(workers, &mut self.gathered);
            exchange_shift_tangent(&mut self.gathered, &mut run.mpi, run.world, &run.run);
            self.gathered.take_sparse(&mut self.sparse);
            self.hint = target_cutoff(&self.sparse, target, self.hint, |update| update.dn.abs());
            compress_sparse(&mut self.sparse, self.hint, rng);
        }

        // Propagate the compressed tangent through the overlap.
        self.propagated.fill(0.0);
        apply_overlap_population_changes(
            (&mut self.propagated, &self.sparse),
            run.data,
            overlap,
            &run.run,
            (run.world, &mut run.mpi),
            scratch,
        );
        &self.propagated
    }
}

/// Return the Newton scale of the tangent shift controller,
/// `s = \zeta N_M(\partial N_M/\partial E_s)^{-1}`, where for `N_M = ||N'||_1` and the propagated
/// tangent `T = \partial N'/\partial E_s`,
/// `\partial N_M/\partial E_s = \operatorname{sign}(N')^T T`.
/// The tangent already contains every report-cycle `dt` factor, so no `1/(dt\,n_\text{cyc})`
/// enters the scale.
/// # Arguments:
/// - `populations`: Rank-local populations `N'` after the step.
/// - `tangent`: Rank-local propagated tangent `T`.
/// - `norms`: Current and previous population norms `(N_M, N_M^{\text{prev}})`.
/// - `damping`: Shift damping `\zeta`.
/// - `run`: Rank-local determinant ownership metadata.
/// - `world`: MPI communicator.
/// # Returns:
/// - `Option<f64>`: Newton scale, or `None` when the norms or the Jacobian are unusable.
pub(in crate::noci::stochastic) fn tangent_scale(
    populations: &[f64],
    tangent: &[f64],
    norms: (f64, f64),
    damping: f64,
    run: &QMCRunInfo,
    world: &impl Communicator,
) -> Option<f64> {
    let local = populations
        .par_iter()
        .zip(tangent.par_iter())
        .map(|(&population, &derivative)| (population.signum() * derivative, derivative.abs()))
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    let (derivative, norm) = if run.nranks == 1 {
        local
    } else {
        let mut global = [0.0; 2];
        world.all_reduce_into(&[local.0, local.1], &mut global, SystemOperation::sum());
        (global[0], global[1])
    };

    // The step needs positive finite norms and a Jacobian well above rounding of the tangent.
    let (current, previous) = norms;
    let usable = current.is_finite()
        && current > 0.0
        && previous.is_finite()
        && previous > 0.0
        && derivative.is_finite()
        && norm.is_finite()
        && norm > 0.0
        && derivative.abs() > f64::EPSILON * norm;
    usable.then(|| damping * current / derivative)
}

/// Exchange remote range-propagator tangent contributions and add them to owner-local storage.
/// # Arguments:
/// - `tangent`: Report-level shift tangent.
/// - `mpi`: Reusable MPI communication scratch.
/// - `world`: MPI communicator.
/// - `run`: Rank-local determinant ownership metadata.
/// # Returns:
/// - `()`: Adds received tangent updates to owner-local dense storage.
fn exchange_shift_tangent(
    tangent: &mut ShiftTangent,
    mpi: &mut NOCIMPIScratch,
    world: &impl CommunicatorCollectives,
    run: &QMCRunInfo,
) {
    if run.nranks <= 1 {
        tangent.remote.clear();
        return;
    }
    // Each rank holds partial contributions to `B_w = \sum_a dt \sum_x S_{wx}\tilde N_x`.
    // Redistribute them by determinant ownership before report-level FRI.
    mpi.send_ranked.append(&mut tangent.remote);
    for update in exchange_to_owners(world, run.irank, mpi) {
        tangent.add(update.index as usize, update.dn, true);
    }
    mpi.send_contig.clear();
}

/// Reduce thread-local dense retained-space tangents into one report-level vector.
/// Computes `B_w = \sum_t B_w^{(t)}` in determinant-major order for either range propagator.
/// # Arguments:
/// - `workers`: Per-thread dense tangent accumulators.
/// - `tangent`: Dense report-level tangent receiving thread sum.
/// # Returns:
/// - `()`: Replaces `tangent` with report tangent and clears thread-local tangents.
fn reduce_shift_tangent<T: PropagationWorker + Send>(
    workers: &mut [Mutex<T>],
    tangent: &mut [f64],
) {
    // Single-rank B is nearly dense. Reduce determinant-major to avoid touched-index bookkeeping
    // and sparse materialisation before target-NNZ compression.
    let sources = workers
        .iter_mut()
        .map(|worker| worker.get_mut().unwrap().tangent().values.as_slice())
        .collect::<Vec<_>>();

    tangent.par_iter_mut().enumerate().for_each(|(det, value)| {
        *value = sources.iter().map(|source| source[det]).sum();
    });

    drop(sources);
    workers.par_iter_mut().for_each(|worker| {
        worker.get_mut().unwrap().tangent().values.fill(0.0);
    });
}

/// Drain sparse worker shift tangents into the report accumulator for MPI exchange.
/// # Arguments:
/// - `workers`: Persistent SApply or BApply propagation workers.
/// - `tangent`: Owner-local report tangent and remote contributions.
/// # Returns:
/// - `()`: Adds touched worker values and drains remote updates.
fn collect_shift_tangent<T: PropagationWorker>(
    workers: &mut [Mutex<T>],
    tangent: &mut ShiftTangent,
) {
    for worker in workers.iter_mut() {
        let source = worker.get_mut().unwrap().tangent();
        for det in source.changed.drain(..) {
            let dn = source.values[det];
            source.values[det] = 0.0;
            tangent.add(det, dn, true);
        }
        tangent.remote.append(&mut source.remote);
    }
}
