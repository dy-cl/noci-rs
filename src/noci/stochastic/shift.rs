// noci/stochastic/shift.rs

// Standard library imports.
use std::sync::Mutex;

// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;
use rayon::prelude::*;

// Crate-root imports.
use crate::input::Input;
use crate::qmc::{exchange_to_owners, update_shift as qmc_update_shift};

// Parent/sibling imports.
use super::state::{
    NOCIMPIScratch, PopulationStats, PropagationState, QMCRunInfo, ShiftTangent, TangentWorker,
};

/// Update the unshifted propagator's population-control shift.
/// # Arguments:
/// - `stats`: Current population statistics.
/// - `state`: Current propagation state.
/// - `shift`: Population-control shift.
/// - `input`: User input options.
/// # Returns:
/// - `()`: Updates the shift and cached previous population.
pub(in crate::noci::stochastic) fn update_shift(
    stats: &PopulationStats,
    state: &mut PropagationState,
    shift: &mut f64,
    input: &Input,
) {
    let qmc = input.qmc.as_ref().unwrap();
    let dteff = input.prop_ref().dt * qmc.ncycles as f64;

    qmc_update_shift(
        shift,
        &mut state.reached,
        (stats.nw, state.prev_pop.nw),
        (qmc.target_population, qmc.shift_damping, 0.0, dteff),
    );

    state.prev_pop = *stats;
}

/// Update a range-population shift from its already propagated tangent
/// `T = \partial N'/\partial E_s`.
/// The damped Newton step uses
/// `E_s' = E_s - \zeta N_M(\partial N_M/\partial E_s)^{-1}`
/// `[\ln(N_M/N_M^{prev}) + \kappa\ln(N_M^{prev}/N_*)]`, where
/// `\partial N_M/\partial E_s = \operatorname{sign}(N')^T T`.
/// # Arguments:
/// - `stats`: Current population statistics.
/// - `state`: Current propagation state.
/// - `shift`: Range-population control shift.
/// - `tangent`: Rank-local propagated tangent `T`.
/// - `run`: Rank-local determinant ownership metadata.
/// - `world`: MPI communicator.
/// - `input`: User input options.
/// # Returns:
/// - `()`: Updates the shared range-population shift when its Jacobian is usable.
pub(in crate::noci::stochastic) fn update_shift_tangent(
    stats: &PopulationStats,
    state: &mut PropagationState,
    shift: &mut f64,
    tangent: &[f64],
    run: &QMCRunInfo,
    world: &impl Communicator,
    input: &Input,
) {
    // For `N_M = ||N'||_1` and an already propagated tangent
    // `T = \partial N'/\partial E_s`, `\partial N_M/\partial E_s = sign(N')^T T`.
    let local = state
        .mc
        .populations
        .par_iter()
        .zip(tangent.par_iter())
        .map(|(&population, &derivative)| (population.signum() * derivative, derivative.abs()))
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    let (range_derivative, tangent_norm) = if run.nranks == 1 {
        local
    } else {
        let mut global = [0.0; 2];
        world.all_reduce_into(&[local.0, local.1], &mut global, SystemOperation::sum());
        (global[0], global[1])
    };
    let qmc = input.qmc.as_ref().unwrap();
    let previous = state.prev_pop.nw;
    let current = stats.nw;

    if !state.reached && current >= qmc.target_population {
        state.reached = true;
    }

    let usable = state.reached
        && current.is_finite()
        && current > 0.0
        && previous.is_finite()
        && previous > 0.0
        && range_derivative.is_finite()
        && tangent_norm.is_finite()
        && tangent_norm > 0.0
        && range_derivative.abs() > f64::EPSILON * tangent_norm;

    if usable {
        // The supplied tangent contains every report-cycle `dt` factor. No additional
        // `1/(dt*ncycles)` belongs in the Newton update.
        let growth = (current / previous).ln();
        let residual = if qmc.population_restoring == 0.0 {
            growth
        } else {
            growth + qmc.population_restoring * (previous / qmc.target_population).ln()
        };
        let next = *shift - qmc.shift_damping * current / range_derivative * residual;

        if next.is_finite() {
            *shift = next;
        }
    }

    state.prev_pop = *stats;
}

/// Exchange remote range-propagator tangent contributions and add them to owner-local storage.
/// # Arguments:
/// - `tangent`: Report-level shift tangent.
/// - `mpi`: Reusable MPI communication scratch.
/// - `world`: MPI communicator.
/// - `run`: Rank-local determinant ownership metadata.
/// # Returns:
/// - `()`: Adds received tangent updates to owner-local dense storage.
pub(in crate::noci::stochastic) fn exchange_shift_tangent(
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
        tangent.add(update.det as usize, update.dn, true);
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
pub(in crate::noci::stochastic) fn reduce_shift_tangent<T: TangentWorker + Send>(
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
pub(in crate::noci::stochastic) fn collect_shift_tangent<T: TangentWorker>(
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
