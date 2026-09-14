// stochastic/shift.rs

// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;
use rayon::prelude::*;

// Crate-root imports.
use crate::input::Input;

// Parent/sibling imports.
use super::state::{PopulationStats, PropagationState, QMCRunInfo};

/// Update the ordinary-walker population-control shift.
/// # Arguments:
/// - `stats`: Current population statistics.
/// - `state`: Current propagation state.
/// - `shift`: Population-control shift.
/// - `input`: User input options.
/// # Returns:
/// - `()`: Updates the shift and cached previous population.
pub(in crate::stochastic) fn update_shift(
    stats: &PopulationStats,
    state: &mut PropagationState,
    shift: &mut f64,
    input: &Input,
) {
    let qmc = input.qmc.as_ref().unwrap();
    let dteff = input.prop_ref().dt * qmc.ncycles as f64;

    if !state.reached && stats.nw >= qmc.target_population {
        state.reached = true;
    }

    if state.reached {
        *shift -= (qmc.shift_damping / dteff) * (stats.nw / state.prev_pop.nw).ln();
    }

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
pub(in crate::stochastic) fn update_shift_tangent(
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
    let (metric_derivative, tangent_norm) = if run.nranks == 1 {
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
        && metric_derivative.is_finite()
        && tangent_norm.is_finite()
        && tangent_norm > 0.0
        && metric_derivative.abs() > f64::EPSILON * tangent_norm;

    if usable {
        // The supplied tangent contains every report-cycle `dt` factor. No additional
        // `1/(dt*ncycles)` belongs in the Newton update.
        let growth = (current / previous).ln();
        let residual = if qmc.population_restoring == 0.0 {
            growth
        } else {
            growth + qmc.population_restoring * (previous / qmc.target_population).ln()
        };
        let next = *shift - qmc.shift_damping * current / metric_derivative * residual;

        if next.is_finite() {
            *shift = next;
        }
    }

    state.prev_pop = *stats;
}
