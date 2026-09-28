// nocc/deterministic/mod.rs
//! Deterministic solution of the GNOCC amplitude equations.

mod solver;

// Restricted type re-exports.
pub(crate) use solver::AmplitudeSolution;

// Restricted function re-exports.
pub(crate) use solver::solve_amplitudes;
