// nocc/deterministic/mod.rs
//! Deterministic NOCC: imaginary-time propagation of the Jeziorski-Monkhorst NOCC populations,
//! the noise-free counterpart of the stochastic propagation, with analytic Jacobian
//! diagnostics, and, behind the `gnocc` feature, solution of the GNOCC amplitude equations.

mod jmnoccmc;
#[cfg(feature = "gnocc")]
mod solver;
mod write;

// Public type re-exports.
pub use jmnoccmc::{JMEquations, JMPopulations, JMSpace};

// Public function re-exports.
pub use jmnoccmc::{jm_space, propagate_populations, report_converged};

// Restricted type re-exports.
#[cfg(feature = "gnocc")]
pub(crate) use solver::AmplitudeSolution;

// Restricted function re-exports.
#[cfg(feature = "gnocc")]
pub(crate) use solver::solve_amplitudes;
