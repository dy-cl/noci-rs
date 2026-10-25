// nocc/deterministic/mod.rs
//! Deterministic NOCC: imaginary-time propagation of the Jeziorski-Monkhorst NOCC amplitudes
//! and, behind the `gnocc` feature, solution of the GNOCC amplitude equations.

mod jmnoccmc;
#[cfg(feature = "gnocc")]
mod solver;
mod write;

// Restricted type re-exports.
pub(crate) use jmnoccmc::JMEquations;
#[cfg(feature = "gnocc")]
pub(crate) use solver::AmplitudeSolution;

// Restricted function re-exports.
pub(crate) use jmnoccmc::{jm_space, propagate_amplitudes, report_converged};
#[cfg(feature = "gnocc")]
pub(crate) use solver::solve_amplitudes;
