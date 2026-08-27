// driver/mod.rs
//! Top-level orchestration of an `noci-rs` calculation.
//!
//! The driver connects input parsing, molecular-integral generation, SCF-state construction,
//! reference-basis generation and the requested post-SCF calculation. It determines the order
//! in which the scientific modules are invoked but does not itself implement their numerical
//! methods.
//!
//! For each molecular geometry, the driver:
//!
//! 1. Generates atomic-orbital integrals and initial data;
//! 2. Converges the requested real and holomorphic SCF states;
//! 3. Constructs and reports the selected reference NOCI basis;
//! 4. Prepares molecular-orbital caches and nonorthogonal Wick intermediates;
//! 5. Evaluates the reference NOCI state;
//! 6. Runs deterministic propagation, NOCIQMC, NOCI-PT2 or SNOCI as requested;
//! 7. Writes the requested orbital, matrix, coefficient, histogram and restart data.
//!
//! During a geometry scan, converged states from the previous geometry may be reused as
//! initial guesses so that corresponding SCF branches can be followed between geometries.

mod config;
mod deterministic;
mod geometry;
mod post;
mod reference;
mod report;
mod run;
mod scf;
mod snoci;
mod stochastic;
mod types;

// Public function re-exports.
pub use config::load_config;
pub use run::run;
