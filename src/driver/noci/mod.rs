// driver/noci/mod.rs
//! Orchestration of the NOCI-family post-reference methods.

mod deterministic;
mod selected;
mod stochastic;

// Restricted function re-exports.
pub(super) use deterministic::run_qmc_deterministic_noci;
pub(super) use selected::run_snoci;
pub(super) use stochastic::run_qmc_stochastic_noci;
