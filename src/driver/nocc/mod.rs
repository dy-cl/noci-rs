// driver/nocc/mod.rs
//! Orchestration of the NOCC-family post-reference methods.

#[cfg(feature = "gnocc")]
mod gnocc;
mod jmnoccmc;

// Restricted function re-exports.
#[cfg(feature = "gnocc")]
pub(super) use gnocc::run_gnocc;
pub(super) use jmnoccmc::run_jmnoccmc;
