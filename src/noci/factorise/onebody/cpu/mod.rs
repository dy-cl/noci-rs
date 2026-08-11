// noci/factorise/onebody/cpu/mod.rs
//! CPU implementation of factorised one-body NOCI operator contractions.

mod backend;
mod contract;
mod diagonals;
mod factors;
mod orthogonal;

// Crate-visible type re-exports.
pub(crate) use backend::CpuOneBodyBackend;
