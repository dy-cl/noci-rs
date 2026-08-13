// noci/factorise/onebody/gpu/mod.rs
//! CubeCL GPU implementation of factorised one-body NOCI operator contractions.

mod backend;
mod consts;
mod contract;
mod data;
mod diagonals;
mod factors;
mod orthogonal;

// Crate-visible type re-exports.
pub(crate) use backend::GpuOneBodyBackend;
