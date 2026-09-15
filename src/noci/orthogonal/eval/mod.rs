// noci/orthogonal/eval/mod.rs
//! Scalar, batched, and SIMD prepared orthogonal Hamiltonian kernels.

mod dispatch;
mod preparehamiltonian;

// Crate-visible function re-exports.
pub(crate) use preparehamiltonian::{
    xw_hamiltonian_orthogonal_prepared, xw_hamiltonian_orthogonal_prepared_batched,
};
