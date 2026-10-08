// elements/orthogonal/eval/mod.rs

//! Scalar, batched, and SIMD fixed-rank orthogonal overlap, Fock, and Hamiltonian kernels.

mod dispatch;
mod fock;
mod hamiltonian;
mod overlap;

// Crate-visible function re-exports.
pub(crate) use fock::{xw_fock_orthogonal, xw_fock_orthogonal_batched};
pub(crate) use hamiltonian::{xw_hamiltonian_orthogonal, xw_hamiltonian_orthogonal_batched};
pub(crate) use overlap::{xw_overlap_orthogonal, xw_overlap_orthogonal_batched};
