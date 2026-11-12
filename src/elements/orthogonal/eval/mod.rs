// elements/orthogonal/eval/mod.rs

//! Scalar and SIMD fixed-rank orthogonal overlap, Fock, and Hamiltonian kernels, one file per
//! operator family, with the shared parent-local batches in `batch.rs` on the lane batching of
//! `elements/batch.rs`.

mod batch;
mod dispatch;
mod fock;
mod hamiltonian;
mod overlap;

// Crate-visible type re-exports.
pub(crate) use fock::OrthogonalFock;
pub(crate) use hamiltonian::OrthogonalHamiltonian;
pub(crate) use overlap::OrthogonalOverlap;

// Crate-visible function re-exports.
pub(crate) use batch::xw_orthogonal_batched;
pub(crate) use fock::xw_fock_orthogonal;
pub(crate) use overlap::xw_overlap_orthogonal;
