// noci/orthogonal/mod.rs
//! Prepared Slater-Condon evaluation in one parent's orthonormal MO determinant basis.

mod eval;

// Crate-visible function re-exports.
pub(crate) use eval::{
    xw_hamiltonian_orthogonal_prepared, xw_hamiltonian_orthogonal_prepared_batched,
};
