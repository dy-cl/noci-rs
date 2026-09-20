// noci/orthogonal/mod.rs

//! Prepared Slater-Condon evaluation in one parent's orthonormal MO determinant basis.
//!
//! # References
//!
//! - Mayer, "Determinant Wave Functions," in *Simple Theorems, Proofs, and Derivations in
//!   Quantum Chemistry* (Springer, 2003),
//!   [doi:10.1007/978-1-4757-6519-9_5](https://doi.org/10.1007/978-1-4757-6519-9_5).

mod connection;
mod eval;

// Crate-visible type re-exports.
pub(crate) use connection::OrthogonalConnection;

// Crate-visible function re-exports.
pub(crate) use eval::{
    xw_hamiltonian_orthogonal_prepared, xw_hamiltonian_orthogonal_prepared_batched,
};
