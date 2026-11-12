// elements/orthogonal/mod.rs

//! Fixed-rank Slater-Condon evaluation in one parent's orthonormal MO determinant basis.
//!
//! # References
//!
//! - Mayer, "Determinant Wave Functions," in *Simple Theorems, Proofs, and Derivations in
//!   Quantum Chemistry* (Springer, 2003),
//!   [doi:10.1007/978-1-4757-6519-9_5](https://doi.org/10.1007/978-1-4757-6519-9_5).

mod eval;
mod pairs;

// Crate-visible type re-exports.
pub(crate) use pairs::OrthogonalHamiltonianScratch;

// Restricted type re-exports.
pub(in crate::elements) use eval::{OrthogonalFock, OrthogonalHamiltonian, OrthogonalOverlap};

// Crate-visible function re-exports.
pub(crate) use pairs::{
    calculate_f_pair_orthogonal, calculate_h_pairs_orthogonal_batched, calculate_s_pair_orthogonal,
};

// Restricted function re-exports.
pub(in crate::elements) use eval::xw_orthogonal_batched;
pub(in crate::elements) use pairs::{calculate_pairs_orthogonal_batched, connection_data};
