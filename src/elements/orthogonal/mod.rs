// elements/orthogonal/mod.rs

//! Prepared Slater-Condon evaluation in one parent's orthonormal MO determinant basis.
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

// Crate-visible function re-exports.
pub(crate) use eval::{
    xw_hamiltonian_orthogonal_prepared, xw_hamiltonian_orthogonal_prepared_batched,
};
pub(crate) use pairs::{
    calculate_f_pair_orthogonal, calculate_h_pairs_orthogonal_batched, calculate_s_pair_orthogonal,
};

// Restricted function re-exports.
pub(in crate::elements) use pairs::{calculate_hs_pair_orthogonal, calculate_m_pair_orthogonal};
