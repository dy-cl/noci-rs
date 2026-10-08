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

// Crate-visible function re-exports.
pub(crate) use eval::{
    xw_fock_orthogonal, xw_fock_orthogonal_batched, xw_hamiltonian_orthogonal,
    xw_hamiltonian_orthogonal_batched, xw_overlap_orthogonal, xw_overlap_orthogonal_batched,
};
pub(crate) use pairs::{
    calculate_f_pair_orthogonal, calculate_h_pairs_orthogonal_batched, calculate_s_pair_orthogonal,
};

// Restricted function re-exports.
pub(in crate::elements) use pairs::{
    calculate_f_pairs_orthogonal_batched, calculate_hs_pair_orthogonal,
    calculate_hs_pairs_orthogonal_batched, calculate_m_pair_orthogonal,
    calculate_m_pairs_orthogonal_batched, calculate_s_pairs_orthogonal_batched,
};
