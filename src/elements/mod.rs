// elements/mod.rs
//! Matrix elements and expectation values between nonorthogonal determinants.
//!
//! Every determinant-pair quantity, the overlap, Hamiltonian, generalised Fock, shifted
//! `M = F - E_0 S` and transition reduced density matrices, is evaluated by one of three
//! engines chosen per pair:
//!
//! - `orthogonal`: Slater-Condon rules when both determinants descend from the same
//!   orthonormal parent;
//! - `naive`: the generalised Slater-Condon rules for nonorthogonal pairs;
//! - `nonorthogonalwicks`: the extended nonorthogonal Wick theorem for nonorthogonal pairs.
//!
//! `operator` defines the overlap, shifted Fock and Hamiltonian operators by their evaluation in
//! each engine, and `pairs` routes every pair of a batch to its engine for any operator; `rdm`
//! assembles the reduced density matrices of NOCI states from the pair densities. Both method
//! families, `noci` and `nocc`, evaluate their matrix elements here.

pub mod nonorthogonalwicks;

mod cache;
mod naive;
mod operator;
mod orthogonal;
mod pairs;
mod rdm;
mod types;

// Public type re-exports.
pub use types::{FockMOCache, MOCache, NOCIData};

// Public function re-exports.
pub use cache::build_mo_cache;
pub use nonorthogonalwicks::build_wicks_shared;

// Crate-visible type re-exports.
pub(crate) use operator::{Fock, HamiltonianOverlap, Overlap};
pub(crate) use orthogonal::OrthogonalHamiltonianScratch;
#[cfg(feature = "gnocc")]
pub(crate) use rdm::{RDM1, RDM2, RDM3, RDM4};
pub(crate) use types::{DetPair, FockData};

// Crate-visible function re-exports.
pub(crate) use cache::build_fock_mo_cache;
pub(crate) use naive::{build_s_pair, build_s_pair_floored, occ_coeffs, pair_density};
pub(crate) use nonorthogonalwicks::update_wicks_fock;
pub(crate) use orthogonal::{
    calculate_f_pair_orthogonal, calculate_h_pairs_orthogonal_batched, calculate_s_pair_orthogonal,
};
pub(crate) use pairs::{calculate_pairs, calculate_pairs_dense, compare_pairs_wicks_naive};
pub(crate) use rdm::noci_density;
#[cfg(feature = "gnocc")]
pub(crate) use rdm::{rdm1, rdm2, rdm3, rdm4};
