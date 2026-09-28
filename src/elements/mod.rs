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
//! The operator files at this level dispatch each pair to its engine; `rdm` assembles the
//! reduced density matrices of NOCI states from the pair densities. Both method families,
//! `noci` and `nocc`, evaluate their matrix elements here.

pub mod nonorthogonalwicks;

mod cache;
mod fock;
mod hs;
mod m;
mod naive;
mod orthogonal;
mod overlap;
mod rdm;
mod types;

// Public type re-exports.
pub use types::{FockMOCache, MOCache, NOCIData};

// Public function re-exports.
pub use cache::build_mo_cache;
pub use nonorthogonalwicks::build_wicks_shared;

// Crate-visible type re-exports.
pub(crate) use orthogonal::OrthogonalHamiltonianScratch;
#[cfg(feature = "nocc")]
pub(crate) use rdm::{RDM1, RDM2, RDM3, RDM4};
pub(crate) use types::{DetPair, FockData};

// Crate-visible function re-exports.
pub(crate) use cache::build_fock_mo_cache;
pub(crate) use fock::{calculate_f_pair, compare_f_pair_wicks_naive};
pub(crate) use hs::{
    calculate_hs_pair, calculate_hs_pairs_wicks_batched, compare_hs_pair_wicks_naive,
};
pub(crate) use m::calculate_m_pair;
pub(crate) use naive::{build_s_pair, calculate_s_pair_naive, occ_coeffs, pair_density};
pub(crate) use nonorthogonalwicks::update_wicks_fock;
pub(crate) use orthogonal::{
    calculate_f_pair_orthogonal, calculate_h_pairs_orthogonal_batched, calculate_s_pair_orthogonal,
};
pub(crate) use overlap::calculate_s_pair;
pub(crate) use rdm::noci_density;
#[cfg(feature = "nocc")]
pub(crate) use rdm::{rdm1, rdm2, rdm3, rdm4};
