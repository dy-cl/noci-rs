// noci/mod.rs
//! Core determinant representation and matrix-element layer for NOCI.
//!
//! This module defines the data, determinant-pair representation and molecular-orbital caches
//! used by all NOCI-based methods. It evaluates overlap, Hamiltonian, generalised-Fock and
//! related transition quantities between reference or excited determinants and constructs full
//! matrices when required.
//!
//! Matrix-element evaluation is selected according to the determinant pair:
//!
//! - Determinants generated from a common orthonormal parent use orthogonal
//!   Slater-Condon shortcuts;
//! - Nonorthogonal pairs use the extended nonorthogonal Wick implementation when enabled;
//! - The generalised Slater-Condon implementation provides the direct alternative.
//!
//! The module centralises determinant-pair ordering, excitation phases and matrix-element
//! cache access so that reference NOCI, deterministic propagation, NOCIQMC and
//! NOCI-PT2/SNOCI use consistent matrix elements.
//!
//! Full Hamiltonian, overlap and generalised-Fock matrices may be constructed for
//! deterministic calculations. The resulting generalised eigenvalue problem
//!
//! `\mathbf H\mathbf c = E\mathbf S\mathbf c`
//!
//! is then solved.

mod cache;
mod factorise;
mod fock;
mod hs;
mod m;
mod matrix;
mod naive;
mod overlap;
mod types;
mod wicks;

// Public type re-exports.
pub use types::{FockMOCache, MOCache, NOCIData, NOCIScalar};

// Public function re-exports.
pub use cache::{build_fock_mo_cache, build_mo_cache};
pub use matrix::{build_noci_hs, build_noci_s, calculate_noci_energy};
pub use naive::noci_density;
pub use wicks::{build_wicks_shared, update_wicks_fock};

// Crate-visible type re-exports.
pub(crate) use factorise::{
    OneBodyFactorisation, OneBodyScratch, OverlapScratch, SpinFactorisation,
};
pub(crate) use types::{DetPair, FockData};

// Crate-visible function re-exports.
pub(crate) use fock::calculate_f_pair;
pub(crate) use hs::calculate_hs_pair;
pub(crate) use m::calculate_m_pair;
pub(crate) use matrix::build_noci_fock;
pub(crate) use naive::occ_coeffs;
#[cfg(feature = "nocc")]
pub(crate) use naive::{build_s_pair, pair_density};
pub(crate) use overlap::calculate_s_pair;
