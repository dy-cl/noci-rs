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
//!
//! # References
//!
//! - NOCI reference-state construction: Thom and Head-Gordon, *J. Chem. Phys.* **131**, 124113
//!   (2009), [doi:10.1063/1.3236841](https://doi.org/10.1063/1.3236841); Burton and Thom,
//!   *J. Chem. Theory Comput.* **15**, 4851 (2019),
//!   [doi:10.1021/acs.jctc.9b00441](https://doi.org/10.1021/acs.jctc.9b00441).

mod auxiliary;
mod cache;
mod factorise;
mod fock;
mod hs;
mod m;
mod matrix;
mod naive;
mod orthogonal;
mod overlap;
mod space;
mod types;
mod wicks;

// Public type re-exports.
pub use crate::determinant::ParentDeterminant;
pub use space::{NOCIDeterminantState, NOCIIndex, NOCISpace};
pub use types::{FockMOCache, MOCache, NOCIData, NOCIScalar};

// Public function re-exports.
pub use cache::build_mo_cache;
pub use matrix::{build_noci_hs, calculate_noci_energy};
pub use wicks::build_wicks_shared;

// Crate-visible type re-exports.
pub(crate) use auxiliary::{
    AuxiliaryDeterminantState, AuxiliaryIndex, AuxiliarySpace, AuxiliarySpinIndex,
};
pub(crate) use factorise::{
    OneBodyFactorisation, OneBodyScratch, OverlapFactors, OverlapScratch, SpinFactorisation,
};
pub(crate) use orthogonal::OrthogonalConnection;
pub(crate) use space::{NOCISpinIndex, ReducedOneSpinNOCIDeterminantState};
pub(crate) use types::{DetPair, FockData};

// Crate-visible function re-exports.
pub(crate) use cache::build_fock_mo_cache;
pub(crate) use fock::calculate_f_pair;
pub(crate) use hs::{
    OrthogonalHamiltonianScratch, calculate_h_pairs_orthogonal_batched, calculate_hs_pair,
    calculate_hs_pairs_wicks_batched,
};
pub(crate) use m::calculate_m_pair;
pub(crate) use matrix::{build_noci_fock, build_noci_s};
#[cfg(feature = "nocc")]
pub(crate) use naive::{build_s_pair, pair_density};
pub(crate) use naive::{noci_density, occ_coeffs};
pub(crate) use overlap::calculate_s_pair;
pub(crate) use wicks::update_wicks_fock;
