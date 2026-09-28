// noci/mod.rs
//! NOCI family: full NOCI matrices, the reference NOCI state and factorised operators.
//!
//! This shared layer builds the full overlap, Hamiltonian and generalised-Fock matrices of a
//! determinant space from the pair matrix elements of [`crate::elements`], solves the
//! generalised eigenvalue problem
//!
//! `\mathbf H\mathbf c = E\mathbf S\mathbf c`
//!
//! for the reference NOCI state, and provides spin-factorised overlap and one-body tables. The
//! method subfolders below it use these consistently: deterministic propagation in
//! [`deterministic`], NOCIQMC in [`stochastic`], and selected NOCI with NOCI-PT2 in
//! [`selected`]. Each method subfolder depends only on this shared layer and the modules below
//! it, never on a sibling method subfolder.
//!
//! # References
//!
//! - NOCI reference-state construction: Thom and Head-Gordon, *J. Chem. Phys.* **131**, 124113
//!   (2009), [doi:10.1063/1.3236841](https://doi.org/10.1063/1.3236841); Burton and Thom,
//!   *J. Chem. Theory Comput.* **15**, 4851 (2019),
//!   [doi:10.1021/acs.jctc.9b00441](https://doi.org/10.1021/acs.jctc.9b00441).

pub mod deterministic;
pub mod selected;
pub mod stochastic;

mod factorise;
mod matrix;

// Public function re-exports.
pub use matrix::{build_noci_hs, calculate_noci_energy};

// Crate-visible type re-exports.
pub(crate) use factorise::{
    OneBodyFactorisation, OneBodyScratch, OverlapFactors, OverlapScratch, SpinFactorisation,
};

// Crate-visible function re-exports.
pub(crate) use matrix::{build_noci_fock, build_noci_s};
