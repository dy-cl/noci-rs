// elements/naive/mod.rs
//! Generalised Slater–Condon matrix elements for nonorthogonal determinants.
//!
//! # References
//!
//! - Burton, *J. Chem. Phys.* **154**, 144109 (2021),
//!   [doi:10.1063/5.0045442](https://doi.org/10.1063/5.0045442).

mod pairs;
#[cfg(feature = "nocc")]
mod rdm;
mod rules;

// Crate-visible function re-exports.
pub(crate) use pairs::calculate_s_pair_naive;
pub(crate) use rules::{build_s_pair, occ_coeffs, pair_density};

// Restricted function re-exports.
pub(in crate::elements) use pairs::{
    calculate_f_pair_naive, calculate_hs_pair_naive, calculate_m_pair_naive,
};
#[cfg(feature = "nocc")]
pub(in crate::elements) use rdm::{
    rdm1_pair_naive, rdm2_pair_naive, rdm3_pair_naive, rdm4_pair_naive,
};
