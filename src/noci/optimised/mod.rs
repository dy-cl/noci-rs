// noci/optimised/mod.rs
//! Orbital-optimised reference NOCI by resonating Hartree-Fock (ResHF).
//!
//! ResHF minimises the state-averaged NOCI energy
//!
//! `E_{\text{SA}} = \sum_I w_I E_I`, with `H c_I = S c_I E_I`,
//!
//! with respect to the orbitals of every determinant and the expansion coefficients together.
//! The determinants it returns are ordinary [`crate::SCFState`]s, so the reference NOCI state
//! and every method subfolder use them unchanged. This approach depends only on the NOCI shared
//! layer and on [`crate::elements`], [`crate::determinant`], [`crate::maths`] and [`crate::scf`].
//!
//! # References
//!
//! - Miller and Parker, *J. Chem. Phys.* **162** (2025), "Numerically stable resonating
//!   Hartree-Fock", [arXiv:2411.00712](https://arxiv.org/abs/2411.00712).

mod ci;
mod pair;

// Public type re-exports.
pub use ci::ResHFExpansion;
pub use pair::{ResHFPair, ResHFPairs};

// Public function re-exports.
pub use ci::{reshf_expansion, reshf_solve};
pub use pair::{reshf_pair, reshf_pairs};
