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
//! layer and on [`crate::elements`], [`crate::input`], [`crate::maths`] and [`crate::scf`].
//!
//! # References
//!
//! - Miller and Parker, *J. Chem. Phys.* **162**, 104115 (2025),
//!   [doi:10.1063/5.0246790](https://doi.org/10.1063/5.0246790).

mod ci;
mod fock;
mod optimise;
mod pair;

// Public function re-exports.
pub use optimise::reshf_optimise;
