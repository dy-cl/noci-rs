// scf/mod.rs
//! Real and holomorphic Hartree-Fock optimisation.
//!
//! This module implements the numerical SCF procedures used to obtain the independently
//! optimised orbital sets forming the reference NOCI basis. Conventional calculations support
//! restricted and unrestricted Fock construction, density and energy evaluation, orbital
//! occupations, DIIS acceleration and convergence testing.
//!
//! Holomorphic SCF calculations replace the Hermitian density with the complex-analytic
//! density and optimise the resulting holomorphic energy with respect to complex orbital
//! rotations. The module provides construction and normalisation of holomorphic states,
//! orbital gradients, seeded holomorphic orbitals and lookup structures used to follow
//! corresponding states.
//!
//! This module performs the numerical orbital optimisation. State recipes, biased initial
//! guesses, metadynamics searches, duplicate detection and assembly of the resulting states
//! into a reference NOCI basis are handled by the basis and driver modules.

mod bias;
mod cycle;
mod diis;
mod h;
mod kernels;
mod occupation;
mod print;
mod select;

// Public type re-exports.
pub use h::{HSCFGenerationLookups, StateLookups};
pub use kernels::DensityMode;
pub use occupation::SpinOccupation;

// Public function re-exports.
pub use cycle::scf_cycle;
pub use h::{build_hscf_state, h_seed_orbitals, hscf_cycle, normalise_hermitian};
pub use kernels::{density, energy, fock, orbital_energies, orbital_gradient};
pub use occupation::{occ_first, spin_occupation};
