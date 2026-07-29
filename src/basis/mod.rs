// basis/mod.rs
//! Construction of reference and excited determinant bases.
//!
//! This module converts user-defined state-generation recipes and converged SCF solutions
//! into the determinant bases used by NOCI and its post-reference methods. Reference states
//! may be obtained from maximum-overlap recipes or SCF metadynamics using spin-density,
//! spatial-density biases.
//!
//! Candidate SCF solutions are compared using density and electron-distance measures so that
//! coalesced or duplicate states can be identified before entering the reference NOCI basis.
//! The selected states retain their orbital coefficients, occupations, labels and reference
//! ordering.
//!
//! Excited determinant bases are generated from the molecular-orbital set of
//! each selected reference. Their excitation descriptors, parent-reference indices and
//! fermionic phases are retained for subsequent matrix-element evaluation.

mod atoms;
mod bias;
mod duplicate;
mod excitation;
mod generate;
mod metadynamics;
mod mom;
mod normalise;
mod types;

// Public type re-exports.
pub use types::ReferenceBasis;

// Public function re-exports.
pub use duplicate::{density_distance, electron_distance};
pub use excitation::{excitation_phase, generate_excited_basis};
pub use generate::generate_reference_noci_basis;
pub use normalise::hermitian_hnoci_basis;
