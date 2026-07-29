// snoci/mod.rs
//! Selected NOCI and second-order perturbative corrections to NOCI.
//!
//! This module iteratively enlarges a variational NOCI space using excited determinants
//! generated from the configured reference orbital sets. At each iteration it:
//!
//! 1. Solves the current-space generalised eigenvalue problem;
//! 2. Constructs or updates the external candidate pool;
//! 3. Builds current-candidate overlap, Hamiltonian and generalised-Fock quantities;
//! 4. Projects the candidates against the current NOCI ground state;
//! 5. Solves the projected NOCI-PT2 amplitude equation;
//! 6. Assigns perturbative scores and selects candidates for the next variational space.
//!
//! In the projected candidate space, the amplitudes satisfy
//!
//! \mathbf M^{(k)}(\epsilon)\mathbf a^{(k)}(\epsilon) = -\mathbf V^{(k)},
//!
//! where \mathbf M^{(k)} contains the projected shifted generalised-Fock operator and metric,
//! \mathbf V^{(k)} couples the current NOCI state to the candidate space, and \epsilon is an
//! optional imaginary shift.
//!
//! The projected linear system is solved using restarted GMRES. Diagonal and low-rank
//! Woodbury preconditioners are available, and candidate-space matrix-vector products may be
//! evaluated from a stored packed matrix or on demand. MPI distributes candidate-space
//! operator applications while Rayon parallelises matrix-element evaluation within each rank.
//!
//! Candidate importance is obtained from the NOCI-PT2 amplitudes and couplings. Determinants
//! above the configured threshold are added to the selected space until the perturbative
//! correction, selection, iteration or dimension stopping condition is reached.

mod candidate;
mod gmres;
mod operators;
mod step;
mod types;

// Public type re-exports.
pub use types::{SNOCIPT2Result, SNOCIState};

// Public function re-exports.
pub use step::snoci_step;

// Restricted type re-exports.
pub(in crate::snoci) use candidate::CandidatePool;
pub(in crate::snoci) use types::{
    ArnoldiCycle, ArnoldiParams, GMRESResult, PT2ProjectedOperator, PT2Projection, Preconditioner,
    SNOCIFocks, SNOCIOverlaps,
};

// Restricted function re-exports.
pub(in crate::snoci) use gmres::gmres;
pub(in crate::snoci) use operators::{
    apply_shifted_omega_m, apply_shifted_omega_m_mpi, build_candidate_current_h,
    build_candidate_m, build_candidate_m_diag, build_candidate_m_disk, build_candidate_s_diag,
    build_candidate_v, build_omega_v, build_preconditioner, build_snoci_focks,
    build_snoci_overlaps, build_snoci_projection, select_candidates, solve_current_space,
};
