// snoci/types.rs

use ndarray::{Array1, Array2};

use crate::SCFState;

/// Storage for the result of a selected NOCI step.
pub struct SNOCIState {
    /// Current selected-space NOCI energy.
    pub ecurrent: f64,
    /// Current selected-space ground-state eigenvector.
    pub coeffs: Array1<f64>,
    /// Hamiltonian matrix in the current selected space.
    pub hcurrent: Array2<f64>,
    /// Overlap matrix in the current selected space.
    pub scurrent: Array2<f64>,
    /// Candidate determinants considered on the current iteration.
    pub candidates: Vec<SCFState>,
    /// Candidate determinants selected for addition to the current space.
    pub selected: Vec<SCFState>,
    /// Importance scores for the current candidate determinants.
    pub candidate_scores: Vec<f64>,
    /// Second-order Epstein-Nesbet-like correction from the current candidate space.
    pub ept2: f64,
}

/// Result of a GMRES linear solve.
pub(in crate::snoci) struct GmresResult {
    /// Approximate solution vector.
    pub(in crate::snoci) x: Array1<f64>,
    /// Root-mean-square residual norm.
    pub(in crate::snoci) residual_rms: f64,
    /// Number of GMRES iterations performed.
    pub(in crate::snoci) iterations: usize,
    /// Whether the residual reached the requested tolerance.
    pub(in crate::snoci) converged: bool,
}

/// Candidate-space quantities after projection out of the current selected space.
pub(in crate::snoci) struct ProjectedCandidateSpaceElems {
    /// Current pool candidates.
    pub(in crate::snoci) candidates: Vec<SCFState>,
    /// Candidate-candidate overlap projected into the complement of the selected space.
    /// This is given by S_{ab}^{\Omega} = S_{ab} - S_{ai} S^{ij} S_{jb}.
    pub(in crate::snoci) s_ab_omega: Array2<f64>,
    /// Non-projected candidate-current overlap for the current pool.
    pub(in crate::snoci) s_ai: Array2<f64>,
    /// Transpose of `s_ai`.
    pub(in crate::snoci) s_ia: Array2<f64>,
}

/// Fock matrix elements between current and candidate spaces.
pub(in crate::snoci) struct FockMatrixElems {
    /// Current-current space Fock matrix.
    pub(in crate::snoci) f_ii: Array2<f64>,
    /// Candidate-current space Fock matrix.
    pub(in crate::snoci) f_ai: Array2<f64>,
    /// Current-candidate space Fock matrix.
    pub(in crate::snoci) f_ia: Array2<f64>,
    /// Candidate-candidate space Fock matrix.
    pub(in crate::snoci) f_ab: Array2<f64>,
}

