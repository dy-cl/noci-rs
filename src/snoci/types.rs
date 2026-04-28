// snoci/types.rs
use std::time::Instant;

use ndarray::{Array1, Array2};

use crate::SCFState;
use crate::noci::{FockData, NOCIData};

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
pub(in crate::snoci) struct GMRES {
    /// Approximate solution vector.
    pub(in crate::snoci) x: Array1<f64>,
    /// Root-mean-square residual norm.
    pub(in crate::snoci) residual_rms: f64,
    /// Number of GMRES iterations performed.
    pub(in crate::snoci) iterations: usize,
    /// Whether the residual reached the requested tolerance.
    pub(in crate::snoci) converged: bool,
}

/// Candidate-space overlap blocks required for projection out of the current selected space.
pub(in crate::snoci) struct SNOCIOverlaps {
    /// Candidate-current overlap, S_ai.
    pub(in crate::snoci) s_ai: Array2<f64>,
    /// Current-candidate overlap, S_ia.
    pub(in crate::snoci) s_ia: Array2<f64>,
}

/// Fock matrix blocks required to build the SNOCI response problem.
pub(in crate::snoci) struct SNOCIFocks {
    /// Current-current Fock matrix, F_ij.
    pub(in crate::snoci) f_ii: Array2<f64>,
    /// Candidate-current Fock matrix, F_ai.
    pub(in crate::snoci) f_ai: Array2<f64>,
    /// Current-candidate Fock matrix, F_ia.
    pub(in crate::snoci) f_ia: Array2<f64>,
}

/// Projector quantities required to remove the current selected-space NOCI state
/// from the NOCI-PT2 first-order interacting space.
pub(in crate::snoci) struct PT2Projection {
    /// Zeroth-order NOCI generalised-Fock energy.
    pub(in crate::snoci) e0: f64,
    /// Candidate-reference overlap, S_a0.
    pub(in crate::snoci) s_a0: Array1<f64>,
    /// Reference-candidate overlap, S_0a.
    pub(in crate::snoci) s_0a: Array1<f64>,
    /// Candidate-reference Fock contraction, F_a0.
    pub(in crate::snoci) f_a0: Array1<f64>,
    /// Reference-candidate Fock contraction, F_0a.
    pub(in crate::snoci) f_0a: Array1<f64>,
}

/// Matrix-free projected NOCI-PT2 operator.
pub(in crate::snoci) struct PT2ProjectedOperator<'a, 'data, 'fock> {
    /// Shared NOCI matrix-element data.
    pub(in crate::snoci) data: &'a NOCIData<'data>,
    /// Fock-specific matrix-element data.
    pub(in crate::snoci) fock: &'a FockData<'fock>,
    /// Candidate determinants defining the first-order interacting space.
    pub(in crate::snoci) candidates: &'a [SCFState],
    /// Precomputed projection quantities.
    pub(in crate::snoci) projection: &'a PT2Projection,
}

/// Storage for a single restarted Arnoldi cycle.
pub(in crate::snoci) struct ArnoldiCycle {
    /// Orthonormal Krylov vectors.
    pub(in crate::snoci) q: Vec<Array1<f64>>,
    /// Upper Hessenberg matrix after Givens rotations.
    pub(in crate::snoci) h: Array2<f64>,
    /// Rotated residual right-hand side.
    pub(in crate::snoci) g: Array1<f64>,
    /// Number of Arnoldi iterations completed in the current cycle.
    pub(in crate::snoci) kfinal: usize,
}

/// Parameters for a single restarted Arnoldi cycle.
pub(in crate::snoci) struct ArnoldiParams<'a> {
    /// Maximum number of Arnoldi iterations in this restart cycle.
    pub(in crate::snoci) inner_max: usize,
    /// GMRES restart cycle index.
    pub(in crate::snoci) restart_id: usize,
    /// Total number of GMRES iterations before this cycle.
    pub(in crate::snoci) total_iter: usize,
    /// Square-root of the vector length.
    pub(in crate::snoci) rms: f64,
    /// Wall-time for GMRES.
    pub(in crate::snoci) gmres_start: &'a Instant,
}

/// Rank-2 Woodbury preconditioner for the projected NOCI-PT2 shifted Fock matrix.
pub(in crate::snoci) struct Preconditioner {
    /// Inverse diagonal of the unprojected candidate-candidate matrix.
    dinv: Array1<f64>,
    /// First diagonal-scaled left update vector `D^{-1} u_0`.
    z0: Array1<f64>,
    /// Second diagonal-scaled left update vector `D^{-1} u_1`.
    z1: Array1<f64>,
    /// First right update vector.
    v0: Array1<f64>,
    /// Second right update vector.
    v1: Array1<f64>,
    /// `(0,0)` element of the inverse Woodbury core.
    w00: f64,
    /// `(0,1)` element of the inverse Woodbury core.
    w01: f64,
    /// `(1,0)` element of the inverse Woodbury core.
    w10: f64,
    /// `(1,1)` element of the inverse Woodbury core.
    w11: f64,
    /// Whether the rank-2 Woodbury correction is numerically safe to apply.
    active: bool,
}

impl Preconditioner {
    /// Build a rank-2 Woodbury preconditioner from an unprojected diagonal and projection contractions.
    /// # Arguments:
    /// - `m_diag`: Diagonal of the unprojected candidate-candidate matrix `M`.
    /// - `p`: Projection contractions used to form `M^Omega`.
    /// # Returns:
    /// - `OmegaRank2Preconditioner`: Rank-2 preconditioner.
    pub (in crate::snoci) fn new(m_diag: &Array1<f64>, p: &PT2Projection) -> Self {
        let dmax = m_diag.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
        let dfloor = (1e-12_f64 * dmax).max(1e-14_f64);

        let dinv = Array1::from_iter(m_diag.iter().map(|&x| if x.abs() > dfloor {1.0 / x} else {1.0}));

        let u0 = Array1::from_iter(p.f_a0.iter().zip(p.s_a0.iter()).map(|(&f, &s)| -f + 2.0 * p.e0 * s));
        let u1 = p.s_a0.mapv(|s| -s);
        let v0 = p.s_0a.clone();
        let v1 = p.f_0a.clone();

        let z0 = Array1::from_iter(dinv.iter().zip(u0.iter()).map(|(&d, &u)| d * u));
        let z1 = Array1::from_iter(dinv.iter().zip(u1.iter()).map(|(&d, &u)| d * u));

        let c00 = 1.0 + v0.dot(&z0);
        let c01 = v0.dot(&z1);
        let c10 = v1.dot(&z0);
        let c11 = 1.0 + v1.dot(&z1);

        let det = c00 * c11 - c01 * c10;
        let active = det.abs() > 1e-14_f64;

        let (w00, w01, w10, w11) = if active {
            (c11 / det, -c01 / det, -c10 / det, c00 / det)
        } else {
            (1.0, 0.0, 0.0, 1.0)
        };

        Preconditioner {dinv, z0, z1, v0, v1, w00, w01, w10, w11, active}
    }

    /// Apply the rank-2 Woodbury preconditioner to a vector.
    /// # Arguments:
    /// - `v`: Vector to precondition.
    /// # Returns:
    /// - `Array1<f64>`: Approximate action of `(M^Omega)^{-1} v`.
    pub(in crate::snoci) fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut y = Array1::from_iter(v.iter().zip(self.dinv.iter()).map(|(&vi, &di)| vi * di));

        if !self.active {
            return y;
        }

        let t0 = self.v0.dot(&y);
        let t1 = self.v1.dot(&y);

        let c0 = self.w00 * t0 + self.w01 * t1;
        let c1 = self.w10 * t0 + self.w11 * t1;

        for i in 0..y.len() {
            y[i] -= self.z0[i] * c0 + self.z1[i] * c1;
        }
        y
    }
}
