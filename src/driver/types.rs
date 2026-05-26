// driver/types.rs

use num_complex::Complex64;

use crate::driver::post::PostReferenceResults;
use crate::driver::reference::ReferenceRun;
use crate::timers;
use crate::{HSCFState, SCFState};

/// Atom specification for one geometry.
pub type Atoms = Vec<String>;

/// Printed results for a given geometry.
pub struct GeometryResults {
    /// Geometry parameter for this calculation.
    pub r: f64,
    /// SCF states generated for this geometry.
    pub states: Vec<SCFState>,
    /// Complex h-SCF states generated for this geometry.
    pub hstates: Vec<HSCFState>,
    /// RHF energy at this geometry.
    pub e_rhf: f64,
    /// Reference-space NOCI energy at this geometry.
    pub e_noci_ref: f64,
    /// Deterministic NOCI-QMC energy if calculated.
    pub e_noci_qmc_det: Option<f64>,
    /// Stochastic NOCI-QMC energy if calculated.
    pub e_noci_qmc_stoch: Option<f64>,
    /// Selected NOCI energy if calculated.
    pub e_snoci: Option<f64>,
    /// NOCI-PT2 energy if calculated.
    pub e_pt2: Option<Vec<(f64, f64)>>,
    /// FCI energy if available.
    pub e_fci: Option<f64>,
    /// Number of MPI ranks for printing.
    pub nranks: usize,
    /// Total timings associated with this geometry.
    pub timings: timers::Totals,
}

impl GeometryResults {
    /// Construct results from a real-reference run.
    /// # Arguments:
    /// - `r`: Current geometry.
    /// - `states`: Real SCF states generated for this geometry.
    /// - `reference`: Finished reference-space calculation.
    /// - `post`: Optional post-reference calculation results.
    /// - `e_fci`: Optional FCI energy.
    /// - `nranks`: Number of MPI ranks for printing.
    /// - `timings`: Total timings associated with this geometry.
    /// # Returns:
    /// - `GeometryResults`: Printable geometry results.
    pub fn from_real(
        r: f64,
        states: Vec<SCFState>,
        reference: ReferenceRun<f64>,
        post: PostReferenceResults,
        e_fci: Option<f64>,
        nranks: usize,
        timings: timers::Totals,
    ) -> Self {
        let e_rhf = states[0].e;
        Self {
            r,
            states,
            hstates: Vec::new(),
            e_rhf,
            e_noci_ref: reference.e_noci,
            e_noci_qmc_det: post.e_noci_qmc_det,
            e_noci_qmc_stoch: post.e_noci_qmc_stoch,
            e_snoci: post.e_snoci,
            e_pt2: post.e_pt2,
            e_fci,
            nranks,
            timings,
        }
    }

    /// Construct results from a holomorphic-reference run.
    /// # Arguments:
    /// - `r`: Current geometry.
    /// - `state_sets`: Real SCF states and complex h-SCF states generated for this geometry.
    /// - `reference`: Finished reference-space calculation.
    /// - `post`: Optional post-reference calculation results.
    /// - `e_fci`: Optional FCI energy.
    /// - `nranks`: Number of MPI ranks for printing.
    /// - `timings`: Total timings associated with this geometry.
    /// # Returns:
    /// - `GeometryResults`: Printable geometry results.
    pub fn from_holomorphic(
        r: f64,
        state_sets: (Vec<SCFState>, Vec<HSCFState>),
        reference: ReferenceRun<Complex64>,
        post: PostReferenceResults,
        e_fci: Option<f64>,
        nranks: usize,
        timings: timers::Totals,
    ) -> Self {
        let (states, hstates) = state_sets;
        let e_rhf = states.first().map(|st| st.e).unwrap_or(0.0);
        Self {
            r,
            states,
            hstates,
            e_rhf,
            e_noci_ref: reference.e_noci,
            e_noci_qmc_det: post.e_noci_qmc_det,
            e_noci_qmc_stoch: post.e_noci_qmc_stoch,
            e_snoci: post.e_snoci,
            e_pt2: post.e_pt2,
            e_fci,
            nranks,
            timings,
        }
    }
}
