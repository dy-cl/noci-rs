// deterministic.rs 
use ndarray::{s, Array1, Array2};
use ndarray_linalg::{Eigh, UPLO, Norm};

use crate::input::{Input, Propagator};

use crate::maths::parallel_matvec_real;

// Storage for the chosen propagator expressed in the basis of projectors.
pub struct ProjPropagator {
    pub urr: Array2<f64>,
    pub unn: Array2<f64>,
    pub unr: Array2<f64>,
    pub urn: Array2<f64>,
}

// Storage for projectors that project the coefficients to relevant and null subspaces.
pub struct Projectors {
    ur: Array2<f64>,
    ur_dag: Array2<f64>,
    un: Array2<f64>,
    un_dag: Array2<f64>,
}

// Storage for coefficients at each time-step.
pub struct Coefficients {
    pub iter: usize, 
    pub c_full: Array1<f64>,
    pub c_relevant: Array1<f64>,
    pub c_null: Array1<f64>,
}

impl Projectors {
    /// Calculate projectors onto the relevant and null subsapces of the overlap matrix S by
    /// diagonalising S as S = U \Lambda U^\dagger and paritioning the eigenvectors by an
    /// eigenvalue threshold. The null subspace is spanned by eigenvectors with \lambda < eps and
    /// the relevant subsapces by eigenvectors with \lambda > eps. The partioned eigenvector
    /// matrices U_r (relevant) and U_n (null) are used to form the projectors as:
    ///     P_r = U_r U_r^\dagger, P_n = U_n U_n^\dagger.
    /// # Arguments 
    ///     `s`: Array2, overlap matrix in full NOCI-QMC basis.
    ///     `eps`: f64, tolerance for an eigenvalue being null or relevant.
    pub fn calculate_projectors(s: &Array2<f64>, eps: f64) -> Self {
        // S = U \Lambda U^\dagger
        let (lambda, u) = s.eigh(UPLO::Lower).unwrap();

        // Split indices of eigenvalues into relevant and null spaces.
        let mut relevant = Vec::new();
        let mut null = Vec::new();
        for i in 0..lambda.len() {
            if lambda[i].abs() > eps {
                relevant.push(i);
            } else {
                null.push(i);
            }
        }

        // Construct U_{relevant} and U_{null}, i.e., the eigenvector matrices of S corresponding to
        // the relevant and null subspaces.
        let mut ur = Array2::<f64>::zeros((lambda.len(), relevant.len()));
        for (j, &icol) in relevant.iter().enumerate() {
            let col = u.slice(s![.., icol]);
            ur.slice_mut(s![.., j]).assign(&col);
        }
        let ur_dag = ur.t().to_owned();
        let mut un = Array2::<f64>::zeros((lambda.len(), null.len()));
        for (j, &icol) in null.iter().enumerate() {
            let col = u.slice(s![.., icol]);
            un.slice_mut(s![.., j]).assign(&col);
        }
        let un_dag = un.t().to_owned();
        
        println!("Projectors: eps = {:.3e}, dim(S) = {}, relevant = {}, null = {}", eps, lambda.len(), relevant.len(), null.len());
        Projectors {ur, ur_dag, un, un_dag}
    }

    /// Project a full NOCI-QMC coefficient vector c into the relevant and null subsapces of the
    /// overlap matrix S as:
    ///     c_r = P_r c = U_r U_r^\dagger, c_n = P_n c = U_n U_n^\dagger c.
    /// # Arguments
    ///     `c`: Array1, coefficient vector in the full NOCI-QMC basis.
    pub fn project(&self, c: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        // C_r = U_r U_r^\dagger C  
        let yr = parallel_matvec_real(&self.ur_dag, c);
        let c_relevant = parallel_matvec_real(&self.ur, &yr);
        // C_n = U_n U_n^\dagger C
        let yn = parallel_matvec_real(&self.un_dag, c);
        let c_null = parallel_matvec_real(&self.un, &yn);

        (c_relevant, c_null)
    }
}

impl ProjPropagator {
    /// Express a propjector in the null and relevant subspace basis by forming the matrix (U_{rr},
    /// U_{nr} \\ U_{rn} U_{nn}). All elements of the propragator can be projected by doing for
    /// example: H_{rn} = U_r^\dagger H U_n. 
    pub fn calculate_projected_propagator(h: &Array2<f64>, s: &Array2<f64>, p: &Projectors, es: f64, dt: f64, prop: &Propagator) -> Self {
        // H_{ur} = H U_r, H_{un} = H U_n, S_{ur} = S U_r, S_{un} = S U_n.
        let hur = h.dot(&p.ur); let hun = h.dot(&p.un); let sur = s.dot(&p.ur); let sun = s.dot(&p.un);

        let hrr = p.ur_dag.dot(&hur); let hnn = p.un_dag.dot(&hun); let hnr = p.un_dag.dot(&hur); let hrn = p.ur_dag.dot(&hun);
        let srr = p.ur_dag.dot(&sur); let snn = p.un_dag.dot(&sun); let snr = p.un_dag.dot(&sur); let srn = p.ur_dag.dot(&sun);
        
        let identityr = Array2::<f64>::eye(hrr.nrows());
        let identityn = Array2::<f64>::eye(hnn.nrows());

        let identityfac = match prop {
            Propagator::Unshifted => 1.0, 
            Propagator::Shifted => 1.0 + dt * es,
            // Not yet implemented. Redirect to normal shift.
            Propagator::DoublyShifted => 1.0 + dt * es,
            Propagator::DifferenceDoublyShifted => 1.0 + dt * es,
        };

        let urr = &(identityfac * &identityr) - &(dt * (&hrr - &srr.mapv(|z| es * z)));
        let unn = &(identityfac * &identityn) - &(dt * (&hnn - &snn.mapv(|z| es * z)));
        let unr = -(dt * (&hnr - &snr.mapv(|z| es * z)));
        let urn = -(dt * (&hrn - &srn.mapv(|z| es * z)));

        ProjPropagator {urr, unn, unr, urn}
    }
} 

/// Perform one deterministic update step of NOCI-QMC unshifted propagator:
/// # Arguments
///     `h`: Array2, NOCI Hamiltonian in full NOCI-QMC basis.
///     `s`: Array2, Overlap matrix in full NOCI-QMC basis.
///     `c`: Array1, NOCI-QMC coefficient vector.
///     `esc`: Scalar, Energy shift.
///     `dt`: Propagation time step.
pub fn propagate_step_unshifted(h: &Array2<f64>, s: &Array2<f64>, c: &Array1<f64>, esc: f64, dt: f64) 
                                -> Array1<f64> {
    let hc = parallel_matvec_real(h, c);
    let sc = parallel_matvec_real(s, c);
    let htildec = hc - sc.mapv(|z| esc * z);
    let dtc = htildec.mapv(|z| dt * z);
    c - &dtc 
}

/// Perform one deterministic update step of NOCI-QMC shifted propagator:
/// - \Delta\tau \sum_{\Gamma \neq \Lambda}(H_{\Lambda\Gamma} - E_s S_{\Lambda\Gamma})C^\Gamma(\tau).
/// # Arguments
///     `h`: Array2, NOCI Hamiltonian in full NOCI-QMC basis.
///     `s`: Array2, Overlap matrix in full NOCI-QMC basis.
///     `c`: Array1, NOCI-QMC coefficient vector.
///     `esc`: Scalar, Energy shift, we just use the reference NOCI energy for this here.
///     `dt`: Propagation time step.
pub fn propagate_step_shifted(h: &Array2<f64>, s: &Array2<f64>, c: &Array1<f64>, esc: f64, dt: f64) 
                              -> Array1<f64> {
    let hc = parallel_matvec_real(h, c);
    let sc = parallel_matvec_real(s, c);
    let htildec = hc - sc.mapv(|z| esc * z);
    let rhs = htildec - c.mapv(|z| esc * z);
    let dtc = rhs.mapv(|z| dt * z);
    c - &dtc
}

/// Propagate nsteps number of time-step updates or until convergence in the energy.
/// # Arguments
///     `h`: Array2, NOCI Hamiltonian in full NOCI-QMC basis. 
///     `s`: Array2, Overlap matrix in full NOCI-QMC basis.
///     `c0`: Array1, Initial NOCI-QMC coefficient vector, start from reference NOCI coefficients.
///     `es`: Scalar, Energy shift, we just use the reference HF energy for this here.
///     `max_steps`: Maximum number of time-step updates to perform.
///     `e_tol`: Energy tolerance which determines convergence.
pub fn propagate(h: &Array2<f64>, s: &Array2<f64>, c0: &Array1<f64>, mut es: f64, history: &mut Vec<Coefficients>, 
                 input: &Input) -> Option<Array1<f64>> {

    let mut c_norm = c0.clone(); 
    let mut e_prev = projected_energy(h, s, c0);
    let mut logamp: f64 = 0.0;
    let de_max = 10.0;

    // Unwrap deterministic propagation specific options
    let det = input.det.as_ref().unwrap();
    
    // If we're doing deterministic investigation into relevant and null subspaces we need to
    // calculate projectors onto these spaces which involves diagonalising S. Of course for larger
    // systems this should not be done as diagonalising S has equal cost to solving GEVP of full
    // NOCI-QMC basis.
    let mut projectors: Option<Projectors> = None;
    if input.write.write_deterministic_coeffs {
        let p = Projectors::calculate_projectors(s, 1e-14);
        let (c0_relevant, c0_null) = p.project(&c_norm);

        // Calculate diagnostics.
        println!("{}", "=".repeat(100));
        let sc0n = s.dot(&c0_null);
        let hc0n = h.dot(&c0_null);
        println!("Action of S and H on initial null vector: ||Scn|| = {}, ||Hcn|| = {}.", sc0n.norm(), hc0n.norm());
        let proj_propagator = ProjPropagator::calculate_projected_propagator(h, s, &p, es, input.prop.dt, &input.prop.propagator);
        println!("With initial shift: {}, ||Unn|| = {}, ||Urr|| = {}, ||Urn|| = {}, ||Unr|| = {}.",
                 es, &proj_propagator.unn.norm(), &proj_propagator.urr.norm(), &proj_propagator.urn.norm(), &proj_propagator.unr.norm());
        let nnull = proj_propagator.unn.nrows();
        if nnull > 0 {
            let (evals_unn, _) = proj_propagator.unn.eigh(UPLO::Lower).unwrap();
            println!("Null-space propagator Unn eigenvalues: {}", evals_unn);
        } else {
            println!("Null-space dimension is 0 (no eigenvalues of propagator Unn).");
        }
        // Add initial coefficients to the history. 
        history.push(Coefficients {iter: 0, c_full: c0.clone(), c_relevant: c0_relevant, c_null: c0_null});
        projectors = Some(p);
    }

    // Print table header.
    println!("{}", "=".repeat(100));
    println!("{:<6} {:>16} {:>16} {:>16} {:>16} {:>16}", 
            "iter", "E", "|dE|", "Shift", "||C||", "C^†SC");
    // Print initial.
    let c0_1norm = c0.iter().map(|z| z.abs()).sum::<f64>();
    let den = c0.dot(&s.dot(c0));
    println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", 
            0, e_prev, 0, es, c0_1norm, den);

    for it in 0..input.prop.max_steps {

        // Select propagator.
        let mut c_new_norm = match input.prop.propagator {
            Propagator::Shifted => propagate_step_shifted(h, s, &c_norm, es, input.prop.dt),
            Propagator::Unshifted => propagate_step_unshifted(h, s, &c_norm, es, input.prop.dt),
            // Not implemented yet. Redirect to normal shift. 
            Propagator::DoublyShifted => propagate_step_shifted(h, s, &c_norm, es, input.prop.dt),
            Propagator::DifferenceDoublyShifted => propagate_step_shifted(h, s, &c_norm, es, input.prop.dt),
        };

        // Normalise.
        let sc = s.dot(&c_new_norm);
        let norm: f64 = c_new_norm.iter().zip(sc.iter()).map(|(ci, sci)| ci * sci).sum::<f64>().sqrt();
        c_new_norm.mapv_inplace(|z| z / norm);
        // Calculate C S C^\dagger post normalisation.
        let den = c_new_norm.dot(&s.dot(&c_new_norm));
        // Calculate energy.
        let e = projected_energy(h, s, &c_new_norm);
        let de = (e - e_prev).abs();
        
        logamp += norm.ln();
        
        // Update shift dynamically if requested.
        if det.dynamic_shift {
            let a = det.dynamic_shift_alpha;  
            es = (1.0 - a) * es + a * e;
        }

        // If p exists we are doing projection into subspaces.
        if let Some(ref p) = projectors {
            let scale = logamp.exp();
            // Project coefficients into relevant and null subspaces.
            let (c_relevant, mut c_null) = p.project(&c_new_norm);
            c_null = c_null.mapv(|z| z * scale);
            // Add coefficients to the history. 
            history.push(Coefficients {iter: it + 1, c_full: c_new_norm.clone(), c_relevant, c_null});
        }

        // Print table rows.
        let c1norm = c_new_norm.iter().map(|z| z.abs()).sum::<f64>();
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", 
                it + 1, e, de, es, c1norm, den);
        // If our energy change between iterations is large we likely have problems with
        // singularity and very low eigenvalues or a time-step that is too large. 
        if de > de_max {
            println!("Energy change too large at iter {}: |dE| = {}. 
            Either time-step too large or likely converging to un-physical eigenvalues with singular S or H", it + 1, de);
            return None
        }

        if de < det.e_tol {
            return Some(c_new_norm)
        }
        c_norm = c_new_norm;
        e_prev = e;
    }
    Some(c_norm)
}

/// E(\tau) = \frac{C^\Lambda\langle\Psi_\Lambda|\hat H|\Psi_\Gamma\rangle C^\Gamma}{C^\Lambda\langle\Psi_\Lambda|\Psi_\Gamma\rangle C^\Gamma} 
/// = \frac{C^\Lambda H_{\Lambda\Gamma}C^{\Gamma} }{C^\Lambda S_{\Lambda\Gamma}C^\Gamma}.
/// # Arguments
///     `h`: Array2, NOCI Hamiltonian in full NOCI-QMC basis. Shifted by E_s * S.
///     `s`: Array2, Overlap matrix in full NOCI-QMC basis.
///     `c`: Array1, NOCI-QMC coefficient vector.
pub fn projected_energy(h: &Array2<f64>, s: &Array2<f64>, c: &Array1<f64>) -> f64 {
    let hc = h.dot(c);
    let num = c.iter().zip(hc.iter()).map(|(ci, hci)| ci * hci).sum::<f64>();
    let sc = s.dot(c);
    let den = c.iter().zip(sc.iter()).map(|(ci, sci)| ci * sci).sum::<f64>();
    num / den 
}

