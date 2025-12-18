// deterministic.rs 
use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use ndarray_linalg::{Eigh, UPLO, Norm};

use crate::maths::parallel_matvec;
use crate::input::{Input, Propagator};

// Storage for the chosen propagator expressed in the basis of projectors.
pub struct ProjPropagator {
    pub urr: Array2<Complex64>,
    pub unn: Array2<Complex64>,
    pub unr: Array2<Complex64>,
    pub urn: Array2<Complex64>,
}

// Storage for projectors that project the coefficients to relevant and null subspaces.
pub struct Projectors {
    ur: Array2<Complex64>,
    ur_dag: Array2<Complex64>,
    un: Array2<Complex64>,
    un_dag: Array2<Complex64>,
}

// Storage for coefficients at each time-step.
pub struct Coefficients {
    pub iter: usize, 
    pub c_full: Array1<Complex64>,
    pub c_relevant: Array1<Complex64>,
    pub c_null: Array1<Complex64>,
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
    pub fn calculate_projectors(s: &Array2<Complex64>, eps: f64) -> Self {
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
        let mut ur = Array2::<Complex64>::zeros((lambda.len(), relevant.len()));
        for (j, &icol) in relevant.iter().enumerate() {
            let col = u.slice(s![.., icol]);
            ur.slice_mut(s![.., j]).assign(&col);
        }
        let ur_dag = ur.map(|z| z.conj()).t().to_owned();
        let mut un = Array2::<Complex64>::zeros((lambda.len(), null.len()));
        for (j, &icol) in null.iter().enumerate() {
            let col = u.slice(s![.., icol]);
            un.slice_mut(s![.., j]).assign(&col);
        }
        let un_dag = un.map(|z| z.conj()).t().to_owned();
        
        println!("Projectors: eps = {:.3e}, dim(S) = {}, relevant = {}, null = {}", eps, lambda.len(), relevant.len(), null.len());
        Projectors {ur, ur_dag, un, un_dag}
    }

    /// Project a full NOCI-QMC coefficient vector c into the relevant and null subsapces of the
    /// overlap matrix S as:
    ///     c_r = P_r c = U_r U_r^\dagger, c_n = P_n c = U_n U_n^\dagger c.
    /// # Arguments
    ///     `c`: Array1, coefficient vector in the full NOCI-QMC basis.
    pub fn project(&self, c: &Array1<Complex64>) -> (Array1<Complex64>, Array1<Complex64>) {
        // C_r = U_r U_r^\dagger C  
        let yr = parallel_matvec(&self.ur_dag, c);
        let c_relevant = parallel_matvec(&self.ur, &yr);
        // C_n = U_n U_n^\dagger C
        let yn = parallel_matvec(&self.un_dag, c);
        let c_null = parallel_matvec(&self.un, &yn);

        (c_relevant, c_null)
    }
}

impl ProjPropagator {
    pub fn calculate_projected_propagator(h: &Array2<Complex64>, s: &Array2<Complex64>, p: &Projectors, es: f64, dt: f64, prop: &Propagator) -> Self {
        let esc = Complex64::new(es, 0.0);
        let dtc = Complex64::new(dt, 0.0); 
        
        // H_{ur} = H U_r, H_{un} = H U_n, S_{ur} = S U_r, S_{un} = S U_n.
        let hur = h.dot(&p.ur); let hun = h.dot(&p.un); let sur = s.dot(&p.ur); let sun = s.dot(&p.un);

        let hrr = p.ur_dag.dot(&hur); let hnn = p.un_dag.dot(&hun); let hnr = p.un_dag.dot(&hur); let hrn = p.ur_dag.dot(&hun);
        let srr = p.ur_dag.dot(&sur); let snn = p.un_dag.dot(&sun); let snr = p.un_dag.dot(&sur); let srn = p.ur_dag.dot(&sun);
        
        let identityr = Array2::<Complex64>::eye(hrr.nrows());
        let identityn = Array2::<Complex64>::eye(hnn.nrows());

        let identityfac = match prop {Propagator::Unshifted => Complex64::new(1.0, 0.0), Propagator::Shifted   => Complex64::new(1.0 + dt * es, 0.0)};

        let urr = &(identityfac * &identityr) - &(dtc * (&hrr - &srr.mapv(|z| esc * z)));
        let unn = &(identityfac * &identityn) - &(dtc * (&hnn - &snn.mapv(|z| esc * z)));
        let unr = -(dtc * (&hnr - &snr.mapv(|z| esc * z)));
        let urn = -(dtc * (&hrn - &srn.mapv(|z| esc * z)));

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
pub fn propagate_step_unshifted(h: &Array2<Complex64>, s: &Array2<Complex64>, c: &Array1<Complex64>, esc: Complex64, dt: f64) 
                                -> Array1<Complex64> {
    let hc = parallel_matvec(h, c);
    let sc = parallel_matvec(s, c);
    let htildec = hc - sc.mapv(|z| esc * z);
    let dtc = htildec.mapv(|z| Complex64::new(dt, 0.0) * z);
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
pub fn propagate_step_shifted(h: &Array2<Complex64>, s: &Array2<Complex64>, c: &Array1<Complex64>, esc: Complex64, dt: f64) 
                              -> Array1<Complex64> {
    let hc = parallel_matvec(h, c);
    let sc = parallel_matvec(s, c);
    let htildec = hc - sc.mapv(|z| esc * z);
    let rhs = htildec - c.mapv(|z| esc * z);
    let dtc = rhs.mapv(|z| Complex64::new(dt, 0.0) * z);
    c - &dtc
}

/// Propagate nsteps number of time-step updates.
/// # Arguments
///     `h`: Array2, NOCI Hamiltonian in full NOCI-QMC basis. 
///     `s`: Array2, Overlap matrix in full NOCI-QMC basis.
///     `c0`: Array1, Initial NOCI-QMC coefficient vector, start from reference NOCI coefficients.
///     `es`: Scalar, Energy shift, we just use the reference HF energy for this here.
///     `max_steps`: Maximum number of time-step updates to perform.
///     `e_tol`: Energy tolerance which determines convergence.
pub fn propagate(h: &Array2<Complex64>, s: &Array2<Complex64>, c0: &Array1<Complex64>, mut es: f64, history: &mut Vec<Coefficients>, 
                 input: &Input) -> Option<Array1<Complex64>> {

    let mut c_norm = c0.clone(); 
    let mut e_prev = projected_energy(h, s, c0);
    
    let mut logamp: f64 = 0.0;

    let de_max = 10.0;
    
    // If we're doing deterministic investigation into relevant and null subspaces we need to
    // calculate projectors onto these spaces which involves diagonalising S. Of course for larger
    // systems this should not be done as diagonalising S has equal cost to solving GEVP of full
    // NOCI-QMC basis.
    let mut projectors: Option<Projectors> = None;
    if input.write.write_coeffs {
        let p = Projectors::calculate_projectors(s, 1e-12);
        let (c0_relevant, c0_null) = p.project(&c_norm);

        // Calculate diagnostics.
        println!("{}", "=".repeat(100));
        let sc0n = s.dot(&c0_null);
        let hc0n = h.dot(&c0_null);
        println!("Action of S and H on initial null vector: ||Scn|| = {}, ||Hcn|| = {}.", sc0n.norm(), hc0n.norm());
        let proj_propagator = ProjPropagator::calculate_projected_propagator(h, s, &p, es, input.qmc.dt, &input.qmc.propagator);
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
    let c0_1norm = c0.iter().map(|z| z.norm()).sum::<f64>();
    let den = c0.mapv(|z| z.conj()).dot(&s.dot(c0));
    println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", 
            0, e_prev, 0, es, c0_1norm, den.re);

    for it in 0..input.qmc.max_steps {
        let esc = Complex64::new(es, 0.0);

        // Select propagator.
        let mut c_new_norm = match input.qmc.propagator {
            // U_{\Pi\Lambda}(\Delta\tau) = (1 + \Delta\tau E_s)\delta_{\Lambda}^\Pi - \Delta\tau(H_{\Pi\Lambda} - E_s S_{\Pi\Lambda})
            Propagator::Shifted => propagate_step_shifted(h, s, &c_norm, esc, input.qmc.dt),
            // U_{\Pi\Lambda}(\Delta\tau) = \delta_\Lambda^\Pi - \Delta\tau(H_{\Pi\Lambda} - E_sS_{\Pi\Lambda})
            Propagator::Unshifted => propagate_step_unshifted(h, s, &c_norm, esc, input.qmc.dt),
        };

        // Normalise.
        let sc = s.dot(&c_new_norm);
        let norm: Complex64 = c_new_norm.iter().zip(sc.iter()).map(|(ci, sci)| ci.conj() * sci).sum::<Complex64>().sqrt();
        c_new_norm.mapv_inplace(|z| z / norm);
        // Calculate C S C^\dagger post normalisation.
        let den = c_new_norm.mapv(|z| z.conj()).dot(&s.dot(&c_new_norm));
        // Calculate energy.
        let e = projected_energy(h, s, &c_new_norm);
        let de = (e - e_prev).abs();
        
        let alpha = norm.norm();
        logamp += alpha.ln();
        
        // Update shift dynamically if requested.
        if input.qmc.dynamic_shift {
            let a = input.qmc.dynamic_shift_alpha;  
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
        let c1norm = c_new_norm.iter().map(|z| z.norm()).sum::<f64>();
        
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", 
                it + 1, e, de, esc.re, c1norm, den.re);
        // If our energy change between iterations is large we likely have problems with
        // singularity and very low eigenvalues or a time-step that is too large. 
        if de > de_max {
            println!("Energy change too large at iter {}: |dE| = {}. 
            Either time-step too large or likely converging to un-physical eigenvalues with singular S or H", it + 1, de);
            return None
        }

        if de < input.qmc.e_tol {
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
pub fn projected_energy(h: &Array2<Complex64>, s: &Array2<Complex64>, c: &Array1<Complex64>) -> f64 {
    let hc = h.dot(c);
    let num = c.iter().zip(hc.iter()).map(|(ci, hci)| ci.conj() * hci).sum::<Complex64>();
    let sc = s.dot(c);
    let den = c.iter().zip(sc.iter()).map(|(ci, sci)| ci.conj() * sci).sum::<Complex64>();

    (num / den).re 
}

