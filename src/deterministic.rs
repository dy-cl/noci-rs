// deterministic.rs 
use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use ndarray_linalg::{Eigh, UPLO};

use crate::maths::parallel_matvec;
use crate::input::{Input, Propagator};

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
            if lambda[i] > eps {
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

/// Perform one deterministic update step of NOCI-QMC unshifted propagator:
/// # Arguments
///     `h`: Array2, Shifted NOCI Hamiltonian in full NOCI-QMC basis: \tilde{H} = (H - E_s * S)
///                  where E_s is the shift energy and S the NOCI overlap matrix.
///     `c`: Array1, NOCI-QMC coefficient vector.
///     `dt`: Propagation time step.
pub fn propagate_step_unshifted(h: &Array2<Complex64>, c: &Array1<Complex64>, dt: f64) -> Array1<Complex64> {
    let hc = parallel_matvec(h, c);
    let dtc = hc.mapv(|z| Complex64::new(dt, 0.0) * z);
    c - &dtc
}

/// Perform one deterministic update step of NOCI-QMC shifted propagator:
/// C^\Lambda(\tau + \Delta\tau) = C^\Lambda(\tau) - \Delta\tau(H_{\Lambda\Lambda} - E_sS_{\Lambda\Lambda} - E_s)C^\Lambda(\tau) 
/// - \Delta\tau \sum_{\Gamma \neq \Lambda}(H_{\Lambda\Gamma} - E_s S_{\Lambda\Gamma})C^\Gamma(\tau).
/// # Arguments
///     `h`: Array2, Shifted NOCI Hamiltonian in full NOCI-QMC basis: \tilde{H} = (H - E_s * S)
///                  where E_s is the shift energy and S the NOCI overlap matrix.
///     `c`: Array1, NOCI-QMC coefficient vector.
///     `esc`: Scalar, Energy shift, we just use the reference NOCI energy for this here.
///     `dt`: Propagation time step.
pub fn propagate_step_shifted(h: &Array2<Complex64>, c: &Array1<Complex64>, esc: Complex64, dt: f64) -> Array1<Complex64> {
    let hc = parallel_matvec(h, c);
    let esc_c = c.mapv(|z| esc * z);
    let dtc = (hc - esc_c).mapv(|z| Complex64::new(dt, 0.0) * z);
    c - &dtc
}

/// Propagate nsteps number of time-step updates.
/// # Arguments
///     `h`: Array2, NOCI Hamiltonian in full NOCI-QMC basis. Shifted by E_s * S.
///     `s`: Array2, Overlap matrix in full NOCI-QMC basis.
///     `c0`: Array1, Initial NOCI-QMC coefficient vector, start from reference NOCI coefficients.
///     `es`: Scalar, Energy shift, we just use the reference HF energy for this here.
///     `max_steps`: Maximum number of time-step updates to perform.
///     `e_tol`: Energy tolerance which determines convergence.
pub fn propagate(h: &Array2<Complex64>, s: &Array2<Complex64>, c0: &Array1<Complex64>, es: f64, history: &mut Vec<Coefficients>, 
                 input: &Input) -> Option<Array1<Complex64>> {

    let mut c_norm = c0.clone();
    let mut e_prev = projected_energy(h, s, c0, es);
    let esc = Complex64::new(es, 0.0);
    let de_max = 10.0;
    
    // If we're doing deterministic investigation into relevant and null subspaces we need to
    // calculate projectors onto these spaces which involves diagonalising S. Of course for larger
    // systems this should not be done as diagonalising S has equal cost to solving GEVP of full
    // NOCI-QMC basis.
    let mut projectors: Option<Projectors> = None;
    if input.write.write_coeffs {
        let p = Projectors::calculate_projectors(s, 1e-12);
        let (c0_relevant, c0_null) = p.project(c0);
        // Add initial coefficients to the history. 
        history.push(Coefficients {iter: 0, c_full: c0.clone(), c_relevant: c0_relevant, c_null: c0_null});
        projectors = Some(p);
    }

    // Print table header.
    println!("{}", "=".repeat(100));
    println!("{:<6} {:>10} {:>10}, {:>10} {:>10}", "iter", "E", "|dE|", "||C||", "√(C^† S C)");
    // Print initial.
    let c0_1norm = c0.iter().map(|z| z.norm()).sum::<f64>();
    let den = c0.mapv(|z| z.conj()).dot(&s.dot(c0));
    let den_sqrt = den.sqrt();
    println!("{:<6} {:>10.6} {:>10.3e}, {:>10.6} {:>10.6}", 0, e_prev, 0, c0_1norm, den_sqrt);

    for it in 0..input.qmc.max_steps {
        // Select propagator.
        let mut c_new_norm = match input.qmc.propagator {
            // U_{\Pi\Lambda}(\Delta\tau) = (1 + \Delta\tau E_s)\delta_{\Lambda}^\Pi - \Delta\tau(H_{\Pi\Lambda} - E_s S_{\Pi\Lambda})
            Propagator::Shifted => propagate_step_shifted(h, &c_norm, esc, input.qmc.dt),
            // U_{\Pi\Lambda}(\Delta\tau) = \delta_\Lambda^\Pi - \Delta\tau(H_{\Pi\Lambda} - E_sS_{\Pi\Lambda})
            Propagator::Unshifted => propagate_step_unshifted(h, &c_norm, input.qmc.dt),
        };

        // Normalise (enforce C S C^\dagger = 1) and calculate energy.
        let den = c_new_norm.mapv(|z| z.conj()).dot(&s.dot(&c_new_norm));
        let den_sqrt = den.sqrt();
        let sc = s.dot(&c_new_norm);
        let norm: Complex64 = c_new_norm.iter().zip(sc.iter()).map(|(ci, sci)| ci.conj() * sci).sum::<Complex64>().sqrt(); 
        c_new_norm.mapv_inplace(|z| z / norm); 
        let e = projected_energy(h, s, &c_new_norm, es);
        let de = (e - e_prev).abs();

        // If p exists we are doing projection into subspaces.
        if let Some(ref p) = projectors {
            // Project coefficients into relevant and null subspaces.
            let (c_relevant, c_null) = p.project(&c_new_norm);
            // Add coefficients to the history. 
            history.push(Coefficients {iter: it + 1, c_full: c_new_norm.clone(), c_relevant: c_relevant.clone(), c_null: c_null.clone()});
        }

        // Print table rows.
        let c1norm = c_new_norm.iter().map(|z| z.norm()).sum::<f64>();
        
        println!("{:<6} {:>10.6} {:>10.3e} {:>10.6} {:>10.6}", it + 1, e, de, c1norm, den_sqrt);
        // If our energy change between iterations is large we likely have problems with
        // singularity and very low eigenvalues or a time-step that is too large. 
        if de > de_max {
            println!("Energy change too large at iter {}: |dE| = {}. 
            Either time-step too large or likely converging to un-physical eigenvalues with singular S or H", it + 1, de);
            return None
        }
        // If the calculation is converging to an eigenvalue that is particularly low we again
        // likely have probleems with singularities. Set a lower bound as x% the RHF energy.
        // Adjust if needed.
        if e < 1.5 * es {
            println!("Energy too low at iter {}: E = {}. 
            Either time-step too large or likely converging to un-physical eigenvalues with singular S or H.", it + 1, e);
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

/// Calculate the projected energy using a given NOCI-QMC coefficient vector as:
/// E(\tau) = \frac{C^\Lambda\langle\Psi_\Lambda|\hat H|\Psi_\Gamma\rangle C^\Gamma}{C^\Lambda\langle\Psi_\Lambda|\Psi_\Gamma\rangle C^\Gamma} 
/// = \frac{C^\Lambda H_{\Lambda\Gamma}C^{\Gamma} }{C^\Lambda S_{\Lambda\Gamma}C^\Gamma}.
/// # Arguments
///     `h`: Array2, NOCI Hamiltonian in full NOCI-QMC basis. Shifted by E_s * S.
///     `s`: Array2, Overlap matrix in full NOCI-QMC basis.
///     `c`: Array1, NOCI-QMC coefficient vector.
///     `es`: Scalar, Energy shift, we just use the reference HF energy for this here.
pub fn projected_energy(h: &Array2<Complex64>, s: &Array2<Complex64>, c: &Array1<Complex64>, es: f64) -> f64 {
    let hc = h.dot(c);
    let num = c.iter().zip(hc.iter()).map(|(ci, hci)| ci.conj() * hci).sum::<Complex64>();
    let sc = s.dot(c);
    let den = c.iter().zip(sc.iter()).map(|(ci, sci)| ci.conj() * sci).sum::<Complex64>();

    (num / den).re + es
}

