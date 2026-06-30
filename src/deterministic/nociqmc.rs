// deterministic.rs
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Eigh, Norm, UPLO};

use crate::input::{Input, Propagator};
use crate::maths::{adjoint, parallel_matvec};
use crate::noci::NOCIScalar;

pub struct ProjPropagator<T: NOCIScalar> {
    /// Propagator block coupling the relevant subspace to itself.
    pub urr: Array2<T>,
    /// Propagator block coupling the null subspace to itself.
    pub unn: Array2<T>,
    /// Propagator block coupling the relevant subspace into the null subspace.
    pub unr: Array2<T>,
    /// Propagator block coupling the null subspace into the relevant subspace.
    pub urn: Array2<T>,
}

pub struct Projectors<T: NOCIScalar> {
    /// Eigenvectors spanning the relevant subspace of the overlap matrix.
    ur: Array2<T>,
    /// Transpose of the relevant-subspace eigenvector matrix.
    ur_dag: Array2<T>,
    /// Eigenvectors spanning the null subspace of the overlap matrix.
    un: Array2<T>,
    /// Transpose of the null-subspace eigenvector matrix.
    un_dag: Array2<T>,
}

pub struct Coefficients<T: NOCIScalar> {
    /// Iteration number at which these coefficients were recorded.
    pub iter: usize,
    /// Full coefficient vector in the complete NOCI-QMC basis.
    pub c_full: Array1<T>,
    /// Coefficient vector projected into the relevant subspace.
    pub c_relevant: Array1<T>,
    /// Coefficient vector projected into the null subspace.
    pub c_null: Array1<T>,
}

/// Return the overlap and identity shifts for a propagator.
/// # Arguments
/// - `prop`: Propagator choice.
/// - `es_s`: Overlap-transformed shift.
/// - `es`: Non-overlap-transformed shift.
/// # Returns
/// - `(f64, f64)`: Overlap shift and identity shift.
fn propagator_shifts(
    prop: &Propagator,
    es_s: f64,
    es: f64,
) -> (f64, f64) {
    match prop {
        Propagator::Unshifted => (es_s, 0.0),
        Propagator::Shifted => (es_s, es_s),
        Propagator::DoublyShifted => (es_s, es),
        Propagator::DifferenceDoublyShiftedU1 => (0.5 * (es + es_s), es - es_s),
        Propagator::DifferenceDoublyShiftedU2 => (es_s, es - es_s),
    }
}

impl<T: NOCIScalar> Projectors<T> {
    /// Calculate projectors onto the relevant and null subsapces of the overlap matrix S by
    /// diagonalising S as S = U \Lambda U^\dagger and paritioning the eigenvectors by an
    /// eigenvalue threshold. The null subspace is spanned by eigenvectors with \lambda < eps and
    /// the relevant subsapces by eigenvectors with \lambda > eps. The partioned eigenvector
    /// matrices U_r (relevant) and U_n (null) are used to form the projectors as:
    ///     P_r = U_r U_r^\dagger, P_n = U_n U_n^\dagger.
    /// # Arguments
    /// `s`: Array2, overlap matrix in full NOCI-QMC basis.
    /// `eps`: f64, tolerance for an eigenvalue being null or relevant.
    /// # Returns
    /// `Projectors`, projectors onto the relevant and null subspaces of the overlap matrix.
    pub fn calculate_projectors(
        s: &Array2<T>,
        eps: f64,
    ) -> Self {
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
        let mut ur = Array2::<T>::zeros((lambda.len(), relevant.len()));
        for (j, &icol) in relevant.iter().enumerate() {
            let col = u.slice(s![.., icol]);
            ur.slice_mut(s![.., j]).assign(&col);
        }
        let ur_dag = adjoint(&ur);
        let mut un = Array2::<T>::zeros((lambda.len(), null.len()));
        for (j, &icol) in null.iter().enumerate() {
            let col = u.slice(s![.., icol]);
            un.slice_mut(s![.., j]).assign(&col);
        }
        let un_dag = adjoint(&un);

        println!(
            "Projectors: eps = {:.3e}, dim(S) = {}, relevant = {}, null = {}",
            eps,
            lambda.len(),
            relevant.len(),
            null.len()
        );
        Projectors {
            ur,
            ur_dag,
            un,
            un_dag,
        }
    }

    /// Project a full NOCI-QMC coefficient vector c into the relevant and null subsapces of the
    /// overlap matrix S as:
    ///     c_r = P_r c = U_r U_r^\dagger c, c_n = P_n c = U_n U_n^\dagger c.
    /// # Arguments
    /// `c`: Array1, coefficient vector in the full NOCI-QMC basis.
    /// # Returns
    /// `(Array1<f64>, Array1<f64>)`, coefficients projected into the relevant and null subspaces.
    pub fn project(
        &self,
        c: &Array1<T>,
    ) -> (Array1<T>, Array1<T>) {
        // C_r = U_r U_r^\dagger C
        let yr = parallel_matvec(&self.ur_dag, c);
        let c_relevant = parallel_matvec(&self.ur, &yr);
        // C_n = U_n U_n^\dagger C
        let yn = parallel_matvec(&self.un_dag, c);
        let c_null = parallel_matvec(&self.un, &yn);

        (c_relevant, c_null)
    }
}

impl<T: NOCIScalar> ProjPropagator<T> {
    /// Express a propjector in the null and relevant subspace basis by forming the matrix (U_{rr},
    /// U_{nr} \\ U_{rn} U_{nn}). All elements of the propragator can be projected by doing for
    /// example: H_{rn} = U_r^\dagger H U_n.
    /// # Arguments
    /// `h`: Array2, NOCI Hamiltonian in the full NOCI-QMC basis.
    /// `s`: Array2, overlap matrix in the full NOCI-QMC basis.
    /// `p`: Projectors, projectors onto the relevant and null subspaces.
    /// `es`: f64, energy shift.
    /// `dt`: f64, propagation time step.
    /// `prop`: Propagator, propagator choice.
    /// # Returns
    /// `ProjPropagator`, propagator blocks expressed in the relevant and null subspace basis.
    pub fn calculate_projected_propagator(
        h: &Array2<T>,
        s: &Array2<T>,
        p: &Projectors<T>,
        es_s: f64,
        es: f64,
        dt: f64,
        prop: &Propagator,
    ) -> Self {
        let (es_s, es) = propagator_shifts(prop, es_s, es);

        let es_s = T::from_real(es_s);
        let es = T::from_real(es);
        let dt = T::from_real(dt);

        // H_{ur} = H U_r, H_{un} = H U_n, S_{ur} = S U_r, S_{un} = S U_n.
        let hur = h.dot(&p.ur);
        let hun = h.dot(&p.un);
        let sur = s.dot(&p.ur);
        let sun = s.dot(&p.un);

        let hrr = p.ur_dag.dot(&hur);
        let hnn = p.un_dag.dot(&hun);
        let hnr = p.un_dag.dot(&hur);
        let hrn = p.ur_dag.dot(&hun);
        let srr = p.ur_dag.dot(&sur);
        let snn = p.un_dag.dot(&sun);
        let snr = p.un_dag.dot(&sur);
        let srn = p.ur_dag.dot(&sun);

        let identityr = Array2::<T>::eye(hrr.nrows());
        let identityn = Array2::<T>::eye(hnn.nrows());
        let identityfac = T::from_real(1.0) + dt * es;

        let urr = identityr.mapv(|z| identityfac * z)
            - (&hrr - &srr.mapv(|z| es_s * z)).mapv(|z| dt * z);

        let unn = identityn.mapv(|z| identityfac * z)
            - (&hnn - &snn.mapv(|z| es_s * z)).mapv(|z| dt * z);

        let unr = (&hnr - &snr.mapv(|z| es_s * z)).mapv(|z| -dt * z);

        let urn = (&hrn - &srn.mapv(|z| es_s * z)).mapv(|z| -dt * z);

        Self { urr, unn, unr, urn }
    }
}

/// Perform one deterministic NOCI-QMC propagation step.
/// # Arguments
/// - `h`: NOCI Hamiltonian in the full NOCI-QMC basis.
/// - `s`: Overlap matrix in the full NOCI-QMC basis.
/// - `c`: NOCI-QMC coefficient vector.
/// - `es_s`: Overlap-transformed energy shift.
/// - `es`: Non-overlap-transformed energy shift.
/// - `dt`: Propagation time step.
/// - `prop`: Propagator choice.
/// # Returns
/// - `Array1<T>`: Updated NOCI-QMC coefficient vector.
pub fn propagate_step<T: NOCIScalar>(
    h: &Array2<T>,
    s: &Array2<T>,
    c: &Array1<T>,
    es_s: f64,
    es: f64,
    dt: f64,
    prop: &Propagator,
) -> Array1<T> {
    let (es_s, es) = propagator_shifts(prop, es_s, es);

    let es_s = T::from_real(es_s);
    let es = T::from_real(es);
    let dt = T::from_real(dt);

    let hc = parallel_matvec(h, c);
    let sc = parallel_matvec(s, c);

    let residual = hc - sc.mapv(|z| es_s * z) - c.mapv(|z| es * z);

    c - &residual.mapv(|z| dt * z)
}

/// Propagate nsteps number of time-step updates or until convergence in the energy.
/// # Arguments
/// - `h`: NOCI Hamiltonian in full NOCI-QMC basis.
/// - `s`: Overlap matrix in full NOCI-QMC basis.
/// - `c0`: Initial NOCI-QMC coefficient vector, start from reference NOCI coefficients.
/// - `es`: Initial value of the non-overlap and overlap-transformed shifts.
/// - `history`: Storage for coefficient history during propagation.
/// - `input`: User inputted options.
/// # Returns
/// - `Option<Array1<T>>`: Converged coefficient vector if propagation succeeds, otherwise `None`.
pub fn propagate<T: NOCIScalar>(
    h: &Array2<T>,
    s: &Array2<T>,
    c0: &Array1<T>,
    mut es: f64,
    history: &mut Vec<Coefficients<T>>,
    input: &Input,
) -> Option<Array1<T>> {
    let mut es_s = es;
    let mut c_norm = c0.clone();
    let mut e_prev = projected_energy(h, s, c0);
    let mut logamp = 0.0;
    let de_max = 10.0;

    // Unwrap deterministic propagation specific options.
    let det = input.det.as_ref().unwrap();

    // Calculate initial populations.
    let sc0 = parallel_matvec(s, c0);
    let pop_c0 = c0.iter().map(|z| z.abs()).sum::<f64>();
    let pop_sc0 = sc0.iter().map(|z| z.abs()).sum::<f64>();

    if !pop_c0.is_finite()
        || !pop_sc0.is_finite()
        || pop_c0 <= 0.0
        || pop_sc0 <= 0.0
    {
        println!(
            "Invalid initial deterministic populations: ||C|| = {}, ||SC|| = {}.",
            pop_c0, pop_sc0
        );
        return None;
    }

    let mut log_pop_c = pop_c0.ln();
    let mut log_pop_sc = pop_sc0.ln();

    // If we're doing deterministic investigation into relevant and null subspaces we need to
    // calculate projectors onto these spaces which involves diagonalising S. Of course for larger
    // systems this should not be done as diagonalising S has equal cost to solving GEVP of full
    // NOCI-QMC basis.
    let mut projectors: Option<Projectors<T>> = None;
    if input.write.write_deterministic_coeffs {
        let p = Projectors::calculate_projectors(s, 1e-14);
        let (c0_relevant, c0_null) = p.project(&c_norm);

        // Calculate diagnostics.
        println!("{}", "=".repeat(100));
        let sc0n = parallel_matvec(s, &c0_null);
        let hc0n = parallel_matvec(h, &c0_null);
        println!(
            "Action of S and H on initial null vector: ||Scn|| = {}, ||Hcn|| = {}.",
            sc0n.norm(),
            hc0n.norm()
        );

        let proj_propagator = ProjPropagator::calculate_projected_propagator(
            h,
            s,
            &p,
            es_s,
            es,
            input.prop_ref().dt,
            &input.prop_ref().propagator,
        );

        println!(
            "With initial shifts E_s = {}, E_s^S = {}, ||Unn|| = {}, ||Urr|| = {}, ||Urn|| = {}, ||Unr|| = {}.",
            es,
            es_s,
            proj_propagator.unn.norm(),
            proj_propagator.urr.norm(),
            proj_propagator.urn.norm(),
            proj_propagator.unr.norm()
        );

        let nnull = proj_propagator.unn.nrows();
        if nnull > 0 {
            let (evals_unn, _) = proj_propagator.unn.eigh(UPLO::Lower).unwrap();
            println!("Null-space propagator Unn eigenvalues: {}", evals_unn);
        } else {
            println!("Null-space dimension is 0 (no eigenvalues of propagator Unn).");
        }

        // Add initial coefficients to the history.
        history.push(Coefficients {
            iter: 0,
            c_full: c0.clone(),
            c_relevant: c0_relevant,
            c_null: c0_null,
        });
        projectors = Some(p);
    }

    // Print table header.
    println!("{}", "=".repeat(140));
    println!(
        "{:<6} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
        "iter",
        "E",
        "|dE|",
        "Shift (Es)",
        "Shift (EsS)",
        "||C||",
        "||SC||",
        "C^†SC"
    );

    // Print initial row.
    let den0 = c0
        .iter()
        .zip(sc0.iter())
        .map(|(&ci, &sci)| ci.conj() * sci)
        .sum::<T>()
        .re();

    println!(
        "{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
        0,
        e_prev,
        0.0,
        es,
        es_s,
        pop_c0,
        pop_sc0,
        den0
    );

    for it in 0..det.max_steps {
        // Perform one propagation step using the current shifts.
        let mut c_new_norm = propagate_step(
            h,
            s,
            &c_norm,
            es_s,
            es,
            input.prop_ref().dt,
            &input.prop_ref().propagator,
        );

        // Calculate the metric norm before normalisation.
        let sc = parallel_matvec(s, &c_new_norm);
        let norm_squared = c_new_norm
            .iter()
            .zip(sc.iter())
            .map(|(&ci, &sci)| ci.conj() * sci)
            .sum::<T>()
            .re();

        if !norm_squared.is_finite() || norm_squared <= 0.0 {
            println!(
                "Invalid metric norm at iter {}: C^†SC = {}.",
                it + 1,
                norm_squared
            );
            return None;
        }

        let norm = norm_squared.sqrt();

        // Normalise coefficients but retain the removed amplitude in logamp.
        c_new_norm.mapv_inplace(|z| z / T::from_real(norm));
        logamp += norm.ln();

        // Calculate post-normalisation quantities.
        let sc = parallel_matvec(s, &c_new_norm);
        let den = c_new_norm
            .iter()
            .zip(sc.iter())
            .map(|(&ci, &sci)| ci.conj() * sci)
            .sum::<T>()
            .re();

        let e = projected_energy(h, s, &c_new_norm);
        let de = (e - e_prev).abs();

        // Calculate the coefficient norms after metric normalisation.
        let c1norm = c_new_norm.iter().map(|z| z.abs()).sum::<f64>();
        let sc1norm = sc.iter().map(|z| z.abs()).sum::<f64>();

        if !c1norm.is_finite()
            || !sc1norm.is_finite()
            || c1norm <= 0.0
            || sc1norm <= 0.0
        {
            println!(
                "Invalid deterministic populations at iter {}: ||C|| = {}, ||SC|| = {}.",
                it + 1,
                c1norm,
                sc1norm
            );
            return None;
        }

        // Reconstruct the unnormalised populations removed by metric normalisation.
        let log_pop_c_new = logamp + c1norm.ln();
        let log_pop_sc_new = logamp + sc1norm.ln();
        let pop_c = log_pop_c_new.exp();
        let pop_sc = log_pop_sc_new.exp();

        if !pop_c.is_finite() || !pop_sc.is_finite() {
            println!(
                "Deterministic populations overflowed at iter {}: ||C|| = {}, ||SC|| = {}.",
                it + 1,
                pop_c,
                pop_sc
            );
            return None;
        }

        // Update the non-overlap and overlap-transformed shifts independently.
        if det.dynamic_shift {
            let fac = det.dynamic_shift_alpha / input.prop_ref().dt;
            es -= fac * (log_pop_c_new - log_pop_c);
            es_s -= fac * (log_pop_sc_new - log_pop_sc);
        }

        log_pop_c = log_pop_c_new;
        log_pop_sc = log_pop_sc_new;

        // If p exists we are doing projection into subspaces.
        if let Some(ref p) = projectors {
            let scale = logamp.exp();

            // Project coefficients into relevant and null subspaces.
            let (c_relevant, mut c_null) = p.project(&c_new_norm);
            c_null.mapv_inplace(|z| z * T::from_real(scale));

            // Add coefficients to the history.
            history.push(Coefficients {
                iter: it + 1,
                c_full: c_new_norm.clone(),
                c_relevant,
                c_null,
            });
        }

        // Print table row.
        println!(
            "{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
            it + 1,
            e,
            de,
            es,
            es_s,
            pop_c,
            pop_sc,
            den
        );

        // If our energy change between iterations is large we likely have problems with
        // singularity and very low eigenvalues or a time-step that is too large.
        if de > de_max {
            println!(
                "Energy change too large at iter {}: |dE| = {}.\n\
                 Either time-step too large or likely converging to un-physical eigenvalues with \
                 singular S or H",
                it + 1,
                de
            );
            return None;
        }

        if de < det.e_tol {
            return Some(c_new_norm);
        }

        c_norm = c_new_norm;
        e_prev = e;
    }

    Some(c_norm)
}

/// E(\tau) = \frac{C^\Lambda\langle\Psi_\Lambda|\hat H|\Psi_\Gamma\rangle C^\Gamma}{C^\Lambda\langle\Psi_\Lambda|\Psi_\Gamma\rangle C^\Gamma}
/// = \frac{C^\Lambda H_{\Lambda\Gamma}C^{\Gamma} }{C^\Lambda S_{\Lambda\Gamma}C^\Gamma}.
/// # Arguments
/// - `h`: NOCI Hamiltonian in full NOCI-QMC basis. Shifted by E_s * S.
/// - `s`: Overlap matrix in full NOCI-QMC basis.
/// - `c`: NOCI-QMC coefficient vector.
/// # Returns
/// - `f64`: Projected energy corresponding to coefficient vector `c`.
pub fn projected_energy<T: NOCIScalar>(
    h: &Array2<T>,
    s: &Array2<T>,
    c: &Array1<T>,
) -> f64 {
    let hc = parallel_matvec(h, c);
    let num = c
        .iter()
        .zip(hc.iter())
        .map(|(&ci, &hci)| ci.conj() * hci)
        .sum::<T>();
    let sc = parallel_matvec(s, c);
    let den = c
        .iter()
        .zip(sc.iter())
        .map(|(&ci, &sci)| ci.conj() * sci)
        .sum::<T>();
    (num / den).re()
}
