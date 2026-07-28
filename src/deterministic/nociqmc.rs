// deterministic.rs
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Eigh, Norm, UPLO};

use super::write::{
    print_canonical_wavefunction, print_initial_null_diagnostics, print_overlap_spectrum_gaps,
    print_projected_matrix_norms, print_projected_propagator_diagnostics,
    print_projector_spectrum_diagnostics, print_propagation_table_header,
    print_propagation_table_row, print_retained_subspace_diagnostics,
};
use crate::DetState;
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
    pub(super) ur: Array2<T>,
    /// Overlap eigenvalues in the relevant subspace.
    pub(super) lambda_r: Array1<f64>,
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
        Propagator::DirectOverlap => {
            panic!("Propagator::DirectOverlap cannot be specified in this way.")
        }
    }
}

/// Diagonalise the Hamiltonian in the canonically orthogonalised relevant subspace.
/// # Arguments
/// - `h`: Hamiltonian matrix in the full NOCI-QMC basis.
/// - `p`: Projectors onto the relevant and null subspaces of the overlap matrix.
/// # Returns
/// - `(Array1<f64>, Array2<T>)`: Eigenvalues and eigenvectors in the canonical relevant basis.
fn diagonalise_retained_hamiltonian<T: NOCIScalar>(
    h: &Array2<T>,
    p: &Projectors<T>,
) -> (Array1<f64>, Array2<T>) {
    // H_r = U_r^\dagger H U_r.
    let hr = p.ur_dag.dot(&h.dot(&p.ur));
    let mut hbar = hr.clone();

    // \bar H_r = \Lambda_r^{-1/2} H_r \Lambda_r^{-1/2}.
    for i in 0..hbar.nrows() {
        for j in 0..hbar.ncols() {
            hbar[(i, j)] *= T::from_real(1.0 / (p.lambda_r[i] * p.lambda_r[j]).sqrt());
        }
    }

    hbar.eigh(UPLO::Lower).unwrap()
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

        // \lambda_{\text{scale}} = \max(1, \max_i |\lambda_i|)
        let scale = lambda
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);

        // Eigenvalue cutoff is user chosen epsilon scaled by scale.
        let nulltol = eps * scale;
        let negativetol = 100.0 * nulltol;

        // Identify largest gap in eigenvalue spectrum of the overlap.
        print_overlap_spectrum_gaps(&lambda);

        let mut relevant: Vec<usize> = Vec::new();
        let mut null: Vec<usize> = Vec::new();

        for i in 0..lambda.len() {
            if lambda[i] < -negativetol {
                // Overlap should be positive semidefinite so a sufficiently large negative
                // eigenvalue suggests something has gone wrong somewhere.
                panic!(
                    "Overlap matrix has significantly negative eigenvalue {}. Null tolerance: {}, negative tolerance: {}.",
                    lambda[i], nulltol, negativetol,
                );
            // Catergorise eigenvalues into null and non-null spaces.
            } else if lambda[i] > nulltol {
                relevant.push(i);
            } else {
                null.push(i);
            }
        }

        // Construct U_{\text{relevant}}, that is, the matrix of eigenvectors of retained non-null eigenvalues.
        let mut ur = Array2::<T>::zeros((lambda.len(), relevant.len()));
        for (j, &icol) in relevant.iter().enumerate() {
            let col = u.slice(s![.., icol]);
            ur.slice_mut(s![.., j]).assign(&col);
        }
        let lambda_r = Array1::from_vec(relevant.iter().map(|&i| lambda[i]).collect());
        let ur_dag = adjoint(&ur);
        // Construct U_{\text{null}}, that is, the matrix of eigenvectors of discarded null eigenvalues.
        let mut un = Array2::<T>::zeros((lambda.len(), null.len()));
        for (j, &icol) in null.iter().enumerate() {
            let col = u.slice(s![.., icol]);
            un.slice_mut(s![.., j]).assign(&col);
        }
        let un_dag = adjoint(&un);

        print_projector_spectrum_diagnostics(
            eps,
            &lambda,
            scale,
            nulltol,
            negativetol,
            &relevant,
            &null,
        );

        Projectors {
            ur,
            lambda_r,
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
        if matches!(prop, Propagator::DirectOverlap) {
            let es_s = T::from_real(es_s);
            let dt = T::from_real(dt);

            // Actions on relevant and null source subspaces.
            let hur = h.dot(&p.ur);
            let hun = h.dot(&p.un);
            let sur = s.dot(&p.ur);
            let sun = s.dot(&p.un);

            // (H - EsS S) U_r and (H - EsS S) U_n.
            let residual_r = hur - sur.mapv(|z| es_s * z);

            let residual_n = hun - sun.mapv(|z| es_s * z);

            // S(H - EsS S) U_r and S(H - EsS S) U_n.
            let action_r = s.dot(&residual_r);
            let action_n = s.dot(&residual_n);

            // Project the direct-overlap action into the relevant/null
            // subspace basis.
            let arr = p.ur_dag.dot(&action_r);
            let anr = p.un_dag.dot(&action_r);
            let arn = p.ur_dag.dot(&action_n);
            let ann = p.un_dag.dot(&action_n);

            let identity_r = Array2::<T>::eye(arr.nrows());
            let identity_n = Array2::<T>::eye(ann.nrows());

            let urr = identity_r - arr.mapv(|z| dt * z);

            let unn = identity_n - ann.mapv(|z| dt * z);

            let unr = anr.mapv(|z| -dt * z);

            let urn = arn.mapv(|z| -dt * z);

            return Self { urr, unn, unr, urn };
        }

        let (es_s, es) = propagator_shifts(prop, es_s, es);

        let es_s = T::from_real(es_s);
        let es = T::from_real(es);
        let dt = T::from_real(dt);

        // H U_r, H U_n, S U_r, and S U_n.
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

        print_projected_matrix_norms(&sun, &hun, &snn, &hnn, &hrn);

        let identity_r = Array2::<T>::eye(hrr.nrows());
        let identity_n = Array2::<T>::eye(hnn.nrows());

        let identity_fac = T::from_real(1.0) + dt * es;

        let urr = identity_r.mapv(|z| identity_fac * z)
            - (&hrr - &srr.mapv(|z| es_s * z)).mapv(|z| dt * z);

        let unn = identity_n.mapv(|z| identity_fac * z)
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
    match prop {
        Propagator::DirectOverlap => {
            let es_s = T::from_real(es_s);
            let dt = T::from_real(dt);

            let hc = parallel_matvec(h, c);
            let sc = parallel_matvec(s, c);

            let residual = hc - sc.mapv(|z| es_s * z);

            let overlap_residual = parallel_matvec(s, &residual);

            c - &overlap_residual.mapv(|z| dt * z)
        }

        _ => {
            let (es_s, es) = propagator_shifts(prop, es_s, es);

            let es_s = T::from_real(es_s);
            let es = T::from_real(es);
            let dt = T::from_real(dt);

            let hc = parallel_matvec(h, c);
            let sc = parallel_matvec(s, c);

            let residual = hc - sc.mapv(|z| es_s * z) - c.mapv(|z| es * z);

            c - &residual.mapv(|z| dt * z)
        }
    }
}

/// Propagate nsteps number of time-step updates or until convergence in the energy.
/// # Arguments
/// - `h`: NOCI Hamiltonian in full NOCI-QMC basis.
/// - `s`: Overlap matrix in full NOCI-QMC basis.
/// - `c0`: Initial NOCI-QMC coefficient vector, start from reference NOCI coefficients.
/// - `es`: Initial value of the non-overlap and overlap-transformed shifts.
/// - `history`: Storage for coefficient history during propagation.
/// - `input`: User inputted options.
/// - `basis`: Original NOCI-QMC determinant basis in the ordering used for H and S.
/// # Returns
/// - `Option<Array1<T>>`: Converged coefficient vector if propagation succeeds, otherwise `None`.
pub fn propagate<T: NOCIScalar>(
    h: &Array2<T>,
    s: &Array2<T>,
    c0: &Array1<T>,
    mut es: f64,
    history: &mut Vec<Coefficients<T>>,
    input: &Input,
    basis: &[DetState<T>],
) -> Option<Array1<T>> {
    let mut es_s = es;
    let doverlap = matches!(input.prop_ref().propagator, Propagator::DirectOverlap);

    // There is no identity shift in direct-overlap propagation.
    if doverlap {
        es = 0.0;
    }

    // Old style propagators evolve coefficient vector directly whilst
    // direct overlap propagation evolves coefficient vector acted on by
    // the overlap.
    let mut c_norm = if doverlap {
        parallel_matvec(s, c0)
    } else {
        c0.clone()
    };

    // Initialise projected.
    let mut e_prev = projected_energy(h, s, &c_norm);

    // Keep track of algorithmic ampltiudes.
    let mut logamp = 0.0;
    // Arbitrary maximum change in energy between iterations that
    // serves to detect if a calculation id diverging. Probably is a better
    // method of detection.
    let de_max = 10.0;

    // Unwrap deterministic propagation specific options.
    let det = input.det.as_ref().unwrap();

    // Calculate initial populations.
    let sc0 = parallel_matvec(s, &c_norm);
    let pop_c0 = c_norm.iter().map(|z| z.abs()).sum::<f64>();
    let pop_sc0 = sc0.iter().map(|z| z.abs()).sum::<f64>();

    if !pop_c0.is_finite() || !pop_sc0.is_finite() || pop_c0 <= 0.0 || pop_sc0 <= 0.0 {
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
        let p = Projectors::calculate_projectors(s, det.projector_eps);
        let (retained_e, retained_c) = diagonalise_retained_hamiltonian(h, &p);
        print_retained_subspace_diagnostics(&retained_e, &retained_c);
        print_canonical_wavefunction(
            &retained_c.slice(s![.., 0]).to_owned(),
            &p,
            basis,
            det.canonical_states_n,
            det.canonical_terms_m,
        );
        let (c0_relevant, c0_null) = p.project(&c_norm);

        // Calculate diagnostics.
        let sc0n = parallel_matvec(s, &c0_null);
        let hc0n = parallel_matvec(h, &c0_null);
        let cn_norm = c0_null.norm();
        print_initial_null_diagnostics(&sc0n, &hc0n, cn_norm);

        let proj_propagator = ProjPropagator::calculate_projected_propagator(
            h,
            s,
            &p,
            es_s,
            es,
            input.prop_ref().dt,
            &input.prop_ref().propagator,
        );

        print_projected_propagator_diagnostics(&proj_propagator, es, es_s, doverlap);

        // Add initial coefficients to the history.
        history.push(Coefficients {
            iter: 0,
            c_relevant: c0_relevant,
            c_null: c0_null,
        });
        projectors = Some(p);
    }

    print_propagation_table_header(doverlap);

    // Print initial row.
    let den0 = c_norm
        .iter()
        .zip(sc0.iter())
        .map(|(&ci, &sci)| ci.conj() * sci)
        .sum::<T>()
        .re();

    print_propagation_table_row(0, (e_prev, 0.0), (es, es_s), (pop_c0, pop_sc0), den0);

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

        if !c1norm.is_finite() || !sc1norm.is_finite() || c1norm <= 0.0 || sc1norm <= 0.0 {
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

            if doverlap {
                es_s -= fac * (log_pop_c_new - log_pop_c);
            } else {
                es -= fac * (log_pop_c_new - log_pop_c);
                es_s -= fac * (log_pop_sc_new - log_pop_sc);
            }
        }

        log_pop_c = log_pop_c_new;
        log_pop_sc = log_pop_sc_new;

        if let Some(ref p) = projectors {
            let scale = T::from_real(logamp.exp());

            let (mut c_relevant, mut c_null) = p.project(&c_new_norm);

            c_relevant.mapv_inplace(|z| z * scale);
            c_null.mapv_inplace(|z| z * scale);

            history.push(Coefficients {
                iter: it + 1,
                c_relevant,
                c_null,
            });
        }

        print_propagation_table_row(it + 1, (e, de), (es, es_s), (pop_c, pop_sc), den);

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
