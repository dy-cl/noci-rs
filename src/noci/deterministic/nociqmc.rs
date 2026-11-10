// noci/deterministic/nociqmc.rs

// External crate imports.
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Eigh, Norm, UPLO};

// Crate-root imports.
use crate::NOCIScalar;
use crate::input::{Input, Propagator};
use crate::maths::{adjoint, parallel_matvec};

// Parent/sibling imports.
use super::write::{
    print_canonical_wavefunction, print_initial_null_diagnostics, print_overlap_spectrum_gaps,
    print_projected_matrix_norms, print_projected_propagator_diagnostics,
    print_projector_spectrum_diagnostics, print_propagation_table_header,
    print_propagation_table_row, print_retained_subspace_diagnostics,
};

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

    // \bar H_r = x_r^{-1/2} H_r x_r^{-1/2}.
    for i in 0..hbar.nrows() {
        for j in 0..hbar.ncols() {
            hbar[(i, j)] *= T::from_real(1.0 / (p.lambda_r[i] * p.lambda_r[j]).sqrt());
        }
    }

    hbar.eigh(UPLO::Lower).unwrap()
}

impl<T: NOCIScalar> Projectors<T> {
    /// Construct relevant and null projectors from `S = U \Lambda U^\dagger`.
    /// Eigenvectors above the cutoff span `P_r = U_r U_r^\dagger`; the remaining vectors span
    /// `P_n = U_n U_n^\dagger`.
    /// # Arguments:
    /// - `s`: Overlap matrix in the full NOCI-QMC basis.
    /// - `eps`: Relative cutoff separating relevant and null eigenvectors.
    /// # Returns
    /// - `Projectors`: Relevant and null subspace projectors.
    /// # Panics
    /// - Panics if the overlap spectrum contains a significantly negative eigenvalue.
    pub fn calculate_projectors(
        s: &Array2<T>,
        eps: f64,
    ) -> Self {
        // S = U x U^\dagger
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
    ///     `c_r = P_r c = U_r U_r^\dagger c, c_n = P_n c = U_n U_n^\dagger c.`
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
    /// Express the propagator in relevant and null subspace blocks, such as
    /// `U_{rn} = -\Delta\tau\, U_r^\dagger A U_n`, where the action is `A = S(H - E_s S)` for
    /// S-apply and `A = H - E_s S` otherwise. For B-apply this is exact, because the factor
    /// obeys `B^\dagger B = S` and `B^\dagger \hat H B = H`, so `B^\dagger(\hat H - E_s)B = H - E_s S`.
    /// # Arguments:
    /// - `h`: Hamiltonian in the full NOCI-QMC basis.
    /// - `s`: Overlap matrix in the full NOCI-QMC basis.
    /// - `p`: Relevant and null subspace projectors.
    /// - `es`: Shift `E_s`.
    /// - `dt`: Propagation time step.
    /// - `prop`: Propagator choice.
    /// # Returns
    /// - `ProjPropagator`: Propagator blocks in the relevant and null subspace basis.
    pub fn calculate_projected_propagator(
        h: &Array2<T>,
        s: &Array2<T>,
        p: &Projectors<T>,
        es: f64,
        dt: f64,
        prop: &Propagator,
    ) -> Self {
        let es = T::from_real(es);
        let dt = T::from_real(dt);

        // H U_r, H U_n, S U_r and S U_n.
        let hur = h.dot(&p.ur);
        let hun = h.dot(&p.un);
        let sur = s.dot(&p.ur);
        let sun = s.dot(&p.un);

        // The null-space couplings of `H` and `S` are diagnostic only for the unshifted
        // propagator, whose population is not confined to the range of the overlap.
        if matches!(prop, Propagator::Unshifted) {
            print_projected_matrix_norms(
                &sun,
                &hun,
                &p.un_dag.dot(&sun),
                &p.un_dag.dot(&hun),
                &p.ur_dag.dot(&hun),
            );
        }

        // `(H - E_s S) U`, with `S` applied once more for S-apply.
        let residual_r = &hur - &sur.mapv(|z| es * z);
        let residual_n = &hun - &sun.mapv(|z| es * z);

        let (action_r, action_n) = if matches!(prop, Propagator::SApply) {
            (s.dot(&residual_r), s.dot(&residual_n))
        } else {
            (residual_r, residual_n)
        };

        // `A_{rn} = U_r^\dagger A U_n` in the relevant and null subspace blocks.
        let arr = p.ur_dag.dot(&action_r);
        let anr = p.un_dag.dot(&action_r);
        let arn = p.ur_dag.dot(&action_n);
        let ann = p.un_dag.dot(&action_n);

        // `U = I - \Delta\tau A`, with the identity only on the diagonal blocks.
        let urr = Array2::<T>::eye(arr.nrows()) - arr.mapv(|z| dt * z);
        let unn = Array2::<T>::eye(ann.nrows()) - ann.mapv(|z| dt * z);
        let unr = anr.mapv(|z| -dt * z);
        let urn = arn.mapv(|z| -dt * z);

        Self { urr, unn, unr, urn }
    }
}

/// Perform one deterministic NOCI-QMC propagation step from the Hamiltonian and overlap
/// products of the current vector, so that the unshifted and B-apply propagators need no further
/// matrix-vector products and S-apply needs only `S` applied to the residual:
/// `c' = c - \Delta\tau (H - E_s S)c` for unshifted, `N' = N - \Delta\tau S(H - E_s S)N` for
/// S-apply and `N' = N - \Delta\tau B^\dagger(\hat H - E_s)BN = N - \Delta\tau(H - E_s S)N` for
/// B-apply, the expectation of its stochastic update.
/// # Arguments
/// - `s`: Overlap matrix in the full NOCI-QMC basis.
/// - `c`: Propagated vector, the coefficients `c` or the range population `N`.
/// - `products`: Hamiltonian and overlap products `Hc` and `Sc`.
/// - `es`: Shift `E_s`.
/// - `dt`: Propagation time step.
/// - `prop`: Propagator choice.
/// # Returns
/// - `Array1<T>`: Updated propagated vector.
pub fn propagate_step<T: NOCIScalar>(
    s: &Array2<T>,
    c: &Array1<T>,
    products: (&Array1<T>, &Array1<T>),
    es: f64,
    dt: f64,
    prop: &Propagator,
) -> Array1<T> {
    let (hc, sc) = products;
    let es = T::from_real(es);
    let dt = T::from_real(dt);

    // Residual `(H - E_s S)c` from the cached products, with `S` applied once more for S-apply.
    let residual = hc - &sc.mapv(|z| es * z);

    match prop {
        Propagator::Unshifted | Propagator::BApply => c - &residual.mapv(|z| dt * z),
        Propagator::SApply => c - &parallel_matvec(s, &residual).mapv(|z| dt * z),
    }
}

/// Propagate nsteps number of time-step updates or until convergence in the energy.
/// # Arguments
/// - `h`: NOCI Hamiltonian in full NOCI-QMC basis.
/// - `s`: Overlap matrix in full NOCI-QMC basis.
/// - `c0`: Initial NOCI-QMC coefficient vector, start from reference NOCI coefficients.
/// - `es`: Initial value of the shift `E_s`.
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
    basis: &crate::determinant::NOCISpace<T>,
) -> Option<Array1<T>> {
    let range = matches!(
        input.prop_ref().propagator,
        Propagator::SApply | Propagator::BApply
    );

    // The unshifted propagator evolves the coefficients `c`, and S-apply and B-apply the range
    // population `N = Sc`.
    let mut c_norm = if range {
        parallel_matvec(s, c0)
    } else {
        c0.clone()
    };

    // Hamiltonian and overlap products of the current coefficients, carried between steps.
    let mut hc = parallel_matvec(h, &c_norm);
    let mut sc = parallel_matvec(s, &c_norm);

    // Initialise projected.
    let mut e_prev = rayleigh_quotient(&c_norm, &hc, &sc);

    // Keep track of algorithmic ampltiudes.
    let mut logamp = 0.0;
    // Arbitrary maximum change in energy between iterations that
    // serves to detect if a calculation id diverging. Probably is a better
    // method of detection.
    let de_max = 10.0;

    // Unwrap deterministic propagation specific options.
    let det = input.det.as_ref().unwrap();

    // Initial population `\|c\|_1` of the propagated vector, which controls the shift.
    let pop_c0 = c_norm.iter().map(|z| z.abs()).sum::<f64>();

    if !pop_c0.is_finite() || pop_c0 <= 0.0 {
        println!(
            "Invalid initial deterministic population: ||C|| = {}.",
            pop_c0
        );
        return None;
    }

    let mut log_pop_c = pop_c0.ln();

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
            es,
            input.prop_ref().dt,
            &input.prop_ref().propagator,
        );

        print_projected_propagator_diagnostics(&proj_propagator, es, range);

        // Add initial coefficients to the history.
        history.push(Coefficients {
            iter: 0,
            c_relevant: c0_relevant,
            c_null: c0_null,
        });
        projectors = Some(p);
    }

    print_propagation_table_header(range);

    // Print initial row.
    let den0 = c_norm
        .iter()
        .zip(sc.iter())
        .map(|(&ci, &sci)| ci.conj() * sci)
        .sum::<T>()
        .re();

    print_propagation_table_row(0, (e_prev, 0.0), es, pop_c0, den0);

    for it in 0..det.max_steps {
        // Perform one propagation step using the current shifts and products.
        let mut c_new_norm = propagate_step(
            s,
            &c_norm,
            (&hc, &sc),
            es,
            input.prop_ref().dt,
            &input.prop_ref().propagator,
        );
        let mut hc_new = parallel_matvec(h, &c_new_norm);
        let mut sc_new = parallel_matvec(s, &c_new_norm);

        // Calculate the metric norm before normalisation.
        let norm_squared = c_new_norm
            .iter()
            .zip(sc_new.iter())
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

        // Normalise coefficients and, by linearity, their products, but retain the removed
        // amplitude in logamp.
        c_new_norm.mapv_inplace(|z| z / T::from_real(norm));
        hc_new.mapv_inplace(|z| z / T::from_real(norm));
        sc_new.mapv_inplace(|z| z / T::from_real(norm));
        logamp += norm.ln();

        // Calculate post-normalisation quantities.
        let den = c_new_norm
            .iter()
            .zip(sc_new.iter())
            .map(|(&ci, &sci)| ci.conj() * sci)
            .sum::<T>()
            .re();

        let e = rayleigh_quotient(&c_new_norm, &hc_new, &sc_new);
        let de = (e - e_prev).abs();

        // Population of the propagated vector after metric normalisation.
        let c1norm = c_new_norm.iter().map(|z| z.abs()).sum::<f64>();

        if !c1norm.is_finite() || c1norm <= 0.0 {
            println!(
                "Invalid deterministic population at iter {}: ||C|| = {}.",
                it + 1,
                c1norm
            );
            return None;
        }

        // Reconstruct the unnormalised population removed by metric normalisation.
        let log_pop_c_new = logamp + c1norm.ln();
        let pop_c = log_pop_c_new.exp();

        if !pop_c.is_finite() {
            println!(
                "Deterministic population overflowed at iter {}: ||C|| = {}.",
                it + 1,
                pop_c
            );
            return None;
        }

        // `E_s \leftarrow E_s - (\alpha/\Delta\tau) \ln(\|c_{k+1}\|_1/\|c_k\|_1)`.
        if det.dynamic_shift {
            es -= det.dynamic_shift_alpha / input.prop_ref().dt * (log_pop_c_new - log_pop_c);
        }

        log_pop_c = log_pop_c_new;

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

        print_propagation_table_row(it + 1, (e, de), es, pop_c, den);

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
        hc = hc_new;
        sc = sc_new;
        e_prev = e;
    }

    Some(c_norm)
}

/// `E(\tau) = \frac{C^x\langle\Psi_x|\hat H|\Psi_w\rangle C^w}{C^x\langle\Psi_x|\Psi_w\rangle C^w}`
/// `= \frac{C^x H_{xw}C^{w} }{C^x S_{xw}C^w}.`
/// # Arguments
/// - `h`: `NOCI Hamiltonian in full NOCI-QMC basis. Shifted by E_s * S.`
/// - `s`: Overlap matrix in full NOCI-QMC basis.
/// - `c`: NOCI-QMC coefficient vector.
/// # Returns
/// - `f64`: Projected energy corresponding to coefficient vector `c`.
pub fn projected_energy<T: NOCIScalar>(
    h: &Array2<T>,
    s: &Array2<T>,
    c: &Array1<T>,
) -> f64 {
    rayleigh_quotient(c, &parallel_matvec(h, c), &parallel_matvec(s, c))
}

/// Projected energy `E = c^\dagger (Hc) / c^\dagger (Sc)` from precomputed products.
/// # Arguments
/// - `c`: NOCI-QMC coefficient vector.
/// - `hc`: Hamiltonian product `Hc`.
/// - `sc`: Overlap product `Sc`.
/// # Returns
/// - `f64`: Projected energy corresponding to coefficient vector `c`.
fn rayleigh_quotient<T: NOCIScalar>(
    c: &Array1<T>,
    hc: &Array1<T>,
    sc: &Array1<T>,
) -> f64 {
    let num = c
        .iter()
        .zip(hc.iter())
        .map(|(&ci, &hci)| ci.conj() * hci)
        .sum::<T>();
    let den = c
        .iter()
        .zip(sc.iter())
        .map(|(&ci, &sci)| ci.conj() * sci)
        .sum::<T>();
    (num / den).re()
}
