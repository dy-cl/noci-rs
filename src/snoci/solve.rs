// snoci/solve.rs

use ndarray::{Array1, Array2};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::{WicksShared};
use crate::noci::{MOCache, FockMOCache, NOCIData, FockData};
use super::{CandidatePool, GmresResult, ProjectedCandidateSpaceElems, FockMatrixElems};
use crate::time_call;

use crate::noci::{build_noci_fock, build_noci_hs};
use crate::maths::{general_evp_real, parallel_matvec_real};

/// Solve a linear system using (currently) unrestarted GMRES.
/// # Arguments:
/// - `m`: Matrix defining the linear system Mx = b.
/// - `b`: Right-hand side vector.
/// - `max_iter`: Maximum number of GMRES iterations.
/// - `tol`: Residual RMS convergence tolerance.
/// # Returns:
/// - `GmresResult`: Approximate solution vector together with final residual RMS, number of
///   iterations performed, and convergence flag.
pub (in crate::snoci) fn gmres(m: &Array2<f64>, b: &Array1<f64>, max_iter: usize, tol: f64) -> GmresResult {
    time_call!(crate::timers::snoci::add_gmres, {
        let n = b.len();
        // Initial guess as all zeros.
        let mut x = Array1::<f64>::zeros(n);
        
        // Empty system is converged.
        if n == 0 {return GmresResult {x, residual_rms: 0.0, iterations: 0, converged: true};}
        
        // Initial residual is simply b as initial guess is zero.
        let r = b.clone();
        // Residual norm.
        let beta = r.dot(&r).sqrt();
        // Check if initial residual is converged and return if so.
        let mut final_residual_rms = beta / (n as f64).sqrt();
        let mut converged = final_residual_rms < tol;
        if converged {return GmresResult {x, residual_rms: final_residual_rms, iterations: 0, converged: true};}
        
        // Arnoldi basis vectors.
        let mut q: Vec<Array1<f64>> = Vec::with_capacity(max_iter + 1);
        q.push(r.mapv(|ri| ri / beta));
        
        // Upper Hessenberg matrix.
        let mut h = Array2::<f64>::zeros((max_iter + 1, max_iter));
        //  Rotations for Givens rotations.
        let mut cs = vec![0.0; max_iter];
        let mut sn = vec![0.0; max_iter];
        // RHS  of least-squares problem.
        let mut g = Array1::<f64>::zeros(max_iter + 1);
        g[0] = beta;
        
        // Counter for number of GMRES iterations.
        let mut kfinal = 0usize;

        for k in 0..max_iter {
            // Arnoldi step.
            let mut w = parallel_matvec_real(m, &q[k]);
            
            // Gram-Schmidt orthogonalisation of w against existing Krylov basis.
            for j in 0..=k {
                h[(j, k)] = q[j].dot(&w);
                w.scaled_add(-h[(j, k)], &q[j]);
            }
            h[(k + 1, k)] = w.dot(&w).sqrt();
            // Non-zero remainder means we append next Arnoldi basis vector.
            if h[(k + 1, k)] > tol {
                q.push(w.mapv(|x| x / h[(k + 1, k)]));
            }
            
            // Apply accumulated Givens rotations.
            for j in 0..k {
                let temp = cs[j] * h[(j, k)] + sn[j] * h[(j + 1, k)];
                h[(j + 1, k)] = -sn[j] * h[(j, k)] + cs[j] * h[(j + 1, k)];
                h[(j, k)] = temp;
            }
            // Construct new Givens rotation
            let hk = h[(k, k)];
            let hk1 = h[(k + 1, k)];
            let denom = (hk * hk + hk1 * hk1).sqrt();
            if denom > tol {
                cs[k] = hk / denom;
                sn[k] = hk1 / denom;
            } else {
                cs[k] = 1.0;
                sn[k] = 0.0;
            }

            // Apply new Givens rotation to  Hessenberg column and RHS.
            h[(k, k)] = cs[k] * hk + sn[k] * hk1;
            h[(k + 1, k)] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] *= cs[k];

            kfinal = k + 1;
            
            // Residual is magnitude of last component of rotated RHS
            let res_rms = g[k + 1].abs() / (n as f64).sqrt();
            if res_rms < tol {break;}
        }
        
        // Solve upper-triangular system via back substitution.
        let mut y = Array1::<f64>::zeros(kfinal);
        for ii in 0..kfinal {
            let i = kfinal - 1 - ii;
            let mut rhs = g[i];
            for j in (i + 1)..kfinal {
                rhs -= h[(i, j)] * y[j];
            }
            y[i] = rhs / h[(i, i)];
        }
        
        // Reconstruct GMRES solution.
        for j in 0..kfinal {
            x.scaled_add(y[j], &q[j]);
        }
        
        //  Find true residual and norm.
        let mut rtrue = b.clone();
        rtrue.scaled_add(-1.0, &parallel_matvec_real(m, &x));
        final_residual_rms = rtrue.dot(&rtrue).sqrt() / (n as f64).sqrt();
        converged = final_residual_rms < tol;

        GmresResult {x, residual_rms: final_residual_rms, iterations: kfinal, converged}
    })
}

/// Build Hamiltonian and overlap matrix elements for the current space and solve the resulting
/// generalised eigenvalue problem.
/// # Arguments:
/// - `ao`: AO integrals and other system data.
/// - `current_space`: Current selected nonorthogonal determinant space.
/// - `input`: User-defined input options.
/// - `wicks`: Optional shared Wick's intermediates.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns:
/// - `(Array2<f64>, Array2<f64>, f64, Array1<f64>)`: Hamiltonian matrix in the current space,
///   overlap matrix in the current space, lowest eigenvalue, and corresponding eigenvector.
pub (in crate::snoci) fn solve_current_space(ao: &AoData, current_space: &[SCFState], input: &Input, wicks: Option<&WicksShared>, 
                       mocache: &[MOCache], tol: f64)  -> (Array2<f64>, Array2<f64>, f64, Array1<f64>)  {
    time_call!(crate::timers::snoci::add_solve_current_space, {
        let wview = wicks.as_ref().map(|ws| ws.view());
        let data = NOCIData::new(ao, current_space, input, tol, wview).withmocache(mocache);

        // Get Hamiltonian and overlap matrix elements for the current space.
        let (hcurrent, scurrent, _) = build_noci_hs(&data, current_space, current_space, true);
        // Solve current space GEVP.
        let (evals, c) = general_evp_real(&hcurrent, &scurrent, true, tol);
        // Extract lowest eigenvalue and eigenvector.
        let ecurrent = evals[0];
        let coeffs = c.column(0).to_owned();
        (hcurrent, scurrent, ecurrent, coeffs)
    })
}

/// Project the current pool of candidates into complement of current selected space. Which is:
/// |\Omega_\alpha^{(k)}\rangle = \hat Q^{(k)} |\Phi_\alpha^{(k)} \rangle, where the operator \hat
/// Q is \hat Q = \hat I - \hat P^{(k)}. In projecting the candidate space, we need only form S_{ab}^{\Omega}.
/// # Arguments:
/// - `pool`: Candidate pool containing candidate-candidate and candidate-current overlaps.
/// - `s_ij_inv`: Inverse metric in the current selected space.
/// # Returns:
/// - `ProjectedCandidateSpaceElems`: Candidate-space matrices projected into the complement
///   of the current selected space.
pub (in crate::snoci) fn project_candidate_space(pool: &CandidatePool, s_ij_inv: &Array2<f64>) -> ProjectedCandidateSpaceElems {
    time_call!(crate::timers::snoci::add_project_candidate_space, {
        let s_ai = pool.s_ai.clone();
        let s_ia = s_ai.t().to_owned();
        // T_{aj} = S_{ai} S^{ij}.
        let t = s_ai.dot(s_ij_inv);
        // S_{ab, \Omega} = S_{ab} - S_{ai} S^{ij} S_{jb}.
        let s_ab_omega = &pool.s_ab - &t.dot(&s_ia);

        ProjectedCandidateSpaceElems {candidates: pool.candidates.clone(), s_ab_omega, s_ai, s_ia}
    })
}

/// Build current-current, candidate-current, and candidate-candidate Fock matrix elements.
/// # Arguments:
/// - `ao`: AO integrals and other system data.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// - `projected_space`: Projected candidate-space matrix elements.
/// - `fa`: Alpha-spin generalised Fock matrix.
/// - `fb`: Beta-spin generalised Fock matrix.
/// - `noci_reference_basis`: Vector of only the reference determinants.
/// - `wicks`: Optional shared Wick's intermediates.
/// - `fock_mocache`: MO-basis Fock integral caches.
/// - `input`: User-defined input options.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns:
/// - `FockMatrixElems`: Current-current, candidate-current, and candidate-candidate Fock matrix
///   elements.
pub (in crate::snoci) fn build_focks(ao: &AoData, selected_space: &[SCFState], projected_space: &ProjectedCandidateSpaceElems, fa: &Array2<f64>, 
               fb: &Array2<f64>, wicks: Option<&WicksShared>, fock_mocache: &[FockMOCache], input: &Input, tol: f64) -> FockMatrixElems {
    let wview = wicks.as_ref().map(|ws| ws.view());
    let data = NOCIData::new(ao, selected_space, input, tol, wview);
    let fock = FockData::new(fock_mocache, fa, fb);

    // Current-current Fock.
    let (f_ii, _) = build_noci_fock(&data, &fock, selected_space, selected_space, true);
    // Candidate-current Fock.
    let (f_ai, _) = build_noci_fock(&data, &fock, &projected_space.candidates, selected_space, false);
    let f_ia = f_ai.t().to_owned();
    // Candidate-candidate Fock.
    let (f_ab, _) = build_noci_fock(&data, &fock, &projected_space.candidates, &projected_space.candidates, true);
    FockMatrixElems {f_ii, f_ai, f_ia, f_ab}
}

/// Build the projected Omega-space Fock operator and coupling vector.
/// # Arguments:
/// - `projected_space`: Projected candidate-space matrix elements.
/// - `focks`: Current-current, candidate-current, and candidate-candidate Fock
///   matrix elements.
/// - `hcurrent`: Hamiltonian matrix in the current selected space.
/// - `coeffs`: Current-space ground-state eigenvector.
/// - `s_ij_inv`: Inverse metric in the current selected space.
/// # Returns:
/// - `(Array2<f64>, Array1<f64>)`: Projected Omega-space Fock matrix and coupling vector.
pub (in crate::snoci) fn build_omega_fock(projected_space: &ProjectedCandidateSpaceElems, focks: FockMatrixElems, hcurrent: &Array2<f64>, 
                    coeffs: &Array1<f64>, s_ij_inv: &Array2<f64>, h_ai: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    // F_{ai} S^{ij} S_{jb}.
    let f_ai_proj = focks.f_ai.dot(&s_ij_inv.dot(&projected_space.s_ia));
    // S_{ai} S^{ij} F_{jb}.
    let f_ia_proj = projected_space.s_ai.dot(&s_ij_inv.dot(&focks.f_ia));
    // S_{ai} S^{ij} F_{jk} S^{kl} S_{lb}.
    let f_ii_proj = projected_space.s_ai.dot(&s_ij_inv.dot(&focks.f_ii.dot(&s_ij_inv.dot(&projected_space.s_ia))));
    // Projected candidate space Fock.
    let f_ab_omega = &focks.f_ab - &f_ai_proj - &f_ia_proj + &f_ii_proj;
    // Projected candidate-current Hamiltonian.
    let h_ai_omega = projected_space.s_ai.dot(&s_ij_inv.dot(hcurrent));
    let v_omega = (h_ai - &h_ai_omega).dot(coeffs);
    (f_ab_omega, v_omega)
}

/// Select the highest-scoring candidates above the selection threshold.
/// # Arguments:
/// - `candidates`: Candidate determinants currently present in the projected pool.
/// - `candidate_scores`: Candidate importance scores.
/// - `sigma`: Selection threshold.
/// - `max_add`: Maximum number of candidates to add.
/// # Returns:
/// - `Vec<SCFState>`: Selected candidates sorted by decreasing score.
pub (in crate::snoci) fn select_candidates(candidates: &[SCFState], candidate_scores: &[f64], sigma: f64, max_add: usize,) -> Vec<SCFState> {
    let mut ranked: Vec<(SCFState, f64)> = candidates.iter().cloned().zip(candidate_scores.iter().copied()).filter(|(_, score)| *score > sigma).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.into_iter().take(max_add).map(|(state, _)| state).collect()
}

