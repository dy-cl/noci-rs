//snoci.rs
use std::time::{Instant, Duration};
use std::collections::HashSet;

use ndarray::{Array1, Array2, Axis, concatenate};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::{WicksShared};

use crate::basis::generate_excited_basis;
use crate::noci::{build_noci_fock, build_noci_hs, build_noci_s, noci_density, update_wicks_fock};
use crate::maths::{general_evp_real, loewdin_x_real, parallel_matvec_real};
use crate::scf::form_fock_matrices;

pub struct SNOCIState {
    pub ecurrent: f64,
    pub coeffs: Array1<f64>,
    pub hcurrent: Array2<f64>,
    pub scurrent: Array2<f64>,
    pub candidates: Vec<SCFState>,
    pub selected: Vec<SCFState>,
    pub candidate_scores: Vec<f64>,
    pub ept2: f64,
}

struct GmresResult {
    pub x: Array1<f64>,
    pub residual_rms: f64,
    pub iterations: usize,
    pub converged: bool,
}

struct ProjectedCandidateSpaceElems {
    // Current pool candidates.
    candidates: Vec<SCFState>,
    // Candidate-candidate overlap projected into complement of current selected space. This is
    // given by S_{ab}^{\Omega} = S_{ab} - S_{ai} S^{ij} S_{jb}.
    s_ab_omega: Array2<f64>,
    // Non-projected candidate-current overlap for current pool.
    s_ai: Array2<f64>,
    // Transpose of s_ai.
    s_ia: Array2<f64>,
}

struct CandidatePool {
    // Current pool candidates.
    candidates: Vec<SCFState>,
    // Candidate-candidate overlap.
    s_ab: Array2<f64>,
    // Candidate-current overlap.
    s_ai: Array2<f64>, 
}

impl CandidatePool {
    /// Construct the initial candidate pool of determinants from the current selected space.
    /// # Arguments
    /// - `ao`: AO integrals and other system data.
    /// - `selected_space`: Current selected nonorthogonal determinant space.
    /// - `input`: User-defined input options.
    /// - `wicks`: Optional shared Wick's intermediates.
    /// - `tol`: Tolerance for whether a number is considered zero.
    /// # Returns
    /// - `CandidatePool`: Initial candidate pool containing all generated candidates together
    ///   with candidate-candidate and candidate-current overlap matrices.
    fn new(ao: &AoData, selected_space: &[SCFState], input: &Input, wicks: Option<&WicksShared>, tol: f64) -> Self {
        let candidates = generate_excited_basis(selected_space, input, false);
        if candidates.is_empty() {
            return Self {
                candidates,
                s_ab: Array2::zeros((0, 0)),
                s_ai: Array2::zeros((0, selected_space.len())),
            };
        }

        let (s_ab, _) = build_noci_s(ao, input, &candidates, &candidates, tol, wicks.as_ref().map(|ws| ws.view()), true);
        let (s_ai, _) = build_noci_s(ao, input, &candidates, selected_space, tol, wicks.as_ref().map(|ws| ws.view()), false);

        Self {candidates, s_ab, s_ai}
    }

    /// Remove any candidates from the pool that have just been selected.
    /// # Arguments
    /// - `selected`: Newly selected determinants that should no longer remain in the pool.
    /// # Returns
    /// - `()`: Updates the candidate pool in place.
    fn remove_selected(&mut self, selected: &[SCFState]) {
        let selected_keys: HashSet<&str> = selected.iter().map(|st| st.label.as_str()).collect();
        let keep: Vec<usize> = self.candidates.iter().enumerate().filter_map(|(i, st)| (!selected_keys.contains(st.label.as_str())).then_some(i)).collect();
        self.candidates = keep.iter().map(|&i| self.candidates[i].clone()).collect();
        self.s_ab = self.s_ab.select(Axis(0), &keep).select(Axis(1), &keep);
        self.s_ai = self.s_ai.select(Axis(0), &keep);
    }

    /// Remove candidates whose projected norm in the complement of the current selected space is
    /// numerically zero. Such candidates have directions almost entirely accounted for by the
    /// current selected space and would simply add null directions if included.
    /// # Arguments
    /// - `s_ij_inv`: Inverse metric in the current selected space.
    /// - `metric_tol`: Threshold below which projected candidate norms are discarded.
    /// # Returns
    /// - `()`: Updates the candidate pool in place.
    fn filter_candidates(&mut self, s_ij_inv: &Array2<f64>, metric_tol: f64) {
        if self.candidates.is_empty() {return;}
        
        // T_{aj} = S_{ai} S^{ij}.
        let t = self.s_ai.dot(s_ij_inv);
        // S_{ab, \Omega} = S_{ab} - S_{ai} S^{ij} S_{jb}.
        let s_omega_diag = self.s_ab.diag().to_owned() - (&t * &self.s_ai).sum_axis(Axis(1));

        let keep: Vec<usize> = s_omega_diag.iter().enumerate().filter_map(|(a, &d)| (d > metric_tol).then_some(a)).collect();
        if keep.len() == self.candidates.len() {return;}

        self.candidates = keep.iter().map(|&i| self.candidates[i].clone()).collect();
        self.s_ab = self.s_ab.select(Axis(0), &keep).select(Axis(1), &keep);
        self.s_ai = self.s_ai.select(Axis(0), &keep);
    }

    /// Update the candidate pool and overlap matrices once the selected space has grown.
    /// # Arguments
    /// - `ao`: AO integrals and other system data.
    /// - `selected_space`: Updated selected nonorthogonal determinant space.
    /// - `newly_selected`: Determinants added on the most recent SNOCI iteration.
    /// - `input`: User-defined input options.
    /// - `wicks`: Optional shared Wick's intermediates.
    /// - `tol`: Tolerance for whether a number is considered zero.
    /// # Returns
    /// - `()`: Updates the pool in place by removing newly selected states, extending the
    ///   candidate-current overlap block, and appending overlaps for genuinely new candidates.
    fn update(&mut self, ao: &AoData, selected_space: &[SCFState], newly_selected: &[SCFState], input: &Input, wicks: Option<&WicksShared>, tol: f64) {
        // If nothing was selected there is nothing to be done.
        if newly_selected.is_empty() {return;}
        // Remove new selections from the candidate pool.
        self.remove_selected(newly_selected);
        
        // Find candidate-newlyselected overlap and append to existing overlap. 
        if !self.candidates.is_empty() {
            let (s_aj, _) = build_noci_s(ao, input, &self.candidates, newly_selected, tol, wicks.as_ref().map(|ws| ws.view()), false);
            self.s_ai = concatenate(Axis(1), &[self.s_ai.view(), s_aj.view()]).unwrap();
        }
        
        // Generate the candidates corresponding to excitations of the newly selected determinants.
        let mut new_candidates = generate_excited_basis(newly_selected, input, false);
        let existing: HashSet<&str> = selected_space.iter().chain(self.candidates.iter()).map(|st| st.label.as_str()).collect();
        new_candidates.retain(|st| !existing.contains(st.label.as_str()));
        if new_candidates.is_empty() {return;}
        
        // Generate candidate-candidate overlap for new candidates only.
        let (s_bb, _) = build_noci_s(ao, input, &new_candidates, &new_candidates, tol, wicks.as_ref().map(|ws| ws.view()), true);
        // Get candidate-candidate overlap between new candidates and old candidate pool.
        let s_ba = if self.candidates.is_empty() {
            Array2::<f64>::zeros((new_candidates.len(), 0))
        } else {
            build_noci_s(ao, input, &new_candidates, &self.candidates, tol, wicks.as_ref().map(|ws| ws.view()), false).0
        };
        // Candidate-current overlap between new candidates and current selected space.
        let (s_bi, _) = build_noci_s(ao, input, &new_candidates, selected_space, tol, wicks.as_ref().map(|ws| ws.view()), false);
    
        // Assemble the full new candidate-candidate overlap matrix.
        if self.candidates.is_empty() {
            self.s_ab = s_bb;
            self.s_ai = s_bi;
            self.candidates.extend(new_candidates);
            return;
        }

        self.s_ab = {
            let top = concatenate(Axis(1), &[self.s_ab.view(), s_ba.t().view()]).unwrap();
            let bot = concatenate(Axis(1), &[s_ba.view(), s_bb.view()]).unwrap();
            concatenate(Axis(0), &[top.view(), bot.view()]).unwrap()
        };

        // Assemble full new candidate-current overlap matrix.
        self.s_ai = concatenate(Axis(0), &[self.s_ai.view(), s_bi.view()]).unwrap();
        self.candidates.extend(new_candidates);
    }
}

struct FockMatrixElems {
    // Current-current space Fock.
    f_ii: Array2<f64>,
    // Candidate-current space Fock.
    f_ai: Array2<f64>,
    // Candidate-current space Fock transposed.
    f_ia: Array2<f64>,
    // Candidate-candidate space Fock.
    f_ab: Array2<f64>,
}

#[derive(Default)]
pub struct SNOCIStepTimings {
    pub current_space: Duration,
    pub initial_candidate_generation: Duration,
    pub pool_overlap_update: Duration,
    pub candidate_h_ai: Duration,
    pub pseudoinverse: Duration,
    pub s_omega: Duration,
    pub generalised_fock: Duration,
    pub update_wicks: Duration,
    pub candidate_fock: Duration,
    pub f_omega: Duration,
    pub gmres: Duration,
    pub select: Duration,
}

/// Solve a linear system using (currently) unrestarted GMRES.
/// # Arguments:
/// - `m`: Matrix defining the linear system Mx = b.
/// - `b`: Right-hand side vector.
/// - `max_iter`: Maximum number of GMRES iterations.
/// - `tol`: Residual RMS convergence tolerance.
/// # Returns:
/// - `GmresResult`: Approximate solution vector together with final residual RMS, number of
///   iterations performed, and convergence flag.
fn gmres(m: &Array2<f64>, b: &Array1<f64>, max_iter: usize, tol: f64) -> GmresResult {
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
}

/// Build Hamiltonian and overlap matrix elements for the current space and solve the resulting
/// generalised eigenvalue problem.
/// # Arguments:
/// - `ao`: AO integrals and other system data.
/// - `current_space`: Current selected nonorthogonal determinant space.
/// - `input`: User-defined input options.
/// - `wicks`: Optional shared Wick's intermediates.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns:
/// - `(Array2<f64>, Array2<f64>, f64, Array1<f64>)`: Hamiltonian matrix in the current space,
///   overlap matrix in the current space, lowest eigenvalue, and corresponding eigenvector.
fn solve_current_space(ao: &AoData, current_space: &[SCFState], input: &Input, 
                       wicks: Option<&WicksShared>, tol: f64)  -> (Array2<f64>, Array2<f64>, f64, Array1<f64>)  {
        // Get Hamiltonian and overlap matrix elements for the current space.
        let (hcurrent, scurrent, _) = build_noci_hs(ao, input, current_space, current_space, tol, wicks.as_ref().map(|ws| ws.view()), true);
        // Solve current space GEVP.
        let (evals, c) = general_evp_real(&hcurrent, &scurrent, true, tol);
        // Extract lowest eigenvalue and eigenvector.
        let ecurrent = evals[0];
        let coeffs = c.column(0).to_owned();
        (hcurrent, scurrent, ecurrent, coeffs)
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
fn project_candidate_space(pool: &CandidatePool, s_ij_inv: &Array2<f64>) -> ProjectedCandidateSpaceElems {
    let s_ai = pool.s_ai.clone();
    let s_ia = s_ai.t().to_owned();
    // T_{aj} = S_{ai} S^{ij}.
    let t = s_ai.dot(s_ij_inv);
    // S_{ab, \Omega} = S_{ab} - S_{ai} S^{ij} S_{jb}.
    let s_ab_omega = &pool.s_ab - &t.dot(&s_ia);

    ProjectedCandidateSpaceElems {candidates: pool.candidates.clone(), s_ab_omega, s_ai, s_ia}
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
/// - `input`: User-defined input options.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns:
/// - `FockMatrixElems`: Current-current, candidate-current, and candidate-candidate Fock matrix
///   elements.
fn build_focks(ao: &AoData, selected_space: &[SCFState], projected_space: &ProjectedCandidateSpaceElems, fa: &Array2<f64>, 
               fb: &Array2<f64>, wicks: Option<&WicksShared>, input: &Input, tol: f64) -> FockMatrixElems {
    // Current-current Fock.
    let (f_ii, _) = build_noci_fock(ao, selected_space, selected_space, fa, fb, wicks.as_ref().map(|ws| ws.view()), tol, true, input);
    // Candidate-current Fock.
    let (f_ai, _) = build_noci_fock(ao, &projected_space.candidates, selected_space, fa, fb, wicks.as_ref().map(|ws| ws.view()), tol, false, input);
    let f_ia = f_ai.t().to_owned();
    // Candidate-candidate Fock.
    let (f_ab, _) = build_noci_fock(ao, &projected_space.candidates, &projected_space.candidates, fa, fb, wicks.as_ref().map(|ws| ws.view()), tol, true, input);
    FockMatrixElems {f_ii, f_ai, f_ia, f_ab}
}

/// Return a SNOCI state with empty selected, candidate score, and EPT2 fields.
/// # Arguments:
/// - `ecurrent`: Current NOCI energy.
/// - `coeffs`: Current-space ground-state eigenvector.
/// - `hcurrent`: Hamiltonian matrix in the current space.
/// - `scurrent`: Overlap matrix in the current space.
/// - `candidates`: Candidate determinants associated with the current iteration.
/// # Returns:
/// - `SNOCIState`: SNOCI state with empty selected determinants, empty candidate scores, and
///   zero EPT2 correction.
fn empty_state(ecurrent: f64, coeffs: Array1<f64>, hcurrent: Array2<f64>, scurrent: Array2<f64>, candidates: Vec<SCFState>) -> SNOCIState {
    SNOCIState {ecurrent, coeffs, hcurrent, scurrent, candidates, selected: Vec::new(), candidate_scores: Vec::new(), ept2: 0.0}
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
fn build_omega_fock(projected_space: &ProjectedCandidateSpaceElems, focks: FockMatrixElems, hcurrent: &Array2<f64>, 
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
fn select_candidates(candidates: &[SCFState], candidate_scores: &[f64], sigma: f64, max_add: usize,) -> Vec<SCFState> {
    let mut ranked: Vec<(SCFState, f64)> = candidates.iter().cloned().zip(candidate_scores.iter().copied()).filter(|(_, score)| *score > sigma).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.into_iter().take(max_add).map(|(state, _)| state).collect()
}

/// Print the SNOCI iteration table header.
/// # Arguments:
/// None.
/// # Returns:
/// - `()`: Prints the SNOCI iteration header to standard output.
fn print_snoci_header() {
    println!("{}", "=".repeat(100));
    println!("{:>6} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
             "iter", "NCurr", "NCand (R)", "NCand", "NSelect", "E", "Ecorr", "EPT2", "E + EPT2", "Res (GMRES)", "iter (GMRES)", "Converged (GMRES)");
}

/// Print a single SNOCI iteration summary line.
/// # Arguments:
/// - `it`: SNOCI iteration index.
/// - `n_current`: Number of determinants in the current selected space.
/// - `npool_pre`: Candidate-pool size before projected-norm filter.
/// - `npool_post`: Candidate-pool size after projected-norm filter.
/// - `e0`: RHF reference energy used to define the correlation energy.
/// - `state`: SNOCI state for the current iteration.
/// - `gmres`: GMRES solve information for the current iteration.
/// # Returns:
/// - `()`: Prints the SNOCI iteration summary to standard output.
fn print_snoci_iteration(it: usize, n_current: usize, e0: f64, state: &SNOCIState, gmres: &GmresResult,
                         npool_pre: usize, npool_post: usize) {
    println!("{:>6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
             it, n_current, npool_pre, npool_post, state.selected.len(), state.ecurrent, state.ecurrent - e0,
             state.ept2, state.ecurrent + state.ept2, gmres.residual_rms, gmres.iterations, gmres.converged);
}

/// Perform selected NOCI with selection from single and double excitations of current space.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `current_space`: Current selected nonorthogonal determinant space.
/// - `noci_reference_basis`: Vector of only the reference determinants.
/// - `input`: User-defined input options.
/// - `tol`: Tolerance for whether a number is zero.
/// - `wicks`: Mutable Wick's intermediates as we need to update Fock intermediates.
/// # Returns:
/// - `(SNOCIState, SNOCIStepTimings)`: Final SNOCI state from the last completed iteration and
///   timings for each major component of the SNOCI step.
pub fn snoci_step(ao: &AoData, current_space: &[SCFState], noci_reference_basis: &[SCFState], input: &Input, 
                  tol: f64, mut wicks: Option<&mut WicksShared>) -> (SNOCIState, SNOCIStepTimings) {
    // Unwrap SNOCI options and initialise timings, selected_space array, and the final state. 
    let opts = input.snoci.as_ref().expect("snoci_step called without input.snoci.");
    let mut timings = SNOCIStepTimings::default();
    let mut selected_space = current_space.to_vec();
    let mut final_state: Option<SNOCIState> = None;
    let mut candidate_pool: Option<CandidatePool> = None;

    // We define correlation from the RHF energy.
    let e0 = noci_reference_basis[0].e;
    print_snoci_header();

    for it in 0..opts.max_iter {
        // Solve NOCI GEVP in the current selected space.
        let t0 = Instant::now();
        let (hcurrent, scurrent, ecurrent, coeffs) = solve_current_space(ao, &selected_space, input, wicks.as_deref(), tol);
        timings.current_space += t0.elapsed();

        // Build the candidate pool once, then reuse it.
        if candidate_pool.is_none() {
            let t0 = Instant::now();
            candidate_pool = Some(CandidatePool::new(ao, &selected_space, input, wicks.as_deref(), tol));
            timings.initial_candidate_generation += t0.elapsed();
        }
          
        // Find psuedoinvserse S^{ij}. This is used to define the projector onto the current
        // selected space as:
        //  \hat P = \sum_{i,j \in \mathcal{V}_N^{(k)}} |\Psi_i \rangle S^{ij} \langle \Psi_j |.
        let t0 = Instant::now();
        let x = loewdin_x_real(&scurrent, true, tol);
        let s_ij_inv = x.dot(&x);
        timings.pseudoinverse += t0.elapsed();
   
        let npool_pre = candidate_pool.as_ref().unwrap().candidates.len();
        if let Some(pool) = candidate_pool.as_mut() {
            let t0 = Instant::now();
            // Remove pool candidates whose projected norm
            //     S_{aa}^{\Omega} = S_{aa} - S_{ai} S^{ij} S_{ja}
            // falls below the metric threshold. These candidates have directions already described
            // ny the current space and would only add linear dependence.
            pool.filter_candidates(&s_ij_inv, opts.gmres.metric_tol);
            timings.s_omega += t0.elapsed();
        }
        let pool = candidate_pool.as_ref().unwrap();
        let npool_post = pool.candidates.len();

        // No candidates left in the pool.
        if pool.candidates.is_empty() {
            return (empty_state(ecurrent, coeffs, hcurrent, scurrent, Vec::new()), timings);
        }

        //  where \hat P is as defined above.
        // Project the current candidate pool out of the selected space to form the Omega-space
        // overlap used in the linear problem.
        let t0 = Instant::now();
        let projected_candidate_space = project_candidate_space(pool, &s_ij_inv); 
        timings.s_omega += t0.elapsed();
        
        // Build candidate-current Hamiltonian only for surviving candidates.
        let t0 = Instant::now();
        let (h_ai, _, _) = build_noci_hs(ao, input, &projected_candidate_space.candidates, &selected_space, 
                                         tol, wicks.as_ref().map(|ws| ws.view()), false);
        timings.candidate_h_ai += t0.elapsed();

        // Get multireference NOCI density and form the generalised Fock matrix.
        let t0 = Instant::now();
        let (da, db) = noci_density(ao, &selected_space, &coeffs, tol);
        let (fa, fb) = form_fock_matrices(&ao.h, &ao.eri_coul, &da, &db);
        timings.generalised_fock += t0.elapsed();

        // Update Wick's intermediates for Fock matrix element calculation if using. 
        let t0 = Instant::now();
        if input.wicks.enabled && let Some(ws) = wicks.as_deref_mut() {
            update_wicks_fock(&fa, &fb, noci_reference_basis, ws);
        }
        timings.update_wicks += t0.elapsed();

        // Calculate current-current, candidate-current and candidate-candidate Fock matrix elements.
        let t0 = Instant::now();
        let focks = build_focks(ao, &selected_space, &projected_candidate_space, &fa, &fb, wicks.as_deref(), input, tol);
        timings.candidate_fock += t0.elapsed();

        // Find the projected space Fock operator and coupling vector of the linear system to solve.
        let t0 = Instant::now();
        let (f_ab_omega, v_omega) = build_omega_fock(&projected_candidate_space, focks, &hcurrent, &coeffs, &s_ij_inv, &h_ai); 
        timings.f_omega += t0.elapsed();
        
        // Set-up linear system and pass to GMRES to solve for amplitude vector A_\alpha^{(k)}
        let t0 = Instant::now();
        let m_omega_ab = &f_ab_omega - &(projected_candidate_space.s_ab_omega.mapv(|x| ecurrent * x));
        let rhs = v_omega.mapv(|x| -x);
        let a = gmres(&m_omega_ab, &rhs, opts.gmres.max_iter, opts.gmres.res_tol);
        timings.gmres += t0.elapsed();
        
        // NOCIPT2 energy is simply \sum_\alpha A_\alpha^{(k)} V_\alpha^{(k)}.
        let ept2 = a.x.dot(&v_omega);
        // Score candidates with metric |A_\alpha^{(k)} V_\alpha^{(k)}|
        let candidate_scores: Vec<f64> = a.x.iter().zip(v_omega.iter()).map(|(&a, &v)| (a * v).abs()).collect();
        
        // Select top candidates to add to current space.
        let t0 = Instant::now();
        let selected = select_candidates(&projected_candidate_space.candidates, &candidate_scores, opts.sigma, opts.max_add);
        timings.select += t0.elapsed();
        
        let state = SNOCIState {ecurrent, coeffs, hcurrent, scurrent, candidates: projected_candidate_space.candidates, selected, candidate_scores, ept2};
        print_snoci_iteration(it, selected_space.len(), e0, &state, &a, npool_pre, npool_post);

        // Convergence or stopping.
        if state.selected.is_empty() {
            println!("SNOCI stopped at iteration {}: no candidates satisfied the selection threshold ({}).", it, opts.sigma);
            return (state, timings);
        }
        if state.ept2.abs() < opts.tol {
            println!("SNOCI stopped at iteration {}: |EPT2|: {:.12} fell below tolerance {:.12}.", it, state.ept2.abs(), opts.tol);
            return (state, timings);
        }

        // Extend selected space and update the pool.
        selected_space.extend(state.selected.iter().cloned());
        if let Some(pool) = candidate_pool.as_mut() {
            let t0 = Instant::now();
            pool.update(ao, &selected_space, &state.selected, input, wicks.as_deref(), tol);
            timings.pool_overlap_update += t0.elapsed();
        }
        final_state = Some(state);
    }
    println!("SNOCI stopped: Maximum iteration was reached ({}).", opts.max_iter);
    (final_state.unwrap(), timings)
}
