//snoci.rs
use std::time::{Instant, Duration};

use ndarray::{Array1, Array2, Axis};

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

struct CandidateSpaceMatrixElems {
    // List of all candidate states.
    candidates: Vec<SCFState>,
    // Candidate-candidate space overlap.
    s_ab: Array2<f64>,
    // Candidate-current space Hamiltonian.
    h_ai: Array2<f64>,
    // Candidate-current space overlap.
    s_ai: Array2<f64>,
}

struct FilteredCandidateSpaceMatrixElems {
    // List of all filtered candidate states.
    candidates: Vec<SCFState>,
    // Filtered candidate-candidate space after projecting out current space directions.
    s_ab_omega: Array2<f64>,
    // Filtered candidate-candidate space Hamiltonian.
    h_ai: Array2<f64>,
    // Filtered candidate-candidate space overlap.
    s_ai: Array2<f64>,
    // Filtered candidate-candidate space overlap transposed.
    s_ia: Array2<f64>,
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
    pub generate_candidates: Duration,
    pub candidate_hs: Duration,
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
/// - `noci_reference_basis`: Vector of only the reference determinants.
/// - `input`: User-defined input options.
/// - `wicks`: Optional shared Wick's intermediates.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns:
/// - `(Array2<f64>, Array2<f64>, f64, Array1<f64>)`: Hamiltonian matrix in the current space,
///   overlap matrix in the current space, lowest eigenvalue, and corresponding eigenvector.
fn solve_current_space(ao: &AoData, current_space: &[SCFState], noci_reference_basis: &[SCFState], input: &Input, 
                       wicks: Option<&WicksShared>, tol: f64)  -> (Array2<f64>, Array2<f64>, f64, Array1<f64>)  {
        // Get Hamiltonian and overlap matrix elements for the current space.
        let (hcurrent, scurrent, _) = build_noci_hs(ao, input, current_space, current_space, noci_reference_basis, tol, wicks.as_ref().map(|ws| ws.view()), true);
        // Solve current space GEVP.
        let (evals, c) = general_evp_real(&hcurrent, &scurrent, true, tol);
        // Extract lowest eigenvalue and eigenvector.
        let ecurrent = evals[0];
        let coeffs = c.column(0).to_owned();
        (hcurrent, scurrent, ecurrent, coeffs)
}

/// Build candidate-candidate and candidate-current overlap and Hamiltonian matrix elements.
/// # Arguments:
/// - `ao`: AO integrals and other system data.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// - `candidates`: Candidate determinants generated from the selected space.
/// - `noci_reference_basis`: Vector of only the reference determinants.
/// - `wicks`: Optional shared Wick's intermediates.
/// - `tol`: Tolerance for whether a number is considered zero.
/// - `input`: User-defined input options.
/// # Returns:
/// - `CandidateSpaceMatrixElems`: Candidate-space overlap and Hamiltonian matrix elements.
fn build_candidate_space(ao: &AoData, selected_space: &[SCFState], candidates: &[SCFState], noci_reference_basis: &[SCFState],
                            wicks: Option<&WicksShared>, tol: f64, input: &Input) -> CandidateSpaceMatrixElems {
    // Candidate-candidate overlap.
    let (s_ab, _) = build_noci_s(ao, input, candidates, candidates, noci_reference_basis, tol, wicks.as_ref().map(|ws| ws.view()), true);
    // Candidate-current Hamiltonian and overlap.
    let (h_ai, s_ai, _) = build_noci_hs(ao, input, candidates, selected_space, noci_reference_basis, tol, wicks.as_ref().map(|ws| ws.view()), false);
    CandidateSpaceMatrixElems {candidates: candidates.to_vec(), s_ab, h_ai, s_ai}
}

/// Project the candidate space out of the current selected space and remove linearly dependent candidates.
/// # Arguments:
/// - `candidate_space`: Candidate-space overlap and Hamiltonian matrix elements.
/// - `s_ij_inv`: Inverse metric in the current space.
/// - `metric_tol`: Threshold below which projected candidate norms are discarded.
/// # Returns:
/// - `Option<FilteredCandidateSpaceMatrixElems>`: Projected and filtered candidate-space matrix
///   elements if any candidates survive, otherwise `None`.
fn filter_candidate_space(candidate_space: CandidateSpaceMatrixElems, s_ij_inv: &Array2<f64>, metric_tol:  f64) -> Option<FilteredCandidateSpaceMatrixElems> {
    // Projection of candidate-space metric into current space S_{ai} S^{ij} S_{jb}.
    let sbi_ij_ja = candidate_space.s_ai.dot(&s_ij_inv.dot(&candidate_space.s_ai.t()));
    // Projected candidate-space metric S_{\Omega, ab} =  S_{ab} - S_{ai} S^{ij} S_{jb}. Components
    // that lie in current space are projected out.
    let s_omega_ab = &candidate_space.s_ab - &sbi_ij_ja;
    //  Keep only candidates with projected norm larger than some tolerance. Norms close to zero
    //  indicate linear dependence and therefore are discarded.
    let keep: Vec<usize> = (0..candidate_space.candidates.len()).filter(|&a| s_omega_ab[(a, a)] > metric_tol).collect();
    if keep.is_empty() {return None;}
    // Keep only surviving candidates.
    let filtered_candidates = keep.iter().map(|&i| candidate_space.candidates[i].clone()).collect();
    // Filter matrices to only have surviving candidates elements.
    let h_ai = candidate_space.h_ai.select(Axis(0), &keep);
    let s_ai = candidate_space.s_ai.select(Axis(0), &keep);
    let s_ia = s_ai.t().to_owned();
    let s_ab_omega = s_omega_ab.select(Axis(0), &keep).select(Axis(1), &keep);

    Some(FilteredCandidateSpaceMatrixElems {candidates: filtered_candidates, h_ai, s_ai, s_ia, s_ab_omega})
}

/// Build current-current, candidate-current, and candidate-candidate Fock matrix elements.
/// # Arguments:
/// - `ao`: AO integrals and other system data.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// - `filtered_space`: Projected and filtered candidate-space
///   matrix elements.
/// - `fa`: Alpha-spin generalised Fock matrix.
/// - `fb`: Beta-spin generalised Fock matrix.
/// - `noci_reference_basis`: Vector of only the reference determinants.
/// - `wicks`: Optional shared Wick's intermediates.
/// - `input`: User-defined input options.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns:
/// - `FockMatrixElems`: Current-current, candidate-current, and candidate-candidate Fock matrix
///   elements.
fn build_focks(ao: &AoData, selected_space: &[SCFState], filtered_space: &FilteredCandidateSpaceMatrixElems, fa: &Array2<f64>, fb: &Array2<f64>,
               noci_reference_basis: &[SCFState], wicks: Option<&WicksShared>, input: &Input, tol: f64) -> FockMatrixElems {
    // Current-current Fock.
    let (f_ii, _) = build_noci_fock(ao, selected_space, selected_space, fa, fb, noci_reference_basis, 
                                   wicks.as_ref().map(|ws| ws.view()), tol, true, input);
    // Candidate-current Fock.
    let (f_ai, _) = build_noci_fock(ao, &filtered_space.candidates, selected_space, fa, fb, noci_reference_basis, 
                                   wicks.as_ref().map(|ws| ws.view()), tol, false, input);
    let f_ia = f_ai.t().to_owned();
    // Candidate-candidate Fock.
    let (f_ab, _) = build_noci_fock(ao, &filtered_space.candidates, &filtered_space.candidates, fa, fb, 
                                   noci_reference_basis, wicks.as_ref().map(|ws| ws.view()), tol, true, input);
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
/// - `filtered_space`: Projected and filtered candidate-space
///   matrix elements.
/// - `focks`: Current-current, candidate-current, and candidate-candidate Fock
///   matrix elements.
/// - `hcurrent`: Hamiltonian matrix in the current selected space.
/// - `coeffs`: Current-space ground-state eigenvector.
/// - `s_ij_inv`: Inverse metric in the current selected space.
/// # Returns:
/// - `(Array2<f64>, Array1<f64>)`: Projected Omega-space Fock matrix and coupling vector.
fn build_omega_fock(filtered_space: &FilteredCandidateSpaceMatrixElems, focks: FockMatrixElems, hcurrent: &Array2<f64>, 
                    coeffs: &Array1<f64>, s_ij_inv: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let f_ai_proj = focks.f_ai.dot(&s_ij_inv.dot(&filtered_space.s_ia));
    let f_ia_proj = filtered_space.s_ai.dot(&s_ij_inv.dot(&focks.f_ia));
    let f_ii_proj = filtered_space.s_ai.dot(&s_ij_inv.dot(&focks.f_ii.dot(&s_ij_inv.dot(&filtered_space.s_ia))));
    let f_ab_omega = &focks.f_ab - &f_ai_proj - &f_ia_proj + &f_ii_proj;
    let h_proj = filtered_space.s_ai.dot(&s_ij_inv.dot(hcurrent));
    let v_omega = (&filtered_space.h_ai - &h_proj).dot(coeffs);
    (f_ab_omega, v_omega)
}

/// Select the highest-scoring candidates above the selection threshold.
/// # Arguments:
/// - `candidates`: Filtered candidate determinants.
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
             "iter", "NCurr", "NCand", "NCand (F)", "NSelect", "E", "Ecorr", "EPT2", "E + EPT2", "Res (GMRES)", "iter (GMRES)", "Converged (GMRES)");
}

/// Print a single SNOCI iteration summary line.
/// # Arguments:
/// - `it`: SNOCI iteration index.
/// - `n_current`: Number of determinants in the current selected space.
/// - `n_generated`: Number of generated candidates before filtering.
/// - `e0`: RHF reference energy used to define the correlation energy.
/// - `state`: SNOCI state for the current iteration.
/// - `gmres`: GMRES solve information for the current iteration.
/// # Returns:
/// - `()`: Prints the SNOCI iteration summary to standard output.
fn print_snoci_iteration(it: usize, n_current: usize, n_generated: usize, e0: f64, state: &SNOCIState, gmres: &GmresResult) {
    println!("{:>6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
             it, n_current, n_generated, state.candidates.len(), state.selected.len(), state.ecurrent, state.ecurrent - e0,
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

    // We define correlation from the RHF energy.
    let e0 = noci_reference_basis[0].e;
    print_snoci_header();

    for it in 0..opts.max_iter {
        // Solve NOCI GEVP in the current selected space.
        let t0 = Instant::now();
        let (hcurrent, scurrent, ecurrent, coeffs) = solve_current_space(ao, &selected_space, noci_reference_basis, input, wicks.as_deref(), tol);
        timings.current_space += t0.elapsed();
        
        // Generate single and double excitations of the current space as potential candidates.
        let t0 = Instant::now();
        // False flag ensures that the current selected space is not returned alongside the
        // potential excited candidates.
        let candidates = generate_excited_basis(&selected_space, input, false);
        timings.generate_candidates += t0.elapsed();

        // If the candidates list is empty we have nothing more to do. 
        if candidates.is_empty() {return (empty_state(ecurrent, coeffs, hcurrent, scurrent, candidates), timings);}

        // Build candidate-candidate and candidate-current matrices. Note that we use incides i, j to refer to
        // the current state space, and a, b (or \alpha and \beta) to refer to the candidate space.
        let t0 = Instant::now();
        let candidate_space = build_candidate_space(ao, &selected_space, &candidates, noci_reference_basis, wicks.as_deref(), tol, input);
        timings.candidate_hs += t0.elapsed();

        // Find psuedoinvserse S^{ij}. This is used to define the projector onto the current
        // selected space as:
        //  \hat P = \sum_{i,j \in \mathcal{V}_N^{(k)}} |\Psi_i \rangle S^{ij} \langle \Psi_j |.
        let t0 = Instant::now();
        let x = loewdin_x_real(&scurrent, true, tol);
        let s_ij_inv = x.dot(&x);
        timings.pseudoinverse += t0.elapsed();
        
        // To avoid redundancy we filter out the potential candidates and remove directions already
        // present in the current space. The projector used is:
        //  \hat Q = \hat I - \hat P,
        //  where \hat P is as defined above.
        let t0 = Instant::now();
        let filtered_candidate_space = filter_candidate_space(candidate_space, &s_ij_inv, opts.gmres.metric_tol); 
        timings.s_omega += t0.elapsed();
        
        // If the resulting filtered candidate space is empty return.
        let filtered_candidate_space = match filtered_candidate_space {
            Some(filtered) => filtered,
            None => return (empty_state(ecurrent, coeffs, hcurrent, scurrent, candidates), timings),
        };

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
        let focks = build_focks(ao, &selected_space, &filtered_candidate_space, &fa, &fb, noci_reference_basis, wicks.as_deref(), input, tol);
        timings.candidate_fock += t0.elapsed();

        // Find the projected space Fock operator and coupling vector of the linear system to solve.
        let t0 = Instant::now();
        let (f_ab_omega, v_omega) = build_omega_fock(&filtered_candidate_space, focks, &hcurrent, &coeffs, &s_ij_inv); 
        timings.f_omega += t0.elapsed();
        
        // Set-up linear system and pass to GMRES to solve for amplitude vector A_\alpha^{(k)}
        let t0 = Instant::now();
        let m_omega_ab = &f_ab_omega - &(filtered_candidate_space.s_ab_omega.mapv(|x| ecurrent * x));
        let rhs = v_omega.mapv(|x| -x);
        let a = gmres(&m_omega_ab, &rhs, opts.gmres.max_iter, opts.gmres.res_tol);
        timings.gmres += t0.elapsed();
        
        // NOCIPT2 energy is simply \sum_\alpha A_\alpha^{(k)} V_\alpha^{(k)}.
        let ept2 = a.x.dot(&v_omega);
        // Score candidates with metric |A_\alpha^{(k)} V_\alpha^{(k)}|
        let candidate_scores: Vec<f64> = a.x.iter().zip(v_omega.iter()).map(|(&a, &v)| (a * v).abs()).collect();
        
        // Select top candidates to add to current space.
        let t0 = Instant::now();
        let selected = select_candidates(&filtered_candidate_space.candidates, &candidate_scores, opts.sigma, opts.max_add);
        timings.select += t0.elapsed();
        
        let state = SNOCIState {ecurrent, coeffs, hcurrent, scurrent, candidates: filtered_candidate_space.candidates, selected, candidate_scores, ept2};
        print_snoci_iteration(it, selected_space.len(), candidates.len(), e0, &state, &a);

        // Convergence or stopping.
        if state.selected.is_empty() {
            println!("SNOCI stopped at iteration {}: no candidates satisfied the selection threshold ({}).", it, opts.sigma);
            return (state, timings);
        }
        if state.ept2.abs() < opts.tol {
            println!("SNOCI stopped at iteration {}: |EPT2|: {:.12} fell below tolerance {:.12}.", it, state.ept2.abs(), opts.tol);
            return (state, timings);
        }

        selected_space.extend(state.selected.iter().cloned());
        final_state = Some(state);
    }
    println!("SNOCI stopped: Maximum iteration was reached ({}).", opts.max_iter);
    (final_state.unwrap(), timings)
}
