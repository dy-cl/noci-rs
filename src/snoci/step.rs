//snoci.rs
use ndarray::{Array1, Array2};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::{WicksShared};
use crate::noci::{MOCache, NOCIData};
use super::{SNOCIState, CandidatePool, GmresResult};
use crate::time_call;

use crate::noci::{build_noci_hs, noci_density, build_fock_mo_cache, update_wicks_fock};
use crate::maths::{loewdin_x_real};
use crate::scf::form_fock_matrices;
use super::{gmres, solve_current_space, project_candidate_space, build_focks, build_omega_fock, select_candidates};

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
/// - `mocache`: MO-basis one and two-electron integral caches.
/// # Returns:
/// - `(SNOCIState, SNOCIStepTimings)`: Final SNOCI state from the last completed iteration and
///   timings for each major component of the SNOCI step.
pub fn snoci_step(ao: &AoData, current_space: &[SCFState], noci_reference_basis: &[SCFState], input: &Input, 
                  mocache: &[MOCache], tol: f64, mut wicks: Option<&mut WicksShared>) -> SNOCIState {
    time_call!(crate::timers::snoci::add_snoci_step, {
        // Unwrap SNOCI options and initialise timings, selected_space array, and the final state. 
        let opts = input.snoci.as_ref().expect("snoci_step called without input.snoci.");
        let mut selected_space = current_space.to_vec();
        let mut final_state: Option<SNOCIState> = None;
        let mut candidate_pool: Option<CandidatePool> = None;

        // We define correlation from the RHF energy.
        let e0 = noci_reference_basis[0].e;
        print_snoci_header();

        for it in 0..opts.max_iter {
            // Solve NOCI GEVP in the current selected space.
            let (hcurrent, scurrent, ecurrent, coeffs) = solve_current_space(ao, &selected_space, input, wicks.as_deref(), mocache, tol);

            // Build the candidate pool once, then reuse it.
            if candidate_pool.is_none() {
                candidate_pool = Some(CandidatePool::new(ao, &selected_space, input, wicks.as_deref(), tol));
            }
              
            // Find psuedoinvserse S^{ij}. This is used to define the projector onto the current
            // selected space as:
            //  \hat P = \sum_{i,j \in \mathcal{V}_N^{(k)}} |\Psi_i \rangle S^{ij} \langle \Psi_j |.
            let s_ij_inv = time_call!(crate::timers::snoci::add_build_pseudoinverse, {
                let x = loewdin_x_real(&scurrent, true, tol);
                x.dot(&x)
            });
       
            let npool_pre = candidate_pool.as_ref().unwrap().candidates.len();
            if let Some(pool) = candidate_pool.as_mut() {
                // Remove pool candidates whose projected norm
                //     S_{aa}^{\Omega} = S_{aa} - S_{ai} S^{ij} S_{ja}
                // falls below the metric threshold. These candidates have directions already described
                // ny the current space and would only add linear dependence.
                pool.filter_candidates(&s_ij_inv, opts.gmres.metric_tol);
            }
            let pool = candidate_pool.as_ref().unwrap();
            let npool_post = pool.candidates.len();

            // No candidates left in the pool.
            if pool.candidates.is_empty() {
                return empty_state(ecurrent, coeffs, hcurrent, scurrent, Vec::new());
            }

            // Project the current candidate pool out of the selected space to form the Omega-space
            // overlap used in the linear problem.
            let projected_candidate_space = project_candidate_space(pool, &s_ij_inv); 
            
            // Build candidate-current Hamiltonian only for surviving candidates.
            let wview = wicks.as_ref().map(|ws| ws.view());
            let data_proj = NOCIData::new(ao, &projected_candidate_space.candidates, input, tol, wview).withmocache(mocache);
            let (h_ai, _, _) = time_call!(crate::timers::snoci::add_build_candidate_h_ai, {
                build_noci_hs(&data_proj, &projected_candidate_space.candidates, &selected_space, false)
            });

            // Get multireference NOCI density and form the generalised Fock matrix.
            let (da, db) = noci_density(ao, &selected_space, &coeffs, tol);
            let (fa, fb) = time_call!(crate::timers::snoci::add_build_generalised_fock, {
                form_fock_matrices(&ao.h, &ao.eri_coul, &da, &db)
            });
            
            let fock_mocache = build_fock_mo_cache(&fa, &fb, noci_reference_basis);

            // Update Wick's intermediates for Fock matrix element calculation if using. 
            if input.wicks.enabled && let Some(ws) = wicks.as_deref_mut() {
                update_wicks_fock(&fa, &fb, noci_reference_basis, ws);
            }

            // Calculate current-current, candidate-current and candidate-candidate Fock matrix elements.
            let focks = build_focks(ao, &selected_space, &projected_candidate_space, &fa, &fb, wicks.as_deref(), &fock_mocache, input, tol);

            // Find the projected space Fock operator and coupling vector of the linear system to solve.
            let (f_ab_omega, v_omega) = build_omega_fock(&projected_candidate_space, focks, &hcurrent, &coeffs, &s_ij_inv, &h_ai); 
            
            // Set-up linear system and pass to GMRES to solve for amplitude vector A_\alpha^{(k)}
            let m_omega_ab = &f_ab_omega - &(projected_candidate_space.s_ab_omega.mapv(|x| ecurrent * x));
            let rhs = v_omega.mapv(|x| -x);
            let a = gmres(&m_omega_ab, &rhs, opts.gmres.max_iter, opts.gmres.res_tol);
            
            // NOCIPT2 energy is simply \sum_\alpha A_\alpha^{(k)} V_\alpha^{(k)}.
            let ept2 = a.x.dot(&v_omega);
            // Score candidates with metric |A_\alpha^{(k)} V_\alpha^{(k)}|
            let candidate_scores: Vec<f64> = a.x.iter().zip(v_omega.iter()).map(|(&a, &v)| (a * v).abs()).collect();
            
            // Select top candidates to add to current space.
            let selected = select_candidates(&projected_candidate_space.candidates, &candidate_scores, opts.sigma, opts.max_add);
            
            let state = SNOCIState {ecurrent, coeffs, hcurrent, scurrent, candidates: projected_candidate_space.candidates, selected, candidate_scores, ept2};
            print_snoci_iteration(it, selected_space.len(), e0, &state, &a, npool_pre, npool_post);

            // Convergence or stopping.
            if state.selected.is_empty() {
                println!("SNOCI stopped at iteration {}: no candidates satisfied the selection threshold ({}).", it, opts.sigma);
                return state;
            }
            if state.ept2.abs() < opts.tol {
                println!("SNOCI stopped at iteration {}: |EPT2|: {:.12} fell below tolerance {:.12}.", it, state.ept2.abs(), opts.tol);
                return state;
            }

            // Extend selected space and update the pool.
            selected_space.extend(state.selected.iter().cloned());
            if let Some(pool) = candidate_pool.as_mut() {
                pool.update(ao, &selected_space, &state.selected, input, wicks.as_deref(), tol);
            }
            final_state = Some(state);
        }
        println!("SNOCI stopped: Maximum iteration was reached ({}).", opts.max_iter);
        final_state.unwrap()
    })
}
