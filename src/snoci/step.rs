// snoci/step.rs

use ndarray::{Array1, Array2};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::WicksShared;
use crate::noci::{MOCache};
use super::{SNOCIState, CandidatePool, GMRES};
use crate::time_call;

use crate::noci::{noci_density, build_fock_mo_cache, update_wicks_fock};
use crate::maths::loewdin_x_real;
use crate::scf::form_fock_matrices;
use super::{gmres, solve_current_space, build_snoci_overlaps, build_snoci_focks, build_candidate_m, 
            build_omega_candidate_m, build_omega_coupling_v, select_candidates, build_candidate_current_h};

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
/// - `e0`: RHF reference energy used to define the correlation energy.
/// - `state`: SNOCI state for the current iteration.
/// - `gmres`: GMRES solve information for the current iteration.
/// - `npool_pre`: Candidate-pool size before projected-norm filter.
/// - `npool_post`: Candidate-pool size after projected-norm filter.
/// # Returns:
/// - `()`: Prints the SNOCI iteration summary to standard output.
fn print_snoci_iteration(it: usize, n_current: usize, e0: f64, state: &SNOCIState, gmres: &GMRES, npool_pre: usize, npool_post: usize) {
    println!("{:>6} {:>16} {:>16} {:>16} {:>16} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16} {:>16}",
        it,
        n_current,
        npool_pre,
        npool_post,
        state.selected.len(),
        state.ecurrent,
        state.ecurrent - e0,
        state.ept2,
        state.ecurrent + state.ept2,
        gmres.residual_rms,
        gmres.iterations,
        gmres.converged);
}

/// Perform selected NOCI with selection from excitations of the current space.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `current_space`: Current selected nonorthogonal determinant space.
/// - `noci_reference_basis`: Vector of only the reference determinants.
/// - `input`: User-defined input options.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `tol`: Tolerance for whether a number is zero.
/// - `wicks`: Mutable Wick's intermediates as we need to update Fock intermediates.
/// # Returns:
/// - `SNOCIState`: Final SNOCI state from the last completed iteration.
pub fn snoci_step(ao: &AoData, current_space: &[SCFState], noci_reference_basis: &[SCFState], input: &Input,
                  mocache: &[MOCache], tol: f64, mut wicks: Option<&mut WicksShared>) -> SNOCIState {
    time_call!(crate::timers::snoci::add_snoci_step, {
        let opts = input.snoci.as_ref().expect("snoci_step called without input.snoci.");

        let mut selected_space = current_space.to_vec();

        let mut final_state: Option<SNOCIState> = None;
        let mut candidate_pool: Option<CandidatePool> = None;

        let e0 = noci_reference_basis[0].e;
        print_snoci_header();

        for it in 0..opts.max_iter {
            // Generate matrix elements for current space and solve GEVP for the energy.
            let (hcurrent, scurrent, ecurrent, coeffs) = solve_current_space(ao, &selected_space, input, wicks.as_deref(), mocache, tol);

            if candidate_pool.is_none() {
                candidate_pool = Some(CandidatePool::new(&selected_space, input));
            }
            let pool = candidate_pool.as_mut().unwrap();
            
            // Get the current space's pseudoinverse.
            let s_ij_inv = time_call!(crate::timers::snoci::add_build_pseudoinverse, {
                let x = loewdin_x_real(&scurrent, true, tol);
                x.dot(&x)
            });
            
            // Build the current-candidate overlap and Hamiltonian and the candidate-candidate overlap.
            let mut overlaps = build_snoci_overlaps(ao, &pool.candidates, &selected_space, input, wicks.as_deref(), tol);
            let h_ai = build_candidate_current_h(ao, &pool.candidates, &selected_space, input, wicks.as_deref(), mocache, tol);
            
            // Filter out any determinants in the candidate space in redundant directions.
            let npoolpre = pool.candidates.len();
            pool.filter_candidates(&mut overlaps, &s_ij_inv, opts.gmres.metric_tol);
            let npoolpost = pool.candidates.len();
            if pool.candidates.is_empty() {
                return empty_state(ecurrent, coeffs, hcurrent, scurrent, Vec::new());
            }
           
            // Form multireference NOCI density and generalised AO Focks.
            let (da, db) = noci_density(ao, &selected_space, &coeffs, tol);
            let (fa, fb) = time_call!(crate::timers::snoci::add_build_generalised_fock, {
                form_fock_matrices(&ao.h, &ao.eri_coul, &da, &db)
            });
            // Transform Focks into MO basis for each reference.
            let fock_mocache = build_fock_mo_cache(&fa, &fb, noci_reference_basis);
            // Update the Wick's intermediates if using them.
            if input.wicks.enabled && let Some(ws) = wicks.as_deref_mut() {
                update_wicks_fock(&fa, &fb, noci_reference_basis, ws);
            }
            
            // Build the candidate-current and candidate-candidate Fock matrix, alongside the shifted Fock `M`.
            let focks = build_snoci_focks(ao, &selected_space, &pool.candidates, &fa, &fb, wicks.as_deref(), &fock_mocache, input, tol);
            let m_ab = build_candidate_m(ao, &selected_space, &pool.candidates, &overlaps, &fa, &fb, wicks.as_deref(), &fock_mocache, input, tol, ecurrent);
            
            // Construct the projected `\Omega` space linear problem to solve. 
            let m_omega_ab = build_omega_candidate_m(&overlaps, focks, m_ab, &scurrent, &s_ij_inv, ecurrent);
            let v_omega = build_omega_coupling_v(&overlaps, &hcurrent, &coeffs, &s_ij_inv, &h_ai);
            
            // Solve.
            let rhs = v_omega.mapv(|x| -x);
            let a = gmres(&m_omega_ab, &rhs, opts.gmres.max_iter, opts.gmres.res_tol);
            
            // Evaluate NOCI-PT2 energies, score and select candidates.
            let ept2 = a.x.dot(&v_omega);
            let candidate_scores: Vec<f64> = a.x.iter().zip(v_omega.iter()).map(|(&a, &v)| (a * v).abs()).collect();
            let remaining = opts.max_dim.saturating_sub(selected_space.len());
            if remaining == 0 {
                println!("SNOCI stopped at iteration {}: selected space reached max_dim ({}).", it, opts.max_dim);
                return SNOCIState {ecurrent, coeffs, hcurrent, scurrent, candidates: pool.candidates.clone(), selected: Vec::new(), candidate_scores, ept2};
            }
            let selected = select_candidates(&pool.candidates, &candidate_scores, opts.sigma, opts.max_add.min(remaining));
            let state = SNOCIState {ecurrent, coeffs, hcurrent, scurrent, candidates: pool.candidates.clone(), selected, candidate_scores, ept2};

            print_snoci_iteration(it, selected_space.len(), e0, &state, &a, npoolpre, npoolpost);

            if state.selected.is_empty() {
                println!("SNOCI stopped at iteration {}: no candidates satisfied the selection threshold ({}).", it, opts.sigma);
                return state;
            }

            if state.ept2.abs() < opts.tol {
                println!("SNOCI stopped at iteration {}: |EPT2|: {:.12} fell below tolerance {:.12}.", it, state.ept2.abs(), opts.tol);
                return state;
            }

            selected_space.extend(state.selected.iter().cloned());
            pool.update(&selected_space, &state.selected, input);
            final_state = Some(state);
        }

        println!("SNOCI stopped: Maximum iteration was reached ({}).", opts.max_iter);
        final_state.unwrap_or_else(|| {
            let (hcurrent, scurrent, ecurrent, coeffs) = solve_current_space(ao, &selected_space, input, wicks.as_deref(), mocache, tol);
            empty_state(ecurrent, coeffs, hcurrent, scurrent, Vec::new())
        })
    })
}
