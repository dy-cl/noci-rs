// snoci/step.rs

use ndarray::{Array1, Array2};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::WicksShared;
use crate::noci::{MOCache, NOCIData, FockData};
use super::{SNOCIState, CandidatePool, GMRES, PT2ProjectedOperator};
use crate::time_call;

use crate::noci::{noci_density, build_fock_mo_cache, update_wicks_fock};
use crate::scf::form_fock_matrices;
use super::{gmres, solve_current_space, build_snoci_overlaps, build_snoci_focks, build_candidate_m_diag, build_snoci_projection, 
            apply_omega_m, build_candidate_v, build_omega_v, select_candidates, build_candidate_current_h, build_preconditioner,
            build_candidate_m};

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

/// Print the start of a SNOCI iteration block.
/// # Arguments:
/// - `it`: SNOCI iteration index.
/// - `n_current`: Number of determinants in the current selected space.
/// - `npoolpre`: Candidate-pool size before projected-norm filter.
/// - `npoolpost`: Candidate-pool size after projected-norm filter.
/// # Returns:
/// - `()`: Prints the SNOCI iteration block header to standard output.
fn print_snoci_iteration_start(it: usize, n_current: usize, npoolpre: usize, npoolpost: usize) {
    println!("SNOCI iteration: {}", it);
    println!("  NCurr:     {}", n_current);
    println!("  NCand (R): {}", npoolpre);
    println!("  NCand:     {}", npoolpost);
}

/// Print the status message for building the cached shifted Fock matrix.
/// # Arguments:
/// - `n`: Number of candidates.
/// # Returns:
/// - `()`: Prints the shifted Fock build message to standard output.
fn print_build_candidate_m(n: usize) {
    let nelem = n * (n + 1) / 2;
    let mib = nelem as f64 * std::mem::size_of::<f64>() as f64 / 1024.0 / 1024.0;
    println!("  Building upper triangle shifted Fock matrix ({} elements, {:.3} MiB)...", nelem, mib);
}

/// Print the SNOCI result for a completed iteration.
/// # Arguments:
/// - `it`: SNOCI iteration index.
/// - `n_current`: Number of determinants in the current selected space.
/// - `e0`: RHF reference energy used to define the correlation energy.
/// - `state`: SNOCI state for the current iteration.
/// - `gmres`: GMRES solve information for the current iteration.
/// # Returns:
/// - `()`: Prints the SNOCI iteration result to standard output.
fn print_snoci_iteration_result(it: usize, n_current: usize, e0: f64, state: &SNOCIState, gmres: &GMRES) {
    println!();
    println!("  SNOCI result");
    println!("  {}", "-".repeat(98));
    println!("  Iteration:          {}", it);
    println!("  NCurr:              {}", n_current);
    println!("  NSelect:            {}", state.selected.len());
    println!("  E:                  {:.12}", state.ecurrent);
    println!("  Ecorr:              {:.12}", state.ecurrent - e0);
    println!("  EPT2:               {:.12}", state.ept2);
    println!("  E + EPT2:           {:.12}", state.ecurrent + state.ept2);
    println!("  GMRES residual:     {:.12}", gmres.residual_rms);
    println!("  GMRES iterations:   {}", gmres.iterations);
    println!("  GMRES converged:    {}", gmres.converged);
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

        for it in 0..opts.max_iter {
            // Generate matrix elements for current space and solve GEVP for the energy.
            let (hcurrent, scurrent, ecurrent, coeffs) = solve_current_space(ao, &selected_space, input, wicks.as_deref(), mocache, tol);

            if candidate_pool.is_none() {
                candidate_pool = Some(CandidatePool::new(&selected_space, input));
            }
            let pool = candidate_pool.as_mut().unwrap();

            let wview = wicks.as_ref().map(|ws| ws.view());
            let candidate_data = NOCIData::new(ao, &pool.candidates, input, tol, wview).withmocache(mocache);

            // Build the current-candidate overlap and its transpose.
            let overlaps = build_snoci_overlaps(&candidate_data, &pool.candidates, &selected_space);
            
            // Filter out any determinants in the candidate space in redundant directions.
            let npoolpre = pool.candidates.len();
            let npoolpost = pool.candidates.len();
            if pool.candidates.is_empty() {
                return empty_state(ecurrent, coeffs, hcurrent, scurrent, Vec::new());
            }
            
            let h_ai = build_candidate_current_h(&candidate_data, &pool.candidates, &selected_space);
           
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
            let wview = wicks.as_ref().map(|ws| ws.view());
            let candidate_data = NOCIData::new(ao, &pool.candidates, input, tol, wview).withmocache(mocache);
            let current_data = NOCIData::new(ao, &selected_space, input, tol, wview).withmocache(mocache);
            let fock = FockData::new(&fock_mocache, &fa, &fb);
            let focks = build_snoci_focks(&current_data, &candidate_data, &fock, &selected_space, &pool.candidates); 

            let e0 = coeffs.dot(&focks.f_ii.dot(&coeffs));
            let projection = build_snoci_projection(&overlaps, &focks, &coeffs, e0);

            let v_a = build_candidate_v(&h_ai, &coeffs);
            let v_omega = build_omega_v(&overlaps.s_ai, &coeffs, v_a, ecurrent);

            let op = PT2ProjectedOperator {data: &candidate_data, fock: &fock, candidates: &pool.candidates, projection: &projection};

            if it > 0 {
                println!("{}", "=".repeat(100));
            }

            print_snoci_iteration_start(it, selected_space.len(), npoolpre, npoolpost);

            let m = if opts.gmres.full_m {
                print_build_candidate_m(op.candidates.len());
                Some(build_candidate_m(&op))
            } else {
                None
            };

            let m_diag = build_candidate_m_diag(&op, m.as_deref());
            let prec = build_preconditioner(&m_diag, op.projection);
            let rhs = v_omega.mapv(|x| -x);

            let a = gmres(|x| apply_omega_m(&op, x, m.as_deref()), |x| prec.apply(x), &rhs, &opts.gmres);

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

            print_snoci_iteration_result(it, selected_space.len(), noci_reference_basis[0].e, &state, &a);
            
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
