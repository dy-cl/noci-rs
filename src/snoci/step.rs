// snoci/step.rs

use mpi::topology::Communicator;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::{CandidatePool, PT2ProjectedOperator, SNOCIPT2Result, SNOCIState};
use crate::noci::{FockData, NOCIData, NOCIScalar};
use crate::nonorthogonalwicks::WicksShared;
use crate::time_call;
use crate::{DetState, PostSCFData, input::Input};

use super::{
    apply_shifted_omega_m, apply_shifted_omega_m_mpi, build_candidate_current_h, build_candidate_m,
    build_candidate_m_diag, build_candidate_s_diag, build_candidate_v, build_omega_v,
    build_preconditioner, build_snoci_focks, build_snoci_overlaps, build_snoci_projection, gmres,
    select_candidates, solve_current_space,
};
use crate::noci::{build_fock_mo_cache, noci_density, update_wicks_fock};
use crate::scf::fock;

/// Return the real component of a scalar used for printed and stored energies.
/// # Arguments:
/// - `z`: Scalar value.
/// # Returns:
/// - `f64`: Real component.
fn scalar_real<T: NOCIScalar + Into<Complex64>>(z: T) -> f64 {
    z.into().re
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
fn empty_state<T: NOCIScalar>(
    ecurrent: f64,
    coeffs: Array1<T>,
    hcurrent: Array2<T>,
    scurrent: Array2<T>,
    candidates: Vec<DetState<T>>,
) -> SNOCIState<T> {
    SNOCIState {
        ecurrent,
        coeffs,
        hcurrent,
        scurrent,
        candidates,
        selected: Vec::new(),
        pt2: Vec::new(),
    }
}

/// Print the start of a SNOCI iteration block.
/// # Arguments:
/// - `it`: SNOCI iteration index.
/// - `n_current`: Number of determinants in the current selected space.
/// - `npoolpre`: Candidate-pool size before projected-norm filter.
/// - `npoolpost`: Candidate-pool size after projected-norm filter.
/// # Returns:
/// - `()`: Prints the SNOCI iteration block header to standard output.
fn print_snoci_iteration_start(
    it: usize,
    n_current: usize,
    npoolpre: usize,
    npoolpost: usize,
) {
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
fn print_build_candidate_m<T: NOCIScalar>(n: usize) {
    let nelem = n * (n + 1) / 2;
    let mib = nelem as f64 * std::mem::size_of::<T>() as f64 / 1024.0 / 1024.0;
    println!(
        "  Building upper triangle shifted Fock matrix ({} elements, {:.3} MiB)...",
        nelem, mib
    );
}

/// Print the SNOCI result for a completed iteration.
/// # Arguments:
/// - `it`: SNOCI iteration index.
/// - `n_current`: Number of determinants in the current selected space.
/// - `e0`: RHF reference energy used to define the correlation energy.
/// - `state`: SNOCI state for the current iteration.
/// # Returns:
/// - `()`: Prints the SNOCI iteration result to standard output.
fn print_snoci_iteration_result<T: NOCIScalar>(
    it: usize,
    n_current: usize,
    e0: f64,
    state: &SNOCIState<T>,
) {
    println!();
    println!("  SNOCI result");
    println!("  {}", "-".repeat(98));
    println!("  Iteration:          {}", it);
    println!("  NCurr:              {}", n_current);
    println!("  NSelect:            {}", state.selected.len());
    println!("  E:                  {:.12}", state.ecurrent);
    println!("  Ecorr:              {:.12}", state.ecurrent - e0);

    for r in &state.pt2 {
        println!();
        println!("  NOCI-PT2 result");
        println!("  Imag shift:         {:.12}", r.imag_shift);
        println!("  EPT2:               {:.12}", r.ept2);
        println!("  E + EPT2:           {:.12}", state.ecurrent + r.ept2);
        println!("  Max |a_i|:          {:.12}", r.max_abs_a);
        println!("  Max |v_i|:          {:.12}", r.max_abs_v);
        println!("  Max |a_i v_i|:      {:.12}", r.max_abs_av);
        println!("  GMRES residual:     {:.12}", r.gmres_residual);
        println!("  GMRES iterations:   {}", r.gmres_iterations);
        println!("  GMRES converged:    {}", r.gmres_converged);
        println!("  {}", "-".repeat(98));
    }
}

/// Perform selected NOCI with selection from excitations of the current space.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `current_space`: Current selected nonorthogonal determinant space.
/// - `input`: User-defined input options.
/// - `wicks`: Mutable Wick's intermediates as we need to update Fock intermediates.
/// - `world`: MPI communicator used to distribute NOCI-PT2 matrix-vector products.
/// # Returns:
/// - `SNOCIState`: Final SNOCI state from the last completed iteration.
pub fn snoci_step<T>(
    post: &PostSCFData<'_, T>,
    current_space: &[DetState<T>],
    input: &Input,
    mut wicks: Option<&mut WicksShared<T>>,
    world: &impl Communicator,
) -> SNOCIState<T>
where
    T: NOCIScalar + Into<Complex64>,
{
    time_call!(crate::timers::snoci::add_snoci_step, {
        let opts = input
            .snoci
            .as_ref()
            .expect("snoci_step called without input.snoci.");

        let mut selected_space = current_space.to_vec();

        let mut final_state: Option<SNOCIState<T>> = None;
        let mut candidate_pool: Option<CandidatePool<T>> = None;

        for it in 0..opts.max_iter {
            // Generate matrix elements for current space and solve GEVP for the energy.
            let (hcurrent, scurrent, ecurrent, coeffs) = solve_current_space(
                post.ao,
                &selected_space,
                input,
                wicks.as_deref(),
                post.mocache,
                post.tol,
            );

            if candidate_pool.is_none() {
                candidate_pool = Some(CandidatePool::new(&selected_space, input));
            }
            let pool = candidate_pool.as_mut().unwrap();

            let wview = wicks.as_ref().map(|ws| ws.view());
            let candidate_data = NOCIData::new(post.ao, &pool.candidates, input, post.tol, wview)
                .withmocache(post.mocache);

            // Build the current-candidate overlap and its transpose.
            let overlaps = build_snoci_overlaps(&candidate_data, &pool.candidates, &selected_space);

            // Filter out any determinants in the candidate space in redundant directions.
            let npoolpre = pool.candidates.len();
            let npoolpost = pool.candidates.len();
            if pool.candidates.is_empty() {
                return empty_state(ecurrent, coeffs, hcurrent, scurrent, Vec::new());
            }

            let h_ai =
                build_candidate_current_h(&candidate_data, &pool.candidates, &selected_space);

            // Form multireference NOCI density and generalised AO Focks.
            let (da, db) = noci_density(post.ao, &selected_space, &coeffs, post.tol);
            let (fa, fb) = time_call!(crate::timers::snoci::add_build_generalised_fock, {
                fock(&post.ao.h, &post.ao.eri_coul, &da, &db)
            });
            // Transform Focks into MO basis for each reference.
            let fock_mocache =
                build_fock_mo_cache(&fa, &fb, post.noci_reference_basis, &post.ao.s, post.tol);
            // Update the Wick's intermediates if using them.
            if input.wicks.enabled
                && let Some(ws) = wicks.as_deref_mut()
            {
                update_wicks_fock(
                    &fa,
                    &fb,
                    post.noci_reference_basis,
                    &post.ao.s,
                    post.tol,
                    ws,
                );
            }

            // Build the candidate-current and candidate-candidate Fock matrix, alongside the shifted Fock `M`.
            let wview = wicks.as_ref().map(|ws| ws.view());
            let candidate_data = NOCIData::new(post.ao, &pool.candidates, input, post.tol, wview)
                .withmocache(post.mocache);
            let current_data = NOCIData::new(post.ao, &selected_space, input, post.tol, wview)
                .withmocache(post.mocache);
            let fock = FockData::new(&fock_mocache, &fa, &fb);
            let focks = build_snoci_focks(
                &current_data,
                &candidate_data,
                &fock,
                &selected_space,
                &pool.candidates,
            );

            let fc = focks.f_ii.dot(&coeffs);
            let e0_z = coeffs
                .iter()
                .zip(fc.iter())
                .fold(T::from_real(0.0), |acc, (&c, &x)| acc + c.conj() * x);
            let e0 = scalar_real(e0_z);
            let projection = build_snoci_projection(&overlaps, &focks, &coeffs, e0);

            let v_a = build_candidate_v(&h_ai, &coeffs);
            let v_omega = build_omega_v(&overlaps.s_ai, &coeffs, v_a, ecurrent);

            let op = PT2ProjectedOperator {
                data: &candidate_data,
                fock: &fock,
                candidates: &pool.candidates,
                projection: &projection,
            };

            if it > 0 && world.rank() == 0 {
                println!("{}", "=".repeat(100));
            }

            if world.rank() == 0 {
                print_snoci_iteration_start(it, selected_space.len(), npoolpre, npoolpost);
            }

            let m = if opts.gmres.full_m {
                if world.rank() == 0 {
                    print_build_candidate_m::<T>(op.candidates.len());
                }
                Some(build_candidate_m(&op))
            } else {
                None
            };

            let m_diag = build_candidate_m_diag(&op, m.as_deref());
            let shifts = if opts.imag_shifts.is_empty() {
                vec![0.0]
            } else {
                opts.imag_shifts.clone()
            };
            let s_diag = shifts
                .iter()
                .any(|&imag_shift| imag_shift != 0.0)
                .then(|| build_candidate_s_diag(&op));
            let rhs = v_omega.mapv(|x| -x);

            // Evaluate NOCI-PT2 energies, scores and diagnostics for each imaginary shift.
            let mut pt2 = Vec::new();
            for &imag_shift in &shifts {
                let prec = build_preconditioner(
                    &m_diag,
                    s_diag.as_ref(),
                    op.projection,
                    opts.preconditioner,
                    imag_shift,
                );

                let a = gmres(
                    |x| {
                        if world.size() > 1 {
                            apply_shifted_omega_m_mpi(&op, x, m.as_deref(), world, imag_shift)
                        } else {
                            apply_shifted_omega_m(&op, x, m.as_deref(), imag_shift)
                        }
                    },
                    |x| prec.apply(x),
                    &rhs,
                    &opts.gmres,
                    world,
                );

                let ma = if world.size() > 1 {
                    apply_shifted_omega_m_mpi(&op, &a.x, m.as_deref(), world, imag_shift)
                } else {
                    apply_shifted_omega_m(&op, &a.x, m.as_deref(), imag_shift)
                };

                let ama =
                    a.x.iter()
                        .zip(ma.iter())
                        .fold(T::from_real(0.0), |acc, (&aa, &maa)| acc + aa.conj() * maa);
                let av =
                    a.x.iter()
                        .zip(v_omega.iter())
                        .fold(T::from_real(0.0), |acc, (&aa, &v)| acc + aa.conj() * v);
                let va = v_omega
                    .iter()
                    .zip(a.x.iter())
                    .fold(T::from_real(0.0), |acc, (&v, &aa)| acc + v.conj() * aa);
                let ept2 = scalar_real(ama + av + va);

                let candidate_scores: Vec<f64> =
                    a.x.iter()
                        .zip(v_omega.iter())
                        .map(|(&a, &v)| (a * v).abs())
                        .collect();

                let max_abs_a = a.x.iter().map(|x| x.abs()).fold(0.0, f64::max);
                let max_abs_v = v_omega.iter().map(|x| x.abs()).fold(0.0, f64::max);
                let max_abs_av = candidate_scores.iter().copied().fold(0.0, f64::max);

                pt2.push(SNOCIPT2Result {
                    imag_shift,
                    ept2,
                    candidate_scores,
                    max_abs_a,
                    max_abs_v,
                    max_abs_av,
                    gmres_residual: a.residual_rms,
                    gmres_iterations: a.iterations,
                    gmres_converged: a.converged,
                });
            }

            let remaining = opts.max_dim.saturating_sub(selected_space.len());
            if remaining == 0 && world.rank() == 0 {
                println!(
                    "SNOCI stopped at iteration {}: selected space reached max_dim ({}).",
                    it, opts.max_dim
                );
                return SNOCIState {
                    ecurrent,
                    coeffs,
                    hcurrent,
                    scurrent,
                    candidates: pool.candidates.clone(),
                    selected: Vec::new(),
                    pt2,
                };
            }

            // Use the final imaginary shift in the input list as the main shift for selection.
            let main_pt2 = pt2
                .last()
                .expect("At least one NOCI-PT2 shift must be evaluated.");

            let selected = select_candidates(
                &pool.candidates,
                &main_pt2.candidate_scores,
                opts.sigma,
                opts.max_add.min(remaining),
            );
            let state = SNOCIState {
                ecurrent,
                coeffs,
                hcurrent,
                scurrent,
                candidates: pool.candidates.clone(),
                selected,
                pt2,
            };

            if world.rank() == 0 {
                print_snoci_iteration_result(
                    it,
                    selected_space.len(),
                    scalar_real(post.noci_reference_basis[0].e),
                    &state,
                );
            }

            if state.selected.is_empty() && world.rank() == 0 {
                println!(
                    "SNOCI stopped at iteration {}: no candidates satisfied the selection threshold ({}).",
                    it, opts.sigma
                );
                return state;
            }

            let main_pt2 = state
                .pt2
                .last()
                .expect("At least one NOCI-PT2 shift must be evaluated.");

            if main_pt2.ept2.abs() < opts.tol {
                if world.rank() == 0 {
                    println!(
                        "SNOCI stopped at iteration {}: |EPT2|: {:.12} fell below tolerance {:.12}.",
                        it,
                        main_pt2.ept2.abs(),
                        opts.tol
                    );
                }
                return state;
            }

            selected_space.extend(state.selected.iter().cloned());
            pool.update(&selected_space, &state.selected, input);
            final_state = Some(state);
        }

        if world.rank() == 0 {
            println!(
                "SNOCI stopped: Maximum iteration was reached ({}).",
                opts.max_iter
            );
        }

        final_state.unwrap_or_else(|| {
            let (hcurrent, scurrent, ecurrent, coeffs) = solve_current_space(
                post.ao,
                &selected_space,
                input,
                wicks.as_deref(),
                post.mocache,
                post.tol,
            );
            empty_state(ecurrent, coeffs, hcurrent, scurrent, Vec::new())
        })
    })
}
