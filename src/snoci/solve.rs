// snoci/solve.rs

use std::time::Instant;

use rayon::prelude::*;
use ndarray::{Array1, Array2};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::{WicksShared, WickScratchSpin};
use crate::noci::{MOCache, NOCIData, FockData, DetPair};
use crate::noci::{build_noci_s, build_noci_fock, build_noci_hs, calculate_f_pair, calculate_s_pair};
use crate::maths::{general_evp_real};
use crate::time_call;

use super::{GMRES, SNOCIOverlaps, SNOCIFocks, PT2ProjectedOperator, PT2Projection};

/// Print the GMRES iteration table header.
/// # Arguments:
/// None.
/// # Returns:
/// - `()`: Prints the GMRES iteration header to standard output.
fn print_gmres_header() {
    println!("  {}", "-".repeat(98));
    println!("  {:>8} {:>8} {:>16} {:>16} {:>16} {:>16}", "restart", "iter", "Res (est.)", "Res (true)", "Apply / s", "Elapsed / s");
}

/// Print a single GMRES iteration summary line.
/// # Arguments:
/// - `restart_id`: GMRES restart cycle index.
/// - `iter`: Total GMRES iteration index.
/// - `residual_est`: Arnoldi/Givens residual estimate for the current Krylov solve.
/// - `apply_secs`: Time spent applying the matrix-free operator on this iteration.
/// - `elapsed_secs`: Total elapsed GMRES wall time.
/// # Returns:
/// - `()`: Prints the GMRES iteration summary to standard output.
fn print_gmres_iteration(restart_id: usize, iter: usize, residual_est: f64, apply_secs: f64, elapsed_secs: f64) {
    println!("  {:>8} {:>8} {:>16.8e} {:>16} {:>16.6} {:>16.6}", restart_id, iter, residual_est, "-", apply_secs, elapsed_secs);
}

/// Print a single GMRES restart summary line using the true residual.
/// # Arguments:
/// - `restart_id`: GMRES restart cycle index.
/// - `iter`: Total GMRES iteration index after the restart.
/// - `residual_true`: True residual RMS after updating the solution.
/// - `elapsed_secs`: Total elapsed GMRES wall time.
/// # Returns:
/// - `()`: Prints the GMRES restart summary to standard output.
fn print_gmres_restart_summary(restart_id: usize, iter: usize, residual_true: f64, elapsed_secs: f64) {
    println!("  {:>8} {:>8} {:>16} {:>16.8e} {:>16} {:>16.6}", restart_id, iter, "-", residual_true, "-", elapsed_secs);
}

/// Solve a linear system using restarted GMRES with an operator callback.
/// # Arguments:
/// - `apply`: Matrix-vector product callback.
/// - `diag`: Optional diagonal used for left Jacobi preconditioning.
/// - `b`: Right-hand side vector.
/// - `restart`: Maximum Krylov subspace size before restart.
/// - `max_iter`: Maximum total GMRES iterations.
/// - `tol`: True residual RMS convergence tolerance.
/// # Returns:
/// - `GMRES`: Approximate solution vector together with final residual RMS, number of
///   iterations performed, and convergence flag.
pub(in crate::snoci) fn gmres<F>(apply: F, diag: Option<&Array1<f64>>, b: &Array1<f64>, restart: usize, max_iter: usize, tol: f64) -> GMRES
where F: Fn(&Array1<f64>) -> Array1<f64> {
    time_call!(crate::timers::snoci::add_gmres, {
        let gmres_start = Instant::now();
        let n = b.len();
        let mut x = Array1::<f64>::zeros(n);

        print_gmres_header();

        if n == 0 {
            return GMRES {x, residual_rms: 0.0, iterations: 0, converged: true};
        }

        let restart = restart.max(1).min(n);
        let rms = (n as f64).sqrt();
        let small = 1e-14_f64;
        let print_stride = 1usize;

        let dinv = diag.map(|d| {
            let dmax = d.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
            let dfloor = (1e-12 * dmax).max(1e-14);
            Array1::from_iter(d.iter().map(|&x| if x.abs() > dfloor {1.0 / x} else {1.0}))
        });

        let apply_prec = |v: &Array1<f64>| -> Array1<f64> {
            match &dinv {
                Some(d) => Array1::from_iter(v.iter().zip(d.iter()).map(|(&vi, &di)| vi * di)),
                None => v.clone(),
            }
        };

        let true_residual = |x: &Array1<f64>| -> Array1<f64> {
            let mut r = b.clone();
            r.scaled_add(-1.0, &apply(x));
            r
        };

        let mut rtrue = true_residual(&x);
        let mut residual_rms = rtrue.dot(&rtrue).sqrt() / rms;

        print_gmres_restart_summary(0, 0, residual_rms, gmres_start.elapsed().as_secs_f64());

        if residual_rms <= tol {
            return GMRES {x, residual_rms, iterations: 0, converged: true};
        }

        let mut total_iter = 0usize;
        let mut restart_id = 0usize;

        while total_iter < max_iter {
            let z = apply_prec(&rtrue);
            let beta = z.dot(&z).sqrt();

            if beta <= small {
                residual_rms = rtrue.dot(&rtrue).sqrt() / rms;
                print_gmres_restart_summary(restart_id, total_iter, residual_rms, gmres_start.elapsed().as_secs_f64());
                return GMRES {x, residual_rms, iterations: total_iter, converged: residual_rms <= tol};
            }

            let inner_max = restart.min(max_iter - total_iter);
            let mut q: Vec<Array1<f64>> = Vec::with_capacity(inner_max + 1);
            q.push(z.mapv(|zi| zi / beta));

            let mut h = Array2::<f64>::zeros((inner_max + 1, inner_max));
            let mut cs = vec![0.0; inner_max];
            let mut sn = vec![0.0; inner_max];
            let mut g = Array1::<f64>::zeros(inner_max + 1);
            g[0] = beta;

            let mut kfinal = 0usize;

            for k in 0..inner_max {
                let t_apply = Instant::now();
                let aq = apply(&q[k]);
                let apply_secs = t_apply.elapsed().as_secs_f64();
                let mut w = apply_prec(&aq);

                for j in 0..=k {
                    h[(j, k)] = q[j].dot(&w);
                    w.scaled_add(-h[(j, k)], &q[j]);
                }

                h[(k + 1, k)] = w.dot(&w).sqrt();

                if h[(k + 1, k)] > small {
                    q.push(w.mapv(|wi| wi / h[(k + 1, k)]));
                }

                for j in 0..k {
                    let temp = cs[j] * h[(j, k)] + sn[j] * h[(j + 1, k)];
                    h[(j + 1, k)] = -sn[j] * h[(j, k)] + cs[j] * h[(j + 1, k)];
                    h[(j, k)] = temp;
                }

                let hk = h[(k, k)];
                let hk1 = h[(k + 1, k)];
                let denom = (hk * hk + hk1 * hk1).sqrt();

                if denom > small {
                    cs[k] = hk / denom;
                    sn[k] = hk1 / denom;
                } else {
                    cs[k] = 1.0;
                    sn[k] = 0.0;
                }

                h[(k, k)] = cs[k] * hk + sn[k] * hk1;
                h[(k + 1, k)] = 0.0;
                g[k + 1] = -sn[k] * g[k];
                g[k] *= cs[k];

                kfinal = k + 1;

                let residual_est = g[k + 1].abs() / rms;
                let iter = total_iter + k + 1;

                if k == 0 || iter.is_multiple_of(print_stride) || residual_est <= tol {
                    print_gmres_iteration(restart_id, iter, residual_est, apply_secs, gmres_start.elapsed().as_secs_f64());
                }
            }

            let mut y = Array1::<f64>::zeros(kfinal);

            for ii in 0..kfinal {
                let i = kfinal - 1 - ii;
                let mut rhs = g[i];

                for j in (i + 1)..kfinal {
                    rhs -= h[(i, j)] * y[j];
                }

                y[i] = if h[(i, i)].abs() > small {rhs / h[(i, i)]} else {0.0};
            }

            for j in 0..kfinal {
                x.scaled_add(y[j], &q[j]);
            }

            total_iter += kfinal;

            rtrue = true_residual(&x);
            residual_rms = rtrue.dot(&rtrue).sqrt() / rms;

            print_gmres_restart_summary(restart_id, total_iter, residual_rms, gmres_start.elapsed().as_secs_f64());

            restart_id += 1;

            if residual_rms <= tol {
                return GMRES {x, residual_rms, iterations: total_iter, converged: true};
            }
        }

        GMRES {x, residual_rms, iterations: total_iter, converged: residual_rms <= tol}
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
pub(in crate::snoci) fn solve_current_space(ao: &AoData, current_space: &[SCFState], input: &Input, wicks: Option<&WicksShared>, 
                                            mocache: &[MOCache], tol: f64) -> (Array2<f64>, Array2<f64>, f64, Array1<f64>) {
    time_call!(crate::timers::snoci::add_solve_current_space, {
        let wview = wicks.as_ref().map(|ws| ws.view());
        let data = NOCIData::new(ao, current_space, input, tol, wview).withmocache(mocache);

        let (hcurrent, scurrent, _) = build_noci_hs(&data, current_space, current_space, true);
        let (evals, c) = general_evp_real(&hcurrent, &scurrent, true, tol);

        let ecurrent = evals[0];
        let coeffs = c.column(0).to_owned();

        (hcurrent, scurrent, ecurrent, coeffs)
    })
}

/// Build candidate-current overlaps.
/// # Arguments:
/// - `data`: Shared NOCI matrix-element data for candidate-left matrix elements.
/// - `candidates`: Candidate determinant space.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// # Returns:
/// - `SNOCIOverlaps`: Candidate-current and current-candidate overlap blocks.
pub(in crate::snoci) fn build_snoci_overlaps(data: &NOCIData<'_>, candidates: &[SCFState], selected_space: &[SCFState]) -> SNOCIOverlaps {
    time_call!(crate::timers::snoci::add_build_snoci_overlaps, {
        let (s_ai, _) = build_noci_s(data, candidates, selected_space, false);
        let s_ia = s_ai.t().to_owned();

        SNOCIOverlaps {s_ai, s_ia}
    })
}

/// Build current-current and candidate-current Fock matrices.
/// # Arguments:
/// - `current_data`: Shared NOCI matrix-element data for current-current matrix elements.
/// - `candidate_data`: Shared NOCI matrix-element data for candidate-left matrix elements.
/// - `fock`: Fock-specific matrix-element data.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// - `candidates`: Candidate determinant space.
/// # Returns:
/// - `SNOCIFocks`: Current-current and candidate-current Fock blocks.
pub(in crate::snoci) fn build_snoci_focks(current_data: &NOCIData<'_>, candidate_data: &NOCIData<'_>, fock: &FockData<'_>,
                                          selected_space: &[SCFState], candidates: &[SCFState]) -> SNOCIFocks {
    time_call!(crate::timers::snoci::add_build_snoci_focks, {
        let (f_ii, _) = build_noci_fock(current_data, fock, selected_space, selected_space, true);
        let (f_ai, _) = build_noci_fock(candidate_data, fock, candidates, selected_space, false);
        let f_ia = f_ai.t().to_owned();

        SNOCIFocks {f_ii, f_ai, f_ia}
    })
}

/// Build candidate-current Hamiltonian matrix elements.
/// # Arguments:
/// - `data`: Shared NOCI matrix-element data for candidate-left matrix elements.
/// - `candidates`: Candidate determinant space.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// # Returns:
/// - `Array2<f64>`: Candidate-current Hamiltonian block `H_ai`.
pub(in crate::snoci) fn build_candidate_current_h(data: &NOCIData<'_>, candidates: &[SCFState], selected_space: &[SCFState]) -> Array2<f64> {
    time_call!(crate::timers::snoci::add_build_candidate_h_ai, {
        build_noci_hs(data, candidates, selected_space, false).0
    })
}

/// Build projection contractions required for the matrix-free NOCI-PT2 operator.
/// # Arguments:
/// - `overlaps`: Candidate-current overlap blocks.
/// - `focks`: Current-current and candidate-current Fock blocks.
/// - `coeffs`: Current-space NOCI eigenvector.
/// - `e0`: Zeroth-order NOCI generalised-Fock energy.
/// # Returns:
/// - `PT2Projection`: Projection contractions used to form `M^Omega`.
pub(in crate::snoci) fn build_snoci_projection(overlaps: &SNOCIOverlaps, focks: &SNOCIFocks, coeffs: &Array1<f64>, e0: f64) -> PT2Projection {
    time_call!(crate::timers::snoci::add_build_snoci_projection, {
        // A projected candidate state is given by:
        // | \Omega_a \rangle = | \Phi_a \rangle - | \Psi_0 \rangle \langle \Psi_0 | \Phi_a \rangle,
        // where | \Psi_0 \rangle is the current NOCI state.
        // We therefore require the following contractions:
        // S_{a0} = \langle \Phi_a | \Psi_0 \rangle = \sum_i S_{ai} c_i,
        let s_a0 = overlaps.s_ai.dot(coeffs);
        // S_{0a} = \langle \Psi_0 | \Phi_a \rangle = \sum_i S_{ia} c_i,
        let s_0a = overlaps.s_ia.t().dot(coeffs);
        // F_{a0} = \langle \Phi_a | \hat F | \Psi_0 \rangle = \sum_i F_{ai} c_i,
        let f_a0 = focks.f_ai.dot(coeffs);
        // F_{0a} = \langle \Psi_0 | \hat F | \Phi_a \rangle = \sum_i F_{ia} c_i.
        let f_0a = focks.f_ia.t().dot(coeffs);

        PT2Projection {e0, s_a0, s_0a, f_a0, f_0a}
    })
}

/// Apply the unprojected candidate-candidate shifted Fock matrix `M` without materialising it.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `x`: Vector to apply `M` to.
/// # Returns:
/// - `Array1<f64>`: Matrix-vector product `M x`.
pub(in crate::snoci) fn apply_candidate_m(op: &PT2ProjectedOperator<'_, '_, '_>, x: &Array1<f64>) -> Array1<f64> {
    time_call!(crate::timers::snoci::add_apply_candidate_m, {
        let n = op.candidates.len();
        
        // Calculate y_a = \sum_b (F_{ab} - E^{(0)} S_{ab}) x_b.
        let y: Vec<f64> = (0..n).into_par_iter().map_init(WickScratchSpin::new, |scratch, a| {
            let ldet = &op.candidates[a];
            let mut ya = 0.0;

            for b in 0..n {
                let xb = x[b];

                if xb == 0.0 {
                    continue;
                }

                let gdet = &op.candidates[b];
                let f_ab = calculate_f_pair(op.data, op.fock, DetPair::new(ldet, gdet), Some(scratch));
                let s_ab = calculate_s_pair(op.data, DetPair::new(ldet, gdet), Some(scratch));
                
                ya += (f_ab - op.projection.e0 * s_ab) * xb;
            }

            ya
        }).collect();
        Array1::from_vec(y)
    })
}

/// Apply the projected NOCI-PT2 shifted Fock matrix `M^Omega` without materialising it.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `x`: Vector to apply `M^Omega` to.
/// # Returns:
/// - `Array1<f64>`: Matrix-vector product `M^Omega x`.
pub(in crate::snoci) fn apply_omega_m(op: &PT2ProjectedOperator<'_, '_, '_>, x: &Array1<f64>) -> Array1<f64> {
    time_call!(crate::timers::snoci::add_apply_omega_m, { 
        let p = op.projection;
        
        // Projected matrix elements are defined as:
        // M_{ab}^\Omega = \langle \Omega_a | \hat F - E^{(0)} | \Omega_b \rangle.
        // Expanding the projected states gives:
        // M_{ab}^\Omega = M_{ab} - F_{a0} S_{0b} - S_{a0} F_{0b} + 2 E^{(0)} S_{a0} S_{0b},
        // where M_{ab} = F_{ab} - E^{(0)} S_{ab}.
        // The following contractions apply the final three terms to `x`.
        let sx = p.s_0a.dot(x);
        let fx = p.f_0a.dot(x);

        // Get the action of unprojected M_{ab} = F_{ab} - E^{(0)} S_{ab} on a vector `x`.
        let mut y = apply_candidate_m(op, x);

        for a in 0..y.len() {
            y[a] += -p.f_a0[a] * sx - p.s_a0[a] * fx + 2.0 * p.e0 * p.s_a0[a] * sx;
        }
        y
    })
}

/// Build the diagonal of the projected NOCI-PT2 shifted Fock matrix without materialising it.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// # Returns:
/// - `Array1<f64>`: Diagonal of `M^\Omega`.
pub(in crate::snoci) fn build_omega_m_diag(op: &PT2ProjectedOperator<'_, '_, '_>) -> Array1<f64> {
    time_call!(crate::timers::snoci::add_build_omega_m_diag, {
        let n = op.candidates.len();
        let p = op.projection;

        let d: Vec<f64> = (0..n).into_par_iter().map_init(WickScratchSpin::new, |scratch, a| {
            let det = &op.candidates[a];
            let pair = DetPair::new(det, det);

            let f_aa = calculate_f_pair(op.data, op.fock, pair, Some(scratch));
            let s_aa = calculate_s_pair(op.data, DetPair::new(det, det), Some(scratch));

            f_aa - p.e0 * s_aa - p.f_a0[a] * p.s_0a[a] - p.s_a0[a] * p.f_0a[a] + 2.0 * p.e0 * p.s_a0[a] * p.s_0a[a]
        }).collect();

        Array1::from_vec(d)
    })
}

/// Build the unprojected candidate-current coupling vector `V`.
/// # Arguments:
/// - `h_ai`: Candidate-current Hamiltonian block.
/// - `coeffs`: Current-space ground-state eigenvector.
/// # Returns:
/// - `Array1<f64>`: Unprojected candidate coupling vector.
pub(in crate::snoci) fn build_candidate_v(h_ai: &Array2<f64>, coeffs: &Array1<f64>) -> Array1<f64> {
    time_call!(crate::timers::snoci::add_build_candidate_v, {
        // V_a = \langle \Phi_a | \hat H | \Psi_0 \rangle = \sum_i H_{ai} c_i.
        h_ai.dot(coeffs)
    })
}

/// Build the NOCI-PT2 state-projected coupling vector.
/// # Arguments:
/// - `s_ai`: Candidate-current overlap block.
/// - `coeffs`: Current-space NOCI eigenvector.
/// - `v_a`: Unprojected candidate-current Hamiltonian coupling vector.
/// - `ecurrent`: Current selected-space NOCI energy.
/// # Returns:
/// - `Array1<f64>`: Omega-projected coupling vector.
pub(in crate::snoci) fn build_omega_v(s_ai: &Array2<f64>, coeffs: &Array1<f64>, v_a: Array1<f64>, ecurrent: f64) -> Array1<f64> {
    time_call!(crate::timers::snoci::add_build_omega_v, {
        // V_a^\Omega = \langle \Omega_a | \hat H | \Psi_0 \rangle.
        // Expanding | \Omega_a \rangle gives:
        // V_a^\Omega = V_a - E_\mathrm{NOCI} S_{a0}.
        let s_a0 = s_ai.dot(coeffs);
        v_a - s_a0.mapv(|x| ecurrent * x)
    })
}

/// Select the highest-scoring candidates above the selection threshold.
/// # Arguments:
/// - `candidates`: Candidate determinants currently present in the pool.
/// - `candidate_scores`: Candidate importance scores.
/// - `sigma`: Selection threshold.
/// - `max_add`: Maximum number of candidates to add.
/// # Returns:
/// - `Vec<SCFState>`: Selected candidates sorted by decreasing score.
pub(in crate::snoci) fn select_candidates(candidates: &[SCFState], candidate_scores: &[f64], sigma: f64, max_add: usize) -> Vec<SCFState> {
    let mut ranked: Vec<(SCFState, f64)> = candidates.iter().cloned()
        .zip(candidate_scores.iter().copied())
        .filter(|(_, score)| *score > sigma)
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.into_iter().take(max_add).map(|(state, _)| state).collect()
}
