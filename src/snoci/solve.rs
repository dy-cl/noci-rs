// snoci/solve.rs

use ndarray::{Array1, Array2};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::WicksShared;
use crate::noci::{MOCache, FockMOCache, NOCIData, FockData};
use crate::noci::{build_noci_s, build_noci_fock, build_noci_hs};
use crate::maths::{general_evp_real, parallel_matvec_real};
use crate::time_call;

use super::{GMRES, SNOCIOverlaps, SNOCIFocks};

/// Solve a linear system using unrestarted GMRES.
/// # Arguments:
/// - `m`: Matrix defining the linear system `Mx = b`.
/// - `b`: Right-hand side vector.
/// - `max_iter`: Maximum number of GMRES iterations.
/// - `tol`: Residual RMS convergence tolerance.
/// # Returns:
/// - `GMRES`: Approximate solution vector together with final residual RMS, number of
///   iterations performed, and convergence flag.
pub(in crate::snoci) fn gmres(m: &Array2<f64>, b: &Array1<f64>, max_iter: usize, tol: f64) -> GMRES {
    time_call!(crate::timers::snoci::add_gmres, {
        let n = b.len();
        let mut x = Array1::<f64>::zeros(n);

        if n == 0 {
            return GMRES {x, residual_rms: 0.0, iterations: 0, converged: true};
        }

        let r = b.clone();
        let beta = r.dot(&r).sqrt();
        let mut final_residual_rms = beta / (n as f64).sqrt();

        if final_residual_rms < tol {
            return GMRES {x, residual_rms: final_residual_rms, iterations: 0, converged: true};
        }

        let mut q: Vec<Array1<f64>> = Vec::with_capacity(max_iter + 1);
        q.push(r.mapv(|ri| ri / beta));

        let mut h = Array2::<f64>::zeros((max_iter + 1, max_iter));
        let mut cs = vec![0.0; max_iter];
        let mut sn = vec![0.0; max_iter];
        let mut g = Array1::<f64>::zeros(max_iter + 1);
        g[0] = beta;

        let mut kfinal = 0usize;

        for k in 0..max_iter {
            let mut w = parallel_matvec_real(m, &q[k]);

            for j in 0..=k {
                h[(j, k)] = q[j].dot(&w);
                w.scaled_add(-h[(j, k)], &q[j]);
            }

            h[(k + 1, k)] = w.dot(&w).sqrt();

            if h[(k + 1, k)] > tol {
                q.push(w.mapv(|x| x / h[(k + 1, k)]));
            }

            for j in 0..k {
                let temp = cs[j] * h[(j, k)] + sn[j] * h[(j + 1, k)];
                h[(j + 1, k)] = -sn[j] * h[(j, k)] + cs[j] * h[(j + 1, k)];
                h[(j, k)] = temp;
            }

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

            h[(k, k)] = cs[k] * hk + sn[k] * hk1;
            h[(k + 1, k)] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] *= cs[k];

            kfinal = k + 1;

            let res_rms = g[k + 1].abs() / (n as f64).sqrt();
            if res_rms <= tol {
                break;
            }
        }

        let mut y = Array1::<f64>::zeros(kfinal);
        for ii in 0..kfinal {
            let i = kfinal - 1 - ii;
            let mut rhs = g[i];
            for j in (i + 1)..kfinal {
                rhs -= h[(i, j)] * y[j];
            }
            y[i] = rhs / h[(i, i)];
        }

        for j in 0..kfinal {
            x.scaled_add(y[j], &q[j]);
        }

        let mut rtrue = b.clone();
        rtrue.scaled_add(-1.0, &parallel_matvec_real(m, &x));
        final_residual_rms = rtrue.dot(&rtrue).sqrt() / (n as f64).sqrt();

        GMRES {x, residual_rms: final_residual_rms, iterations: kfinal, converged: final_residual_rms < tol}
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

/// Build candidate-candidate and candidate-current overlaps.
/// # Arguments:
/// - `ao`: AO integrals and other system data.
/// - `candidates`: Candidate determinant space.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// - `input`: User-defined input options.
/// - `wicks`: Optional shared Wick's intermediates.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns:
/// - `SNOCIOverlaps`: SNOCI overlap matrices.
pub(in crate::snoci) fn build_snoci_overlaps(ao: &AoData, candidates: &[SCFState], selected_space: &[SCFState], input: &Input, 
                                             wicks: Option<&WicksShared>, tol: f64) -> SNOCIOverlaps {
    let wview = wicks.as_ref().map(|ws| ws.view());
    let data = NOCIData::new(ao, candidates, input, tol, wview);

    let (s_ab, _) = build_noci_s(&data, candidates, candidates, true);
    let (s_ai, _) = build_noci_s(&data, candidates, selected_space, false);
    let s_ia = s_ai.t().to_owned();

    SNOCIOverlaps {s_ab, s_ai, s_ia}
}

/// Build current-current and candidate-current Fock matrices.
/// # Arguments:
/// - `ao`: AO integrals and other system data.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// - `candidates`: Candidate determinant space.
/// - `fa`: Alpha-spin generalised Fock matrix.
/// - `fb`: Beta-spin generalised Fock matrix.
/// - `wicks`: Optional shared Wick's intermediates.
/// - `fock_mocache`: MO-basis Fock integral caches.
/// - `input`: User-defined input options.
/// - `tol`: Tolerance for whether a number is zero.
/// # Returns:
/// - `SNOCIFocks`: SNOCI Fock matrices.
pub(in crate::snoci) fn build_snoci_focks(ao: &AoData, selected_space: &[SCFState], candidates: &[SCFState], fa: &Array2<f64>, fb: &Array2<f64>,
                                          wicks: Option<&WicksShared>, fock_mocache: &[FockMOCache], input: &Input, tol: f64) -> SNOCIFocks {
    let wview = wicks.as_ref().map(|ws| ws.view());
    let data = NOCIData::new(ao, selected_space, input, tol, wview);
    let fock = FockData::new(fock_mocache, fa, fb);

    let (f_ii, _) = build_noci_fock(&data, &fock, selected_space, selected_space, true);
    let (f_ai, _) = build_noci_fock(&data, &fock, candidates, selected_space, false);
    let f_ia = f_ai.t().to_owned();

    SNOCIFocks {f_ii, f_ai, f_ia}
}

/// Build candidate-current Hamiltonian matrix elements.
/// # Arguments:
/// - `ao`: AO integrals and other system data.
/// - `candidates`: Candidate determinant space.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// - `input`: User-defined input options.
/// - `wicks`: Optional shared Wick's intermediates.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `tol`: Tolerance for whether a number is zero.
/// # Returns:
/// - `Array2<f64>`: Candidate-current Hamiltonian block `H_ai`.
pub(in crate::snoci) fn build_candidate_current_h(ao: &AoData, candidates: &[SCFState], selected_space: &[SCFState], input: &Input, 
                                                  wicks: Option<&WicksShared>, mocache: &[MOCache], tol: f64) -> Array2<f64> {
    time_call!(crate::timers::snoci::add_build_candidate_h_ai, {
        let wview = wicks.as_ref().map(|ws| ws.view());
        let data = NOCIData::new(ao, candidates, input, tol, wview).withmocache(mocache);
        build_noci_hs(&data, candidates, selected_space, false).0
    })
}

/// Build the unprojected shifted candidate-candidate matrix `M`.
/// # Arguments:
/// - `ao`: AO integrals and other system data.
/// - `candidates`: Candidate determinant space.
/// - `overlaps`: Candidate overlap blocks.
/// - `fa`: Alpha-spin generalised Fock matrix.
/// - `fb`: Beta-spin generalised Fock matrix.
/// - `wicks`: Optional shared Wick's intermediates.
/// - `fock_mocache`: MO-basis Fock integral caches.
/// - `input`: User-defined input options.
/// - `tol`: Tolerance for whether a number is zero.
/// - `ecurrent`: Current NOCI energy used for the shift.
/// # Returns:
/// - `Array2<f64>`: Unprojected shifted candidate-candidate matrix `F_ab - E_0 S_ab`.
pub(in crate::snoci) fn build_candidate_m(ao: &AoData, candidates: &[SCFState], overlaps: &SNOCIOverlaps, fa: &Array2<f64>, fb: &Array2<f64>,
                                          wicks: Option<&WicksShared>, fock_mocache: &[FockMOCache], input: &Input, tol: f64, 
                                          ecurrent: f64) -> Array2<f64> {
    let wview = wicks.as_ref().map(|ws| ws.view());
    let data = NOCIData::new(ao, candidates, input, tol, wview);
    let fock = FockData::new(fock_mocache, fa, fb);

    let (f_ab, _) = build_noci_fock(&data, &fock, candidates, candidates, true);
    f_ab - overlaps.s_ab.mapv(|s| ecurrent * s)
}

/// Build the NOCI-PT2 state-projected shifted candidate-candidate matrix `M`.
/// # Arguments:
/// - `overlaps`: Candidate-current and candidate-candidate overlap blocks.
/// - `focks`: Current-current and candidate-current Fock blocks.
/// - `m_ab`: Unprojected shifted candidate-candidate matrix.
/// - `coeffs`: Current NOCI eigenvector.
/// - `ecurrent`: Current NOCI energy.
/// # Returns:
/// - `Array2<f64>`: State-projected shifted candidate-candidate matrix.
pub(in crate::snoci) fn build_omega_m(overlaps: &SNOCIOverlaps, focks: SNOCIFocks, m_ab: Array2<f64>, coeffs: &Array1<f64>, ecurrent: f64) -> Array2<f64> {
    let s_a0 = overlaps.s_ai.dot(coeffs);
    let s_0a = overlaps.s_ia.t().dot(coeffs);

    let f_a0 = focks.f_ai.dot(coeffs);
    let f_0a = focks.f_ia.t().dot(coeffs);

    let mut out = m_ab;
    
    // M_{ab}^\Omega = \langle \Omega_a | \hat F - E_0 | \Omega_b \rangle.
    // | \Omega_a \rangle = | \Phi_a \rangle - | \Psi_0 \rangle \langle \Psi_0 | \Phi_a \rangle,
    // where | \Psi_0 \rangle is the correct NOCI wavefunction. M_{ab}^\Omega is therefore
    // M_{ab}^\Omega = (F_{ab} - E_0 S_{ab}) - F_{a0} S_{0b} - S_{a0} F_{0b} + S_{a0} (F_{00} + E_0) S_{0b}.
    for a in 0..out.nrows() {
        for b in 0..out.ncols() {
            out[(a, b)] += -f_a0[a] * s_0a[b] - s_a0[a] * f_0a[b] + 2.0 * ecurrent * s_a0[a] * s_0a[b];
        }
    }
    out
}

/// Build the unprojected candidate-current coupling vector `V`.
/// # Arguments:
/// - `h_ai`: Candidate-current Hamiltonian block.
/// - `coeffs`: Current-space ground-state eigenvector.
/// # Returns:
/// - `Array1<f64>`: Unprojected candidate coupling vector.
pub(in crate::snoci) fn build_candidate_v(h_ai: &Array2<f64>, coeffs: &Array1<f64>) -> Array1<f64> {
    h_ai.dot(coeffs)
}

/// Build the NOCI-PT2 state-projected coupling vector `V`.
/// # Arguments:
/// - `overlaps`: Candidate overlap blocks.
/// - `coeffs`: Current-space ground-state eigenvector.
/// - `v_a`: Unprojected candidate coupling vector.
/// - `ecurrent`: Current selected-space NOCI energy.
/// # Returns:
/// - `Array1<f64>`: Omega-projected coupling vector.
pub(in crate::snoci) fn build_omega_v(overlaps: &SNOCIOverlaps, coeffs: &Array1<f64>, v_a: Array1<f64>, ecurrent: f64) -> Array1<f64> {
    // V_a = \langle \Omega_a | \hat H | \Psi_0 \rangle which when expanding as
    // | \Omega_a \rangle = | \Phi_a \rangle - | \Psi_0 \rangle \langle \Psi_0 | \Phi_a \rangle, yields:
    // V_a = \langle \Phi_a | \hat H | \Psi_0 \rangle - S_{a0} \langle \Psi_0 | \hat H | \Psi_0 \rangle.
    let s_a0 = overlaps.s_ai.dot(coeffs);
    v_a - s_a0.mapv(|x| ecurrent * x)
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
