// snoci/solve.rs

use rayon::prelude::*;
use ndarray::{Array1, Array2};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::{WicksShared, WickScratchSpin};
use crate::noci::{MOCache, NOCIData, FockData, DetPair};
use crate::noci::{build_noci_s, build_noci_fock, build_noci_hs, calculate_m_pair};
use crate::maths::{general_evp_real};
use crate::time_call;

use super::{SNOCIOverlaps, SNOCIFocks, PT2ProjectedOperator, PT2Projection};

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
/// Uses symmetry of `M` so that each pair is evaluated only once.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `x`: Vector to apply `M` to.
/// # Returns:
/// - `Array1<f64>`: Matrix-vector product `M x`.
pub(in crate::snoci) fn apply_candidate_m(op: &PT2ProjectedOperator<'_, '_, '_>, x: &Array1<f64>) -> Array1<f64> {
    time_call!(crate::timers::snoci::add_apply_candidate_m, {
        let n = op.candidates.len();

        if n == 0 {
            return Array1::zeros(0);
        }

        let xs = x.as_slice_memory_order().unwrap();
        let min_len = (n / (rayon::current_num_threads() * 8)).max(1);
    
        // Calculate y_a = \sum_b (F_{ab} - E^{(0)} S_{ab}) x_b.
        let y = (0..n).into_par_iter().with_min_len(min_len).fold(
            || (WickScratchSpin::new(), vec![0.0; n]),
            |(mut scratch, mut y), a| {
                let xa = xs[a];
                let ldet = &op.candidates[a];

                for b in a..n {
                    let xb = xs[b];

                    if xa == 0.0 && xb == 0.0 {
                        continue;
                    }

                    let gdet = &op.candidates[b];
                    let m_ab = calculate_m_pair(op.data, op.fock, DetPair::new(ldet, gdet), op.projection.e0, Some(&mut scratch));

                    if xb != 0.0 {
                        y[a] += m_ab * xb;
                    }

                    if b != a && xa != 0.0 {
                        y[b] += m_ab * xa;
                    }
                }

                (scratch, y)
            }
        ).map(|(_, y)| y).reduce(
            || vec![0.0; n],
            |mut lhs, rhs| {
                for i in 0..n {
                    lhs[i] += rhs[i];
                }
                lhs
            }
        );

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
            
            let m_aa = calculate_m_pair(op.data, op.fock, pair, p.e0, Some(scratch));
            
            m_aa - p.f_a0[a] * p.s_0a[a] - p.s_a0[a] * p.f_0a[a] + 2.0 * p.e0 * p.s_a0[a] * p.s_0a[a]
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
