// snoci/solve.rs

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::maths::{adjoint, general_evp};
use crate::noci::{DetPair, FockData, MOCache, NOCIData, NOCIScalar};
use crate::noci::{build_noci_fock, build_noci_hs, build_noci_s, calculate_m_pair};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksShared};
use crate::time_call;
use crate::{AoData, DetState, input::Input};

use super::{PT2ProjectedOperator, PT2Projection, Preconditioner, SNOCIFocks, SNOCIOverlaps};

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
/// - `(Array2<T>, Array2<T>, f64, Array1<T>)`: Hamiltonian matrix in the current space,
///   overlap matrix in the current space, lowest eigenvalue, and corresponding eigenvector.
pub(in crate::snoci) fn solve_current_space<T: NOCIScalar>(
    ao: &AoData,
    current_space: &[DetState<T>],
    input: &Input,
    wicks: Option<&WicksShared<T>>,
    mocache: &[MOCache<T>],
    tol: f64,
) -> (Array2<T>, Array2<T>, f64, Array1<T>) {
    time_call!(crate::timers::snoci::add_solve_current_space, {
        let wview = wicks.as_ref().map(|ws| ws.view());
        let data = NOCIData::new(ao, current_space, input, tol, wview).withmocache(mocache);

        let (hcurrent, scurrent, _) = build_noci_hs(&data, current_space, current_space, true);
        let (evals, c) = general_evp(&hcurrent, &scurrent, true, tol);

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
pub(in crate::snoci) fn build_snoci_overlaps<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    candidates: &[DetState<T>],
    selected_space: &[DetState<T>],
) -> SNOCIOverlaps<T> {
    time_call!(crate::timers::snoci::add_build_snoci_overlaps, {
        let (s_ai, _) = build_noci_s(data, candidates, selected_space, false);
        let s_ia = adjoint(&s_ai);

        SNOCIOverlaps { s_ai, s_ia }
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
pub(in crate::snoci) fn build_snoci_focks<T: NOCIScalar>(
    current_data: &NOCIData<'_, T>,
    candidate_data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    selected_space: &[DetState<T>],
    candidates: &[DetState<T>],
) -> SNOCIFocks<T> {
    time_call!(crate::timers::snoci::add_build_snoci_focks, {
        let (f_ii, _) = build_noci_fock(current_data, fock, selected_space, selected_space, true);
        let (f_ai, _) = build_noci_fock(candidate_data, fock, candidates, selected_space, false);
        let f_ia = adjoint(&f_ai);

        SNOCIFocks { f_ii, f_ai, f_ia }
    })
}

/// Build candidate-current Hamiltonian matrix elements.
/// # Arguments:
/// - `data`: Shared NOCI matrix-element data for candidate-left matrix elements.
/// - `candidates`: Candidate determinant space.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// # Returns:
/// - `Array2<T>`: Candidate-current Hamiltonian block `H_ai`.
pub(in crate::snoci) fn build_candidate_current_h<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    candidates: &[DetState<T>],
    selected_space: &[DetState<T>],
) -> Array2<T> {
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
pub(in crate::snoci) fn build_snoci_projection<T: NOCIScalar>(
    overlaps: &SNOCIOverlaps<T>,
    focks: &SNOCIFocks<T>,
    coeffs: &Array1<T>,
    e0: f64,
) -> PT2Projection<T> {
    time_call!(crate::timers::snoci::add_build_snoci_projection, {
        let coeffs_conj = coeffs.mapv(|x| x.conj());
        // A projected candidate state is given by:
        // | \Omega_a \rangle = | \Phi_a \rangle - | \Psi_0 \rangle \langle \Psi_0 | \Phi_a \rangle,
        // where | \Psi_0 \rangle is the current NOCI state.
        // We therefore require the following contractions:
        // S_{a0} = \langle \Phi_a | \Psi_0 \rangle = \sum_i S_{ai} c_i,
        let s_a0 = overlaps.s_ai.dot(coeffs);
        // S_{0a} = \langle \Psi_0 | \Phi_a \rangle = \sum_i S_{ia} c_i,
        let s_0a = overlaps.s_ia.t().dot(&coeffs_conj);
        // F_{a0} = \langle \Phi_a | \hat F | \Psi_0 \rangle = \sum_i F_{ai} c_i,
        let f_a0 = focks.f_ai.dot(coeffs);
        // F_{0a} = \langle \Psi_0 | \hat F | \Phi_a \rangle = \sum_i F_{ia} c_i.
        let f_0a = focks.f_ia.t().dot(&coeffs_conj);

        PT2Projection {
            e0,
            s_a0,
            s_0a,
            f_a0,
            f_0a,
        }
    })
}

/// Build the upper triangle of the unprojected candidate-candidate shifted Fock matrix `M`.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// # Returns:
/// - `Vec<T>`: Packed upper triangle of `M`.
pub(in crate::snoci) fn build_candidate_m<T: NOCIScalar>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
) -> Vec<T> {
    time_call!(crate::timers::snoci::add_build_candidate_m, {
        let n = op.candidates.len();
        let mut m = vec![T::from_real(0.0); n * (n + 1) / 2];

        m.par_iter_mut()
            .enumerate()
            .for_each_init(WickScratchSpin::new, |scratch, (k, m_ab)| {
                let mut lo = 0usize;
                let mut hi = n;

                while lo < hi {
                    let mid = lo + (hi - lo).div_ceil(2);
                    let prefix = mid * (2 * n - mid + 1) / 2;

                    if prefix <= k {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }

                let a = lo;
                let b = a + (k - a * (2 * n - a + 1) / 2);

                let ldet = &op.candidates[a];
                let gdet = &op.candidates[b];

                *m_ab = calculate_m_pair(
                    op.data,
                    op.fock,
                    DetPair::new(ldet, gdet),
                    op.projection.e0,
                    Some(scratch),
                );
            });
        m
    })
}

/// Build the diagonal of the unprojected candidate-candidate shifted Fock matrix `M`.
/// Uses a cached packed matrix if provided, otherwise evaluates diagonal matrix elements on the fly.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `m`: Optional packed candidate-candidate shifted Fock matrix.
/// # Returns:
/// - `Array1<T>`: Diagonal entries of `M`.
pub(in crate::snoci) fn build_candidate_m_diag<T: NOCIScalar>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    m: Option<&[T]>,
) -> Array1<T> {
    time_call!(crate::timers::snoci::add_build_candidate_m_diag, {
        let n = op.candidates.len();

        if let Some(m) = m {
            let mut diag = Array1::from_elem(n, T::from_real(0.0));

            for i in 0..n {
                let k = i * (2 * n - i + 1) / 2;
                diag[i] = m[k];
            }

            return diag;
        }

        let diag: Vec<T> = (0..n)
            .into_par_iter()
            .map_init(WickScratchSpin::new, |scratch, a| {
                let det = &op.candidates[a];
                calculate_m_pair(
                    op.data,
                    op.fock,
                    DetPair::new(det, det),
                    op.projection.e0,
                    Some(scratch),
                )
            })
            .collect();

        Array1::from_vec(diag)
    })
}

/// Apply the unprojected candidate-candidate shifted Fock matrix `M`.
/// Uses a cached packed matrix if provided, otherwise evaluates matrix elements on the fly.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `x`: Vector to apply `M` to.
/// - `m`: Optional packed candidate-candidate shifted Fock matrix.
/// # Returns:
/// - `Array1<T>`: Matrix-vector product `M x`.
pub(in crate::snoci) fn apply_candidate_m<T: NOCIScalar>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    x: &Array1<T>,
    m: Option<&[T]>,
) -> Array1<T> {
    time_call!(crate::timers::snoci::add_apply_candidate_m, {
        let n = op.candidates.len();

        if n == 0 {
            return Array1::from_vec(Vec::new());
        }

        let xs = x.as_slice_memory_order().unwrap();

        if let Some(m) = m {
            let y: Vec<T> = (0..n)
                .into_par_iter()
                .map(|a| {
                    let mut ya = T::from_real(0.0);

                    for (b, _) in xs.iter().enumerate().take(a) {
                        let k = b * (2 * n - b + 1) / 2 + (a - b);
                        ya += m[k].conj() * xs[b];
                    }

                    let row = a * (2 * n - a + 1) / 2;

                    for (b, _) in xs.iter().enumerate().take(n).skip(a) {
                        let k = row + (b - a);
                        ya += m[k] * xs[b];
                    }

                    ya
                })
                .collect();
            return Array1::from_vec(y);
        }

        let min_len = (n / (rayon::current_num_threads() * 8)).max(1);
        let zero = T::from_real(0.0);

        let y = (0..n)
            .into_par_iter()
            .with_min_len(min_len)
            .fold(
                || (WickScratchSpin::new(), vec![zero; n]),
                |(mut scratch, mut y), a| {
                    let xa = xs[a];
                    let ldet = &op.candidates[a];

                    for b in a..n {
                        let xb = xs[b];

                        if xa == zero && xb == zero {
                            continue;
                        }

                        let gdet = &op.candidates[b];
                        let m_ab = calculate_m_pair(
                            op.data,
                            op.fock,
                            DetPair::new(ldet, gdet),
                            op.projection.e0,
                            Some(&mut scratch),
                        );

                        if xb != zero {
                            y[a] += m_ab * xb;
                        }

                        if b != a && xa != zero {
                            y[b] += m_ab.conj() * xa;
                        }
                    }

                    (scratch, y)
                },
            )
            .map(|(_, y)| y)
            .reduce(
                || vec![zero; n],
                |mut lhs, rhs| {
                    for i in 0..n {
                        lhs[i] += rhs[i];
                    }
                    lhs
                },
            );
        Array1::from_vec(y)
    })
}

/// Apply the projected NOCI-PT2 shifted Fock matrix `M^Omega`.
/// Uses a cached packed candidate-candidate shifted Fock matrix if provided,
/// otherwise applies `M` without materialising it.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `x`: Vector to apply `M^Omega` to.
/// - `m`: Optional packed unprojected candidate-candidate shifted Fock matrix `M`.
/// # Returns:
/// - `Array1<T>`: Matrix-vector product `M^Omega x`.
pub(in crate::snoci) fn apply_omega_m<T: NOCIScalar>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    x: &Array1<T>,
    m: Option<&[T]>,
) -> Array1<T> {
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
        let mut y = apply_candidate_m(op, x, m);
        let two_e0 = T::from_real(2.0 * p.e0);

        for a in 0..y.len() {
            y[a] += -p.f_a0[a] * sx - p.s_a0[a] * fx + two_e0 * p.s_a0[a] * sx;
        }

        y
    })
}

/// Build a rank-2 Woodbury preconditioner for the projected NOCI-PT2 shifted Fock matrix.
/// # Arguments:
/// - `m_diag`: Diagonal of the unprojected candidate-candidate matrix `M`.
/// - `p`: Projection contractions used to form `M^Omega`.
/// # Returns:
/// - `OmegaRank2Preconditioner`: Rank-2 preconditioner for applying an approximate inverse of `M^Omega`.
pub(in crate::snoci) fn build_preconditioner<T: NOCIScalar>(
    m_diag: &Array1<T>,
    p: &PT2Projection<T>,
) -> Preconditioner<T> {
    Preconditioner::new(m_diag, p)
}

/// Build the unprojected candidate-current coupling vector `V`.
/// # Arguments:
/// - `h_ai`: Candidate-current Hamiltonian block.
/// - `coeffs`: Current-space ground-state eigenvector.
/// # Returns:
/// - `Array1<T>`: Unprojected candidate coupling vector.
pub(in crate::snoci) fn build_candidate_v<T: NOCIScalar>(
    h_ai: &Array2<T>,
    coeffs: &Array1<T>,
) -> Array1<T> {
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
/// - `Array1<T>`: Omega-projected coupling vector.
pub(in crate::snoci) fn build_omega_v<T: NOCIScalar>(
    s_ai: &Array2<T>,
    coeffs: &Array1<T>,
    v_a: Array1<T>,
    ecurrent: f64,
) -> Array1<T> {
    time_call!(crate::timers::snoci::add_build_omega_v, {
        // V_a^\Omega = \langle \Omega_a | \hat H | \Psi_0 \rangle.
        // Expanding | \Omega_a \rangle gives:
        // V_a^\Omega = V_a - E_\mathrm{NOCI} S_{a0}.
        let s_a0 = s_ai.dot(coeffs);
        v_a - s_a0.mapv(|x| T::from_real(ecurrent) * x)
    })
}

/// Select the highest-scoring candidates above the selection threshold.
/// # Arguments:
/// - `candidates`: Candidate determinants currently present in the pool.
/// - `candidate_scores`: Candidate importance scores.
/// - `sigma`: Selection threshold.
/// - `max_add`: Maximum number of candidates to add.
/// # Returns:
/// - `Vec<DetState<T>>`: Selected candidates sorted by decreasing score.
pub(in crate::snoci) fn select_candidates<T: NOCIScalar>(
    candidates: &[DetState<T>],
    candidate_scores: &[f64],
    sigma: f64,
    max_add: usize,
) -> Vec<DetState<T>> {
    let mut ranked: Vec<(DetState<T>, f64)> = candidates
        .iter()
        .cloned()
        .zip(candidate_scores.iter().copied())
        .filter(|(_, score)| *score > sigma)
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked
        .into_iter()
        .take(max_add)
        .map(|(state, _)| state)
        .collect()
}
