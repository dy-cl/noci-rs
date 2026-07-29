// nonorthogonalwicks/eval/overlap.rs
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::mix_dets_same;
use super::prepare::prepare_same;
use crate::ExcitationSpin;
use crate::maths::{det, det_lu_l5, det_lu_l6};
use crate::noci::NOCIScalar;
use crate::time_call;

/// Evaluate the same-spin overlap between excited determinants generated from the reference pair
/// \langle{}^x\Psi| and |{}^w\Psi\rangle:
/// \langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
/// = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L=m}}
/// \det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L).
/// Each m_i is zero or one. The lower triangle of \mathbf D_{\mathrm{ov}}, including its diagonal,
/// contains X^{(m_i)} contractions, while its upper triangle contains Y^{(m_i)} contractions.
/// The implementation stores the orbital-pairing phase separately from the product of non-zero
/// singular values forming {}^{xw}\tilde S.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.
/// - `g_ex`: Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.
/// - `scratch`: Prepared contraction determinants and work storage.
/// # Returns
/// - `T`: Same-spin overlap matrix element.
#[inline(always)]
pub fn lg_overlap<T: NOCIScalar>(
    w: &SameSpinView<T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap, {
        // The contraction determinant has dimension L = L_x + L_w.
        let l = l_ex.holes.len() + g_ex.holes.len();

        // A nonzero term requires one contraction for every zero-overlap orbital pair. The
        // constrained sum contains only the all-zero distribution for m = 0, only the all-one
        // distribution for m = L, and every allowed mixed distribution for 0 < m < L.
        if w.m > l {
            <T as From<f64>>::from(0.0)
        } else if w.m == 0 {
            lg_overlap_m0(w, l, scratch)
        } else if w.m == l {
            lg_overlap_ml(w, l, scratch)
        } else {
            lg_overlap_gen(w, l, scratch)
        }
    })
}

/// Evaluate a same-spin overlap for factor-table construction, using the direct overlap-only path
/// when m = 0 and L \leq 6:
/// \langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
/// = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L=m}}
/// \det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L).
/// The direct path avoids preparing reusable Hamiltonian scratch data. Other cases use `prepare_same`
/// followed by the general overlap evaluator.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage used by the prepared evaluation path.
/// # Returns
/// - `f64`: Same-spin overlap excluding excitation phases applied outside the Wick evaluation.
#[inline(always)]
pub(crate) fn lg_overlap_same_f64(
    w: &SameSpinView<'_, f64>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<f64>,
) -> f64 {
    // Determine the contraction-determinant dimension L = L_x + L_w.
    let l = l_ex.holes.len() + g_ex.holes.len();

    // No distribution satisfying \sum_i m_i = m exists when m > L.
    if w.m > l {
        return 0.0;
    }

    // For m = 0 and L \leq 6, construct and evaluate \mathbf D_{\mathrm{ov}}(0,\ldots,0)
    // directly without populating the reusable scratch representation.
    if w.m == 0 && l <= 6 {
        return lg_overlap_m0_direct_f64(w, l_ex, g_ex);
    }

    // Prepare the all-m_i = 0 and, where required, all-m_i = 1 contraction determinants
    // before applying the standard overlap evaluation.
    prepare_same(w, l_ex, g_ex, scratch);
    lg_overlap(w, l_ex, g_ex, scratch)
}

/// Evaluate the same-spin overlap directly when m = 0 and L \leq 6:
/// \langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
/// = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0).
/// The row labels are the x-reference particles followed by the w-reference holes, while the column
/// labels are the x-reference holes followed by the w-reference particles. The determinant contains
/// X^{(0)} on and below the diagonal and Y^{(0)} above the diagonal.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// # Returns
/// - `f64`: Same-spin overlap excluding excitation phases applied outside the Wick evaluation.
#[inline(always)]
pub(crate) fn lg_overlap_m0_direct_f64(
    w: &SameSpinView<'_, f64>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
) -> f64 {
    // Split L into the bra- and ket-reference excitation ranks and form {}^{xw}\tilde S
    // from the separately stored orbital-pairing phase and non-zero singular-value product.
    let nl = l_ex.holes.len();
    let ng = g_ex.holes.len();
    let l = nl + ng;
    let pref = w.phase * w.tilde_s_prod;

    // With no excitation pairs, the determinant is the empty determinant with value one.
    if l == 0 {
        return pref;
    }

    // Read the m_i = 0 fundamental contractions and allocate the row and column labels of
    // \mathbf D_{\mathrm{ov}}(0,\ldots,0).
    let n = w.n();
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    let x0 = w.x_slice(0);
    let y0 = w.y_slice(0);
    let mut rows = [0usize; 6];
    let mut cols = [0usize; 6];

    // The x-reference excitation pairs contribute particle rows a and hole columns i.
    for k in 0..nl {
        rows[k] = l_ex.parts[k] - nocc;
        cols[k] = l_ex.holes[k];
    }

    // The w-reference excitation pairs follow with hole rows j and particle columns b.
    for k in 0..ng {
        let i = nl + k;
        rows[i] = nvirt + g_ex.holes[k];
        cols[i] = g_ex.parts[k];
    }

    // Evaluate {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0) using a
    // fixed-rank determinant expression.
    match l {
        // \det\mathbf D_{\mathrm{ov}}(0) = X_{r_0c_0}^{(0)}.
        1 => pref * x0[rows[0] * n + cols[0]],
        2 => {
            // \mathbf D_{\mathrm{ov}} = [[X_{r_0c_0},Y_{r_0c_1}],
            //                            [X_{r_1c_0},X_{r_1c_1}]].
            let d0 = x0[rows[0] * n + cols[0]];
            let d1 = y0[rows[0] * n + cols[1]];
            let d2 = x0[rows[1] * n + cols[0]];
            let d3 = x0[rows[1] * n + cols[1]];

            pref * (d0 * d3 - d1 * d2)
        }
        3 => {
            // Build the rank-three X/Y contraction determinant explicitly.
            let d0 = x0[rows[0] * n + cols[0]];
            let d1 = y0[rows[0] * n + cols[1]];
            let d2 = y0[rows[0] * n + cols[2]];
            let d3 = x0[rows[1] * n + cols[0]];
            let d4 = x0[rows[1] * n + cols[1]];
            let d5 = y0[rows[1] * n + cols[2]];
            let d6 = x0[rows[2] * n + cols[0]];
            let d7 = x0[rows[2] * n + cols[1]];
            let d8 = x0[rows[2] * n + cols[2]];

            pref * (d0 * (d4 * d8 - d5 * d7) - d1 * (d3 * d8 - d5 * d6) + d2 * (d3 * d7 - d4 * d6))
        }
        4 => {
            // Build \mathbf D_{\mathrm{ov}} with X on and below the diagonal and Y above it.
            let mut d = [0.0; 16];

            for i in 0..4 {
                let row = rows[i] * n;

                for j in 0..4 {
                    d[i * 4 + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            pref * det(&d, 4).unwrap_or(0.0)
        }
        5 => {
            // Build the rank-five contraction determinant before evaluating it by LU factorisation.
            let mut lu = [0.0; 25];

            for i in 0..5 {
                let row = rows[i] * n;

                for j in 0..5 {
                    lu[i * 5 + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            pref * det_lu_l5(&mut lu).unwrap_or(0.0)
        }
        6 => {
            // Build the rank-six contraction determinant before evaluating it by LU factorisation.
            let mut lu = [0.0; 36];

            for i in 0..6 {
                let row = rows[i] * n;

                for j in 0..6 {
                    lu[i * 6 + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            pref * det_lu_l6(&mut lu).unwrap_or(0.0)
        }
        _ => {
            // Retain the general determinant path for completeness; the direct API currently calls
            // this function only for L \leq 6.
            let mut d = [0.0; 36];

            for i in 0..l {
                let row = rows[i] * n;

                for j in 0..l {
                    d[i * l + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            pref * det(&d[..l * l], l).unwrap_or(0.0)
        }
    }
}

/// Evaluate the same-spin overlap when m = 0:
/// \langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
/// = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0).
/// Fixed-rank determinant kernels are used for L = 1,\ldots,6; arbitrary ranks use the general
/// determinant routine. For L = 0, the overlap is the reduced reference overlap {}^{xw}\tilde S.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l`: Total excitation rank L = L_x + L_w.
/// - `scratch`: Prepared \mathbf D_{\mathrm{ov}}(0,\ldots,0).
/// # Returns
/// - `T`: Same-spin overlap matrix element for m = 0.
#[inline(always)]
fn lg_overlap_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0, {
        // Evaluate {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0) with a
        // fixed-rank kernel where available.
        match l {
            // The empty contraction determinant has determinant one.
            0 => w.phase * <T as From<f64>>::from(w.tilde_s_prod),
            1 => lg_overlap_m0_l1(w, scratch),
            2 => lg_overlap_m0_l2(w, scratch),
            3 => lg_overlap_m0_l3(w, scratch),
            4 => lg_overlap_m0_l4(w, scratch),
            5 => lg_overlap_m0_l5(w, scratch),
            6 => lg_overlap_m0_l6(w, scratch),
            _ => {
                // Evaluate the prepared arbitrary-rank contraction determinant directly.
                w.phase
                    * <T as From<f64>>::from(w.tilde_s_prod)
                    * det(scratch.det0.as_slice(), l).unwrap_or(<T as From<f64>>::from(0.0))
            }
        }
    })
}

/// Evaluate the fixed-rank L = 1 overlap when m = 0:
/// {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0) = {}^{xw}\tilde S D_{00}^{(0)}.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-one contraction determinant.
/// # Returns
/// - `T`: Same-spin overlap for L = 1 and m = 0.
#[inline(always)]
fn lg_overlap_m0_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0_l1, {
        // \det\mathbf D_{\mathrm{ov}}(0) = D_{00}^{(0)}.
        let d = scratch.det0.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * d[0]
    })
}

/// Evaluate the fixed-rank L = 2 overlap when m = 0:
/// {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,0).
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-two contraction determinant.
/// # Returns
/// - `T`: Same-spin overlap for L = 2 and m = 0.
#[inline(always)]
fn lg_overlap_m0_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0_l2, {
        // \det\mathbf D_{\mathrm{ov}}(0,0) = D_{00}D_{11} - D_{01}D_{10}.
        let d = scratch.det0.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * (d[0] * d[3] - d[1] * d[2])
    })
}

/// Evaluate the fixed-rank L = 3 overlap when m = 0:
/// {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,0,0).
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-three contraction determinant.
/// # Returns
/// - `T`: Same-spin overlap for L = 3 and m = 0.
#[inline(always)]
fn lg_overlap_m0_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0_l3, {
        // Expand \det\mathbf D_{\mathrm{ov}}(0,0,0) along its first row.
        let d = scratch.det0.as_slice();
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * (d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
                + d[2] * (d[3] * d[7] - d[4] * d[6]))
    })
}

/// Evaluate the fixed-rank L = 4 overlap when m = 0:
/// {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,0,0,0).
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-four contraction determinant.
/// # Returns
/// - `T`: Same-spin overlap for L = 4 and m = 0.
#[inline(always)]
fn lg_overlap_m0_l4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0_l4, {
        // Evaluate the prepared rank-four contraction determinant by first-row cofactor expansion.
        let det = det(&scratch.det0.as_slice()[..16], 4).unwrap_or(<T as From<f64>>::from(0.0));
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det
    })
}

/// Evaluate the fixed-rank L = 5 overlap when m = 0:
/// {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,0,0,0,0).
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-five contraction determinant.
/// # Returns
/// - `T`: Same-spin overlap for L = 5 and m = 0.
#[inline(always)]
fn lg_overlap_m0_l5<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    // Copy \mathbf D_{\mathrm{ov}}(0,0,0,0,0) because the LU determinant routine overwrites its input.
    let mut lu = [T::from_real(0.0); 25];
    lu.copy_from_slice(&scratch.det0.as_slice()[..25]);
    let det = det_lu_l5(&mut lu).unwrap_or(<T as From<f64>>::from(0.0));
    w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det
}

/// Evaluate the fixed-rank L = 6 overlap when m = 0:
/// {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,0,0,0,0,0).
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-six contraction determinant.
/// # Returns
/// - `T`: Same-spin overlap for L = 6 and m = 0.
#[inline(always)]
fn lg_overlap_m0_l6<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    // Copy \mathbf D_{\mathrm{ov}}(0,0,0,0,0,0) because the LU determinant routine overwrites its input.
    let mut lu = [T::from_real(0.0); 36];
    lu.copy_from_slice(&scratch.det0.as_slice()[..36]);
    let det = det_lu_l6(&mut lu).unwrap_or(<T as From<f64>>::from(0.0));
    w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det
}

/// Evaluate the same-spin overlap when m = L. The only allowed distribution is
/// (m_1,\ldots,m_L) = (1,\ldots,1), so:
/// \langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
/// = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,\ldots,1).
/// Fixed-rank determinant kernels are used for L = 1,2,3.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l`: Total excitation rank L = L_x + L_w, equal to m in this path.
/// - `scratch`: Prepared \mathbf D_{\mathrm{ov}}(1,\ldots,1).
/// # Returns
/// - `T`: Same-spin overlap matrix element for m = L.
#[inline(always)]
fn lg_overlap_ml<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml, {
        // Evaluate {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,\ldots,1) with a
        // fixed-rank kernel where available.
        match l {
            // This branch is retained for completeness; lg_overlap dispatches L = m = 0 to the m = 0 path.
            0 => w.phase * <T as From<f64>>::from(w.tilde_s_prod),
            1 => lg_overlap_ml_l1(w, scratch),
            2 => lg_overlap_ml_l2(w, scratch),
            3 => lg_overlap_ml_l3(w, scratch),
            _ => {
                // Evaluate the prepared arbitrary-rank all-m_i = 1 determinant directly.
                w.phase
                    * <T as From<f64>>::from(w.tilde_s_prod)
                    * det(scratch.det1.as_slice(), l).unwrap_or(<T as From<f64>>::from(0.0))
            }
        }
    })
}

/// Evaluate the fixed-rank L = m = 1 overlap:
/// {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1) = {}^{xw}\tilde S D_{00}^{(1)}.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `scratch`: Prepared rank-one m_1 = 1 contraction determinant.
/// # Returns
/// - `T`: Same-spin overlap for L = m = 1.
#[inline(always)]
fn lg_overlap_ml_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml_l1, {
        // \det\mathbf D_{\mathrm{ov}}(1) = D_{00}^{(1)}.
        let d = scratch.det1.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * d[0]
    })
}

/// Evaluate the fixed-rank L = m = 2 overlap:
/// {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,1).
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `scratch`: Prepared rank-two all-m_i = 1 contraction determinant.
/// # Returns
/// - `T`: Same-spin overlap for L = m = 2.
#[inline(always)]
fn lg_overlap_ml_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml_l2, {
        // \det\mathbf D_{\mathrm{ov}}(1,1) = D_{00}D_{11} - D_{01}D_{10}.
        let d = scratch.det1.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * (d[0] * d[3] - d[1] * d[2])
    })
}

/// Evaluate the fixed-rank L = m = 3 overlap:
/// {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,1,1).
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `scratch`: Prepared rank-three all-m_i = 1 contraction determinant.
/// # Returns
/// - `T`: Same-spin overlap for L = m = 3.
#[inline(always)]
fn lg_overlap_ml_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml_l3, {
        // Expand \det\mathbf D_{\mathrm{ov}}(1,1,1) along its first row.
        let d = scratch.det1.as_slice();
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * (d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
                + d[2] * (d[3] * d[7] - d[4] * d[6]))
    })
}

/// Evaluate the same-spin overlap for 0 < m < L by summing every allowed distribution:
/// \langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
/// = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L=m}}
/// \det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L), \qquad m_i \in \{0,1\}.
/// Each distribution selects every column of \mathbf D_{\mathrm{ov}} from the corresponding
/// all-m_i = 0 or all-m_i = 1 contraction determinant.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l`: Total excitation rank L = L_x + L_w.
/// - `scratch`: Prepared all-m_i = 0 and all-m_i = 1 determinants and mixed-determinant storage.
/// # Returns
/// - `T`: Same-spin overlap summed over all \binom{L}{m} allowed distributions.
#[inline(always)]
fn lg_overlap_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_gen, {
        let mut acc = <T as From<f64>>::from(0.0);

        // Enumerate the \binom{L}{m} distributions satisfying \sum_i m_i = m and construct
        // each \mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L) by selecting its columns from `det0` or `det1`.
        mix_dets_same(w, l, 0, scratch, |_, scratch| {
            let d = scratch.det_mix.as_slice();

            // Evaluate the determinant using direct low-rank expressions where available.
            let contrib = match l {
                1 => d[0],
                2 => d[0] * d[3] - d[1] * d[2],
                3 => {
                    d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
                        + d[2] * (d[3] * d[7] - d[4] * d[6])
                }
                _ => det(d, l).unwrap_or(<T as From<f64>>::from(0.0)),
            };
            // Add \det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L) to the constrained sum.
            acc += contrib;
        });

        // Apply the orbital-pairing phase to the product of non-zero singular values to recover
        // {}^{xw}\tilde S\sum_{\{m_i\}}\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L).
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
    })
}
