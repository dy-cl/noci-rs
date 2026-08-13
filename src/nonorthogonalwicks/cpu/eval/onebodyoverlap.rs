// nonorthogonalwicks/cpu/eval/onebodyoverlap.rs
//! Fused CPU overlap and one-body nonorthogonal Wick evaluation.

// Crate-root imports.
use crate::ExcitationSpin;
use crate::maths::adjugate_transpose;
use crate::noci::NOCIScalar;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{bit, column_replacement_correction, mix_dets_same};

/// Evaluate the overlap and generalised-Fock matrix element from one prepared contraction
/// determinant:
/// `S = {}^{xw}\tilde S \sum_{\sum_i m_i=m}\det\mathbf D_{\mathrm{ov}}` and
/// `F = {}^{xw}\tilde S \sum_{\sum_i m_i=m}`
/// `[{}^xF_0^{(m_1)}\det\mathbf D_{\mathrm{ov}}`
/// `-\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}]`.
/// The overlap and one-body terms share every determinant/cofactor evaluation.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Prepared contraction determinants, cofactors and work storage.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Fused `(S,F)` values excluding external determinant excitation phases.
#[inline(always)]
pub(crate) fn xw_overlap_f<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    if w.m == 0 {
        xw_overlap_f_m0(w, l_ex, g_ex, scratch, tol)
    } else {
        xw_overlap_f_gen(w, l_ex, g_ex, scratch, tol)
    }
}

/// Evaluate the fused overlap and generalised-Fock matrix elements when `m = 0`.
/// Fixed-rank evaluators are used for `L = 1,2,3,4`; all other excitation ranks use the
/// singular-safe general cofactor form.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Prepared all-zero contraction determinant and cofactor storage.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Fused `(S,F)` values for `m = 0`.
#[inline(always)]
fn xw_overlap_f_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
    match l {
        0 => {
            let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
            (pref, pref * w.f0f[0])
        }
        1 => xw_overlap_f_m0_l1(w, scratch),
        2 => xw_overlap_f_m0_l2(w, scratch),
        3 => xw_overlap_f_m0_l3(w, scratch, tol),
        4 => xw_overlap_f_m0_l4(w, scratch, tol),
        _ => xw_overlap_f_m0_gen(w, l, scratch, tol),
    }
}

/// Evaluate the fixed-rank `L = 1` fused overlap and generalised-Fock matrix elements for `m = 0`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `scratch`: Prepared rank-one contraction determinant and cofactor storage.
/// # Returns
/// - `(T, T)`: Fused `(S,F)` values for `L = 1` and `m = 0`.
#[inline(always)]
fn xw_overlap_f_m0_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> (T, T) {
    let d = scratch.det0.as_slice()[0];
    let n = w.n();
    let fsl = w.ff_t_slice(0, 0);
    let replacement = fsl[scratch.cols[0] * n + scratch.rows[0]];
    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

    // S = {}^{xw}\tilde S D_{00}^{(0)} and
    // F = {}^{xw}\tilde S[D_{00}^{(0)}{}^xF_0^{(0)} - \mathcal F_{r_0c_0}^{(0,0)}].
    (pref * d, pref * (d * w.f0f[0] - replacement))
}

/// Evaluate the fixed-rank `L = 2` fused overlap and generalised-Fock matrix elements for `m = 0`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `scratch`: Prepared rank-two contraction determinant and cofactor storage.
/// # Returns
/// - `(T, T)`: Fused `(S,F)` values for `L = 2` and `m = 0`.
#[inline(always)]
fn xw_overlap_f_m0_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> (T, T) {
    let d = scratch.det0.as_slice();
    let a00 = d[0];
    let a01 = d[1];
    let a10 = d[2];
    let a11 = d[3];
    let det = a00 * a11 - a01 * a10;
    let n = w.n();
    let fsl = w.ff_t_slice(0, 0);
    let u0 = fsl[scratch.cols[0] * n + scratch.rows[0]];
    let u1 = fsl[scratch.cols[0] * n + scratch.rows[1]];
    let v0 = fsl[scratch.cols[1] * n + scratch.rows[0]];
    let v1 = fsl[scratch.cols[1] * n + scratch.rows[1]];
    let det_c0 = u0 * a11 - a01 * u1;
    let det_c1 = a00 * v1 - v0 * a10;
    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

    // F = {}^{xw}\tilde S[{}^xF_0^{(0)}\det\mathbf D_{\mathrm{ov}}
    // - \det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0}
    // - \det\mathbf D_{\mathrm{ov}}^{1\rightarrow\mathcal F_1}].
    (pref * det, pref * (det * w.f0f[0] - det_c0 - det_c1))
}

/// Evaluate the fixed-rank `L = 3` fused overlap and generalised-Fock matrix elements for `m = 0`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `scratch`: Prepared rank-three contraction determinant and cofactor storage.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate transpose.
/// # Returns
/// - `(T, T)`: Fused `(S,F)` values for `L = 3` and `m = 0`.
#[inline(always)]
fn xw_overlap_f_m0_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    let l = 3;
    let d = &scratch.det0.as_slice()[..l * l];
    let Some(det) = adjugate_transpose(
        scratch.adjt_det.as_mut_slice(),
        scratch.invs.as_mut_slice(),
        scratch.lu.as_mut_slice(),
        d,
        l,
        tol,
    ) else {
        let zero = <T as From<f64>>::from(0.0);
        return (zero, zero);
    };
    let n = w.n();
    let fsl = w.ff_t_slice(0, 0);
    let mut replacement = <T as From<f64>>::from(0.0);

    // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}
    // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}.
    for z in 0..l {
        let base = scratch.cols[z] * n;
        for eta in 0..l {
            replacement += scratch.adjt_det.as_slice()[eta * l + z] * fsl[base + scratch.rows[eta]];
        }
    }

    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
    (pref * det, pref * (det * w.f0f[0] - replacement))
}

/// Evaluate the fixed-rank `L = 4` fused overlap and generalised-Fock matrix elements for `m = 0`.
/// The determinant is obtained from the same cofactor evaluation used by the one-body term.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `scratch`: Prepared rank-four contraction determinant and cofactor storage.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate transpose.
/// # Returns
/// - `(T, T)`: Fused `(S,F)` values for `L = 4` and `m = 0`.
#[inline(always)]
fn xw_overlap_f_m0_l4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    let l = 4;
    let d = &scratch.det0.as_slice()[..l * l];
    let Some(det) = adjugate_transpose(
        scratch.adjt_det.as_mut_slice(),
        scratch.invs.as_mut_slice(),
        scratch.lu.as_mut_slice(),
        d,
        l,
        tol,
    ) else {
        let zero = <T as From<f64>>::from(0.0);
        return (zero, zero);
    };
    let n = w.n();
    let fsl = w.ff_t_slice(0, 0);
    let mut replacement = <T as From<f64>>::from(0.0);

    // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}
    // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}.
    for z in 0..l {
        let base = scratch.cols[z] * n;
        for eta in 0..l {
            replacement += scratch.adjt_det.as_slice()[eta * l + z] * fsl[base + scratch.rows[eta]];
        }
    }

    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
    (pref * det, pref * (det * w.f0f[0] - replacement))
}

/// Evaluate arbitrary-rank fused overlap and generalised-Fock matrix elements for `m = 0`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `l`: Total excitation rank.
/// - `scratch`: Prepared all-zero contraction determinant and cofactor storage.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate transpose.
/// # Returns
/// - `(T, T)`: Fused `(S,F)` values for arbitrary `L` and `m = 0`.
#[inline(always)]
fn xw_overlap_f_m0_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    let d = &scratch.det0.as_slice()[..l * l];
    let Some(det) = adjugate_transpose(
        scratch.adjt_det.as_mut_slice(),
        scratch.invs.as_mut_slice(),
        scratch.lu.as_mut_slice(),
        d,
        l,
        tol,
    ) else {
        let zero = <T as From<f64>>::from(0.0);
        return (zero, zero);
    };
    let n = w.n();
    let fsl = w.ff_t_slice(0, 0);
    let mut contribution = det * w.f0f[0];

    // Subtract \det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z} for each column z.
    for z in 0..l {
        let base = scratch.cols[z] * n;
        let correction =
            column_replacement_correction(l, d, scratch.adjt_det.as_slice(), z, |eta| {
                fsl[base + scratch.rows[eta]]
            });
        contribution -= det + correction;
    }

    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
    (pref * det, pref * contribution)
}

/// Evaluate fused overlap and generalised-Fock matrix elements for `m > 0` by traversing the
/// constrained distributions `\sum_i m_i=m` once.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m > 0`.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Prepared endpoint determinants, cofactors and mixed-determinant storage.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Fused `(S,F)` values summed over all allowed distributions.
#[inline(always)]
fn xw_overlap_f_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
    if l == 0 {
        let zero = <T as From<f64>>::from(0.0);
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        return if w.m == 1 {
            (zero, pref * w.f0f[1])
        } else {
            (zero, zero)
        };
    }

    let mut overlap = <T as From<f64>>::from(0.0);
    let mut one_body = <T as From<f64>>::from(0.0);
    let n = w.n();

    // Bit zero is the operator assignment m_1; the remaining L bits select determinant columns.
    mix_dets_same(w, l, 1, scratch, |bits, scratch| {
        let d = scratch.det_mix.as_slice();
        let Some(det) = adjugate_transpose(
            scratch.adjt_det.as_mut_slice(),
            scratch.invs.as_mut_slice(),
            scratch.lu.as_mut_slice(),
            d,
            l,
            tol,
        ) else {
            return;
        };
        let mi = bit(bits, 0);

        if mi == 0 {
            overlap += det;
        }

        // Preserve the established one-body threshold while retaining every overlap determinant.
        if det.abs() > tol {
            let mut contribution = det * w.f0f[mi];
            for z in 0..l {
                let mj = bit(bits, z + 1);
                let fsl = w.ff_t_slice(mi, mj);
                let base = scratch.cols[z] * n;
                let correction =
                    column_replacement_correction(l, d, scratch.adjt_det.as_slice(), z, |eta| {
                        fsl[base + scratch.rows[eta]]
                    });
                contribution -= det + correction;
            }
            one_body += contribution;
        }
    });

    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
    (pref * overlap, pref * one_body)
}
