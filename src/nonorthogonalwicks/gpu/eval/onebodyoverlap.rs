// nonorthogonalwicks/gpu/eval/onebodyoverlap.rs
//! Fused GPU overlap and one-body nonorthogonal Wick evaluation.

// External crate imports.
use cubecl::prelude::*;

// Crate-root imports.
use crate::maths::gpu::wick::{adjugate_transpose3, adjugate_transpose4, det2};

// Parent/sibling imports.
use super::helpers::{bit, column_replacement_correction};
use super::onebody::fill_cofactors;
use super::overlap::mix_dets_same;
use super::prepare::{GpuSameSpinView, ff_t, prefactor};

/// Fused same-spin overlap and Fock matrix elements.
#[derive(CubeType)]
pub(crate) struct OverlapFock {
    /// Same-spin overlap `S = \langle\Phi_x|\Phi_w\rangle`.
    pub(crate) s: f64,
    /// Same-spin Fock matrix element `F = \langle\Phi_x|\hat F|\Phi_w\rangle`.
    pub(crate) f: f64,
}

/// Evaluate the overlap and generalised-Fock matrix element from one prepared contraction
/// determinant:
/// `S = {}^{xw}\tilde S \sum_{\sum_i m_i=m}\det\mathbf D_{\mathrm{ov}}` and
/// `F = {}^{xw}\tilde S \sum_{\sum_i m_i=m}`
/// `[{}^xF_0^{(m_1)}\det\mathbf D_{\mathrm{ov}}`
/// `-\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}]`.
/// The overlap and one-body terms share every determinant/cofactor evaluation.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Prepared determinant row labels.
/// - `cols`: Prepared determinant column labels.
/// - `det0`: Prepared all-zero endpoint determinant.
/// - `det1`: Prepared all-one endpoint determinant when `m > 0`.
/// - `work`: Mixed determinant storage.
/// - `cof`: Cofactor storage.
/// - `new_col`: Replacement-column storage.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `OverlapFock`: Fused `(S,F)` values excluding external determinant excitation phases.
#[cube]
pub(crate) fn xw_overlap_f(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &Array<f64>,
    det1: &Array<f64>,
    work: &mut Array<f64>,
    cof: &mut Array<f64>,
    new_col: &mut Array<f64>,
    #[comptime] l: usize,
) -> OverlapFock {
    let mut s = 0.0;
    let mut f = 0.0;
    if w.m == 0usize {
        let values = xw_overlap_f_m0(w, rows, cols, det0, cof, l);
        s = values.s;
        f = values.f;
    } else {
        let values = xw_overlap_f_gen(w, rows, cols, det0, det1, work, cof, new_col, l);
        s = values.s;
        f = values.f;
    }
    OverlapFock { s, f }
}

/// Evaluate fused overlap and generalised-Fock matrix elements when `m = 0`.
/// Fixed-rank evaluators are used for `L = 1,2,3,4`; all other ranks use the general cofactor form.
/// # Arguments:
/// - `w`: Device same-spin Wick view with `m = 0`.
/// - `rows`: Prepared determinant row labels.
/// - `cols`: Prepared determinant column labels.
/// - `det0`: Prepared all-zero contraction determinant.
/// - `cof`: Cofactor storage used by ranks `L \geq 3`.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `OverlapFock`: Fused `(S,F)` values for `m = 0`.
#[cube]
pub(crate) fn xw_overlap_f_m0(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &Array<f64>,
    cof: &mut Array<f64>,
    #[comptime] l: usize,
) -> OverlapFock {
    if comptime!(l == 0usize) {
        let pref = prefactor(w);
        OverlapFock {
            s: pref,
            f: pref * w.f0f[0],
        }
    } else if comptime!(l == 1usize) {
        xw_overlap_f_m0_l1(w, rows, cols, det0)
    } else if comptime!(l == 2usize) {
        xw_overlap_f_m0_l2(w, rows, cols, det0)
    } else if comptime!(l == 3usize) {
        xw_overlap_f_m0_l3(w, rows, cols, det0, cof)
    } else if comptime!(l == 4usize) {
        xw_overlap_f_m0_l4(w, rows, cols, det0, cof)
    } else {
        xw_overlap_f_m0_gen(w, rows, cols, det0, cof, l)
    }
}

/// Evaluate the fixed-rank `L = 1` fused overlap and generalised-Fock matrix elements for `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view with `m = 0`.
/// - `rows`: Rank-one determinant row label.
/// - `cols`: Rank-one determinant column label.
/// - `d`: Rank-one contraction determinant.
/// # Returns
/// - `OverlapFock`: Fused `(S,F)` values for `L = 1` and `m = 0`.
#[cube]
pub(crate) fn xw_overlap_f_m0_l1(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &Array<f64>,
) -> OverlapFock {
    let det = d[0];
    let replacement = ff_t(w, 0usize, 0usize, cols[0], rows[0]);
    let pref = prefactor(w);

    // S = {}^{xw}\tilde S D_{00}^{(0)} and
    // F = {}^{xw}\tilde S[D_{00}^{(0)}{}^xF_0^{(0)} - \mathcal F_{r_0c_0}^{(0,0)}].
    OverlapFock {
        s: pref * det,
        f: pref * (det * w.f0f[0] - replacement),
    }
}

/// Evaluate the fixed-rank `L = 2` fused overlap and generalised-Fock matrix elements for `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view with `m = 0`.
/// - `rows`: Rank-two determinant row labels.
/// - `cols`: Rank-two determinant column labels.
/// - `d`: Rank-two contraction determinant.
/// # Returns
/// - `OverlapFock`: Fused `(S,F)` values for `L = 2` and `m = 0`.
#[cube]
pub(crate) fn xw_overlap_f_m0_l2(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &Array<f64>,
) -> OverlapFock {
    let a00 = d[0];
    let a01 = d[1];
    let a10 = d[2];
    let a11 = d[3];
    let det = det2(a00, a01, a10, a11);
    let u0 = ff_t(w, 0usize, 0usize, cols[0], rows[0]);
    let u1 = ff_t(w, 0usize, 0usize, cols[0], rows[1]);
    let v0 = ff_t(w, 0usize, 0usize, cols[1], rows[0]);
    let v1 = ff_t(w, 0usize, 0usize, cols[1], rows[1]);
    let det_c0 = u0 * a11 - a01 * u1;
    let det_c1 = a00 * v1 - v0 * a10;
    let pref = prefactor(w);

    // F = {}^{xw}\tilde S[{}^xF_0^{(0)}\det\mathbf D_{\mathrm{ov}}
    // - \det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0}
    // - \det\mathbf D_{\mathrm{ov}}^{1\rightarrow\mathcal F_1}].
    OverlapFock {
        s: pref * det,
        f: pref * (det * w.f0f[0] - det_c0 - det_c1),
    }
}

/// Evaluate the fixed-rank `L = 3` fused overlap and generalised-Fock matrix elements for `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view with `m = 0`.
/// - `rows`: Rank-three determinant row labels.
/// - `cols`: Rank-three determinant column labels.
/// - `d`: Rank-three contraction determinant.
/// - `cof`: Rank-three cofactor storage.
/// # Returns
/// - `OverlapFock`: Fused `(S,F)` values for `L = 3` and `m = 0`.
#[cube]
pub(crate) fn xw_overlap_f_m0_l3(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &Array<f64>,
    cof: &mut Array<f64>,
) -> OverlapFock {
    let det = adjugate_transpose3(d, cof);
    let mut replacement = 0.0;

    // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}
    // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}.
    for z in 0usize..3usize {
        for eta in 0usize..3usize {
            replacement += cof[eta * 3usize + z] * ff_t(w, 0usize, 0usize, cols[z], rows[eta]);
        }
    }

    let pref = prefactor(w);
    OverlapFock {
        s: pref * det,
        f: pref * (det * w.f0f[0] - replacement),
    }
}

/// Evaluate the fixed-rank `L = 4` fused overlap and generalised-Fock matrix elements for `m = 0`.
/// The determinant is obtained from the same 16 cofactors used by the one-body term.
/// # Arguments:
/// - `w`: Device same-spin Wick view with `m = 0`.
/// - `rows`: Rank-four determinant row labels.
/// - `cols`: Rank-four determinant column labels.
/// - `d`: Rank-four contraction determinant.
/// - `cof`: Rank-four cofactor storage.
/// # Returns
/// - `OverlapFock`: Fused `(S,F)` values for `L = 4` and `m = 0`.
#[cube]
pub(crate) fn xw_overlap_f_m0_l4(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &Array<f64>,
    cof: &mut Array<f64>,
) -> OverlapFock {
    let det = adjugate_transpose4(d, cof);
    let mut replacement = 0.0;

    // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}
    // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}.
    for z in 0usize..4usize {
        for eta in 0usize..4usize {
            replacement += cof[eta * 4usize + z] * ff_t(w, 0usize, 0usize, cols[z], rows[eta]);
        }
    }

    let pref = prefactor(w);
    OverlapFock {
        s: pref * det,
        f: pref * (det * w.f0f[0] - replacement),
    }
}

/// Evaluate arbitrary-rank fused overlap and generalised-Fock matrix elements for `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view with `m = 0`.
/// - `rows`: Prepared determinant row labels.
/// - `cols`: Prepared determinant column labels.
/// - `d`: Prepared all-zero contraction determinant.
/// - `cof`: Cofactor storage.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `OverlapFock`: Fused `(S,F)` values for arbitrary `L` and `m = 0`.
#[cube]
pub(crate) fn xw_overlap_f_m0_gen(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &Array<f64>,
    cof: &mut Array<f64>,
    #[comptime] l: usize,
) -> OverlapFock {
    let det = fill_cofactors(d, cof, l);
    let mut replacement = 0.0;

    // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}
    // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}.
    for z in 0usize..l {
        for eta in 0usize..l {
            replacement += cof[eta * l + z] * ff_t(w, 0usize, 0usize, cols[z], rows[eta]);
        }
    }
    let pref = prefactor(w);

    OverlapFock {
        s: pref * det,
        f: pref * (det * w.f0f[0] - replacement),
    }
}

/// Evaluate fused overlap and generalised-Fock matrix elements for `m > 0` by traversing the
/// constrained distributions `\sum_i m_i=m` once.
/// # Arguments:
/// - `w`: Device same-spin Wick view with `m > 0`.
/// - `rows`: Prepared determinant row labels.
/// - `cols`: Prepared determinant column labels.
/// - `det0`: Prepared all-zero endpoint determinant.
/// - `det1`: Prepared all-one endpoint determinant.
/// - `work`: Mixed determinant storage.
/// - `cof`: Cofactor storage.
/// - `new_col`: Replacement-column storage.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `OverlapFock`: Fused `(S,F)` values summed over all allowed distributions.
#[cube]
pub(crate) fn xw_overlap_f_gen(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &Array<f64>,
    det1: &Array<f64>,
    work: &mut Array<f64>,
    cof: &mut Array<f64>,
    new_col: &mut Array<f64>,
    #[comptime] l: usize,
) -> OverlapFock {
    let mut overlap = 0.0;
    let mut one_body = 0.0;

    if comptime!(l == 0usize) {
        if w.m == 1usize {
            one_body = w.f0f[1];
        }
    } else {
        // Bit zero is m_1; the remaining L bits select the contraction-determinant columns.
        let limit = 1u32 << (l + 1usize);
        for bits in 0u32..limit {
            if usize::cast_from(bits.count_ones()) == w.m {
                let mi = bit(bits, 0usize);
                let cbits = bits >> 1u32;
                mix_dets_same(det0, det1, work, l, cbits);
                let det = fill_cofactors(work, cof, l);
                if mi == 0usize {
                    overlap += det;
                }
                let mut contribution = det * w.f0f[mi];
                for z in 0usize..l {
                    let mj = bit(bits, z + 1usize);
                    for eta in 0usize..l {
                        new_col[eta] = ff_t(w, mi, mj, cols[z], rows[eta]);
                    }
                    let correction = column_replacement_correction(l, work, cof, z, new_col);
                    contribution -= det + correction;
                }
                one_body += contribution;
            }
        }
    }

    let pref = prefactor(w);
    OverlapFock {
        s: pref * overlap,
        f: pref * one_body,
    }
}
