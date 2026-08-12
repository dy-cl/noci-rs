// nonorthogonalwicks/gpu/eval/onebody.rs
//! GPU one-body nonorthogonal Wick evaluation.

// External crate imports.
use cubecl::prelude::*;

// Parent/sibling imports.
use super::helpers::{
    adjugate_transpose3, adjugate_transpose4, bit, column_replacement_correction, det_or_zero, det2,
};
use super::overlap::mix_dets_same;
use super::prepare::{GpuSameSpinView, ff_t, prefactor};

/// Evaluate the generalised-Fock matrix element between excited determinants.
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
/// - `f64`: Same-spin Fock matrix element excluding external determinant excitation phases.
#[cube]
pub(crate) fn xw_f(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &Array<f64>,
    det1: &Array<f64>,
    work: &mut Array<f64>,
    cof: &mut Array<f64>,
    new_col: &mut Array<f64>,
    #[comptime] l: usize,
) -> f64 {
    if w.m == 0u32 {
        xw_one_body_m0(w, rows, cols, det0, cof, l)
    } else {
        xw_one_body_gen(w, rows, cols, det0, det1, work, cof, new_col, l)
    }
}

/// Evaluate the Fock matrix element when `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `det0`: Prepared all-zero determinant.
/// - `cof`: Cofactor storage.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `f64`: Fock matrix element for `m = 0`.
#[cube]
pub(crate) fn xw_one_body_m0(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &Array<f64>,
    cof: &mut Array<f64>,
    #[comptime] l: usize,
) -> f64 {
    if comptime!(l == 0usize) {
        prefactor(w) * w.f0f[0]
    } else if comptime!(l == 1usize) {
        xw_one_body_m0_l1(w, rows, cols, det0)
    } else if comptime!(l == 2usize) {
        xw_one_body_m0_l2(w, rows, cols, det0)
    } else if comptime!(l == 3usize) {
        xw_one_body_m0_l3(w, rows, cols, det0, cof)
    } else if comptime!(l == 4usize) {
        xw_one_body_m0_l4(w, rows, cols, det0, cof)
    } else {
        xw_one_body_m0_gen(w, rows, cols, det0, cof, l)
    }
}

/// Evaluate the fixed-rank L = 1 Fock matrix element for `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `det0`: Rank-one determinant.
/// # Returns
/// - `f64`: Fixed-rank one-body matrix element.
#[cube]
pub(crate) fn xw_one_body_m0_l1(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &Array<f64>,
) -> f64 {
    let repl = ff_t(w, 0usize, 0usize, cols[0], rows[0]);
    prefactor(w) * (det0[0] * w.f0f[0] - repl)
}

/// Evaluate the fixed-rank L = 2 Fock matrix element for `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `d`: Rank-two determinant.
/// # Returns
/// - `f64`: Fixed-rank one-body matrix element.
#[cube]
pub(crate) fn xw_one_body_m0_l2(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &Array<f64>,
) -> f64 {
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

    prefactor(w) * (det * w.f0f[0] - det_c0 - det_c1)
}

/// Evaluate the fixed-rank L = 3 Fock matrix element for `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `d`: Rank-three determinant.
/// - `cof`: Cofactor storage.
/// # Returns
/// - `f64`: Fixed-rank one-body matrix element.
#[cube]
pub(crate) fn xw_one_body_m0_l3(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &Array<f64>,
    cof: &mut Array<f64>,
) -> f64 {
    let det = adjugate_transpose3(d, cof);
    let mut repl = 0.0;
    for z in 0usize..3usize {
        for eta in 0usize..3usize {
            repl += cof[eta * 3usize + z] * ff_t(w, 0usize, 0usize, cols[z], rows[eta]);
        }
    }
    prefactor(w) * (det * w.f0f[0] - repl)
}

/// Evaluate the fixed-rank L = 4 Fock matrix element for `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `d`: Rank-four determinant.
/// - `cof`: Cofactor storage.
/// # Returns
/// - `f64`: Fixed-rank one-body matrix element.
#[cube]
pub(crate) fn xw_one_body_m0_l4(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &Array<f64>,
    cof: &mut Array<f64>,
) -> f64 {
    let det = adjugate_transpose4(d, cof);
    let mut repl = 0.0;
    for z in 0usize..4usize {
        for eta in 0usize..4usize {
            repl += cof[eta * 4usize + z] * ff_t(w, 0usize, 0usize, cols[z], rows[eta]);
        }
    }
    prefactor(w) * (det * w.f0f[0] - repl)
}

/// Evaluate the arbitrary-rank Fock matrix element when `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `d`: Prepared all-zero determinant.
/// - `cof`: Cofactor storage.
/// - `l`: Determinant dimension.
/// # Returns
/// - `f64`: General one-body matrix element.
#[cube]
pub(crate) fn xw_one_body_m0_gen(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &Array<f64>,
    cof: &mut Array<f64>,
    l: usize,
) -> f64 {
    let det = fill_cofactors(d, cof, l);
    let mut contrib = det * w.f0f[0];
    for z in 0usize..l {
        let mut repl = 0.0;
        for eta in 0usize..l {
            repl += cof[eta * l + z] * ff_t(w, 0usize, 0usize, cols[z], rows[eta]);
        }
        contrib -= repl;
    }
    prefactor(w) * contrib
}

/// Evaluate the one-body matrix element when `m > 0` by constrained distributions.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `det0`: All-zero endpoint determinant.
/// - `det1`: All-one endpoint determinant.
/// - `work`: Mixed determinant storage.
/// - `cof`: Cofactor storage.
/// - `new_col`: Replacement-column storage.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `f64`: General one-body matrix element.
#[cube]
pub(crate) fn xw_one_body_gen(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &Array<f64>,
    det1: &Array<f64>,
    work: &mut Array<f64>,
    cof: &mut Array<f64>,
    new_col: &mut Array<f64>,
    #[comptime] l: usize,
) -> f64 {
    let mut acc = 0.0;
    let limit = 1u32 << (l + 1usize);
    for bits in 0u32..limit {
        if bits.count_ones() == w.m {
            let mi = bit(bits, 0usize);
            let cbits = bits >> 1u32;
            mix_dets_same(det0, det1, work, l, cbits);
            let det = fill_cofactors(work, cof, l);
            let mut contrib = det * w.f0f[mi];
            for z in 0usize..l {
                let mj = bit(bits, z + 1usize);
                for eta in 0usize..l {
                    new_col[eta] = ff_t(w, mi, mj, cols[z], rows[eta]);
                }
                let corr = column_replacement_correction(l, work, cof, z, new_col);
                contrib -= det + corr;
            }
            acc += contrib;
        }
    }
    prefactor(w) * acc
}

/// Fill cofactors for a small determinant without using inverse-based formulas.
/// # Arguments:
/// - `d`: Row-major determinant.
/// - `cof`: Row-major cofactor output.
/// - `l`: Matrix dimension.
/// # Returns
/// - `f64`: Determinant.
#[cube]
pub(crate) fn fill_cofactors(
    d: &Array<f64>,
    cof: &mut Array<f64>,
    l: usize,
) -> f64 {
    if l == 1usize {
        cof[0] = 1.0;
        d[0]
    } else if l == 2usize {
        cof[0] = d[3];
        cof[1] = -d[2];
        cof[2] = -d[1];
        cof[3] = d[0];
        det2(d[0], d[1], d[2], d[3])
    } else if l == 3usize {
        adjugate_transpose3(d, cof)
    } else if l == 4usize {
        adjugate_transpose4(d, cof)
    } else {
        let det = det_or_zero(d, l);
        for r in 0usize..l {
            for c in 0usize..l {
                cof[r * l + c] = cofactor_minor(d, l, r, c);
            }
        }
        det
    }
}

/// Evaluate one cofactor by an explicit minor determinant for fallback ranks.
/// # Arguments:
/// - `d`: Row-major determinant.
/// - `l`: Matrix dimension.
/// - `skip_r`: Removed row.
/// - `skip_c`: Removed column.
/// # Returns
/// - `f64`: Cofactor value.
#[cube]
pub(crate) fn cofactor_minor(
    d: &Array<f64>,
    l: usize,
    skip_r: usize,
    skip_c: usize,
) -> f64 {
    let mut minor = Array::<f64>::new(36usize);
    let mut p = 0usize;
    for r in 0usize..l {
        if r != skip_r {
            for c in 0usize..l {
                if c != skip_c {
                    minor[p] = d[r * l + c];
                    p += 1usize;
                }
            }
        }
    }
    let sign = if ((skip_r + skip_c) & 1usize) == 0usize {
        1.0
    } else {
        -1.0
    };
    sign * det_or_zero(&minor, l - 1usize)
}
