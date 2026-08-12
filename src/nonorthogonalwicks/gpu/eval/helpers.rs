// nonorthogonalwicks/gpu/eval/helpers.rs
//! GPU helper routines for nonorthogonal Wick evaluation.

// External crate imports.
use cubecl::prelude::*;

/// Extract one fundamental-contraction assignment from a packed distribution.
/// # Arguments:
/// - `bits`: Packed assignments.
/// - `k`: Assignment index.
/// # Returns
/// - `usize`: `m_k \in \{0,1\}.`
#[cube]
pub(crate) fn bit(
    bits: u32,
    k: usize,
) -> usize {
    (usize::cast_from(bits) >> k) & 1usize
}

/// Evaluate the rank-one determinant.
/// # Arguments:
/// - `a00`: Matrix entry.
/// # Returns
/// - `f64`: Determinant.
#[cube]
pub(crate) fn det1(a00: f64) -> f64 {
    a00
}

/// Evaluate the rank-two determinant.
/// # Arguments:
/// - `a00`: Row 0 column 0.
/// - `a01`: Row 0 column 1.
/// - `a10`: Row 1 column 0.
/// - `a11`: Row 1 column 1.
/// # Returns
/// - `f64`: Determinant.
#[cube]
pub(crate) fn det2(
    a00: f64,
    a01: f64,
    a10: f64,
    a11: f64,
) -> f64 {
    a00 * a11 - a01 * a10
}

/// Evaluate the rank-three determinant by first-row cofactor expansion.
/// # Arguments:
/// - `d`: Row-major rank-three matrix.
/// # Returns
/// - `f64`: Determinant.
#[cube]
pub(crate) fn det3(d: &Array<f64>) -> f64 {
    d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
        + d[2] * (d[3] * d[7] - d[4] * d[6])
}

/// Evaluate the rank-four determinant by first-row cofactor expansion.
/// # Arguments:
/// - `d`: Row-major rank-four matrix.
/// # Returns
/// - `f64`: Determinant.
#[cube]
pub(crate) fn det4(d: &Array<f64>) -> f64 {
    let m0 = det3_minor4(d, 0usize, 0usize);
    let m1 = det3_minor4(d, 0usize, 1usize);
    let m2 = det3_minor4(d, 0usize, 2usize);
    let m3 = det3_minor4(d, 0usize, 3usize);
    d[0] * m0 - d[1] * m1 + d[2] * m2 - d[3] * m3
}

/// Evaluate a rank-three minor of a rank-four matrix.
/// # Arguments:
/// - `d`: Row-major rank-four matrix.
/// - `skip_r`: Removed row.
/// - `skip_c`: Removed column.
/// # Returns
/// - `f64`: Minor determinant.
#[cube]
pub(crate) fn det3_minor4(
    d: &Array<f64>,
    skip_r: usize,
    skip_c: usize,
) -> f64 {
    let mut m = Array::<f64>::new(9usize);
    let mut p = 0usize;
    for r in 0usize..4usize {
        if r != skip_r {
            for c in 0usize..4usize {
                if c != skip_c {
                    m[p] = d[r * 4usize + c];
                    p += 1usize;
                }
            }
        }
    }
    det3(&m)
}

/// Fill the row-major rank-three cofactor matrix.
/// # Arguments:
/// - `d`: Row-major rank-three matrix.
/// - `cof`: Output row-major cofactor matrix.
/// # Returns
/// - `f64`: Determinant.
#[cube]
pub(crate) fn adjugate_transpose3(
    d: &Array<f64>,
    cof: &mut Array<f64>,
) -> f64 {
    let a00 = d[0];
    let a01 = d[1];
    let a02 = d[2];
    let a10 = d[3];
    let a11 = d[4];
    let a12 = d[5];
    let a20 = d[6];
    let a21 = d[7];
    let a22 = d[8];

    cof[0] = a11 * a22 - a12 * a21;
    cof[1] = a12 * a20 - a10 * a22;
    cof[2] = a10 * a21 - a11 * a20;
    cof[3] = a02 * a21 - a01 * a22;
    cof[4] = a00 * a22 - a02 * a20;
    cof[5] = a01 * a20 - a00 * a21;
    cof[6] = a01 * a12 - a02 * a11;
    cof[7] = a02 * a10 - a00 * a12;
    cof[8] = a00 * a11 - a01 * a10;

    a00 * cof[0] + a01 * cof[1] + a02 * cof[2]
}

/// Fill the row-major rank-four cofactor matrix.
/// # Arguments:
/// - `d`: Row-major rank-four matrix.
/// - `cof`: Output row-major cofactor matrix.
/// # Returns
/// - `f64`: Determinant.
#[cube]
pub(crate) fn adjugate_transpose4(
    d: &Array<f64>,
    cof: &mut Array<f64>,
) -> f64 {
    for r in 0usize..4usize {
        for c in 0usize..4usize {
            let sign = if ((r + c) & 1usize) == 0usize {
                1.0
            } else {
                -1.0
            };
            cof[r * 4usize + c] = sign * det3_minor4(d, r, c);
        }
    }
    det4(d)
}

/// Evaluate a small row-major determinant used by general fallback paths.
/// # Arguments:
/// - `d`: Row-major matrix.
/// - `n`: Matrix dimension.
/// # Returns
/// - `f64`: Determinant, or zero for unsupported dimensions.
#[cube]
pub(crate) fn det_or_zero(
    d: &Array<f64>,
    n: usize,
) -> f64 {
    let mut value: f64 = 0.0;
    if n == 0usize {
        value = 1.0;
    } else if n == 1usize {
        value = det1(d[0]);
    } else if n == 2usize {
        value = det2(d[0], d[1], d[2], d[3]);
    } else if n == 3usize {
        value = det3(d);
    } else if n == 4usize {
        value = det4(d);
    } else {
        value = det_elim(d, n);
    }
    value
}

/// Evaluate a small determinant by local Gaussian elimination for fallback ranks.
/// # Arguments:
/// - `d`: Row-major matrix.
/// - `n`: Matrix dimension.
/// # Returns
/// - `f64`: Determinant.
#[cube]
pub(crate) fn det_elim(
    d: &Array<f64>,
    n: usize,
) -> f64 {
    let mut a = Array::<f64>::new(36usize);
    for i in 0usize..(n * n) {
        a[i] = d[i];
    }
    let mut det: f64 = 1.0;
    let mut singular = false;
    for k in 0usize..n {
        let pivot = a[k * n + k];
        if pivot == 0.0 {
            singular = true;
        }
        if !singular {
            det *= pivot;
            for i in (k + 1usize)..n {
                let factor = a[i * n + k] / pivot;
                for j in (k + 1usize)..n {
                    a[i * n + j] -= factor * a[k * n + j];
                }
            }
        }
    }
    let mut value: f64 = 0.0;
    if singular {
        value = 0.0;
    } else {
        value = det;
    }
    value
}

/// Evaluate `\Delta_c = \det D[c\rightarrow N]-\det D` using the cofactor column.
/// # Arguments:
/// - `n`: Matrix dimension.
/// - `old`: Row-major original determinant.
/// - `cof`: Row-major cofactor matrix.
/// - `col`: Replaced column.
/// - `new_col`: Replacement column entries.
/// # Returns
/// - `f64`: Determinant correction.
#[cube]
pub(crate) fn column_replacement_correction(
    n: usize,
    old: &Array<f64>,
    cof: &Array<f64>,
    col: usize,
    new_col: &Array<f64>,
) -> f64 {
    let mut correction = 0.0;
    for r in 0usize..n {
        let i = r * n + col;
        correction += (new_col[r] - old[i]) * cof[i];
    }
    correction
}
