// nonorthogonalwicks/gpu/eval/helpers.rs
//! GPU helper routines for nonorthogonal Wick evaluation.

// External crate imports.
use cubecl::prelude::*;

// Crate-root imports.
use crate::maths::gpu::wick::{det1, det2, det3, det4};

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
    let mut det = 1.0f64;
    let mut active = true;
    for k in 0usize..n {
        let pivot = a[k * n + k];
        if active {
            if pivot == 0.0 {
                det = 0.0f64;
                active = false;
            } else {
                det *= pivot;
                for i in (k + 1usize)..n {
                    let factor = a[i * n + k] / pivot;
                    for j in (k + 1usize)..n {
                        a[i * n + j] -= factor * a[k * n + j];
                    }
                }
            }
        }
    }
    det
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
