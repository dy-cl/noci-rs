// maths/gpu/wick.rs
//! CubeCL contraction-determinant primitives shared by GPU scientific kernels.

// External crate imports.
use cubecl::prelude::*;

/// Calculate the determinant of a `1 x 1` matrix.
/// # Arguments:
/// - `a00`: Matrix element `(0, 0)`.
/// # Returns
/// - `f64`: Determinant of the matrix.
#[cube]
pub(crate) fn det1(a00: f64) -> f64 {
    a00
}

/// Calculate the determinant of a `2 x 2` matrix from scalar row-major elements.
/// # Arguments:
/// - `a00`: Matrix element `(0, 0)`.
/// - `a01`: Matrix element `(0, 1)`.
/// - `a10`: Matrix element `(1, 0)`.
/// - `a11`: Matrix element `(1, 1)`.
/// # Returns
/// - `f64`: Determinant of the matrix.
#[cube]
pub(crate) fn det2(
    a00: f64,
    a01: f64,
    a10: f64,
    a11: f64,
) -> f64 {
    a00 * a11 - a01 * a10
}

/// Calculate the determinant of a row-major `3 x 3` matrix.
/// # Arguments:
/// - `d`: Row-major `3 x 3` matrix.
/// # Returns
/// - `f64`: Determinant of the matrix.
#[cube]
pub(crate) fn det3(d: &Array<f64>) -> f64 {
    d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
        + d[2] * (d[3] * d[7] - d[4] * d[6])
}

/// Calculate the determinant of a `3 x 3` matrix from scalar row-major elements.
/// # Arguments:
/// - `a00`: Matrix element `(0, 0)`.
/// - `a01`: Matrix element `(0, 1)`.
/// - `a02`: Matrix element `(0, 2)`.
/// - `a10`: Matrix element `(1, 0)`.
/// - `a11`: Matrix element `(1, 1)`.
/// - `a12`: Matrix element `(1, 2)`.
/// - `a20`: Matrix element `(2, 0)`.
/// - `a21`: Matrix element `(2, 1)`.
/// - `a22`: Matrix element `(2, 2)`.
/// # Returns
/// - `f64`: Determinant of the matrix.
#[cube]
pub(crate) fn det3_scalar(
    a00: f64,
    a01: f64,
    a02: f64,
    a10: f64,
    a11: f64,
    a12: f64,
    a20: f64,
    a21: f64,
    a22: f64,
) -> f64 {
    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
}

/// Calculate a rank-three minor of a row-major `4 x 4` matrix.
/// # Arguments:
/// - `d`: Row-major `4 x 4` matrix.
/// - `skip_r`: Removed row.
/// - `skip_c`: Removed column.
/// # Returns
/// - `f64`: Minor determinant.
#[cube]
fn det3_minor4(
    d: &Array<f64>,
    skip_r: usize,
    skip_c: usize,
) -> f64 {
    let mut minor = Array::<f64>::new(9usize);
    let mut p = 0usize;
    for r in 0usize..4usize {
        if r != skip_r {
            for c in 0usize..4usize {
                if c != skip_c {
                    minor[p] = d[r * 4usize + c];
                    p += 1usize;
                }
            }
        }
    }
    det3(&minor)
}

/// Calculate the determinant of a row-major `4 x 4` matrix by first-row cofactor expansion.
/// # Arguments:
/// - `d`: Row-major `4 x 4` matrix.
/// # Returns
/// - `f64`: Determinant of the matrix.
#[cube]
pub(crate) fn det4(d: &Array<f64>) -> f64 {
    let m0 = det3_minor4(d, 0usize, 0usize);
    let m1 = det3_minor4(d, 0usize, 1usize);
    let m2 = det3_minor4(d, 0usize, 2usize);
    let m3 = det3_minor4(d, 0usize, 3usize);
    d[0] * m0 - d[1] * m1 + d[2] * m2 - d[3] * m3
}

/// Calculate the determinant and row-major cofactor matrix of a `1 x 1` matrix.
/// # Arguments:
/// - `d`: Row-major `1 x 1` matrix.
/// - `cof`: Scratch space for writing `\operatorname{cof}(d)` row-major.
/// # Returns
/// - `f64`: Determinant of `d`.
#[cube]
pub(crate) fn adjugate_transpose1(
    d: &Array<f64>,
    cof: &mut Array<f64>,
) -> f64 {
    cof[0] = 1.0;
    d[0]
}

/// Calculate the determinant and row-major cofactor matrix of a `2 x 2` matrix.
/// # Arguments:
/// - `d`: Row-major `2 x 2` matrix.
/// - `cof`: Scratch space for writing `\operatorname{cof}(d)` row-major.
/// # Returns
/// - `f64`: Determinant of `d`.
#[cube]
pub(crate) fn adjugate_transpose2(
    d: &Array<f64>,
    cof: &mut Array<f64>,
) -> f64 {
    cof[0] = d[3];
    cof[1] = -d[2];
    cof[2] = -d[1];
    cof[3] = d[0];
    det2(d[0], d[1], d[2], d[3])
}

/// Calculate the determinant and row-major cofactor matrix of a `3 x 3` matrix.
/// # Arguments:
/// - `d`: Row-major `3 x 3` matrix.
/// - `cof`: Scratch space for writing `\operatorname{cof}(d)` row-major.
/// # Returns
/// - `f64`: Determinant of `d`.
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

/// Calculate the determinant and row-major cofactor matrix of a `4 x 4` matrix.
/// The determinant is obtained from the completed first cofactor row.
/// # Arguments:
/// - `d`: Row-major `4 x 4` matrix.
/// - `cof`: Scratch space for writing `\operatorname{cof}(d)` row-major.
/// # Returns
/// - `f64`: Determinant of `d`.
#[cube]
pub(crate) fn adjugate_transpose4(
    d: &Array<f64>,
    cof: &mut Array<f64>,
) -> f64 {
    let a00 = d[0];
    let a01 = d[1];
    let a02 = d[2];
    let a03 = d[3];
    let a10 = d[4];
    let a11 = d[5];
    let a12 = d[6];
    let a13 = d[7];
    let a20 = d[8];
    let a21 = d[9];
    let a22 = d[10];
    let a23 = d[11];
    let a30 = d[12];
    let a31 = d[13];
    let a32 = d[14];
    let a33 = d[15];

    // cof[d]_{rc}=(-1)^{r+c}det d[r|c], stored row-major without transposition.
    cof[0] = det3_scalar(a11, a12, a13, a21, a22, a23, a31, a32, a33);
    cof[1] = -det3_scalar(a10, a12, a13, a20, a22, a23, a30, a32, a33);
    cof[2] = det3_scalar(a10, a11, a13, a20, a21, a23, a30, a31, a33);
    cof[3] = -det3_scalar(a10, a11, a12, a20, a21, a22, a30, a31, a32);
    cof[4] = -det3_scalar(a01, a02, a03, a21, a22, a23, a31, a32, a33);
    cof[5] = det3_scalar(a00, a02, a03, a20, a22, a23, a30, a32, a33);
    cof[6] = -det3_scalar(a00, a01, a03, a20, a21, a23, a30, a31, a33);
    cof[7] = det3_scalar(a00, a01, a02, a20, a21, a22, a30, a31, a32);
    cof[8] = det3_scalar(a01, a02, a03, a11, a12, a13, a31, a32, a33);
    cof[9] = -det3_scalar(a00, a02, a03, a10, a12, a13, a30, a32, a33);
    cof[10] = det3_scalar(a00, a01, a03, a10, a11, a13, a30, a31, a33);
    cof[11] = -det3_scalar(a00, a01, a02, a10, a11, a12, a30, a31, a32);
    cof[12] = -det3_scalar(a01, a02, a03, a11, a12, a13, a21, a22, a23);
    cof[13] = det3_scalar(a00, a02, a03, a10, a12, a13, a20, a22, a23);
    cof[14] = -det3_scalar(a00, a01, a03, a10, a11, a13, a20, a21, a23);
    cof[15] = det3_scalar(a00, a01, a02, a10, a11, a12, a20, a21, a22);

    // det d = sum_c d_{0c} cof(d)_{0c}.
    a00 * cof[0] + a01 * cof[1] + a02 * cof[2] + a03 * cof[3]
}
