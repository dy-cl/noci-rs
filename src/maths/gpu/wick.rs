// maths/gpu/wick.rs
//! CubeCL contraction-determinant primitives shared by GPU scientific kernels.

// External crate imports.
use cubecl::prelude::*;

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
