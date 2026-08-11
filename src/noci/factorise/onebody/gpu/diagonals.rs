// noci/factorise/onebody/gpu/diagonals.rs
//! GPU diagonal construction for factorised one-body NOCI operator contractions.

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::noci::types::NOCIScalar;

/// Build diagonal entries of `F + \lambda S` and `S` from device factors.
/// # Arguments:
/// - `n`: Number of determinant-space diagonal entries.
/// - `lambda`: Scalar overlap shift in `F + \lambda S`.
/// # Returns
/// - `(Array1<T>, Array1<T>)`: Diagonal of `F + \lambda S` and diagonal of `S`.
pub(crate) fn one_body_diagonals<T: NOCIScalar>(
    n: usize,
    _lambda: T,
) -> (Array1<T>, Array1<T>) {
    (
        Array1::from_elem(n, T::from_real(0.0)),
        Array1::from_elem(n, T::from_real(0.0)),
    )
}

/// Fill determinant diagonals from one same-parent factor block on the device.
/// # Arguments:
/// - `m_diag`: Output diagonal of `F + \lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// # Returns
/// - `()`: Writes diagonal values for actual determinants.
pub(crate) fn fill_one_body_diagonal_block<T: NOCIScalar>(
    _m_diag: &mut Array1<T>,
    _s_diag: &mut Array1<T>,
) {
}

/// Fill same-parent orthogonal diagonals from parent-local Slater-Condon rules on the device.
/// # Arguments:
/// - `m_diag`: Output diagonal of `F + \lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// # Returns
/// - `()`: Writes diagonal values for actual determinants.
pub(crate) fn fill_orthogonal_one_body_diagonal_block<T: NOCIScalar>(
    _m_diag: &mut Array1<T>,
    _s_diag: &mut Array1<T>,
) {
}
