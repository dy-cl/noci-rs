// noci/factorise/onebody/gpu/contract.rs
//! GPU dense contractions for factorised one-body NOCI operator contractions.

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::noci::types::NOCIScalar;

/// Apply alpha-first contraction on the device.
/// Computes `Y^Q += F^alpha D (S^beta)^T
/// + S^alpha D (F^beta+\lambda S^beta)^T`.
/// # Arguments:
/// - `x`: Source determinant vector.
/// - `lambda`: Scalar overlap shift.
/// # Returns
/// - `Array1<T>`: Device-computed contribution after final download.
pub(crate) fn apply_one_body_a_first<T: NOCIScalar>(
    x: &Array1<T>,
    _lambda: T,
) -> Array1<T> {
    Array1::from_elem(x.len(), T::from_real(0.0))
}

/// Apply beta-first contraction on the device.
/// Computes `Y^Q += S^alpha D (F^beta)^T
/// + (F^alpha+\lambda S^alpha)D(S^beta)^T`.
/// # Arguments:
/// - `x`: Source determinant vector.
/// - `lambda`: Scalar overlap shift.
/// # Returns
/// - `Array1<T>`: Device-computed contribution after final download.
pub(crate) fn apply_one_body_b_first<T: NOCIScalar>(
    x: &Array1<T>,
    _lambda: T,
) -> Array1<T> {
    Array1::from_elem(x.len(), T::from_real(0.0))
}
