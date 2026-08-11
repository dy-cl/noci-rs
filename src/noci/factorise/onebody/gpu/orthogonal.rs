// noci/factorise/onebody/gpu/orthogonal.rs
//! GPU same-parent orthogonal one-body NOCI operator contractions.

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::noci::types::NOCIScalar;

/// Apply same-parent orthogonal `Y^P += (F^{PP}+\lambda S^{PP})D^P` on the device.
/// # Arguments:
/// - `x`: Source determinant vector.
/// - `lambda`: Scalar overlap shift.
/// # Returns
/// - `Array1<T>`: Same-parent orthogonal contribution.
pub(crate) fn apply_one_body_orthogonal<T: NOCIScalar>(
    _x: &Array1<T>,
    _lambda: T,
) -> Array1<T> {
    eprintln!("GPU orthogonal one-body application is not implemented yet");
    std::process::exit(1);
}

/// Scatter one same-occupation orthogonal contribution to assigned target rows.
/// # Arguments:
/// - `x`: Source determinant vector.
/// # Returns
/// - `()`: Adds same-occupation contributions into the device output buffer.
pub(crate) fn scatter_orthogonal_group<T: NOCIScalar>(_x: &Array1<T>) {
    eprintln!("GPU orthogonal occupation-group scatter is not implemented yet");
    std::process::exit(1);
}

/// Apply all alpha single-excitation orthogonal Fock couplings from one source determinant.
/// # Arguments:
/// - `x`: Source determinant vector.
/// # Returns
/// - `()`: Adds alpha single-excitation Fock contributions into the device output buffer.
pub(crate) fn apply_orthogonal_alpha_singles<T: NOCIScalar>(_x: &Array1<T>) {
    eprintln!("GPU orthogonal alpha singles application is not implemented yet");
    std::process::exit(1);
}

/// Apply all beta single-excitation orthogonal Fock couplings from one source determinant.
/// # Arguments:
/// - `x`: Source determinant vector.
/// # Returns
/// - `()`: Adds beta single-excitation Fock contributions into the device output buffer.
pub(crate) fn apply_orthogonal_beta_singles<T: NOCIScalar>(_x: &Array1<T>) {
    eprintln!("GPU orthogonal beta singles application is not implemented yet");
    std::process::exit(1);
}
