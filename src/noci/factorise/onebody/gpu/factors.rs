// noci/factorise/onebody/gpu/factors.rs
//! GPU factor generation for factorised one-body NOCI operator contractions.

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::noci::types::{NOCIData, NOCIScalar};

/// Build same-spin `S` and `F` factor rows from ordered Wick data on the device.
/// This is the GPU mirror of the CPU `build_spin_one_body_factors` operation and must preserve
/// ordered-parent-pair and `target_left` semantics without introducing conjugate shortcuts.
/// # Arguments:
/// - `data`: Shared NOCI determinant data used to derive GPU-resident Wick and excitation data.
/// - `x`: Source vector, used only to bind the scalar type for kernel specialisation.
/// # Returns
/// - `()`: Factor panels are intended to remain resident on the device.
pub(crate) fn build_spin_one_body_factors<T: NOCIScalar>(
    _data: &NOCIData<'_, T>,
    _x: &Array1<T>,
) {
}
