// nocc/equations/residual.rs
//! GNOCC amplitude residual.

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::nocc::setup::ReferenceState;
use crate::nocc::space::{DenseAmplitudes, Excitation, Spaces};
use crate::nocc::terms::{TermEvaluator, assemble_vector, residual_classes};

/// Build the full residual
/// `R_\mu = \langle\Phi|\hat\tau_\mu^\dagger\hat H\{1 + \hat T + \tfrac12\hat T^2\}|\Phi\rangle_c`.
/// The orders `R_0`, `R_1` and `R_2` are summed in that order into one dense block per class.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `amplitudes`: Dense amplitude tensors of the current cluster operator.
/// # Returns:
/// - `Array1<f64>`: Residual in the raw excitation basis.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eq. (36).
pub(in crate::nocc) fn residual_vector(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
    amplitudes: &DenseAmplitudes,
) -> Array1<f64> {
    let orders = [
        residual_classes(0),
        residual_classes(1),
        residual_classes(2),
    ];
    assemble_vector(
        spaces,
        excitations,
        evaluator,
        TermEvaluator::table_plan,
        &orders,
        &reference.tensors(spaces, Some(amplitudes)),
    )
}
