// nocc/equations/residual.rs
//! GNOCC amplitude residual.

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::NOCIScalar;
use crate::nocc::setup::ReferenceState;
use crate::nocc::space::{DenseAmplitudes, Excitation, Spaces};
use crate::nocc::terms::{Amplitudes, TermEvaluator, assemble_vector, residual_classes};

/// Build the residual
/// `R_\mu = \langle\Phi|\hat\tau_\mu^\dagger\hat H\{1 + \hat T + \tfrac12\hat T^2\}|\Phi\rangle_c`,
/// or only its parts `R_k` of the requested orders `k` in the amplitudes, summed in the given
/// order into one dense block per class.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `amplitudes`: Dense amplitude tensors of the current cluster operator.
/// - `orders`: Amplitude orders `k` of the parts `R_k` to include, `[0, 1, 2]` for the full residual.
/// # Returns:
/// - `Array1<T>`: Residual in the raw excitation basis, in the amplitude scalar type.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eq. (36).
pub(in crate::nocc) fn residual_vector<T: NOCIScalar>(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
    amplitudes: &DenseAmplitudes<T>,
    orders: &[usize],
) -> Array1<T> {
    let orders = orders
        .iter()
        .map(|&k| residual_classes(k))
        .collect::<Vec<_>>();
    assemble_vector(
        spaces,
        excitations,
        evaluator,
        TermEvaluator::table_plan,
        &orders,
        &reference.tensors(spaces, Some(Amplitudes::of(amplitudes))),
    )
}
