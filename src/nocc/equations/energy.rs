// nocc/equations/energy.rs
//! GNOCC correlation energy.

// Crate-root imports.
use crate::NOCIScalar;
use crate::nocc::setup::ReferenceState;
use crate::nocc::space::{DenseAmplitudes, Spaces};
use crate::nocc::terms::{Amplitudes, TermEvaluator, assemble_scalar, e1_terms, e2_terms};

/// Evaluate the correlation energy
/// `E - E_0 = \langle\Phi|\hat H\hat T|\Phi\rangle_c + \tfrac12\langle\Phi|\hat H\{\hat T\hat T\}|\Phi\rangle_c`.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `evaluator`: Term-table evaluator.
/// - `amplitudes`: Dense amplitude tensors of the current cluster operator.
/// # Returns:
/// - `T`: Correlation energy, in the amplitude scalar type.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eq. (35).
pub(in crate::nocc) fn correlation_energy<T: NOCIScalar>(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    evaluator: &TermEvaluator,
    amplitudes: &DenseAmplitudes<T>,
) -> T {
    assemble_scalar(
        evaluator,
        TermEvaluator::table_plan,
        &[e1_terms(), e2_terms()],
        &reference.tensors(spaces, Some(Amplitudes::of(amplitudes))),
    )
}
