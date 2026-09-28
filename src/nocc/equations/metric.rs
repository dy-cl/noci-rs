// nocc/equations/metric.rs
//! Raw FOIS metric.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::nocc::setup::ReferenceState;
use crate::nocc::space::{Excitation, Spaces};
use crate::nocc::terms::{TermEvaluator, assemble_matrix, overlap_blocks};

/// Build the raw FOIS metric `S_{\mu\nu} = \langle\Phi|\hat\tau_\mu^\dagger\hat\tau_\nu|\Phi\rangle`.
/// The metric involves at most the four-body RDM, which the reference provides exactly, so it is
/// evaluated without cumulant truncation.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// # Returns:
/// - `Array2<f64>`: Metric over the raw excitation list.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eq. (37) and Appendix C.
pub(in crate::nocc) fn metric_matrix(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
) -> Array2<f64> {
    assemble_matrix(
        spaces,
        excitations,
        evaluator,
        TermEvaluator::exact_table_plan,
        overlap_blocks(),
        &reference.tensors(spaces, None),
    )
}
