// nocc/space/fois.rs
//! Canonically orthogonalised first-order interacting space.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::input::NOCCMCOptions;
use crate::maths::linalg::{block_loewdin_x, symmetric_blocks};
use crate::nocc::equations::metric_matrix;
use crate::nocc::setup::ReferenceState;
use crate::nocc::terms::TermEvaluator;

// Parent/sibling imports.
use super::excitations::Excitation;
use super::orbitals::Spaces;

/// Raw metric and orthogonalised FOIS basis used by the amplitude solver.
pub(crate) struct FoisBasis {
    /// Raw spin-free FOIS metric S.
    pub metric: Array2<f64>,
    /// Canonical FOIS transformation `Y = X`, with `X^\dagger S X = I`.
    pub y: Array2<f64>,
}

/// Build the FOIS basis by canonical orthogonalisation of the raw excitation metric,
/// `X = U_+ \Lambda_+^{-1/2}` over the eigenvalues of `S` above `fois_tol`.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `options`: Metric eigenvalue threshold.
/// # Returns:
/// - `FoisBasis`: Raw metric and the orthogonalised FOIS basis `Y`.
/// # References
/// - Lee, Tew and Huynh, arXiv:2607.10007 (2026), Eqs. (19)-(21).
pub(crate) fn build_fois_basis(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
    options: &NOCCMCOptions,
) -> FoisBasis {
    // Raw FOIS metric `S_{\mu\nu} = \langle E_\mu^\dagger E_\nu\rangle` from its class-pair blocks.
    let s = metric_matrix(reference, spaces, excitations, evaluator);

    // The metric is block diagonal through its Kronecker deltas, so Löwdin orthogonalisation
    // removes its small eigenmodes block by block.
    let blocks = symmetric_blocks(&s);
    let y = block_loewdin_x(&s, &blocks, options.fois_tol);

    FoisBasis { metric: s, y }
}
