// nocc/overlap.rs

// Standard library imports.
use std::collections::BTreeMap;

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::nocc::common::{class_name, excitation_indices};
use crate::nocc::context::EvaluationContext;
use crate::nocc::contract::{FactorBlocks, evaluate_dense_table};
use crate::nocc::loader::overlap_terms;
use crate::nocc::residual::orbital_positions;
use crate::nocc::space::excitation_class;
use crate::nocc::terms::OverlapTermSet;

/// Assemble a symmetric matrix over the raw excitations from its generated class-pair blocks.
/// Each block is evaluated once as a dense tensor over its left then right free indices, and
/// every listed pair of excitations is gathered from it. Class pairs without a block couple to
/// zero, and each block also fills its transpose.
/// # Arguments:
/// - `ctx`: Reference evaluation context.
/// - `set`: Generated class-pair blocks, such as the metric or the Dyall coupling.
/// # Returns:
/// - `Array2<f64>`: Matrix over the raw excitation list.
pub(crate) fn assemble_block_matrix(
    ctx: &EvaluationContext<'_>,
    set: &OverlapTermSet,
) -> Array2<f64> {
    // Group the raw excitations by class.
    let mut members = BTreeMap::<&'static str, Vec<usize>>::new();
    for (mu, &ex) in ctx.excitations.iter().enumerate() {
        members
            .entry(class_name(excitation_class(ctx.spaces, ex)))
            .or_default()
            .push(mu);
    }

    // Blocks whose classes both occur, with dense factor blocks shared by all of them.
    let blocks = set
        .blocks
        .values()
        .filter(|b| members.contains_key(b.left.as_str()) && members.contains_key(b.right.as_str()))
        .collect::<Vec<_>>();
    let plans = blocks
        .iter()
        .map(|b| ctx.plans.table_plan((&b.terms, &b.indices)))
        .collect::<Vec<_>>();
    let factors = FactorBlocks::build_factor_blocks(
        &plans.iter().map(|p| p.as_ref()).collect::<Vec<_>>(),
        &ctx.tensors(None),
    );

    let positions = orbital_positions(ctx.spaces);
    let n = ctx.excitations.len();
    let mut out = Array2::<f64>::zeros((n, n));

    for (block, plan) in blocks.iter().zip(&plans) {
        let free = [block.left_free.as_slice(), block.right_free.as_slice()].concat();
        let dense = evaluate_dense_table((&block.terms, &block.indices), &free, plan, &factors);

        // Gather every pair from the left then right free-index tuple.
        for &mu in &members[block.left.as_str()] {
            let (left, nl) = excitation_indices(ctx.excitations[mu]);
            for &nu in &members[block.right.as_str()] {
                let (right, nr) = excitation_indices(ctx.excitations[nu]);
                let flat = left[..nl]
                    .iter()
                    .chain(&right[..nr])
                    .zip(&dense.dims)
                    .fold(0, |acc, (&p, &d)| acc * d + positions[p]);
                out[(mu, nu)] = dense.data[flat];
                out[(nu, mu)] = dense.data[flat];
            }
        }
    }

    out
}

/// Build the raw FOIS metric `S_{\mu\nu} = \langle\Phi|\hat\tau_\mu^\dagger\hat\tau_\nu|\Phi\rangle`.
/// # Arguments:
/// - `ctx`: Reference evaluation context.
/// # Returns:
/// - `Array2<f64>`: Metric over the raw excitation list.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eq. (37) and Appendix C.
pub(crate) fn metric_matrix(ctx: &EvaluationContext<'_>) -> Array2<f64> {
    assemble_block_matrix(ctx, overlap_terms())
}
