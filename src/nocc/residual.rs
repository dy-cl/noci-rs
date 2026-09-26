// nocc/residual.rs

// Standard library imports.
use std::collections::BTreeMap;

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::nocc::common::{Tensors, class_name, excitation_indices};
use crate::nocc::contract::{DenseBlock, FactorBlocks, TermEvaluator, evaluate_dense_table};
use crate::nocc::loader::{r0_terms, r1_terms, r2_terms};
use crate::nocc::reference::ReferenceState;
use crate::nocc::space::{
    DenseAmplitudes, Excitation, ExcitationManifold, Spaces, excitation_class,
};
use crate::nocc::terms::ResidualTermSet;

/// Return the position of every orbital within its own orbital space.
/// # Arguments:
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// # Returns:
/// - `Vec<usize>`: Position of each MO in the core, active or virtual list.
pub(crate) fn orbital_positions(spaces: &Spaces) -> Vec<usize> {
    let mut positions = vec![0; spaces.nmo];
    for list in [&spaces.core, &spaces.active, &spaces.virtuals] {
        for (i, &p) in list.iter().enumerate() {
            positions[p] = i;
        }
    }
    positions
}

/// Evaluate the sum of several residual orders at every raw excitation.
/// Each excitation class is evaluated as one dense block over its free-index spaces, summed
/// over the requested orders, and its listed elements are then gathered.
/// # Arguments:
/// - `manifold`: Orbital spaces and raw excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `sets`: Residual term tables of the orders to sum.
/// - `tensors`: Runtime tensors, including the amplitudes when any order needs them.
/// # Returns:
/// - `Array1<f64>`: Summed residual in the raw excitation basis.
/// # Panics
/// - Panics if a table has no terms for an excitation class in the list.
fn residual_orders(
    manifold: &ExcitationManifold<'_>,
    evaluator: &TermEvaluator,
    sets: &[&ResidualTermSet],
    tensors: &Tensors<'_>,
) -> Array1<f64> {
    // Group the raw excitations by class.
    let mut classes = BTreeMap::<&'static str, Vec<usize>>::new();
    for (mu, &ex) in manifold.excitations.iter().enumerate() {
        classes
            .entry(class_name(excitation_class(manifold.spaces, ex)))
            .or_default()
            .push(mu);
    }

    // Dense factor blocks shared by every table of every class.
    let plans = classes
        .keys()
        .flat_map(|&name| sets.iter().map(move |set| &set.classes[name]))
        .map(|c| evaluator.table_plan((&c.terms, &c.indices)))
        .collect::<Vec<_>>();
    let blocks = FactorBlocks::build_factor_blocks(
        &plans.iter().map(|p| p.as_ref()).collect::<Vec<_>>(),
        tensors,
    );

    let positions = orbital_positions(manifold.spaces);
    let mut out = Array1::<f64>::zeros(manifold.excitations.len());

    for (name, members) in &classes {
        // `R = \sum_n R_n` over the requested orders, as one dense block.
        let mut block: Option<DenseBlock> = None;
        for set in sets {
            let class = &set.classes[*name];
            let table = (class.terms.as_slice(), class.indices.as_slice());
            let plan = evaluator.table_plan(table);
            let part = evaluate_dense_table(table, &class.free, &plan, &blocks);
            match &mut block {
                Some(b) => {
                    for (x, y) in b.data.iter_mut().zip(part.data) {
                        *x += y;
                    }
                }
                None => block = Some(part),
            }
        }
        let block = block.expect("at least one residual order");

        // Gather each listed excitation from its free-index tuple.
        for &mu in members {
            let ex: Excitation = manifold.excitations[mu];
            let (values, n) = excitation_indices(ex);
            let flat = values[..n]
                .iter()
                .zip(&block.dims)
                .fold(0, |acc, (&p, &d)| acc * d + positions[p]);
            out[mu] = block.data[flat];
        }
    }

    out
}

/// Build the zeroth-order residual `R_{0,\mu} = \langle\Phi|\hat\tau_\mu^\dagger\hat H|\Phi\rangle_c`.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `manifold`: Orbital spaces and raw excitation list.
/// - `evaluator`: Term-table evaluator.
/// # Returns:
/// - `Array1<f64>`: Zeroth-order residual in the raw excitation basis.
pub(crate) fn zeroth_order_residual(
    reference: &ReferenceState<'_>,
    manifold: &ExcitationManifold<'_>,
    evaluator: &TermEvaluator,
) -> Array1<f64> {
    residual_orders(
        manifold,
        evaluator,
        &[r0_terms()],
        &reference.tensors(manifold.spaces, None),
    )
}

/// Build the first-order residual `R_{1,\mu} = \langle\Phi|\hat\tau_\mu^\dagger\hat H\hat T|\Phi\rangle_c`,
/// linear in the amplitudes.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `manifold`: Orbital spaces and raw excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `amplitudes`: Cluster amplitude vector in the raw excitation basis.
/// # Returns:
/// - `Array1<f64>`: First-order residual in the raw excitation basis.
pub(crate) fn first_order_residual(
    reference: &ReferenceState<'_>,
    manifold: &ExcitationManifold<'_>,
    evaluator: &TermEvaluator,
    amplitudes: &Array1<f64>,
) -> Array1<f64> {
    let dense = manifold.dense_amplitudes(amplitudes);
    residual_orders(
        manifold,
        evaluator,
        &[r1_terms()],
        &reference.tensors(manifold.spaces, Some(&dense)),
    )
}

/// Build the full residual
/// `R_\mu = \langle\Phi|\hat\tau_\mu^\dagger\hat H\{1 + \hat T + \tfrac12\hat T^2\}|\Phi\rangle_c`.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `manifold`: Orbital spaces and raw excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `amplitudes`: Dense amplitude tensors of the current cluster operator.
/// # Returns:
/// - `Array1<f64>`: Residual in the raw excitation basis.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eq. (36).
pub(crate) fn residual_vector(
    reference: &ReferenceState<'_>,
    manifold: &ExcitationManifold<'_>,
    evaluator: &TermEvaluator,
    amplitudes: &DenseAmplitudes,
) -> Array1<f64> {
    residual_orders(
        manifold,
        evaluator,
        &[r0_terms(), r1_terms(), r2_terms()],
        &reference.tensors(manifold.spaces, Some(amplitudes)),
    )
}
