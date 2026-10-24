// nocc/equations/jacobian.rs
//! GNOCC amplitude Jacobian acting on a set of amplitude directions.

// External crate imports.
use ndarray::{Array1, Array2, s};

// Crate-root imports.
use crate::NOCIScalar;
use crate::nocc::setup::ReferenceState;
use crate::nocc::space::{DenseAmplitudes, Excitation, Spaces, dense_amplitudes};
use crate::nocc::terms::{Amplitudes, TermEvaluator, assemble_columns, jacobian_classes};

/// Largest number of directions evaluated together. Every intermediate carrying the batch label
/// grows with the batch, so the batch bounds the memory of the shared contraction graph.
pub(in crate::nocc) const COLUMNS: usize = 8;

/// Build the Jacobian columns `[J v_c]_\mu = \sum_\nu (\partial R_\mu / \partial t_\nu) v_{\nu c}`
/// of the residual `R = R_0 + R_1(t) + R_2(t, t)` at amplitudes `t`, for every direction `v_c`, as
/// `J v_c = J_1 v_c + J_2(t) v_c` with `J_1 v_c = R_1(v_c)` and `J_2(t) v_c = \partial_{v_c} R_2(t, t)`,
/// or only the requested parts. The directions are evaluated in batches of the evaluator's batch
/// size, the last padded with zero directions so every batch keeps the planned extents.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator, whose batch size fixes the directions per batch.
/// - `amplitudes`: Dense amplitude tensors of the current cluster operator.
/// - `directions`: Dense amplitude tensors of every direction `v_c`.
/// - `orders`: Residual orders `k` of the Jacobian parts `J_k` to include, `[1, 2]` for `J`.
/// # Returns:
/// - `Array2<T>`: Jacobian columns over the raw excitations and the directions.
pub(in crate::nocc) fn jacobian_columns<T: NOCIScalar>(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
    amplitudes: &DenseAmplitudes<T>,
    directions: &[DenseAmplitudes<f64>],
    orders: &[usize],
) -> Array2<T> {
    let width = evaluator.batch();
    let orders = orders
        .iter()
        .map(|&k| jacobian_classes(k))
        .collect::<Vec<_>>();
    let zero = dense_amplitudes(
        spaces,
        excitations,
        &Array1::<f64>::zeros(excitations.len()),
    );
    let mut out = Array2::<T>::zeros((excitations.len(), directions.len()));

    for (k, chunk) in directions.chunks(width).enumerate() {
        // Pad the last batch with zero directions, whose columns vanish and are dropped.
        let mut batch = chunk.iter().collect::<Vec<_>>();
        batch.resize(width, &zero);

        let mut tensors = reference.tensors(spaces, Some(Amplitudes::of(amplitudes)));
        tensors.directions = Some(&batch);
        let columns = assemble_columns(
            spaces,
            excitations,
            evaluator,
            TermEvaluator::table_plan,
            &orders,
            &tensors,
            width,
        );

        let start = k * width;
        out.slice_mut(s![.., start..start + chunk.len()])
            .assign(&columns.slice(s![.., ..chunk.len()]));
    }

    out
}
