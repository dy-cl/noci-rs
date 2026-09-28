// nocc/terms/evaluate.rs
//! Evaluation of one planned table over whole orbital-space blocks.

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::maths::contract::{MAXLABELS, contract_tensor_pair, strided_tensor_shape};

// Parent/sibling imports.
use super::factors::FactorBlocks;
use super::plan::{TablePlan, TermTable, label_map};
use super::term::{PlannedTerm, accumulate_term};
use super::workspace::Workspace;

/// Smallest output block, in elements, evaluated as slices over its leading free indices.
const LARGEBLOCK: usize = 1 << 20;

/// Largest slice, in elements, of a large output block accumulated by one worker.
const SLICESIZE: usize = 1 << 16;

/// Smallest number of slices a large output block is split into.
const SLICES: usize = 64;

/// Dense result block over the free indices of one table.
pub(super) struct DenseBlock {
    /// Row-major elements over the free indices in table order.
    pub(super) data: Vec<f64>,
    /// Extent of every free index.
    pub(super) dims: Vec<usize>,
}

/// Evaluate one term table over whole orbital-space blocks.
/// Terms are contracted independently in parallel and their dense contributions summed. Label
/// sets are 64-bit masks, so a table may use at most 64 class-local indices; generated tables
/// use at most eight free indices plus one dummy slot per space and rank.
/// # Arguments:
/// - `table`: Terms and index spaces of the table.
/// - `free`: Class-local ids of the free indices, in output order.
/// - `plan`: Block of every factor of every term.
/// - `blocks`: Dense factor blocks of every block in the plan.
/// # Returns:
/// - `DenseBlock`: Table value at every free-index tuple.
pub(super) fn evaluate_dense_table(
    table: TermTable<'_>,
    free: &[u16],
    plan: &TablePlan,
    blocks: &FactorBlocks,
) -> DenseBlock {
    let (terms, indices) = table;

    // Extent of every label and the data of every table-local block.
    let extent = indices
        .iter()
        .map(|&(_, s)| blocks.dims[s as usize])
        .collect::<Vec<_>>();
    let data = plan
        .keys
        .iter()
        .map(|k| blocks.blocks[k].as_slice())
        .collect::<Vec<_>>();
    let dims = free.iter().map(|&x| extent[x as usize]).collect::<Vec<_>>();
    let size = dims.iter().product::<usize>();

    let planned = |t: usize, index: u32| {
        let range = |starts: &[u32]| starts[t] as usize..starts[t + 1] as usize;
        PlannedTerm {
            factors: &terms[index as usize].3,
            blocks: &plan.factor_blocks[range(&plan.term_starts)],
            steps: &plan.steps[range(&plan.step_starts)],
            map: label_map(&plan.substitutions[range(&plan.substitution_starts)]),
            coefficient: plan.coefficients[t],
        }
    };

    // A large block is split over its leading free indices into cache-sized slices, each
    // accumulated by one worker evaluating every term with those indices fixed.
    if size >= LARGEBLOCK {
        let mut lead = 0;
        let mut slice = size;
        while lead < free.len() && (slice > SLICESIZE || size / slice < SLICES) {
            slice /= dims[lead];
            lead += 1;
        }
        let mut out = vec![0.0; size];
        out.par_chunks_mut(slice)
            .enumerate()
            .for_each_init(Workspace::new, |ws, (s, chunk)| {
                let mut fixed = [(0u16, 0usize); MAXLABELS];
                let mut r = s;
                for k in (0..lead).rev() {
                    fixed[k] = (free[k], r % dims[k]);
                    r /= dims[k];
                }
                for (t, &index) in plan.terms.iter().enumerate() {
                    let term = planned(t, index);
                    accumulate_term(
                        &term,
                        (&data, &extent),
                        (&free[lead..], &fixed[..lead]),
                        None,
                        chunk,
                        ws,
                    );
                }
            });
        return DenseBlock { data: out, dims };
    }

    // A small block is accumulated once per split of the groups and the splits summed; each
    // group contracts its shared product once for all its terms.
    let out = plan
        .groups
        .par_iter()
        .fold(
            || (Workspace::new(), vec![0.0; size], Vec::new()),
            |(mut ws, mut out, buffer), (range, product)| {
                let shared = product.map(|id| {
                    let sp = &plan.shared[id as usize];
                    let a = strided_tensor_shape(&sp.labels[0], &sp.extent);
                    let b = strided_tensor_shape(&sp.labels[1], &sp.extent);
                    contract_tensor_pair(
                        (data[sp.blocks[0] as usize], &a),
                        (data[sp.blocks[1] as usize], &b),
                        sp.keep,
                        buffer,
                    )
                });
                for &t in &plan.order[range.clone()] {
                    let t = t as usize;
                    let term = planned(t, plan.terms[t]);
                    let use_shared = shared
                        .as_ref()
                        .zip(plan.anchors[t].as_ref())
                        .map(|((values, shape), anchor)| (values.as_slice(), shape, anchor));
                    accumulate_term(
                        &term,
                        (&data, &extent),
                        (free, &[]),
                        use_shared,
                        &mut out,
                        &mut ws,
                    );
                }
                (ws, out, shared.map_or_else(Vec::new, |(values, _)| values))
            },
        )
        .map(|(_, out, _)| out)
        .reduce(
            || vec![0.0; size],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );

    DenseBlock { data: out, dims }
}
