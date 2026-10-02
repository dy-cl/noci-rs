// nocc/terms/evaluate.rs
//! Evaluation of one planned table over whole orbital-space blocks.

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::maths::contract::MAXLABELS;

// Parent/sibling imports.
use super::factors::FactorBlocks;
use super::graph::evaluate_graph;
use super::plan::{TablePlan, TermTable, label_map};
use super::term::{PlannedTerm, accumulate_term};
use super::workspace::{Out, Values, Workspace};

/// Smallest output block, in elements, evaluated as slices over its leading free indices.
const LARGEBLOCK: usize = 1 << 20;

/// Largest slice, in elements, of a large output block accumulated by one worker.
const SLICESIZE: usize = 1 << 16;

/// Smallest number of slices a large output block is split into.
const SLICES: usize = 64;

/// Dense result block over the free indices of one table.
pub(super) struct DenseBlock {
    /// Row-major elements over the free indices in table order, complex when any factor is.
    pub(super) data: Values,
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
        .map(|k| blocks.blocks[k].view(0))
        .collect::<Vec<_>>();
    let complex = plan
        .keys
        .iter()
        .any(|k| matches!(blocks.blocks[k], Values::Complex(_)));
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
        let slice_terms = |ws: &mut Workspace, s: usize, mut chunk: Out<'_>| {
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
                    &mut chunk,
                    ws,
                );
            }
        };
        let mut out = Values::zeros(size, complex);
        match &mut out {
            Values::Real(x) => x
                .par_chunks_mut(slice)
                .enumerate()
                .for_each_init(Workspace::new, |ws, (s, chunk)| {
                    slice_terms(ws, s, Out::Real(chunk))
                }),
            Values::Complex(x) => x
                .par_chunks_mut(slice)
                .enumerate()
                .for_each_init(Workspace::new, |ws, (s, chunk)| {
                    slice_terms(ws, s, Out::Complex(chunk))
                }),
        }
        return DenseBlock { data: out, dims };
    }

    // A small block is evaluated over the table's shared contraction graph, with the terms
    // outside the graph contracted one by one.
    let mut out = evaluate_graph(&plan.graph, &data, &extent, free, (size, complex));
    let direct = plan
        .graph
        .direct
        .par_iter()
        .fold(
            || (Workspace::new(), Values::zeros(size, complex)),
            |(mut ws, mut out), &t| {
                let term = planned(t as usize, plan.terms[t as usize]);
                accumulate_term(
                    &term,
                    (&data, &extent),
                    (free, &[]),
                    &mut out.out(),
                    &mut ws,
                );
                (ws, out)
            },
        )
        .map(|(_, out)| out)
        .reduce(
            || Values::zeros(size, complex),
            |mut a, b| {
                a.add_assign(b);
                a
            },
        );
    out.add_assign(direct);

    DenseBlock { data: out, dims }
}
