// nocc/contract.rs
//! Dense tensor-contraction evaluation of generated term tables.
//!
//! A generated table gives one element of a residual, energy or coupling block as a sum of
//! terms `c\prod_k F_k` over dummy indices. Evaluating each element separately repeats every
//! dummy loop for every element. Here each term is instead contracted over whole orbital-space
//! blocks, pairwise in the order that keeps each intermediate cheapest, so one pass yields the
//! term's contribution to every element of the block at once.
//!
//! Tables hold millions of small terms, so per-term overhead is kept off the heap: the dense
//! block of every factor is resolved once per table and cached, operand shapes are fixed-size,
//! label sets are bit masks, and intermediate buffers are reused within each worker.

// Standard library imports.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::maths::contract::{
    MAXLABELS, TensorShape, cheapest_contraction_pair, contract_tensor_pair, strided_tensor_shape,
};
use crate::nocc::common::{Tensors, evaluate_factor, space_orbitals};
use crate::nocc::terms::{GeneratedTerm, TensorFactor};

/// Kronecker delta kind id used for generated delta factors.
const DELTA: u8 = 7;

/// Orbital-space block of one tensor kind: the kind id and the space of every slot, upper then
/// lower.
type BlockKey = (u8, Vec<u8>);

/// One term table: its terms and the name and space of every class-local index.
pub(crate) type TermTable<'a> = (&'a [GeneratedTerm], &'a [(String, u8)]);

/// Dense block of every factor of every kept term in one table, resolved once.
pub(crate) struct TablePlan {
    /// Distinct kind and slot-space patterns used by the kept terms.
    keys: Vec<BlockKey>,
    /// Index of every kept term in the table.
    terms: Vec<u32>,
    /// Block of every delta then tensor factor, for all kept terms in order.
    factor_blocks: Vec<u32>,
    /// Start of every kept term's entries in `factor_blocks`, with a final end marker.
    term_starts: Vec<u32>,
}

/// Evaluator of generated term tables: the cumulant truncation and the plans of every table
/// evaluated in one run, keyed by table address.
pub(crate) struct TermEvaluator {
    /// Highest cumulant rank kept; terms with higher-rank cumulants are dropped.
    max_cumulant: usize,
    /// Cached plans.
    plans: Mutex<HashMap<usize, Arc<TablePlan>>>,
}

/// Dense tensor blocks of every tensor kind and slot-space pattern used by a set of tables.
pub(crate) struct FactorBlocks {
    /// Row-major block data over the orbitals of each slot space.
    blocks: HashMap<BlockKey, Vec<f64>>,
    /// Number of orbitals in the core, active and virtual spaces.
    dims: [usize; 3],
}

/// Dense result block over the free indices of one table.
pub(crate) struct DenseBlock {
    /// Row-major elements over the free indices in table order.
    pub(crate) data: Vec<f64>,
    /// Extent of every free index.
    pub(crate) dims: Vec<usize>,
}

/// Location of an operand's data.
#[derive(Clone, Copy)]
enum Source {
    /// A dense factor block, by table-local block id.
    Block(usize),
    /// An intermediate buffer of the current term.
    Buffer(usize),
}

/// Reusable per-worker storage for contracting terms.
struct Workspace {
    /// Operands of the current term.
    operands: Vec<(Source, TensorShape)>,
    /// Intermediate buffers of the current term.
    buffers: Vec<Vec<f64>>,
    /// Released buffers available for reuse.
    pool: Vec<Vec<f64>>,
    /// Accumulated output block.
    out: Vec<f64>,
}

impl TermEvaluator {
    /// Build an evaluator with no resolved plans.
    /// # Arguments:
    /// - `max_cumulant`: Highest cumulant rank kept in every table.
    /// # Returns:
    /// - `Self`: Evaluator.
    pub(crate) fn new(max_cumulant: usize) -> Self {
        Self {
            max_cumulant,
            plans: Mutex::new(HashMap::new()),
        }
    }

    /// Return the plan of one working-equation table, the energy or a residual, resolving it on
    /// first use. Terms with cumulants above the truncation rank are dropped.
    /// # Arguments:
    /// - `table`: Terms and index spaces of the table.
    /// # Returns:
    /// - `Arc<TablePlan>`: Block of every factor of every kept term.
    pub(crate) fn table_plan(
        &self,
        table: TermTable<'_>,
    ) -> Arc<TablePlan> {
        self.cached_plan(table, self.max_cumulant)
    }

    /// Return the plan of one reference-property table, such as the metric or the zeroth-order
    /// coupling, keeping every term. These tables involve at most the four-body RDM, which the
    /// reference provides exactly, so they are never truncated.
    /// # Arguments:
    /// - `table`: Terms and index spaces of the table.
    /// # Returns:
    /// - `Arc<TablePlan>`: Block of every factor of every term.
    pub(crate) fn exact_table_plan(
        &self,
        table: TermTable<'_>,
    ) -> Arc<TablePlan> {
        self.cached_plan(table, usize::MAX)
    }

    /// Return the cached plan of one table at one truncation rank, resolving it on first use.
    /// Each table is always evaluated at the same rank, so plans are keyed by table address.
    /// # Arguments:
    /// - `table`: Terms and index spaces of the table.
    /// - `max_cumulant`: Highest cumulant rank kept.
    /// # Returns:
    /// - `Arc<TablePlan>`: Block of every factor of every kept term.
    fn cached_plan(
        &self,
        table: TermTable<'_>,
        max_cumulant: usize,
    ) -> Arc<TablePlan> {
        let address = table.0.as_ptr() as usize;
        if let Some(plan) = self.plans.lock().unwrap().get(&address) {
            return plan.clone();
        }

        let plan = Arc::new(resolve_table_plan(table, max_cumulant));
        self.plans.lock().unwrap().insert(address, plan.clone());
        plan
    }
}

/// Resolve the dense block of every factor of every kept term in one table.
/// A term is kept when every cumulant `\Lambda_k` it contains has `k \le k_{\max}`, the
/// GNOCCSD(`k_{\max}`) truncation of Lee and Tew.
/// # Arguments:
/// - `table`: Terms and index spaces of the table.
/// - `max_cumulant`: Highest cumulant rank kept.
/// # Returns:
/// - `TablePlan`: Kept terms, distinct blocks and the block of every factor.
fn resolve_table_plan(
    table: TermTable<'_>,
    max_cumulant: usize,
) -> TablePlan {
    let (terms, indices) = table;
    let space = |x: &u16| indices[*x as usize].1;
    let mut ids = HashMap::<BlockKey, u32>::new();
    let mut keys = Vec::new();
    let mut factor_blocks = Vec::new();
    let mut term_starts = Vec::with_capacity(terms.len() + 1);
    let mut kept = Vec::with_capacity(terms.len());

    let mut intern = |key: BlockKey, keys: &mut Vec<BlockKey>| {
        *ids.entry(key.clone()).or_insert_with(|| {
            keys.push(key);
            (keys.len() - 1) as u32
        })
    };

    // Cumulants `\Lambda_2`, `\Lambda_3` and `\Lambda_4` have kind ids `4`, `5` and `6`.
    let rank = |kind: u8| match kind {
        4..=6 => kind as usize - 2,
        _ => 0,
    };

    for (t, term) in terms.iter().enumerate() {
        if term.3.iter().any(|f| rank(f.0) > max_cumulant) {
            continue;
        }
        kept.push(t as u32);
        term_starts.push(factor_blocks.len() as u32);
        for d in &term.2 {
            factor_blocks.push(intern((DELTA, d.iter().map(space).collect()), &mut keys));
        }
        for f in &term.3 {
            let slots = f.1.iter().chain(&f.2).map(space).collect();
            factor_blocks.push(intern((f.0, slots), &mut keys));
        }
    }
    term_starts.push(factor_blocks.len() as u32);

    TablePlan {
        keys,
        terms: kept,
        factor_blocks,
        term_starts,
    }
}

impl FactorBlocks {
    /// Build every dense factor block used by the given table plans.
    /// Block elements are the runtime tensor elements over the orbitals of each slot space, so
    /// every tensor convention is shared with the element-wise evaluator.
    /// # Arguments:
    /// - `plans`: Plans of the tables to evaluate.
    /// - `tensors`: Runtime tensors, including the current amplitudes when needed.
    /// # Returns:
    /// - `Self`: Dense blocks keyed by tensor kind and slot spaces.
    pub(crate) fn build_factor_blocks(
        plans: &[&TablePlan],
        tensors: &Tensors<'_>,
    ) -> Self {
        let dims = [0u8, 1, 2].map(|s| space_orbitals(tensors.spaces, s).len());

        let mut keys = plans
            .iter()
            .flat_map(|p| p.keys.iter().cloned())
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();

        let blocks = keys
            .into_par_iter()
            .map(|key| {
                let data = dense_factor_block(&key, tensors);
                (key, data)
            })
            .collect();

        Self { blocks, dims }
    }
}

/// Build one dense factor block by evaluating the runtime tensor element at every orbital tuple.
/// # Arguments:
/// - `key`: Tensor kind and slot spaces.
/// - `tensors`: Runtime tensors.
/// # Returns:
/// - `Vec<f64>`: Row-major block elements.
fn dense_factor_block(
    key: &BlockKey,
    tensors: &Tensors<'_>,
) -> Vec<f64> {
    let (kind, spaces) = key;
    let orbitals = spaces
        .iter()
        .map(|&s| space_orbitals(tensors.spaces, s))
        .collect::<Vec<_>>();
    let dims = orbitals.iter().map(|o| o.len()).collect::<Vec<_>>();
    let size = dims.iter().product::<usize>();

    // Slot ids `0..k` index the orbital tuple; the first half are upper slots.
    let k = spaces.len();
    let factor = TensorFactor(
        *kind,
        (0..k as u16 / 2).collect(),
        (k as u16 / 2..k as u16).collect(),
    );

    // A slot over an empty orbital space gives an empty block.
    if size == 0 {
        return Vec::new();
    }

    // Odometer over the orbital tuples in row-major order.
    let mut idx = orbitals.iter().map(|o| o[0]).collect::<Vec<_>>();
    let mut pos = vec![0usize; k];
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        data.push(if *kind == DELTA {
            if idx[0] == idx[1] { 1.0 } else { 0.0 }
        } else {
            evaluate_factor(&factor, &idx, tensors)
        });

        let mut slot = k;
        while slot > 0 {
            slot -= 1;
            pos[slot] += 1;
            if pos[slot] < dims[slot] {
                idx[slot] = orbitals[slot][pos[slot]];
                break;
            }
            pos[slot] = 0;
            idx[slot] = orbitals[slot][0];
        }
    }

    data
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
pub(crate) fn evaluate_dense_table(
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

    let workspace = || Workspace {
        operands: Vec::new(),
        buffers: Vec::new(),
        pool: Vec::new(),
        out: vec![0.0; size],
    };
    let out = plan
        .terms
        .par_iter()
        .enumerate()
        .fold(workspace, |mut ws, (t, &index)| {
            let term = &terms[index as usize];
            let (start, end) = (
                plan.term_starts[t] as usize,
                plan.term_starts[t + 1] as usize,
            );
            accumulate_term(
                term,
                &plan.factor_blocks[start..end],
                &data,
                &extent,
                free,
                &mut ws,
            );
            ws
        })
        .map(|ws| ws.out)
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

/// Contract one term and add its contribution to the workspace output block.
/// # Arguments:
/// - `term`: Generated term.
/// - `factor_blocks`: Table-local block id of every delta then tensor factor.
/// - `data`: Data of every table-local block.
/// - `extent`: Extent of every class-local label.
/// - `free`: Class-local ids of the free indices, in output order.
/// - `ws`: Worker storage, holding the output block.
/// # Returns:
/// - `()`: Mutates `ws`.
fn accumulate_term(
    term: &GeneratedTerm,
    factor_blocks: &[u32],
    data: &[&[f64]],
    extent: &[usize],
    free: &[u16],
    ws: &mut Workspace,
) {
    let free_mask = free.iter().fold(0u64, |m, &x| m | (1 << x));

    // Every delta and tensor factor becomes an operand over its block.
    ws.operands.clear();
    for (k, d) in term.2.iter().enumerate() {
        let shape = strided_tensor_shape(d, extent);
        ws.operands
            .push((Source::Block(factor_blocks[k] as usize), shape));
    }
    for (k, f) in term.3.iter().enumerate() {
        let mut labels = [0u16; 2 * MAXLABELS];
        let n = f.1.len() + f.2.len();
        labels[..f.1.len()].copy_from_slice(&f.1);
        labels[f.1.len()..n].copy_from_slice(&f.2);
        let shape = strided_tensor_shape(&labels[..n], extent);
        let id = factor_blocks[term.2.len() + k] as usize;
        ws.operands.push((Source::Block(id), shape));
    }

    // Contract the cheapest pair until one operand remains.
    while ws.operands.len() > 1 {
        let (i, j) = cheapest_contraction_pair(&ws.operands, extent);
        let (sb, b) = ws.operands.swap_remove(j.max(i));
        let (sa, a) = ws.operands.swap_remove(j.min(i));
        let keep = ws.operands.iter().fold(free_mask, |m, (_, s)| m | s.mask);

        let buffer = ws.pool.pop().unwrap_or_default();
        let (result, shape) = {
            let source = |s: Source| match s {
                Source::Block(id) => data[id],
                Source::Buffer(id) => ws.buffers[id].as_slice(),
            };
            contract_tensor_pair((source(sa), &a), (source(sb), &b), keep, buffer)
        };
        ws.buffers.push(result);
        ws.operands
            .push((Source::Buffer(ws.buffers.len() - 1), shape));
    }

    // Add the final operand, times the coefficient, to the output block.
    let coeff = term.0[0] as f64 / term.0[1] as f64;
    let last = ws.operands.pop().map(|(s, shape)| {
        let values = match s {
            Source::Block(id) => data[id],
            Source::Buffer(id) => ws.buffers[id].as_slice(),
        };
        (values, shape)
    });
    scatter_into_output(last, free, extent, coeff, &mut ws.out);

    // Release this term's buffers for reuse.
    let released = std::mem::take(&mut ws.buffers);
    ws.pool.extend(released);
}

/// Add `c` times the final operand to the output block over the free indices.
/// Free indices absent from the operand are broadcast, labels not free are summed, and a term
/// with no factors is a constant.
/// # Arguments:
/// - `operand`: Final operand data and shape, or `None` for a term without factors.
/// - `free`: Class-local ids of the free indices, in output order.
/// - `extent`: Extent of every class-local label.
/// - `coeff`: Term coefficient.
/// - `out`: Row-major output block, updated in place.
/// # Returns:
/// - `()`: Mutates `out`.
fn scatter_into_output(
    operand: Option<(&[f64], TensorShape)>,
    free: &[u16],
    extent: &[usize],
    coeff: f64,
    out: &mut [f64],
) {
    let Some((data, shape)) = operand else {
        for value in out.iter_mut() {
            *value += coeff;
        }
        return;
    };

    // Stride of every free index in the operand, zero when broadcast.
    let dims = free.iter().map(|&x| extent[x as usize]).collect::<Vec<_>>();
    let strides = free
        .iter()
        .map(|&l| {
            shape.labels[..shape.n]
                .iter()
                .position(|&x| x == l)
                .map_or(0, |k| shape.strides[k])
        })
        .collect::<Vec<_>>();

    // Remaining labels are summed.
    let free_mask = free.iter().fold(0u64, |m, &x| m | (1 << x));
    let summed = (0..shape.n)
        .filter(|&k| free_mask & (1 << shape.labels[k]) == 0)
        .map(|k| (shape.dims[k], shape.strides[k]))
        .collect::<Vec<_>>();
    let rest = summed.iter().map(|&(d, _)| d).product::<usize>();

    for (flat, value) in out.iter_mut().enumerate() {
        let mut offset = 0;
        let mut r = flat;
        for k in (0..dims.len()).rev() {
            offset += (r % dims[k]) * strides[k];
            r /= dims[k];
        }
        let mut total = 0.0;
        for inner in 0..rest {
            let mut extra = 0;
            let mut r = inner;
            for &(d, s) in summed.iter().rev() {
                extra += (r % d) * s;
                r /= d;
            }
            total += data[offset + extra];
        }
        *value += coeff * total;
    }
}
