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
use std::ops::Range;
use std::sync::{Arc, Mutex};

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::maths::contract::{
    LabelSizes, MAXLABELS, TensorShape, contract_tensor_pair, optimal_contraction_steps,
    strided_tensor_shape,
};
use crate::nocc::common::{Tensors, evaluate_factor, space_orbitals};
use crate::nocc::space::Spaces;
use crate::nocc::terms::{GeneratedTerm, TensorFactor};

/// Largest number of terms evaluated together from one computed shared product.
const GROUPTERMS: usize = 1024;

/// Smallest output block, in elements, evaluated as slices over its leading free indices.
const LARGEBLOCK: usize = 1 << 20;

/// Largest slice, in elements, of a large output block accumulated by one worker.
const SLICESIZE: usize = 1 << 16;

/// Smallest number of slices a large output block is split into.
const SLICES: usize = 64;

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
    /// Block of every tensor factor, for all kept terms in order.
    factor_blocks: Vec<u32>,
    /// Start of every kept term's entries in `factor_blocks`, with a final end marker.
    term_starts: Vec<u32>,
    /// Pairwise contraction steps of every kept term in its optimal order, as operand numbers:
    /// factors are numbered delta then tensor and each step's result takes the next number.
    steps: Vec<(u8, u8)>,
    /// Start of every kept term's entries in `steps`, with a final end marker.
    step_starts: Vec<u32>,
    /// Label substitutions `(l, r)` of every kept term that resolve its Kronecker deltas.
    substitutions: Vec<(u16, u16)>,
    /// Start of every kept term's entries in `substitutions`, with a final end marker.
    substitution_starts: Vec<u32>,
    /// Coefficient of every kept term, including the extent of every summed label left only in
    /// resolved deltas.
    coefficients: Vec<f64>,
    /// Products of two factor blocks shared by several kept terms, in canonical labels.
    shared: Vec<SharedProduct>,
    /// Shared product of every kept term, its step number, and the term label of every
    /// canonical label, or `None`.
    anchors: Vec<Option<Anchor>>,
    /// Kept-term positions grouped by shared product, each group with its product.
    groups: Vec<(Range<usize>, Option<u32>)>,
    /// Kept-term positions in group order.
    order: Vec<u32>,
}

/// Product of two factor blocks over canonical labels, `C = A B` summed over the labels not
/// kept, computed once for every kept term that contains it.
struct SharedProduct {
    /// Table-local block of each operand.
    blocks: [u32; 2],
    /// Canonical label of every slot of each operand.
    labels: [Vec<u16>; 2],
    /// Bit mask of the canonical labels kept in the product.
    keep: u64,
    /// Extent of every canonical label.
    extent: Vec<usize>,
}

/// Use of a shared product by one kept term.
#[derive(Clone)]
struct Anchor {
    /// Shared product id.
    product: u32,
    /// Step of the term the product replaces.
    step: u8,
    /// Term label of every canonical label.
    labels: Vec<u16>,
}

/// Evaluator of generated term tables: the cumulant truncation, the orbital-space sizes that
/// fix every contraction order, and the plans of every table evaluated in one run, keyed by
/// table address.
pub(crate) struct TermEvaluator {
    /// Highest cumulant rank kept; terms with higher-rank cumulants are dropped.
    max_cumulant: usize,
    /// Number of orbitals in the core, active and virtual spaces.
    dims: [usize; 3],
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
    /// A dense factor block, by table-local block id, viewed from an element offset.
    Block(usize, usize),
    /// An intermediate buffer of the current term.
    Buffer(usize),
    /// The product shared by the current term's group.
    Shared,
}

/// Reusable per-worker storage for contracting terms.
struct Workspace {
    /// Operands of the current term.
    operands: Vec<(Source, TensorShape)>,
    /// Intermediate buffers of the current term.
    buffers: Vec<Vec<f64>>,
    /// Released buffers available for reuse.
    pool: Vec<Vec<f64>>,
}

impl Workspace {
    /// Build empty worker storage.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Self`: Empty workspace.
    fn new() -> Self {
        Self {
            operands: Vec::new(),
            buffers: Vec::new(),
            pool: Vec::new(),
        }
    }
}

impl TermEvaluator {
    /// Build an evaluator with no resolved plans.
    /// # Arguments:
    /// - `max_cumulant`: Highest cumulant rank kept in every table.
    /// - `spaces`: NOCC orbital spaces.
    /// # Returns:
    /// - `Self`: Evaluator.
    pub(crate) fn new(
        max_cumulant: usize,
        spaces: &Spaces,
    ) -> Self {
        Self {
            max_cumulant,
            dims: [0u8, 1, 2].map(|s| space_orbitals(spaces, s).len()),
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

        let plan = Arc::new(resolve_table_plan(table, max_cumulant, self.dims));
        self.plans.lock().unwrap().insert(address, plan.clone());
        plan
    }
}

/// Resolve the dense block of every factor and the contraction order of every kept term in one
/// table. A term is kept when every cumulant `\Lambda_k` it contains has `k \le k_{\max}`, the
/// GNOCCSD(`k_{\max}`) truncation of Lee and Tew. Each term's order minimises its multiply-adds
/// at the orbital-space sizes of the run.
/// # Arguments:
/// - `table`: Terms and index spaces of the table.
/// - `max_cumulant`: Highest cumulant rank kept.
/// - `dims`: Number of orbitals in the core, active and virtual spaces.
/// # Returns:
/// - `TablePlan`: Kept terms, distinct blocks, the block of every factor and the contraction
///   steps of every term.
fn resolve_table_plan(
    table: TermTable<'_>,
    max_cumulant: usize,
    dims: [usize; 3],
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

    let extent = indices
        .iter()
        .map(|&(_, s)| dims[s as usize])
        .collect::<Vec<_>>();
    let mut substitutions = Vec::new();
    let mut substitution_starts = Vec::with_capacity(terms.len() + 1);
    let mut coefficients = Vec::with_capacity(terms.len());

    for (t, term) in terms.iter().enumerate() {
        if term.3.iter().any(|f| rank(f.0) > max_cumulant) {
            continue;
        }
        // A delta between different orbital spaces vanishes.
        let Some((map, scale)) = resolve_deltas(term, indices, &extent) else {
            continue;
        };
        kept.push(t as u32);
        coefficients.push(scale * term.0[0] as f64 / term.0[1] as f64);
        substitution_starts.push(substitutions.len() as u32);
        substitutions.extend(
            (0..indices.len() as u16)
                .filter(|&l| map[l as usize] != l)
                .map(|l| (l, map[l as usize])),
        );
        term_starts.push(factor_blocks.len() as u32);
        for f in &term.3 {
            let slots = f.1.iter().chain(&f.2).map(space).collect();
            factor_blocks.push(intern((f.0, slots), &mut keys));
        }
    }
    term_starts.push(factor_blocks.len() as u32);
    substitution_starts.push(substitutions.len() as u32);

    // Optimal contraction steps of every kept term over its substituted labels; the
    // representatives of the free labels are kept.
    let sizes = LabelSizes::new(&extent);
    let term_steps = kept
        .par_iter()
        .enumerate()
        .map(|(k, &t)| {
            let term = &terms[t as usize];
            let map = label_map(
                &substitutions
                    [substitution_starts[k] as usize..substitution_starts[k + 1] as usize],
            );
            let masks = term
                .3
                .iter()
                .map(|f| {
                    f.1.iter()
                        .chain(&f.2)
                        .fold(0u64, |m, &x| m | (1 << map[x as usize]))
                })
                .collect::<Vec<_>>();
            let summed = term.1.iter().fold(0u64, |m, &x| m | (1 << x));
            let all = masks.iter().fold(0u64, |m, &x| m | x);
            let mut steps = Vec::with_capacity(masks.len());
            optimal_contraction_steps(&masks, all & !summed, &sizes, &mut steps);
            let blocks = &factor_blocks[term_starts[k] as usize..term_starts[k + 1] as usize];
            let anchor = shared_step(term, &map, blocks, &masks, &steps, all & !summed, &sizes);
            (steps, anchor)
        })
        .collect::<Vec<_>>();
    let mut steps = Vec::new();
    let mut step_starts = Vec::with_capacity(kept.len() + 1);
    let mut shared = Vec::new();
    let mut shared_ids = HashMap::<SharedKey, u32>::new();
    let mut anchors = Vec::with_capacity(kept.len());
    for (s, anchor) in term_steps {
        step_starts.push(steps.len() as u32);
        steps.extend(s);
        anchors.push(anchor.map(|(key, step, labels)| {
            let next = shared.len() as u32;
            let product = *shared_ids.entry(key.clone()).or_insert_with(|| {
                let extent = labels.iter().map(|&l| extent[l as usize]).collect();
                shared.push(SharedProduct {
                    blocks: [key.0, key.2],
                    labels: [key.1.clone(), key.3.clone()],
                    keep: key.4,
                    extent,
                });
                next
            });
            Anchor {
                product,
                step,
                labels,
            }
        }));
    }
    step_starts.push(steps.len() as u32);

    // Group the kept terms by shared product, in pieces small enough to balance the workers;
    // products used by one term are not shared.
    let mut uses = vec![0usize; shared.len()];
    for a in anchors.iter().flatten() {
        uses[a.product as usize] += 1;
    }
    for a in anchors.iter_mut() {
        if a.as_ref().is_some_and(|x| uses[x.product as usize] < 2) {
            *a = None;
        }
    }
    let mut order = (0..kept.len() as u32).collect::<Vec<_>>();
    order.sort_by_key(|&t| anchors[t as usize].as_ref().map_or(u32::MAX, |a| a.product));
    let mut groups = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let product = anchors[order[start] as usize].as_ref().map(|a| a.product);
        let mut end = start + 1;
        if product.is_some() {
            while end < order.len()
                && anchors[order[end] as usize].as_ref().map(|a| a.product) == product
            {
                end += 1;
            }
        }
        for piece in (start..end).step_by(GROUPTERMS) {
            groups.push((piece..(piece + GROUPTERMS).min(end), product));
        }
        start = end;
    }

    TablePlan {
        keys,
        terms: kept,
        factor_blocks,
        term_starts,
        steps,
        step_starts,
        substitutions,
        substitution_starts,
        coefficients,
        shared,
        anchors,
        groups,
        order,
    }
}

/// Canonical form of a product of two factor blocks: each operand's block and canonical slot
/// labels, and the bit mask of kept canonical labels.
type SharedKey = (u32, Vec<u16>, u32, Vec<u16>, u64);

/// Find a term's most expensive step that contracts two factor blocks, in canonical form.
/// Labels are renumbered in order of first appearance over the two operands' slots, in the
/// operand order giving the smaller key, so terms that contract the same blocks in the same way
/// share one key. The kept labels are those the evaluator keeps at that step.
/// # Arguments:
/// - `term`: Generated term.
/// - `map`: Representative of every label.
/// - `blocks`: Table-local block of every tensor factor.
/// - `masks`: Label mask of every tensor factor.
/// - `steps`: Planned contraction steps.
/// - `kept`: Bit mask of labels kept in the final result.
/// - `sizes`: Joint index-space sizes of label masks.
/// # Returns:
/// - `Option<(SharedKey, u8, Vec<u16>)>`: Canonical key, step number, and term label of every
///   canonical label, or `None` when no step contracts two factor blocks.
fn shared_step(
    term: &GeneratedTerm,
    map: &[u16; 64],
    blocks: &[u32],
    masks: &[u64],
    steps: &[(u8, u8)],
    kept: u64,
    sizes: &LabelSizes,
) -> Option<(SharedKey, u8, Vec<u16>)> {
    // Replay the steps with the evaluator's kept labels, recording block-by-block steps.
    let leaves = masks.len();
    let mut ops = masks.to_vec();
    let mut live = (1u64 << ops.len()) - 1;
    let mut best: Option<(f64, u8, usize, usize, u64)> = None;
    for (s, &(i, j)) in steps.iter().enumerate() {
        live &= !((1 << i) | (1 << j));
        let rest = (0..ops.len())
            .filter(|&k| live & (1 << k) != 0)
            .fold(kept, |m, k| m | ops[k]);
        let joint = ops[i as usize] | ops[j as usize];
        let result = joint & rest;
        if (i as usize) < leaves && (j as usize) < leaves {
            let cost = sizes.size(joint);
            if best.is_none_or(|b| cost > b.0) {
                best = Some((cost, s as u8, i as usize, j as usize, result));
            }
        }
        live |= 1 << ops.len();
        ops.push(result);
    }
    let (_, step, i, j) = best.map(|b| (b.0, b.1, b.2, b.3))?;
    let keep = best?.4;

    // Canonical labels in order of first appearance over the two operands' slots.
    let slots = |f: usize| {
        let x = &term.3[f];
        x.1.iter()
            .chain(&x.2)
            .map(|&l| map[l as usize])
            .collect::<Vec<_>>()
    };
    let canonical = |a: usize, b: usize| {
        let mut names = Vec::<u16>::new();
        let mut rename = |l: u16| {
            let k = names.iter().position(|&x| x == l).unwrap_or_else(|| {
                names.push(l);
                names.len() - 1
            });
            k as u16
        };
        let la = slots(a).into_iter().map(&mut rename).collect::<Vec<_>>();
        let lb = slots(b).into_iter().map(&mut rename).collect::<Vec<_>>();
        let keep_c = names
            .iter()
            .enumerate()
            .filter(|&(_, &l)| keep & (1 << l) != 0)
            .fold(0u64, |m, (k, _)| m | (1 << k));
        ((blocks[a], la, blocks[b], lb, keep_c), names)
    };
    let (x, y) = (canonical(i, j), canonical(j, i));
    let (key, names) = if x.0 <= y.0 { x } else { y };
    Some((key, step, names))
}

/// Resolve the Kronecker deltas of one term into label substitutions. Labels joined by deltas
/// are replaced by one representative, a free label when the class holds one, so a delta on a
/// summed label removes that sum and a delta between free labels restricts the term to their
/// diagonal. A class of summed labels that no tensor factor uses sums to its extent.
/// # Arguments:
/// - `term`: Generated term.
/// - `indices`: Name and space of every class-local label.
/// - `extent`: Extent of every class-local label.
/// # Returns:
/// - `Option<([u16; 64], f64)>`: Representative of every label and the coefficient scale, or
///   `None` when a delta joins labels of different orbital spaces and the term vanishes.
fn resolve_deltas(
    term: &GeneratedTerm,
    indices: &[(String, u8)],
    extent: &[usize],
) -> Option<([u16; 64], f64)> {
    let mut map = [0u16; 64];
    for (l, x) in map.iter_mut().enumerate() {
        *x = l as u16;
    }
    let summed = term.1.iter().fold(0u64, |m, &x| m | (1 << x));
    let root = |map: &[u16; 64], mut l: u16| {
        while map[l as usize] != l {
            l = map[l as usize];
        }
        l
    };

    // Join every delta pair, preferring a free representative and then the lower label.
    for d in &term.2 {
        if indices[d[0] as usize].1 != indices[d[1] as usize].1 {
            return None;
        }
        let (a, b) = (root(&map, d[0]), root(&map, d[1]));
        if a == b {
            continue;
        }
        let free = |l: u16| summed & (1 << l) == 0;
        let (keep, drop) = match (free(a), free(b)) {
            (true, false) => (a, b),
            (false, true) => (b, a),
            _ => (a.min(b), a.max(b)),
        };
        map[drop as usize] = keep;
    }
    for l in 0..indices.len() as u16 {
        map[l as usize] = root(&map, l);
    }

    // Summed classes absent from every tensor factor contribute their extent.
    let used = term
        .3
        .iter()
        .flat_map(|f| f.1.iter().chain(&f.2))
        .fold(0u64, |m, &x| m | (1 << map[x as usize]));
    let mut counted = 0u64;
    let mut scale = 1.0;
    for d in &term.2 {
        let r = map[d[0] as usize];
        if summed & (1 << r) != 0 && used & (1 << r) == 0 && counted & (1 << r) == 0 {
            scale *= extent[r as usize] as f64;
            counted |= 1 << r;
        }
    }

    Some((map, scale))
}

/// Build the representative of every label from a term's substitutions.
/// # Arguments:
/// - `substitutions`: Label substitutions `(l, r)` of one term.
/// # Returns:
/// - `[u16; 64]`: Representative of every label.
fn label_map(substitutions: &[(u16, u16)]) -> [u16; 64] {
    let mut map = [0u16; 64];
    for (l, x) in map.iter_mut().enumerate() {
        *x = l as u16;
    }
    for &(l, r) in substitutions {
        map[l as usize] = r;
    }
    map
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
        data.push(evaluate_factor(&factor, &idx, tensors));

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

/// One kept term with its resolved plan.
struct PlannedTerm<'a> {
    /// Tensor factors of the term.
    factors: &'a [TensorFactor],
    /// Table-local block id of every tensor factor.
    blocks: &'a [u32],
    /// Pairwise contraction steps, as operand numbers.
    steps: &'a [(u8, u8)],
    /// Representative of every label after resolving the deltas.
    map: [u16; 64],
    /// Coefficient of the term.
    coefficient: f64,
}

/// Contract one term, with some free indices fixed, and add it to an output block over the
/// remaining free indices.
/// # Arguments:
/// - `term`: Kept term with its plan.
/// - `tensors`: Data of every table-local block and extent of every class-local label.
/// - `indices`: Class-local ids of the free indices of `out` in output order, and the free
///   indices fixed to one value with their values.
/// - `shared`: Data and canonical shape of the term's shared product with its use, which
///   replaces the anchored step, or `None`.
/// - `out`: Row-major output block over the unfixed free indices, updated in place.
/// - `ws`: Worker storage.
/// # Returns:
/// - `()`: Mutates `out` and `ws`.
fn accumulate_term(
    term: &PlannedTerm<'_>,
    tensors: (&[&[f64]], &[usize]),
    indices: (&[u16], &[(u16, usize)]),
    shared: Option<(&[f64], &TensorShape, &Anchor)>,
    out: &mut [f64],
    ws: &mut Workspace,
) {
    let (data, extent) = tensors;
    let (free, fixed) = indices;
    let map = &term.map;

    // Fixed indices by representative; a representative fixed to two values vanishes here.
    let mut reps = [(0u16, 0usize); MAXLABELS];
    let mut nf = 0;
    for &(l, v) in fixed {
        let r = map[l as usize];
        match reps[..nf].iter().find(|x| x.0 == r) {
            Some(&(_, w)) if w != v => return,
            Some(_) => {}
            None => {
                reps[nf] = (r, v);
                nf += 1;
            }
        }
    }
    let fixed = &reps[..nf];
    let fixed_mask = fixed.iter().fold(0u64, |m, x| m | (1 << x.0));
    let free_mask = free.iter().fold(0u64, |m, &x| m | (1 << map[x as usize])) & !fixed_mask;

    // Every tensor factor becomes an operand over its block, with its labels substituted and
    // its fixed labels folded into an offset; a label repeated within one factor becomes a
    // diagonal view.
    ws.operands.clear();
    for (f, &id) in term.factors.iter().zip(term.blocks) {
        let mut labels = [0u16; 2 * MAXLABELS];
        let n = f.1.len() + f.2.len();
        for (slot, &l) in labels.iter_mut().zip(f.1.iter().chain(&f.2)) {
            *slot = map[l as usize];
        }
        let mut shape = strided_tensor_shape(&labels[..n], extent);
        let offset = fixed
            .iter()
            .map(|&(r, v)| fix_label(&mut shape, r, v))
            .sum();
        ws.operands
            .push((Source::Block(id as usize, offset), shape));
    }

    // Contract in the planned order; each step keeps the labels of the operands still to be
    // contracted and the free labels.
    let mut live = (1u64 << ws.operands.len()) - 1;
    for (s, &(i, j)) in term.steps.iter().enumerate() {
        // The anchored step takes the group's shared product, relabelled to this term.
        if let Some((_, canonical, anchor)) = shared
            && s == anchor.step as usize
        {
            live &= !((1 << i) | (1 << j));
            let mut shape = *canonical;
            shape.mask = 0;
            for l in shape.labels[..shape.n].iter_mut() {
                *l = anchor.labels[*l as usize];
                shape.mask |= 1 << *l;
            }
            live |= 1 << ws.operands.len();
            ws.operands.push((Source::Shared, shape));
            continue;
        }
        let (sa, a) = ws.operands[i as usize];
        let (sb, b) = ws.operands[j as usize];
        live &= !((1 << i) | (1 << j));
        let keep = (0..ws.operands.len())
            .filter(|&k| live & (1 << k) != 0)
            .fold(free_mask, |m, k| m | ws.operands[k].1.mask);

        let buffer = ws.pool.pop().unwrap_or_default();
        let (result, shape) = {
            let source = |s: Source| match s {
                Source::Block(id, off) => &data[id][off..],
                Source::Buffer(id) => ws.buffers[id].as_slice(),
                Source::Shared => shared.map_or(&[][..], |x| x.0),
            };
            contract_tensor_pair((source(sa), &a), (source(sb), &b), keep, buffer)
        };
        ws.buffers.push(result);
        live |= 1 << ws.operands.len();
        ws.operands
            .push((Source::Buffer(ws.buffers.len() - 1), shape));
    }

    // Add the final operand, times the coefficient, to the output block.
    let last = ws.operands.pop().map(|(s, shape)| {
        let values = match s {
            Source::Block(id, off) => &data[id][off..],
            Source::Buffer(id) => ws.buffers[id].as_slice(),
            Source::Shared => shared.map_or(&[][..], |x| x.0),
        };
        (values, shape)
    });
    scatter_into_output(last, (free, fixed), map, extent, term.coefficient, out);

    // Release this term's buffers for reuse.
    let released = std::mem::take(&mut ws.buffers);
    ws.pool.extend(released);
}

/// Fix one label of an operand shape to one value, removing it from the shape.
/// # Arguments:
/// - `shape`: Operand shape, updated in place.
/// - `label`: Label to fix.
/// - `value`: Value of the label.
/// # Returns:
/// - `usize`: Element offset of the fixed slice, zero when the operand lacks the label.
fn fix_label(
    shape: &mut TensorShape,
    label: u16,
    value: usize,
) -> usize {
    let Some(k) = shape.labels[..shape.n].iter().position(|&x| x == label) else {
        return 0;
    };
    let offset = value * shape.strides[k];
    for j in k..shape.n - 1 {
        shape.labels[j] = shape.labels[j + 1];
        shape.dims[j] = shape.dims[j + 1];
        shape.strides[j] = shape.strides[j + 1];
    }
    shape.n -= 1;
    shape.mask &= !(1 << label);
    offset
}

/// Add `c` times the final operand to the output block over the free indices.
/// Free indices sharing a representative are written only on their diagonal, representatives
/// absent from the operand are broadcast, labels not free are summed, and a term with no
/// factors is a constant.
/// # Arguments:
/// - `operand`: Final operand data and shape, or `None` for a term without factors.
/// - `indices`: Class-local ids of the free indices of `out` in output order, and the fixed
///   representatives with their values; an output index whose representative is fixed is
///   written only at that value.
/// - `map`: Representative of every label.
/// - `extent`: Extent of every class-local label.
/// - `coeff`: Term coefficient.
/// - `out`: Row-major output block, updated in place.
/// # Returns:
/// - `()`: Mutates `out`.
fn scatter_into_output(
    operand: Option<(&[f64], TensorShape)>,
    indices: (&[u16], &[(u16, usize)]),
    map: &[u16; 64],
    extent: &[usize],
    coeff: f64,
    out: &mut [f64],
) {
    if out.is_empty() {
        return;
    }
    let (free, fixed) = indices;
    let unit = [1.0];
    let (data, shape) = operand.unwrap_or((&unit, strided_tensor_shape(&[], extent)));

    // Extent, output stride and operand stride of every representative of the free indices;
    // free indices sharing a representative add their output strides, and an index whose
    // representative is fixed moves the base offset to its value.
    let mut reps = [(0u16, 0usize, 0usize, 0usize); 2 * MAXLABELS];
    let mut nr = 0;
    let mut base = 0;
    let mut stride = out.len();
    for &l in free {
        stride /= extent[l as usize];
        let r = map[l as usize];
        if let Some(&(_, v)) = fixed.iter().find(|x| x.0 == r) {
            base += v * stride;
            continue;
        }
        match reps[..nr].iter().position(|x| x.0 == r) {
            Some(k) => reps[k].2 += stride,
            None => {
                let from = shape.labels[..shape.n]
                    .iter()
                    .position(|&x| x == r)
                    .map_or(0, |p| shape.strides[p]);
                reps[nr] = (r, extent[r as usize], stride, from);
                nr += 1;
            }
        }
    }
    let reps = &reps[..nr];

    // Remaining operand labels are summed.
    let rep_mask = reps.iter().fold(0u64, |m, x| m | (1 << x.0));
    let mut summed = [(0usize, 0usize); MAXLABELS];
    let mut ns = 0;
    for k in 0..shape.n {
        if rep_mask & (1 << shape.labels[k]) == 0 {
            summed[ns] = (shape.dims[k], shape.strides[k]);
            ns += 1;
        }
    }
    let summed = &summed[..ns];
    let total = |offset: usize| {
        if summed.is_empty() {
            return data[offset];
        }
        let mut idx = [0usize; MAXLABELS];
        let mut extra = 0;
        let mut total = 0.0;
        loop {
            total += data[offset + extra];
            let mut k = summed.len();
            loop {
                if k == 0 {
                    return total;
                }
                k -= 1;
                idx[k] += 1;
                extra += summed[k].1;
                if idx[k] < summed[k].0 {
                    break;
                }
                extra -= summed[k].1 * summed[k].0;
                idx[k] = 0;
            }
        }
    };

    // Walk the representatives with the last one innermost, advancing both offsets by odometer.
    let (last, inner) = match reps.split_last() {
        Some((&(_, d, so, sd), outer)) => ((d, so, sd), outer),
        None => ((1, 0, 0), reps),
    };
    let count = inner.iter().map(|x| x.1).product::<usize>();
    let mut idx = [0usize; 2 * MAXLABELS];
    let (mut o, mut p) = (base, 0usize);
    for _ in 0..count {
        let (d, so, sd) = last;
        if summed.is_empty() {
            for i in 0..d {
                out[o + i * so] += coeff * data[p + i * sd];
            }
        } else {
            for i in 0..d {
                out[o + i * so] += coeff * total(p + i * sd);
            }
        }

        let mut k = inner.len();
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            o += inner[k].2;
            p += inner[k].3;
            if idx[k] < inner[k].1 {
                break;
            }
            o -= inner[k].2 * inner[k].1;
            p -= inner[k].3 * inner[k].1;
            idx[k] = 0;
        }
    }
}
