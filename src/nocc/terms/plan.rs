// nocc/terms/plan.rs
//! Planning of generated term tables: kept terms, factor blocks, contraction orders and
//! shared products.

// Standard library imports.
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::maths::contract::{LabelSizes, optimal_contraction_steps};
use crate::nocc::space::Spaces;

// Parent/sibling imports.
use super::schema::GeneratedTerm;
use super::tensors::{SpaceKind, TensorKind, space_orbitals};

/// Largest number of terms evaluated together from one computed shared product.
const GROUPTERMS: usize = 1024;

/// Orbital-space block of one tensor kind: the kind and the space of every slot, upper then
/// lower.
pub(super) type BlockKey = (TensorKind, Vec<SpaceKind>);

/// One term table: its terms and the name and space of every class-local index.
pub(in crate::nocc) type TermTable<'a> = (&'a [GeneratedTerm], &'a [(String, u8)]);

/// Dense block of every factor of every kept term in one table, resolved once.
pub(in crate::nocc) struct TablePlan {
    /// Distinct kind and slot-space patterns used by the kept terms.
    pub(super) keys: Vec<BlockKey>,
    /// Index of every kept term in the table.
    pub(super) terms: Vec<u32>,
    /// Block of every tensor factor, for all kept terms in order.
    pub(super) factor_blocks: Vec<u32>,
    /// Start of every kept term's entries in `factor_blocks`, with a final end marker.
    pub(super) term_starts: Vec<u32>,
    /// Pairwise contraction steps of every kept term in its optimal order, as operand numbers:
    /// factors are numbered delta then tensor and each step's result takes the next number.
    pub(super) steps: Vec<(u8, u8)>,
    /// Start of every kept term's entries in `steps`, with a final end marker.
    pub(super) step_starts: Vec<u32>,
    /// Label substitutions `(l, r)` of every kept term that resolve its Kronecker deltas.
    pub(super) substitutions: Vec<(u16, u16)>,
    /// Start of every kept term's entries in `substitutions`, with a final end marker.
    pub(super) substitution_starts: Vec<u32>,
    /// Coefficient of every kept term, including the extent of every summed label left only in
    /// resolved deltas.
    pub(super) coefficients: Vec<f64>,
    /// Products of two factor blocks shared by several kept terms, in canonical labels.
    pub(super) shared: Vec<SharedProduct>,
    /// Shared product of every kept term, its step number, and the term label of every
    /// canonical label, or `None`.
    pub(super) anchors: Vec<Option<Anchor>>,
    /// Kept-term positions grouped by shared product, each group with its product.
    pub(super) groups: Vec<(Range<usize>, Option<u32>)>,
    /// Kept-term positions in group order.
    pub(super) order: Vec<u32>,
}

/// Product of two factor blocks over canonical labels, `C = A B` summed over the labels not
/// kept, computed once for every kept term that contains it.
pub(super) struct SharedProduct {
    /// Table-local block of each operand.
    pub(super) blocks: [u32; 2],
    /// Canonical label of every slot of each operand.
    pub(super) labels: [Vec<u16>; 2],
    /// Bit mask of the canonical labels kept in the product.
    pub(super) keep: u64,
    /// Extent of every canonical label.
    pub(super) extent: Vec<usize>,
}

/// Use of a shared product by one kept term.
#[derive(Clone)]
pub(super) struct Anchor {
    /// Shared product id.
    pub(super) product: u32,
    /// Step of the term the product replaces.
    pub(super) step: u8,
    /// Term label of every canonical label.
    pub(super) labels: Vec<u16>,
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
            dims: SpaceKind::ALL.map(|s| space_orbitals(spaces, s).len()),
            plans: Mutex::new(HashMap::new()),
        }
    }

    /// Return the plan of one working-equation table, the energy or a residual, resolving it on
    /// first use. Terms with cumulants above the truncation rank are dropped.
    /// # Arguments:
    /// - `table`: Terms and index spaces of the table.
    /// # Returns:
    /// - `Arc<TablePlan>`: Block of every factor of every kept term.
    pub(in crate::nocc) fn table_plan(
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
    pub(in crate::nocc) fn exact_table_plan(
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
/// # Panics
/// - Panics if a factor has an unknown tensor kind or an index an unknown orbital space.
fn resolve_table_plan(
    table: TermTable<'_>,
    max_cumulant: usize,
    dims: [usize; 3],
) -> TablePlan {
    let (terms, indices) = table;
    let space = |x: &u16| SpaceKind::from_id(indices[*x as usize].1);
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

    let extent = indices
        .iter()
        .map(|&(_, s)| dims[s as usize])
        .collect::<Vec<_>>();
    let mut substitutions = Vec::new();
    let mut substitution_starts = Vec::with_capacity(terms.len() + 1);
    let mut coefficients = Vec::with_capacity(terms.len());

    for (t, term) in terms.iter().enumerate() {
        if term
            .3
            .iter()
            .any(|f| TensorKind::from_id(f.0).cumulant_rank() > max_cumulant)
        {
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
            factor_blocks.push(intern((TensorKind::from_id(f.0), slots), &mut keys));
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
pub(super) fn label_map(substitutions: &[(u16, u16)]) -> [u16; 64] {
    let mut map = [0u16; 64];
    for (l, x) in map.iter_mut().enumerate() {
        *x = l as u16;
    }
    for &(l, r) in substitutions {
        map[l as usize] = r;
    }
    map
}
