// nocc/terms/plan.rs
//! Planning of generated term tables: kept terms, factor blocks, contraction orders and the
//! shared contraction graph.

// Standard library imports.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::maths::contract::{LabelSizes, optimal_contraction_steps};
use crate::nocc::space::Spaces;

// Parent/sibling imports.
use super::graph::{TableGraph, build_table_graph};
use super::schema::GeneratedTerm;
use super::tensors::{SpaceKind, TensorKind, space_orbitals};

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
    /// Shared contraction graph of the kept terms.
    pub(super) graph: TableGraph,
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
            steps
        })
        .collect::<Vec<_>>();
    let mut steps = Vec::new();
    let mut step_starts = Vec::with_capacity(kept.len() + 1);
    for s in term_steps {
        step_starts.push(steps.len() as u32);
        steps.extend(s);
    }
    step_starts.push(steps.len() as u32);

    // The shared contraction graph of the kept terms, built over the finished plan.
    let mut plan = TablePlan {
        keys,
        terms: kept,
        factor_blocks,
        term_starts,
        steps,
        step_starts,
        substitutions,
        substitution_starts,
        coefficients,
        graph: TableGraph::default(),
    };
    plan.graph = build_table_graph(table, &plan, &extent);
    plan
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
