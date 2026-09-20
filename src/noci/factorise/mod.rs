// noci/factorise/mod.rs
//! Spin-factorised determinant topology and factorised NOCI operators.

mod onebody;
mod overlap;
mod storage;

// Crate-visible type re-exports.
pub(crate) use onebody::{OneBodyFactorisation, OneBodyScratch};
pub(crate) use overlap::{OverlapFactors, OverlapScratch};

// Crate-root imports.
use crate::ExcitationSpinCache;

// Parent/sibling imports.
use super::space::{NOCISpace, NOCISpinIndex, ReducedOneSpinNOCIDeterminantState};
use super::types::{NOCIData, NOCIScalar};

/// Actual determinant entry in a parent-local spin factorisation.
#[derive(Clone, Copy)]
pub(crate) struct FactorEntry {
    /// Global determinant index `I`.
    pub(crate) det: usize,
    /// Parent-local alpha component `a_I`.
    pub(crate) a: usize,
    /// Parent-local beta component `b_I`.
    pub(crate) b: usize,
}

/// Parent-local determinant space in the shared spin factorisation.
#[derive(Default)]
pub(super) struct ParentSpinSpace {
    /// Reduced representative for each parent-local alpha component.
    pub(super) areps: Vec<ReducedOneSpinNOCIDeterminantState>,
    /// Alpha component IDs in fixed-rank Wick evaluation order.
    pub(super) a_eval_order: Vec<usize>,
    /// Alpha excitation caches in fixed-rank Wick evaluation order.
    pub(super) a_eval_caches: Vec<ExcitationSpinCache>,
    /// Alpha excitation phases in fixed-rank Wick evaluation order.
    pub(super) a_eval_phases: Vec<f64>,
    /// Boundaries of equal-rank, common-hole alpha evaluation groups.
    pub(super) a_eval_groups: Vec<usize>,
    /// Reduced representative for each parent-local beta component.
    pub(super) breps: Vec<ReducedOneSpinNOCIDeterminantState>,
    /// Beta component IDs in fixed-rank Wick evaluation order.
    pub(super) b_eval_order: Vec<usize>,
    /// Beta excitation caches in fixed-rank Wick evaluation order.
    pub(super) b_eval_caches: Vec<ExcitationSpinCache>,
    /// Beta excitation phases in fixed-rank Wick evaluation order.
    pub(super) b_eval_phases: Vec<f64>,
    /// Boundaries of equal-rank, common-hole beta evaluation groups.
    pub(super) b_eval_groups: Vec<usize>,
    /// Actual determinants belonging to this parent as `(I,a_I,b_I)`.
    pub(super) entries: Vec<FactorEntry>,
    /// Determinant indices grouped by parent-local alpha component.
    pub(super) entries_by_a: Vec<Vec<usize>>,
    /// Determinant indices grouped by parent-local beta component.
    pub(super) entries_by_b: Vec<Vec<usize>>,
}

/// Numerical spin-factorised evaluation plan for the retained NOCI space.
pub(crate) struct SpinFactorisation {
    /// Largest parent-local alpha factor dimension.
    pub(super) ma: usize,
    /// Largest parent-local beta factor dimension.
    pub(super) mb: usize,
    /// Parent-local factor evaluation orders and retained entries.
    pub(super) parents: Vec<ParentSpinSpace>,
}

impl SpinFactorisation {
    /// Prepare parent-local factor evaluation orders from authoritative component IDs.
    /// # Arguments:
    /// - `data`: Retained NOCI topology and matrix-element data.
    /// # Returns:
    /// - `Self`: Numerical spin-factorised evaluation plan.
    pub(crate) fn new<T: NOCIScalar>(data: &NOCIData<'_, T>) -> Self {
        let parents = build_parent_spin_spaces(data.space);
        let ma = data
            .space
            .components
            .iter()
            .map(|parent| parent.na())
            .max()
            .unwrap_or(0);
        let mb = data
            .space
            .components
            .iter()
            .map(|parent| parent.nb())
            .max()
            .unwrap_or(0);

        Self { ma, mb, parents }
    }

    /// Return the number of parent reference blocks in the numerical plan.
    /// # Arguments:
    /// - `self`: Prepared factorisation plan.
    /// # Returns:
    /// - `usize`: Parent count.
    pub(crate) fn nparents(&self) -> usize {
        self.parents.len()
    }

    /// Return the retained alpha and beta factor dimensions for one parent.
    /// # Arguments:
    /// - `self`: Prepared factorisation plan.
    /// - `parent`: Parent orbital-frame index.
    /// # Returns:
    /// - `(usize, usize)`: Alpha and beta factor dimensions.
    pub(crate) fn parent_component_counts(
        &self,
        parent: usize,
    ) -> (usize, usize) {
        (
            self.parents[parent].areps.len(),
            self.parents[parent].breps.len(),
        )
    }

    /// Return retained entries grouped by one parent for factor contractions.
    /// # Arguments:
    /// - `self`: Prepared factorisation plan.
    /// - `parent`: Parent orbital-frame index.
    /// # Returns:
    /// - `&[FactorEntry]`: Ordered retained entries.
    pub(crate) fn parent_entries(
        &self,
        parent: usize,
    ) -> &[FactorEntry] {
        &self.parents[parent].entries
    }
}

/// Ordered Wick parent pair `(x,w)` and whether target parent is left.
/// Existing factorised overlap evaluates Wick factors with the earlier parent block on the left.
/// This preserves that ordering convention for every operator using the shared topology.
/// # Arguments:
/// - `factorisation`: Shared determinant-space spin topology.
/// - `target_parent`: Target parent `Q`.
/// - `source_parent`: Source parent `P`.
/// # Returns
/// - `(usize, usize, bool)`: Ordered pair `(lp,gp,target_left)`.
pub(super) fn ordered_parent_pair(
    factorisation: &SpinFactorisation,
    target_parent: usize,
    source_parent: usize,
) -> (usize, usize, bool) {
    let target_first = factorisation.parents[target_parent]
        .entries
        .first()
        .map_or(usize::MAX, |entry| entry.det);
    let source_first = factorisation.parents[source_parent]
        .entries
        .first()
        .map_or(usize::MAX, |entry| entry.det);

    if target_first <= source_first {
        (target_parent, source_parent, true)
    } else {
        (source_parent, target_parent, false)
    }
}

/// Build the numerical factor plan from canonical parent-local component identities.
/// The retained determinant set is sparse within each `A_P \\times B_P` product, so entries are
/// recorded in basis order while one-spin evaluation orders come from the component caches.
/// # Arguments:
/// - `space`: Authoritative retained NOCI determinant topology.
/// # Returns:
/// - `Vec<ParentSpinSpace>`: Parent-local factor orders and retained entries.
fn build_parent_spin_spaces<T: NOCIScalar>(space: &NOCISpace<T>) -> Vec<ParentSpinSpace> {
    // Allocate one independent factor topology per reference parent.
    let mut parents = (0..space.parents.len())
        .map(|_| ParentSpinSpace::default())
        .collect::<Vec<_>>();

    // Index each retained determinant by both of its one-spin component identities.
    for (det, state) in space.states.iter().enumerate() {
        let parent = &mut parents[state.parent];
        let aid = state.aid.0;
        let bid = state.bid.0;

        parent.entries.push(FactorEntry {
            det,
            a: aid,
            b: bid,
        });

        if parent.entries_by_a.len() <= aid {
            parent.entries_by_a.resize_with(aid + 1, Vec::new);
        }
        parent.entries_by_a[aid].push(det);

        if parent.entries_by_b.len() <= bid {
            parent.entries_by_b.resize_with(bid + 1, Vec::new);
        }
        parent.entries_by_b[bid].push(det);
    }

    // Materialise canonical reduced representatives and contraction evaluation orders.
    for (parent_id, parent) in parents.iter_mut().enumerate() {
        let components = space.parent_components(parent_id);

        parent.areps = components
            .alpha
            .iter()
            .enumerate()
            .map(|(component, state)| ReducedOneSpinNOCIDeterminantState {
                parent: parent_id,
                component: NOCISpinIndex(component),
                state: state.reduced,
            })
            .collect();
        parent.breps = components
            .beta
            .iter()
            .enumerate()
            .map(|(component, state)| ReducedOneSpinNOCIDeterminantState {
                parent: parent_id,
                component: NOCISpinIndex(component),
                state: state.reduced,
            })
            .collect();

        // Fixed-rank Wick rows are grouped by rank and common holes so the numerical evaluator
        // can reuse contraction preparation independently of determinant-pair identities.
        parent.a_eval_order = (0..parent.areps.len()).collect();
        parent.a_eval_order.sort_unstable_by(|&i, &j| {
            let ic = parent.areps[i].state.excitation_cache;
            let jc = parent.areps[j].state.excitation_cache;
            ic.rank
                .cmp(&jc.rank)
                .then_with(|| ic.holes.cmp(&jc.holes))
                .then_with(|| ic.particles.cmp(&jc.particles))
                .then_with(|| i.cmp(&j))
        });
        parent.a_eval_groups.push(0);
        for position in 1..parent.a_eval_order.len() {
            let previous = parent.areps[parent.a_eval_order[position - 1]]
                .state
                .excitation_cache;
            let current = parent.areps[parent.a_eval_order[position]]
                .state
                .excitation_cache;
            if current.rank != previous.rank || current.holes != previous.holes {
                parent.a_eval_groups.push(position);
            }
        }
        if !parent.a_eval_order.is_empty() {
            parent.a_eval_groups.push(parent.a_eval_order.len());
        }
        parent.a_eval_caches = parent
            .a_eval_order
            .iter()
            .map(|&id| parent.areps[id].state.excitation_cache)
            .collect();
        parent.a_eval_phases = parent
            .a_eval_order
            .iter()
            .map(|&id| parent.areps[id].state.phase)
            .collect();

        // Mirror the rank-and-hole grouping for beta-spin contraction rows.
        parent.b_eval_order = (0..parent.breps.len()).collect();
        parent.b_eval_order.sort_unstable_by(|&i, &j| {
            let ic = parent.breps[i].state.excitation_cache;
            let jc = parent.breps[j].state.excitation_cache;
            ic.rank
                .cmp(&jc.rank)
                .then_with(|| ic.holes.cmp(&jc.holes))
                .then_with(|| ic.particles.cmp(&jc.particles))
                .then_with(|| i.cmp(&j))
        });
        parent.b_eval_groups.push(0);
        for position in 1..parent.b_eval_order.len() {
            let previous = parent.breps[parent.b_eval_order[position - 1]]
                .state
                .excitation_cache;
            let current = parent.breps[parent.b_eval_order[position]]
                .state
                .excitation_cache;
            if current.rank != previous.rank || current.holes != previous.holes {
                parent.b_eval_groups.push(position);
            }
        }
        if !parent.b_eval_order.is_empty() {
            parent.b_eval_groups.push(parent.b_eval_order.len());
        }
        parent.b_eval_caches = parent
            .b_eval_order
            .iter()
            .map(|&id| parent.breps[id].state.excitation_cache)
            .collect();
        parent.b_eval_phases = parent
            .b_eval_order
            .iter()
            .map(|&id| parent.breps[id].state.phase)
            .collect();
    }

    parents
}
