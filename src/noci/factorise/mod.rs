// noci/factorise/mod.rs
//! Spin-factorised determinant topology and factorised NOCI operators.

mod onebody;
mod overlap;
mod storage;

// Crate-visible type re-exports.
pub(crate) use onebody::{OneBodyFactorisation, OneBodyScratch};
pub(crate) use overlap::{AuxiliaryOverlapScratch, OverlapFactors, OverlapScratch};

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::{NOCISpace, SpinStringData};
use crate::elements::NOCIData;
use crate::elements::nonorthogonalwicks::SpinStringSources;

/// Actual determinant entry in a parent-local spin factorisation.
#[derive(Clone, Copy)]
pub(crate) struct FactorEntry {
    /// Global determinant index `I`.
    pub(crate) det: usize,
    /// Parent-local alpha spin string `a_I`.
    pub(crate) a: usize,
    /// Parent-local beta spin string `b_I`.
    pub(crate) b: usize,
}

/// Parent-local determinant space in the shared spin factorisation.
#[derive(Default)]
pub(super) struct ParentSpinSpace {
    /// Payload of each parent-local alpha spin string.
    pub(super) areps: Vec<SpinStringData>,
    /// Alpha representatives in packed Wick evaluation order.
    pub(super) a_sources: SpinStringSources,
    /// Payload of each parent-local beta spin string.
    pub(super) breps: Vec<SpinStringData>,
    /// Beta representatives in packed Wick evaluation order.
    pub(super) b_sources: SpinStringSources,
    /// Actual determinants belonging to this parent as `(I,a_I,b_I)`.
    pub(super) entries: Vec<FactorEntry>,
    /// Determinant indices grouped by parent-local alpha spin string.
    pub(super) entries_by_a: Vec<Vec<usize>>,
    /// Determinant indices grouped by parent-local beta spin string.
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
    /// Prepare parent-local factor evaluation orders from authoritative spin-string IDs.
    /// # Arguments:
    /// - `data`: Retained NOCI topology and matrix-element data.
    /// # Returns:
    /// - `Self`: Numerical spin-factorised evaluation plan.
    pub(crate) fn new<T: NOCIScalar>(data: &NOCIData<'_, T>) -> Self {
        let parents = build_parent_spin_spaces(data.space);
        let ma = data
            .space
            .spin_strings
            .iter()
            .map(|parent| parent.na())
            .max()
            .unwrap_or(0);
        let mb = data
            .space
            .spin_strings
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

/// Build the numerical factor plan from canonical parent-local spin-string identities.
/// The retained determinant set is sparse within each `A_P \\times B_P` product, so entries are
/// recorded in basis order while one-spin evaluation orders come from the spin-string payloads.
/// # Arguments:
/// - `space`: Authoritative retained NOCI determinant topology.
/// # Returns:
/// - `Vec<ParentSpinSpace>`: Parent-local factor orders and retained entries.
fn build_parent_spin_spaces<T: NOCIScalar>(space: &NOCISpace<T>) -> Vec<ParentSpinSpace> {
    // Allocate one independent factor topology per reference parent.
    let mut parents = (0..space.parents.len())
        .map(|_| ParentSpinSpace::default())
        .collect::<Vec<_>>();

    // Index each retained determinant by both of its spin-string identities.
    for (det, state) in space.pairs.iter().enumerate() {
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

    // Copy each parent's canonical spin-string data in ID order: FactorEntry.a/b index these
    // arrays, so their order must continue to match the retained determinant topology.
    for (parent_id, parent) in parents.iter_mut().enumerate() {
        let strings = space.parent_spin_strings(parent_id);
        parent.areps = strings.alpha.iter().map(|string| string.data).collect();
        parent.breps = strings.beta.iter().map(|string| string.data).collect();

        // Build a separate evaluation order grouped by excitation rank and common holes.
        // SpinStringSources keeps the original columns so factor rows scatter back to ID order.
        parent.a_sources = SpinStringSources::new(&parent.areps);
        parent.b_sources = SpinStringSources::new(&parent.breps);
    }

    parents
}
