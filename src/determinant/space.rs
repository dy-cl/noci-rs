// determinant/space.rs
//! The retained NOCI determinant space: parent frames, the interned spin strings of every parent
//! and the spin-string pair behind each `NOCIIndex`.

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use itertools::Itertools;

// Crate-root imports.
use crate::basis::{excitation_between, undo_excitation};
use crate::determinant::{
    DeterminantState, ParentDeterminant, ParentSpinStrings, SpinStringIndex, SpinStringPair,
    SpinStringState,
};
use crate::input::Input;
use crate::{ExcitationSpin, SCFState, StateScalar};

// Parent/sibling imports.
use super::state::determinant_index;

determinant_index!(NOCISpinStringIndex, SpinStringIndex);

/// Global retained determinant index in the ordered NOCI basis.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NOCIIndex(
    /// Ordered retained-basis position.
    pub usize,
);

/// Authoritative ordered NOCI determinant topology and identity registry.
/// Active retained, selected or candidate subspaces are represented by ordered
/// collections of `NOCIIndex` values into this space.
pub struct NOCISpace<T: StateScalar> {
    /// Shared parent orbital frames defining all determinants in this space.
    pub(crate) parents: Arc<[ParentDeterminant<T>]>,
    /// Canonical parent-local spin strings referenced by determinant identities.
    pub(crate) spin_strings: Vec<ParentSpinStrings<NOCISpinStringIndex>>,
    /// Ordered determinant identities registered in this space.
    pub(crate) pairs: Vec<SpinStringPair<NOCISpinStringIndex>>,
    /// Labels ordered identically to `pairs`.
    pub(crate) labels: Vec<String>,
}

/// Format the total parent-relative excitation using the established basis labels.
/// # Arguments:
/// - `alpha_holes`: Removed parent alpha orbitals.
/// - `alpha_parts`: Added alpha orbitals.
/// - `beta_holes`: Removed parent beta orbitals.
/// - `beta_parts`: Added beta orbitals.
/// # Returns:
/// - `String`: Spin-resolved excitation label.
fn excitation_label(
    alpha_holes: u128,
    alpha_parts: u128,
    beta_holes: u128,
    beta_parts: u128,
) -> String {
    let format_mask = |mut bits: u128| {
        let mut orbitals = Vec::with_capacity(bits.count_ones() as usize);

        while bits != 0 {
            let orbital = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            orbitals.push(orbital.to_string());
        }

        orbitals.join(" ")
    };

    let mut labels = Vec::new();
    if alpha_holes != 0 {
        labels.push(format!(
            "alpha {} -> {}",
            format_mask(alpha_holes),
            format_mask(alpha_parts),
        ));
    }
    if beta_holes != 0 {
        labels.push(format!(
            "beta {} -> {}",
            format_mask(beta_holes),
            format_mask(beta_parts),
        ));
    }

    format!("({})", labels.join("; "))
}

impl<T: StateScalar> NOCISpace<T> {
    /// Construct reference topology from selected SCF solutions in their existing order.
    /// # Arguments:
    /// - `states`: Selected and occupied-first SCF references.
    /// # Returns:
    /// - `Self`: One parent and one retained reference determinant per SCF state.
    pub fn from_scf(states: &[SCFState<T>]) -> Self {
        let parents = states
            .iter()
            .map(ParentDeterminant::from_scf)
            .collect::<Vec<_>>();
        let mut space = Self::from_parents(parents);

        for parent in 0..space.parents.len() {
            let reference = &space.parents[parent];
            let (oa, ob, label) = (reference.oa, reference.ob, reference.label.clone());
            space.push(parent, oa, ob, label);
        }

        space
    }

    /// Generate requested excitations from retained sources in their supplied order.
    /// Each source may already be excited, but every child is interned relative to the
    /// original parent frame rather than to the immediate source determinant.
    /// # Arguments:
    /// - `self`: Space containing the retained source determinants.
    /// - `sources`: Source indices in candidate-generation order.
    /// - `input`: Requested excitation orders.
    /// - `include_sources`: Whether source determinants precede their children.
    /// # Returns:
    /// - `Self`: New ordered retained topology sharing the selected parent frames.
    pub fn excited_from(
        &self,
        sources: &[NOCIIndex],
        input: &Input,
        include_sources: bool,
    ) -> Self {
        let mut out = Self {
            parents: Arc::clone(&self.parents),
            spin_strings: (0..self.parents.len())
                .map(|_| ParentSpinStrings::new())
                .collect(),
            pairs: Vec::new(),
            labels: Vec::new(),
        };

        for &source in sources {
            let state = self.spin_string_pair(source);
            let alpha = self.alpha(source);
            let beta = self.beta(source);

            if include_sources {
                out.push(
                    state.parent,
                    alpha.occupation,
                    beta.occupation,
                    self.labels[source.0].clone(),
                );
            }

            let mut orders = if input.excit.all {
                let maximum = (alpha.occupied.len() + beta.occupied.len())
                    .min(alpha.virtuals.len() + beta.virtuals.len());
                (1..=maximum).collect::<Vec<_>>()
            } else {
                input.excit.orders.clone()
            };
            orders.sort_unstable();
            orders.dedup();

            for &order in &orders {
                for alpha_rank in 0..=order {
                    let beta_rank = order - alpha_rank;

                    for alpha_holes in alpha.occupied.iter().copied().combinations(alpha_rank) {
                        for alpha_parts in alpha.virtuals.iter().copied().combinations(alpha_rank) {
                            for beta_holes in beta.occupied.iter().copied().combinations(beta_rank)
                            {
                                for beta_parts in
                                    beta.virtuals.iter().copied().combinations(beta_rank)
                                {
                                    // Apply the requested excitation to the supplied source,
                                    // then let `push` derive total metadata from its parent.
                                    let oa =
                                        alpha_holes.iter().fold(alpha.occupation, |occ, &hole| {
                                            occ & !(1u128 << hole)
                                        });
                                    let oa = alpha_parts
                                        .iter()
                                        .fold(oa, |occ, &part| occ | (1u128 << part));
                                    let ob = beta_holes
                                        .iter()
                                        .fold(beta.occupation, |occ, &hole| occ & !(1u128 << hole));
                                    let ob = beta_parts
                                        .iter()
                                        .fold(ob, |occ, &part| occ | (1u128 << part));

                                    // Undo the source's parent-relative excitation before
                                    // labeling the total parent-to-child excitation.
                                    let parent_oa = undo_excitation(
                                        alpha.occupation,
                                        alpha.data.excitation.holes,
                                        alpha.data.excitation.parts,
                                    );
                                    let parent_ob = undo_excitation(
                                        beta.occupation,
                                        beta.data.excitation.holes,
                                        beta.data.excitation.parts,
                                    );
                                    let (aholes, aparts) = excitation_between(parent_oa, oa);
                                    let (bholes, bparts) = excitation_between(parent_ob, ob);
                                    let label = excitation_label(aholes, aparts, bholes, bparts);

                                    out.push(
                                        state.parent,
                                        oa,
                                        ob,
                                        format!("{} {}", self.labels[source.0], label),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        out
    }

    /// Construct empty retained topology around already selected parent orbital frames.
    /// # Arguments:
    /// - `parents`: Selected parent orbital frames in reference order.
    /// # Returns:
    /// - `Self`: Empty retained topology with those frames.
    pub(crate) fn from_parents(parents: Vec<ParentDeterminant<T>>) -> Self {
        let strings = (0..parents.len())
            .map(|_| ParentSpinStrings::new())
            .collect();

        Self {
            parents: parents.into(),
            spin_strings: strings,
            pairs: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// Build an ordered temporary NOCI topology for a numerical determinant subset.
    /// Parent orbital frames remain shared; spin-string payloads is interned in subset order.
    /// # Arguments:
    /// - `self`: Source retained space.
    /// - `indices`: Determinants required by the numerical operation.
    /// # Returns:
    /// - `Self`: Ordered determinant subset with dense local indices.
    pub fn subset(
        &self,
        indices: &[NOCIIndex],
    ) -> Self {
        let mut out = Self {
            parents: Arc::clone(&self.parents),
            spin_strings: (0..self.parents.len())
                .map(|_| ParentSpinStrings::new())
                .collect(),
            pairs: Vec::new(),
            labels: Vec::new(),
        };

        for &index in indices {
            let state = self.spin_string_pair(index);
            let (oa, ob) = self.occupations(index);
            out.push(state.parent, oa, ob, self.labels[index.0].clone());
        }

        out
    }

    /// Append an ordered retained determinant, interning its spin occupations once.
    /// # Arguments:
    /// - `parent`: Parent orbital-frame index.
    /// - `oa`: Physical alpha occupation.
    /// - `ob`: Physical beta occupation.
    /// - `label`: Existing determinant label.
    /// # Returns:
    /// - `NOCIIndex`: Index of the appended retained determinant.
    pub(crate) fn push(
        &mut self,
        parent: usize,
        oa: u128,
        ob: u128,
        label: String,
    ) -> NOCIIndex {
        let reference = &self.parents[parent];
        let strings = &mut self.spin_strings[parent];

        let aid =
            strings.intern_alpha(SpinStringState::new(reference.oa, oa, reference.ca.ncols()));
        let bid = strings.intern_beta(SpinStringState::new(reference.ob, ob, reference.cb.ncols()));

        let index = NOCIIndex(self.pairs.len());

        self.pairs.push(SpinStringPair { parent, aid, bid });
        self.labels.push(label);

        index
    }

    /// Return the retained determinant count.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// # Returns:
    /// - `usize`: Number of retained determinants.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Return whether the retained basis has no determinant identities.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// # Returns:
    /// - `bool`: Whether the ordered retained basis is empty.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Return the selected parent orbital frames in reference order.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// # Returns:
    /// - `&[ParentDeterminant<T>]`: Shared orbital frames.
    pub fn parents(&self) -> &[ParentDeterminant<T>] {
        &self.parents
    }

    /// Return the label belonging to one ordered retained determinant.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Ordered retained determinant index.
    /// # Returns:
    /// - `&str`: User-visible determinant label.
    pub fn label(
        &self,
        index: NOCIIndex,
    ) -> &str {
        &self.labels[index.0]
    }

    /// Return the spin-string handles of one retained determinant.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `SpinStringPair<NOCISpinStringIndex>`: Parent and spin-string positions.
    #[inline(always)]
    pub(crate) fn spin_string_pair(
        &self,
        index: NOCIIndex,
    ) -> SpinStringPair<NOCISpinStringIndex> {
        self.pairs[index.0]
    }

    /// Return the parent orbital frame of one retained determinant.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `usize`: Parent frame index.
    #[inline(always)]
    pub fn parent(
        &self,
        index: NOCIIndex,
    ) -> usize {
        self.pairs[index.0].parent
    }

    /// Return one parent's canonical spin-string storage.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `parent`: Parent orbital-frame index.
    /// # Returns:
    /// - `&ParentSpinStrings<NOCISpinStringIndex>`: Canonical retained spin strings.
    #[inline(always)]
    pub(crate) fn parent_spin_strings(
        &self,
        parent: usize,
    ) -> &ParentSpinStrings<NOCISpinStringIndex> {
        &self.spin_strings[parent]
    }

    /// Return the parent determinant of one retained determinant.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `&ParentDeterminant<T>`: Shared parent orbital frame.
    pub(crate) fn parent_determinant(
        &self,
        index: NOCIIndex,
    ) -> &ParentDeterminant<T> {
        &self.parents[self.parent(index)]
    }

    /// Return the retained alpha spin string of one determinant.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `&SpinStringState`: Canonical alpha spin string.
    pub(crate) fn alpha(
        &self,
        index: NOCIIndex,
    ) -> &SpinStringState {
        let state = self.spin_string_pair(index);
        self.spin_strings[state.parent].alpha(state.aid)
    }

    /// Return the retained beta spin string of one determinant.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `&SpinStringState`: Canonical beta spin string.
    pub(crate) fn beta(
        &self,
        index: NOCIIndex,
    ) -> &SpinStringState {
        let state = self.spin_string_pair(index);
        self.spin_strings[state.parent].beta(state.bid)
    }

    /// Return the physical alpha and beta occupations of one retained determinant.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `(u128, u128)`: Physical spin occupations.
    pub fn occupations(
        &self,
        index: NOCIIndex,
    ) -> (u128, u128) {
        (self.alpha(index).occupation, self.beta(index).occupation)
    }

    /// Return a retained determinant's parent-relative spin excitations.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `(&ExcitationSpin, &ExcitationSpin)`: Alpha and beta excitations.
    pub(crate) fn excitations(
        &self,
        index: NOCIIndex,
    ) -> (&ExcitationSpin, &ExcitationSpin) {
        (
            &self.alpha(index).data.excitation,
            &self.beta(index).data.excitation,
        )
    }

    /// Return one retained determinant's parent-relative two-spin phase.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `f64`: Product of alpha and beta phases.
    pub(crate) fn phase(
        &self,
        index: NOCIIndex,
    ) -> f64 {
        self.alpha(index).data.phase * self.beta(index).data.phase
    }

    /// Return the identity of one retained determinant, its parent and occupations.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `DeterminantState`: Parent and alpha and beta occupations.
    pub(crate) fn determinant(
        &self,
        index: NOCIIndex,
    ) -> DeterminantState {
        DeterminantState {
            parent: self.parent(index),
            alpha: self.alpha(index).occupation,
            beta: self.beta(index).occupation,
        }
    }
}
