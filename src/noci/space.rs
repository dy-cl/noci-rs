// noci/space.rs

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use itertools::Itertools;

// Crate-root imports.
use crate::basis::{excitation_between, undo_excitation};
use crate::determinant::{
    DeterminantState, ParentComponents, ParentDeterminant, ReducedOneSpinDeterminantState,
    ReducedTwoSpinDeterminantState, SpinDeterminantIndex, SpinDeterminantState,
};
use crate::input::Input;
use crate::{ExcitationSpin, ReducedTwoSpinState, SCFState, StateScalar};

macro_rules! determinant_index {
    ($name:ident, $trait_name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub(crate) struct $name(
            /// Parent-local spin-component position.
            pub(crate) usize,
        );

        impl $trait_name for $name {
            /// Construct a compact index from its position.
            /// # Arguments:
            /// - `index`: Parent-local position.
            /// # Returns:
            /// - `Self`: Typed component index.
            #[inline(always)]
            fn from_usize(index: usize) -> Self {
                Self(index)
            }

            /// Return the position encoded by this index.
            /// # Arguments:
            /// - `self`: Typed component index.
            /// # Returns:
            /// - `usize`: Parent-local position.
            #[inline(always)]
            fn as_usize(self) -> usize {
                self.0
            }
        }
    };
}

determinant_index!(NOCISpinIndex, SpinDeterminantIndex);

/// Global retained determinant index in the ordered NOCI basis.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NOCIIndex(
    /// Ordered retained-basis position.
    pub usize,
);

/// Parent and spin-component identity of one retained determinant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NOCIDeterminantState {
    /// Orbital-frame parent shared with other descendants.
    pub parent: usize,
    /// Retained alpha component ID within that parent.
    pub(crate) aid: NOCISpinIndex,
    /// Retained beta component ID within that parent.
    pub(crate) bid: NOCISpinIndex,
}

impl DeterminantState for NOCIDeterminantState {
    type SpinIndex = NOCISpinIndex;

    /// Return the retained determinant's parent orbital frame.
    /// # Arguments:
    /// - `self`: Retained determinant identity.
    /// # Returns:
    /// - `usize`: Parent frame index.
    fn parent(&self) -> usize {
        self.parent
    }

    /// Return the retained alpha component ID.
    /// # Arguments:
    /// - `self`: Retained determinant identity.
    /// # Returns:
    /// - `NOCISpinIndex`: Alpha component ID.
    fn aid(&self) -> Self::SpinIndex {
        self.aid
    }

    /// Return the retained beta component ID.
    /// # Arguments:
    /// - `self`: Retained determinant identity.
    /// # Returns:
    /// - `NOCISpinIndex`: Beta component ID.
    fn bid(&self) -> Self::SpinIndex {
        self.bid
    }
}

/// Parent-local retained one-spin identity and reduced kernel payload.
pub(crate) type ReducedOneSpinNOCIDeterminantState = ReducedOneSpinDeterminantState<NOCISpinIndex>;
/// Retained determinant identity and reduced two-spin kernel payload.
pub(crate) type ReducedTwoSpinNOCIDeterminantState = ReducedTwoSpinDeterminantState<NOCIIndex>;

/// Authoritative ordered NOCI determinant topology and identity registry.
/// Active retained, selected or candidate subspaces are represented by ordered
/// collections of `NOCIIndex` values into this space.
pub struct NOCISpace<T: StateScalar> {
    /// Shared parent orbital frames defining all determinants in this space.
    pub(crate) parents: Arc<[ParentDeterminant<T>]>,
    /// Canonical parent-local spin components referenced by determinant identities.
    pub(crate) components: Vec<ParentComponents<NOCISpinIndex>>,
    /// Ordered determinant identities registered in this space.
    pub(crate) states: Vec<NOCIDeterminantState>,
    /// Labels ordered identically to `states`.
    pub(crate) labels: Vec<String>,
    /// Parent-relative numerical payloads ordered identically to `states`.
    pub(crate) reduced: Vec<ReducedTwoSpinState>,
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
            components: (0..self.parents.len())
                .map(|_| ParentComponents::new())
                .collect(),
            states: Vec::new(),
            labels: Vec::new(),
            reduced: Vec::new(),
        };

        for &source in sources {
            let state = self.state(source);
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
                                        alpha.excitation.holes,
                                        alpha.excitation.parts,
                                    );
                                    let parent_ob = undo_excitation(
                                        beta.occupation,
                                        beta.excitation.holes,
                                        beta.excitation.parts,
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
        let components = (0..parents.len())
            .map(|_| ParentComponents::new())
            .collect();

        Self {
            parents: parents.into(),
            components,
            states: Vec::new(),
            labels: Vec::new(),
            reduced: Vec::new(),
        }
    }

    /// Build an ordered temporary NOCI topology for a numerical determinant subset.
    /// Parent orbital frames remain shared; component metadata is interned in subset order.
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
            components: (0..self.parents.len())
                .map(|_| ParentComponents::new())
                .collect(),
            states: Vec::new(),
            labels: Vec::new(),
            reduced: Vec::new(),
        };

        for &index in indices {
            let state = self.state(index);
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
        let components = &mut self.components[parent];

        let aid = components.intern_alpha(SpinDeterminantState::new(
            reference.oa,
            oa,
            reference.ca.ncols(),
        ));
        let bid = components.intern_beta(SpinDeterminantState::new(
            reference.ob,
            ob,
            reference.cb.ncols(),
        ));

        let reduced =
            ReducedTwoSpinState::from_spin_states(components.alpha(aid), components.beta(bid));
        let index = NOCIIndex(self.states.len());

        self.states.push(NOCIDeterminantState { parent, aid, bid });
        self.labels.push(label);
        self.reduced.push(reduced);

        index
    }

    /// Return the retained determinant count.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// # Returns:
    /// - `usize`: Number of retained determinants.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Return whether the retained basis has no determinant identities.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// # Returns:
    /// - `bool`: Whether the ordered retained basis is empty.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
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

    /// Return one retained determinant identity.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `NOCIDeterminantState`: Parent and component IDs.
    #[inline(always)]
    pub fn state(
        &self,
        index: NOCIIndex,
    ) -> NOCIDeterminantState {
        self.states[index.0]
    }

    /// Return one parent's canonical spin-component storage.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `parent`: Parent orbital-frame index.
    /// # Returns:
    /// - `&ParentComponents<NOCISpinIndex>`: Canonical retained components.
    #[inline(always)]
    pub(crate) fn parent_components(
        &self,
        parent: usize,
    ) -> &ParentComponents<NOCISpinIndex> {
        &self.components[parent]
    }

    /// Return the parent orbital frame of one retained determinant.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `&ParentDeterminant<T>`: Shared parent orbital frame.
    pub(crate) fn parent(
        &self,
        index: NOCIIndex,
    ) -> &ParentDeterminant<T> {
        &self.parents[self.state(index).parent]
    }

    /// Return the retained alpha component of one determinant.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `&SpinDeterminantState`: Canonical alpha component.
    pub(crate) fn alpha(
        &self,
        index: NOCIIndex,
    ) -> &SpinDeterminantState {
        let state = self.state(index);
        self.components[state.parent].alpha(state.aid)
    }

    /// Return the retained beta component of one determinant.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `&SpinDeterminantState`: Canonical beta component.
    pub(crate) fn beta(
        &self,
        index: NOCIIndex,
    ) -> &SpinDeterminantState {
        let state = self.state(index);
        self.components[state.parent].beta(state.bid)
    }

    /// Return an ordered retained determinant's numerical payload.
    /// # Arguments:
    /// - `self`: Retained NOCI space.
    /// - `index`: Global retained determinant index.
    /// # Returns:
    /// - `ReducedTwoSpinNOCIDeterminantState`: Derived identity-bearing numerical view.
    pub(crate) fn reduced(
        &self,
        index: NOCIIndex,
    ) -> ReducedTwoSpinNOCIDeterminantState {
        ReducedTwoSpinNOCIDeterminantState {
            det: index,
            state: self.reduced[index.0],
        }
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
        (&self.alpha(index).excitation, &self.beta(index).excitation)
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
        self.reduced(index).state.phase
    }
}
