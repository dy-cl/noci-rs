// noci/auxiliary.rs

// Crate-root imports.
use crate::StateScalar;
use crate::determinant::{
    DeterminantState, ParentComponents, SpinDeterminantIndex, SpinDeterminantState,
};

// Parent/sibling imports.
use super::orthogonal::OrthogonalConnection;
use super::space::{NOCIIndex, NOCISpace};

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

determinant_index!(AuxiliarySpinIndex, SpinDeterminantIndex);

/// Global index in the implicit auxiliary Cartesian product.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct AuxiliaryIndex(
    /// Flattened physical determinant position.
    pub(crate) usize,
);

/// Parent and spin-component identity of one physical auxiliary determinant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AuxiliaryDeterminantState {
    /// Orbital-frame parent shared with the retained NOCI space.
    pub(crate) parent: usize,
    /// Physical alpha component ID within that parent.
    pub(crate) aid: AuxiliarySpinIndex,
    /// Physical beta component ID within that parent.
    pub(crate) bid: AuxiliarySpinIndex,
}

impl DeterminantState for AuxiliaryDeterminantState {
    type SpinIndex = AuxiliarySpinIndex;

    /// Return the auxiliary determinant's parent orbital frame.
    /// # Arguments:
    /// - `self`: Auxiliary determinant identity.
    /// # Returns:
    /// - `usize`: Parent frame index.
    fn parent(&self) -> usize {
        self.parent
    }

    /// Return the auxiliary alpha component ID.
    /// # Arguments:
    /// - `self`: Auxiliary determinant identity.
    /// # Returns:
    /// - `AuxiliarySpinIndex`: Alpha component ID.
    fn aid(&self) -> Self::SpinIndex {
        self.aid
    }

    /// Return the auxiliary beta component ID.
    /// # Arguments:
    /// - `self`: Auxiliary determinant identity.
    /// # Returns:
    /// - `AuxiliarySpinIndex`: Beta component ID.
    fn bid(&self) -> Self::SpinIndex {
        self.bid
    }
}

/// Physical BApply closure with NOCI components as exact alpha/beta prefixes.
/// The two-spin Cartesian product is represented only by component lists and offsets.
pub(crate) struct AuxiliarySpace {
    /// Parent-local physical components; each retained NOCI list is an exact prefix.
    components: Vec<ParentComponents<AuxiliarySpinIndex>>,
    /// Start of each parent's flattened implicit product, with a final end offset.
    offsets: Vec<usize>,
}

impl AuxiliarySpace {
    /// Build the H-connected component closure with retained NOCI components as exact prefixes.
    /// # Arguments:
    /// - `noci`: Authoritative retained determinant space.
    /// # Returns:
    /// - `Self`: Implicit auxiliary product and deterministic offsets.
    pub(crate) fn new<T: StateScalar>(noci: &NOCISpace<T>) -> Self {
        let mut components = Vec::with_capacity(noci.parents.len());
        let mut offsets = Vec::with_capacity(noci.parents.len() + 1);
        offsets.push(0);

        for (parent, retained) in noci.components.iter().enumerate() {
            let reference = &noci.parents[parent];
            let mut auxiliary = ParentComponents::new();

            for component in &retained.alpha {
                auxiliary.intern_alpha(component.clone());
            }

            for component in &retained.beta {
                auxiliary.intern_beta(component.clone());
            }

            // Each one-spin H connection changes one or two electrons. Pairing the independent
            // alpha and beta single closures also covers every opposite-spin double child.
            let mut extra_alpha = Vec::new();
            let mut extra_beta = Vec::new();
            for component in &retained.alpha {
                reachable_spin_occupations(component, &mut extra_alpha);
            }

            for component in &retained.beta {
                reachable_spin_occupations(component, &mut extra_beta);
            }

            extra_alpha.sort_unstable();
            extra_alpha.dedup();

            extra_beta.sort_unstable();
            extra_beta.dedup();

            for occupation in extra_alpha {
                auxiliary.intern_alpha(SpinDeterminantState::new(
                    reference.oa,
                    occupation,
                    reference.ca.ncols(),
                ));
            }

            for occupation in extra_beta {
                auxiliary.intern_beta(SpinDeterminantState::new(
                    reference.ob,
                    occupation,
                    reference.cb.ncols(),
                ));
            }

            let next = offsets[parent] + auxiliary.na() * auxiliary.nb();
            offsets.push(next);
            components.push(auxiliary);
        }

        Self {
            components,
            offsets,
        }
    }

    /// Flatten a structured auxiliary determinant within its parent's implicit product.
    /// # Arguments:
    /// - `self`: Auxiliary determinant space.
    /// - `state`: Parent and physical auxiliary component IDs.
    /// # Returns:
    /// - `AuxiliaryIndex`: Deterministic flattened determinant index.
    pub(crate) fn index(
        &self,
        state: AuxiliaryDeterminantState,
    ) -> AuxiliaryIndex {
        let nb = self.components[state.parent].nb();
        AuxiliaryIndex(self.offsets[state.parent] + state.aid.0 * nb + state.bid.0)
    }

    /// Decode one flattened auxiliary determinant without allocating product states.
    /// # Arguments:
    /// - `self`: Auxiliary determinant space.
    /// - `index`: Deterministic flattened determinant index.
    /// # Returns:
    /// - `AuxiliaryDeterminantState`: Parent and component IDs.
    pub(crate) fn state(
        &self,
        index: AuxiliaryIndex,
    ) -> AuxiliaryDeterminantState {
        let parent = self.offsets.partition_point(|&offset| offset <= index.0) - 1;
        let local = index.0 - self.offsets[parent];
        let nb = self.components[parent].nb();

        AuxiliaryDeterminantState {
            parent,
            aid: AuxiliarySpinIndex(local / nb),
            bid: AuxiliarySpinIndex(local % nb),
        }
    }

    /// Embed a retained determinant using the exact component-prefix invariant.
    /// # Arguments:
    /// - `self`: Auxiliary determinant space.
    /// - `noci`: Retained determinant space with matching parent frames.
    /// - `index`: Retained determinant index.
    /// # Returns:
    /// - `AuxiliaryIndex`: Physical auxiliary identity.
    pub(crate) fn embed_noci<T: StateScalar>(
        &self,
        noci: &NOCISpace<T>,
        index: NOCIIndex,
    ) -> AuxiliaryIndex {
        let state = noci.state(index);
        self.index(AuxiliaryDeterminantState {
            parent: state.parent(),
            aid: AuxiliarySpinIndex(state.aid().0),
            bid: AuxiliarySpinIndex(state.bid().0),
        })
    }

    /// Resolve a surviving orthogonal H spawn through parent-local occupation maps.
    /// # Arguments:
    /// - `self`: Auxiliary determinant space.
    /// - `noci`: Retained source determinant space.
    /// - `source`: Retained source determinant index.
    /// - `connection`: Surviving source-relative H connection.
    /// # Returns:
    /// - `AuxiliaryIndex`: Physical auxiliary child identity.
    pub(crate) fn connected<T: StateScalar>(
        &self,
        noci: &NOCISpace<T>,
        source: NOCIIndex,
        connection: OrthogonalConnection,
    ) -> AuxiliaryIndex {
        let state = noci.state(source);
        let retained = noci.parent_components(state.parent());
        let (oa, ob) =
            connection.child_occupations(retained.alpha(state.aid()), retained.beta(state.bid()));

        let components = &self.components[state.parent()];
        self.index(AuxiliaryDeterminantState {
            parent: state.parent(),
            aid: components.aids[&oa],
            bid: components.bids[&ob],
        })
    }

    /// Return one parent's auxiliary spin components.
    /// # Arguments:
    /// - `self`: Auxiliary determinant space.
    /// - `parent`: Parent orbital-frame index.
    /// # Returns:
    /// - `&ParentComponents<AuxiliarySpinIndex>`: Auxiliary spin components.
    pub(crate) fn parent_components(
        &self,
        parent: usize,
    ) -> &ParentComponents<AuxiliarySpinIndex> {
        &self.components[parent]
    }
}

/// Enumerate all physical one-spin occupations connected by a single or same-spin double.
/// # Arguments:
/// - `state`: Source component with rank-to-orbital lookup tables.
/// - `out`: Accumulator for occupations subsequently sorted and deduplicated.
/// # Returns
/// - `()`: Appends reachable occupations to `out`.
fn reachable_spin_occupations(
    state: &SpinDeterminantState,
    out: &mut Vec<u128>,
) {
    for &hole in &state.occupied {
        for &particle in &state.virtuals {
            out.push((state.occupation & !(1u128 << hole)) | (1u128 << particle));
        }
    }
    // Choose unordered hole and particle pairs so each same-spin double
    // occupation is generated once before the caller deduplicates results.
    for i in 0..state.occupied.len() {
        for j in i + 1..state.occupied.len() {
            for a in 0..state.virtuals.len() {
                for b in a + 1..state.virtuals.len() {
                    let holes = (1u128 << state.occupied[i]) | (1u128 << state.occupied[j]);
                    let parts = (1u128 << state.virtuals[a]) | (1u128 << state.virtuals[b]);
                    out.push((state.occupation & !holes) | parts);
                }
            }
        }
    }
}
