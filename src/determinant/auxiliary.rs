// determinant/auxiliary.rs

// Crate-root imports.
use crate::StateScalar;
use crate::determinant::{NOCIIndex, NOCISpace};
use crate::determinant::{ParentSpinStrings, SpinStringIndex, SpinStringPair, SpinStringState};

// Parent/sibling imports.
use super::connection::ConnectionState;
use super::state::determinant_index;

determinant_index!(AuxiliarySpinStringIndex, SpinStringIndex);

/// Global index in the implicit auxiliary Cartesian product.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct AuxiliaryIndex(
    /// Flattened physical determinant position.
    pub(crate) usize,
);

/// Physical BApply closure with NOCI components as exact alpha/beta prefixes.
/// The two-spin Cartesian product is represented only by component lists and offsets.
pub(crate) struct AuxiliarySpace {
    /// Parent-local physical components; each retained NOCI list is an exact prefix.
    spin_strings: Vec<ParentSpinStrings<AuxiliarySpinStringIndex>>,
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

        for (parent, retained) in noci.spin_strings.iter().enumerate() {
            let reference = &noci.parents[parent];
            let mut auxiliary = ParentSpinStrings::new();

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
                auxiliary.intern_alpha(SpinStringState::new(
                    reference.oa,
                    occupation,
                    reference.ca.ncols(),
                ));
            }

            for occupation in extra_beta {
                auxiliary.intern_beta(SpinStringState::new(
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
            spin_strings: components,
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
        state: SpinStringPair<AuxiliarySpinStringIndex>,
    ) -> AuxiliaryIndex {
        let nb = self.spin_strings[state.parent].nb();
        AuxiliaryIndex(self.offsets[state.parent] + state.aid.0 * nb + state.bid.0)
    }

    /// Decode one flattened auxiliary determinant without allocating product states.
    /// # Arguments:
    /// - `self`: Auxiliary determinant space.
    /// - `index`: Deterministic flattened determinant index.
    /// # Returns:
    /// - `SpinStringPair<AuxiliarySpinStringIndex>`: Parent and spin-string positions.
    pub(crate) fn spin_string_pair(
        &self,
        index: AuxiliaryIndex,
    ) -> SpinStringPair<AuxiliarySpinStringIndex> {
        let parent = self.offsets.partition_point(|&offset| offset <= index.0) - 1;
        let local = index.0 - self.offsets[parent];
        let nb = self.spin_strings[parent].nb();

        SpinStringPair {
            parent,
            aid: AuxiliarySpinStringIndex(local / nb),
            bid: AuxiliarySpinStringIndex(local % nb),
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
        let state = noci.spin_string_pair(index);
        self.index(SpinStringPair {
            parent: state.parent,
            aid: AuxiliarySpinStringIndex(state.aid.0),
            bid: AuxiliarySpinStringIndex(state.bid.0),
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
        connection: ConnectionState,
    ) -> AuxiliaryIndex {
        let state = noci.spin_string_pair(source);
        let retained = noci.parent_spin_strings(state.parent);
        let (oa, ob) =
            connection.child_occupations(retained.alpha(state.aid), retained.beta(state.bid));

        let components = &self.spin_strings[state.parent];
        self.index(SpinStringPair {
            parent: state.parent,
            aid: components.aids[&oa],
            bid: components.bids[&ob],
        })
    }

    /// Return one parent's auxiliary spin components.
    /// # Arguments:
    /// - `self`: Auxiliary determinant space.
    /// - `parent`: Parent orbital-frame index.
    /// # Returns:
    /// - `&ParentSpinStrings<AuxiliarySpinStringIndex>`: Auxiliary spin components.
    pub(crate) fn parent_spin_strings(
        &self,
        parent: usize,
    ) -> &ParentSpinStrings<AuxiliarySpinStringIndex> {
        &self.spin_strings[parent]
    }
}

/// Enumerate all physical one-spin occupations connected by a single or same-spin double.
/// # Arguments:
/// - `state`: Source component with rank-to-orbital lookup tables.
/// - `out`: Accumulator for occupations subsequently sorted and deduplicated.
/// # Returns
/// - `()`: Appends reachable occupations to `out`.
fn reachable_spin_occupations(
    state: &SpinStringState,
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
