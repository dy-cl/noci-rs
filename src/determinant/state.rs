// determinant/state.rs

// Standard library imports.
use std::collections::HashMap;

// Crate-root imports.
use crate::basis::{excitation_between, excitation_phase_bits};
use crate::{ExcitationSpin, ReducedOneSpinState};

/// Compact index of a spin component in one parent determinant.
pub(crate) trait SpinDeterminantIndex: Copy + Eq {
    /// Construct an index from its parent-local position.
    /// # Arguments:
    /// - `index`: Parent-local component position.
    /// # Returns:
    /// - `Self`: Typed component index.
    fn from_usize(index: usize) -> Self;

    /// Return the parent-local position.
    /// # Arguments:
    /// - `self`: Typed component index.
    /// # Returns:
    /// - `usize`: Parent-local component position.
    fn as_usize(self) -> usize;
}

/// Compact identity of one determinant in a two-spin space.
pub(crate) trait DeterminantState: Copy {
    type SpinIndex: SpinDeterminantIndex;

    /// Return the orbital-frame parent.
    /// # Arguments:
    /// - `self`: Determinant identity.
    /// # Returns:
    /// - `usize`: Parent frame index.
    fn parent(&self) -> usize;

    /// Return the parent-local alpha component.
    /// # Arguments:
    /// - `self`: Determinant identity.
    /// # Returns:
    /// - `Self::SpinIndex`: Alpha component index.
    fn aid(&self) -> Self::SpinIndex;

    /// Return the parent-local beta component.
    /// # Arguments:
    /// - `self`: Determinant identity.
    /// # Returns:
    /// - `Self::SpinIndex`: Beta component index.
    fn bid(&self) -> Self::SpinIndex;
}

/// Canonical spin determinant in one known parent orbital frame.
#[derive(Clone)]
pub(crate) struct SpinDeterminantState {
    /// Physical occupation bitstring in the parent orbital frame.
    pub(crate) occupation: u128,
    /// Excitation relative to the parent spin determinant.
    pub(crate) excitation: ExcitationSpin,
    /// Phase plus fixed-rank cache used by numerical kernels.
    pub(crate) reduced: ReducedOneSpinState,
    /// Physical orbital labels indexed by occupied rank.
    pub(crate) occupied: Vec<u8>,
    /// Physical orbital labels indexed by virtual rank.
    pub(crate) virtuals: Vec<u8>,
}

impl SpinDeterminantState {
    /// Derive parent-relative excitation, phase, cache and rank lookup tables from occupations.
    /// # Arguments:
    /// - `parent_occupation`: Occupation of the parent spin reference.
    /// - `occupation`: Physical occupation of this spin component.
    /// - `norb`: Number of orbitals in the parent spin frame.
    /// # Returns:
    /// - `Self`: Canonical parent-local spin component.
    pub(crate) fn new(
        parent_occupation: u128,
        occupation: u128,
        norb: usize,
    ) -> Self {
        let (holes, parts) = excitation_between(parent_occupation, occupation);
        let excitation = ExcitationSpin { holes, parts };
        let phase = excitation_phase_bits(parent_occupation, holes, parts);
        let reduced = ReducedOneSpinState::new(phase, excitation.cache());

        let noccupied = occupation.count_ones() as usize;
        let mut occupied = Vec::with_capacity(noccupied);
        let mut virtuals = Vec::with_capacity(norb - noccupied);
        for orbital in 0..norb {
            if occupation & (1u128 << orbital) == 0 {
                virtuals.push(orbital as u8);
            } else {
                occupied.push(orbital as u8);
            }
        }

        Self {
            occupation,
            excitation,
            reduced,
            occupied,
            virtuals,
        }
    }
}

/// Unique alpha and beta components in one parent and one determinant space.
#[derive(Clone)]
pub(crate) struct ParentComponents<I: SpinDeterminantIndex> {
    /// Canonical alpha components indexed by typed spin ID.
    pub(crate) alpha: Vec<SpinDeterminantState>,
    /// Canonical beta components indexed by typed spin ID.
    pub(crate) beta: Vec<SpinDeterminantState>,
    /// Alpha occupation-to-component-ID map.
    pub(crate) aids: HashMap<u128, I>,
    /// Beta occupation-to-component-ID map.
    pub(crate) bids: HashMap<u128, I>,
}

impl<I: SpinDeterminantIndex> ParentComponents<I> {
    /// Construct empty parent-local component storage.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Self`: Empty alpha and beta component tables.
    pub(crate) fn new() -> Self {
        Self {
            alpha: Vec::new(),
            beta: Vec::new(),
            aids: HashMap::new(),
            bids: HashMap::new(),
        }
    }

    /// Return one alpha spin component by compact ID.
    /// # Arguments:
    /// - `self`: Parent-local component storage.
    /// - `aid`: Alpha component ID.
    /// # Returns:
    /// - `&SpinDeterminantState`: Canonical alpha component.
    pub(crate) fn alpha(
        &self,
        aid: I,
    ) -> &SpinDeterminantState {
        &self.alpha[aid.as_usize()]
    }

    /// Return one beta spin component by compact ID.
    /// # Arguments:
    /// - `self`: Parent-local component storage.
    /// - `bid`: Beta component ID.
    /// # Returns:
    /// - `&SpinDeterminantState`: Canonical beta component.
    pub(crate) fn beta(
        &self,
        bid: I,
    ) -> &SpinDeterminantState {
        &self.beta[bid.as_usize()]
    }

    /// Return the number of unique alpha components.
    /// # Arguments:
    /// - `self`: Parent-local component storage.
    /// # Returns:
    /// - `usize`: Number of alpha components.
    pub(crate) fn na(&self) -> usize {
        self.alpha.len()
    }

    /// Return the number of unique beta components.
    /// # Arguments:
    /// - `self`: Parent-local component storage.
    /// # Returns:
    /// - `usize`: Number of beta components.
    pub(crate) fn nb(&self) -> usize {
        self.beta.len()
    }

    /// Intern an alpha component by physical occupation within this parent.
    /// # Arguments:
    /// - `self`: Parent-local component storage.
    /// - `state`: Canonical alpha component to intern.
    /// # Returns:
    /// - `I`: Existing or newly assigned alpha component ID.
    pub(crate) fn intern_alpha(
        &mut self,
        state: SpinDeterminantState,
    ) -> I {
        if let Some(&id) = self.aids.get(&state.occupation) {
            return id;
        }

        let id = I::from_usize(self.alpha.len());
        self.aids.insert(state.occupation, id);
        self.alpha.push(state);
        id
    }

    /// Intern a beta component by physical occupation within this parent.
    /// # Arguments:
    /// - `self`: Parent-local component storage.
    /// - `state`: Canonical beta component to intern.
    /// # Returns:
    /// - `I`: Existing or newly assigned beta component ID.
    pub(crate) fn intern_beta(
        &mut self,
        state: SpinDeterminantState,
    ) -> I {
        if let Some(&id) = self.bids.get(&state.occupation) {
            return id;
        }

        let id = I::from_usize(self.beta.len());
        self.bids.insert(state.occupation, id);
        self.beta.push(state);
        id
    }
}
