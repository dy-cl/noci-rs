// determinant/state.rs

// Standard library imports.
use std::collections::HashMap;

// Crate-root imports.
use crate::basis::{excitation_between, excitation_phase_bits};
use crate::{ExcitationSpin, ExcitationSpinCache};

/// Compact index of a spin string in one parent determinant.
pub(crate) trait SpinStringIndex: Copy + Eq {
    /// Construct an index from its parent-local position.
    /// # Arguments:
    /// - `index`: Parent-local spin-string position.
    /// # Returns:
    /// - `Self`: Typed spin-string index.
    fn from_usize(index: usize) -> Self;

    /// Return the parent-local position.
    /// # Arguments:
    /// - `self`: Typed spin-string index.
    /// # Returns:
    /// - `usize`: Parent-local spin-string position.
    fn as_usize(self) -> usize;
}

/// Define a typed spin-string handle, a parent-local position implementing `SpinStringIndex`.
macro_rules! determinant_index {
    ($name:ident, $trait_name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub(crate) struct $name(
            /// Parent-local spin-string position.
            pub(crate) usize,
        );

        impl $trait_name for $name {
            /// Construct a compact index from its position.
            /// # Arguments:
            /// - `index`: Parent-local position.
            /// # Returns:
            /// - `Self`: Typed spin-string index.
            #[inline(always)]
            fn from_usize(index: usize) -> Self {
                Self(index)
            }

            /// Return the position encoded by this index.
            /// # Arguments:
            /// - `self`: Typed spin-string index.
            /// # Returns:
            /// - `usize`: Parent-local position.
            #[inline(always)]
            fn as_usize(self) -> usize {
                self.0
            }
        }
    };
}

pub(super) use determinant_index;

/// A determinant's spin-string handles in one space: its parent and the positions of its alpha
/// and beta spin strings among that parent's spin strings.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SpinStringPair<I: SpinStringIndex> {
    /// Orbital-frame parent of the determinant.
    pub(crate) parent: usize,
    /// Position of the alpha spin string among the parent's alpha spin strings.
    pub(crate) aid: I,
    /// Position of the beta spin string among the parent's beta spin strings.
    pub(crate) bid: I,
}

/// One spin string relative to a reference spin string of the same frame: its hole and particle
/// masks, its fermionic phase and its fixed-rank cache. It carries no identity, so every spin string
/// with the same excitation has the same data.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpinStringData {
    /// Holes and particles relative to the reference spin string.
    pub(crate) excitation: ExcitationSpin,
    /// Fermionic phase relative to the reference spin string.
    pub(crate) phase: f64,
    /// Excitation rank and fixed-rank orbital labels.
    pub(crate) cache: ExcitationSpinCache,
}

impl SpinStringData {
    /// Derive the excitation, phase and fixed-rank cache of a spin string relative to a reference
    /// spin string of the same frame.
    /// # Arguments:
    /// - `reference`: Occupation of the reference spin string.
    /// - `occupation`: Occupation of the spin string.
    /// # Returns:
    /// - `Self`: Spin string data relative to the reference.
    #[inline(always)]
    pub(crate) fn between(
        reference: u128,
        occupation: u128,
    ) -> Self {
        let (holes, parts) = excitation_between(reference, occupation);
        let excitation = ExcitationSpin { holes, parts };
        Self {
            excitation,
            phase: excitation_phase_bits(reference, holes, parts),
            cache: excitation.cache(),
        }
    }
}

/// One spin string of a parent, stored once per space: its occupation, its data relative to the
/// parent, and the orbital of every occupied and virtual rank.
#[derive(Clone)]
pub(crate) struct SpinStringState {
    /// Occupation bitstring in the parent orbital frame.
    pub(crate) occupation: u128,
    /// Excitation, phase and fixed-rank cache relative to the parent spin string.
    pub(crate) data: SpinStringData,
    /// Physical orbital labels indexed by occupied rank.
    pub(crate) occupied: Vec<u8>,
    /// Physical orbital labels indexed by virtual rank.
    pub(crate) virtuals: Vec<u8>,
}

impl SpinStringState {
    /// Derive parent-relative excitation, phase, cache and rank lookup tables from occupations.
    /// # Arguments:
    /// - `parent_occupation`: Occupation of the parent spin reference.
    /// - `occupation`: Physical occupation of this spin string.
    /// - `norb`: Number of orbitals in the parent spin frame.
    /// # Returns:
    /// - `Self`: Canonical parent-local spin string.
    pub(crate) fn new(
        parent_occupation: u128,
        occupation: u128,
        norb: usize,
    ) -> Self {
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
            data: SpinStringData::between(parent_occupation, occupation),
            occupied,
            virtuals,
        }
    }
}

/// Unique alpha and beta spin strings in one parent and one determinant space.
#[derive(Clone)]
pub(crate) struct ParentSpinStrings<I: SpinStringIndex> {
    /// Canonical alpha spin strings indexed by typed spin ID.
    pub(crate) alpha: Vec<SpinStringState>,
    /// Canonical beta spin strings indexed by typed spin ID.
    pub(crate) beta: Vec<SpinStringState>,
    /// Alpha occupation-to-spin-string-ID map.
    pub(crate) aids: HashMap<u128, I>,
    /// Beta occupation-to-spin-string-ID map.
    pub(crate) bids: HashMap<u128, I>,
}

impl<I: SpinStringIndex> ParentSpinStrings<I> {
    /// Construct empty parent-local spin-string storage.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Self`: Empty alpha and beta spin-string tables.
    pub(crate) fn new() -> Self {
        Self {
            alpha: Vec::new(),
            beta: Vec::new(),
            aids: HashMap::new(),
            bids: HashMap::new(),
        }
    }

    /// Return one alpha spin string by compact ID.
    /// # Arguments:
    /// - `self`: Parent-local spin-string storage.
    /// - `aid`: Alpha spin-string ID.
    /// # Returns:
    /// - `&SpinStringState`: Canonical alpha spin string.
    pub(crate) fn alpha(
        &self,
        aid: I,
    ) -> &SpinStringState {
        &self.alpha[aid.as_usize()]
    }

    /// Return one beta spin string by compact ID.
    /// # Arguments:
    /// - `self`: Parent-local spin-string storage.
    /// - `bid`: Beta spin-string ID.
    /// # Returns:
    /// - `&SpinStringState`: Canonical beta spin string.
    pub(crate) fn beta(
        &self,
        bid: I,
    ) -> &SpinStringState {
        &self.beta[bid.as_usize()]
    }

    /// Return the number of unique alpha spin strings.
    /// # Arguments:
    /// - `self`: Parent-local spin-string storage.
    /// # Returns:
    /// - `usize`: Number of alpha spin strings.
    pub(crate) fn na(&self) -> usize {
        self.alpha.len()
    }

    /// Return the number of unique beta spin strings.
    /// # Arguments:
    /// - `self`: Parent-local spin-string storage.
    /// # Returns:
    /// - `usize`: Number of beta spin strings.
    pub(crate) fn nb(&self) -> usize {
        self.beta.len()
    }

    /// Intern an alpha spin string by physical occupation within this parent.
    /// # Arguments:
    /// - `self`: Parent-local spin-string storage.
    /// - `state`: Canonical alpha spin string to intern.
    /// # Returns:
    /// - `I`: Existing or newly assigned alpha spin-string ID.
    pub(crate) fn intern_alpha(
        &mut self,
        state: SpinStringState,
    ) -> I {
        if let Some(&id) = self.aids.get(&state.occupation) {
            return id;
        }

        let id = I::from_usize(self.alpha.len());
        self.aids.insert(state.occupation, id);
        self.alpha.push(state);
        id
    }

    /// Intern a beta spin string by physical occupation within this parent.
    /// # Arguments:
    /// - `self`: Parent-local spin-string storage.
    /// - `state`: Canonical beta spin string to intern.
    /// # Returns:
    /// - `I`: Existing or newly assigned beta spin-string ID.
    pub(crate) fn intern_beta(
        &mut self,
        state: SpinStringState,
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
