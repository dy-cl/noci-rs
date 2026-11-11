// determinant/excited.rs

// Crate-root imports.
use crate::StateScalar;

// Parent/sibling imports.
use super::parent::ParentDeterminant;
use super::state::SpinStringData;

/// The identity of a determinant: its parent orbital frame and its alpha and beta occupations in
/// that frame. It is the same whichever space holds the determinant, and it is the determinant of
/// every matrix-element request.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DeterminantState {
    /// Parent orbital frame of the determinant.
    pub(crate) parent: usize,
    /// Alpha occupation bitstring in the parent frame.
    pub(crate) alpha: u128,
    /// Beta occupation bitstring in the parent frame.
    pub(crate) beta: u128,
}

/// A determinant relative to a reference determinant of the same frame, an alpha spin string times
/// a beta spin string: for a determinant's own data the reference is its parent, and for a
/// connection it is the source. It carries no identity.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeterminantData {
    /// Alpha spin string relative to the reference.
    pub(crate) alpha: SpinStringData,
    /// Beta spin string relative to the reference.
    pub(crate) beta: SpinStringData,
}

impl DeterminantState {
    /// Derive the data of the determinant relative to its parent, `|D\rangle = \phi\,\hat\tau|P\rangle`.
    /// # Arguments:
    /// - `self`: Determinant identity.
    /// - `parents`: Parent determinants indexed by parent frame.
    /// # Returns:
    /// - `DeterminantData`: Excitation, phase and fixed-rank cache of each spin string.
    pub(crate) fn data<T: StateScalar>(
        &self,
        parents: &[ParentDeterminant<T>],
    ) -> DeterminantData {
        let parent = &parents[self.parent];
        DeterminantData::between((parent.oa, parent.ob), (self.alpha, self.beta))
    }
}

impl DeterminantData {
    /// Derive the data of a determinant relative to a reference determinant of the same frame.
    /// # Arguments:
    /// - `reference`: Alpha and beta occupations of the reference determinant.
    /// - `occupations`: Alpha and beta occupations of the determinant.
    /// # Returns:
    /// - `Self`: Data of both spin strings relative to the reference.
    #[inline(always)]
    pub(crate) fn between(
        reference: (u128, u128),
        occupations: (u128, u128),
    ) -> Self {
        Self {
            alpha: SpinStringData::between(reference.0, occupations.0),
            beta: SpinStringData::between(reference.1, occupations.1),
        }
    }

    /// Return the fermionic phase of the determinant, `p = p_\alpha p_\beta`.
    /// # Arguments:
    /// - `self`: Determinant data.
    /// # Returns:
    /// - `f64`: Product of the spin-string phases.
    #[inline(always)]
    pub(crate) fn phase(&self) -> f64 {
        self.alpha.phase * self.beta.phase
    }
}
