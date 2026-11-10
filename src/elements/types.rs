// elements/types.rs

// External crate imports.
use ndarray::{Array2, Array4};

// Crate-root imports.
use crate::determinant::{ExcitedDeterminant, ExcitedState, NOCIIndex, NOCISpace};
use crate::input::Input;
use crate::{AoData, NOCIScalar};

// Parent/sibling imports.
use super::nonorthogonalwicks::WicksView;

/// Shared data required for NOCI matrix-element evaluation.
pub struct NOCIData<'a, T: NOCIScalar> {
    /// AO-basis integrals and other system-wide data.
    pub ao: &'a AoData,
    /// Authoritative retained determinant space.
    pub(crate) space: &'a NOCISpace<T>,
    /// User input controlling matrix-element evaluation and optional Wick's usage.
    pub input: &'a Input,
    /// Numerical tolerance used to decide when quantities are treated as zero.
    pub tol: f64,
    /// Optional precomputed Wick's intermediates for non-orthogonal evaluation.
    pub wicks: Option<&'a WicksView<T>>,
    /// MO-basis Hamiltonian caches for orthogonal-parent matrix elements.
    pub mocache: Option<&'a [MOCache<T>]>,
}

impl<'a, T: NOCIScalar> NOCIData<'a, T> {
    /// Construct the shared data required for NOCI matrix-element evaluation.
    /// # Arguments:
    /// - `ao`: Contains AO integrals and other system data.
    /// - `space`: Authoritative retained determinant space.
    /// - `input`: User defined input options.
    /// - `tol`: Tolerance for a number being zero.
    /// - `wicks`: View to the intermediates required for non-orthogonal Wick's theorem.
    /// # Returns:
    /// - `NOCIData<'a, T>`: Shared data for NOCI matrix-element evaluation.
    pub fn new(
        ao: &'a AoData,
        space: &'a NOCISpace<T>,
        input: &'a Input,
        tol: f64,
        wicks: Option<&'a WicksView<T>>,
    ) -> Self {
        Self {
            ao,
            space,
            input,
            tol,
            wicks,
            mocache: None,
        }
    }

    /// Attach the MO-basis Hamiltonian caches required for orthogonal Hamiltonian matrix elements.
    /// # Arguments:
    /// - `mocache`: MO-basis one and two-electron integral caches.
    /// # Returns:
    /// - `NOCIData<'a, T>`: Shared data for Hamiltonian and overlap matrix-element evaluation.
    pub fn withmocache(
        mut self,
        mocache: &'a [MOCache<T>],
    ) -> Self {
        self.mocache = Some(mocache);
        self
    }
}

/// Fock-specific data required for scalar-generic NOCI matrix-element evaluation.
pub(crate) struct FockData<'a, T: NOCIScalar> {
    /// Optional MO-basis Fock caches for orthogonal-parent Fock matrix elements.
    pub(crate) fock_mocache: &'a [FockMOCache<T>],
    /// Spin-alpha Fock matrix in the AO basis.
    pub(crate) fa: &'a Array2<T>,
    /// Spin-beta Fock matrix in the AO basis.
    pub(crate) fb: &'a Array2<T>,
}

impl<'a, T: NOCIScalar> FockData<'a, T> {
    /// Construct the Fock-specific data required for evaluation of Fock matrix elements.
    /// # Arguments:
    /// - `fock_mocache`: MO-basis Fock integral caches.
    /// - `fa`: Spin-alpha Fock matrix in the AO basis.
    /// - `fb`: Spin-beta Fock matrix in the AO basis.
    /// # Returns:
    /// - `FockData<'a, T>`: Fock-specific data for NOCI matrix-element evaluation.
    pub(crate) fn new(
        fock_mocache: &'a [FockMOCache<T>],
        fa: &'a Array2<T>,
        fb: &'a Array2<T>,
    ) -> Self {
        Self {
            fock_mocache,
            fa,
            fb,
        }
    }
}

/// Stores the pair of determinants whose matrix element is being evaluated.
#[derive(Clone, Copy)]
pub(crate) struct DetPair<D = NOCIIndex> {
    /// Left determinant in the matrix element.
    pub(crate) ldet: D,
    /// Right determinant in the matrix element.
    pub(crate) gdet: D,
}

impl<D> DetPair<D> {
    /// Construct the pair of determinants whose matrix element is to be evaluated.
    /// # Arguments:
    /// - `ldet`: Left determinant in the matrix element.
    /// - `gdet`: Right determinant in the matrix element.
    /// # Returns:
    /// - `DetPair<D>`: Pair of determinants to be passed to matrix-element routines.
    pub(crate) fn new(
        ldet: D,
        gdet: D,
    ) -> Self {
        Self { ldet, gdet }
    }
}

/// A determinant that can enter a matrix element. The dispatch needs only its parent frame and
/// occupations, and prepares the parent-relative excitation, spin phases and fixed-rank payload
/// only for pairs evaluated by the nonorthogonal engines.
pub(crate) trait PairDeterminant<T: NOCIScalar>: Copy {
    /// Return the parent orbital frame of the determinant.
    /// # Arguments:
    /// - `self`: Determinant.
    /// - `space`: Determinant space holding the parent frames.
    /// # Returns:
    /// - `usize`: Parent frame index.
    fn parent(
        self,
        space: &NOCISpace<T>,
    ) -> usize;

    /// Return the alpha and beta occupations of the determinant in its parent frame.
    /// # Arguments:
    /// - `self`: Determinant.
    /// - `space`: Determinant space holding the parent frames.
    /// # Returns:
    /// - `(u128, u128)`: Alpha and beta occupation bitstrings.
    fn occupations(
        self,
        space: &NOCISpace<T>,
    ) -> (u128, u128);

    /// Return the determinant in its prepared form.
    /// # Arguments:
    /// - `self`: Determinant.
    /// - `space`: Determinant space holding the parent frames.
    /// # Returns:
    /// - `ExcitedState`: Parent-relative excitation, spin phases and fixed-rank payload.
    fn state(
        self,
        space: &NOCISpace<T>,
    ) -> ExcitedState;
}

impl<T: NOCIScalar> PairDeterminant<T> for NOCIIndex {
    /// Return the parent frame of a retained determinant.
    /// # Arguments:
    /// - `self`: Retained determinant index.
    /// - `space`: Determinant space containing the determinant.
    /// # Returns:
    /// - `usize`: Parent frame index.
    fn parent(
        self,
        space: &NOCISpace<T>,
    ) -> usize {
        space.state(self).parent
    }

    /// Return the occupations of a retained determinant.
    /// # Arguments:
    /// - `self`: Retained determinant index.
    /// - `space`: Determinant space containing the determinant.
    /// # Returns:
    /// - `(u128, u128)`: Alpha and beta occupation bitstrings.
    fn occupations(
        self,
        space: &NOCISpace<T>,
    ) -> (u128, u128) {
        space.occupations(self)
    }

    /// Assemble the prepared form of a retained determinant from the stored payload.
    /// # Arguments:
    /// - `self`: Retained determinant index.
    /// - `space`: Determinant space containing the determinant.
    /// # Returns:
    /// - `ExcitedState`: Stored excitation, spin phases and fixed-rank payload.
    fn state(
        self,
        space: &NOCISpace<T>,
    ) -> ExcitedState {
        space.excited_state(self)
    }
}

impl<T: NOCIScalar> PairDeterminant<T> for &ExcitedState {
    /// Return the parent frame of a prepared determinant.
    /// # Arguments:
    /// - `self`: Prepared determinant.
    /// - `space`: Unused, as the determinant carries its parent.
    /// # Returns:
    /// - `usize`: Parent frame index.
    fn parent(
        self,
        _space: &NOCISpace<T>,
    ) -> usize {
        self.det.parent
    }

    /// Return the occupations of a prepared determinant.
    /// # Arguments:
    /// - `self`: Prepared determinant.
    /// - `space`: Unused, as the determinant carries its occupations.
    /// # Returns:
    /// - `(u128, u128)`: Alpha and beta occupation bitstrings.
    fn occupations(
        self,
        _space: &NOCISpace<T>,
    ) -> (u128, u128) {
        (self.det.alpha, self.det.beta)
    }

    /// Return a copy of the prepared determinant.
    /// # Arguments:
    /// - `self`: Prepared determinant.
    /// - `space`: Unused, as the determinant is already prepared.
    /// # Returns:
    /// - `ExcitedState`: The prepared determinant.
    fn state(
        self,
        _space: &NOCISpace<T>,
    ) -> ExcitedState {
        *self
    }
}

impl<T: NOCIScalar> PairDeterminant<T> for ExcitedDeterminant {
    /// Return the parent frame of an occupation-identified determinant.
    /// # Arguments:
    /// - `self`: Occupation-identified determinant.
    /// - `space`: Unused, as the determinant carries its parent.
    /// # Returns:
    /// - `usize`: Parent frame index.
    fn parent(
        self,
        _space: &NOCISpace<T>,
    ) -> usize {
        self.parent
    }

    /// Return the occupations of an occupation-identified determinant.
    /// # Arguments:
    /// - `self`: Occupation-identified determinant.
    /// - `space`: Unused, as the determinant carries its occupations.
    /// # Returns:
    /// - `(u128, u128)`: Alpha and beta occupation bitstrings.
    fn occupations(
        self,
        _space: &NOCISpace<T>,
    ) -> (u128, u128) {
        (self.alpha, self.beta)
    }

    /// Prepare an occupation-identified determinant relative to its parent.
    /// # Arguments:
    /// - `self`: Occupation-identified determinant.
    /// - `space`: Determinant space holding the parent frames.
    /// # Returns:
    /// - `ExcitedState`: Derived excitation, spin phases and fixed-rank payload.
    fn state(
        self,
        space: &NOCISpace<T>,
    ) -> ExcitedState {
        ExcitedDeterminant::state(&self, space.parents())
    }
}

/// MO-basis caches for orthogonal-parent matrix elements.
pub struct MOCache<T: NOCIScalar> {
    /// One-electron Hamiltonian in parent alpha MO basis.
    pub ha: Array2<T>,
    /// One-electron Hamiltonian in parent beta MO basis.
    pub hb: Array2<T>,
    /// Antisymmetrised same-spin ERIs in parent alpha MO basis.
    pub eri_aa_asym: Array4<T>,
    /// Antisymmetrised same-spin ERIs in parent beta MO basis.
    pub eri_bb_asym: Array4<T>,
    /// Coulomb different-spin ERIs in parent alpha/beta MO basis.
    pub eri_ab_coul: Array4<T>,
    /// Whether the parent is suitable for ordinary orthogonal Slater-Condon rules.
    pub orthogonal_slater_condon: bool,
}

/// MO-basis Fock caches for orthogonal-parent matrix elements.
pub struct FockMOCache<T: NOCIScalar> {
    /// Spin-alpha Fock matrix in the parent alpha MO basis.
    pub fa: Array2<T>,
    /// Spin-beta Fock matrix in the parent beta MO basis.
    pub fb: Array2<T>,
    /// Whether the parent is suitable for ordinary orthogonal Slater-Condon rules.
    pub orthogonal_slater_condon: bool,
}
