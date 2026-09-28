// elements/types.rs

// External crate imports.
use ndarray::{Array2, Array4};

// Crate-root imports.
use crate::determinant::{NOCIIndex, NOCISpace};
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
pub(crate) struct DetPair {
    /// Left determinant in the matrix element.
    pub(crate) ldet: NOCIIndex,
    /// Right determinant in the matrix element.
    pub(crate) gdet: NOCIIndex,
}

impl DetPair {
    /// Construct the pair of determinants whose matrix element is to be evaluated.
    /// # Arguments:
    /// - `ldet`: Left determinant in the matrix element.
    /// - `gdet`: Right determinant in the matrix element.
    /// # Returns:
    /// - `DetPair<'a, T>`: Pair of determinants to be passed to matrix-element routines.
    pub(crate) fn new(
        ldet: NOCIIndex,
        gdet: NOCIIndex,
    ) -> Self {
        Self { ldet, gdet }
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
