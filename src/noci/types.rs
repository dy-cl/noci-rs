// noci/types.rs
use ndarray::{Array2, Array4};

use crate::{AoData, SCFState};
use crate::input::Input;
use crate::nonorthogonalwicks::WicksView;

/// Shared data required for NOCI matrix-element evaluation.
pub struct NOCIData<'a> {
    /// AO-basis integrals and other system-wide data.
    pub ao: &'a AoData,
    /// List of the current determinants in the basis.
    pub basis: &'a [SCFState],
    /// User input controlling matrix-element evaluation and optional Wick's usage.
    pub input: &'a Input,
    /// Numerical tolerance used to decide when quantities are treated as zero.
    pub tol: f64,
    /// Optional precomputed Wick's intermediates for non-orthogonal evaluation.
    pub wicks: Option<&'a WicksView>,
    /// MO-basis Hamiltonian caches for orthogonal-parent matrix elements.
    pub mocache: Option<&'a [MOCache]>,
}

impl<'a> NOCIData<'a> {
    /// Construct the shared data required for NOCI matrix-element evaluation.
    /// # Arguments:
    /// - `ao`: Contains AO integrals and other system data.
    /// - `basis`: Determinant basis with respect to which matrix elements are being evaluated.
    /// - `input`: User defined input options.
    /// - `tol`: Tolerance for a number being zero.
    /// - `wicks`: View to the intermediates required for non-orthogonal Wick's theorem.
    /// # Returns:
    /// - `NOCIData<'a>`: Shared data for NOCI matrix-element evaluation.
    pub fn new(ao: &'a AoData, basis: &'a[SCFState], input: &'a Input, tol: f64, wicks: Option<&'a WicksView>) -> Self {
        Self {ao, basis, input, tol, wicks, mocache: None}
    }

    /// Attach the MO-basis Hamiltonian caches required for Hamiltonian matrix elements.
    /// # Arguments:
    /// - `mocache`: MO-basis one and two-electron integral caches.
    /// # Returns:
    /// - `NOCIData<'a>`: Shared data for Hamiltonian and overlap matrix-element evaluation.
    pub fn withmocache(mut self, mocache: &'a [MOCache]) -> Self {
        self.mocache = Some(mocache);
        self
    }
}

pub(crate) struct FockData<'a> {
    /// Optional MO-basis Fock caches for orthogonal-parent Fock matrix elements.
    pub(crate) fock_mocache: &'a [FockMOCache],
    /// Optional spin-alpha Fock matrix in the AO basis.
    pub(crate) fa: &'a Array2<f64>,
    /// Optional spin-beta Fock matrix in the AO basis.
    pub(crate) fb: &'a Array2<f64>,
}

impl<'a> FockData<'a> {
    /// Construct the Fock-specific data required for evaluation of Fock matrix elements.
    /// # Arguments:
    /// - `fock_mocache`: MO-basis Fock integral caches.
    /// - `fa`: Spin-alpha Fock matrix in the AO basis.
    /// - `fb`: Spin-beta Fock matrix in the AO basis.
    /// # Returns:
    /// - `FockData<'a>`: Fock-specific data for NOCI matrix-element evaluation.
    pub(crate) fn new(fock_mocache: &'a [FockMOCache], fa: &'a Array2<f64>, fb: &'a Array2<f64>) -> Self {
        Self {fock_mocache, fa, fb}
    }
}

/// Stores the pair of determinants whose matrix element is being evaluated.
pub(crate) struct DetPair<'a> {
    /// Left determinant in the matrix element.
    pub(crate) ldet: &'a SCFState,
    /// Right determinant in the matrix element.
    pub(crate) gdet: &'a SCFState,
}

impl<'a> DetPair<'a> {
    /// Construct the pair of determinants whose matrix element is to be evaluated.
    /// # Arguments:
    /// - `ldet`: Left determinant in the matrix element.
    /// - `gdet`: Right determinant in the matrix element.
    /// # Returns:
    /// - `DetPair<'a>`: Pair of determinants to be passed to matrix-element evaluation routines.
    pub(crate) fn new(ldet: &'a SCFState, gdet: &'a SCFState) -> Self {
        Self {ldet, gdet}
    }
}

// Trait which defines how returned determinant-pair quatity should be scattered into matrices.
// Used such that we can have generic scatter functions which return 1 or 2 matrices.
pub(in crate::noci) trait ScatterValue: Sized {
    type Output;
    /// Construct zero initialised output. 
    /// # Arguments:
    /// - `nl`: Length of determinant set 1.
    /// - `nr`: Length of determinant set 2.
    fn zeros(nl: usize, nr: usize) -> Self::Output;
    /// Write a value into the output at indices i, j.
    /// # Arguments:
    /// - `out`: Output container to write into.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: Determinant-pair value to scatter.
    fn write(out: &mut Self::Output, i: usize, j: usize, val: Self);
}

impl ScatterValue for f64 {
    type Output = Array2<f64>;
    /// Construct zero initialised output. 
    /// # Arguments:
    /// - `nl`: Length of determinant set 1.
    /// - `nr`: Length of determinant set 2.
    fn zeros(nl: usize, nr: usize) -> Self::Output {
        Array2::<f64>::zeros((nl, nr))
    }
    /// Write scalar value into matrix position (i, j).
    /// # Arguments:
    /// - `out`: Output matrix.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: Matrix element value.
    fn write(out: &mut Self::Output, i: usize, j: usize, val: Self) {
        out[(i, j)] = val;
    }
}

impl ScatterValue for (f64, f64) {
    type Output = (Array2<f64>, Array2<f64>);

    /// Construct zero initialised output. 
    /// # Arguments:
    /// - `nl`: Length of determinant set 1.
    /// - `nr`: Length of determinant set 2.
    fn zeros(nl: usize, nr: usize) -> Self::Output {
        (Array2::<f64>::zeros((nl, nr)), Array2::<f64>::zeros((nl, nr)))
    }

    /// Write scalar value into matrix position (i, j) in both matrices.
    /// # Arguments:
    /// - `out`: `Array2<f64>`), output matrices.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: F64), matrix element values.
    fn write(out: &mut Self::Output, i: usize, j: usize, val: Self) {
        out.0[(i, j)] = val.0;
        out.1[(i, j)] = val.1;
    }
}

// Storage of quantities required to compute matrix elements between determinant pairs in the naive fashion.
pub(in crate::noci) struct Pair {
    /// Full overlap matrix element for the determinant pair.
    pub(in crate::noci) s: f64,
    /// Product of the non-zero singular values of the occupied-overlap matrix.
    pub(in crate::noci) s_red: f64,
    /// Indices of singular values treated as zero.
    pub(in crate::noci) zeros: Vec<usize>,
    /// Weighted co-density matrix used when no singular values are zero.
    pub(in crate::noci) w: Option<Array2<f64>>,
    /// Co-density matrix associated with the first zero singular value.
    pub(in crate::noci) p_i: Option<Array2<f64>>,
    /// Co-density matrix associated with the second zero singular value.
    pub(in crate::noci) p_j: Option<Array2<f64>>,
    /// Overall phase from the SVD rotations of the occupied-overlap matrix.
    pub(in crate::noci) phase: f64,
}

pub struct MOCache {
    /// One-electron Hamiltonian in parent alpha MO basis.
    pub ha: Array2<f64>,
    /// One-electron Hamiltonian in parent beta MO basis.
    pub hb: Array2<f64>,
    /// Antisymmetrised same-spin ERIs in parent alpha MO basis.
    pub eri_aa_asym: Array4<f64>,
    /// Antisymmetrised same-spin ERIs in parent beta MO basis.
    pub eri_bb_asym: Array4<f64>,
    /// Coulomb different-spin ERIs in parent alpha/beta MO basis.
    pub eri_ab_coul: Array4<f64>,
}

pub struct FockMOCache {
    /// Spin-alpha Fock matrix in the parent alpha MO basis.
    pub fa: Array2<f64>,
    /// Spin-beta Fock matrix in the parent beta MO basis.
    pub fb: Array2<f64>,
}


