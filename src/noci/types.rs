// noci/types.rs

use ndarray::{Array2, Array4};
use ndarray_linalg::Scalar;
use num_complex::Complex64;

use crate::input::Input;
use crate::maths::ERIScalar;
use crate::nonorthogonalwicks::WicksView;
use crate::{AoData, DetState, StateScalar};

use crate::maths::{
    einsum_ba_ab_complex, einsum_ba_ab_complex_real, einsum_ba_ab_real, einsum_ba_abcd_cd_complex,
    einsum_ba_abcd_cd_complex_real, einsum_ba_abcd_cd_real,
};

/// Scalar type accepted by generic NOCI matrix-element code.
pub trait NOCIScalar: StateScalar + From<f64> + Scalar<Real = f64> + ERIScalar {
    /// Construct a purely imaginary scalar.
    fn from_imag(x: f64) -> Self;

    /// Calculate Einstein summation of scalar matrices `g` and `h` as \sum_{a,b} g_{b,a} h_{a,b}.
    /// Assumes `g` and `h` are of identical shape.
    /// # Arguments
    /// - `g`: Scalar matrix 1.
    /// - `h`: Scalar matrix 2.
    /// # Returns
    /// - `Self`: Contracted scalar.
    fn einsum_ba_ab(
        g: &Array2<Self>,
        h: &Array2<Self>,
    ) -> Self;

    /// Calculate Einstein summation of scalar matrix `g` and real matrix `h` as \sum_{a,b} g_{b,a} h_{a,b}.
    /// Assumes `g` and `h` are of identical shape.
    /// # Arguments
    /// - `g`: Scalar matrix 1.
    /// - `h`: Real matrix 2.
    /// # Returns
    /// - `Self`: Contracted scalar.
    fn einsum_ba_ab_realop(
        g: &Array2<Self>,
        h: &Array2<f64>,
    ) -> Self;

    /// Calculate Einstein summation of scalar matrices `g` and `h` and scalar 4D tensor `t` as
    /// \sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.
    /// Assumes `g`, `h` and `t` all have axes of equal length.
    /// # Arguments
    /// - `g`: Scalar matrix 1.
    /// - `t`: Scalar 4D tensor.
    /// - `h`: Scalar matrix 2.
    /// # Returns
    /// - `Self`: Contracted scalar.
    fn einsum_ba_abcd_cd(
        g: &Array2<Self>,
        t: &Array4<Self>,
        h: &Array2<Self>,
    ) -> Self;

    /// Calculate Einstein summation of scalar matrices `g` and `h` and real 4D tensor `t` as
    /// \sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.
    /// Assumes `g`, `h` and `t` all have axes of equal length.
    /// # Arguments
    /// - `g`: Scalar matrix 1.
    /// - `t`: Real 4D tensor.
    /// - `h`: Scalar matrix 2.
    /// # Returns
    /// - `Self`: Contracted scalar.
    fn einsum_ba_abcd_cd_realop(
        g: &Array2<Self>,
        t: &Array4<f64>,
        h: &Array2<Self>,
    ) -> Self;
}

impl NOCIScalar for f64 {
    fn from_imag(x: f64) -> Self {
        if x == 0.0 {
            0.0
        } else {
            panic!("non-zero SNOCI imaginary shift requires complex arithmetic")
        }
    }

    /// Calculate Einstein summation of real matrices `g` and `h` as \sum_{a,b} g_{b,a} h_{a,b}.
    /// Assumes `g` and `h` are of identical shape.
    /// # Arguments
    /// - `g`: Real matrix 1.
    /// - `h`: Real matrix 2.
    /// # Returns
    /// - `f64`: Contracted scalar.
    fn einsum_ba_ab(
        g: &Array2<Self>,
        h: &Array2<Self>,
    ) -> Self {
        einsum_ba_ab_real(g, h)
    }

    /// Calculate Einstein summation of real matrices `g` and `h` as \sum_{a,b} g_{b,a} h_{a,b}.
    /// Assumes `g` and `h` are of identical shape.
    /// # Arguments
    /// - `g`: Real matrix 1.
    /// - `h`: Real matrix 2.
    /// # Returns
    /// - `f64`: Contracted scalar.
    fn einsum_ba_ab_realop(
        g: &Array2<Self>,
        h: &Array2<f64>,
    ) -> Self {
        einsum_ba_ab_real(g, h)
    }

    /// Calculate Einstein summation of real matrices `g` and `h` and real 4D tensor `t` as
    /// \sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.
    /// Assumes `g`, `h` and `t` all have axes of equal length.
    /// # Arguments
    /// - `g`: Real matrix 1.
    /// - `t`: Real 4D tensor.
    /// - `h`: Real matrix 2.
    /// # Returns
    /// - `f64`: Contracted scalar.
    fn einsum_ba_abcd_cd(
        g: &Array2<Self>,
        t: &Array4<Self>,
        h: &Array2<Self>,
    ) -> Self {
        einsum_ba_abcd_cd_real(g, t, h)
    }

    /// Calculate Einstein summation of real matrices `g` and `h` and real 4D tensor `t` as
    /// \sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.
    /// Assumes `g`, `h` and `t` all have axes of equal length.
    /// # Arguments
    /// - `g`: Real matrix 1.
    /// - `t`: Real 4D tensor.
    /// - `h`: Real matrix 2.
    /// # Returns
    /// - `f64`: Contracted scalar.
    fn einsum_ba_abcd_cd_realop(
        g: &Array2<Self>,
        t: &Array4<f64>,
        h: &Array2<Self>,
    ) -> Self {
        einsum_ba_abcd_cd_real(g, t, h)
    }
}

impl NOCIScalar for Complex64 {
    fn from_imag(x: f64) -> Self {
        Complex64::new(0.0, x)
    }

    /// Calculate Einstein summation of complex matrices `g` and `h` as \sum_{a,b} g_{b,a} h_{a,b}.
    /// Assumes `g` and `h` are of identical shape.
    /// # Arguments
    /// - `g`: Complex matrix 1.
    /// - `h`: Complex matrix 2.
    /// # Returns
    /// - `Complex64`: Contracted scalar.
    fn einsum_ba_ab(
        g: &Array2<Self>,
        h: &Array2<Self>,
    ) -> Self {
        einsum_ba_ab_complex(g, h)
    }

    /// Calculate Einstein summation of complex matrix `g` and real matrix `h` as \sum_{a,b} g_{b,a} h_{a,b}.
    /// Assumes `g` and `h` are of identical shape.
    /// # Arguments
    /// - `g`: Complex matrix.
    /// - `h`: Real matrix.
    /// # Returns
    /// - `Complex64`: Contracted scalar.
    fn einsum_ba_ab_realop(
        g: &Array2<Self>,
        h: &Array2<f64>,
    ) -> Self {
        einsum_ba_ab_complex_real(g, h)
    }

    /// Calculate Einstein summation of complex matrices `g` and `h` and complex 4D tensor `t` as
    /// \sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.
    /// Assumes `g`, `h` and `t` all have axes of equal length.
    /// # Arguments
    /// - `g`: Complex matrix 1.
    /// - `t`: Complex 4D tensor.
    /// - `h`: Complex matrix 2.
    /// # Returns
    /// - `Complex64`: Contracted scalar.
    fn einsum_ba_abcd_cd(
        g: &Array2<Self>,
        t: &Array4<Self>,
        h: &Array2<Self>,
    ) -> Self {
        einsum_ba_abcd_cd_complex(g, t, h)
    }

    /// Calculate Einstein summation of complex matrices `g` and `h` and real 4D tensor `t` as
    /// \sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.
    /// Assumes `g`, `h` and `t` all have axes of equal length.
    /// # Arguments
    /// - `g`: Complex matrix 1.
    /// - `t`: Real 4D tensor.
    /// - `h`: Complex matrix 2.
    /// # Returns
    /// - `Complex64`: Contracted scalar.
    fn einsum_ba_abcd_cd_realop(
        g: &Array2<Self>,
        t: &Array4<f64>,
        h: &Array2<Self>,
    ) -> Self {
        einsum_ba_abcd_cd_complex_real(g, t, h)
    }
}

/// Shared data required for NOCI matrix-element evaluation.
pub struct NOCIData<'a, T: NOCIScalar> {
    /// AO-basis integrals and other system-wide data.
    pub ao: &'a AoData,
    /// List of the current determinants in the basis.
    pub basis: &'a [DetState<T>],
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
    /// - `basis`: Determinant basis with respect to which matrix elements are being evaluated.
    /// - `input`: User defined input options.
    /// - `tol`: Tolerance for a number being zero.
    /// - `wicks`: View to the intermediates required for non-orthogonal Wick's theorem.
    /// # Returns:
    /// - `NOCIData<'a, T>`: Shared data for NOCI matrix-element evaluation.
    pub fn new(
        ao: &'a AoData,
        basis: &'a [DetState<T>],
        input: &'a Input,
        tol: f64,
        wicks: Option<&'a WicksView<T>>,
    ) -> Self {
        Self {
            ao,
            basis,
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
pub(crate) struct DetPair<'a, T: NOCIScalar> {
    /// Left determinant in the matrix element.
    pub(crate) ldet: &'a DetState<T>,
    /// Right determinant in the matrix element.
    pub(crate) gdet: &'a DetState<T>,
}

impl<'a, T: NOCIScalar> DetPair<'a, T> {
    /// Construct the pair of determinants whose matrix element is to be evaluated.
    /// # Arguments:
    /// - `ldet`: Left determinant in the matrix element.
    /// - `gdet`: Right determinant in the matrix element.
    /// # Returns:
    /// - `DetPair<'a, T>`: Pair of determinants to be passed to matrix-element routines.
    pub(crate) fn new(
        ldet: &'a DetState<T>,
        gdet: &'a DetState<T>,
    ) -> Self {
        Self { ldet, gdet }
    }
}

/// Trait which defines how returned determinant-pair quantities should be scattered into matrices.
pub(in crate::noci) trait ScatterValue: Sized + Copy {
    type Output;

    /// Construct zero initialised output.
    /// # Arguments:
    /// - `nl`: Length of determinant set 1.
    /// - `nr`: Length of determinant set 2.
    /// # Returns:
    /// - `Self::Output`: Zero initialised output container.
    fn zeros(
        nl: usize,
        nr: usize,
    ) -> Self::Output;

    /// Write a value into the output at indices i, j.
    /// # Arguments:
    /// - `out`: Output container to write into.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: Matrix element value.
    fn write(
        out: &mut Self::Output,
        i: usize,
        j: usize,
        val: Self,
    );

    /// Value to write into the mirrored Hermitian position.
    /// # Arguments:
    /// - `self`: Matrix element value.
    /// # Returns:
    /// - `Self`: Complex-conjugated mirrored value.
    fn mirror(self) -> Self;
}

impl<T: NOCIScalar> ScatterValue for T {
    type Output = Array2<T>;

    /// Construct zero initialised matrix.
    /// # Arguments:
    /// - `nl`: Number of rows.
    /// - `nr`: Number of columns.
    /// # Returns:
    /// - `Array2<T>`: Zero initialised matrix.
    fn zeros(
        nl: usize,
        nr: usize,
    ) -> Self::Output {
        Array2::<T>::zeros((nl, nr))
    }

    /// Write scalar value into matrix at row `i` and column `j`.
    /// # Arguments:
    /// - `out`: Matrix to write into.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: Value to write.
    fn write(
        out: &mut Self::Output,
        i: usize,
        j: usize,
        val: Self,
    ) {
        out[(i, j)] = val;
    }

    /// Return Hermitian mirrored value for the lower triangle.
    /// # Arguments:
    /// - `self`: Matrix element value.
    /// # Returns:
    /// - `Self`: Complex conjugated matrix element value.
    fn mirror(self) -> Self {
        self.conj()
    }
}

impl<T: NOCIScalar> ScatterValue for (T, T) {
    type Output = (Array2<T>, Array2<T>);

    /// Construct pair of zero initialised matrices.
    /// # Arguments:
    /// - `nl`: Number of rows.
    /// - `nr`: Number of columns.
    /// # Returns:
    /// - `(Array2<T>, Array2<T>)`: Pair of zero initialised matrices.
    fn zeros(
        nl: usize,
        nr: usize,
    ) -> Self::Output {
        (Array2::<T>::zeros((nl, nr)), Array2::<T>::zeros((nl, nr)))
    }

    /// Write pair of scalar values into pair of matrices at row `i` and column `j`.
    /// # Arguments:
    /// - `out`: Pair of matrices to write into.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: Pair of values to write.
    fn write(
        out: &mut Self::Output,
        i: usize,
        j: usize,
        val: Self,
    ) {
        out.0[(i, j)] = val.0;
        out.1[(i, j)] = val.1;
    }

    /// Return Hermitian mirrored values for the lower triangle.
    /// # Arguments:
    /// - `self`: Pair of matrix element values.
    /// # Returns:
    /// - `Self`: Pair of complex conjugated matrix element values.
    fn mirror(self) -> Self {
        (self.0.conj(), self.1.conj())
    }
}

/// Storage of quantities required to compute matrix elements between determinant pairs in the naive fashion.
pub(in crate::noci) struct Pair<T: NOCIScalar> {
    /// Full overlap matrix element for the determinant pair.
    pub(in crate::noci) s: T,
    /// Product of the non-zero singular values of the occupied-overlap matrix.
    pub(in crate::noci) s_red: f64,
    /// Indices of singular values treated as zero.
    pub(in crate::noci) zeros: Vec<usize>,
    /// Weighted co-density matrix used when no singular values are zero.
    pub(in crate::noci) w: Option<Array2<T>>,
    /// Co-density matrix associated with the first zero singular value.
    pub(in crate::noci) p_i: Option<Array2<T>>,
    /// Co-density matrix associated with the second zero singular value.
    pub(in crate::noci) p_j: Option<Array2<T>>,
    /// Overall phase from the SVD rotations of the occupied-overlap matrix.
    pub(in crate::noci) phase: T,
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
    /// Whether parent alpha and beta MO spaces are Hermitian-orthonormal.
    pub hermitian_orthonormal: bool,
}

/// MO-basis Fock caches for orthogonal-parent matrix elements.
pub struct FockMOCache<T: NOCIScalar> {
    /// Spin-alpha Fock matrix in the parent alpha MO basis.
    pub fa: Array2<T>,
    /// Spin-beta Fock matrix in the parent beta MO basis.
    pub fb: Array2<T>,
    /// Whether parent alpha and beta MO spaces are Hermitian-orthonormal.
    pub hermitian_orthonormal: bool,
}
