// Standard library imports.
use std::{ops::AddAssign, sync::Arc};

// External crate imports.
use ndarray::{Array2, Array4, LinalgScalar};
use ndarray_linalg::{Lapack, Scalar};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

// Crate-root imports.
use crate::maths::{
    ERIScalar, einsum_ba_ab_complex, einsum_ba_ab_complex_real, einsum_ba_ab_real,
    einsum_ba_abcd_cd_complex, einsum_ba_abcd_cd_complex_real, einsum_ba_abcd_cd_real,
};

// Scalar generic marker trait for SCF states.
pub trait StateScalar:
    LinalgScalar
    + Scalar<Real = f64>
    + Lapack
    + AddAssign
    + Send
    + Sync
    + Serialize
    + for<'de> Deserialize<'de>
{
}

impl StateScalar for f64 {}
impl StateScalar for Complex64 {}

/// Scalar-valued converged SCF solution.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "T: StateScalar", deserialize = "T: StateScalar"))]
pub struct SCFState<T: StateScalar = f64> {
    /// Energy of SCF state in Ha.
    pub e: T,
    /// MO occupancy vector for spin alpha orbitals as bitstring.
    pub oa: u128,
    /// MO occupancy vector for spin beta orbitals as bitstring.
    pub ob: u128,
    /// MO coefficients for spin alpha electrons, (nao, nao).
    pub ca: Arc<Array2<T>>,
    /// MO coefficients for spin beta electrons, (nao, nao).
    pub cb: Arc<Array2<T>>,
    /// SCF converged density matrix spin alpha.
    pub da: Arc<Array2<T>>,
    /// SCF converged density matrix spin beta.
    pub db: Arc<Array2<T>>,
    /// Label defined in user input.
    pub label: String,
    /// Is this state used in the NOCI basis?
    pub noci_basis: bool,
}

/// Complex-valued holomorphic SCF determinant state.
pub type HSCFState = SCFState<Complex64>;

impl HSCFState {
    /// Promote a real SCF state to a complex h-SCF state.
    /// # Arguments:
    /// - `st`: Real SCF state.
    /// # Returns:
    /// - `HSCFState`: Complex state with zero imaginary components.
    pub fn from_real(st: &SCFState) -> Self {
        Self {
            e: Complex64::new(st.e, 0.0),
            oa: st.oa,
            ob: st.ob,
            ca: Arc::new(st.ca.mapv(|x| Complex64::new(x, 0.0))),
            cb: Arc::new(st.cb.mapv(|x| Complex64::new(x, 0.0))),
            da: Arc::new(st.da.mapv(|x| Complex64::new(x, 0.0))),
            db: Arc::new(st.db.mapv(|x| Complex64::new(x, 0.0))),
            label: st.label.clone(),
            noci_basis: st.noci_basis,
        }
    }
}

/// Scalar type accepted by generic NOCI matrix-element code.
pub trait NOCIScalar: StateScalar + From<f64> + Scalar<Real = f64> + ERIScalar {
    /// Construct a purely imaginary scalar.
    /// # Arguments:
    /// - `x`: Imaginary component.
    /// # Returns
    /// - `Self`: Purely imaginary scalar.
    fn from_imag(x: f64) -> Self;

    /// `Calculate Einstein summation of scalar matrices g and h as \sum_{a,b} g_{b,a} h_{a,b}.`
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

    /// `Calculate Einstein summation of scalar matrix g and real matrix h as \sum_{a,b} g_{b,a} h_{a,b}.`
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
    /// `\sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.`
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
    /// `\sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.`
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
    /// Convert a zero imaginary component to a real scalar.
    /// # Arguments:
    /// - `x`: Imaginary component, which must be zero.
    /// # Returns
    /// - `f64`: Zero when `x` is zero.
    /// # Panics
    /// - Panics if `x` is nonzero because `f64` cannot represent an imaginary value.
    fn from_imag(x: f64) -> Self {
        if x == 0.0 {
            0.0
        } else {
            panic!("non-zero SNOCI imaginary shift requires complex arithmetic")
        }
    }

    /// `Calculate Einstein summation of real matrices g and h as \sum_{a,b} g_{b,a} h_{a,b}.`
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

    /// `Calculate Einstein summation of real matrices g and h as \sum_{a,b} g_{b,a} h_{a,b}.`
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
    /// `\sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.`
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
    /// `\sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.`
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
    /// Construct a purely imaginary complex scalar.
    /// # Arguments:
    /// - `x`: Imaginary component.
    /// # Returns
    /// - `Complex64`: Complex number with zero real part.
    fn from_imag(x: f64) -> Self {
        Complex64::new(0.0, x)
    }

    /// `Calculate Einstein summation of complex matrices g and h as \sum_{a,b} g_{b,a} h_{a,b}.`
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

    /// `Calculate Einstein summation of complex matrix g and real matrix h as \sum_{a,b} g_{b,a} h_{a,b}.`
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
    /// `\sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.`
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
    /// `\sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.`
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
