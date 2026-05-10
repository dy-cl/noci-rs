use std::sync::Arc;

use ndarray::{Array2, LinalgScalar};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;

use crate::Excitation;

/// Scalar type used in determinant states and scalar-generic SCF kernels.
pub trait StateScalar: LinalgScalar + AddAssign + Serialize + for<'de> Deserialize<'de> {
    /// Construct scalar from real value.
    /// # Arguments:
    /// - `x`: Real value.
    /// # Returns:
    /// - `Self`: Scalar value.
    fn from_f64(x: f64) -> Self;

    /// Complex conjugate of scalar.
    /// # Arguments:
    /// - `self`: Scalar value.
    /// # Returns:
    /// - `Self`: Conjugated scalar.
    fn conj(self) -> Self;

    /// Real part of scalar.
    /// # Arguments:
    /// - `self`: Scalar value.
    /// # Returns:
    /// - `f64`: Real component.
    fn re(self) -> f64;

    /// Magnitude of scalar.
    /// # Arguments:
    /// - `self`: Scalar value.
    /// # Returns:
    /// - `f64`: Scalar magnitude.
    fn abs(self) -> f64;
}

impl StateScalar for f64 {
    fn from_f64(x: f64) -> Self {x}
    fn conj(self) -> Self {self}
    fn re(self) -> f64 {self}
    fn abs(self) -> f64 {self.abs()}
}

impl StateScalar for Complex64 {
    fn from_f64(x: f64) -> Self {Complex64::new(x, 0.0)}
    fn conj(self) -> Self {Complex64::new(self.re, -self.im)}
    fn re(self) -> f64 {self.re}
    fn abs(self) -> f64 {self.norm()}
}

/// SCF determinant state with scalar-valued orbital data.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound = "T: StateScalar")]
pub struct DetState<T: StateScalar> {
    /// Energy of SCF state in Ha.
    pub e: T,
    /// MO occupancy vector for spin alpha orbitals as bitstring.
    pub oa: u128,
    /// MO occupancy vector for spin beta orbitals as bitstring.
    pub ob: u128,
    /// Fermionic phase relative to parent for spin alpha electrons.
    pub pha: f64,
    /// Fermionic phase relative to parent for spin beta electrons.
    pub phb: f64,
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
    /// Index of reference parent determinant if excited for QMC basis.
    pub parent: usize,
    /// Excitation relative to parent if excited for QMC basis.
    pub excitation: Excitation,
}

/// Real-valued SCF determinant state.
pub type SCFState = DetState<f64>;

/// Complex-valued holomorphic SCF determinant state.
pub type HSCFState = DetState<Complex64>;

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
            pha: st.pha,
            phb: st.phb,
            ca: Arc::new(st.ca.mapv(Complex64::from)),
            cb: Arc::new(st.cb.mapv(Complex64::from)),
            da: Arc::new(st.da.mapv(Complex64::from)),
            db: Arc::new(st.db.mapv(Complex64::from)),
            label: st.label.clone(),
            noci_basis: st.noci_basis,
            parent: st.parent,
            excitation: st.excitation.clone(),
        }
    }
}
