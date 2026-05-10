use std::{ops::AddAssign, sync::Arc};

use ndarray::{Array2, LinalgScalar};
use ndarray_linalg::{Lapack, Scalar};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::Excitation;

// Scalar generic marker trait for SCF states.
pub trait StateScalar: LinalgScalar + Scalar<Real = f64> + Lapack + AddAssign + Send + Sync + Serialize + for<'de> Deserialize<'de> {}

impl StateScalar for f64 {}
impl StateScalar for Complex64 {}

/// SCF determinant state with scalar-valued orbital data.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "T: StateScalar", deserialize = "T: StateScalar"))]
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
            ca: Arc::new(st.ca.mapv(|x| Complex64::new(x, 0.0))),
            cb: Arc::new(st.cb.mapv(|x| Complex64::new(x, 0.0))),
            da: Arc::new(st.da.mapv(|x| Complex64::new(x, 0.0))),
            db: Arc::new(st.db.mapv(|x| Complex64::new(x, 0.0))),
            label: st.label.clone(),
            noci_basis: st.noci_basis,
            parent: st.parent,
            excitation: st.excitation.clone(),
        }
    }
}
