// determinant/parent.rs

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::{SCFState, StateScalar};

/// One selected SCF reference's orbital frame and occupations.
#[derive(Clone)]
pub struct ParentDeterminant<T: StateScalar> {
    /// SCF energy associated with this parent reference.
    pub(crate) e: T,
    /// Alpha-spin occupation of the parent determinant.
    pub(crate) oa: u128,
    /// Beta-spin occupation of the parent determinant.
    pub(crate) ob: u128,
    /// Alpha-spin MO coefficients defining the parent orbital frame.
    pub(crate) ca: Arc<Array2<T>>,
    /// Beta-spin MO coefficients defining the parent orbital frame.
    pub(crate) cb: Arc<Array2<T>>,
    /// User-visible label inherited from the selected SCF state.
    pub(crate) label: String,
}

impl<T: StateScalar> ParentDeterminant<T> {
    /// Retain only the orbital-frame data of a selected SCF solution.
    /// # Arguments:
    /// - `state`: Selected SCF solution.
    /// # Returns:
    /// - `Self`: Parent reference shared by descendant determinants.
    pub(crate) fn from_scf(state: &SCFState<T>) -> Self {
        Self {
            e: state.e,
            oa: state.oa,
            ob: state.ob,
            ca: Arc::clone(&state.ca),
            cb: Arc::clone(&state.cb),
            label: state.label.clone(),
        }
    }
}
