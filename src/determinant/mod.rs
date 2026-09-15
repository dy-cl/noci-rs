// determinant/mod.rs
//! Shared determinant-space foundations.

pub(crate) mod parent;
pub(crate) mod reduced;
pub(crate) mod state;

pub use parent::ParentDeterminant;
pub(crate) use reduced::{ReducedOneSpinDeterminantState, ReducedTwoSpinDeterminantState};
pub(crate) use state::{
    DeterminantState, ParentComponents, SpinDeterminantIndex, SpinDeterminantState,
};
