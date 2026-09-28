// determinant/mod.rs
//! Shared determinant-space foundations: parent orbital frames, determinant identities, the
//! retained NOCI determinant space and its auxiliary spaces.

pub(crate) mod auxiliary;
pub(crate) mod connection;
pub(crate) mod parent;
pub(crate) mod reduced;
pub(crate) mod space;
pub(crate) mod state;

// Public type re-exports.
pub use parent::ParentDeterminant;
pub use space::{NOCIDeterminantState, NOCIIndex, NOCISpace};

// Crate-visible type re-exports.
pub(crate) use auxiliary::{
    AuxiliaryDeterminantState, AuxiliaryIndex, AuxiliarySpace, AuxiliarySpinIndex,
};
pub(crate) use connection::OrthogonalConnection;
pub(crate) use reduced::{ReducedOneSpinDeterminantState, ReducedTwoSpinDeterminantState};
pub(crate) use space::{NOCISpinIndex, ReducedOneSpinNOCIDeterminantState};
pub(crate) use state::{
    DeterminantState, ParentComponents, SpinDeterminantIndex, SpinDeterminantState,
};
