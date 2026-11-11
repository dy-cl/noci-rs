// determinant/mod.rs
//! Shared determinant-space foundations: parent orbital frames, determinant identities, the
//! retained NOCI determinant space and its auxiliary spaces, and occupation-identified
//! determinants outside any precomputed space.

pub(crate) mod auxiliary;
pub(crate) mod connection;
pub(crate) mod excited;
pub(crate) mod parent;
pub(crate) mod space;
pub(crate) mod state;

// Public type re-exports.
pub use parent::ParentDeterminant;
pub use space::{NOCIIndex, NOCISpace};

// Crate-visible type re-exports.
pub(crate) use auxiliary::{AuxiliaryIndex, AuxiliarySpace, AuxiliarySpinStringIndex};
pub(crate) use connection::ConnectionState;
pub(crate) use excited::{DeterminantData, DeterminantState};
pub(crate) use space::NOCISpinStringIndex;
pub(crate) use state::{
    ParentSpinStrings, SpinStringData, SpinStringIndex, SpinStringPair, SpinStringState,
};
