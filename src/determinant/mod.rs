// determinant/mod.rs
//! Shared determinant-space foundations: parent orbital frames, the identities and payloads of
//! determinants and their spin strings, the connections between determinants of one frame, and
//! the retained NOCI determinant space with its auxiliary spaces.
//!
//! Every determinant and spin string has a handle (`…Index`) into a space, an identity
//! (`…State`) independent of any space, and a payload (`…Data`) measured from a reference of the
//! same frame.

pub(crate) mod auxiliary;
pub(crate) mod connection;
pub(crate) mod parent;
pub(crate) mod space;
pub(crate) mod state;

// Public type re-exports.
pub use parent::ParentDeterminant;
pub use space::{NOCIIndex, NOCISpace};

// Crate-visible type re-exports.
pub(crate) use auxiliary::{AuxiliaryIndex, AuxiliarySpace, AuxiliarySpinStringIndex};
pub(crate) use connection::ConnectionState;
pub(crate) use space::NOCISpinStringIndex;
pub(crate) use state::{
    DeterminantData, DeterminantState, ParentSpinStrings, SpinStringData, SpinStringIndex,
    SpinStringPair, SpinStringState,
};
