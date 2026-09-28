// nocc/space/mod.rs
//! Orbital spaces, the spin-free excitation manifold and the first-order interacting space.

mod excitations;
mod fois;
mod orbitals;

// Restricted type re-exports.
pub(crate) use excitations::Excitation;
pub(in crate::nocc) use excitations::{DenseAmplitudes, ExcitationClass};
pub(crate) use fois::FoisBasis;
pub(crate) use orbitals::Spaces;

// Restricted function re-exports.
pub(crate) use excitations::build_excitations;
pub(in crate::nocc) use excitations::{dense_amplitudes, excitation_class};
pub(crate) use fois::build_fois_basis;
pub(in crate::nocc) use fois::project_onto_fois;
pub(crate) use orbitals::build_spaces;
