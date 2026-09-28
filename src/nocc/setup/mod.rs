// nocc/setup/mod.rs
//! Construction of the normal-ordered reference from a NOCI state.

mod orbitals;
mod reference;

// Restricted type re-exports.
pub(crate) use reference::ReferenceState;

// Restricted function re-exports.
pub(crate) use orbitals::{
    noci_natural_orbitals, print_noci_natural_orbitals, transform_ao_data, transform_noci_basis,
};
pub(crate) use reference::reference_energy;
