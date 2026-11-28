// elements/rdm/mod.rs
//! Reduced density matrices of NOCI states, assembled from determinant-pair transition
//! densities.

mod density;
#[cfg(feature = "gnocc")]
mod rdm1;
#[cfg(feature = "gnocc")]
mod rdm2;
#[cfg(feature = "gnocc")]
mod rdm3;
#[cfg(feature = "gnocc")]
mod rdm4;
mod transition;

// Crate-visible type re-exports.
#[cfg(feature = "gnocc")]
pub(crate) use rdm1::RDM1;
#[cfg(feature = "gnocc")]
pub(crate) use rdm2::RDM2;
#[cfg(feature = "gnocc")]
pub(crate) use rdm3::RDM3;
#[cfg(feature = "gnocc")]
pub(crate) use rdm4::RDM4;
pub(crate) use transition::SpinTransition;

// Crate-visible function re-exports.
pub(crate) use density::{build_noci_spin_square, noci_density};
#[cfg(feature = "gnocc")]
pub(crate) use rdm1::rdm1;
#[cfg(feature = "gnocc")]
pub(crate) use rdm2::rdm2;
#[cfg(feature = "gnocc")]
pub(crate) use rdm3::rdm3;
#[cfg(feature = "gnocc")]
pub(crate) use rdm4::rdm4;
