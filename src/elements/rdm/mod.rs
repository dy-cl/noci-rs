// elements/rdm/mod.rs
//! Reduced density matrices of NOCI states, assembled from determinant-pair transition
//! densities.

#[cfg(feature = "nocc")]
mod common;
mod density;
#[cfg(feature = "nocc")]
mod rdm1;
#[cfg(feature = "nocc")]
mod rdm2;
#[cfg(feature = "nocc")]
mod rdm3;
#[cfg(feature = "nocc")]
mod rdm4;

// Crate-visible type re-exports.
#[cfg(feature = "nocc")]
pub(crate) use rdm1::RDM1;
#[cfg(feature = "nocc")]
pub(crate) use rdm2::RDM2;
#[cfg(feature = "nocc")]
pub(crate) use rdm3::RDM3;
#[cfg(feature = "nocc")]
pub(crate) use rdm4::RDM4;

// Crate-visible function re-exports.
pub(crate) use density::noci_density;
#[cfg(feature = "nocc")]
pub(crate) use rdm1::rdm1;
#[cfg(feature = "nocc")]
pub(crate) use rdm2::rdm2;
#[cfg(feature = "nocc")]
pub(crate) use rdm3::rdm3;
#[cfg(feature = "nocc")]
pub(crate) use rdm4::rdm4;

// Restricted type re-exports.
#[cfg(feature = "nocc")]
pub(in crate::elements) use common::RDMDeterminantView;

// Restricted function re-exports.
#[cfg(feature = "nocc")]
pub(in crate::elements) use common::resolve_rdm_determinant;
