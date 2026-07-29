// nocc/rdm/mod.rs

mod common;
mod rdm1;
mod rdm2;
mod rdm3;
mod rdm4;

// Crate-visible type re-exports.
pub(crate) use self::rdm1::RDM1;
pub(crate) use self::rdm2::RDM2;
pub(crate) use self::rdm3::RDM3;
pub(crate) use self::rdm4::RDM4;

// Crate-visible function re-exports.
pub(crate) use self::rdm1::rdm1;
pub(crate) use self::rdm2::rdm2;
pub(crate) use self::rdm3::rdm3;
pub(crate) use self::rdm4::rdm4;
