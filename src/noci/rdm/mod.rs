// noci/rdm/mod.rs

mod common;
mod rdm1;
mod rdm2;
mod rdm3;
mod rdm4;

pub(crate) use self::rdm1::{rdm1, RDM1};
pub(crate) use self::rdm2::{rdm2, RDM2};
pub(crate) use self::rdm3::{rdm3, RDM3};
pub(crate) use self::rdm4::{rdm4, RDM4};
