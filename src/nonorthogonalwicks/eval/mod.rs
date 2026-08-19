mod helpers;
mod onebody;
mod overlap;
mod prepare;
mod preparehamiltonianoverlap;
mod prepareonebodyoverlap;
#[cfg(feature = "nocc")]
mod rdm1;
#[cfg(feature = "nocc")]
mod rdm2diff;
#[cfg(feature = "nocc")]
mod rdm2same;
#[cfg(feature = "nocc")]
mod rdmksame;

// Public function re-exports.
pub use overlap::xw_overlap;
pub use prepare::prepare_same;

// Crate-visible function re-exports.
pub(crate) use onebody::xw_f;
pub(crate) use overlap::{xw_overlap_same_f64, xw_overlap_same_f64_batched};
pub(crate) use preparehamiltonianoverlap::{
    xw_hamiltonian_overlap_prepared, xw_hamiltonian_overlap_prepared_batched,
};
pub(crate) use prepareonebodyoverlap::xw_f_overlap_prepared;
#[cfg(target_arch = "x86_64")]
pub(crate) use prepareonebodyoverlap::{
    xw_f_overlap_m0_prepared_f64x4, xw_f_overlap_m0_prepared_f64x8,
};
#[cfg(feature = "nocc")]
pub(crate) use rdm1::xw_rdm1;
#[cfg(feature = "nocc")]
pub(crate) use rdm2diff::xw_rdm2_diff;
#[cfg(feature = "nocc")]
pub(crate) use rdm2same::xw_rdm2_same;
#[cfg(feature = "nocc")]
pub(crate) use rdmksame::xw_rdm_same_element;
