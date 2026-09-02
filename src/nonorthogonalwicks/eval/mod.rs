mod dispatch;
mod helpers;
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
#[cfg(target_arch = "x86_64")]
mod simd;

// Crate-visible type re-exports.
pub(crate) use overlap::SameSpinOverlapBatch;
pub(crate) use prepareonebodyoverlap::SameSpinOneBodyBatch;

// Crate-visible function re-exports.
#[cfg(feature = "nocc")]
pub(crate) use overlap::xw_overlap;
pub(crate) use overlap::{xw_overlap_prepared, xw_overlap_prepared_batched};
#[cfg(feature = "nocc")]
pub(crate) use prepare::prepare_same;
pub(crate) use preparehamiltonianoverlap::{
    xw_hamiltonian_overlap_prepared, xw_hamiltonian_overlap_prepared_batched,
};
pub(crate) use prepareonebodyoverlap::{xw_f_overlap_prepared, xw_f_overlap_prepared_batched};
#[cfg(feature = "nocc")]
pub(crate) use rdm1::xw_rdm1;
#[cfg(feature = "nocc")]
pub(crate) use rdm2diff::xw_rdm2_diff;
#[cfg(feature = "nocc")]
pub(crate) use rdm2same::xw_rdm2_same;
#[cfg(feature = "nocc")]
pub(crate) use rdmksame::xw_rdm_same_element;
