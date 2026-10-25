// elements/nonorthogonalwicks/eval/mod.rs

//! Prepared and batched extended nonorthogonal Wick evaluators.

mod dispatch;
mod helpers;
mod overlap;
mod prepare;
mod preparehamiltonianoverlap;
mod prepareonebodyoverlap;
#[cfg(feature = "gnocc")]
mod rdm;

// Crate-visible type re-exports.
pub(crate) use overlap::{SameSpinOrthogonalOverlapBatch, SameSpinOverlapBatch};
pub(crate) use prepareonebodyoverlap::SameSpinOneBodyBatch;

// Crate-visible function re-exports.
pub(crate) use overlap::{
    xw_overlap_orthogonal_prepared_batched, xw_overlap_prepared, xw_overlap_prepared_batched,
};
pub(crate) use preparehamiltonianoverlap::{
    xw_hamiltonian_overlap_prepared, xw_hamiltonian_overlap_prepared_batched,
};
pub(crate) use prepareonebodyoverlap::{xw_f_overlap_prepared, xw_f_overlap_prepared_batched};
#[cfg(feature = "gnocc")]
pub(crate) use rdm::xw_rdmk_same_prepared_batched;
