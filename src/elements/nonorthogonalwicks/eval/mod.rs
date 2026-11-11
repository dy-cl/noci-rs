// elements/nonorthogonalwicks/eval/mod.rs

//! Prepared and batched extended nonorthogonal Wick evaluators, one file per operator family:
//! `overlap.rs`, `onebody.rs` and `hamiltonian.rs`, with the shared same-spin factor rows in
//! `batch.rs`.

mod batch;
mod dispatch;
mod hamiltonian;
mod helpers;
mod onebody;
mod overlap;
mod prepare;
#[cfg(feature = "gnocc")]
mod rdm;

// Crate-visible type re-exports.
pub(crate) use batch::{SpinFactor, SpinFactorRow, SpinStringSources};
pub(crate) use onebody::OneBodyFactor;
pub(crate) use overlap::OverlapFactor;

// Crate-visible function re-exports.
pub(crate) use batch::xw_spin_factor_row;
pub(crate) use hamiltonian::xw_hamiltonian_overlap_prepared_batched;
#[cfg(feature = "gnocc")]
pub(crate) use overlap::xw_overlap_prepared;
#[cfg(feature = "gnocc")]
pub(crate) use rdm::xw_rdmk_same_prepared_batched;
