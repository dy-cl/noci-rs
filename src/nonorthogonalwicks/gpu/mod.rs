// nonorthogonalwicks/gpu/mod.rs
//! GPU representation of nonorthogonal Wick intermediates.

mod build;
pub(crate) mod eval;
mod layout;
mod pack;
mod scratch;
mod storage;
pub(crate) mod types;
mod view;

// Crate-visible type re-exports.
pub(crate) use eval::prepare::GpuSameSpinView;
pub(crate) use storage::WicksShared;
pub(crate) use types::DeviceWicksShared;
pub(crate) use view::{SameSpinView, WicksPairView, WicksView};

// Crate-visible function re-exports.
pub(crate) use eval::onebody::xw_f;
pub(crate) use eval::onebodyoverlap::{xw_overlap_f, xw_overlap_f_m0};
pub(crate) use eval::overlap::xw_overlap;
pub(crate) use eval::prepare::prepare_same;
pub(crate) use pack::pack_wicks;
