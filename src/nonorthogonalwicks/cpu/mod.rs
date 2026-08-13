// nonorthogonalwicks/cpu/mod.rs
//! CPU implementation of extended nonorthogonal Wick matrix elements.

pub(crate) mod build;
pub(crate) mod eval;
pub(crate) mod layout;
pub(crate) mod scratch;
pub(crate) mod storage;
pub(crate) mod view;

// Public type re-exports.
pub use storage::WicksShared;
pub use view::WicksView;

// Crate-visible type re-exports.
pub(crate) use build::{DiffSpinBuild, SameSpinBuild};
pub(crate) use scratch::WickScratchSpin;
pub(crate) use storage::{WicksDiskMeta, WicksRma};
pub(crate) use view::WicksPairView;

// Crate-visible function re-exports.
pub(crate) use eval::{
    prepare_same, xw_f, xw_h1, xw_h2_diff, xw_h2_same, xw_overlap, xw_overlap_f,
    xw_overlap_same_f64,
};
#[cfg(feature = "nocc")]
pub(crate) use eval::{xw_rdm_same_element, xw_rdm1, xw_rdm2_diff, xw_rdm2_same};
pub(crate) use layout::{assign_offsets, write_diff_spin, write_same_spin, write2t};
pub(crate) use storage::{create_wicks_mmap, load_wicks_mmap};
