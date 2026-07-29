// nonorthogonalwicks/mod.rs
mod build;
mod eval;
mod layout;
mod scratch;
mod storage;
mod types;
mod view;

pub(crate) use build::{DiffSpinBuild, SameSpinBuild};
pub(crate) use scratch::WickScratchSpin;
pub(crate) use types::{DiffSpinMeta, PairMeta, PairZeroCounts, SameSpinMeta};

pub use storage::WicksShared;
pub(crate) use storage::{WicksDiskMeta, WicksRma};
pub(crate) use view::WicksPairView;
pub use view::WicksView;

pub(crate) use eval::{
    prepare_same, xw_f, xw_h1, xw_h2_diff, xw_h2_same, xw_overlap, xw_overlap_same_f64,
};
#[cfg(feature = "nocc")]
pub(crate) use eval::{xw_rdm_same_element, xw_rdm1, xw_rdm2_diff, xw_rdm2_same};
pub(crate) use layout::{assign_offsets, write_diff_spin, write_same_spin, write2t};
pub(crate) use storage::{create_wicks_mmap, load_wicks_mmap};
