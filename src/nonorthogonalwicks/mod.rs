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

pub(crate) use eval::{lg_f, lg_h1, lg_h2_diff, lg_h2_same, lg_overlap, prepare_same};
#[cfg(feature = "nocc")]
pub(crate) use eval::{lg_rdm_same_element, lg_rdm1, lg_rdm2_diff, lg_rdm2_same};
pub(crate) use layout::{assign_offsets, write_diff_spin, write_same_spin, write2t};
pub(crate) use storage::{create_wicks_mmap, load_wicks_mmap};
