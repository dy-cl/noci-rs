// nonorthogonalwicks/mod.rs
mod storage;
mod view;
mod types;
mod layout;
mod scratch;
mod build;
mod eval;

pub(crate) use types::{SameSpinMeta, DiffSpinMeta, PairMeta};
pub(crate) use build::{SameSpinBuild, DiffSpinBuild};
pub(crate) use scratch::{WickScratchSpin};

pub use storage::WicksShared;
pub use view::WicksView;
pub(crate) use storage::{WicksDiskMeta, WicksRma};

pub(crate) use eval::{prepare_same, lg_overlap, lg_f, lg_h1, lg_h2_same, lg_h2_diff};
pub(crate) use layout::{assign_offsets, write2, write_same_spin, write_diff_spin};
pub(crate) use storage::{create_wicks_mmap, load_wicks_mmap};

