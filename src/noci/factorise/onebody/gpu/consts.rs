// noci/factorise/onebody/gpu/consts.rs
//! Constants for factorised one-body GPU contractions.

/// Square tile dimension for cooperative `f64` rank-block contractions.
pub(super) const GEMM_TILE_DIM: usize = 16;

/// Number of `u32` fields in one grouped stage block descriptor.
pub(super) const GROUPED_STAGE_BLOCK_FIELDS: usize = 8;

/// Number of `u32` fields in one grouped stage contribution descriptor.
pub(super) const GROUPED_STAGE_CONTRIBUTION_FIELDS: usize = 4;

/// Number of `u32` fields in one grouped final block descriptor.
pub(super) const GROUPED_FINAL_BLOCK_FIELDS: usize = 9;

/// Number of `u32` fields in one grouped final contribution descriptor.
pub(super) const GROUPED_FINAL_CONTRIBUTION_FIELDS: usize = 2;
