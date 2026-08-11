// nonorthogonalwicks/gpu/scratch.rs
//! GPU scratch representation for nonorthogonal Wick evaluation.

/// Compile-time maximum excitation rank placeholder for future CubeCL Wick kernels.
pub(crate) const MAX_RANK: usize = 16;

/// Compile-time maximum contraction determinant size placeholder for future CubeCL Wick kernels.
pub(crate) const MAX_L: usize = 32;
