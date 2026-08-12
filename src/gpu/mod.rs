// gpu/mod.rs
//! Common CubeCL accelerator infrastructure.

pub(crate) mod runtime;
pub(crate) mod scalar;

// Crate-visible type re-exports.
pub(crate) use runtime::{GpuBuffer, GpuContext, GpuRuntime};
pub(crate) use scalar::GpuComplex64;

// Crate-visible function re-exports.
pub(crate) use runtime::runtime_name;
