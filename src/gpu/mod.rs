// gpu/mod.rs
//! NVIDIA GPU support for the `gpu` feature: the device context and the CUDA kernels of
//! `kernels.cu`, compiled by `build/build.rs` into one fatbin with native code for compute
//! capabilities 8.6 and 9.0.

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use cudarc::driver::{CudaContext, CudaModule, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;

/// Kernels compiled by `build/build.rs`.
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.fatbin"));

/// One GPU with its default stream and the loaded kernels.
pub(crate) struct Device {
    /// Stream on which every transfer and kernel of this device is issued, in order.
    pub(crate) stream: Arc<CudaStream>,
    /// Module holding the kernels of `kernels.cu`.
    pub(crate) kernels: Arc<CudaModule>,
}

impl Device {
    /// Open a GPU, take its default stream and load the kernels.
    /// # Arguments:
    /// - `ordinal`: Index of the GPU among the visible devices.
    /// # Returns:
    /// - `Result<Device, DriverError>`: The device, or the driver error that prevented opening it.
    pub(crate) fn new(ordinal: usize) -> Result<Self, DriverError> {
        let context = CudaContext::new(ordinal)?;
        let stream = context.default_stream();
        let kernels = context.load_module(Ptx::from_binary(KERNELS.to_vec()))?;

        Ok(Self { stream, kernels })
    }
}
