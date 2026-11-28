// gpu/mod.rs
//! NVIDIA GPU support for the `gpu` feature: the device context shared by every GPU counterpart of
//! a host routine. Each counterpart sits beside its host module, its kernels in a `.cu` file that
//! `build/build.rs` compiles to PTX at the same path under `OUT_DIR`, for example
//! `noci/factorise/onebody/device.{rs,cu}` beside `noci/factorise/onebody.rs`.

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use cudarc::driver::{CudaContext, CudaModule, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;

/// One GPU with the stream on which its work is issued.
pub(crate) struct Device {
    /// Context of the GPU.
    context: Arc<CudaContext>,
    /// Stream on which every transfer and kernel of this device is issued, in order.
    pub(crate) stream: Arc<CudaStream>,
}

impl Device {
    /// Open a GPU and take its default stream.
    /// # Arguments:
    /// - `ordinal`: Index of the GPU among the visible devices.
    /// # Returns:
    /// - `Result<Device, DriverError>`: The device, or the driver error that prevented opening it.
    pub(crate) fn new(ordinal: usize) -> Result<Self, DriverError> {
        let context = CudaContext::new(ordinal)?;
        let stream = context.default_stream();

        Ok(Self { context, stream })
    }

    /// Load the kernels of one PTX module, which the driver compiles for this GPU.
    /// # Arguments:
    /// - `self`: Device.
    /// - `ptx`: PTX compiled by `build/build.rs`.
    /// # Returns:
    /// - `Result<Arc<CudaModule>, DriverError>`: Module holding the PTX's kernels.
    pub(crate) fn load(
        &self,
        ptx: &str,
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.context.load_module(Ptx::from_src(ptx))
    }
}
