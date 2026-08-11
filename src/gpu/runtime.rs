// gpu/runtime.rs
//! CubeCL runtime selection and generic accelerator context.

/// CubeCL runtime selected by Cargo feature.
#[cfg(feature = "gpu-cuda")]
pub(crate) type GpuRuntime = cubecl::cuda::CudaRuntime;

/// CubeCL runtime selected by Cargo feature.
#[cfg(feature = "gpu-hip")]
pub(crate) type GpuRuntime = cubecl::hip::HipRuntime;

/// CubeCL runtime selected by Cargo feature.
#[cfg(feature = "gpu-vulkan")]
pub(crate) type GpuRuntime = cubecl::wgpu::WgpuRuntime;

/// Common GPU context for backend-independent CubeCL infrastructure.
pub(crate) struct GpuContext {
    /// Runtime name selected by Cargo feature.
    runtime: &'static str,
}

impl GpuContext {
    /// Create a lightweight GPU context descriptor for the selected runtime.
    /// # Returns
    /// - `GpuContext`: Runtime descriptor shared by GPU scientific modules.
    pub(crate) fn new() -> Self {
        Self {
            runtime: runtime_name(),
        }
    }

    /// Return the selected CubeCL runtime name.
    /// # Arguments:
    /// - `self`: GPU context descriptor.
    /// # Returns
    /// - `&'static str`: Runtime name selected at compile time.
    pub(crate) fn runtime_name(&self) -> &'static str {
        self.runtime
    }
}

/// Return the selected CubeCL runtime name.
/// # Returns
/// - `&'static str`: Runtime name selected at compile time.
pub(crate) fn runtime_name() -> &'static str {
    #[cfg(feature = "gpu-cuda")]
    {
        "cuda"
    }
    #[cfg(feature = "gpu-hip")]
    {
        "hip"
    }
    #[cfg(feature = "gpu-vulkan")]
    {
        "vulkan"
    }
}
