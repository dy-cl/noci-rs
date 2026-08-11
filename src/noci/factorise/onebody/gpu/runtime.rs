// noci/factorise/onebody/gpu/runtime.rs
//! CubeCL runtime selection for the factorised one-body GPU backend.

/// CubeCL runtime selected by Cargo feature.
#[cfg(feature = "gpu-cuda")]
pub(crate) type GpuRuntime = cubecl::cuda::CudaRuntime;

/// CubeCL runtime selected by Cargo feature.
#[cfg(feature = "gpu-hip")]
pub(crate) type GpuRuntime = cubecl::hip::HipRuntime;

/// CubeCL runtime selected by Cargo feature.
#[cfg(feature = "gpu-vulkan")]
pub(crate) type GpuRuntime = cubecl::wgpu::WgpuRuntime;

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
