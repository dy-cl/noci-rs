// gpu/runtime.rs
//! CubeCL runtime selection, device buffers and generic accelerator context.

// Standard library imports.
use std::marker::PhantomData;

// External crate imports.
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

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
    /// CubeCL compute client for the selected runtime.
    client: ComputeClient<GpuRuntime>,
    /// Runtime name selected by Cargo feature.
    runtime: &'static str,
}

impl GpuContext {
    /// Create a CubeCL context for the selected runtime on the default device.
    /// # Returns
    /// - `GpuContext`: Runtime client shared by GPU scientific modules.
    pub(crate) fn new() -> Self {
        let device = <GpuRuntime as Runtime>::Device::default();
        Self {
            client: GpuRuntime::client(&device),
            runtime: runtime_name(),
        }
    }

    /// Borrow the CubeCL compute client.
    /// # Arguments:
    /// - `self`: GPU context.
    /// # Returns
    /// - `&ComputeClient<GpuRuntime>`: Runtime client for buffer operations and kernel launches.
    pub(crate) fn client(&self) -> &ComputeClient<GpuRuntime> {
        &self.client
    }

    /// Return the selected CubeCL runtime name.
    /// # Arguments:
    /// - `self`: GPU context.
    /// # Returns
    /// - `&'static str`: Runtime name selected at compile time.
    pub(crate) fn runtime_name(&self) -> &'static str {
        self.runtime
    }
}

/// Typed CubeCL device buffer backed by a raw runtime `Handle`.
pub(crate) struct GpuBuffer<T: CubeElement> {
    /// Raw CubeCL allocation handle.
    pub(crate) handle: Handle,
    /// Number of typed elements.
    pub(crate) len: usize,
    marker: PhantomData<T>,
}

impl<T: CubeElement> GpuBuffer<T> {
    /// Upload a typed host slice into device memory.
    /// # Arguments:
    /// - `context`: CubeCL context owning the target client.
    /// - `values`: Host values to upload.
    /// # Returns
    /// - `GpuBuffer<T>`: Device allocation containing `values`.
    pub(crate) fn from_slice(
        context: &GpuContext,
        values: &[T],
    ) -> Self {
        let handle = context.client().create_from_slice(T::as_bytes(values));
        Self {
            handle,
            len: values.len(),
            marker: PhantomData,
        }
    }

    /// Allocate uninitialised device memory for `len` typed values.
    /// # Arguments:
    /// - `context`: CubeCL context owning the target client.
    /// - `len`: Number of typed elements to allocate.
    /// # Returns
    /// - `GpuBuffer<T>`: Uninitialised device allocation.
    pub(crate) fn empty(
        context: &GpuContext,
        len: usize,
    ) -> Self {
        let handle = context.client().empty(len * core::mem::size_of::<T>());
        Self {
            handle,
            len,
            marker: PhantomData,
        }
    }

    /// Download the full device buffer to host memory.
    /// # Arguments:
    /// - `self`: Device buffer to read.
    /// - `context`: CubeCL context owning the source client.
    /// # Returns
    /// - `Vec<T>`: Host copy of the device buffer.
    pub(crate) fn read(
        &self,
        context: &GpuContext,
    ) -> Vec<T> {
        let bytes = context
            .client()
            .read_one(self.handle.clone())
            .expect("CubeCL device read failed");
        T::from_bytes(&bytes).to_vec()
    }

    /// Return the number of typed elements in the buffer.
    /// # Arguments:
    /// - `self`: Device buffer.
    /// # Returns
    /// - `usize`: Number of typed elements.
    pub(crate) fn len(&self) -> usize {
        self.len
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
