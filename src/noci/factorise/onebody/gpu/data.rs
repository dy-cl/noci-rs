// noci/factorise/onebody/gpu/data.rs
//! GPU-resident data layout for factorised one-body NOCI operator contractions.

// Standard library imports.
use std::marker::PhantomData;

// Crate-root imports.
use crate::noci::types::{NOCIData, NOCIScalar};

/// Persistent GPU topology and Wick data for factorised one-body contractions.
pub(crate) struct GpuOneBodyData<T: NOCIScalar> {
    /// Number of determinants in the candidate basis.
    pub(crate) ndet: usize,
    /// Scalar marker for host-to-device conversion state.
    pub(crate) scalar: PhantomData<T>,
}

impl<T: NOCIScalar> GpuOneBodyData<T> {
    /// Pack persistent determinant topology for CubeCL kernels.
    /// # Arguments:
    /// - `data`: Shared NOCI data defining the candidate determinant basis and Wick views.
    /// # Returns
    /// - `GpuOneBodyData<T>`: GPU data descriptor.
    pub(crate) fn new(data: &NOCIData<'_, T>) -> Self {
        Self {
            ndet: data.basis.len(),
            scalar: PhantomData,
        }
    }
}
