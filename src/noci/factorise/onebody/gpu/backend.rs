// noci/factorise/onebody/gpu/backend.rs
//! GPU backend for spin-factorised one-body NOCI operator contractions.

// Standard library imports.
use std::any::TypeId;
use std::path::Path;

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::gpu::GpuContext;
use crate::input::SNOCIStorage;
use crate::noci::types::{FockData, NOCIData, NOCIScalar};
use crate::nonorthogonalwicks::gpu::types::DeviceWicksShared;
use crate::nonorthogonalwicks::{WicksRequirements, gpu};

// Parent/sibling imports.
use super::super::super::SpinFactorisation;
use super::super::plan::OneBodyPlan;
use super::data::{DeviceOneBodyData, GpuOneBodyData};

/// CubeCL factorised one-body backend for the current generalised Fock.
pub(crate) struct GpuOneBodyBackend<T: NOCIScalar> {
    /// Common CubeCL context descriptor.
    context: GpuContext,
    /// Shared determinant-space factorisation `I <-> (P,a_I,b_I)`.
    spin: SpinFactorisation,
    /// Shared one-body topology and contraction plan.
    plan: OneBodyPlan,
    /// GPU-packed Wick data required for NOCI-PT2 one-body evaluation.
    wicks: gpu::WicksShared<T>,
    /// Device-resident real Wick data.
    device_wicks: DeviceWicksShared,
    /// Factorised-operator GPU topology data.
    data: GpuOneBodyData,
    /// Device-resident determinant topology and decoded excitations.
    device_data: DeviceOneBodyData,
}

impl<T: NOCIScalar + 'static> GpuOneBodyBackend<T> {
    /// Build GPU-resident data for the current generalised Fock operator.
    /// # Arguments:
    /// - `data`: Shared NOCI data with Wick intermediates for the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data, already reflected in Wick intermediates.
    /// - `cache`: Directory for persistent file-backed factor blocks.
    /// - `rank`: MPI rank used in factor-cache filenames.
    /// - `iteration`: SNOCI iteration used in factor-cache filenames.
    /// - `storage`: Requested persistent factor-table storage backend.
    /// # Returns
    /// - `GpuOneBodyBackend<T>`: GPU one-body backend descriptor.
    pub(crate) fn new(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        _cache: &Path,
        _rank: i32,
        _iteration: usize,
        storage: SNOCIStorage,
    ) -> Self {
        if TypeId::of::<T>() != TypeId::of::<f64>() {
            eprintln!("snoci.backend = \"gpu\" currently supports real f64 NOCI-PT2 data only");
            std::process::exit(1);
        }
        if !matches!(storage, SNOCIStorage::None) {
            eprintln!("snoci.backend = \"gpu\" requires snoci.gmres.factor_tables = \"none\"");
            std::process::exit(1);
        }
        let Some(wicks) = data.wicks else {
            eprintln!("snoci.backend = \"gpu\" requires Wick intermediates");
            std::process::exit(1);
        };
        let spin = SpinFactorisation::new(data);
        let plan = OneBodyPlan::new(&spin, fock);
        let requirements = WicksRequirements::one_body();
        let wicks = gpu::pack_wicks(wicks, requirements);
        let gpu_data = GpuOneBodyData::new(&spin, data);
        let context = GpuContext::new();
        let device_wicks = unsafe { upload_real_wicks(&wicks, &context) };
        let device_data = gpu_data.upload(&context);
        Self {
            context,
            spin,
            plan,
            wicks,
            device_wicks,
            data: gpu_data,
            device_data,
        }
    }

    /// Apply `Y = (F + \lambda S)x` using GPU-resident factor generation and contractions.
    /// The intended arithmetic is `Y^Q += F^alpha D (S^beta)^T
    /// + S^alpha D (F^beta + \lambda S^beta)^T` for A-first blocks and the corresponding
    /// beta-first expression for B-first blocks.
    /// # Arguments:
    /// - `x`: Source vector over actual candidate determinants.
    /// - `data`: Shared NOCI data used by same-parent orthogonal blocks.
    /// - `fock`: Current generalised-Fock data used by same-parent orthogonal blocks.
    /// - `lambda`: Scalar shift multiplying the overlap operator.
    /// - `partition`: Worker index and worker count for target rows.
    /// # Returns
    /// - `Array1<T>`: Partial or complete determinant-space result vector.
    pub(crate) fn apply_one_body(
        &mut self,
        _x: &Array1<T>,
        _data: &NOCIData<'_, T>,
        _fock: &FockData<'_, T>,
        _lambda: T,
        _partition: (usize, usize),
    ) -> Array1<T> {
        eprintln!(
            "CubeCL GPU one-body orchestration for runtime '{}' is not complete: nonorthogonal kernels are present, but panel contraction launch wiring remains unsupported on this boundary",
            self.context.runtime_name()
        );
        std::process::exit(1);
    }

    /// Build diagonal entries of `F + \lambda S` and `S` using GPU one-body arithmetic.
    /// # Arguments:
    /// - `data`: Shared NOCI data used by same-parent orthogonal blocks.
    /// - `fock`: Current generalised-Fock data used by same-parent orthogonal blocks.
    /// - `lambda`: Scalar overlap shift in `F + \lambda S`.
    /// # Returns
    /// - `(Array1<T>, Array1<T>)`: Diagonal of `F + \lambda S` and diagonal of `S`.
    pub(crate) fn one_body_diagonals(
        &mut self,
        _data: &NOCIData<'_, T>,
        _fock: &FockData<'_, T>,
        _lambda: T,
    ) -> (Array1<T>, Array1<T>) {
        eprintln!(
            "CubeCL GPU one-body diagonal orchestration for runtime '{}' is not complete",
            self.context.runtime_name()
        );
        std::process::exit(1);
    }
}

/// Upload host-packed Wick storage after the caller has proven `T = f64`.
/// # Arguments:
/// - `wicks`: Host-packed Wick storage with real scalar layout.
/// - `context`: CubeCL context owning the target device.
/// # Returns
/// - `DeviceWicksShared`: Device Wick buffers.
unsafe fn upload_real_wicks<T: NOCIScalar>(
    wicks: &gpu::WicksShared<T>,
    context: &GpuContext,
) -> DeviceWicksShared {
    let real = &*(wicks as *const gpu::WicksShared<T> as *const gpu::WicksShared<f64>);
    real.upload_f64(context)
}
