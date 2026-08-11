// noci/factorise/onebody/gpu/backend.rs
//! GPU backend for spin-factorised one-body NOCI operator contractions.

// Standard library imports.
use std::path::Path;

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::gpu::GpuContext;
use crate::input::SNOCIStorage;
use crate::noci::types::{FockData, NOCIData, NOCIScalar};
use crate::nonorthogonalwicks::{WicksRequirements, gpu};

// Parent/sibling imports.
use super::super::super::SpinFactorisation;
use super::super::plan::OneBodyPlan;
use super::data::GpuOneBodyData;

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
    /// Factorised-operator GPU topology data.
    data: GpuOneBodyData,
}

impl<T: NOCIScalar> GpuOneBodyBackend<T> {
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
        Self {
            context: GpuContext::new(),
            spin,
            plan,
            wicks,
            data: gpu_data,
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
            "CubeCL GPU one-body arithmetic for runtime '{}' is not implemented yet",
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
            "CubeCL GPU one-body diagonals for runtime '{}' are not implemented yet",
            self.context.runtime_name()
        );
        std::process::exit(1);
    }
}
