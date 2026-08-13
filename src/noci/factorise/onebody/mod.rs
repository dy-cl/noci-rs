// noci/factorise/onebody/mod.rs
//! Backend dispatch for factorised one-body NOCI operator contractions.

pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;

mod plan;

// Crate-visible type re-exports.
pub(crate) use plan::{OneBodyBlockPlan, OneBodyContraction, OneBodyPlan};

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::input::{SNOCIBackend, SNOCIStorage};
use crate::noci::types::{FockData, NOCIData, NOCIScalar};

// Parent/sibling imports.
use self::cpu::CpuOneBodyBackend;
#[cfg(feature = "gpu")]
use self::gpu::GpuOneBodyBackend;

/// Factorised one-body backend selected for the current SNOCI solve.
pub(crate) enum OneBodyBackend<T: NOCIScalar> {
    /// CPU implementation with Rayon and host factor storage.
    CPU(CpuOneBodyBackend<T>),
    /// CubeCL implementation for the selected GPU runtime.
    #[cfg(feature = "gpu")]
    GPU(GpuOneBodyBackend<T>),
}

impl<T: NOCIScalar + 'static> OneBodyBackend<T> {
    /// Build a factorised one-body backend for the current generalised Fock operator.
    /// # Arguments:
    /// - `backend`: Runtime backend requested by input.
    /// - `data`: Shared NOCI data with Wick intermediates for the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data, already reflected in Wick intermediates.
    /// - `cache`: Directory for persistent file-backed factor blocks.
    /// - `rank`: MPI rank used in factor-cache filenames.
    /// - `iteration`: SNOCI iteration used in factor-cache filenames.
    /// - `storage`: Requested persistent factor-table storage backend.
    /// # Returns
    /// - `OneBodyBackend<T>`: Backend-specific factorised one-body operator.
    pub(crate) fn new(
        backend: SNOCIBackend,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        cache: &std::path::Path,
        rank: i32,
        iteration: usize,
        storage: SNOCIStorage,
    ) -> Self {
        match backend {
            SNOCIBackend::CPU => Self::CPU(CpuOneBodyBackend::new(
                data, fock, cache, rank, iteration, storage,
            )),
            SNOCIBackend::GPU => Self::new_gpu(data, fock, cache, rank, iteration, storage),
        }
    }

    /// Count raw factor storage bytes for `S^{alpha}`, `F^{alpha}`, `S^{beta}` and `F^{beta}`.
    /// Same-parent orthogonal Slater-Condon blocks require no dense factor storage.
    /// # Arguments:
    /// - `data`: Shared NOCI data defining the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data used to identify orthogonal same-parent blocks.
    /// # Returns
    /// - `usize`: Number of bytes required to store all nonorthogonal raw factor tables.
    pub(crate) fn storage_bytes(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> usize {
        CpuOneBodyBackend::storage_bytes(data, fock)
    }

    /// Apply `Y = (F + \lambda S)x` using the selected backend.
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
        x: &Array1<T>,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        lambda: T,
        partition: (usize, usize),
    ) -> Array1<T> {
        match self {
            Self::CPU(backend) => backend.apply_one_body(x, data, fock, lambda, partition),
            #[cfg(feature = "gpu")]
            Self::GPU(backend) => backend.apply_one_body(x, data, fock, lambda, partition),
        }
    }

    /// Build diagonal entries of `F + \lambda S` and `S` using the selected backend.
    /// # Arguments:
    /// - `data`: Shared NOCI data used by same-parent orthogonal blocks.
    /// - `fock`: Current generalised-Fock data used by same-parent orthogonal blocks.
    /// - `lambda`: Scalar overlap shift in `F + \lambda S`.
    /// # Returns
    /// - `(Array1<T>, Array1<T>)`: Diagonal of `F + \lambda S` and diagonal of `S`.
    pub(crate) fn one_body_diagonals(
        &mut self,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        lambda: T,
    ) -> (Array1<T>, Array1<T>) {
        match self {
            Self::CPU(backend) => backend.one_body_diagonals(data, fock, lambda),
            #[cfg(feature = "gpu")]
            Self::GPU(backend) => backend.one_body_diagonals(data, fock, lambda),
        }
    }

    /// Print backend memory-configuration diagnostics when the selected backend has them.
    /// # Returns
    /// - `()`: Writes backend diagnostics to standard output.
    pub(crate) fn report_memory_configuration(&self) {
        match self {
            Self::CPU(_) => {}
            #[cfg(feature = "gpu")]
            Self::GPU(backend) => backend.report_memory_configuration(),
        }
    }

    /// Build a GPU backend or terminate clearly when the executable lacks GPU support.
    /// # Arguments:
    /// - `data`: Shared NOCI data with Wick intermediates for the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data, already reflected in Wick intermediates.
    /// - `cache`: Directory for persistent file-backed factor blocks.
    /// - `rank`: MPI rank used in factor-cache filenames.
    /// - `iteration`: SNOCI iteration used in factor-cache filenames.
    /// - `storage`: Requested persistent factor-table storage backend.
    /// # Returns
    /// - `OneBodyBackend<T>`: GPU one-body backend.
    #[cfg(feature = "gpu")]
    fn new_gpu(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        cache: &std::path::Path,
        rank: i32,
        iteration: usize,
        storage: SNOCIStorage,
    ) -> Self {
        Self::GPU(GpuOneBodyBackend::new(
            data, fock, cache, rank, iteration, storage,
        ))
    }

    /// Report a clear fatal error when GPU is requested without a GPU build.
    /// # Arguments:
    /// - `data`: Shared NOCI data with Wick intermediates for the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data, already reflected in Wick intermediates.
    /// - `cache`: Directory for persistent file-backed factor blocks.
    /// - `rank`: MPI rank used in factor-cache filenames.
    /// - `iteration`: SNOCI iteration used in factor-cache filenames.
    /// - `storage`: Requested persistent factor-table storage backend.
    /// # Returns
    /// - `OneBodyBackend<T>`: This function exits instead of returning.
    #[cfg(not(feature = "gpu"))]
    fn new_gpu(
        _data: &NOCIData<'_, T>,
        _fock: &FockData<'_, T>,
        _cache: &std::path::Path,
        _rank: i32,
        _iteration: usize,
        _storage: SNOCIStorage,
    ) -> Self {
        eprintln!(
            "snoci.backend = \"gpu\" requires rebuilding with one of --features gpu-cuda, --features gpu-hip, or --features gpu-vulkan"
        );
        std::process::exit(1);
    }
}
