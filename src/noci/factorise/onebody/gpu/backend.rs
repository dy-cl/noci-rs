// noci/factorise/onebody/gpu/backend.rs
//! GPU backend for spin-factorised one-body NOCI operator contractions.

// Standard library imports.
use std::marker::PhantomData;
use std::path::Path;

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::input::SNOCIStorage;
use crate::noci::types::{FockData, NOCIData, NOCIScalar};

// Parent/sibling imports.
use super::data::GpuOneBodyData;
use super::runtime::runtime_name;

/// CubeCL factorised one-body backend for the current generalised Fock.
pub(crate) struct GpuOneBodyBackend<T: NOCIScalar> {
    /// Persistent GPU topology and Wick data.
    data: GpuOneBodyData<T>,
    /// Scalar marker for reusable CubeCL buffers.
    scalar: PhantomData<T>,
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
        _fock: &FockData<'_, T>,
        _cache: &Path,
        _rank: i32,
        _iteration: usize,
        storage: SNOCIStorage,
    ) -> Self {
        if !matches!(storage, SNOCIStorage::None) {
            eprintln!("snoci.backend = \"gpu\" requires snoci.gmres.factor_tables = \"none\"");
            std::process::exit(1);
        }
        Self {
            data: GpuOneBodyData::new(data),
            scalar: PhantomData,
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
            "CubeCL GPU one-body arithmetic for runtime '{}' is not enabled in this source build",
            runtime_name()
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
            "CubeCL GPU one-body diagonals for runtime '{}' are not enabled in this source build",
            runtime_name()
        );
        std::process::exit(1);
    }
}
