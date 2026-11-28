// elements/orthogonal/device.rs
//! `DeviceOrthogonal`, the GPU counterpart of the `MOCache` and `FockMOCache` integrals the
//! orthogonal Slater-Condon evaluators read: for every parent in an orthonormal frame, its Fock
//! matrices and, when its Hamiltonian cache is orthonormal too, its one-electron matrices and
//! antisymmetrised and Coulomb two-electron tensors, copied into one device array. The kernels that
//! read them are in `eval/device.cuh`. The Fock matrices follow the current generalised Fock, so a
//! new `DeviceOrthogonal` is built whenever the Fock changes. Real scalars only.

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, DriverError};

// Crate-root imports.
use crate::NOCIScalar;

// Parent/sibling imports.
use super::super::{FockData, NOCIData};

/// Offsets of one parent's integrals in the device slab. Its layout is fixed so the CUDA struct of
/// the same name in `eval/device.cuh` reads it field by field; every field is eight bytes wide.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct DeviceOrthogonalParent {
    /// One when the Fock matrices are in an orthonormal frame and copied.
    fock: u64,
    /// One when the Hamiltonian integrals are in an orthonormal frame and copied.
    hamiltonian: u64,
    /// Number of molecular orbitals.
    n: u64,
    /// Offset of the alpha Fock matrix, row-major `n x n`.
    fa: u64,
    /// Offset of the beta Fock matrix, row-major `n x n`.
    fb: u64,
    /// Offset of the alpha one-electron matrix, row-major `n x n`.
    ha: u64,
    /// Offset of the beta one-electron matrix, row-major `n x n`.
    hb: u64,
    /// Offset of the antisymmetrised alpha-alpha tensor `\langle pq||rs\rangle`, row-major `n^4`.
    aa: u64,
    /// Offset of the antisymmetrised beta-beta tensor, row-major `n^4`.
    bb: u64,
    /// Offset of the alpha-beta Coulomb tensor `(pq|rs)`, row-major `n^4`.
    ab: u64,
}

// Plain data with a C layout, so the driver may copy it byte for byte.
unsafe impl DeviceRepr for DeviceOrthogonalParent {}

/// Orthogonal Slater-Condon integrals of every orthonormal parent on a GPU.
pub(crate) struct DeviceOrthogonal {
    /// Whether each parent's Fock elements are evaluated by the orthogonal rules on the GPU.
    pub(crate) fock: Vec<bool>,
    /// Whether each parent's Hamiltonian and overlap elements are evaluated by the orthogonal rules
    /// on the GPU.
    pub(crate) hamiltonian: Vec<bool>,
    /// Record of each parent.
    pub(crate) parents: CudaSlice<DeviceOrthogonalParent>,
    /// Every copied matrix and tensor, located by the records.
    pub(crate) integrals: CudaSlice<f64>,
}

impl DeviceOrthogonal {
    /// Copy the integrals of every orthonormal parent.
    /// # Arguments:
    /// - `stream`: Stream of the GPU to copy to.
    /// - `data`: Shared NOCI data with the Hamiltonian MO caches, if any.
    /// - `fock`: Current generalised-Fock data with the Fock MO caches.
    /// # Returns:
    /// - `Result<DeviceOrthogonal, DriverError>`: The integrals on the GPU.
    pub(crate) fn new<T: NOCIScalar>(
        stream: &Arc<CudaStream>,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> Result<Self, DriverError> {
        let mut slab: Vec<f64> = Vec::new();
        let nparent = fock.fock_mocache.len();
        let mut parents = vec![DeviceOrthogonalParent::default(); nparent];
        let mut fockflags = vec![false; nparent];
        let mut hamiltonian = vec![false; nparent];

        // Append the real parts of one array in logical order and return where they start.
        let mut push = |values: &mut dyn Iterator<Item = f64>| {
            let offset = slab.len() as u64;
            slab.extend(values);
            offset
        };

        for (p, record) in parents.iter_mut().enumerate() {
            let cache = &fock.fock_mocache[p];
            if cache.orthogonal_slater_condon {
                fockflags[p] = true;
                record.fock = 1;
                record.n = cache.fa.nrows() as u64;
                record.fa = push(&mut cache.fa.iter().map(|v| v.re()));
                record.fb = push(&mut cache.fb.iter().map(|v| v.re()));
            }

            if let Some(cache) = data.mocache.map(|mocache| &mocache[p])
                && cache.orthogonal_slater_condon
            {
                hamiltonian[p] = true;
                record.hamiltonian = 1;
                record.n = cache.ha.nrows() as u64;
                record.ha = push(&mut cache.ha.iter().map(|v| v.re()));
                record.hb = push(&mut cache.hb.iter().map(|v| v.re()));
                record.aa = push(&mut cache.eri_aa_asym.iter().map(|v| v.re()));
                record.bb = push(&mut cache.eri_bb_asym.iter().map(|v| v.re()));
                record.ab = push(&mut cache.eri_ab_coul.iter().map(|v| v.re()));
            }
        }

        Ok(Self {
            fock: fockflags,
            hamiltonian,
            parents: stream.clone_htod(&parents)?,
            integrals: stream.clone_htod(&slab)?,
        })
    }
}
