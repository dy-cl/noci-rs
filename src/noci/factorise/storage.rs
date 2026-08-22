// noci/factorise/storage.rs
//! Backing storage for persistent factorised NOCI operator tables.

// Standard library imports.
use std::fs::{OpenOptions, remove_file};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

// External crate imports.
use memmap2::{MmapMut, MmapOptions};

// Crate-root imports.
use crate::input::SNOCIStorage;
use crate::noci::types::NOCIScalar;

/// Mutable storage policy state for assigning factor blocks to RAM or disk.
pub(super) struct OneBodyStoragePlan {
    /// Directory for file-backed factor blocks.
    cache: PathBuf,
    /// MPI rank used to make factor filenames unique.
    rank: i32,
    /// SNOCI iteration used to make factor filenames unique.
    iteration: usize,
    /// Requested factor-table storage backend.
    storage: SNOCIStorage,
}

impl OneBodyStoragePlan {
    /// Construct a persistent factor storage plan for one SNOCI iteration.
    /// # Arguments:
    /// - `cache`: Directory for file-backed factor blocks.
    /// - `rank`: MPI rank used in filenames.
    /// - `iteration`: SNOCI iteration used in filenames.
    /// - `storage`: Requested factor-table storage backend.
    /// # Returns
    /// - `OneBodyStoragePlan`: Storage allocator with the requested backend.
    pub(super) fn new(
        cache: &Path,
        rank: i32,
        iteration: usize,
        storage: SNOCIStorage,
    ) -> Self {
        Self {
            cache: cache.to_path_buf(),
            rank,
            iteration,
            storage,
        }
    }

    /// Allocate `S^{alpha}`, `F^{alpha}`, `S^{beta}` and `F^{beta}` storage for one block.
    /// The requested SNOCI storage backend determines whether the raw factor tables live in RAM
    /// or in a writable file-backed map.
    /// # Arguments:
    /// - `target_parent`: Target parent `Q`.
    /// - `source_parent`: Source parent `P`.
    /// - `na`: Number of alpha factor entries.
    /// - `nb`: Number of beta factor entries.
    /// # Returns
    /// - `OneBodyFactorStorage<T>`: RAM or disk storage for the raw factor tables.
    pub(super) fn allocate<T: NOCIScalar>(
        &mut self,
        target_parent: usize,
        source_parent: usize,
        na: usize,
        nb: usize,
    ) -> OneBodyFactorStorage<T> {
        match self.storage {
            SNOCIStorage::RAM => OneBodyFactorStorage::Ram(OneBodyRamFactors {
                sa: vec![T::from_real(0.0); na],
                fa: vec![T::from_real(0.0); na],
                sb: vec![T::from_real(0.0); nb],
                fb: vec![T::from_real(0.0); nb],
            }),
            SNOCIStorage::Disk => {
                let path = self.cache.join(format!(
                    "snoci_factors_rank{}_iter{}_q{}_p{}.bin",
                    self.rank, self.iteration, target_parent, source_parent
                ));
                OneBodyFactorStorage::Disk(OneBodyDiskFactors::create(&path, na, nb))
            }
            SNOCIStorage::None => {
                panic!("factor table storage must be 'ram' or 'disk' when factors are built")
            }
        }
    }
}

/// Mutable storage policy state for assigning overlap factor blocks to RAM or disk.
pub(super) struct OverlapStoragePlan {
    /// Directory for file-backed factor blocks.
    cache: PathBuf,
    /// MPI rank used to make factor filenames unique.
    rank: i32,
    /// Requested factor-table storage backend.
    storage: SNOCIStorage,
}

impl OverlapStoragePlan {
    /// Construct a persistent overlap-factor storage plan for one QMC run.
    /// # Arguments:
    /// - `cache`: Directory for file-backed factor blocks.
    /// - `rank`: MPI rank used in filenames.
    /// - `storage`: Requested factor-table storage backend.
    /// # Returns
    /// - `OverlapStoragePlan`: Storage allocator with the requested backend.
    pub(super) fn new(
        cache: &Path,
        rank: i32,
        storage: SNOCIStorage,
    ) -> Self {
        Self {
            cache: cache.to_path_buf(),
            rank,
            storage,
        }
    }

    /// Allocate `A^{QP}` and `B^{QP}` overlap factors and optional source-major CDF storage.
    /// # Arguments:
    /// - `target_parent`: Target parent `Q`.
    /// - `source_parent`: Source parent `P`.
    /// - `na`: Number of alpha factor entries.
    /// - `nb`: Number of beta factor entries.
    /// - `build_cdfs`: Whether to allocate alpha and beta CDF tables.
    /// # Returns
    /// - `OverlapFactorStorage`: RAM or disk storage for the overlap factor tables.
    pub(super) fn allocate(
        &mut self,
        target_parent: usize,
        source_parent: usize,
        na: usize,
        nb: usize,
        build_cdfs: bool,
    ) -> OverlapFactorStorage {
        match self.storage {
            SNOCIStorage::RAM => OverlapFactorStorage::Ram(OverlapRamFactors {
                afac: vec![0.0; na],
                bfac: vec![0.0; nb],
                acdf: if build_cdfs {
                    vec![0.0; na]
                } else {
                    Vec::new()
                },
                bcdf: if build_cdfs {
                    vec![0.0; nb]
                } else {
                    Vec::new()
                },
            }),
            SNOCIStorage::Disk => {
                let path = self.cache.join(format!(
                    "qmc_overlap_factors_rank{}_q{}_p{}.bin",
                    self.rank, target_parent, source_parent
                ));
                OverlapFactorStorage::Disk(OverlapDiskFactors::create(&path, na, nb, build_cdfs))
            }
            SNOCIStorage::None => {
                panic!("factor table storage must be 'ram' or 'disk' when factors are built")
            }
        }
    }
}

/// Raw one-body factor table backing for one parent-pair block.
pub(super) enum OneBodyFactorStorage<T: NOCIScalar> {
    /// Anonymous RAM vectors for all raw factors.
    Ram(OneBodyRamFactors<T>),
    /// Writable file-backed memory map for all raw factors.
    Disk(OneBodyDiskFactors<T>),
}

impl<T: NOCIScalar> OneBodyFactorStorage<T> {
    /// Borrow `S^{alpha,QP}`, `F^{alpha,QP}`, `S^{beta,QP}` and `F^{beta,QP}` as row-major slices.
    /// # Arguments:
    /// - `self`: Raw factor storage.
    /// # Returns
    /// - `(&[T], &[T], &[T], &[T])`: Row-major raw factor tables.
    pub(super) fn factors(&self) -> (&[T], &[T], &[T], &[T]) {
        match self {
            Self::Ram(storage) => (
                storage.sa.as_slice(),
                storage.fa.as_slice(),
                storage.sb.as_slice(),
                storage.fb.as_slice(),
            ),
            Self::Disk(storage) => {
                let len = 2usize
                    .checked_mul(
                        storage
                            .na
                            .checked_add(storage.nb)
                            .expect("one-body factor length overflow"),
                    )
                    .expect("one-body factor length overflow");
                let full = unsafe {
                    // The map was allocated to `len * size_of::<T>()`, and every returned slice is
                    // bounded by the stored alpha/beta factor lengths.
                    std::slice::from_raw_parts(storage.map.as_ptr() as *const T, len)
                };
                let (sa, rest) = full.split_at(storage.na);
                let (fa, rest) = rest.split_at(storage.na);
                let (sb, fb) = rest.split_at(storage.nb);
                (sa, fa, sb, fb)
            }
        }
    }

    /// Mutably borrow `S^{alpha,QP}` and `F^{alpha,QP}` as disjoint row-major slices.
    /// # Arguments:
    /// - `self`: Raw factor storage.
    /// # Returns
    /// - `(&mut [T], &mut [T])`: Row-major alpha overlap and Fock factor entries.
    pub(super) fn alpha_mut(&mut self) -> (&mut [T], &mut [T]) {
        match self {
            Self::Ram(storage) => (storage.sa.as_mut_slice(), storage.fa.as_mut_slice()),
            Self::Disk(storage) => {
                let len = 2usize
                    .checked_mul(
                        storage
                            .na
                            .checked_add(storage.nb)
                            .expect("one-body factor length overflow"),
                    )
                    .expect("one-body factor length overflow");
                let full = unsafe {
                    // The map was allocated to `len * size_of::<T>()`, and this mutable borrow is
                    // unique through `&mut self`.
                    std::slice::from_raw_parts_mut(storage.map.as_mut_ptr() as *mut T, len)
                };
                let (sa, rest) = full.split_at_mut(storage.na);
                let (fa, _) = rest.split_at_mut(storage.na);
                (sa, fa)
            }
        }
    }

    /// Mutably borrow `S^{beta,QP}` and `F^{beta,QP}` as disjoint row-major slices.
    /// # Arguments:
    /// - `self`: Raw factor storage.
    /// # Returns
    /// - `(&mut [T], &mut [T])`: Row-major beta overlap and Fock factor entries.
    pub(super) fn beta_mut(&mut self) -> (&mut [T], &mut [T]) {
        match self {
            Self::Ram(storage) => (storage.sb.as_mut_slice(), storage.fb.as_mut_slice()),
            Self::Disk(storage) => {
                let len = 2usize
                    .checked_mul(
                        storage
                            .na
                            .checked_add(storage.nb)
                            .expect("one-body factor length overflow"),
                    )
                    .expect("one-body factor length overflow");
                let full = unsafe {
                    // The map was allocated to `len * size_of::<T>()`, and this mutable borrow is
                    // unique through `&mut self`.
                    std::slice::from_raw_parts_mut(storage.map.as_mut_ptr() as *mut T, len)
                };
                let start = 2usize
                    .checked_mul(storage.na)
                    .expect("one-body factor offset overflow");
                let (_, beta) = full.split_at_mut(start);
                let (sb, rest) = beta.split_at_mut(storage.nb);
                let (fb, _) = rest.split_at_mut(storage.nb);
                (sb, fb)
            }
        }
    }

    /// Flush file-backed raw factors when this storage uses a writable map.
    /// # Arguments:
    /// - `self`: Raw factor storage.
    /// # Returns
    /// - `()`: Completes any required flush or panics on filesystem failure.
    pub(super) fn flush(&mut self) {
        if let Self::Disk(storage) = self {
            storage
                .map
                .flush()
                .expect("failed to flush one-body factor map");
        }
    }
}

/// Raw overlap factor table backing for one parent-pair block.
pub(super) enum OverlapFactorStorage {
    /// Anonymous RAM vectors for overlap factors and optional CDFs.
    Ram(OverlapRamFactors),
    /// Writable file-backed memory map for overlap factors and optional CDFs.
    Disk(OverlapDiskFactors),
}

impl OverlapFactorStorage {
    /// Borrow `A^{QP}`, `B^{QP}` and optional alpha and beta CDFs as row-major slices.
    /// # Arguments:
    /// - `self`: Raw overlap factor storage.
    /// # Returns
    /// - `(&[f64], &[f64], &[f64], &[f64])`: Alpha factors, beta factors and optional CDFs.
    pub(super) fn factors(&self) -> (&[f64], &[f64], &[f64], &[f64]) {
        match self {
            Self::Ram(storage) => (
                storage.afac.as_slice(),
                storage.bfac.as_slice(),
                storage.acdf.as_slice(),
                storage.bcdf.as_slice(),
            ),
            Self::Disk(storage) => {
                let nfactor = storage
                    .na
                    .checked_add(storage.nb)
                    .expect("overlap factor length overflow");
                let len = if storage.build_cdfs {
                    2usize
                        .checked_mul(nfactor)
                        .expect("overlap factor length overflow")
                } else {
                    nfactor
                };
                let full = unsafe {
                    // The map was allocated to `len * size_of::<f64>()`, and every returned slice
                    // is bounded by the stored alpha/beta factor lengths.
                    std::slice::from_raw_parts(storage.map.as_ptr() as *const f64, len)
                };
                let (afac, rest) = full.split_at(storage.na);
                let (bfac, rest) = rest.split_at(storage.nb);
                if storage.build_cdfs {
                    let (acdf, bcdf) = rest.split_at(storage.na);
                    (afac, bfac, acdf, bcdf)
                } else {
                    (afac, bfac, &[], &[])
                }
            }
        }
    }

    /// Mutably borrow `A^{QP}`, `B^{QP}` and optional alpha and beta CDFs as row-major slices.
    /// # Arguments:
    /// - `self`: Raw overlap factor storage.
    /// # Returns
    /// - `(&mut [f64], &mut [f64], &mut [f64], &mut [f64])`: Mutable factor and CDF tables.
    pub(super) fn factors_mut(&mut self) -> (&mut [f64], &mut [f64], &mut [f64], &mut [f64]) {
        match self {
            Self::Ram(storage) => (
                storage.afac.as_mut_slice(),
                storage.bfac.as_mut_slice(),
                storage.acdf.as_mut_slice(),
                storage.bcdf.as_mut_slice(),
            ),
            Self::Disk(storage) => {
                let nfactor = storage
                    .na
                    .checked_add(storage.nb)
                    .expect("overlap factor length overflow");
                let len = if storage.build_cdfs {
                    2usize
                        .checked_mul(nfactor)
                        .expect("overlap factor length overflow")
                } else {
                    nfactor
                };
                let full = unsafe {
                    // The map was allocated to `len * size_of::<f64>()`, and this mutable borrow
                    // is unique through `&mut self`.
                    std::slice::from_raw_parts_mut(storage.map.as_mut_ptr() as *mut f64, len)
                };
                let (afac, rest) = full.split_at_mut(storage.na);
                let (bfac, rest) = rest.split_at_mut(storage.nb);
                if storage.build_cdfs {
                    let (acdf, bcdf) = rest.split_at_mut(storage.na);
                    (afac, bfac, acdf, bcdf)
                } else {
                    let (acdf, bcdf) = rest.split_at_mut(0);
                    (afac, bfac, acdf, bcdf)
                }
            }
        }
    }

    /// Flush file-backed overlap factors when this storage uses a writable map.
    /// # Arguments:
    /// - `self`: Raw overlap factor storage.
    /// # Returns
    /// - `()`: Completes any required flush or panics on filesystem failure.
    pub(super) fn flush(&mut self) {
        if let Self::Disk(storage) = self {
            storage
                .map
                .flush()
                .expect("failed to flush overlap factor map");
        }
    }
}

/// Anonymous RAM storage for the four raw one-body factor tables.
pub(super) struct OneBodyRamFactors<T: NOCIScalar> {
    /// Row-major `S^{alpha,QP}` entries.
    sa: Vec<T>,
    /// Row-major `F^{alpha,QP}` entries.
    fa: Vec<T>,
    /// Row-major `S^{beta,QP}` entries.
    sb: Vec<T>,
    /// Row-major `F^{beta,QP}` entries.
    fb: Vec<T>,
}

/// Anonymous RAM storage for overlap factor tables and optional proposal CDFs.
pub(super) struct OverlapRamFactors {
    /// Row-major `A^{QP}` entries.
    afac: Vec<f64>,
    /// Row-major `B^{QP}` entries.
    bfac: Vec<f64>,
    /// Source-major alpha CDF entries, or an empty vector when not requested.
    acdf: Vec<f64>,
    /// Source-major beta CDF entries, or an empty vector when not requested.
    bcdf: Vec<f64>,
}

/// Writable file-backed storage for the four raw one-body factor tables.
pub(super) struct OneBodyDiskFactors<T: NOCIScalar> {
    /// Path removed when the factor storage is dropped.
    path: PathBuf,
    /// Writable memory map containing `S^\alpha`, `F^\alpha`, `S^\beta`, and `F^\beta`.
    map: MmapMut,
    /// Number of alpha factor entries.
    na: usize,
    /// Number of beta factor entries.
    nb: usize,
    /// Scalar marker for typed access to the byte map.
    marker: PhantomData<T>,
}

impl<T: NOCIScalar> OneBodyDiskFactors<T> {
    /// Create a writable file-backed raw factor map with checked byte layout.
    /// # Arguments:
    /// - `path`: File path for the raw factor map.
    /// - `na`: Number of alpha factor entries.
    /// - `nb`: Number of beta factor entries.
    /// # Returns
    /// - `OneBodyDiskFactors<T>`: Writable file-backed raw factor storage.
    fn create(
        path: &Path,
        na: usize,
        nb: usize,
    ) -> Self {
        let nentries = 2usize
            .checked_mul(na.checked_add(nb).expect("one-body factor length overflow"))
            .expect("one-body factor length overflow");
        let nbytes = nentries
            .checked_mul(std::mem::size_of::<T>())
            .expect("one-body factor byte size overflow");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .expect("failed to create one-body factor file");
        file.set_len(nbytes as u64)
            .expect("failed to size one-body factor file");
        let map = unsafe {
            // The file was resized to exactly `nbytes`, and all typed slices below stay within
            // this mapping. `MmapMut` is page-aligned, which is sufficient for `T`.
            MmapOptions::new()
                .len(nbytes)
                .map_mut(&file)
                .expect("failed to map one-body factor file")
        };

        Self {
            path: path.to_path_buf(),
            map,
            na,
            nb,
            marker: PhantomData,
        }
    }
}

impl<T: NOCIScalar> Drop for OneBodyDiskFactors<T> {
    /// Remove the temporary factor file after the mapping is destroyed.
    fn drop(&mut self) {
        let _ = remove_file(&self.path);
    }
}

/// Writable file-backed storage for overlap factor tables and optional proposal CDFs.
pub(super) struct OverlapDiskFactors {
    /// Path removed when the factor storage is dropped.
    path: PathBuf,
    /// Writable memory map containing `A^{QP}`, `B^{QP}` and optional proposal CDFs.
    map: MmapMut,
    /// Number of alpha factor entries.
    na: usize,
    /// Number of beta factor entries.
    nb: usize,
    /// Whether alpha and beta CDF tables are present after the raw factors.
    build_cdfs: bool,
}

impl OverlapDiskFactors {
    /// Create a writable file-backed overlap factor map with checked byte layout.
    /// # Arguments:
    /// - `path`: File path for the raw factor map.
    /// - `na`: Number of alpha factor entries.
    /// - `nb`: Number of beta factor entries.
    /// - `build_cdfs`: Whether alpha and beta CDF tables are stored after the raw factors.
    /// # Returns
    /// - `OverlapDiskFactors`: Writable file-backed overlap factor storage.
    fn create(
        path: &Path,
        na: usize,
        nb: usize,
        build_cdfs: bool,
    ) -> Self {
        let nfactor = na.checked_add(nb).expect("overlap factor length overflow");
        let nentries = if build_cdfs {
            2usize
                .checked_mul(nfactor)
                .expect("overlap factor length overflow")
        } else {
            nfactor
        };
        let nbytes = nentries
            .checked_mul(std::mem::size_of::<f64>())
            .expect("overlap factor byte size overflow");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .expect("failed to create overlap factor file");
        file.set_len(nbytes as u64)
            .expect("failed to size overlap factor file");
        let map = unsafe {
            // The file was resized to exactly `nbytes`, and all typed slices below stay within
            // this mapping. `MmapMut` is page-aligned, which is sufficient for `f64`.
            MmapOptions::new()
                .len(nbytes)
                .map_mut(&file)
                .expect("failed to map overlap factor file")
        };

        Self {
            path: path.to_path_buf(),
            map,
            na,
            nb,
            build_cdfs,
        }
    }
}

impl Drop for OverlapDiskFactors {
    /// Remove the temporary overlap factor file after the mapping is destroyed.
    fn drop(&mut self) {
        let _ = remove_file(&self.path);
    }
}
