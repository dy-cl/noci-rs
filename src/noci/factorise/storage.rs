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
