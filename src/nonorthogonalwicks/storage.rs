// nonorthogonalwicks/storage.rs
// Standard library imports.
use std::fs::{File, OpenOptions};
use std::ptr::NonNull;

// External crate imports.
use memmap2::{Mmap, MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};

// Crate-root imports.
use crate::mpiutils::Sharedffi;
use crate::noci::NOCIScalar;

// Parent/sibling imports.
use super::types::{PairMeta, PairOffset};
use super::view::WicksView;

/// Backing allocation used by the contiguous slab of precomputed nonorthogonal Wick intermediates.
#[allow(dead_code)]
pub(crate) enum WicksBacking<T: NOCIScalar> {
    /// Node-shared RMA allocation.
    Shared(WicksRma<T>),
    /// Read-only file-backed memory map.
    Mmap(Mmap),
    /// Writable file-backed memory map.
    MmapCow(MmapMut),
}

/// Owner of the backing allocation and its read-only address, offset and metadata view.
/// Keeping both objects together guarantees that the raw slab pointer in `WicksView` remains
/// valid for the lifetime of this storage object.
pub struct WicksShared<T: NOCIScalar> {
    /// Backing allocation for the contiguous Wick tensor slab.
    backing: WicksBacking<T>,
    /// View over the tensor slab and per-reference-pair metadata.
    view: WicksView<T>,
}

impl<T: NOCIScalar> WicksShared<T> {
    /// Borrow the read-only view over the stored Wick intermediates.
    /// # Arguments:
    /// - `self`: Wick storage and associated view.
    /// # Returns
    /// - `&WicksView<T>`: Shared view over the tensor slab.
    pub fn view(&self) -> &WicksView<T> {
        &self.view
    }

    /// Mutably borrow the view metadata without changing the ownership of the backing allocation.
    /// # Arguments:
    /// - `self`: Wick storage and associated view.
    /// # Returns
    /// - `&mut WicksView<T>`: Mutable borrow of the slab view and metadata.
    pub fn view_mut(&mut self) -> &mut WicksView<T> {
        &mut self.view
    }

    /// Return a mutable slice over the full contiguous tensor slab. This is used when initially
    /// writing intermediates or when overwriting generalised-Fock intermediates in place.
    /// # Arguments:
    /// - `self`: Writable Wick storage and associated view.
    /// # Returns
    /// - `&mut [T]`: Mutable slice over the full tensor slab.
    /// # Panics
    /// - Panics when the backing allocation is a read-only memory map.
    pub fn slab_mut(&mut self) -> &mut [T] {
        let ptr = self.base_mut_ptr();
        let len = self.view.slab_len;
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Return a mutable pointer to the first entry of the contiguous tensor slab.
    /// # Arguments:
    /// - `self`: Writable Wick storage and associated view.
    /// # Returns
    /// - `*mut T`: Mutable pointer to the first slab entry.
    /// # Panics
    /// - Panics when the backing allocation is a read-only memory map.
    fn base_mut_ptr(&mut self) -> *mut T {
        match &mut self.backing {
            WicksBacking::Shared(rma) => rma.base_ptr as *mut T,
            WicksBacking::Mmap(_) => panic!("Wick's slab is read-only"),
            WicksBacking::MmapCow(map) => map.as_mut_ptr() as *mut T,
        }
    }

    /// Flush a writable file-backed tensor slab to disk. Shared-memory and read-only mappings
    /// require no explicit flush through this interface.
    /// # Arguments:
    /// - `self`: Wick storage and associated view.
    /// # Returns
    /// - `std::io::Result<()>`: Success when no flush is required or the writable map is flushed.
    pub(crate) fn flush_mmap(&mut self) -> std::io::Result<()> {
        match &mut self.backing {
            WicksBacking::MmapCow(map) => map.flush(),
            _ => Ok(()),
        }
    }

    /// Construct storage backed by a node-shared RMA allocation.
    /// # Arguments:
    /// - `rma`: Shared-memory backing allocation.
    /// - `view`: View over the contiguous Wick tensor slab.
    /// # Returns
    /// - `WicksShared<T>`: Shared-memory-backed Wick storage.
    pub(crate) fn from_shared(
        rma: WicksRma<T>,
        view: WicksView<T>,
    ) -> Self {
        Self {
            backing: WicksBacking::Shared(rma),
            view,
        }
    }

    /// Construct storage backed by a read-only file mapping.
    /// # Arguments:
    /// - `mmap`: Read-only memory map containing the Wick tensor slab.
    /// - `view`: View over the mapped tensor slab.
    /// # Returns
    /// - `WicksShared<T>`: Read-only file-backed Wick storage.
    pub(crate) fn from_mmap(
        mmap: Mmap,
        view: WicksView<T>,
    ) -> Self {
        Self {
            backing: WicksBacking::Mmap(mmap),
            view,
        }
    }

    /// Construct storage backed by a writable file mapping.
    /// # Arguments:
    /// - `mmap`: Writable memory map containing the Wick tensor slab.
    /// - `view`: View over the mapped tensor slab.
    /// # Returns
    /// - `WicksShared<T>`: Writable file-backed Wick storage.
    pub(crate) fn from_mmap_cow(
        mmap: MmapMut,
        view: WicksView<T>,
    ) -> Self {
        Self {
            backing: WicksBacking::MmapCow(mmap),
            view,
        }
    }
}

/// Shared-memory backing for the contiguous Wick tensor slab.
pub(crate) struct WicksRma<T: NOCIScalar> {
    /// Allocation handle retained to keep the node-shared memory region alive.
    pub(crate) _shared: Sharedffi,
    /// Raw byte pointer to the first entry of the shared tensor slab.
    pub(crate) base_ptr: *mut u8,
    /// Total size of the shared allocation in bytes.
    pub(crate) _nbytes: usize,
    /// Marker for the scalar type represented by the raw allocation.
    pub(crate) _marker: std::marker::PhantomData<T>,
}

/// Serialised metadata required to reconstruct a file-backed `WicksView` over a raw tensor slab.
#[derive(Serialize, Deserialize)]
#[serde(bound = "T: NOCIScalar")]
pub(crate) struct WicksDiskMeta<T: NOCIScalar> {
    /// On-disk metadata format version.
    pub(crate) version: u32,
    /// Number of reference determinants.
    pub(crate) nref: usize,
    /// Total slab length in units of `T`.
    pub(crate) slab_len: usize,
    /// Per-reference-pair offset tables into the contiguous tensor slab.
    pub(crate) off: Vec<PairOffset>,
    /// Per-reference-pair scalar metadata stored outside the tensor slab.
    pub(crate) meta: Vec<PairMeta<T>>,
}

/// Create a writable file-backed memory map for the contiguous Wick tensor slab. The associated
/// scalar metadata is initialised in memory and can be serialised separately once construction is complete.
/// # Arguments:
/// - `slab_path`: Path to the raw file storing the contiguous tensor slab.
/// - `nref`: Number of reference determinants.
/// - `off`: Per-reference-pair offset tables into the tensor slab.
/// - `slab_len`: Total slab length in units of `T`.
/// # Returns
/// - `std::io::Result<WicksShared<T>>`: Writable file-backed Wick storage and slab view.
pub(crate) fn create_wicks_mmap<T: NOCIScalar>(
    slab_path: &std::path::Path,
    nref: usize,
    off: Vec<PairOffset>,
    slab_len: usize,
) -> std::io::Result<WicksShared<T>> {
    let nbytes = slab_len * std::mem::size_of::<T>();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(slab_path)?;
    file.set_len(nbytes as u64)?;

    let mut mmap = unsafe { MmapOptions::new().len(nbytes).map_mut(&file)? };
    let ptr = mmap.as_mut_ptr() as *mut T;
    let slab = unsafe { std::slice::from_raw_parts_mut(ptr, slab_len) };
    slab.fill(<T as From<f64>>::from(f64::NAN));

    let meta = vec![PairMeta::<T>::default(); nref * nref];
    let view = WicksView {
        slab: NonNull::new(ptr).unwrap(),
        slab_len,
        nref,
        off,
        meta,
    };

    Ok(WicksShared::from_mmap_cow(mmap, view))
}

/// Load a read-only file-backed tensor slab together with its serialised offsets and scalar metadata.
/// # Arguments:
/// - `slab_path`: Path to the raw file storing the contiguous tensor slab.
/// - `meta_path`: Path to the serialised `WicksDiskMeta` object.
/// # Returns
/// - `std::io::Result<WicksShared<T>>`: Read-only file-backed Wick storage and slab view.
pub(crate) fn load_wicks_mmap<T: NOCIScalar>(
    slab_path: &std::path::Path,
    meta_path: &std::path::Path,
) -> std::io::Result<WicksShared<T>> {
    let disk_meta: WicksDiskMeta<T> = bincode::deserialize(&std::fs::read(meta_path)?).unwrap();
    let file = File::open(slab_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let ptr = mmap.as_ptr() as *mut T;

    let view = WicksView {
        slab: NonNull::new(ptr).unwrap(),
        slab_len: disk_meta.slab_len,
        nref: disk_meta.nref,
        off: disk_meta.off,
        meta: disk_meta.meta,
    };

    Ok(WicksShared::from_mmap(mmap, view))
}
