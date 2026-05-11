// nonorthogonalwicks/storage.rs
use std::ptr::NonNull;
use std::fs::{File, OpenOptions};

use serde::{Serialize, Deserialize};
use memmap2::{Mmap, MmapMut, MmapOptions};

use crate::mpiutils::Sharedffi;
use crate::noci::NOCIScalar;
use super::view::WicksView;
use super::types::{PairOffset, PairMeta};

#[allow(dead_code)]
pub(crate) enum WicksBacking<T: NOCIScalar> {
    Shared(WicksRma<T>),
    Mmap(Mmap),
    MmapCow(MmapMut),
}

/// Storage in which we split the Wicks data into the shared remote memory access (RMA) and a view
/// for reading said data.
pub struct WicksShared<T: NOCIScalar> {
    /// Backing storage for the contiguous Wick's tensor slab.
    backing: WicksBacking<T>,
    /// View of the tensor slab.
    view: WicksView<T>,
}

impl<T: NOCIScalar> WicksShared<T> {
    /// Get a shared reference to the WicksView object.
    /// # Arguments:
    /// - `self`: View and RMA for Wick's intermediates.
    /// # Returns
    /// - `&WicksView<T>`: Shared view of the Wick's intermediates.
    pub fn view(&self) -> &WicksView<T> {&self.view}
    
    /// Get a mutable reference to the WicksView object.
    /// # Arguments:
    /// - `self`: View and RMA for Wick's intermediates.
    /// # Returns
    /// - `&mut WicksView<T>`: Mutable view of the Wick's intermediates.
    pub fn view_mut(&mut self) -> &mut WicksView<T> {
        &mut self.view
    }
    
    /// Get a mutable slice over the full contiguous shared tensor storage.
    /// The returned slice may be used to overwrite stored matrices or tensors in place.
    /// # Arguments:
    /// - `self`: View and RMA for Wick's intermediates.
    /// # Returns
    /// - `&mut [T]`: Mutable slice over the full shared tensor slab.
    pub fn slab_mut(&mut self) -> &mut [T] {
        let ptr = self.base_mut_ptr();
        let len = self.view.slab_len;
        unsafe {std::slice::from_raw_parts_mut(ptr, len)}
    }
    
    /// Return a mutable pointer to the start of the contiguous tensor storage.
    /// # Arguments:
    /// - `self`: View and backing storage for Wick's intermediates.
    /// # Returns
    /// - `*mut T`: Mutable pointer to the start of the tensor slab.
    /// # Panics
    /// - Panics if the backing storage is read-only.
    fn base_mut_ptr(&mut self) -> *mut T {
        match &mut self.backing {
            WicksBacking::Shared(rma) => rma.base_ptr as *mut T,
            WicksBacking::Mmap(_) => panic!("Wick's slab is read-only"),
            WicksBacking::MmapCow(map) => map.as_mut_ptr() as *mut T,
        }
    }

    /// Flush a writable file-backed Wick's slab to disk if present.
    /// # Arguments:
    /// - `self`: Wick's storage and associated view.
    /// # Returns
    /// - `std::io::Result<()>`: Ok if no flush is required or if flushing succeeds.
    pub(crate) fn flush_mmap(&mut self) -> std::io::Result<()> {
        match &mut self.backing {
            WicksBacking::MmapCow(map) => map.flush(),
            _ => Ok(()),
        }
    }

    /// Construct shared-memory-backed Wick's storage.
    /// # Arguments:
    /// - `rma`: Shared-memory backing region.
    /// - `view`: View over the contiguous Wick's slab.
    /// # Returns
    /// - `WicksShared<T>`: Wick's storage and associated view.
    pub(crate) fn from_shared(rma: WicksRma<T>, view: WicksView<T>) -> Self {
        Self {backing: WicksBacking::Shared(rma), view}
    }

    /// Construct file-backed read-only Wick's storage.
    /// # Arguments:
    /// - `mmap`: Read-only memory map of the Wick's slab.
    /// - `view`: View over the contiguous Wick's slab.
    /// # Returns
    /// - `WicksShared<T>`: Wick's storage and associated view.
    pub(crate) fn from_mmap(mmap: Mmap, view: WicksView<T>) -> Self {
        Self {backing: WicksBacking::Mmap(mmap), view}
    }

    /// Construct file-backed writable Wick's storage.
    /// # Arguments:
    /// - `mmap`: Writable memory map of the Wick's slab.
    /// - `view`: View over the contiguous Wick's slab.
    /// # Returns
    /// - `WicksShared<T>`: Wick's storage and associated view.
    pub(crate) fn from_mmap_cow(mmap: MmapMut, view: WicksView<T>) -> Self {
        Self {backing: WicksBacking::MmapCow(mmap), view}
    }
}

/// Storage for the RMA data of the Wick's objects.
pub(crate) struct WicksRma<T: NOCIScalar> {
    /// Shared-memory allocation handle.
    pub(crate) _shared: Sharedffi,
    /// Raw pointer to the beginning of the shared tensor slab.
    pub(crate) base_ptr: *mut u8,
    /// Total size of the shared allocation in bytes.
    pub(crate) _nbytes: usize,
    /// Marker for the scalar type stored in the slab.
    pub(crate) _marker: std::marker::PhantomData<T>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "T: NOCIScalar")]
pub(crate) struct WicksDiskMeta<T: NOCIScalar> {
    /// On-disk metadata format version.
    pub(crate) version: u32,
    /// Number of reference determinants.
    pub(crate) nref: usize,
    /// Total slab length in units of T.
    pub(crate) slab_len: usize,
    /// Per-reference-pair offset tables into the contiguous slab.
    pub(crate) off: Vec<PairOffset>,
    /// Per-reference-pair scalar metadata stored outside the slab.
    pub(crate) meta: Vec<PairMeta<T>>,
}

/// Create a file-backed mutable memory map for the contiguous Wick's tensor slab and
/// write the associated metadata to disk.
/// # Arguments:
/// - `slab_path`: Path to the raw file storing the contiguous tensor slab.
/// - `nref`: Number of reference determinants.
/// - `off`: Per-pair offset table into the contiguous tensor slab.
/// - `slab_len`: Total slab length in units of `T`.
/// # Returns
/// - `std::io::Result<WicksShared<T>>`: File-backed Wick's storage and view over the mapped slab.
pub(crate) fn create_wicks_mmap<T: NOCIScalar>(slab_path: &std::path::Path, nref: usize, off: Vec<PairOffset>, slab_len: usize) -> std::io::Result<WicksShared<T>> {
    let nbytes = slab_len * std::mem::size_of::<T>();
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(slab_path)?;
    file.set_len(nbytes as u64)?;

    let mut mmap = unsafe {MmapOptions::new().len(nbytes).map_mut(&file)?};
    let ptr = mmap.as_mut_ptr() as *mut T;
    let slab = unsafe {std::slice::from_raw_parts_mut(ptr, slab_len)};
    slab.fill(<T as From<f64>>::from(f64::NAN));

    let meta = vec![PairMeta::<T>::default(); nref * nref];
    let view = WicksView {slab: NonNull::new(ptr).unwrap(), slab_len, nref, off, meta};
    
    Ok(WicksShared::from_mmap_cow(mmap, view))
}

/// Load a file-backed read-only memory map for a previously written Wick's tensor slab
/// together with its serialised metadata.
/// # Arguments:
/// - `slab_path`: Path to the raw file storing the contiguous tensor slab.
/// - `meta_path`: Path to the file storing serialised Wick's metadata.
/// # Returns
/// - `std::io::Result<WicksShared<T>>`: File-backed Wick's storage and view over the mapped slab.
pub(crate) fn load_wicks_mmap<T: NOCIScalar>(slab_path: &std::path::Path, meta_path: &std::path::Path) -> std::io::Result<WicksShared<T>> {
    let disk_meta: WicksDiskMeta<T> = bincode::deserialize(&std::fs::read(meta_path)?).unwrap();
    let file = File::open(slab_path)?;
    let mmap = unsafe {MmapOptions::new().map(&file)?};
    let ptr = mmap.as_ptr() as *mut T;

    let view = WicksView {slab: NonNull::new(ptr).unwrap(), slab_len: disk_meta.slab_len, nref: disk_meta.nref, off: disk_meta.off, meta: disk_meta.meta};
    
    Ok(WicksShared::from_mmap(mmap, view))
}
