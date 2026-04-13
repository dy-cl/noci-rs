// nonorthogonalwicks/storage.rs
use std::ptr::NonNull;
use std::fs::{File, OpenOptions};

use serde::{Serialize, Deserialize};
use memmap2::{Mmap, MmapMut, MmapOptions};

use crate::mpiutils::Sharedffi;
use super::view::WicksView;
use super::types::{PairOffset, PairMeta};

#[allow(dead_code)]
pub(crate) enum WicksBacking {
    Shared(WicksRma),
    Mmap(Mmap),
    MmapCow(MmapMut),
}

/// Storage in which we split the Wicks data into the shared remote memory access (RMA) and a view 
/// for reading said data.
pub struct WicksShared {
    /// Backing storage for the contiguous Wick's tensor slab.
    backing: WicksBacking,
    /// View of the tensor slab.
    view: WicksView, 
}

impl WicksShared {
    /// Get a shared reference to the WicksView object.
    /// # Arguments:
    /// - `self`: View and RMA for Wick's intermediates.
    /// # Returns
    /// - `&WicksView`: Shared view of the Wick's intermediates.
    pub fn view(&self) -> &WicksView {&self.view}
    
    /// Get a mutable reference to the WicksView object.
    /// # Arguments:
    /// - `self`: View and RMA for Wick's intermediates.
    /// # Returns
    /// - `&mut WicksView`: Mutable view of the Wick's intermediates.
    pub fn view_mut(&mut self) -> &mut WicksView {
        &mut self.view
    }
    
    /// Get a mutable slice over the full contiguous shared tensor storage.
    /// The returned slice may be used to overwrite stored matrices or tensors in place.
    /// # Arguments:
    /// - `self`: View and RMA for Wick's intermediates.
    /// # Returns
    /// - `&mut [f64]`: Mutable slice over the full shared tensor slab.
    pub fn slab_mut(&mut self) -> &mut [f64] {
        let ptr = self.base_mut_ptr();
        let len = self.view.slab_len;
        unsafe {std::slice::from_raw_parts_mut(ptr, len)}
    }
    
    /// Return a mutable pointer to the start of the contiguous tensor storage.
    /// # Arguments:
    /// - `self`: View and backing storage for Wick's intermediates.
    /// # Returns
    /// - `*mut f64`: Mutable pointer to the start of the tensor slab.
    /// # Panics
    /// - Panics if the backing storage is read-only.
    fn base_mut_ptr(&mut self) -> *mut f64 {
        match &mut self.backing {
            WicksBacking::Shared(rma) => rma.base_ptr as *mut f64,
            WicksBacking::Mmap(_) => panic!("Wick's slab is read-only"),
            WicksBacking::MmapCow(map) => map.as_mut_ptr() as *mut f64,
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
    /// - `WicksShared`: Wick's storage and associated view.
    pub(crate) fn from_shared(rma: WicksRma, view: WicksView) -> Self {
        Self {backing: WicksBacking::Shared(rma), view}
    }

    /// Construct file-backed read-only Wick's storage.
    /// # Arguments:
    /// - `mmap`: Read-only memory map of the Wick's slab.
    /// - `view`: View over the contiguous Wick's slab.
    /// # Returns
    /// - `WicksShared`: Wick's storage and associated view.
    pub(crate) fn from_mmap(mmap: Mmap, view: WicksView) -> Self {
        Self {backing: WicksBacking::Mmap(mmap), view}
    }

    /// Construct file-backed writable Wick's storage.
    /// # Arguments:
    /// - `mmap`: Writable memory map of the Wick's slab.
    /// - `view`: View over the contiguous Wick's slab.
    /// # Returns
    /// - `WicksShared`: Wick's storage and associated view.
    pub(crate) fn from_mmap_cow(mmap: MmapMut, view: WicksView) -> Self {
        Self {backing: WicksBacking::MmapCow(mmap), view}
    }
}

/// Storage for the RMA data of the Wick's objects.
pub(crate) struct WicksRma {
    /// Shared-memory allocation handle.
    pub(crate) _shared: Sharedffi,
    /// Raw pointer to the beginning of the shared tensor slab.
    pub(crate) base_ptr: *mut u8,
    /// Total size of the shared allocation in bytes.
    pub(crate) _nbytes: usize,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct WicksDiskMeta {
    /// On-disk metadata format version.
    pub(crate) version: u32,
    /// Number of reference determinants.
    pub(crate) nref: usize,
    /// Total slab length in units of f64.
    pub(crate) slab_len: usize,
    /// Per-reference-pair offset tables into the contiguous slab.
    pub(crate) off: Vec<PairOffset>,
    /// Per-reference-pair scalar metadata stored outside the slab.
    pub(crate) meta: Vec<PairMeta>,
}

/// Create a file-backed mutable memory map for the contiguous Wick's tensor slab and
/// write the associated metadata to disk.
/// # Arguments:
/// - `slab_path`: Path to the raw file storing the contiguous tensor slab.
/// - `meta_path`: Path to the file storing serialised Wick's metadata.
/// - `nref`: Number of reference determinants.
/// - `off`: Per-pair offset table into the contiguous tensor slab.
/// - `meta`: Per-pair metadata stored outside the tensor slab.
/// - `slab_len`: Total slab length in units of `f64`.
/// # Returns
/// - `std::io::Result<WicksShared>`: File-backed Wick's storage and view over the mapped slab.
pub(crate) fn create_wicks_mmap(slab_path: &std::path::Path, nref: usize, off: Vec<PairOffset>, slab_len: usize) -> std::io::Result<WicksShared> {
    let nbytes = slab_len * std::mem::size_of::<f64>();
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(slab_path)?;
    file.set_len(nbytes as u64)?;

    let mut mmap = unsafe {MmapOptions::new().len(nbytes).map_mut(&file)?};
    let ptr = mmap.as_mut_ptr() as *mut f64;
    let slab = unsafe {std::slice::from_raw_parts_mut(ptr, slab_len)};
    slab.fill(f64::NAN);

    let meta = vec![PairMeta::default(); nref * nref];
    let view = WicksView {slab: NonNull::new(ptr).unwrap(), slab_len, nref, off, meta};
    
    Ok(WicksShared::from_mmap_cow(mmap, view))
}

/// Load a file-backed read-only memory map for a previously written Wick's tensor slab
/// together with its serialised metadata.
/// # Arguments:
/// - `slab_path`: Path to the raw file storing the contiguous tensor slab.
/// - `meta_path`: Path to the file storing serialised Wick's metadata.
/// # Returns
/// - `std::io::Result<WicksShared>`: File-backed Wick's storage and view over the mapped slab.
pub(crate) fn load_wicks_mmap(slab_path: &std::path::Path, meta_path: &std::path::Path) -> std::io::Result<WicksShared> {
    let disk_meta: WicksDiskMeta = bincode::deserialize(&std::fs::read(meta_path)?).unwrap();
    let file = File::open(slab_path)?;
    let mmap = unsafe {MmapOptions::new().map(&file)?};
    let ptr = mmap.as_ptr() as *mut f64;

    let view = WicksView {slab: NonNull::new(ptr).unwrap(), slab_len: disk_meta.slab_len, nref: disk_meta.nref, off: disk_meta.off, meta: disk_meta.meta};
    
    Ok(WicksShared::from_mmap(mmap, view))
}

