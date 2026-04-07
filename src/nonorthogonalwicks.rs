// nonorthogonalwicks.rs 
use std::ptr::NonNull;
use std::fs::{File, OpenOptions};

use ndarray::{Array1, Array2, ArrayView2, Array4, ArrayView4, Axis, s};
use ndarray_linalg::{SVD, Determinant};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use memmap2::{Mmap, MmapMut, MmapOptions};

use crate::{ExcitationSpin};

use crate::maths::{einsum_ba_ab_real, eri_ao2mo, loewdin_x_real, minor, adjugate_transpose, det, mix_columns, build_d};
use crate::mpiutils::Sharedffi;
use crate::noci::occ_coeffs;

pub enum WicksBacking {
    Shared(WicksRma),
    Mmap(Mmap),
    MmapCow(MmapMut),
}

// Storage in which we split the Wicks data into the shared remote memory access (RMA) and a view 
// for reading said data.
pub struct WicksShared {
    pub backing: WicksBacking,
    pub view: WicksView, 
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
}

// Storage for the RMA data of the Wick's objects.
pub struct WicksRma {
    pub shared: Sharedffi, 
    pub base_ptr: *mut u8, 
    pub nbytes: usize,     
}

#[derive(Serialize, Deserialize)]
pub struct WicksDiskMeta {
    pub version: u32,
    pub nref: usize,
    pub slab_len: usize,
    pub off: Vec<PairOffset>,
    pub meta: Vec<PairMeta>,
}

// Storage for data which allows the Wicks objects to be viewed.
#[derive(Clone)]
pub struct WicksView {
    // Pointer to contiguous data which contains all intermediates.
    pub slab: NonNull<f64>, 
    // Length of storage.
    pub slab_len: usize,
    // Number of reference determinants.
    pub nref: usize,
    // Offset gives where in the storage each tensor for pair p begins.
    pub off: Vec<PairOffset>,
    // Scalars that are cheap to store.
    pub meta: Vec<PairMeta>,
}

// Implying that WicksView can be shared across threads.
unsafe impl Sync for WicksView {}
unsafe impl Send for WicksView {}

impl WicksView {
    /// Map a pair index (lp, gp) into a 1D flattened index.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `lp`: Pair index 1.
    /// - `gp`: Pair index 2.
    /// # Returns
    /// - `usize`: Flattened pair index.
    pub fn idx(&self, lp: usize, gp: usize) -> usize {
        lp * self.nref + gp
    }
    
    /// Get a pointer to the start of the shared tensor storage.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// # Returns
    /// - `*const f64`: Pointer to the start of the shared tensor slab.
    fn slab_ptr(&self) -> *const f64 {
        self.slab.as_ptr()
    }
    
    /// Read tensor slab beginning at a given offset and interpret the following n * n elements as
    /// a n by n matrix. Lifetime elision '_ ensures that the ArrayView2 may not outlive the borrow
    /// of self (WicksView) which in turn is only valid while the remote memory storage is valid.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `off_f64`: Offset from the beginning of the tensor slab in units of f64.
    /// - `n`: Size of matrix to be read.
    /// # Returns
    /// - `ArrayView2<'_, f64>`: Matrix view into the tensor slab.
    pub fn view2(&self, off_f64: usize, n: usize) -> ArrayView2<'_, f64> {
        unsafe {ArrayView2::from_shape_ptr((n, n), self.slab_ptr().add(off_f64))}
    }

    /// Read tensor slab beginning at a given offset and interpret the following n * n * n * n elements as
    /// a n by n by n by n tensor. Lifetime elision '_ ensures that the ArrayView4 may not outlive the borrow
    /// of self (WicksView) which in turn is only valid while the remote memory storage is valid.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `off_f64`: Offset from the beginning of the tensor slab in units of f64.
    /// - `n`: Size of tensor to be read.
    /// # Returns
    /// - `ArrayView4<'_, f64>`: 4D tensor view into the tensor slab.
    pub fn view4(&self, off_f64: usize, n: usize) -> ArrayView4<'_, f64> {
        unsafe {ArrayView4::from_shape_ptr((n, n, n, n), self.slab_ptr().add(off_f64))}
    }
    
    /// Return a view for precomputed intermediates for a given lp, gp. Lifetime elision '_ ensures 
    /// that the ArrayView4 may not outlive the borrow of self (WicksView) which in turn is only valid 
    /// while the remote memory storage is valid.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `lp`: Pair index 1.
    /// - `gp`: Pair index 2.
    /// # Returns
    /// - `WicksPairView<'_>`: Grouped view of the same-spin and different-spin intermediates for the pair.
    pub fn pair(&self, lp: usize, gp: usize) -> WicksPairView<'_> {
        let idx = self.idx(lp, gp);

        let aa = SameSpinView {nmo: self.meta[idx].aa.nmo, m: self.meta[idx].aa.m, tilde_s_prod: self.meta[idx].aa.tilde_s_prod, 
                               phase: self.meta[idx].aa.phase, f0f: self.meta[idx].aa.f0f, f0h: self.meta[idx].aa.f0h, v0: self.meta[idx].aa.v0, w: self, off: self.off[idx].aa};
        let bb = SameSpinView {nmo: self.meta[idx].bb.nmo, m: self.meta[idx].bb.m, tilde_s_prod: self.meta[idx].bb.tilde_s_prod, 
                               phase: self.meta[idx].bb.phase, f0f: self.meta[idx].bb.f0f, f0h: self.meta[idx].bb.f0h, v0: self.meta[idx].bb.v0, w: self, off: self.off[idx].bb};
        let ab = DiffSpinView {nmo: self.meta[idx].ab.nmo, vab0: self.meta[idx].ab.vab0, vba0: self.meta[idx].ab.vba0, w: self, off: self.off[idx].ab};

        WicksPairView {aa, bb, ab}
    }
}

// Read only view of same-spin Wick's intermediates. Lifetime parameter 'a ensures that 
// the view cannot exist longer than the referenced WicksView object.
#[derive(Clone, Copy)]
pub struct SameSpinView<'a> {
    pub nmo: usize,
    pub m: usize,
    pub tilde_s_prod: f64,
    pub phase: f64,
    pub f0f: [f64; 2],
    pub f0h: [f64; 2],
    pub v0: [f64; 3],
    w: &'a WicksView,
    off: SameSpinOffset,
}

impl<'a> SameSpinView<'a> {
    /// Get tensor dimension n. 
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// # Returns
    /// - `usize`: Tensor dimension `2 * nmo`.
    fn n(&self) -> usize {2 * self.nmo}
    
    /// Get a view to the `X[mi]` matrix.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi`: Zero distribution selector. 
    /// # Returns
    /// - `ArrayView2<'_, f64>`: View of the `X[mi]` matrix.
    pub fn x(&self, mi: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.x[mi], self.n())
    }
    
    /// Get a view to the `Y[mi]` matrix.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi`: Zero distribution selector. 
    /// # Returns
    /// - `ArrayView2<'_, f64>`: View of the `Y[mi]` matrix.
    pub fn y(&self, mi: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.y[mi], self.n())
    }
    
    /// Get a view to the transpoed Hamiltonian `F[mi][mj]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi, mj`: Zero distribution selectors. 
    /// # Returns
    /// - `ArrayView2<'_, f64>`: View of the transpoed Hamiltonian `F[mi][mj]` matrix.
    pub fn fh_t(&self, mi: usize, mj: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.fh[mi][mj], self.n())
    }

    /// Get a view to the transpoed Fock `F[mi][mj]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi, mj`: Zero distribution selectors. 
    /// # Returns
    /// - `ArrayView2<'_, f64>`: View of the transpoed Fock `F[mi][mj]` matrix.
    pub fn ff_t(&self, mi: usize, mj: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.ff[mi][mj], self.n())
    }
    
    /// Get a view to the transpoed `V[mi][mj][mk]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi, mj, mk`: Zero distribution selector. 
    /// # Returns
    /// - `ArrayView2<'_, f64>`: View of the transpoed `V[mi][mj][mk]` matrix.
    pub fn v_t(&self, mi: usize, mj: usize, mk: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.v[mi][mj][mk], self.n())
    }
    
    /// Get a view to the `J[mi][mj][mk][ml]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `slot`: Compressed storage slot for the requested J tensor.
    /// # Returns
    /// - `ArrayView4<'_, f64>`: View of the requested J tensor.
    pub fn j(&self, slot: usize) -> ArrayView4<'_, f64> {
        self.w.view4(self.off.j[slot], self.n())
    }
}

// Read only view of diff-spin Wick's intermediates. Lifetime parameter 'a ensures that 
// the view cannot exist longer than the referenced WicksView object.
#[derive(Clone, Copy)]
pub struct DiffSpinView<'a> {
    pub nmo: usize,
    pub vab0: [[f64; 2]; 2],
    pub vba0: [[f64; 2]; 2],
    w: &'a WicksView,
    off: DiffSpinOffset,
}

impl<'a> DiffSpinView<'a> {
    /// Get tensor dimension n. 
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// # Returns
    /// - `usize`: Tensor dimension `2 * nmo`.
    fn n(&self) -> usize {2 * self.nmo}
    
    /// Get a view to the transpoed `Vab[ma0][mb0][mak]` tensor.
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// - `ma0, mb0, mak`: Zero distribution selector. 
    /// # Returns
    /// - `ArrayView2<'a, f64>`: View of the transpoed `Vab[ma0][mb0][mak]` matrix.
    pub fn vab_t(&self, ma0: usize, mb0: usize, mak: usize) -> ArrayView2<'a, f64> {
        self.w.view2(self.off.vab[ma0][mb0][mak], self.n())
    }

    /// Get a view to the transpoed `Vba[mb0][ma0][mak]` tensor.
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// - `mb0, ma0, mbk`: Zero distribution selector.
    /// # Returns
    /// - `ArrayView2<'a, f64>`: View of the transpoed `Vba[mb0][ma0][mbk]` matrix.
    pub fn vba_t(&self, mb0: usize, ma0: usize, mbk: usize) -> ArrayView2<'a, f64> {
        self.w.view2(self.off.vba[mb0][ma0][mbk], self.n())
    }

    /// Get a view to the `IIab[ma0][maj][mb0][mbj]` tensor.
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// - `ma0, maj, mb0, mbj`: Zero distribution selector.
    /// # Returns
    /// - `ArrayView4<'a, f64>`: View of the `IIab[ma0][maj][mb0][mbj]` tensor.
    pub fn iiab(&self, ma0: usize, maj: usize, mb0: usize, mbj: usize) -> ArrayView4<'a, f64> {
        self.w.view4(self.off.iiab[ma0][maj][mb0][mbj], self.n())
    }
}

// Storage for views of each type of spin pairing.
#[derive(Clone, Copy)]
pub struct WicksPairView<'a> {
    pub aa: SameSpinView<'a>,
    pub bb: SameSpinView<'a>,
    pub ab: DiffSpinView<'a>,
}

// Storage for same-spin metadata and lightweight scalars that we can store outside the shared
// memory region.
#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub struct SameSpinMeta {
    pub tilde_s_prod: f64,
    pub phase: f64,
    pub m: usize,
    pub nmo: usize,
    pub f0f: [f64; 2],
    pub f0h: [f64; 2],
    pub v0: [f64; 3],
}

// Storage for diff-spin metadata and lightweight scalars that we can store outside the shared
// memory region.
#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub struct DiffSpinMeta {
    pub nmo: usize,
    pub vab0: [[f64; 2]; 2],
    pub vba0: [[f64; 2]; 2],
}

// Storage for same-spin per reference-pair offset tables into the shared contiguous tensor storage. 
#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug)]
pub struct SameSpinOffset {
    pub x: [usize; 2],
    pub y: [usize; 2],
    pub fh: [[usize; 2]; 2],
    pub ff: [[usize; 2]; 2],
    pub v: [[[usize; 2]; 2]; 2],
    pub j: [usize; 10], // 10 / 16 4D tensors.
}

// Storage for diff-spin per reference-pair offset tables into the shared contiguous tensor storage.
#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug)]
pub struct DiffSpinOffset {
    pub vab: [[[usize; 2]; 2]; 2],
    pub vba: [[[usize; 2]; 2]; 2],
    pub iiab: [[[[usize; 2]; 2]; 2]; 2],
}

// Storage for all per reference-pair pair offset tables.
#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub struct PairOffset {
    pub aa: SameSpinOffset,
    pub bb: SameSpinOffset,
    pub ab: DiffSpinOffset,
}

// Storage for all pair metadata.
#[derive(Clone, Default, Serialize, Deserialize, Debug)]
pub struct PairMeta {
    pub aa: SameSpinMeta,
    pub bb: SameSpinMeta,
    pub ab: DiffSpinMeta,
}

// Owning struct for the same-spin computed intermediates.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SameSpinBuild {
    pub x: [Array2<f64>; 2], // X[mi]
    pub y: [Array2<f64>; 2], // Y[mi]

    pub f0f: [f64; 2],
    pub f0h: [f64; 2],
    pub fh: [[Array2<f64>; 2]; 2], // F[mi][mj] for one electron Hamiltonian.
    pub ff: [[Array2<f64>; 2]; 2], // F[mi][mj] for Fock. 

    pub v0: [f64; 3], // V0[mi][mj]
    pub v: [[[Array2<f64>; 2]; 2]; 2], // V[mi][mj][mk]

    pub j: [Array4<f64>; 10], // J[mi][mj][mk][ml], but only store 10 / 16.

    pub tilde_s_prod: f64,
    pub phase: f64,
    pub m: usize,
    pub nmo: usize,
}

// Owning struct for the diff-spin computed intermediates.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct DiffSpinBuild {
    pub vab0: [[f64; 2]; 2], // vab0[ma0][mb0]
    pub vab:  [[[Array2<f64>; 2]; 2]; 2], // vab[ma0][mb0][mak]

    pub vba0: [[f64; 2]; 2], // vba0[mb0][ma0]
    pub vba:  [[[Array2<f64>; 2]; 2]; 2], // vba[mb0][ma0][mbk]

    pub iiab: [[[[Array4<f64>; 2]; 2]; 2]; 2], // iiab[ma0][mak][mb0][mbj]
}

// Owning struct for all computed intermediates.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct WicksReferencePairBuild {
    pub aa: SameSpinBuild,
    pub bb: SameSpinBuild,
    pub ab: DiffSpinBuild,
}

// Whether a given orbital index belongs to the bra or ket.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Side {Gamma, Lambda}

#[derive(Debug, Copy, Clone)]
pub enum Type {Hole, Part}

#[derive(Clone, Copy)]
enum OneBody {H1, Fock}

/// Read the scalar zeroth-order one-body contribution for the chosen branch.
/// # Arguments:
/// - `w`: Same-spin Wick's view.
/// - `ob`: Selects Hamiltonian or Fock intermediates.
/// - `mi`: Branch selector for the operator contraction.
/// # Returns
/// - `f64`: Zeroth-order one-body scalar contribution.
#[inline(always)]
fn one_body_scalar(w: &SameSpinView<'_>, ob: OneBody, mi: usize) -> f64 {
    match ob {OneBody::H1 => w.f0h[mi], OneBody::Fock => w.f0f[mi]}
}

/// Read the first-order one-body matrix block for the chosen branch pair.
/// # Arguments:
/// - `w`: Same-spin Wick's view.
/// - `ob`: Selects Hamiltonian or Fock intermediates.
/// - `mi`: Branch selector associated with the one-body operator contraction.
/// - `mj`: Branch selector associated with the chosen determinant column.
/// # Returns
/// - `ArrayView2<'a, f64>`: Borrowed view of the requested one-body block.
#[inline(always)]
fn one_body_block<'a>(w: &'a SameSpinView<'a>, ob: OneBody, mi: usize, mj: usize,) -> ArrayView2<'a, f64> {
    match ob {OneBody::H1 => w.fh_t(mi, mj), OneBody::Fock => w.ff_t(mi, mj)}
}

pub type Label = (Side, Type, usize);

/// Calculate and store all the required offsets from the beginning of the large contiguous shared
/// tensor storage to a given matrix or tensor.
/// # Arguments:
/// - `nref`: Number of references.
/// - `nmo`: Number of molecular orbitals.
/// # Returns
/// - `(Vec<PairOffset>, usize)`: Per-pair offset table and total slab length in units of `f64`.
pub fn assign_offsets(nref: usize, nmo: usize) -> (Vec<PairOffset>, usize) {
    
    let n = 2 * nmo;
    let nn2 = n * n;
    let nn4 = n * n * n * n;
    let mut off = vec![PairOffset::default(); nref * nref];
    let mut i: usize = 0;
    
    // For each reference-pair we have n by n f64s or n by n by n by n f64s.
    // For every pair p we assign all offset locations for all required tensors.
    for p in off.iter_mut() {
        for mi in 0..2 {p.aa.x[mi] = i; i += nn2;}
        for mi in 0..2 {p.aa.y[mi] = i; i += nn2;}
        for mi in 0..2 { for mj in 0..2 {p.aa.fh[mi][mj] = i; i += nn2;}}
        for mi in 0..2 { for mj in 0..2 {p.aa.ff[mi][mj] = i; i += nn2;}}
        for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {p.aa.v[mi][mj][mk] = i; i += nn2;}}}
        for s in 0..10 {p.aa.j[s] = i; i += nn4;}

        for mi in 0..2 {p.bb.x[mi] = i; i += nn2;}
        for mi in 0..2 {p.bb.y[mi] = i; i += nn2;}
        for mi in 0..2 {for mj in 0..2 {p.bb.fh[mi][mj] = i; i += nn2;}}
        for mi in 0..2 {for mj in 0..2 {p.bb.ff[mi][mj] = i; i += nn2;}}
        for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {p.bb.v[mi][mj][mk] = i; i += nn2;}}}
        for s in 0..10 {p.bb.j[s] = i; i += nn4;}

        for ma0 in 0..2 {for mb0 in 0..2 {for mk in 0..2 {p.ab.vab[ma0][mb0][mk] = i; i += nn2;}}}
        for mb0 in 0..2 {for ma0 in 0..2 {for mk in 0..2 {p.ab.vba[mb0][ma0][mk] = i; i += nn2;}}}

        for ma0 in 0..2 {for maj in 0..2 {for mb0 in 0..2 {for mbj in 0..2 {p.ab.iiab[ma0][maj][mb0][mbj] = i; i += nn4;}}}}
    }
    (off, i) 
}

/// Fill the same-spin data owning structs with the same-spin Wick's intermediates using the
/// prescribed offsets.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `o`: Offsets into the storage.
/// - `w`: Owned Wick's intermediates.
/// # Returns
/// - `()`: Writes the same-spin intermediates into the slab.
pub fn write_same_spin(slab: &mut [f64], o: &SameSpinOffset, w: &SameSpinBuild) {
    write2(slab, o.x[0], &w.x[0]);
    write2(slab, o.x[1], &w.x[1]);
    write2(slab, o.y[0], &w.y[0]);
    write2(slab, o.y[1], &w.y[1]);
    for mi in 0..2 {for mj in 0..2 {write2t(slab, o.fh[mi][mj], &w.fh[mi][mj]);}}
    for mi in 0..2 {for mj in 0..2 {write2t(slab, o.ff[mi][mj], &w.ff[mi][mj]);}}
    for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {write2t(slab, o.v[mi][mj][mk], &w.v[mi][mj][mk]);}}}
    for s in 0..10 {write4ijrc(slab, o.j[s], &w.j[s]);}
}

/// Fill the diff-spin data owning structs with the diff-spin Wick's intermediates using the
/// prescribed offsets.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `o`: Offsets into the storage.
/// - `w`: Owned Wick's intermediates.
/// # Returns
/// - `()`: Writes the diff-spin intermediates into the slab.
pub fn write_diff_spin(slab: &mut [f64], o: &DiffSpinOffset, w: &DiffSpinBuild) {
    for ma0 in 0..2 {for mb0 in 0..2 {for mk in 0..2 {write2t(slab, o.vab[ma0][mb0][mk], &w.vab[ma0][mb0][mk]);}}}
    for mb0 in 0..2 {for ma0 in 0..2 {for mk in 0..2 {write2t(slab, o.vba[mb0][ma0][mk], &w.vba[mb0][ma0][mk]);}}}
    for ma0 in 0..2 {for maj in 0..2 {for mb0 in 0..2 {for mbj in 0..2 {write4rcij(slab, o.iiab[ma0][maj][mb0][mbj], &w.iiab[ma0][maj][mb0][mbj]);}}}}
}

/// Copy matrix into tensor slab provided it is contiguous.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset for the start position.
/// - `a`: Matrix to copy.
/// # Returns
/// - `()`: Writes the matrix into the tensor slab.
pub fn write2(slab: &mut [f64], off: usize, a: &ndarray::Array2<f64>) {
    let src = a.as_slice().expect("Array2 must be contiguous");
    slab[off..off + src.len()].copy_from_slice(src);
}

/// Copy transpoed matrix into tensor slab provided it is contiguous.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset for the start position.
/// - `a`: Matrix to copy.
/// # Returns
/// - `()`: Writes the matrix into the tensor slab.
pub fn write2t(slab: &mut [f64], off: usize, a: &ndarray::Array2<f64>) {
    let (nr, nc) = a.dim();
    let src = a.as_slice().expect("Array2 must be contiguous");
    let dst = &mut slab[off..off + nr * nc];
    for r in 0..nr {
        let src_row = &src[r * nc..(r + 1) * nc];
        for c in 0..nc {
            dst[c * nr + r] = src_row[c];
        }
    }
}

/// Copy tensor in [r, c, i, j] form into tensor slab provided it is contiguous.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset for the start position.
/// - `a`: Tensor to copy.
/// # Returns
/// - `()`: Writes the tensor into the tensor slab.
fn write4rcij(slab: &mut [f64], off: usize, a: &ndarray::Array4<f64>) {
    let src = a.as_slice().expect("Array4 must be contiguous");
    slab[off..off + src.len()].copy_from_slice(src);
}

/// Copy tensor in [i, j, r, c] form into tensor slab provided it is contiguous.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset for the start position.
/// - `a`: Tensor to copy.
/// # Returns
/// - `()`: Writes the tensor into the tensor slab.
fn write4ijrc(slab: &mut [f64], off: usize, a: &ndarray::Array4<f64>) {
    let sh = a.shape();
    let nr = sh[0];
    let nc = sh[1];
    let ni = sh[2];
    let nj = sh[3];

    let src = a.as_slice().expect("Array4 must be contiguous");
    let dst = &mut slab[off..off + src.len()];

    for r in 0..nr {
        for c in 0..nc {
            for i in 0..ni {
                for j in 0..nj {
                    let src_idx = ((r * nc + c) * ni + i) * nj + j;
                    let dst_idx = ((i * nj + j) * nr + r) * nc + c;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

/// Convert row and column indices into a row-major flat index.
/// # Arguments:
/// - `ncols`: Number of columns in the flattened matrix.
/// - `r`: Row index.
/// - `c`: Column index.
/// # Returns
/// - `usize`: Flat row-major index corresponding to `(r, c)`.
#[inline(always)]
fn idx(ncols: usize, r: usize, c: usize) -> usize {
    r * ncols + c
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
pub fn create_wicks_mmap(slab_path: &std::path::Path, meta_path: &std::path::Path, nref: usize, off: Vec<PairOffset>, 
                         meta: Vec<PairMeta>, slab_len: usize,) -> std::io::Result<WicksShared> {
    let nbytes = slab_len * std::mem::size_of::<f64>();
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(slab_path)?;
    file.set_len(nbytes as u64)?;

    let mut mmap = unsafe {MmapOptions::new().len(nbytes).map_mut(&file)?};

    let ptr = mmap.as_mut_ptr() as *mut f64;
    let slab = unsafe { std::slice::from_raw_parts_mut(ptr, slab_len) };
    slab.fill(0.0);

    mmap.flush()?;

    let disk_meta = WicksDiskMeta {version: 1, nref, slab_len, off, meta};
    std::fs::write(meta_path, bincode::serialize(&disk_meta).unwrap())?;

    let view = WicksView {slab: NonNull::new(ptr).unwrap(), slab_len, nref, off: disk_meta.off.clone(), meta: disk_meta.meta.clone(),};

    Ok(WicksShared {backing: WicksBacking::MmapCow(mmap), view})
}

/// Load a file-backed read-only memory map for a previously written Wick's tensor slab
/// together with its serialised metadata.
/// # Arguments:
/// - `slab_path`: Path to the raw file storing the contiguous tensor slab.
/// - `meta_path`: Path to the file storing serialised Wick's metadata.
/// # Returns
/// - `std::io::Result<WicksShared>`: File-backed Wick's storage and view over the mapped slab.
pub fn load_wicks_mmap(slab_path: &std::path::Path, meta_path: &std::path::Path) -> std::io::Result<WicksShared> {
    let disk_meta: WicksDiskMeta = bincode::deserialize(&std::fs::read(meta_path)?).unwrap();
    let file = File::open(slab_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let ptr = mmap.as_ptr() as *mut f64;

    let view = WicksView {slab: NonNull::new(ptr).unwrap(), slab_len: disk_meta.slab_len, nref: disk_meta.nref, off: disk_meta.off, meta: disk_meta.meta,};

    Ok(WicksShared {backing: WicksBacking::Mmap(mmap), view})
}

/// Write 2D slice of 4D J or IIab tensors into provided output scratch. The given slice is
/// `t[i, j, r, c]` where `i, j` are fixed indices and `r, c` are rows and columns.
/// # Arguments:
/// - `out`: Preallocated output scratch.
/// - `l`: Excitation rank.
/// - `t`: View of a 4D tensor.
/// - `rows`: Length excitation-rank map from row labels to tensor index.
/// - `cols`: Length excitation-rank map from col labels to tensor index.
/// - `i_fixed`: Fixed tensor index for the `i` dimension.
/// - `j_fixed`: Fixed tensor index for the `j` dimension.
/// # Returns
/// - `()`: Writes the requested 2D slice into `out`.
fn slice4ijrc(out: &mut [f64], l: usize, t: &ArrayView4<f64>, rows: &[usize], cols: &[usize], i_fixed: usize, j_fixed: usize) {
    // 4D tensor `t` is stored as
    // base + a * strides[0] + b * strides[1] + c * strides[2] + d * strides[3]
    // for element at t[a, b, c, d], where base is the start of this memory in the slab.
    let strides = t.strides();
    let base = t.as_ptr();

    unsafe {
        // For the fixed indices i, j we know that any t[i, j, r, c] can be found as
        // base + i * strides[0] + j * strides[1] + r * strides[2] + c * strides[3].
        let fixed = (i_fixed as isize) * strides[0] + (j_fixed as isize) * strides[1];

        // Iterate over rows of output.
        for r in 0..l {
            // Output row r is given by tensor row rr = rows[r].
            let rr = *rows.get_unchecked(r) as isize;
            let row_base = r * l;

            // Base offset for current output row is t[i, j, rr, 0] or
            // base + i * strides[0] + j * strides[1] + rr * strides[2] + 0, which is
            // equivalent to fixed + rr * strides[2].
            let off = fixed + rr * strides[2];

            // Iterate over columns of output.
            for c in 0..l {
                // Output col c is given by tensor col cc = cols[c].
                let cc = *cols.get_unchecked(c) as isize;

                // Element t[i, j, rr, cc] is
                // base + i * strides[0] + j * strides[1] + rr * strides[2] + cc * strides[3]
                // or off + cc * strides[3]. Write into output buffer as row-major.
                *out.get_unchecked_mut(row_base + c) = *base.offset(off + cc * strides[3]);
            }
        }
    }
}

/// Write 2D slice of 4D J or IIab tensors into provided output scratch. The given slice is
/// `t[r, c, i, j]` where `r, c` are rows and columns and `i, j` are fixed indices.
/// # Arguments:
/// - `out`: Preallocated output scratch.
/// - `l`: Excitation rank.
/// - `t`: View of a 4D tensor.
/// - `rows`: Length excitation-rank map from row labels to tensor index.
/// - `cols`: Length excitation-rank map from col labels to tensor index.
/// - `i_fixed`: Fixed tensor index for the `i` dimension.
/// - `j_fixed`: Fixed tensor index for the `j` dimension.
/// # Returns
/// - `()`: Writes the requested 2D slice into `out`.
fn slice4rcij(out: &mut [f64], l: usize, t: &ArrayView4<f64>, rows: &[usize], cols: &[usize], i_fixed: usize, j_fixed: usize) {
    // 4D tensor `t` is stored as
    // base + a * strides[0] + b * strides[1] + c * strides[2] + d * strides[3]
    // for element at t[a, b, c, d], where base is the start of this memory in the slab.
    let strides = t.strides();
    let base = t.as_ptr();

    unsafe {
        // For the fixed indices i, j we know that any t[r, c, i, j] can be found as
        // base + r * strides[0] + c * strides[1] + i * strides[2] + j * strides[3].
        let fixed = (i_fixed as isize) * strides[2] + (j_fixed as isize) * strides[3];

        // Iterate over rows of output.
        for r in 0..l {
            // Output row r is given by tensor row rr = rows[r].
            let rr = *rows.get_unchecked(r) as isize;
            let row_base = r * l;

            // Base offset for current output row is t[rr, 0, i, j] or
            // base + rr * strides[0] + 0 + i * strides[2] + j * strides[3], which is
            // equivalent to rr * strides[0] + fixed.
            let off = rr * strides[0] + fixed;

            // Iterate over columns of output.
            for c in 0..l {
                // Output col c is given by tensor col cc = cols[c].
                let cc = *cols.get_unchecked(c) as isize;

                // Element t[rr, cc, i, j] is
                // base + rr * strides[0] + cc * strides[1] + i * strides[2] + j * strides[3]
                // or off + cc * strides[1]. Write into output buffer as row-major.
                *out.get_unchecked_mut(row_base + c) = *base.offset(off + cc * strides[1]);
            }
        }
    }
}

/// Map selectors (mi, mj, mk, ml) to the compressed storage slot for J and
/// whether the requested tensor should be read with swapped indices.
/// # Arguments:
/// - `mi, mj, mk, ml`: Zero distribution selectors.
/// # Returns
/// - `(usize, bool)`: Compressed slot index and whether swapped access is required.
fn jslot(mi: usize, mj: usize, mk: usize, ml: usize,) -> (usize, bool) {
    match (mi, mj, mk, ml) {
        (0, 0, 0, 0) => (0, false),
        (0, 0, 0, 1) => (1, false),
        (0, 1, 0, 0) => (1, true),
        (0, 0, 1, 0) => (2, false),
        (1, 0, 0, 0) => (2, true),
        (0, 0, 1, 1) => (3, false),
        (1, 1, 0, 0) => (3, true),
        (0, 1, 0, 1) => (4, false),
        (0, 1, 1, 0) => (5, false),
        (1, 0, 0, 1) => (5, true),
        (0, 1, 1, 1) => (6, false),
        (1, 1, 0, 1) => (6, true),
        (1, 0, 1, 0) => (7, false),
        (1, 0, 1, 1) => (8, false),
        (1, 1, 1, 0) => (8, true),
        (1, 1, 1, 1) => (9, false),
        _ => unreachable!(),
    }
}

#[derive(Default)]
pub struct Vec1 {
    data: Vec<f64>,
    len: usize,
}

impl Vec1 {
    /// Ensure the storage can hold at least `len` elements.
    /// # Arguments:
    /// - `self`: Scratch vector to resize.
    /// - `len`: Required logical length.
    /// # Returns
    /// - `()`: Grows storage if required and updates length.
    #[inline(always)]
    pub fn ensure(&mut self, len: usize) {
        if self.data.len() < len {
            self.data.resize(len, 0.0);
        }
        self.len = len;
    }

    /// Get the contents of the scratch vector as an immutable slice.
    /// # Arguments:
    /// - `self`: Scratch vector.
    /// # Returns
    /// - `&[f64]`: Immutable slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &[f64] {
        &self.data[..self.len]
    }
    
    /// Get the contents of the scratch vector as a mutable slice.
    /// # Arguments:
    /// - `self`: Scratch vector.
    /// # Returns
    /// - `&mut [f64]`: Mutable slice.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data[..self.len]
    }
}

#[derive(Default)]
pub struct Vec2 {
    data: Vec<f64>,
    nrows: usize,
    ncols: usize,
}

impl Vec2 {
    /// Ensure the storage can hold at least `nrows * ncols` elements and set the shape.
    /// # Arguments:
    /// - `self`: Scratch matrix to resize.
    /// - `nrows`: Required number of rows.
    /// - `ncols`: Required number of columns.
    /// # Returns
    /// - `()`: Grows storage if required and updates the shape.
    #[inline(always)]
    pub fn ensure(&mut self, nrows: usize, ncols: usize) {
        let need = nrows * ncols;
        if self.data.len() < need {
            self.data.resize(need, 0.0);
        }
        self.nrows = nrows;
        self.ncols = ncols;
    }
    
    /// Get the contents of the scratch matrix as an immutable slice.
    /// # Arguments:
    /// - `self`: Scratch matrix.
    /// # Returns
    /// - `&[f64]`: Immutable slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &[f64] {
        &self.data[..self.nrows * self.ncols]
    }
    
    /// Get the contents of the scratch matrix as a mutable slice.
    /// # Arguments:
    /// - `self`: Scratch matrix.
    /// # Returns
    /// - `&mut [f64]`: Mutable slice.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        let len = self.nrows * self.ncols;
        &mut self.data[..len]
    }
}

// Storage for preallocated Wick's data terms such that we do not have to reallocate all of these
// everytime a matrix element evaluation routine is called.
#[derive(Default)]
pub struct WickScratch {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,

    pub det0: Vec2,
    pub det1: Vec2,
    pub det_mix: Vec2,

    pub fcol: Vec1,
    pub dv: Vec1,

    pub v1: Vec1,
    pub dv1: Vec1,
    pub dv1m: Vec1,
    pub jslice_full: Vec2,
    pub jslice2: Vec2,
    pub det_mix2: Vec2,

    pub rows_a: Vec<usize>,
    pub cols_a: Vec<usize>,
    pub rows_b: Vec<usize>,
    pub cols_b: Vec<usize>,

    pub deta0: Vec2,
    pub deta1: Vec2,
    pub deta_mix: Vec2,
    pub detb0: Vec2,
    pub detb1: Vec2,
    pub detb_mix: Vec2,

    pub v1a: Vec1,
    pub v1b: Vec1,
    pub dv1a: Vec1,
    pub dv1b: Vec1,

    pub iislicea: Vec2,
    pub iisliceb: Vec2,
    pub deta_mix_minor: Vec2,
    pub detb_mix_minor: Vec2,

    pub adjt_det: Vec2,
    pub adjt_deta: Vec2,
    pub adjt_detb: Vec2,
    pub adjt_det2: Vec2,
    pub adjt_deta_mix_minor: Vec2,
    pub adjt_detb_mix_minor: Vec2,

    pub invs: Vec1,
    pub invsla: Vec1,
    pub invslb: Vec1,
    pub invslm1: Vec1,
    pub invslam1: Vec1,
    pub invslbm1: Vec1,

    pub lu: Vec2,
    pub lua: Vec2,
    pub lub: Vec2,
}

impl WickScratch {
    /// Construct empty scratch storage for Wick's quantities.
    /// # Arguments:
    /// # Returns
    /// - `WickScratch`: Default-initialised scratch storage.
    pub fn new() -> Self {Self::default()}
    
    /// Pre-allocate the Wick' scratch buffers to the largest sizes needed for the
    /// current calculation in order to reduce repeated allocations.
    /// # Arguments:
    /// - `maxsame`: Maximum total excitation rank required for same-spin terms.
    /// - `maxla`: Maximum total alpha-spin excitation rank required for different-spin terms.
    /// - `maxlb`: Maximum total beta-spin excitation rank required for different-spin terms.
    /// # Returns
    /// - `WickScratch`: Scratch storage with the requested capacities reserved.
    #[inline]
    pub fn with_sizes(maxsame: usize, maxla: usize, maxlb: usize) -> Self {
        let mut s = Self::default();
        s.ensure_same(maxsame);
        s.ensure_diff(maxla, maxlb);
        s
    }
    
    /// If the previously allocated size of the scratch space is the not the same in  
    /// the same spin case resize all the scratch space quantities to be correct.
    /// # Arguments:
    /// - `self`: Scratch space for Wick's quantities.
    /// - `l`: Excitation rank.
    /// # Returns
    /// - `()`: Resizes same-spin scratch storage in place.
    #[inline(always)]
    pub fn ensure_same(&mut self, l: usize) {
        self.rows.clear();
        self.cols.clear();
        self.rows.reserve(l);
        self.cols.reserve(l);

        self.det0.ensure(l, l);
        self.det1.ensure(l, l);
        self.det_mix.ensure(l, l);
        self.adjt_det.ensure(l, l);
        self.jslice_full.ensure(l, l);

        self.fcol.ensure(l);
        self.dv.ensure(l);
        self.v1.ensure(l);
        self.dv1.ensure(l);
        self.invs.ensure(l);

        self.lu.ensure(6, 6);

        let lm1 = l.saturating_sub(1);
        self.dv1m.ensure(lm1);
        self.invslm1.ensure(lm1);
        self.det_mix2.ensure(lm1, lm1);
        self.jslice2.ensure(lm1, lm1);
        self.adjt_det2.ensure(lm1, lm1);
    }

    /// If the previously allocated size of the scratch space is the not the same in  
    /// the different spin case resize all the scratch space quantities to be correct.
    /// # Arguments:
    /// - `self`: Scratch space for Wick's quantities.
    /// - `la`: Excitation rank spin alpha.
    /// - `lb`: Excitation rank spin beta.
    /// # Returns
    /// - `()`: Resizes different-spin scratch storage in place.
    #[inline(always)]
    pub fn ensure_diff(&mut self, la: usize, lb: usize) {
        self.deta0.ensure(la, la);
        self.deta1.ensure(la, la);
        self.deta_mix.ensure(la, la);
        self.adjt_deta.ensure(la, la);
        self.v1a.ensure(la);
        self.dv1a.ensure(la);
        self.iislicea.ensure(la, la);
        self.invsla.ensure(la);
        // This just needs to be of size LMAX + 2.
        self.lua.ensure(6, 6);

        let lam1 = la.saturating_sub(1);
        self.deta_mix_minor.ensure(lam1, lam1);
        self.adjt_deta_mix_minor.ensure(lam1, lam1);
        self.invslam1.ensure(lam1);

        self.detb0.ensure(lb, lb);
        self.detb1.ensure(lb, lb);
        self.detb_mix.ensure(lb, lb);
        self.adjt_detb.ensure(lb, lb);
        self.v1b.ensure(lb);
        self.dv1b.ensure(lb);
        self.iisliceb.ensure(lb, lb);
        self.invslb.ensure(lb);
        // This just needs to be of size LMAX + 2.
        self.lub.ensure(6, 6);

        let lbm1 = lb.saturating_sub(1);
        self.detb_mix_minor.ensure(lbm1, lbm1);
        self.adjt_detb_mix_minor.ensure(lbm1, lbm1);
        self.invslbm1.ensure(lbm1);
    }
}

impl SameSpinBuild {
    /// Constructor for the WicksReferencePair object of SameSpin which contains the precomputed values
    /// per pair of reference determinants required to evaluate arbitrary excited determinants 
    /// in O(1) time when the excitations are of the same spin.
    /// # Arguments:
    /// - `eri`: Electron repulsion integrals. 
    /// - `h_munu`: AO core Hamiltonian.
    /// - `s_munu`: AO overlap matrix.
    /// - `g_c`: Full AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// - `l_c`: Full AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// - `go`: Occupancy vector for |^\Gamma\Psi\rangle.
    /// - `lo`: Occupancy vector for |^\Lambda\Psi\rangle. 
    /// - `tol`: Tolerance for whether a number is considered zero.
    /// # Returns
    /// - `SameSpinBuild`: Precomputed same-spin Wick's intermediates for the reference pair.
    pub fn new(eri: &Array4<f64>, h_munu: &Array2<f64>, s_munu: &Array2<f64>, g_c: &Array2<f64>, l_c: &Array2<f64>, go: u128, lo: u128, tol: f64) -> Self {
        let nmo = g_c.ncols();
        let nbas = l_c.nrows();

        let mut ccat = Array2::<f64>::zeros((nbas, 2 * nmo));
        ccat.slice_mut(s![.., 0..nmo]).assign(l_c);
        ccat.slice_mut(s![.., nmo..2*nmo]).assign(g_c);

        let l_c_occ = occ_coeffs(l_c, lo);
        let g_c_occ = occ_coeffs(g_c, go);

        // SVD and rotate the occupied orbitals.
        let (tilde_s_occ, g_tilde_c_occ, l_tilde_c_occ, phase) = Self::perform_ortho_and_svd_and_rotate(s_munu, &l_c_occ, &g_c_occ, 1e-20);

        // Multiply diagonal non-zero values of {}^{\Gamma\Lambda} \tilde{S} together.
        let tilde_s_prod = tilde_s_occ.iter().filter(|&&x| x.abs() > tol).product::<f64>();

        // Find indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S} and count them.
        let zeros: Vec<usize> = tilde_s_occ.iter().enumerate().filter_map(|(k, &sk)| if sk.abs() <= tol {Some(k)} else {None}).collect();
        let m = zeros.len();
        
        // Construct the {}^{\Gamma\Lambda} M^{\sigma\tau, 0} and {}^{\Gamma\Lambda} M^{\sigma\tau, 1} matrices. 
        // The argument order here is swapped compared to what the function asks for, but it does not match the naive
        // implementation if we swap them.
        let (m0, m1) = Self::construct_m(&tilde_s_occ, &l_tilde_c_occ, &g_tilde_c_occ, &zeros, tol);
        let mao: [Array2<f64>; 2] = [m0, m1];

        // Construct the {}^{\Gamma\Lambda} X_{ij}^{m_k} and {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices.
        let (x0, y0) = Self::construct_xy(g_c, l_c, s_munu, &mao[0], true);
        let (x1, y1) = Self::construct_xy(g_c, l_c, s_munu, &mao[1], false);
        let x: [Array2<f64>; 2] = [x0, x1];
        let y: [Array2<f64>; 2] = [y0, y1];

        // Construct Coulomb and exchange contractions of ERIs with  {}^{\Gamma\Lambda} M^{\sigma\tau, m_k}, 
        // {}^{\Gamma\Lambda} J_{\mu\nu}^{m_k} and {}^{\Gamma\Lambda} K_{\mun\u}^{m_k}. These
        // quantities are used in many of the following intermediates so we precompute here.
        let nbas = mao[0].nrows();
        let mut jkao: [Array2<f64>; 2] = [Array2::<f64>::zeros((nbas, nbas)), Array2::<f64>::zeros((nbas, nbas))];
        for mi in 0..2 {
            let j = Self::build_j_coulomb(eri, &mao[mi]);
            let k = Self::build_k_exchange(eri, &mao[mi]);
            jkao[mi] = &j - &k;
        }
        
        //if maxabs(&mao[0]) > 1e5 || maxabs(&mao[1]) > 1e5 {println!("WARNING: HUGE M")};
        //println!("tilde_s_occ: {:.3e}, tilde_s_prod: {:.3e}", tilde_s_occ, tilde_s_prod);
        //println!("M0 max: {:.3e}, frob: {:.3e} | M1 max: {:.3e}, frob: {:.3e}", maxabs(&mao[0]), frob(&mao[0]), maxabs(&mao[1]), frob(&mao[1]));
        //println!("(J-K)0 max: {:.3e}, frob: {:.3e} | (J-K)1 max: {:.3e}, frob: {:.3e}", maxabs(&jkao[0]), frob(&jkao[0]), maxabs(&jkao[1]), frob(&jkao[1]));
        //println!();

        // Construct the {}^{\Gamma\Lambda} F_0^{m_k} and {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j}
        // intermediates required for one electron Hamiltonian matrix elements.
        let (f0_0h, f00h) = Self::construct_f(l_c, h_munu, &x[0], &y[0]);
        let (_, f01h) = Self::construct_f(l_c, h_munu, &x[0], &y[1]);
        let (_, f10h) = Self::construct_f(l_c, h_munu, &x[1], &y[0]);
        let (f0_1h, f11h) = Self::construct_f(l_c, h_munu, &x[1], &y[1]);
        let f0h: [f64; 2] = [f0_0h, f0_1h];
        let fh: [[Array2<f64>; 2]; 2] = [[f00h, f01h], [f10h, f11h]];
        
        // Initialise {}^{\Gamma\Lambda} F_0^{m_k} and {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j} for 
        // Fock matrix elements to zero as when used in SNOCI these will change per iteration.
        let f0f: [f64; 2] = [0.0, 0.0];
        let ff: [[Array2<f64>; 2]; 2] = [[Array2::zeros((2 * nmo, 2 * nmo)), Array2::zeros((2 * nmo, 2 * nmo))],
                                         [Array2::zeros((2 * nmo, 2 * nmo)), Array2::zeros((2 * nmo, 2 * nmo))]];
        
        // Calculate the left and right factorisations of the {}^{\Gamma\Lambda} X_{ij}^{m_k} and 
        // {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices, {}^{\Gamma\Lambda} so as to avoid branching
        // down the line.
        let mut cx: [Array2<f64>; 2] = [Array2::<f64>::zeros((nbas, 2*nmo)), Array2::<f64>::zeros((nbas, 2*nmo))];
        let mut xc: [Array2<f64>; 2] = [Array2::<f64>::zeros((nbas, 2*nmo)), Array2::<f64>::zeros((nbas, 2*nmo))];
        for mi in 0..2 {
            (cx[mi], xc[mi]) = DiffSpinBuild::build_cx_xc(&mao[mi], s_munu, l_c, g_c, mi);
        }

        // Construct {}^{\Lambda\Gamma} V_0^{m_i, m_j} = \sum_{prqs} ({}^{\Lambda}(pr|qs) -
        // {}^{\Lambda}(ps|qr)) {}^{\Lambda\Gamma} X_{sq}^{m_i} {}^{\Lambda\Gamma}. This can be
        // rewritten (and thus calculated) as V_0^{m_i, m_j} = \sum_{pr} (J_{\mu\nu}^{m_i} - K_{\mu\nu}^{m_i}) 
        // {}^{\Gamma\Lambda} M^{\sigma\tau, m_j}.
        let mut v0 = [0.0f64; 3];
        v0[0] = einsum_ba_ab_real(&jkao[0], &mao[0]);
        if m > 1 {
            v0[1] = 2.0 * einsum_ba_ab_real(&jkao[0], &mao[1]);
            v0[2] = einsum_ba_ab_real(&jkao[1], &mao[1]);
        } else {
            v0[1] = 0.0;
        }

        // Construct {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} intermediates for two electron
        // Hamiltonian matrix elements. These are given as:
        //      {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} = \sum_{pq} {}^{\Lambda\Gamma} Y_{ap}^{m_1}
        //      (\sum_{rs} ({}^{\Lambda}(pr|qs) - {}^{\Lambda}(ps|qr)) {}^{\Lambda\Gamma} X_{sr}^{m_2}) 
        //      {}^{\Lambda\Gamma} X_{qb}^{m_3},
        // where the use of X or Y on the left and righthand sides depends on the ordering of
        // \Lambda and \Gamma. Again using our precomputed quantities we rewrite as:
        //      {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} = \sum_{pq} C_{L,ap}^{m_1} (J_{\mu\nu}^{m_2} - K_{\mu\nu}^{m_2})
        //      C_{R,qb}^{m_3}.
        let mut v: [[[Array2<f64>; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| Array2::<f64>::zeros((2 * nmo, 2 * nmo)))
            })
        });
        let combos: Vec<(usize, usize, usize)> =
            (0..2).flat_map(|mi|
            (0..2).flat_map(move |mj|
            (0..2).map(move |mk| (mi, mj, mk))
        )).collect();
        let blocks: Vec<((usize, usize, usize), Array2<f64>)> =
            combos.into_par_iter().map(|(mi, mj, mk)| {
                let blk = cx[mi].t().dot(&jkao[mk]).dot(&xc[mj]);
                ((mi, mj, mk), blk)
            }).collect();
        for ((mi, mj, mk), blk) in blocks {
            v[mi][mj][mk] = blk;
        }

        // Construct {}^{\Lambda\Gamma} J_{ab,cd}^{m_1,m_2,m_3,m_4} intermediates for two electron
        // Hamiltonian matrix elements. These are given as:
        //      {}^{\Lambda\Gamma} J_{ab,cd}^{m_1,m_2,m_3,m_4} = \sum_{prqs} ({}^{\Lambda}(pr|qs) - {}^{\Lambda}(ps|qr))
        //      {}^{\Lambda\Gamma} Y_{ap}^{m_1} {}^{\Lambda\Gamma} X_{rb}^{m_2} {}^{\Lambda\Gamma} Y_{cq}^{m_3} {}^{\Lambda\Gamma} X_{sd}^{m_4},
        // where the use of X or Y in each part depends on the ordering of \Lambda and \Gamma. Again using our quantities
        // this may instead be calculated as
        //      {}^{\Lambda\Gamma} J_{ab,cd}^{m_1, m_2, m_3, m_4} = \sum_{\mu\nu\tau\sigma} C_{L,a\mu}^{m_1} C_{R,b\nu}^{m_2}
        //      ((\mu\nu|\tau\sigma) - (\mu\sigma|\tau\nu)) C_{L,c\tau}^{m_3} C_{R,d\sigma}^{m_4},
        // which is achieved by antisymmetrising the AO integrals and transforming from AO to MO
        // basis. Only 10 of 16 4D tensors of J need be stored due to symmetry.
        let mut j: [Array4<f64>; 10] = std::array::from_fn(|_| {
            Array4::<f64>::zeros((2 * nmo, 2 * nmo, 2 * nmo, 2 * nmo))
        });
        let combos: [(usize, usize, usize, usize); 10] = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 1), 
                                                          (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1)];
        let blocks: Vec<(usize, Array4<f64>)> = combos.into_par_iter().enumerate().map(|(s, (mi, mj, mk, ml))| {
            let mut blk = eri_ao2mo(eri, &cx[mi], &xc[mj], &cx[mk], &xc[ml]);
            let ex = blk.view().permuted_axes([0, 2, 1, 3]).to_owned().as_standard_layout().to_owned();
            blk -= &ex;
            (s, blk.as_standard_layout().to_owned())
        }).collect();
        for (s, blk) in blocks {
            j[s] = blk;
        }

        Self {x, y, f0h, fh, f0f, ff, v0, v, j, tilde_s_prod, phase, m, nmo}
    }

    /// Calculate the overlap matrix between two sets of occupied orbitals as:
    ///     {}^{\Gamma\Lambda} S_{ij} = \sum_{\mu\nu} ({}^\Gamma C^*)_i^\mu S_{\mu\nu} ({}^\Lambda C)_j^\nu
    /// # Arguments:
    /// - `g_c_occ`: Occupied coefficients ({}^\Gamma C^*)_i^\mu.
    /// - `l_c_occ`: Occupied coefficients ({}^\Lambda C)_j^\nu 
    /// - `s_munu`: AO overlap matrix S_{\mu\nu}.
    /// # Returns
    /// - `Array2<f64>`: Occupied-orbital overlap matrix.
    pub fn calculate_mo_overlap_matrix(l_c_occ: &Array2<f64>, g_c_occ: &Array2<f64>, s_munu: &Array2<f64>) -> Array2<f64> {
        l_c_occ.t().dot(&s_munu.dot(g_c_occ))
    }
    
    /// Perform singular value decomposition on the occupied orbital overlap matrix {}^{\Gamma\Lambda} S_{ij} as:
    ///     {}^{\Gamma\Lambda} \mathbf{S} = \mathbf{U} {}^{\Gamma\Lambda} \mathbf{\tilde{S}} \mathbf{V}^\dagger,
    /// and rotate the occupied coefficients:
    ///     |{}^\Gamma \Psi_i\rangle = \sum_{\mu} {}^\Gamma c_i^\mu U_{ij} |\phi_\mu \rangle.
    ///     |{}^\Lambda \Psi_j\rangle = \sum_{\nu} {}^\Lambda c_j^\nu V_{ij} |\phi_\nu \rangle.
    /// # Arguments:
    /// - `g_c_occ`: Occupied coefficients ({}^\Gamma C^*)_i^\mu.
    /// - `l_c_occ`: Occupied coefficients ({}^\Lambda C)_j^\nu.
    /// - `tol`: Tolerance for the orthonormalisation step.
    /// # Returns
    /// - `(Array1<f64>, Array2<f64>, Array2<f64>, f64)`: Singular values, rotated occupied coefficients
    ///   for \Gamma and \Lambda, and the phase associated with the rotation.
    pub fn perform_ortho_and_svd_and_rotate(s_munu: &Array2<f64>, l_c_occ: &Array2<f64>, g_c_occ: &Array2<f64>, tol: f64) 
                                            -> (Array1<f64>, Array2<f64>, Array2<f64>, f64) {
        
        // Orthonormalise.
        let s_ll = l_c_occ.t().dot(&s_munu.dot(l_c_occ));
        let s_gg = g_c_occ.t().dot(&s_munu.dot(g_c_occ));
        let x_l = loewdin_x_real(&s_ll, true, tol);
        let x_g = loewdin_x_real(&s_gg, true, tol);
        let l_c_occ_ortho = l_c_occ.dot(&x_l);
        let g_c_occ_ortho = g_c_occ.dot(&x_g);

        let lg_s = l_c_occ_ortho.t().dot(&s_munu.dot(&g_c_occ_ortho));

        // SVD.
        let (u, lg_tilde_s, v_dag) = lg_s.svd(true, true).unwrap();
        let u = u.unwrap();
        let v = v_dag.unwrap().t().to_owned();
        
        // Rotate MOs.
        let l_tilde_c = l_c_occ_ortho.dot(&u);
        let g_tilde_c = g_c_occ_ortho.dot(&v);
        
        // Calculate phase associated with rotation.
        let det_u = u.det().unwrap();
        let det_v = v.det().unwrap();
        let ph = det_u * det_v;
        
        (lg_tilde_s, g_tilde_c, l_tilde_c, ph)
    }
    
    /// Form the matrices {}^{\Gamma\Lambda} M^{\sigma\tau, 0} and{}^{\Gamma\Lambda} M^{\sigma\tau, 1} as:
    ///     {}^{\Gamma\Lambda} M^{\sigma\tau, 0} = {}^{\Gamma\Lambda} W^{\sigma\tau} + {}^{\Gamma\Lambda} P^{\sigma\tau} + {}^{\Gamma\Gamma} P^{\sigma\tau}
    ///     {}^{\Gamma\Lambda} M^{\sigma\tau, 1} = {}^{\Gamma\Lambda} P^{\sigma\tau}.
    /// The components {}^{\Gamma\Lambda} W^{\sigma\tau}, {}^{\Gamma\Lambda} P^{\sigma\tau}, 
    /// {}^{\Gamma\Gamma} P^{\sigma\tau} are constructed sequentially and added into the correct
    /// matrix.
    /// # Arguments:
    /// - `gl_tilde_s`: Vector of diagonal single values of {}^{\Gamma\Lambda} \tilde{S}.
    /// - `g_c_tilde_occ`: Rotated occupied coefficients ({}^\Gamma \tilde{C}^*)_i^\mu.
    /// - `l_c_tilde_occ`: Rotated occupied coefficients ({}^\Lambda \tilde{C})_j^\nu.
    /// - `zeros`: Indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S}. 
    /// - `tol`: Tolerance for whether a singular value is considered zero.
    /// # Returns
    /// - `(Array2<f64>, Array2<f64>)`: The M^{0} and M^{1} matrices.
    pub fn construct_m(lg_tilde_s: &Array1<f64>, g_tilde_c_occ: &Array2<f64>, l_tilde_c_occ: &Array2<f64>, zeros: &Vec<usize>, tol: f64) -> (Array2<f64>, Array2<f64>) {
        let nbas = g_tilde_c_occ.nrows();
        let nocc = g_tilde_c_occ.ncols();
        
        // Calculate {}^{\Lambda\Gamma} W^{\sigma\tau} (weighted co-density matrix) as:
        //      {}^{\Lambda\Gamma} W^{\sigma\tau} = \sum_i (^\Lambda \tilde{C})^\sigma_i (1 /
        //      {}^{\Lambda\Gamma} \tilde{S}_i) (^\Gamma \tilde{C}^*)^\tau_i
        //  for all i where {}^{\Lambda\Gamma} \tilde{S}_i != 0. This result is stored in
        //  {}^{\Lambda\Gamma} M^{\sigma\tau, 0}, where the zero indicates this quantity will be
        //  used when m_k = 0. 
        let mut l_tilde_c_occ_scaled = l_tilde_c_occ.clone();
        for k in 0..nocc {
            let s = lg_tilde_s[k];
            if s.abs() > tol {
                let scale = 1.0 / s;
            let mut col = l_tilde_c_occ_scaled.column_mut(k);
            col *= scale;
            } else {
                l_tilde_c_occ_scaled.column_mut(k).fill(0.0);
            }
        }
        let mut lg_m0 = l_tilde_c_occ_scaled.dot(&g_tilde_c_occ.t());
        let mut lg_m1 = Array2::<f64>::zeros((nbas, nbas));
        let mut ll_m0 = Array2::<f64>::zeros((nbas, nbas));
        
        // Calculate {}^{\Lambda\Lambda} P^{\sigma\tau}_k (co-density matrix) as:
        //      {}^{\Lambda\Lambda} P^{\sigma\tau}_k = ({}^\Lambda \tilde{C})^\sigma_k ({}^\Lambda
        //      \tilde{C}^*)^\tau_k
        // for all k where {}^{\Lambda\Lambda} \tilde{S}_k = 0 and sum together to form {}^{\Lambda\Lambda} P^{\sigma\tau}.
        // This result is added to {}^{\Lambda\Gamma} M^{\sigma\tau, 0} which now
        // contains contributions from the \Lambda, \Lambda co-density matrix and \Lambda \Gamma
        // weighted co-density matrix.
        for &k in zeros {
            let l_tilde_c_occ_k = l_tilde_c_occ.column(k).to_owned();
            let outer = l_tilde_c_occ_k.view().insert_axis(Axis(1)).dot(&l_tilde_c_occ_k.view().insert_axis(Axis(0)));
            ll_m0 += &outer;
        }
        lg_m0 += &ll_m0;
        
        // Calculate {}^{\Lambda\Gamma} P^{\sigma\tau}_k (co-density matrix) as:
        //      {}^{\Lambda\Gamma} P^{\sigma\tau}_k = ({}^\Lambda \tilde{C})^\sigma_k ({}^\Gamma
        //      \tilde{C}^*)^\tau_k
        // for all k where {}^{\Lambda\Gamma} \tilde{S}_k = 0 and sum together to form {}^{\Lambda\Gamma} P^{\sigma\tau}.
        // This result is added to {}^{\Lambda\Gamma} M^{\sigma\tau, 0} which now
        // contains the correct contributions to be:
        //  {}^{\Lambda\Gamma} M^{\sigma\tau, 0} = {}^{\Lambda\Gamma} W^{\sigma\tau} + {}^{\Lambda\Gamma} P^{\sigma\tau} + {}^{\Lambda\Lambda} P^{\sigma\tau} 
        //  as required. Similarly we make {}^{\Lambda\Gamma} M^{\sigma\tau, 1} = {}^{\Lambda\Gamma} P^{\sigma\tau}.  
        for &k in zeros {
            let l_tilde_c_occ_k = l_tilde_c_occ.column(k).to_owned();
            let g_tilde_c_occ_k = g_tilde_c_occ.column(k).to_owned(); 
            let outer = l_tilde_c_occ_k.view().insert_axis(Axis(1)).dot(&g_tilde_c_occ_k.view().insert_axis(Axis(0)));
            lg_m1 += &outer;
            lg_m0 += &outer;
        }

        (lg_m0, lg_m1)
    }

    /// Form the matrices {}^{\Gamma\Lambda} X^{\sigma\tau, m_k}_{ij} and {\Gamma\Lambda} Y_{ij}^{m_k} as:
    ///     {}^{\Gamma\Lambda} X_{ij}^{m_k} = \sum_{\mu\nu\sigma\tau} ({}^\Gamma C^*)_i^\mu S_{\mu\nu} 
    ///     (^{\Gammma\Lambda} M^{m_k})^{\sigma\tau} S_{\mu\nu} (^\Lambda C)_j^\nu.
    ///     {}^{\Gammma\Lambda} Y_{ij}^{0} = {\Gamma\Lambda} X_{ij}^{0} - {}^{\Gammma\Lambda} S_{ij}.
    ///     {}^{\Gammma\Lambda} Y_{ij}^{1} = {\Gamma\Lambda} X_{ij}^{1}.
    /// # Arguments:
    /// - `s_munu`: AO overlap matrix.
    /// - `g_c`: Full AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// - `l_c`: Full AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// - `gl_m`: M matrix{}^{\Gamma\Lambda} M^{\sigma\tau, 0} or  {}^{\Gamma\Lambda} M^{\sigma\tau, 1}. 
    /// - `subtract`: Whether to use m_k = 0 or m_k = 1. 
    /// # Returns
    /// - `(Array2<f64>, Array2<f64>)`: The X and Y matrices.
    fn construct_xy(g_c: &Array2<f64>, l_c: &Array2<f64>, s_munu: &Array2<f64>, gl_m: &Array2<f64>, subtract: bool) -> (Array2<f64>, Array2<f64>) {
        let nbas = g_c.nrows();
        let nmo = g_c.ncols();
        
        // Concatenate coefficient matrices into one.
        let mut lg_c = Array2::<f64>::zeros((nbas, 2 * nmo));
        lg_c.slice_mut(s![.., 0..nmo]).assign(l_c);
        lg_c.slice_mut(s![.., nmo..2 * nmo]).assign(g_c);

        // {}^{\Gamma\Lambda} X_{ij}^{m_k} = \sum_{\mu\nu\sigma\tau} ({}^\Gamma C^*)_i^\mu
        // S_{\mu\nu} (^{\Gammma\Lambda} M^{m_k})^{\sigma\tau} S_{\mu\nu} (^\Lambda C)_j^\nu. Note 
        // that this expression if for computing specifically the {}^{\Gamma\Lambda} quadrant of
        // X_{ij}^{m_k}. By using the concatenated coefficient matrices we compute it all in one go. 
        let sm = s_munu.dot(gl_m);
        let sms = sm.dot(s_munu);
        let x = lg_c.t().dot(&sms).dot(&lg_c);
        
        // {}^{\Gammma\Lambda} Y_{ij}^{0} = {\Gamma\Lambda} X_{ij}^{0} - {}^{\Gammma\Lambda} S_{ij}.
        // {}^{\Gammma\Lambda} Y_{ij}^{1} = {\Gamma\Lambda} X_{ij}^{1}.
        let ymiddle = if subtract {&sms - s_munu} else {sms}; 
        let y = lg_c.t().dot(&ymiddle).dot(&lg_c);

        (x, y)
    }
    
    /// Construct the {}^{\Gamma\Lambda} F_0^{m_k} and {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j}
    /// intermediates required for one-body coupling as:
    ///     {}^{\Lambda\Lambda} F_0^{m_i} = \sum_{pq} {}^\Lambda f_{pq} {\Lambda\Lambda} X_{qp}^{m_i}
    /// where {}^{\Lambda} f_{pq} is the required onebody operator in the MO basis for determinant
    /// \Lambda, and:
    ///     {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j} = \sum_{pq} {\Gamma\Lambda} X_{ap}^{m_i}
    ///     {\Lambda\Lambda} f_{pq} {\Lambda\Lambda} X_{qb}^{m_j},
    /// where the use of X or Y and their quadrants depends on the requested ordering of \Lambda,
    /// \Gamma in {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j}.
    /// # Arguments:
    /// - `g_c`: Full AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// - `l_c`: Full AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// - `h_munu`: One-electron core AO hamiltonian.
    /// - `x`: {}^{\Gamma\Lambda} X_{ij}^{m_k}.
    /// - `y`: {}^{\Gamma\Lambda} Y_{ij}^{m_k}.
    /// # Returns
    /// - `(f64, Array2<f64>)`: Scalar F_0^{m_k} and matrix F_{ab}^{m_i,m_j}.
    pub fn construct_f(l_c: &Array2<f64>, h_munu: &Array2<f64>, x: &Array2<f64>, y: &Array2<f64>) -> (f64, Array2<f64>) {
        let nmo = l_c.ncols();
        let ll_h = l_c.t().dot(h_munu).dot(l_c);
        
        let ll_x = x.slice(s![0..nmo, 0..nmo]).to_owned();  
        let gl_x = x.slice(s![nmo..2 * nmo, 0..nmo]).to_owned();   
        let ll_y = y.slice(s![0..nmo, 0..nmo]).to_owned();          
        let lg_y = y.slice(s![0..nmo, nmo..2 * nmo]).to_owned(); 

        let ll_f0 = einsum_ba_ab_real(&ll_x, &ll_h);

        let ll_f = ll_y.dot(&ll_h).dot(&ll_x);
        let gl_f = gl_x.dot(&ll_h).dot(&ll_x);
        let lg_f = ll_y.dot(&ll_h).dot(&lg_y);
        let gg_f = gl_x.dot(&ll_h).dot(&lg_y);
        
        let mut f = Array2::<f64>::zeros((2 * nmo, 2 * nmo));
        f.slice_mut(s![0..nmo, 0..nmo]).assign(&ll_f);              
        f.slice_mut(s![0..nmo, nmo..2 * nmo]).assign(&lg_f);        
        f.slice_mut(s![nmo..2 * nmo, 0..nmo]).assign(&gl_f);          
        f.slice_mut(s![nmo..2 * nmo, nmo..2 * nmo]).assign(&gg_f); 

        (ll_f0, f)
    }
    
    /// Calculate the Coulomb contraction J^{m_k}_{\mu\nu} required for the two electron
    /// intermediates as:
    ///     J^{m_k}_{\mu\nu} = \sum_{\sigma\tau} ({}^{\Lambda}(\mu\nu|\sigma\tau)) {}^{\Gamma\Lambda} M^{\mu\nu, m_k}
    /// where {}^{\Gamma\Lambda} M^{m_k} is the AO-space M matrix. 
    /// # Arguments:
    /// - `eri`: AO basis ERIs (not-antisymmetrised).
    /// - `m`: {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
    /// # Returns
    /// - `Array2<f64>`: Coulomb contraction matrix.
    fn build_j_coulomb(eri: &Array4<f64>, m: &Array2<f64>) -> Array2<f64> {
        let n = m.nrows();
        let mut j = Array2::<f64>::zeros((n, n));

        j.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(s, mut row)| {
            for t in 0..n {
                let mut acc = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        acc += eri[(s, t, mu, nu)] * m[(mu, nu)];
                    }
                }
                row[t] = acc;
            }
        });
        j
    }

    /// Calculate the Coulomb contraction K^{m_k}_{\mu\nu} required for the two electron
    /// intermediates as:
    ///     K^{m_k}_{\mu\nu} = \sum_{\sigma\tau} ({}^{\Lambda}(\mu\sigma|\tau\nu)) {}^{\Gamma\Lambda} M^{\mu\nu, m_k}
    /// where {}^{\Gamma\Lambda} M^{m_k} is the AO-space M matrix. 
    /// # Arguments:
    /// - `eri`: AO basis ERIs (not-antisymmetrised).
    /// - `m`: {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
    /// # Returns
    /// - `Array2<f64>`: Exchange contraction matrix.
    fn build_k_exchange(eri: &Array4<f64>, m: &Array2<f64>) -> Array2<f64> {
        let n = m.nrows();
        let mut k = Array2::<f64>::zeros((n, n));

        k.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(s, mut row)| {
            for t in 0..n {
                let mut acc = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        acc += eri[(s, mu, nu, t)] * m[(mu, nu)];
                    }
                }
                row[t] = acc;
            }
        });
        k
    }
}

impl DiffSpinBuild {
    /// Constructor for the WicksReferencePair object of DiffSpin which contains the precomputed values
    /// per pair of reference determinants required to evaluate arbitrary excited determinants 
    /// in O(1) time when the excitations are of different spin. As such, the quantities here are
    /// lesser as we only need to evaluate two electron terms with these intermediates.
    /// # Arguments:
    /// - `eri`: Electron repulsion integrals. 
    /// - `h_munu`: AO core Hamiltonian.
    /// - `s_munu`: AO overlap matrix.
    /// - `g_ca`: Spin alpha AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// - `g_cb`: Spin beta AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// - `l_ca`: Spin alpha AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// - `l_cb`: Spin beta AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// - `goa`: Alpha occupancy vector for |^\Gamma\Psi\rangle.
    /// - `gob`: Beta occupancy vector for |^\Gamma\Psi\rangle.
    /// - `loa`: Alpha occupancy vector for |^\Lambda\Psi\rangle.
    /// - `lob`: Beta occupancy vector for |^\Lambda\Psi\rangle.
    /// - `tol`: Tolerance for whether a number is considered zero.
    /// # Returns
    /// - `DiffSpinBuild`: Precomputed different-spin Wick's intermediates for the reference pair.
    pub fn new(eri: &Array4<f64>, s_munu: &Array2<f64>, g_ca: &Array2<f64>, g_cb: &Array2<f64>, l_ca: &Array2<f64>, l_cb: &Array2<f64>, 
           goa: u128, gob: u128, loa: u128, lob: u128, tol: f64) -> Self {
        let nmo = g_ca.ncols();

        let l_ca_occ = occ_coeffs(l_ca, loa);
        let g_ca_occ = occ_coeffs(g_ca, goa);
        let l_cb_occ = occ_coeffs(l_cb, lob);
        let g_cb_occ = occ_coeffs(g_cb, gob);

        // SVD and rotate the occupied orbitals per spin.
        let (tilde_sa_occ, g_tilde_ca_occ, l_tilde_ca_occ, _phase) = SameSpinBuild::perform_ortho_and_svd_and_rotate(s_munu, &l_ca_occ, &g_ca_occ, 1e-20);
        let (tilde_sb_occ, g_tilde_cb_occ, l_tilde_cb_occ, _phase) = SameSpinBuild::perform_ortho_and_svd_and_rotate(s_munu, &l_cb_occ, &g_cb_occ, 1e-20);
        
        // Find indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S} and count them per spin.
        // No longer writing per spin from here onwards, hopefully it is clear.
        let zerosa: Vec<usize> = tilde_sa_occ.iter().enumerate().filter_map(|(k, &sk)| if sk.abs() <= tol {Some(k)} else {None}).collect();
        let zerosb: Vec<usize> = tilde_sb_occ.iter().enumerate().filter_map(|(k, &sk)| if sk.abs() <= tol {Some(k)} else {None}).collect();

        // Construct the {}^{\Gamma\Lambda} M^{\sigma\tau, 0} and {}^{\Gamma\Lambda} M^{\sigma\tau, 1} matrices.
        let (m0a, m1a) = SameSpinBuild::construct_m(&tilde_sa_occ, &l_tilde_ca_occ, &g_tilde_ca_occ, &zerosa, tol);
        let (m0b, m1b) = SameSpinBuild::construct_m(&tilde_sb_occ, &l_tilde_cb_occ, &g_tilde_cb_occ, &zerosb, tol);
        let ma = [&m0a, &m1a];
        let mb = [&m0b, &m1b];
        
        // Construct only the Coulomb contraction of ERIs with  {}^{\Gamma\Lambda} M^{\sigma\tau, m_k}, 
        // {}^{\Gamma\Lambda} J_{\mu\nu}^{m_k}. No exchange here due to differing spins.
        let ja = [SameSpinBuild::build_j_coulomb(eri, ma[0]), SameSpinBuild::build_j_coulomb(eri, ma[1])];
        let jb = [SameSpinBuild::build_j_coulomb(eri, mb[0]), SameSpinBuild::build_j_coulomb(eri, mb[1])];

        //let tilde_sa_prod = tilde_sa_occ.iter().filter(|&&x| x.abs() > tol).product::<f64>();
        //let tilde_sb_prod = tilde_sb_occ.iter().filter(|&&x| x.abs() > tol).product::<f64>();
        //if maxabs(ma[0]) > 1e5 || maxabs(ma[1]) > 1e5 {println!("WARNING: HUGE MA")};
        //println!("tilde_sa_occ: {:.3e}, tilde_sa_prod: {:.3e}", tilde_sa_occ, tilde_sa_prod);
        //println!("M0a max: {:.3e}, frob: {:.3e} | M1a max: {:.3e}, frob: {:.3e}", maxabs(ma[0]), frob(ma[0]), maxabs(ma[1]), frob(ma[1]));
        //println!("J0a max: {:.3e}, frob: {:.3e} | J1a max: {:.3e}, frob: {:.3e}", maxabs(&ja[0]), frob(&ja[0]), maxabs(&ja[1]), frob(&ja[1]));
        //println!();

        //if maxabs(mb[0]) > 1e5 || maxabs(mb[1]) > 1e5 {println!("WARNING: HUGE MB")};
        //println!("tilde_sb_occ: {:.3e}, tilde_sb_prod: {:.3e}", tilde_sb_occ, tilde_sb_prod);
        //println!("M0b max: {:.3e}, frob: {:.3e} | M1b max: {:.3e}, frob: {:.3e}", maxabs(&mb[0]), frob(&mb[0]), maxabs(&mb[1]), frob(&mb[1]));
        //println!("J0b max: {:.3e}, frob: {:.3e} | J1b max: {:.3e}, frob: {:.3e}", maxabs(&jb[0]), frob(&jb[0]), maxabs(&jb[1]), frob(&jb[1]));
        //println!();

        // Construct {}^{\Lambda\Gamma} V_{ab,0}^{m_i, m_j} = \sum_{prqs} ({}^{\Lambda}(pr|qs)) X_{sq}^{m_i} {}^{\Lambda\Gamma}. 
        // This can be rewritten (and thus calculated) as V_{ab, 0}^{m_i, m_j} = \sum_{pr} (J_{\mu\nu}^{m_i}) {}^{\Gamma\Lambda} M^{\sigma\tau, m_j}.
        // This is directly analogous to {}^{\Lambda\Gamma} V_0^{m_i, m_j} in the same spin case
        // but with exchange omitted.
        let mut vab0 = [[0.0f64; 2]; 2]; 
        let mut vba0 = [[0.0f64; 2]; 2]; 
        for i in 0..2 {
            for j in 0..2 {
                vab0[i][j] = einsum_ba_ab_real(&ja[i], mb[j]); 
                vba0[j][i] = einsum_ba_ab_real(&jb[j], ma[i]); 
            }
        }
           
        // Calculate the left and right factorisations of the {}^{\Gamma\Lambda} X_{ij}^{m_k} and 
        // {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices, {}^{\Gamma\Lambda} so as to avoid branching
        // down the line. Again analogous to the same spin case but with spin resolved quantities.
        let (cx_a0, xc_a0) = Self::build_cx_xc(ma[0], s_munu, l_ca, g_ca, 0);
        let (cx_a1, xc_a1) = Self::build_cx_xc(ma[1], s_munu, l_ca, g_ca, 1);
        let (cx_b0, xc_b0) = Self::build_cx_xc(mb[0], s_munu, l_cb, g_cb, 0);
        let (cx_b1, xc_b1) = Self::build_cx_xc(mb[1], s_munu, l_cb, g_cb, 1);
        let cx_a = [&cx_a0, &cx_a1];
        let xc_a = [&xc_a0, &xc_a1];
        let cx_b = [&cx_b0, &cx_b1];
        let xc_b = [&xc_b0, &xc_b1];
        
        // Construct {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} intermediates for two electron
        // Hamiltonian matrix elements. These are given as:
        //      {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} = \sum_{pq} {}^{\Lambda\Gamma} Y_{ap}^{m_1}
        //      (\sum_{rs} ({}^{\Lambda}(pr|qs)) {}^{\Lambda\Gamma} X_{sr}^{m_2}) {}^{\Lambda\Gamma} X_{qb}^{m_3},
        // where the use of X or Y on the left and righthand sides depends on the ordering of
        // \Lambda and \Gamma. Again using our precomputed quantities we rewrite as:
        //      {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} = \sum_{pq} C_{L,ap}^{m_1} (J_{\mu\nu}^{m_2})
        //      C_{R,qb}^{m_3}.
        //  Once more this is analogous to the SameSpin case but with exchange removed. The
        //  similarities here hint at a possible generalisation of the code.
        let mut vab: [[[Array2<f64>; 2]; 2]; 2] = std::array::from_fn(|_| {std::array::from_fn(|_| std::array::from_fn(|_| Array2::<f64>::zeros((2*nmo, 2*nmo))))});
        let mut vba: [[[Array2<f64>; 2]; 2]; 2] = std::array::from_fn(|_| {std::array::from_fn(|_| std::array::from_fn(|_| Array2::<f64>::zeros((2*nmo, 2*nmo))))});
        let combos: Vec<(usize, usize, usize)> =
            (0..2).flat_map(|ma0|
            (0..2).flat_map(move |mb0|
            (0..2).map(move |mk| (ma0, mb0, mk))
        )).collect();
        let vabblocks: Vec<((usize, usize, usize), Array2<f64>)> =
            combos.clone().into_par_iter().map(|(ma0, mb0, mak)| {
                let blk = cx_a[ma0].t().dot(&jb[mb0]).dot(xc_a[mak]);
                ((ma0, mb0, mak), blk)
            }).collect();
        for ((ma0, mb0, mak), blk) in vabblocks {
            vab[ma0][mb0][mak] = blk;
        }
        let vbablocks: Vec<((usize, usize, usize), Array2<f64>)> =
            combos.into_par_iter().map(|(ma0, mb0, mbk)| {
                let blk = cx_b[mb0].t().dot(&ja[ma0]).dot(xc_b[mbk]);
                ((mb0, ma0, mbk), blk)
            }).collect();
        for ((mb0, ma0, mbk), blk) in vbablocks {
            vba[mb0][ma0][mbk] = blk;
        }

        // Construct {}^{\Lambda\Gamma} II_{ab,cd}^{m_1,m_2,m_3,m_4} intermediates for two electron
        // Hamiltonian matrix elements. These are given as:
        //      {}^{\Lambda\Gamma} II_{ab,cd}^{m_1,m_2,m_3,m_4} = \sum_{prqs} ({}^{\Lambda}(pr|qs))
        //      {}^{\Lambda\Gamma} Y_{ap}^{m_1} {}^{\Lambda\Gamma} X_{rb}^{m_2} {}^{\Lambda\Gamma} Y_{cq}^{m_3} {}^{\Lambda\Gamma} X_{sd}^{m_4},
        // where the use of X or Y in each part depends on the ordering of \Lambda and \Gamma. Again using our quantities
        // this may instead be calculated as
        //      {}^{\Lambda\Gamma} II_{ab,cd}^{m_1, m_2, m_3, m_4} = \sum_{\mu\nu\tau\sigma} C_{L,a\mu}^{m_1} C_{R,b\nu}^{m_2}
        //      ((\mu\nu|\tau\sigma)) C_{L,c\tau}^{m_3} C_{R,d\sigma}^{m_4},
        // which is achieved by antisymmetrising the AO integrals and transforming from AO to MO
        // basis. This is unsuprisngly analogous to the 4-index J tensor in SameSpin.
        let mut iiab: [[[[Array4<f64>; 2]; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| Array4::<f64>::zeros((2 * nmo, 2 * nmo, 2 * nmo, 2 * nmo))))
            })
        });

        let combos: Vec<(usize, usize, usize, usize)> =
            (0..2).flat_map(|mi| (0..2).flat_map(move |mj|
            (0..2).flat_map(move |mk| (0..2).map(move |ml| (mi, mj, mk, ml)))
        )).collect();

        let blocks: Vec<((usize, usize, usize, usize), Array4<f64>)> = combos.into_par_iter().map(|(ma0, maj, mb0, mbj)| {
            let blk = eri_ao2mo(eri, cx_a[ma0], xc_a[maj], cx_b[mb0], xc_b[mbj]).as_standard_layout().to_owned();
            ((ma0, maj, mb0, mbj), blk)
        }).collect();

        for ((ma0, maj, mb0, mbj), blk) in blocks {
            iiab[ma0][maj][mb0][mbj] = blk;
        }

        Self {vab0, vab, vba0, vba, iiab}
    }
        
    /// Build the left and right factorisations of the {}^{\Gamma\Lambda} X_{ij}^{m_k} and 
    /// {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices such that our intermediates can be computed more
    /// easily.
    /// # Arguments:
    /// - `m`: {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
    /// - `s`: AO overlap matrix S_{\mu\nu}.
    /// - `cx`: AO coefficient matrix for \Lambda. Should be renamed.
    /// - `cw`: AO coefficient matrix for \Gamma. Should be renamed.
    /// - `i`: Selector for m being 0 or 1.
    /// # Returns
    /// - `(Array2<f64>, Array2<f64>)`: Left and right factorisation matrices.
    fn build_cx_xc(m: &Array2<f64>, s: &Array2<f64>, cx: &Array2<f64>, cw: &Array2<f64>, i: usize) -> (Array2<f64>, Array2<f64>) {
        let nao = cx.nrows();
        let nmo = cx.ncols();
        let mut cx_out = Array2::<f64>::zeros((nao, 2 * nmo));
        let mut xc_out = Array2::<f64>::zeros((nao, 2 * nmo));

        let one_minus_i = (1 - i) as f64;

        let ms = m.dot(s);
        let mts = m.t().dot(s);

        cx_out.slice_mut(s![.., 0..nmo]).assign(&(mts.dot(cx) - &(cx * one_minus_i)));
        xc_out.slice_mut(s![.., 0..nmo]).assign(&ms.dot(cx));

        cx_out.slice_mut(s![.., nmo..2 * nmo]).assign(&mts.dot(cw));
        xc_out.slice_mut(s![.., nmo..2 * nmo]).assign(&(ms.dot(cw) - &(cw * one_minus_i)));

        (cx_out, xc_out)
    }
}

/// Extract bit `k` from a distribution bitstring. Each possible bitstring assigns `m` zero-overlap
/// orbital pairs to a given contraction. The bits are also used to select whether a given
/// contraction uses the `0` or `1` branch of intermediates.
/// # Arguments:
/// - `bits`: Bitstring encoding a zero distribution.
/// - `k`: Index of the bit to extract.
/// # Returns
/// - `usize`: `0` or `1` depending on the selected branch for contraction `k`.
#[inline(always)]
fn bit(bits: u64, k: usize) -> usize {
    ((bits >> k) & 1) as usize
}

/// Calculate determinant correction obtained by replacing one column of a determinant. If one
/// column of determinant `D` is replaced by a new column, the determinant update correction `C` may be
/// written as C = \sum_r (N_r - O_r) * A_r, where N_r is the new entry in row `r` of replacement
/// column, O_r is the old entry and A_r is the cofactor of row `r` in that column. 
/// # Arguments:
/// - `n`: Dimension of the determinant.
/// - `old`: Row-major storage of the original determinant.
/// - `cof`: Row-major storage of the adjugate-transpose / cofactor matrix.
/// - `col`: Column index to replace.
/// - `new_at`: Closure returning the new value for row `r` in the replacement column.
/// # Returns
/// - `f64`: Cofactor contraction for the chosen column replacement.
#[inline(always)]
fn column_replacement_correction(n: usize, old: &[f64], cof: &[f64], col: usize, mut new_at: impl FnMut(usize) -> f64) -> f64 {
    let mut correction = 0.0;
    for r in 0..n {
        let i = idx(n, r, col);
        correction += (new_at(r) - old[i]) * cof[i];
    }
    correction
}

/// Form same spin mixed contraction determinants for all allowed bitstrings. For a same-spin matrix element 
/// with excitation rank `l` we require a sum over all possible ways to distribute `m` zero-overlap 
/// orbital pairs across contractions. In determinant form this is a sum over mixed determinants 
/// obtained by choosing columns from the `det0` and `det1` branch.
/// # Arguments:
/// - `w`: Same-spin view containing the number of zero-overlap pairs `m`.
/// - `l`: Excitation rank.
/// - `pbits`: Number of leading bits reserved for operator-specific selectors
///   before the determinant-column bits begin.
/// - `scratch`: Wick's scratch space containing `det0`, `det1`, and `det_mix`.
/// - `f`: Closure applied once for each mixed determinant.
/// # Returns
/// - `()`: Writes the mixed determinant into `scratch.det_mix` and calls `f`.
#[inline(always)]
fn mix_dets_same(w: &SameSpinView<'_>, l: usize, pbits: usize, scratch: &mut WickScratch, mut f: impl FnMut(u64, &mut WickScratch),) {
    for_each_m_combination(l + pbits, w.m, |bits| {
        let cbits = bits >> pbits;
        mix_columns(scratch.det_mix.as_mut_slice(), scratch.det0.as_slice(), scratch.det1.as_slice(), l, cbits);
        f(bits, scratch);
    });
}

/// Get determinant and adjugate transpose of each same-spin determinant corresponding to an allowed
/// bitstring. Mixes determinants for each bitstring and calls determinant routines.
/// # Arguments:
/// - `w`: Same-spin Wick's view.
/// - `l`: Determinant dimension.
/// - `pbits`: Number of non-column selector bits at the front of the bitstring.
/// - `scratch`: Wick's scratch space containing `det0`, `det1`, and `det_mix`.
/// - `scratch`: Scratch space for mixed determinant, inverse workspace, and cofactors.
/// - `tol`: Singularity threshold.
/// - `f`: Closure receiving the packed bitstring, scratch space, and determinant value.
/// # Returns
/// - `()`: Calls `f` only for nonsingular mixed determinants.
#[inline(always)]
fn get_det_adjt_same(w: &SameSpinView<'_>, l: usize, pbits: usize, scratch: &mut WickScratch, tol: f64, mut f: impl FnMut(u64, &mut WickScratch, f64)) {
    mix_dets_same(w, l, pbits, scratch, |bits, scratch| {
        if let Some(det_det) = adjugate_transpose(scratch.adjt_det.as_mut_slice(), scratch.invs.as_mut_slice(), scratch.lu.as_mut_slice(), scratch.det_mix.as_slice(), l, tol) {
            f(bits, scratch, det_det);
        }
    });
}

/// Get determinant and adjugate transpose of each different-spin determinant corresponding to an allowed
/// bitstring. Mixes determinants for each bitstring and calls determinant routines.
/// # Arguments:
/// - `w`: Diff-spin pair Wick's view.
/// - `l`: Determinant dimension.
/// - `pbits`: Number of non-column selector bits at the front of the bitstring.
/// - `scratch`: Wick's scratch space containing `det0`, `det1`, and `det_mix`.
/// - `scratch`: Scratch space for mixed determinant, inverse workspace, and cofactors.
/// - `tol`: Singularity threshold.
/// - `f`: Closure receiving the packed bitstring, scratch space, and determinant value.
/// # Returns
/// - `()`: Calls `f` only for nonsingular mixed determinants.
#[inline(always)]
fn get_det_adjt_diff(w: &WicksPairView<'_>, la: usize, lb: usize, scratch: &mut WickScratch, tol: f64, mut f: impl FnMut(u64, u64, &mut WickScratch, f64, f64)) {
    for_each_m_combination(la + 1, w.aa.m, |bits_a| {
        let inda = bits_a >> 1;
        mix_columns(scratch.deta_mix.as_mut_slice(), scratch.deta0.as_slice(), scratch.deta1.as_slice(), la, inda,);

        if let Some(det_a) = adjugate_transpose(scratch.adjt_deta.as_mut_slice(), scratch.invsla.as_mut_slice(), scratch.lua.as_mut_slice(), scratch.deta_mix.as_slice(), la, tol) {
            for_each_m_combination(lb + 1, w.bb.m, |bits_b| {

                let indb = bits_b >> 1;
                mix_columns(scratch.detb_mix.as_mut_slice(), scratch.detb0.as_slice(), scratch.detb1.as_slice(), lb, indb);

                if let Some(det_b) = adjugate_transpose(scratch.adjt_detb.as_mut_slice(), scratch.invslb.as_mut_slice(), scratch.lub.as_mut_slice(), scratch.detb_mix.as_slice(), lb, tol) {
                    f(bits_a, bits_b, scratch, det_a, det_b);
                }
            });
        }
    });
}

/// Form the `l - 1` by `l - 1` minor of a determinant and find its adjugate transpose.
/// # Arguments:
/// - `full`: Row-major storage of the determinant.
/// - `l`: Dimension of the determinant.
/// - `i`: Row to remove.
/// - `j`: Column to remove.
/// - `minorb`: Scratch storage for the minor determinant.
/// - `adjtb`: Scratch storage for the adjugate-transpose of the minor.
/// - `invsb`: Scratch storage for inverse-related data.
/// - `lun`: Scratch storage for the LU factorization.
/// - `tol`: Singularity threshold.
/// - `f`: Closure receiving the minor dimension, minor entries, cofactors, and determinant.
/// # Returns
/// - `()`: Calls `f` only if the minor determinant is nonsingular.
#[inline(always)]
fn minor_adjt(full: &[f64], l: usize, i: usize, j: usize, minorb: &mut Vec2, adjtb: &mut Vec2, invsb: &mut Vec1, lub: &mut Vec2, tol: f64, mut f: impl FnMut(usize, &[f64], &[f64], f64)) {
    let lm1 = l.saturating_sub(1);
    minor(minorb.as_mut_slice(), full, l, i, j);

    if let Some(det_minor) = adjugate_transpose(adjtb.as_mut_slice(), invsb.as_mut_slice(), lub.as_mut_slice(), minorb.as_slice(), lm1, tol) {
        f(lm1, minorb.as_slice(), adjtb.as_slice(), det_minor);
    }
}

/// Construct the row and column indices used for a contraction determinant.
/// Indices are written in the concatenated orbital space `[Lambda orbitals; Gamma orbitals]`, 
/// so any Gamma index is offset by `nmo`.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra (`Lambda`) determinant.
/// - `g_ex`: Excitation defining the ket (`Gamma`) determinant.
/// - `nmo`: Number of molecular orbitals in a single determinant.
/// - `rows`: Output row indices.
/// - `cols`: Output column indices.
/// # Returns
/// - `()`: Writes the contraction determinant indices into `rows` and `cols`.
#[inline(always)]
fn construct_determinant_indices(l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, nmo: usize, rows: &mut Vec<usize>, cols: &mut Vec<usize>) {
    let nl = l_ex.holes.len();
    let ng = g_ex.holes.len();

    rows.clear();
    cols.clear();
    rows.reserve(nl + ng);
    cols.reserve(nl + ng);

    if nl == 0 && ng == 0 {
        return;
    }

    if nl > 0 && ng == 0 {
        for &a in &l_ex.parts {
            rows.push(a);
        }
        for &i in &l_ex.holes {
            cols.push(i);
        }
        return;
    }

    if nl == 0 && ng > 0 {
        for &i in &g_ex.holes {
            rows.push(nmo + i);
        }
        for &a in &g_ex.parts {
            cols.push(nmo + a);
        }
        return;
    }

    for &a in &l_ex.parts {
        rows.push(a);
    }
    for &i in &g_ex.holes {
        rows.push(nmo + i);
    }
    for &i in &l_ex.holes {
        cols.push(i);
    }
    for &a in &g_ex.parts {
        cols.push(nmo + a);
    }
}

/// Call `f` for every bitstring of length `l` containing exactly `m` set bits.
/// These bitstrings enumerate all allowed zero-distribution patterns.
/// # Arguments:
/// - `l`: Bitstring length.
/// - `m`: Number of set bits required.
/// - `f`: Callback applied to each valid bitstring.
/// # Returns
/// - `()`: Calls `f` once for each valid combination.
#[inline(always)]
fn for_each_m_combination(l: usize, m: usize, mut f: impl FnMut(u64)) {
    if m > l {
        return;
    }
    if m == 0 {
        f(0);
        return;
    }
    if m == l {
        f((1u64 << l) - 1);
        return;
    }

    let limit = 1u64 << l;
    let mut x = (1u64 << m) - 1;

    while x < limit {
        f(x);

        let c = x & x.wrapping_neg();
        let r = x + c;
        x = (((r ^ x) >> 2) / c) | r;
    }
}

/// All the same spin routines require a number of the same quantities. It is more efficient to
/// precompute them once here rather than inside each matrix element routine.
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle. 
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `()`: Prepares the shared same-spin scratch quantities in place.
pub fn prepare_same(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) { 
    let l = l_ex.holes.len() + g_ex.holes.len();
    scratch.ensure_same(l);

    construct_determinant_indices(l_ex, g_ex, w.nmo, &mut scratch.rows, &mut scratch.cols);

    let x0 = w.x(0); 
    let y0 = w.y(0);
    let x1 = w.x(1); 
    let y1 = w.y(1);

    // Build two full contraction determinants each having exclusively (X0, Y0) or (X1, Y1). 
    build_d(scratch.det0.as_mut_slice(), l, &x0, &y0, scratch.rows.as_slice(), scratch.cols.as_slice());
    build_d(scratch.det1.as_mut_slice(), l, &x1, &y1, scratch.rows.as_slice(), scratch.cols.as_slice());
}

/// Calculate overlap matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. Utilises
/// a sum over possible ways to distribute zeros across the columns of the L by L determinant.
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle. 
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element.
pub fn lg_overlap(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) -> f64 {
    let l = l_ex.holes.len() + g_ex.holes.len();
    if w.m > l {
        return 0.0;
    }

    if w.m == 0 {
        return w.phase * w.tilde_s_prod * det(scratch.det0.as_slice(), l).unwrap_or(0.0);
    }
    if w.m == l {
        return w.phase * w.tilde_s_prod * det(scratch.det1.as_slice(), l).unwrap_or(0.0);
    }

    let mut acc = 0.0;
    mix_dets_same(w, l, 0, scratch, |_, scratch| {
        if let Some(d) = det(scratch.det_mix.as_slice(), l) {
            acc += d;
        }
    });
    w.phase * w.tilde_s_prod * acc
}

/// Calculate one electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: One-electron Hamiltonian matrix element.
#[inline(always)]
pub fn lg_h1(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    lg_one_body(w, l_ex, g_ex, scratch, tol, OneBody::H1)
}

/// Calculate one electron Fock matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: One-electron Fock matrix element.
#[inline(always)]
pub fn lg_f(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64,) -> f64 {
    lg_one_body(w, l_ex, g_ex, scratch, tol, OneBody::Fock)
}

/// Calculate one body matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: One-electron Hamiltonian matrix element.
#[inline(always)]
fn lg_one_body(w: &SameSpinView<'_>, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64, ob: OneBody,) -> f64 {
    let l = l_ex.holes.len() + g_ex.holes.len();
    if w.m > l + 1 {
        return 0.0;
    }
    let mut acc = 0.0;

    get_det_adjt_same(w, l, 1, scratch, tol, |bits, scratch, det_det| {
        let mi = bit(bits, 0);
        let mut contrib = det_det * one_body_scalar(w, ob, mi);

        for b in 0..l {
            let mj = bit(bits, b + 1);
            let cb = scratch.cols[b];
            let f = one_body_block(w, ob, mi, mj);

            let corr = column_replacement_correction(l, scratch.det_mix.as_slice(), scratch.adjt_det.as_slice(), b, |r| f[(cb, scratch.rows[r])]);
            contrib -= det_det + corr;
        }
        acc += contrib;
    });
    w.phase * w.tilde_s_prod * acc
}

/// Calculate the same-spin two electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Same-spin two-electron Hamiltonian matrix element.
#[inline(always)]
pub fn lg_h2_same(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    let l = l_ex.holes.len() + g_ex.holes.len();
    if w.m > l + 2 {
        return 0.0;
    }
    let mut acc = 0.0;

    get_det_adjt_same(w, l, 2, scratch, tol, |bits, scratch, det_det| {
        let m1 = bit(bits, 0);
        let m2 = bit(bits, 1);

        let mut contrib = w.v0[m1 + m2] * det_det;

        for k in 0..l {
            let mk = bit(bits, k + 2);
            let v_t = w.v_t(m1, m2, mk);
            let ck = scratch.cols[k];

            let corr = column_replacement_correction(l, scratch.det_mix.as_slice(), scratch.adjt_det.as_slice(), k, |r| v_t[(ck, scratch.rows[r])],);
            contrib -= 2.0 * (det_det + corr);
        }

        for i in 0..l {
            for j in 0..l {
                let phase = if ((i + j) & 1) == 0 {1.0} else {-1.0};
                let ri_fixed = scratch.rows[i];
                let cj_fixed = scratch.cols[j];
                let mj = bit(bits, j + 2);

                minor_adjt(scratch.det_mix.as_slice(), l, i, j, &mut scratch.det_mix2, &mut scratch.adjt_det2, &mut scratch.invslm1, 
                          &mut scratch.lu, tol, |lm1, det_minor, cof_minor, det_det2| {

                    for k2 in 0..lm1 {
                        let k_full = if k2 < j {k2} else {k2 + 1};
                        let mk = bit(bits, k_full + 2);
                        let (slot, swap) = jslot(m1, m2, mk, mj);
                        let j4 = w.j(slot);

                        if !swap {
                            slice4ijrc(scratch.jslice_full.as_mut_slice(), l, &j4, scratch.rows.as_slice(), scratch.cols.as_slice(), ri_fixed, cj_fixed);
                        } else {
                            slice4rcij(scratch.jslice_full.as_mut_slice(), l, &j4, scratch.rows.as_slice(), scratch.cols.as_slice(), ri_fixed, cj_fixed);
                        }

                        minor(scratch.jslice2.as_mut_slice(), scratch.jslice_full.as_slice(), l, i, j);
                        let corr = column_replacement_correction(lm1, det_minor, cof_minor, k2, |r| scratch.jslice2.as_slice()[idx(lm1, r, k2)]);
                        contrib += phase * (det_det2 + corr);
                        }
                    },
                );
            }
        }
        acc += contrib;
    });
    w.phase * w.tilde_s_prod * acc
}
    
/// Calculate the different-spin two electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// - `w`: Same-spin and different-spin Wick's reference pair intermediates.
/// - `l_ex_a`: Spin alpha excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex_a`: Spin alpha excitation array for |{}^\Gamma \Psi\rangle.
/// - `l_ex_b`: Spin beta excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex_b`: Spin beta excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Different-spin two-electron Hamiltonian matrix element.
#[inline(always)]
pub fn lg_h2_diff(w: &WicksPairView, l_ex_a: &ExcitationSpin, g_ex_a: &ExcitationSpin, l_ex_b: &ExcitationSpin, g_ex_b: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    let la = l_ex_a.holes.len() + g_ex_a.holes.len();
    let lb = l_ex_b.holes.len() + g_ex_b.holes.len();

    if w.aa.m > la + 1 {return 0.0;}
    if w.bb.m > lb + 1 {return 0.0;}
    scratch.ensure_diff(la, lb);

    construct_determinant_indices(l_ex_a, g_ex_a, w.aa.nmo, &mut scratch.rows_a, &mut scratch.cols_a);
    construct_determinant_indices(l_ex_b, g_ex_b, w.bb.nmo, &mut scratch.rows_b, &mut scratch.cols_b);

    build_d(scratch.deta0.as_mut_slice(), la, &w.aa.x(0), &w.aa.y(0), &scratch.rows_a, &scratch.cols_a);
    build_d(scratch.deta1.as_mut_slice(), la, &w.aa.x(1), &w.aa.y(1), &scratch.rows_a, &scratch.cols_a);
    build_d(scratch.detb0.as_mut_slice(), lb, &w.bb.x(0), &w.bb.y(0), &scratch.rows_b, &scratch.cols_b);
    build_d(scratch.detb1.as_mut_slice(), lb, &w.bb.x(1), &w.bb.y(1), &scratch.rows_b, &scratch.cols_b);

    let mut acc = 0.0;

    get_det_adjt_diff(w, la, lb, scratch, tol, |bits_a, bits_b, scratch, det_deta, det_detb| {
        let ma0 = bit(bits_a, 0);
        let mb0 = bit(bits_b, 0);
        let mut contrib = w.ab.vab0[ma0][mb0] * det_deta * det_detb;

        for k in 0..la {
            let mak = bit(bits_a, k + 1);
            let v_t = w.ab.vab_t(ma0, mb0, mak);
            let ck = scratch.cols_a[k];
            let corr = column_replacement_correction(la, scratch.deta_mix.as_slice(), scratch.adjt_deta.as_slice(), k, |r| v_t[(ck, scratch.rows_a[r])]);
            contrib -= (det_deta + corr) * det_detb;
        }

        for k in 0..lb {
            let mbk = bit(bits_b, k + 1);
            let v_t = w.ab.vba_t(mb0, ma0, mbk);
            let ck = scratch.cols_b[k];
            let corr = column_replacement_correction(lb, scratch.detb_mix.as_slice(), scratch.adjt_detb.as_slice(), k, |r| v_t[(ck, scratch.rows_b[r])]);
            contrib -= (det_detb + corr) * det_deta;
        }

        for (i, &ra) in scratch.rows_a.iter().enumerate() {
            for (j, &ca) in scratch.cols_a.iter().enumerate() {
                let cofa = scratch.adjt_deta.as_slice()[idx(la, i, j)];
                let ma1 = bit(bits_a, j + 1);

                for k in 0..lb {
                    let mbk = bit(bits_b, k + 1);
                    let iib = w.ab.iiab(ma0, ma1, mb0, mbk);
                    slice4ijrc(scratch.iisliceb.as_mut_slice(), lb, &iib, scratch.rows_b.as_slice(), scratch.cols_b.as_slice(), ra, ca);
                    let corr = column_replacement_correction(lb, scratch.detb_mix.as_slice(), scratch.adjt_detb.as_slice(), k, |r| scratch.iisliceb.as_slice()[idx(lb, r, k)]);
                    contrib += 0.5 * cofa * (det_detb + corr);
                }
            }
        }

        for (i, &rb) in scratch.rows_b.iter().enumerate() {
            for (j, &cb) in scratch.cols_b.iter().enumerate() {
                let cofb = scratch.adjt_detb.as_slice()[idx(lb, i, j)];
                let mb1 = bit(bits_b, j + 1);

                for k in 0..la {
                    let mak = bit(bits_a, k + 1);
                    let iia = w.ab.iiab(ma0, mak, mb0, mb1);
                    slice4rcij(scratch.iislicea.as_mut_slice(), la, &iia, scratch.rows_a.as_slice(), scratch.cols_a.as_slice(), rb, cb);
                    let corr = column_replacement_correction(la, scratch.deta_mix.as_slice(), scratch.adjt_deta.as_slice(), k, |r| scratch.iislicea.as_slice()[idx(la, r, k)]);
                    contrib += 0.5 * cofb * (det_deta + corr);
                }
            }
        }
        acc += contrib;
    });
    (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * acc
}

