// nonorthogonalwicks.rs 
use std::ptr::NonNull;

use ndarray::{Array1, Array2, ArrayView2, Array4, ArrayView4, Axis, s};
use ndarray_linalg::{SVD, Determinant};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::{ExcitationSpin};

use crate::maths::{einsum_ba_ab_real, eri_ao2mo, loewdin_x_real, minor, adjugate_transpose, det, mix_columns, build_d};
use crate::mpiutils::Sharedffi;
use crate::noci::occ_coeffs;

// Storage in which we split the Wicks data into the shared remote memory access (RMA) and a view 
// for reading said data.
pub struct WicksShared {
    pub rma: WicksRma,   
    pub view: WicksView, 
}

impl WicksShared {
    /// Get a shared reference to the WicksView object.
    /// # Arguments:
    /// `self`: `WicksShared`, view and RMA for Wick's intermediates.
    /// # Returns
    /// `&WicksView`, shared view of the Wick's intermediates.
    pub fn view(&self) -> &WicksView {&self.view}
    
    /// Get a mutable reference to the WicksView object.
    /// # Arguments:
    /// `self`: `WicksShared`, view and RMA for Wick's intermediates.
    /// # Returns
    /// `&mut WicksView`, mutable view of the Wick's intermediates.
    pub fn view_mut(&mut self) -> &mut WicksView {
        &mut self.view
    }
    
    /// Get a mutable slice over the full contiguous shared tensor storage.
    /// The returned slice may be used to overwrite stored matrices or tensors in place.
    /// # Arguments:
    /// `self`: `WicksShared`, view and RMA for Wick's intermediates.
    /// # Returns
    /// `&mut [f64]`, mutable slice over the full shared tensor slab.
    pub fn slab_mut(&mut self) -> &mut [f64] {
        let ptr = self.rma.base_ptr as *mut f64;
        let len = self.view.slab_len;
        unsafe {std::slice::from_raw_parts_mut(ptr, len)}
    }
}

// Storage for the RMA data of the Wick's objects.
pub struct WicksRma {
    pub shared: Sharedffi, 
    pub base_ptr: *mut u8, 
    pub nbytes: usize,     
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
    /// `self`: `WicksView`, view into Wick's intermediates.
    /// `lp`: `usize`, pair index 1.
    /// `gp`: `usize`, pair index 2.
    /// # Returns
    /// `usize`, flattened pair index.
    pub fn idx(&self, lp: usize, gp: usize) -> usize {
        lp * self.nref + gp
    }
    
    /// Get a pointer to the start of the shared tensor storage.
    /// # Arguments:
    /// `self`: `WicksView`, view into Wick's intermediates.
    /// # Returns
    /// `*const f64`, pointer to the start of the shared tensor slab.
    fn slab_ptr(&self) -> *const f64 {
        self.slab.as_ptr()
    }
    
    /// Read tensor slab beginning at a given offset and interpret the following n * n elements as
    /// a n by n matrix. Lifetime elision '_ ensures that the ArrayView2 may not outlive the borrow
    /// of self (WicksView) which in turn is only valid while the remote memory storage is valid.
    /// # Arguments:
    /// `self`: `WicksView`, view into Wick's intermediates.
    /// `off_f64`: `usize`, offset from the beginning of the tensor slab in units of f64.
    /// `n`: `usize`, size of matrix to be read.
    /// # Returns
    /// `ArrayView2<'_, f64>`, matrix view into the tensor slab.
    pub fn view2(&self, off_f64: usize, n: usize) -> ArrayView2<'_, f64> {
        unsafe {ArrayView2::from_shape_ptr((n, n), self.slab_ptr().add(off_f64))}
    }

    /// Read tensor slab beginning at a given offset and interpret the following n * n * n * n elements as
    /// a n by n by n by n tensor. Lifetime elision '_ ensures that the ArrayView4 may not outlive the borrow
    /// of self (WicksView) which in turn is only valid while the remote memory storage is valid.
    /// # Arguments:
    /// `self`: `WicksView`, view into Wick's intermediates.
    /// `off_f64`: `usize`, offset from the beginning of the tensor slab in units of f64.
    /// `n`: `usize`, size of tensor to be read.
    /// # Returns
    /// `ArrayView4<'_, f64>`, 4D tensor view into the tensor slab.
    pub fn view4(&self, off_f64: usize, n: usize) -> ArrayView4<'_, f64> {
        unsafe {ArrayView4::from_shape_ptr((n, n, n, n), self.slab_ptr().add(off_f64))}
    }
    
    /// Return a view for precomputed intermediates for a given lp, gp. Lifetime elision '_ ensures 
    /// that the ArrayView4 may not outlive the borrow of self (WicksView) which in turn is only valid 
    /// while the remote memory storage is valid.
    /// # Arguments:
    /// `self`: `WicksView`, view into Wick's intermediates.
    /// `lp`: `usize`, pair index 1.
    /// `gp`: `usize`, pair index 2.
    /// # Returns
    /// `WicksPairView<'_>`, grouped view of the same-spin and different-spin intermediates for the pair.
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
    /// `self`: `SameSpinView`, view to same-spin Wick's intermediates.
    /// # Returns
    /// `usize`, tensor dimension `2 * nmo`.
    fn n(&self) -> usize {2 * self.nmo}
    
    /// Get a view to the `X[mi]` matrix.
    /// # Arguments:
    /// `self`: `SameSpinView`, view to same-spin Wick's intermediates.
    /// `mi`: `usize`, zero distribution selector. 
    /// # Returns
    /// `ArrayView2<'_, f64>`, view of the `X[mi]` matrix.
    pub fn x(&self, mi: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.x[mi], self.n())
    }
    
    /// Get a view to the `Y[mi]` matrix.
    /// # Arguments:
    /// `self`: `SameSpinView`, view to same-spin Wick's intermediates.
    /// `mi`: `usize`, zero distribution selector. 
    /// # Returns
    /// `ArrayView2<'_, f64>`, view of the `Y[mi]` matrix.
    pub fn y(&self, mi: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.y[mi], self.n())
    }
    
    /// Get a view to the Hamiltonian `F[mi][mj]` tensor.
    /// # Arguments:
    /// `self`: `SameSpinView`, view to same-spin Wick's intermediates.
    /// `mi, mj`: `usize`, zero distribution selectors. 
    /// # Returns
    /// `ArrayView2<'_, f64>`, view of the Hamiltonian `F[mi][mj]` matrix.
    pub fn fh(&self, mi: usize, mj: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.fh[mi][mj], self.n())
    }

    /// Get a view to the Fock `F[mi][mj]` tensor.
    /// # Arguments:
    /// `self`: `SameSpinView`, view to same-spin Wick's intermediates.
    /// `mi, mj`: `usize`, zero distribution selectors. 
    /// # Returns
    /// `ArrayView2<'_, f64>`, view of the Fock `F[mi][mj]` matrix.
    pub fn ff(&self, mi: usize, mj: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.ff[mi][mj], self.n())
    }
    
    /// Get a view to the `V[mi][mj][mk]` tensor.
    /// # Arguments:
    /// `self`: `SameSpinView`, view to same-spin Wick's intermediates.
    /// `mi, mj, mk`: `usize`, zero distribution selector. 
    /// # Returns
    /// `ArrayView2<'_, f64>`, view of the `V[mi][mj][mk]` matrix.
    pub fn v(&self, mi: usize, mj: usize, mk: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.v[mi][mj][mk], self.n())
    }
    
    /// Get a view to the `J[mi][mj][mk][ml]` tensor.
    /// # Arguments:
    /// `self`: `SameSpinView`, view to same-spin Wick's intermediates.
    /// `slot`: `usize`, compressed storage slot for the requested J tensor.
    /// # Returns
    /// `ArrayView4<'_, f64>`, view of the requested J tensor.
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
    /// `self`: `DiffSpinView`, view to diff-spin Wick's intermediates.
    /// # Returns
    /// `usize`, tensor dimension `2 * nmo`.
    fn n(&self) -> usize {2 * self.nmo}
    
    /// Get a view to the `Vab[ma0][mb0][mak]` tensor.
    /// # Arguments:
    /// `self`: `DiffSpinView`, view to diff-spin Wick's intermediates.
    /// `ma0, mb0, mak`: `usize`, zero distribution selector. 
    /// # Returns
    /// `ArrayView2<'a, f64>`, view of the `Vab[ma0][mb0][mak]` matrix.
    pub fn vab(&self, ma0: usize, mb0: usize, mak: usize) -> ArrayView2<'a, f64> {
        self.w.view2(self.off.vab[ma0][mb0][mak], self.n())
    }

    /// Get a view to the `Vba[mb0][ma0][mak]` tensor.
    /// # Arguments:
    /// `self`: `DiffSpinView`, view to diff-spin Wick's intermediates.
    /// `mb0, ma0, mbk`: `usize`, zero distribution selector.
    /// # Returns
    /// `ArrayView2<'a, f64>`, view of the `Vba[mb0][ma0][mbk]` matrix.
    pub fn vba(&self, mb0: usize, ma0: usize, mbk: usize) -> ArrayView2<'a, f64> {
        self.w.view2(self.off.vba[mb0][ma0][mbk], self.n())
    }

    /// Get a view to the `IIab[ma0][maj][mb0][mbj]` tensor.
    /// # Arguments:
    /// `self`: `DiffSpinView`, view to diff-spin Wick's intermediates.
    /// `ma0, maj, mb0, mbj`: `usize`, zero distribution selector.
    /// # Returns
    /// `ArrayView4<'a, f64>`, view of the `IIab[ma0][maj][mb0][mbj]` tensor.
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

pub type Label = (Side, Type, usize);

/// Calculate and store all the required offsets from the beginning of the large contiguous shared
/// tensor storage to a given matrix or tensor.
/// # Arguments:
/// `nref`: `usize`, number of references.
/// `nmo`: `usize`, number of molecular orbitals.
/// # Returns
/// `(Vec<PairOffset>, usize)`, per-pair offset table and total slab length in units of `f64`.
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
/// `slab`: `[f64]`, contiguous tensor storage.
/// `o`: `SameSpinOffset`, offsets into the storage.
/// `w`: `SameSpinBuild`, owned Wick's intermediates.
/// # Returns
/// `()`, writes the same-spin intermediates into the slab.
pub fn write_same_spin(slab: &mut [f64], o: &SameSpinOffset, w: &SameSpinBuild) {
    write2(slab, o.x[0], &w.x[0]);
    write2(slab, o.x[1], &w.x[1]);
    write2(slab, o.y[0], &w.y[0]);
    write2(slab, o.y[1], &w.y[1]);
    for mi in 0..2 {for mj in 0..2 {write2(slab, o.fh[mi][mj], &w.fh[mi][mj]);}}
    for mi in 0..2 {for mj in 0..2 {write2(slab, o.ff[mi][mj], &w.ff[mi][mj]);}}
    for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {write2(slab, o.v[mi][mj][mk], &w.v[mi][mj][mk]);}}}
    for s in 0..10 {write4(slab, o.j[s], &w.j[s]);}
}

/// Fill the diff-spin data owning structs with the diff-spin Wick's intermediates using the
/// prescribed offsets.
/// # Arguments:
/// `slab`: `[f64]`, contiguous tensor storage.
/// `o`: `DiffSpinOffset`, offsets into the storage.
/// `w`: `DiffSpinBuild`, owned Wick's intermediates.
/// # Returns
/// `()`, writes the diff-spin intermediates into the slab.
pub fn write_diff_spin(slab: &mut [f64], o: &DiffSpinOffset, w: &DiffSpinBuild) {
    for ma0 in 0..2 {for mb0 in 0..2 {for mk in 0..2 {write2(slab, o.vab[ma0][mb0][mk], &w.vab[ma0][mb0][mk]);}}}
    for mb0 in 0..2 {for ma0 in 0..2 {for mk in 0..2 {write2(slab, o.vba[mb0][ma0][mk], &w.vba[mb0][ma0][mk]);}}}
    for ma0 in 0..2 {for maj in 0..2 {for mb0 in 0..2 {for mbj in 0..2 {write4(slab, o.iiab[ma0][maj][mb0][mbj], &w.iiab[ma0][maj][mb0][mbj]);}}}}
}

/// Copy matrix into tensor slab provided it is contiguous.
/// # Arguments:
/// `slab`: `[f64]`, contiguous tensor storage.
/// `off`: `usize`, offset for the start position.
/// `a`: `Array2`, matrix to copy.
/// # Returns
/// `()`, writes the matrix into the tensor slab.
pub fn write2(slab: &mut [f64], off: usize, a: &ndarray::Array2<f64>) {
    let src = a.as_slice().expect("Array2 must be contiguous");
    slab[off..off + src.len()].copy_from_slice(src);
}

/// Copy tensor into tensor slab provided it is contiguous.
/// # Arguments:
/// `slab`: `[f64]`, contiguous tensor storage.
/// `off`: `usize`, offset for the start position.
/// `a`: `Array4`, tensor to copy.
/// # Returns
/// `()`, writes the tensor into the tensor slab.
fn write4(slab: &mut [f64], off: usize, a: &ndarray::Array4<f64>) {
    let src = a.as_slice().expect("Array4 must be contiguous");
    slab[off..off + src.len()].copy_from_slice(src);
}

/// Write 2D slice of 4D J or IIab tensors into provided output scratch. The given slice is t[r, c, i, j]  
/// where r, c are rows, columns and i, j are fixed indices.
/// # Arguments:
/// `out`: `Array2`, preallocated output scratch.
/// `t`: `ArrayView4`, view of a 4D tensor.
/// `rows`: `[usize]`, length excitation rank map from row labels to tensor index.
/// `cols`: `[usize]`, length excitation rank map from col labels to tensor index.
/// `i_fixed`: `usize`, fixed tensor indices for the j dimension.
/// `j_fixed`: `usize`, fixed tensor indices for the j dimension.
/// # Returns
/// `()`, writes the requested 2D slice into `out`.
fn slice4(out: &mut Array2<f64>, t: &ArrayView4<f64>, rows: &[usize], cols: &[usize], i_fixed: usize, j_fixed: usize) {
    let l = rows.len();
    // `out` is contiguous so we write into flat buffer with row-major index as out[(r, c)] == buf[r * ncols + c]
    let ncols = out.ncols();
    let buf = out.as_slice_mut().unwrap();
    // 4D tensor `t` is stored as base + a * strides[0] + b * strides[1] + c * strides[2] + d * strides[3] for 
    // element at t[a, b, c, d], where base is the start of this memory in the slab.
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
            // Base offset for current output row is t[rr, 0, i, j] or 
            // base + rr * strides[0] + 0 + i * strides[2] + j * strides[3] which is
            // equivalent to rr * strides[0] + fixed.
            let off = rr * strides[0] + fixed;
            
            // Iterate over columns of output.
            for c in 0..l {
                // Output col cc given by tensor col cc = cols[c].
                let cc = *cols.get_unchecked(c) as isize;
                // Element t[rr, cc, i, j] is base + rr * strides[0] + cc * strides[1] + i * strides[2] + j * strides[3] 
                // or off + cc * strides[1]. Write into output buffer as row major.
                buf[(r * ncols) + c] = *base.offset(off + cc * strides[1]);
            }
        }
    }
}

/// Write 2D slice of 4D J and IIab tensor into provided output scratch. The given slice is t[r, c, i, j]  
/// where r, c are rows, columns and i, j are fixed indices.
/// # Arguments:
/// `out`: `Array2`, preallocated output scratch.
/// `t`: `ArrayView4`, view of a 4D tensor.
/// `rows`: `[usize]`, length excitation rank map from row labels to tensor index.
/// `cols`: `[usize]`, length excitation rank map from col labels to tensor index.
/// `i_fixed`: `usize`, fixed tensor indices for the j dimension.
/// `j_fixed`: `usize`, fixed tensor indices for the j dimension.
/// # Returns
/// `()`, writes the requested swapped 2D slice into `out`.
fn slice4swap(out: &mut Array2<f64>, t: &ArrayView4<f64>, rows: &[usize], cols: &[usize], i_fixed: usize, j_fixed: usize) {
    let l = rows.len();
    // `out` is contiguous so we write into flat buffer with row-major index as out[(r, c)] == buf[r * ncols + c]
    let ncols = out.ncols();
    let buf = out.as_slice_mut().unwrap();
    // 4D tensor `t` is stored as base + a * strides[0] + b * strides[1] + c * strides[2] + d * strides[3] for 
    // element at t[a, b, c, d], where base is the start of this memory in the slab.
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
            // Base offset for current output row is t[i, j, rr, 0] or 
            // base + i * strides[0] + j * strides[1] + rr * strides[2] + 0, which is
            // equivalent to fixed + rr * strides[2].
            let off = fixed + rr * strides[2];
            
            // Iterate over columns of output.
            for c in 0..l {
                // Output col cc given by tensor col cc = cols[c].
                let cc = *cols.get_unchecked(c) as isize;
                // Element t[i, j, rr, cc] is base + i * strides[0] + j * strides[1] + rr * strides[2] + cc * strides[3]
                // or off + cc * strides[3]. Write into output buffer as row major.
                buf[(r * ncols) + c] = *base.offset(off + cc * strides[3]);
            }
        }
    }
}

/// Map selectors (mi, mj, mk, ml) to the compressed storage slot for J and
/// whether the requested tensor should be read with swapped indices.
/// # Arguments:
/// `mi, mj, mk, ml`: `usize`, zero distribution selectors.
/// # Returns
/// `(usize, bool)`, compressed slot index and whether swapped access is required.
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

// Storage for preallocated Wick's data terms such that we do not have to reallocate all of these
// everytime a matrix element evaluation routine is called.
#[derive(Default)]
pub struct WickScratch {
    pub rows_label: Vec<Label>,
    pub cols_label: Vec<Label>,

    pub rows_label_a: Vec<Label>,
    pub cols_label_a: Vec<Label>,

    pub rows_label_b: Vec<Label>,
    pub cols_label_b: Vec<Label>,

    pub rows: Vec<usize>,
    pub cols: Vec<usize>,

    pub det0: Array2<f64>,
    pub det1: Array2<f64>,
    pub det_mix: Array2<f64>,

    pub fcol: Array1<f64>,
    pub dv: Array1<f64>,

    pub v1: Array1<f64>,
    pub dv1: Array1<f64>,
    pub dv1m: Array1<f64>,          
    pub jslice_full: Array2<f64>,
    pub jslice2: Array2<f64>,       
    pub det_mix2: Array2<f64>,  

    pub rows_a: Vec<usize>,
    pub cols_a: Vec<usize>,
    pub rows_b: Vec<usize>,
    pub cols_b: Vec<usize>,

    pub deta0: Array2<f64>,
    pub deta1: Array2<f64>,
    pub deta_mix: Array2<f64>,
    pub detb0: Array2<f64>,
    pub detb1: Array2<f64>,
    pub detb_mix: Array2<f64>,

    pub v1a: Array1<f64>,
    pub v1b: Array1<f64>,
    pub dv1a: Array1<f64>,
    pub dv1b: Array1<f64>,

    pub iislicea: Array2<f64>,  
    pub iisliceb: Array2<f64>,      
    pub deta_mix_minor: Array2<f64>,
    pub detb_mix_minor: Array2<f64>,
    
    pub adjt_det: Array2<f64>,
    pub adjt_deta: Array2<f64>,
    pub adjt_detb: Array2<f64>,
    pub adjt_det2: Array2<f64>,
    pub adjt_deta_mix_minor: Array2<f64>,
    pub adjt_detb_mix_minor: Array2<f64>,

    pub invs: Array1<f64>,
    pub invsla: Array1<f64>,
    pub invslb: Array1<f64>,
    pub invslm1:  Array1<f64>,
    pub invslam1: Array1<f64>,
    pub invslbm1: Array1<f64>,

    pub lu: Array2<f64>,
    pub lua: Array2<f64>,
    pub lub: Array2<f64>,
}

impl WickScratch {
    /// Construct empty scratch storage for Wick's quantities.
    /// # Arguments:
    /// # Returns
    /// `WickScratch`, default-initialised scratch storage.
    pub fn new() -> Self {Self::default()}
    
    /// If the previously allocated size of the scratch space is the not the same in  
    /// the same spin case resize all the scratch space quantities to be correct.
    /// # Arguments:
    /// `self`: `WickScratch`, scratch space for Wick's quantities.
    /// `l`: `usize`, excitation rank.
    /// # Returns
    /// `()`, resizes same-spin scratch storage in place.
    pub fn resizel(&mut self, l: usize) {
        if self.det0.nrows() != l {
            self.det0 = Array2::zeros((l, l));
            self.det1 = Array2::zeros((l, l));
            self.det_mix = Array2::zeros((l, l));
            self.adjt_det = Array2::zeros((l, l));
            self.jslice_full = Array2::zeros((l, l));
            self.fcol = Array1::zeros(l);
            self.dv = Array1::zeros(l);
            self.v1 = Array1::zeros(l);
            self.dv1 = Array1::zeros(l);
            self.invs = Array1::zeros(l);
            // This just needs to be of size LMAX + 2.
            self.lu = Array2::zeros((6, 6));

            let lm1 = l.saturating_sub(1);
            self.dv1m = Array1::zeros(lm1);
            self.invslm1 = Array1::zeros(lm1);
            self.jslice2 = Array2::zeros((lm1, lm1));
            self.det_mix2 = Array2::zeros((lm1, lm1));
            self.adjt_det2 = Array2::zeros((lm1, lm1));
        }
    }

    /// If the previously allocated size of the scratch space is the not the same in  
    /// the different spin case resize all the scratch space quantities to be correct.
    /// # Arguments:
    /// `self`: `WickScratch`, scratch space for Wick's quantities.
    /// `la`: `usize`, excitation rank spin alpha.
    /// `lb`: `usize`, excitation rank spin beta.
    /// # Returns
    /// `()`, resizes different-spin scratch storage in place.
    pub fn resizelalb(&mut self, la: usize, lb: usize) {
        if self.deta0.nrows() != la {
            self.deta0 = Array2::zeros((la, la));
            self.deta1 = Array2::zeros((la, la));
            self.deta_mix = Array2::zeros((la, la));
            self.adjt_deta = Array2::zeros((la, la));
            self.v1a = Array1::zeros(la);
            self.dv1a = Array1::zeros(la);
            self.iislicea = Array2::zeros((la, la));
            self.invsla = Array1::zeros(la);
             // This just needs to be of size LMAX + 2.
            self.lua = Array2::zeros((6, 6));

            let lam1 = la.saturating_sub(1);
            self.deta_mix_minor = Array2::zeros((lam1, lam1));
            self.adjt_deta_mix_minor = Array2::zeros((lam1, lam1));
            self.invslam1 = Array1::zeros(lam1);
        }
        if self.detb0.nrows() != lb {
            self.detb0 = Array2::zeros((lb, lb));
            self.detb1 = Array2::zeros((lb, lb));
            self.detb_mix = Array2::zeros((lb, lb));
            self.adjt_detb = Array2::zeros((lb, lb));
            self.v1b = Array1::zeros(lb);
            self.dv1b = Array1::zeros(lb);
            self.iisliceb = Array2::zeros((lb, lb));
            self.invslb = Array1::zeros(lb);
            // This just needs to be of size LMAX + 2.
            self.lub = Array2::zeros((6, 6));

            let lbm1 = lb.saturating_sub(1);
            self.detb_mix_minor = Array2::zeros((lbm1, lbm1));
            self.adjt_detb_mix_minor = Array2::zeros((lbm1, lbm1));
            self.invslbm1 = Array1::zeros(lbm1);
        }
    }
}

impl SameSpinBuild {
    /// Constructor for the WicksReferencePair object of SameSpin which contains the precomputed values
    /// per pair of reference determinants required to evaluate arbitrary excited determinants 
    /// in O(1) time when the excitations are of the same spin.
    /// # Arguments:
    /// `eri`: `Array4`, electron repulsion integrals. 
    /// `h_munu`: `Array2`, AO core Hamiltonian.
    /// `s_munu`: `Array2`, AO overlap matrix.
    /// `g_c`: `Array2`, full AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// `l_c`: `Array2`, full AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// `go`: `Array1`, occupancy vector for |^\Gamma\Psi\rangle.
    /// `lo`: `Array1`, occupancy vector for |^\Lambda\Psi\rangle. 
    /// `tol`: `f64`, tolerance for whether a number is considered zero.
    /// # Returns
    /// `SameSpinBuild`, precomputed same-spin Wick's intermediates for the reference pair.
    pub fn new(eri: &Array4<f64>, h_munu: &Array2<f64>, s_munu: &Array2<f64>, g_c: &Array2<f64>, l_c: &Array2<f64>, go: &Array1<f64>, lo: &Array1<f64>, tol: f64) -> Self {
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
    /// `g_c_occ`: `Array2`, occupied coefficients ({}^\Gamma C^*)_i^\mu.
    /// `l_c_occ`: `Array2`, occupied coefficients ({}^\Lambda C)_j^\nu 
    /// `s_munu`: `Array2`, AO overlap matrix S_{\mu\nu}.
    /// # Returns
    /// `Array2<f64>`, occupied-orbital overlap matrix.
    pub fn calculate_mo_overlap_matrix(l_c_occ: &Array2<f64>, g_c_occ: &Array2<f64>, s_munu: &Array2<f64>) -> Array2<f64> {
        l_c_occ.t().dot(&s_munu.dot(g_c_occ))
    }
    
    /// Perform singular value decomposition on the occupied orbital overlap matrix {}^{\Gamma\Lambda} S_{ij} as:
    ///     {}^{\Gamma\Lambda} \mathbf{S} = \mathbf{U} {}^{\Gamma\Lambda} \mathbf{\tilde{S}} \mathbf{V}^\dagger,
    /// and rotate the occupied coefficients:
    ///     |{}^\Gamma \Psi_i\rangle = \sum_{\mu} {}^\Gamma c_i^\mu U_{ij} |\phi_\mu \rangle.
    ///     |{}^\Lambda \Psi_j\rangle = \sum_{\nu} {}^\Lambda c_j^\nu V_{ij} |\phi_\nu \rangle.
    /// # Arguments:
    /// `g_c_occ`: `Array2`, occupied coefficients ({}^\Gamma C^*)_i^\mu.
    /// `l_c_occ`: `Array2`, occupied coefficients ({}^\Lambda C)_j^\nu.
    /// `tol`: `f64`, tolerance for the orthonormalisation step.
    /// # Returns
    /// `(Array1<f64>, Array2<f64>, Array2<f64>, f64)`, singular values, rotated occupied coefficients
    /// for \Gamma and \Lambda, and the phase associated with the rotation.
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
    /// `gl_tilde_s`: `Array1`, vector of diagonal single values of {}^{\Gamma\Lambda} \tilde{S}.
    /// `g_c_tilde_occ`: `Array2`, rotated occupied coefficients ({}^\Gamma \tilde{C}^*)_i^\mu.
    /// `l_c_tilde_occ`: `Array2`, rotated occupied coefficients ({}^\Lambda \tilde{C})_j^\nu.
    /// `zeros`: `Vec<usize>`, indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S}. 
    /// `tol`: `f64`, tolerance for whether a singular value is considered zero.
    /// # Returns
    /// `(Array2<f64>, Array2<f64>)`, the M^{0} and M^{1} matrices.
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
    /// `s_munu`: `Array2`, AO overlap matrix.
    /// `g_c`: `Array2`, full AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// `l_c`: `Array2`, full AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// `gl_m`: `Array2`, M matrix{}^{\Gamma\Lambda} M^{\sigma\tau, 0} or  {}^{\Gamma\Lambda} M^{\sigma\tau, 1}. 
    /// `subtract`: `bool`, whether to use m_k = 0 or m_k = 1. 
    /// # Returns
    /// `(Array2<f64>, Array2<f64>)`, the X and Y matrices.
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
    /// `g_c`: `Array2`, full AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// `l_c`: `Array2`, full AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// `h_munu`: `Array2`, one-electron core AO hamiltonian.
    /// `x`: `Array2`, {}^{\Gamma\Lambda} X_{ij}^{m_k}.
    /// `y`: `Array2`, {}^{\Gamma\Lambda} Y_{ij}^{m_k}.
    /// # Returns
    /// `(f64, Array2<f64>)`, scalar F_0^{m_k} and matrix F_{ab}^{m_i,m_j}.
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
    /// `eri`: `Array4`, AO basis ERIs (not-antisymmetrised).
    /// `m`: `Array2`, {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
    /// # Returns
    /// `Array2<f64>`, Coulomb contraction matrix.
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
    /// `eri`: `Array4`, AO basis ERIs (not-antisymmetrised).
    /// `m`: `Array2`, {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
    /// # Returns
    /// `Array2<f64>`, exchange contraction matrix.
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
    /// `eri`: `Array4`, electron repulsion integrals. 
    /// `h_munu`: `Array2`, AO core Hamiltonian.
    /// `s_munu`: `Array2`, AO overlap matrix.
    /// `g_ca`: `Array2`, spin alpha AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// `g_cb`: `Array2`, spin beta AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// `l_ca`: `Array2`, spin alpha AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// `l_cb`: `Array2`, spin beta AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// `goa`: `Array1`, alpha occupancy vector for |^\Gamma\Psi\rangle.
    /// `gob`: `Array1`, beta occupancy vector for |^\Gamma\Psi\rangle.
    /// `loa`: `Array1`, alpha occupancy vector for |^\Lambda\Psi\rangle.
    /// `lob`: `Array1`, beta occupancy vector for |^\Lambda\Psi\rangle.
    /// `tol`: `f64`, tolerance for whether a number is considered zero.
    /// # Returns
    /// `DiffSpinBuild`, precomputed different-spin Wick's intermediates for the reference pair.
    pub fn new(eri: &Array4<f64>, s_munu: &Array2<f64>, g_ca: &Array2<f64>, g_cb: &Array2<f64>, l_ca: &Array2<f64>, l_cb: &Array2<f64>, 
           goa: &Array1<f64>, gob: &Array1<f64>, loa: &Array1<f64>, lob: &Array1<f64>, tol: f64) -> Self {
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
    /// `m`: `Array2`, {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
    /// `s`: `Array2`, AO overlap matrix S_{\mu\nu}.
    /// `cx`: `Array2`, AO coefficient matrix for \Lambda. Should be renamed.
    /// `cw`: `Array2`, AO coefficient matrix for \Gamma. Should be renamed.
    /// `i`: `usize`, selector for m being 0 or 1.
    /// # Returns
    /// `(Array2<f64>, Array2<f64>)`, left and right factorisation matrices.
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

/// Given the orbitals excited from and to  relative to the reference determinants, construct the 
/// index labels for the L by L contraction matrix. The contraction matrix is blocked by quadrant
/// as: [\Lambda\Lambda & \Lambda\Gamma \\ \Gamma\Lambda & \Gamma\Gamma]. The rows correspond to
/// operations that act as creation operators after normal ordering to the bra (\Lambda), whilst
/// columns correspond to operations that are annhilations after normal ordering to the bra.
/// # Arguments:
/// `l_ex`: `Excitation`, left excitations relative to reference \Lambda in the bra.
/// `g_ex`: `Excitation`, right excitations relative to reference \Gamma in the ket.
/// `rows`: `Vec<Label>`, output row labels.
/// `cols`: `Vec<Label>`, output column labels.
/// # Returns
/// `()`, writes the contraction determinant labels into `rows` and `cols`.
fn construct_determinant_lables(l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, rows: &mut Vec<Label>, cols: &mut Vec<Label>) {
    // Integer L is the total number of combined excitations from the bra (\Lambda) and ket (\Gamma).
    let nl = l_ex.holes.len();
    let ng = g_ex.holes.len();

    rows.clear();
    cols.clear();

    if nl == 0 && ng == 0 {return;}
    
    // If number of holes in \Lambda determinant is non-zero and number of holes in \Gamma
    // determinant is zero we only need to consider \Lambda determinant. The ordering is:
    //  rows = [\Lambda parts], cols = [\Lambda holes].
    if nl > 0 && ng == 0 {
        for &a in &l_ex.parts {rows.push((Side::Lambda, Type::Part, a));}
        for &i in &l_ex.holes {cols.push((Side::Lambda, Type::Hole, i));}
        return;
    }
    
    // If number of holes in \Gamma determinant is non-zero and number of holes in \Lambda
    // determinant is zero we only need to consider \Gamma determinant. The ordering is:
    //  rows = [\Gamma holes], cols = [\Gamma parts].
    if nl == 0 && ng > 0 {
        for &i in &g_ex.holes {rows.push((Side::Gamma, Type::Hole, i));}
        for &a in &g_ex.parts {cols.push((Side::Gamma, Type::Part, a));}
        return;
    }

    // If both are non-zero we have the ordering:
    //  rows = [\Lambda parts ; \Gamma holes], cols = [\Lambda holes ; \Gamma parts]
    for &a in &l_ex.parts {rows.push((Side::Lambda, Type::Part, a));}
    for &i in &g_ex.holes {rows.push((Side::Gamma, Type::Hole, i));}
    for &i in &l_ex.holes {cols.push((Side::Lambda, Type::Hole, i));}
    for &a in &g_ex.parts {cols.push((Side::Gamma, Type::Part, a));}
}

/// Given the total excitation rank L and the number of zero-overlap orbital couplings find all the
/// possible combinations of (m_1,..., m_L) \in {0, 1}^L which satisfy
///     m_1 + .... + m_L = m.
/// The return is some type with Iterator implemented which is not specified due to long name.
/// # Arguments:
/// `l`: `usize`, total excitation rank.
/// `m`: `usize`, number of zero-overlap orbital couplings.
/// # Returns
/// `impl Iterator<Item = u64>`, iterator over all bitstrings satisfying the required number of ones.
fn iter_m_combinations(l: usize, m: usize) -> impl Iterator<Item = u64> {
    // If the total excitation rank is greater than 64 the below calculation will not work.
    assert!(l < 64);
    // Left shift 000....0001 (64) by L positions to get 2^L combinations of (m_1,..., m_L) \in {0, 1}^L.
    let max = 1u64 << l;
    // Iterate over all 2^L bitstrings and keep only those with m bits set to 1 as required by 
    //  m_1 + ..... + m_L = m.
    let mut bitstrings = Vec::new();
    for bitstring in 0..max {
        let ones: usize = bitstring.count_ones() as usize;
        if ones == m {bitstrings.push(bitstring)}
    }
    bitstrings.into_iter()
}

/// Convert the generated contraction determinant labels into position indices.
/// # Arguments:
/// `side`: `Side`, whether the label is from the bra (\Lambda) or ket (\Gamma).
/// `p`: `usize`, orbital index in MO basis.
/// `nmo`: `usize`, number of MOs.
/// # Returns
/// `usize`, flattened orbital index in the concatenated orbital space.
fn label_to_idx(side: Side, p: usize, nmo: usize) -> usize {
    match side {
        Side::Lambda => p,        // First block.
        Side::Gamma  => nmo + p,  // Second block.
    }
}

/// All the same spin routines require a number of the same quantities. It is more efficient to
/// precompute them once here rather than inside each matrix element routine.
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// `l_ex`: `ExcitationSpin`, spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// `g_ex`: `ExcitationSpin`, spin resolved excitation array for |{}^\Gamma \Psi\rangle. 
/// `scratch`: `WickScratch`, scratch space for Wick's quantities.
/// # Returns
/// `()`, prepares the shared same-spin scratch quantities in place.
pub fn prepare_same(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) { 
    let l = l_ex.holes.len() + g_ex.holes.len();
    scratch.resizel(l);

    construct_determinant_lables(l_ex, g_ex, &mut scratch.rows_label, &mut scratch.cols_label);

    scratch.rows.clear();
    scratch.rows.extend(scratch.rows_label.iter().map(|(s, _t, i)| label_to_idx(*s, *i, w.nmo)));
    scratch.cols.clear();
    scratch.cols.extend(scratch.cols_label.iter().map(|(s, _t, i)| label_to_idx(*s, *i, w.nmo)));

    let x0 = w.x(0); 
    let y0 = w.y(0);
    let x1 = w.x(1); 
    let y1 = w.y(1);

    // Build two full contraction determinants each having exclusively (X0, Y0) or (X1, Y1). 
    build_d(&mut scratch.det0, &x0, &y0, &scratch.rows, &scratch.cols);
    build_d(&mut scratch.det1, &x1, &y1, &scratch.rows, &scratch.cols);
}

/// Calculate overlap matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. Utilises
/// a sum over possible ways to distribute zeros across the columns of the L by L determinant.
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// `l_ex`: `ExcitationSpin`, spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// `g_ex`: `ExcitationSpin`, spin resolved excitation array for |{}^\Gamma \Psi\rangle. 
/// `scratch`: `WickScratch`, scratch space for Wick's quantities.
/// # Returns
/// `f64`, overlap matrix element.
pub fn lg_overlap(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) -> f64 {

    // If the total excitation rank is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the overlap element is zero.
    let l = l_ex.holes.len() + g_ex.holes.len();
    if w.m > l {return 0.0;}

    if w.m == 0 {return w.phase * w.tilde_s_prod * det(&scratch.det0).unwrap_or(0.0);}
    if w.m == l {return w.phase * w.tilde_s_prod * det(&scratch.det1).unwrap_or(0.0);}

    let mut acc = 0.0;

    // Iterate over all possible distributions of zeros amongst the columns.
    for bitstring in iter_m_combinations(l, w.m) {
        mix_columns(&mut scratch.det_mix, &scratch.det0, &scratch.det1, bitstring);
        let Some(d) = det(&scratch.det_mix) else {continue;};
        acc += d;
    }
    w.phase * w.tilde_s_prod * acc
}

/// Calculate one electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// `l_ex`: `ExcitationSpin`, spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// `g_ex`: `ExcitationSpin`, spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// `scratch`: `WickScratch`, scratch space for Wick's quantities.
/// `tol`: `f64`, tolerance for singularity handling in determinant evaluation.
/// # Returns
/// `f64`, one-electron Hamiltonian matrix element.
#[inline(always)]
pub fn lg_h1(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    
    // If the total excitation rank L + 1 is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the one electron matrix element is zero.
    let l = l_ex.holes.len() + g_ex.holes.len();
    if w.m > (l + 1) {return 0.0;}

    let mut acc = 0.0;

    // Iterate over all possible distributions of zeros amongst the columns.
    for bitstring in iter_m_combinations(l + 1, w.m) {
        let mi = ((bitstring & 1) == 1) as usize;

        let mcol: u64 = bitstring >> 1;
        mix_columns(&mut scratch.det_mix, &scratch.det0, &scratch.det1, mcol);
        
        // First term in the sum of Eqn 26 involves the same sum over all possible bitstrings with
        // the F0 matrix multiplying the contraction determinant.
        let Some(det_det) = adjugate_transpose(&mut scratch.adjt_det, &mut scratch.invs, &mut scratch.lu, &scratch.det_mix, tol) else {continue;};
        let mut contrib = det_det * w.f0h[mi];

        for b in 0..l {
            let mj = ((bitstring >> (b + 1)) & 1) as usize;
            let f = w.fh(mi, mj);
            let cb = scratch.cols[b];

            let mut corr = 0.0;
            for a in 0..l {
                let ra = scratch.rows[a];
                let new = f[(ra, cb)];
                corr += (new - scratch.det_mix[(a, b)]) * scratch.adjt_det[(a, b)];
            }
            contrib -= det_det + corr;
        }
        acc += contrib;
    }
    w.phase * w.tilde_s_prod * acc
}

/// Calculate one electron Fock matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// `l_ex`: `ExcitationSpin`, spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// `g_ex`: `ExcitationSpin`, spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// `scratch`: `WickScratch`, scratch space for Wick's quantities.
/// `tol`: `f64`, tolerance for singularity handling in determinant evaluation.
/// # Returns
/// `f64`, one-electron Fock matrix element.
#[inline(always)]
pub fn lg_f(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    
    // If the total excitation rank L + 1 is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the one electron matrix element is zero.
    let l = l_ex.holes.len() + g_ex.holes.len();
    if w.m > (l + 1) {return 0.0;}

    let mut acc = 0.0;

    // Iterate over all possible distributions of zeros amongst the columns.
    for bitstring in iter_m_combinations(l + 1, w.m) {
        let mi = ((bitstring & 1) == 1) as usize;

        let mcol: u64 = bitstring >> 1;
        mix_columns(&mut scratch.det_mix, &scratch.det0, &scratch.det1, mcol);
        
        // First term in the sum of Eqn 26 involves the same sum over all possible bitstrings with
        // the F0 matrix multiplying the contraction determinant.
        let Some(det_det) = adjugate_transpose(&mut scratch.adjt_det, &mut scratch.invs, &mut scratch.lu, &scratch.det_mix, tol) else {continue;};
        let mut contrib = det_det * w.f0f[mi];

        for b in 0..l {
            let mj = ((bitstring >> (b + 1)) & 1) as usize;
            let f = w.ff(mi, mj);
            let cb = scratch.cols[b];

            let mut corr = 0.0;
            for a in 0..l {
                let ra = scratch.rows[a];
                let new = f[(ra, cb)];
                corr += (new - scratch.det_mix[(a, b)]) * scratch.adjt_det[(a, b)];
            }
            contrib -= det_det + corr;
        }
        acc += contrib;
    }
    w.phase * w.tilde_s_prod * acc
}

/// Calculate the same-spin two electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// `l_ex`: `ExcitationSpin`, spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// `g_ex`: `ExcitationSpin`, spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// `scratch`: `WickScratch`, scratch space for Wick's quantities.
/// `tol`: `f64`, tolerance for singularity handling in determinant evaluation.
/// # Returns
/// `f64`, same-spin two-electron Hamiltonian matrix element.
#[inline(always)]
pub fn lg_h2_same(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    
    // If the total excitation rank L + 2 is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the two electron matrix element is zero.
    let l = l_ex.holes.len() + g_ex.holes.len();
    if w.m > (l + 2) {return 0.0;}
    
    let mut acc = 0.0;

    // Iterate over all possible distributions of zeros amongst the columns.
    for bits in iter_m_combinations(l + 2, w.m) {
        let m1 = (bits & 1) as usize;
        let m2 = ((bits >> 1) & 1) as usize;
        let ind: u64 = bits >> 2;
        
        // Construct mixed X0, Y0, X1, Y1 determinant for the given bitstring of size L + 2.
        mix_columns(&mut scratch.det_mix, &scratch.det0, &scratch.det1, ind);
        let Some(det_det) = adjugate_transpose(&mut scratch.adjt_det, &mut scratch.invs, &mut scratch.lu, &scratch.det_mix, tol) else {continue;};

        let mut contrib = 0.0f64;

        // Equation 30 sum over all distributions of zeros with V0 matrix multiplying size L + 2 determinant.
        contrib += w.v0[m1 + m2] * det_det;

        // Equation 34 sum over all distributions of zeros and iterating over all columns replacing
        // the current column with the V matrices to form a new determinant.
        for k in 0..l {
            let mk = ((bits >> (k + 2)) & 1) as usize;
            // Choose correct column of V based upon zero distributions.
            let vcol = w.v(m1, m2, mk);

            let ck = scratch.cols[k];

            let mut corr = 0.0;
            for r in 0..l {
                let new = vcol[(scratch.rows[r], ck)];
                corr += (new - scratch.det_mix[(r, k)]) * scratch.adjt_det[(r, k)];
            }

            contrib -= 2.0 * (det_det + corr);
        }

        // Equation 38 double sum over excitation ranks (i, j) which vary the two electron 4-index tensor
        // J accordingly, and within each double sum is another sum over columns of determinant
        // where column k is replaced with J matrices to form a new determinant.
        for i in 0..l {
            for j in 0..l {
                let phase = if ((i + j) & 1) == 0 {1.0} else {-1.0};
                
                // Find the (L - 1) by (L - 1) determinant removing row and column i, j.
                minor(&mut scratch.det_mix2, &scratch.det_mix, i, j);
                let Some(det_det2) = adjugate_transpose(&mut scratch.adjt_det2, &mut scratch.invslm1, &mut scratch.lu, &scratch.det_mix2, tol) else {continue;};

                // Select indices that give the correct slice of J.
                let ri_fixed = scratch.rows[i];
                let cj_fixed = scratch.cols[j];

                let mj = ((bits >> (j + 2)) & 1) as usize;
                
                // Inner sum iterates over all columns of the L - 1 determinant and replaces column
                // k with the appropriate column of J.
                for k2 in 0..(l - 1) {
                    let k_full = if k2 < j {k2} else {k2 + 1};
                    let mk = ((bits >> (k_full + 2)) & 1) as usize;

                    // Extract slice of J corresponding to the current distribution of zeros and
                    // get the correct minor matrix so as to align with the L - 1 dimensions.
                    let (slot, swap) = jslot(m1, m2, mk, mj);
                    let j4 = w.j(slot);
                    //if maxabs4(j4) > 1e16 {
                    //    println!("HUGE maxabs4(j4): {}",  maxabs4(j4));
                    //    println!("(m1, m2, mk, mj): ({},{},{},{})", m1 as usize, m2 as usize, mk as usize, mj as usize);
                    //    println!();
                    //}
                    if !swap {slice4(&mut scratch.jslice_full, &j4, &scratch.rows, &scratch.cols, ri_fixed, cj_fixed);} 
                    else {slice4swap(&mut scratch.jslice_full, &j4, &scratch.rows, &scratch.cols, ri_fixed, cj_fixed);}
                    minor(&mut scratch.jslice2, &scratch.jslice_full, i, j);

                    let mut corr = 0.0;
                    for r in 0..(l - 1) {
                        let new = scratch.jslice2[(r, k2)];
                        corr += (new - scratch.det_mix2[(r, k2)]) * scratch.adjt_det2[(r, k2)];
                    }
                    contrib += phase * (det_det2 + corr);
                }
            }
        }
        acc += contrib;
    }
    w.phase * w.tilde_s_prod * acc
}

/// Calculate the different-spin two electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: `WicksPairView`, same-spin and different-spin Wick's reference pair intermediates.
/// `l_ex_a`: `ExcitationSpin`, spin alpha excitation array for |{}^\Lambda \Psi\rangle.
/// `g_ex_a`: `ExcitationSpin`, spin alpha excitation array for |{}^\Gamma \Psi\rangle.
/// `l_ex_b`: `ExcitationSpin`, spin beta excitation array for |{}^\Lambda \Psi\rangle.
/// `g_ex_b`: `ExcitationSpin`, spin beta excitation array for |{}^\Gamma \Psi\rangle.
/// `scratch`: `WickScratch`, scratch space for Wick's quantities.
/// `tol`: `f64`, tolerance for singularity handling in determinant evaluation.
/// # Returns
/// `f64`, different-spin two-electron Hamiltonian matrix element.
#[inline(always)]
pub fn lg_h2_diff(w: &WicksPairView, l_ex_a: &ExcitationSpin, g_ex_a: &ExcitationSpin, l_ex_b: &ExcitationSpin, g_ex_b: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {

    // If the per-spin excitation rank L + 1 is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the two electron matrix element is zero.
    let la = l_ex_a.holes.len() + g_ex_a.holes.len();
    let lb = l_ex_b.holes.len() + g_ex_b.holes.len();
    if w.aa.m > la + 1 {return 0.0;}
    if w.bb.m > lb + 1 {return 0.0;}
    scratch.resizelalb(la, lb);

    construct_determinant_lables(l_ex_a, g_ex_a, &mut scratch.rows_label_a, &mut scratch.cols_label_a);
    construct_determinant_lables(l_ex_b, g_ex_b, &mut scratch.rows_label_b, &mut scratch.cols_label_b);

    // Convert the contraction determinant labels into actual indices.
    scratch.rows_a.clear();
    for &(s, _t, i) in &scratch.rows_label_a {scratch.rows_a.push(label_to_idx(s, i, w.aa.nmo));}
    scratch.cols_a.clear();
    for &(s, _t, i) in &scratch.cols_label_a {scratch.cols_a.push(label_to_idx(s, i, w.aa.nmo));}
    scratch.rows_b.clear();
    for &(s, _t, i) in &scratch.rows_label_b {scratch.rows_b.push(label_to_idx(s, i, w.bb.nmo));}
    scratch.cols_b.clear();
    for &(s, _t, i) in &scratch.cols_label_b {scratch.cols_b.push(label_to_idx(s, i, w.bb.nmo));}
    
    let x0aa = w.aa.x(0);
    let y0aa = w.aa.y(0);
    let x1aa = w.aa.x(1);
    let y1aa = w.aa.y(1);
    let x0bb = w.bb.x(0);
    let y0bb = w.bb.y(0);
    let x1bb = w.bb.x(1);
    let y1bb = w.bb.y(1);

    // Build two full contraction determinants each having exclusively (X0, Y0) or (X1, Y1) per spin.
    build_d(&mut scratch.deta0, &x0aa, &y0aa, &scratch.rows_a, &scratch.cols_a);    
    build_d(&mut scratch.deta1, &x1aa, &y1aa, &scratch.rows_a, &scratch.cols_a);
    build_d(&mut scratch.detb0, &x0bb, &y0bb, &scratch.rows_b, &scratch.cols_b);
    build_d(&mut scratch.detb1, &x1bb, &y1bb, &scratch.rows_b, &scratch.cols_b);
    
    let mut acc = 0.0;

    // Iterate over all possible distributions of alpha zeros amongst the columns.
    for bits_a in iter_m_combinations(la + 1, w.aa.m) {
        let ma0 = (bits_a & 1) as usize;
        let inda: u64 = bits_a >> 1;
        
        // Construct mixed X0, Y0, X1, Y1 determinant for the given bitstring of size L + 1 for
        // spin alpha.
        mix_columns(&mut scratch.deta_mix, &scratch.deta0, &scratch.deta1, inda);
        let Some(det_deta) = adjugate_transpose(&mut scratch.adjt_deta, &mut scratch.invsla, &mut scratch.lua, &scratch.deta_mix, tol) else {continue;};
        
        // Iterate over all possible distributions of beta zeros amongst the columns.
        for bits_b in iter_m_combinations(lb + 1, w.bb.m) {
            let mb0 = (bits_b & 1) as usize;
            let indb: u64 = bits_b >> 1;
            
            // Construct mixed X0, Y0, X1, Y1 determinant for the given bitstring of size L + 1 for
            // spin beta.
            mix_columns(&mut scratch.detb_mix, &scratch.detb0, &scratch.detb1, indb);
            let Some(det_detb) = adjugate_transpose(&mut scratch.adjt_detb, &mut scratch.invslb, &mut scratch.lub, &scratch.detb_mix, tol) else {continue;};

            let mut contrib = 0.0f64;

            // Equation 30 sum over all distributions of zeros with V0_{ab} matrix multiplying the
            // alpha and beta size La and Lb determinants respectively.
            contrib += w.ab.vab0[ma0][mb0] * det_deta * det_detb;

            // Equation 34 alpha sum over all distributions of zeros and iterating over all columns replacing
            // the current column with the V matrices to form a new determinant.
            for k in 0..la {
                // Choose correct column of V based upon zero distributions.
                let mak = ((bits_a >> (k + 1)) & 1) as usize;
                let vcol = &w.ab.vab(ma0, mb0, mak);
                let ck = scratch.cols_a[k];

                let mut corr = 0.0;
                for r in 0..la {
                    let new = vcol[(scratch.rows_a[r], ck)];
                    corr += (new - scratch.deta_mix[(r, k)]) * scratch.adjt_deta[(r, k)];
                }
                contrib -= (det_deta + corr) * det_detb;
            }

            // Equation 34 beta sum over all distributions of zeros and iterating over all columns replacing
            // the current column with the V matrices to form a new determinant.
            for k in 0..lb {
                // Choose correct column of V based upon zero distributions.
                let mbk = ((bits_b >> (k + 1)) & 1) as usize;
                let vcol = &w.ab.vba(mb0, ma0, mbk);
                let ck = scratch.cols_b[k];

                let mut corr = 0.0;
                for r in 0..lb {
                    let new = vcol[(scratch.rows_b[r], ck)];
                    corr += (new - scratch.detb_mix[(r, k)]) * scratch.adjt_detb[(r, k)];
                }
                contrib -= (det_detb + corr) * det_deta;
            }

            // Equation 38 beta double sum over excitation ranks (i, j) which vary the two electron 4-index tensor
            // II (no exchange, but otherwise analogous to J in same spin case) accordingly, and within each double 
            // sum is another sum over columns of determinant where column k is replaced with II matrices to form a new determinant.
            for (i, &ra) in scratch.rows_a.iter().enumerate() {
                for (j, &ca) in scratch.cols_a.iter().enumerate() {

                    let cofa = scratch.adjt_deta[(i, j)];
                    let ma1 = ((bits_a >> (j + 1)) & 1) as usize;
                
                    // Inner sum iterates over all columns of the determinant and replaces column
                    // k with the appropriate column of II.
                    for k in 0..lb {
                        let mbk = ((bits_b >> (k + 1)) & 1) as usize;
                        
                        // Extract slice of II corresponding to the current distribution of zeros and
                        // get the correct minor matrix so as to align with the L - 1 dimensions.
                        // We want tensor IIba here, but we only store IIab but may read IIab and
                        // use the relation IIba[mb0][mbk][ma0][ma1] = IIab[ma0][ma1][mb0][mbk].
                        let iib = w.ab.iiab(ma0, ma1, mb0, mbk);
                        slice4swap(&mut scratch.iisliceb, &iib, &scratch.rows_b, &scratch.cols_b, ra, ca);

                        let mut corr = 0.0;
                        for r in 0..lb {
                            let new = scratch.iisliceb[(r, k)];
                            corr += (new - scratch.detb_mix[(r, k)]) * scratch.adjt_detb[(r, k)];
                        }
                        contrib += 0.5 * cofa * (det_detb + corr);
                    }
                }
            }

            // Equation 38 alpha double sum over excitation ranks (i, j) which vary the two electron 4-index tensor
            // II (no exchange, but otherwise analogous to J in same spin case) accordingly, and within each double 
            // sum is another sum over columns of determinant where column k is replaced with II matrices to form a new determinant.
            for (i, &rb) in scratch.rows_b.iter().enumerate() {
                for (j, &cb) in scratch.cols_b.iter().enumerate() {

                    let cofb = scratch.adjt_detb[(i, j)];
                    let mb1 = ((bits_b >> (j + 1)) & 1) as usize;

                    // Inner sum iterates over all columns of the L determinant and replaces column
                    // k with the appropriate column of II.
                    for k in 0..la {
                        let mak = ((bits_a >> (k + 1)) & 1) as usize;
                        
                        // Extract slice of II corresponding to the current distribution of zeros and
                        // get the correct minor matrix so as to align with the L - 1 dimensions.
                        let iia = &w.ab.iiab(ma0, mak, mb0, mb1);
                        slice4(&mut scratch.iislicea, iia, &scratch.rows_a, &scratch.cols_a, rb, cb);

                        let mut corr = 0.0;
                        for r in 0..la {
                            let new = scratch.iislicea[(r, k)];
                            corr += (new - scratch.deta_mix[(r, k)]) * scratch.adjt_deta[(r, k)];
                        }
                        contrib += 0.5 * cofb * (det_deta + corr);
                    }
                }
            }
            acc += contrib;
        }
    }
    (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * acc
}

