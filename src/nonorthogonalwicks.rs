// nonorthogonalwicks.rs 
use std::ptr::NonNull;

use ndarray::{Array1, Array2, ArrayView2, Array4, ArrayView4, Axis, s};
use ndarray_linalg::{SVD, Determinant};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::{ExcitationSpin};

use crate::maths::{einsum_ba_ab_real, eri_ao2mo, loewdin_x_real, minor, adjugate_transpose, det_thresh, mix_columns, build_d};
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
    ///     `self`: WicksShared, view and RMA for Wick's intermediates.
    pub fn view(&self) -> &WicksView {&self.view}
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
    ///     `self`: WicksView, view into Wick's intermediates.
    ///     `lp`: usize, pair index 1.
    ///     `gp`: usize, pair index 2.
    pub fn idx(&self, lp: usize, gp: usize) -> usize {
        lp * self.nref + gp
    }
    
    /// Get a pointer to the start of the shared tensor storage.
    /// # Arguments:
    ///     `self`: WicksView, view into Wick's intermediates.
    fn slab_ptr(&self) -> *const f64 {
        self.slab.as_ptr()
    }
    
    /// Read tensor slab beginning at a given offset and interpret the following n * n elements as
    /// a n by n matrix. Lifetime elision '_ ensures that the ArrayView2 may not outlive the borrow
    /// of self (WicksView) which in turn is only valid while the remote memory storage is valid.
    /// # Arguments:
    ///     `self`: WicksView, view into Wick's intermediates.
    ///     `off_f64`: usize, offset from the beginning of the tensor slab in units of f64.
    ///     `n`: usize, size of matrix to be read.
    pub fn view2(&self, off_f64: usize, n: usize) -> ArrayView2<'_, f64> {
        unsafe {ArrayView2::from_shape_ptr((n, n), self.slab_ptr().add(off_f64))}
    }

    /// Read tensor slab beginning at a given offset and interpret the following n * n * n * n elements as
    /// a n by n by n by n tensor. Lifetime elision '_ ensures that the ArrayView4 may not outlive the borrow
    /// of self (WicksView) which in turn is only valid while the remote memory storage is valid.
    /// # Arguments:
    ///     `self`: WicksView, view into Wick's intermediates.
    ///     `off_f64`: usize, offset from the beginning of the tensor slab in units of f64.
    ///     `n`: usize, size of tensor to be read.
    pub fn view4(&self, off_f64: usize, n: usize) -> ArrayView4<'_, f64> {
        unsafe {ArrayView4::from_shape_ptr((n, n, n, n), self.slab_ptr().add(off_f64))}
    }
    
    /// Return a view for precomputed intermediates for a given lp, gp. Lifetime elision '_ ensures 
    /// that the ArrayView4 may not outlive the borrow of self (WicksView) which in turn is only valid 
    /// while the remote memory storage is valid.
    /// # Arguments:
    ///     `self`: WicksView, view into Wick's intermediates.
    ///     `lp`: usize, pair index 2. 
    pub fn pair(&self, lp: usize, gp: usize) -> WicksPairView<'_> {
        let idx = self.idx(lp, gp);

        let aa = SameSpinView {nmo: self.meta[idx].aa.nmo, m: self.meta[idx].aa.m, tilde_s_prod: self.meta[idx].aa.tilde_s_prod, 
                               phase: self.meta[idx].aa.phase, f0: self.meta[idx].aa.f0, v0: self.meta[idx].aa.v0, w: self, off: self.off[idx].aa};
        let bb = SameSpinView {nmo: self.meta[idx].bb.nmo, m: self.meta[idx].bb.m, tilde_s_prod: self.meta[idx].bb.tilde_s_prod, 
                               phase: self.meta[idx].bb.phase, f0: self.meta[idx].bb.f0, v0: self.meta[idx].bb.v0, w: self, off: self.off[idx].bb};
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
    pub f0: [f64; 2],
    pub v0: [f64; 3],
    w: &'a WicksView,
    off: SameSpinOffset,
}

impl<'a> SameSpinView<'a> {
    /// Get tensor dimension n. 
    /// # Arguments:
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    fn n(&self) -> usize {2 * self.nmo}
    
    /// Get a view to the X[mi] matrix.
    /// # Arguments:
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    ///     `mi`: usize, zero distribution selector. 
    pub fn x(&self, mi: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.x[mi], self.n())
    }
    
    /// Get a view to the Y[mi] matrix.
    /// # Arguments:
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    ///     `mi`: usize, zero distribution selector. 
    pub fn y(&self, mi: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.y[mi], self.n())
    }
    
    /// Get a view to the F[mi][mj] tensor.
    /// # Arguments:
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    ///     `mi, mj`: usize, zero distribution selectors. 
    pub fn f(&self, mi: usize, mj: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.f[mi][mj], self.n())
    }
    
    /// Get a view to the V[mi][mj][mk] tensor.
    /// # Arguments:
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    ///     `mi, mj, mk`: usize, zero distribution selector. 
    pub fn v(&self, mi: usize, mj: usize, mk: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.v[mi][mj][mk], self.n())
    }
    
    /// Get a view to the J[mi][mj][mk][ml] tensor.
    /// # Arguments:
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    ///     `mi, mj, mk, ml`: usize, zero distribution selector. 
    pub fn j(&self, mi: usize, mj: usize, mk: usize, ml: usize) -> ArrayView4<'_, f64> {
        self.w.view4(self.off.j[mi][mj][mk][ml], self.n())
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
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    fn n(&self) -> usize {2 * self.nmo}
    
    /// Get a view to the Vab[ma0][mb0][mak] tensor.
    /// # Arguments:
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    ///     `ma0, mb0, mak`: usize, zero distribution selector. 
    pub fn vab(&self, ma0: usize, mb0: usize, mak: usize) -> ArrayView2<'a, f64> {
        self.w.view2(self.off.vab[ma0][mb0][mak], self.n())
    }

    /// Get a view to the Vba[mb0][ma0][mak] tensor.
    /// # Arguments:
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    ///     `ma0, mb0, mak`: usize, zero distribution selector.
    pub fn vba(&self, mb0: usize, ma0: usize, mbk: usize) -> ArrayView2<'a, f64> {
        self.w.view2(self.off.vba[mb0][ma0][mbk], self.n())
    }

    /// Get a view to the IIab[ma0][maj][mb0][mbj] tensor.
    /// # Arguments:
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    ///     `ma0, maj, mb0, mbj`: usize, zero distribution selector.
    pub fn iiab(&self, ma0: usize, maj: usize, mb0: usize, mbj: usize) -> ArrayView4<'a, f64> {
        self.w.view4(self.off.iiab[ma0][maj][mb0][mbj], self.n())
    }

    /// Get a view to the IIba[mb0][mbj][ma0][maj] tensor.
    /// # Arguments:
    ///     `self`: SameSpinView, view to same-spin Wick's intermediates.
    ///     `ma0, maj, mb0, mbj`: usize, zero distribution selector.
    pub fn iiba(&self, mb0: usize, mbj: usize, ma0: usize, maj: usize) -> ArrayView4<'a, f64> {
        self.w.view4(self.off.iiba[mb0][mbj][ma0][maj], self.n())
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
    pub f0: [f64; 2],
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
    pub f: [[usize; 2]; 2],
    pub v: [[[usize; 2]; 2]; 2],
    pub j: [[[[usize; 2]; 2]; 2]; 2],
}

// Storage for diff-spin per reference-pair offset tables into the shared contiguous tensor storage.
#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug)]
pub struct DiffSpinOffset {
    pub vab: [[[usize; 2]; 2]; 2],
    pub vba: [[[usize; 2]; 2]; 2],
    pub iiab: [[[[usize; 2]; 2]; 2]; 2],
    pub iiba: [[[[usize; 2]; 2]; 2]; 2],
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

    pub f0: [f64; 2], // F0[mi]
    pub f: [[Array2<f64>; 2]; 2], // F[mi][mj]

    pub v0: [f64; 3], // V0[mi][mj]
    pub v: [[[Array2<f64>; 2]; 2]; 2], // V[mi][mj][mk]

    pub j: [[[[Array4<f64>; 2]; 2]; 2]; 2],  // J[mi][mj][mk][ml]

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
    pub iiba: [[[[Array4<f64>; 2]; 2]; 2]; 2], // iiba[mb0][mbk][ma0][maj]
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
///     `nref`: usize, number of references.
///     `nmo`: usize, number of molecular orbitals.
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
        for mi in 0..2 {for mj in 0..2 {p.aa.f[mi][mj] = i; i += nn2; }}
        for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {p.aa.v[mi][mj][mk] = i; i += nn2;}}}
        for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {for ml in 0..2 {p.aa.j[mi][mj][mk][ml] = i; i += nn4;}}}}

        for mi in 0..2 {p.bb.x[mi] = i; i += nn2;}
        for mi in 0..2 {p.bb.y[mi] = i; i += nn2;}
        for mi in 0..2 {for mj in 0..2 {p.bb.f[mi][mj] = i; i += nn2;}}
        for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {p.bb.v[mi][mj][mk] = i; i += nn2;}}}
        for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {for ml in 0..2 {p.bb.j[mi][mj][mk][ml] = i; i += nn4;}}}}

        for ma0 in 0..2 {for mb0 in 0..2 {for mk in 0..2 {p.ab.vab[ma0][mb0][mk] = i; i += nn2; }}}
        for mb0 in 0..2 {for ma0 in 0..2 {for mk in 0..2 {p.ab.vba[mb0][ma0][mk] = i; i += nn2; }}}

        for ma0 in 0..2 {for maj in 0..2 {for mb0 in 0..2 {for mbj in 0..2 {p.ab.iiab[ma0][maj][mb0][mbj] = i; i += nn4;}}}}
        for mb0 in 0..2 {for mbj in 0..2 {for ma0 in 0..2 {for maj in 0..2 {p.ab.iiba[mb0][mbj][ma0][maj] = i; i += nn4;}}}}
    }
    (off, i) 
}

/// Fill the same-spin data owning structs with the same-spin Wick's intermediates using the
/// prescribed offsets.
/// # Arguments:
///     `slab`: [f64], contiguous tensor storage.
///     `o`: SameSpinOffset, offsets into the storage.
///     `w`: SameSpinBuild, owned Wick's intermediates.
pub fn write_same_spin(slab: &mut [f64], o: &SameSpinOffset, w: &SameSpinBuild) {
    write2(slab, o.x[0], &w.x[0]);
    write2(slab, o.x[1], &w.x[1]);
    write2(slab, o.y[0], &w.y[0]);
    write2(slab, o.y[1], &w.y[1]);
    for mi in 0..2 {for mj in 0..2 {write2(slab, o.f[mi][mj], &w.f[mi][mj]);}}
    for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {write2(slab, o.v[mi][mj][mk], &w.v[mi][mj][mk]);}}}
    for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {for ml in 0..2 {write4(slab, o.j[mi][mj][mk][ml], &w.j[mi][mj][mk][ml]);}}}}
}

/// Fill the diff-spin data owning structs with the diff-spin Wick's intermediates using the
/// prescribed offsets.
/// # Arguments:
///     `slab`: [f64], contiguous tensor storage.
///     `o`: DiffSpinOffset, offsets into the storage.
///     `w`: DiffSpinBuild, owned Wick's intermediates.
pub fn write_diff_spin(slab: &mut [f64], o: &DiffSpinOffset, w: &DiffSpinBuild) {
    for ma0 in 0..2 {for mb0 in 0..2 {for mk in 0..2 {write2(slab, o.vab[ma0][mb0][mk], &w.vab[ma0][mb0][mk]);}}}
    for mb0 in 0..2 {for ma0 in 0..2 {for mk in 0..2 {write2(slab, o.vba[mb0][ma0][mk], &w.vba[mb0][ma0][mk]);}}}
    for ma0 in 0..2 {for maj in 0..2 {for mb0 in 0..2 {for mbj in 0..2 {
        write4(slab, o.iiab[ma0][maj][mb0][mbj], &w.iiab[ma0][maj][mb0][mbj]);
        write4(slab, o.iiba[mb0][mbj][ma0][maj], &w.iiba[mb0][mbj][ma0][maj]);
    }}}}
}

/// Copy matrix into tensor slab provided it is contiguous.
/// # Arguments:
///     `slab`: [f64], contiguous tensor storage.
///     `off`: usize, offset for the start position.
///     `a`: Array2, matrix to copy.
fn write2(slab: &mut [f64], off: usize, a: &ndarray::Array2<f64>) {
    let src = a.as_slice().expect("Array2 must be contiguous");
    slab[off..off + src.len()].copy_from_slice(src);
}

/// Copy tensor into tensor slab provided it is contiguous.
/// # Arguments:
///     `slab`: [f64], contiguous tensor storage.
///     `off`: usize, offset for the start position.
///     `a`: Array4, tensor to copy.
fn write4(slab: &mut [f64], off: usize, a: &ndarray::Array4<f64>) {
    let src = a.as_slice().expect("Array4 must be contiguous");
    slab[off..off + src.len()].copy_from_slice(src);
}

/// Write 2D slice of 4D J or II tensors into provided output scratch. The given slice is t[r, c, i, j]  
/// where r, c are rows, columns and i, j are fixed indices.
/// # Arguments:
///     `out`: Array2, preallocated output scratch.
///     `t`: ArrayView4, view of a 4D tensor.
///     `rows`: [usize], length excitation rank map from row labels to tensor index.
///     `cols`: [usize], length excitation rank map from col labels to tensor index.
///     `i_fixed`: usize, fixed tensor indices for the j dimension.
///     `j_fixed`: usize, fixed tensor indices for the j dimension.
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
    pub fn new() -> Self {Self::default()}
    
    /// If the previously allocated size of the scratch space is the not the same in  
    /// the same spin case resize all the scratch space quantities to be correct.
    /// # Arguments:
    ///     `self`: WickScratch, scratch space for Wick's quantities.
    ///     `l`: usize, excitation rank.
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
    ///     `self`: WickScratch, scratch space for Wick's quantities.
    ///     `la`: usize, excitation rank spin alpha.
    ///     `lb`: usize, excitation rank spin beta.
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
            self.invslam1 = Array1::zeros(lbm1);
        }
    }
}

impl SameSpinBuild {
    /// Constructor for the WicksReferencePair object of SameSpin which contains the precomputed values
    /// per pair of reference determinants required to evaluate arbitrary excited determinants 
    /// in O(1) time when the excitations are of the same spin.
    /// # Arguments:
    ///     `eri`: Array4, electron repulsion integrals. 
    ///     `h_munu`: Array2, AO core Hamiltonian.
    ///     `s_munu`: Array2, AO overlap matrix.
    ///     `g_c`: Array2, full AO coefficient matrix of |^\Gamma\Psi\rangle.
    ///     `l_c`: Array2, full AO coefficient matrix of |^\Lambda\Psi\rangle.
    ///     `go`: Array1, occupancy vector for |^\Gamma\Psi\rangle.
    ///     `lo`: Array1, occupancy vector for |^\Lambda\Psi\rangle. 
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

        // Construct the {}^{\Gamma\Lambda} F_0^{m_k} and {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j}
        // intermediates required for one body matrix elements.
        let (f0_0, f00) = Self::construct_f(l_c, h_munu, &x[0], &y[0]);
        let (_, f01) = Self::construct_f(l_c, h_munu, &x[0], &y[1]);
        let (_, f10) = Self::construct_f(l_c, h_munu, &x[1], &y[0]);
        let (f0_1, f11) = Self::construct_f(l_c, h_munu, &x[1], &y[1]);
        let f0: [f64; 2] = [f0_0, f0_1];
        let f: [[Array2<f64>; 2]; 2] = [[f00, f01], [f10, f11]];
        
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
        // basis.
        let mut j: [[[[Array4<f64>; 2]; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| {
                    std::array::from_fn(|_| {
                        Array4::<f64>::zeros((2 * nmo, 2 * nmo, 2 * nmo, 2 * nmo))
                    })
                })
            })
        });
        let combos: Vec<(usize, usize, usize, usize)> =
            (0..2).flat_map(|mi| (0..2).flat_map(move |mj|
            (0..2).flat_map(move |mk| (0..2).map(move |ml| (mi, mj, mk, ml)))
        )).collect();
        let blocks: Vec<((usize, usize, usize, usize), Array4<f64>)> = combos.into_par_iter().map(|(mi,mj,mk,ml)| {
            let mut blk = eri_ao2mo(eri, &cx[mi], &xc[mj], &cx[mk], &xc[ml]);
            let ex = blk.view().permuted_axes([0, 2, 1, 3]).to_owned().as_standard_layout().to_owned();
            blk -= &ex;
            let blk = blk.as_standard_layout().to_owned();
            ((mi, mj, mk, ml), blk)
        }).collect();
        for ((mi,mj,mk,ml), blk) in blocks {
            j[mi][mj][mk][ml] = blk;
        }

        Self {x, y, f0, f, v0, v, j, tilde_s_prod, phase, m, nmo}
    }

    /// Calculate the overlap matrix between two sets of occupied orbitals as:
    ///     {}^{\Gamma\Lambda} S_{ij} = \sum_{\mu\nu} ({}^\Gamma C^*)_i^\mu S_{\mu\nu} ({}^\Lambda C)_j^\nu
    /// # Arguments:
    ///     `g_c_occ`: Array2, occupied coefficients ({}^\Gamma C^*)_i^\mu.
    ///     `l_c_occ`: Array2, occupied coefficients ({}^\Lambda C)_j^\nu 
    ///     `s_munu`: Array2, AO overlap matrix S_{\mu\nu}.
    pub fn calculate_mo_overlap_matrix(l_c_occ: &Array2<f64>, g_c_occ: &Array2<f64>, s_munu: &Array2<f64>) -> Array2<f64> {
        l_c_occ.t().dot(&s_munu.dot(g_c_occ))
    }
    
    /// Perform singular value decomposition on the occupied orbital overlap matrix {}^{\Gamma\Lambda} S_{ij} as:
    ///     {}^{\Gamma\Lambda} \mathbf{S} = \mathbf{U} {}^{\Gamma\Lambda} \mathbf{\tilde{S}} \mathbf{V}^\dagger,
    /// and rotate the occupied coefficients:
    ///     |{}^\Gamma \Psi_i\rangle = \sum_{\mu} {}^\Gamma c_i^\mu U_{ij} |\phi_\mu \rangle.
    ///     |{}^\Lambda \Psi_j\rangle = \sum_{\nu} {}^\Lambda c_j^\nu V_{ij} |\phi_\nu \rangle.
    /// # Arguments:
    ///     `g_c_occ`: Array2, occupied coefficients ({}^\Gamma C^*)_i^\mu.
    ///     `l_c_occ`: Array2, occupied coefficients ({}^\Lambda C)_j^\nu.
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
    ///     `gl_tilde_s`: Array1, vector of diagonal single values of {}^{\Gamma\Lambda} \tilde{S}.
    ///     `g_c_tilde_occ`: Array2, rotated occupied coefficients ({}^\Gamma \tilde{C}^*)_i^\mu.
    ///     `l_c_tilde_occ`: Array2, rotated occupied coefficients ({}^\Lambda \tilde{C})_j^\nu.
    ///     `zeros`: Vec<usize>, indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S}. 
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
    ///     `s_munu`: Array2, AO overlap matrix.
    ///     `g_c`: Array2, full AO coefficient matrix of |^\Gamma\Psi\rangle.
    ///     `l_c`: Array2, full AO coefficient matrix of |^\Lambda\Psi\rangle.
    ///     `gl_m`: Array2, M matrix{}^{\Gamma\Lambda} M^{\sigma\tau, 0} or  {}^{\Gamma\Lambda} M^{\sigma\tau, 1}. 
    ///     `subtract`: bool, whether to use m_k = 0 or m_k = 1. 
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
    ///     `g_c`: Array2, full AO coefficient matrix of |^\Gamma\Psi\rangle.
    ///     `l_c`: Array2, full AO coefficient matrix of |^\Lambda\Psi\rangle.
    ///     `h_munu`: Array2, one-electron core AO hamiltonian.
    ///     `x`: Array2, {}^{\Gamma\Lambda} X_{ij}^{m_k}.
    ///     `y`: Array2, {}^{\Gamma\Lambda} Y_{ij}^{m_k}
    fn construct_f(l_c: &Array2<f64>, h_munu: &Array2<f64>, x: &Array2<f64>, y: &Array2<f64>) -> (f64, Array2<f64>) {
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
    ///     `eri`: Array4, AO basis ERIs (not-antisymmetrised).
    ///     `m`: Array2, {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
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
    ///     `eri`: Array4, AO basis ERIs (not-antisymmetrised).
    ///     `m`: Array2, {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
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
    ///     `eri`: Array4, electron repulsion integrals. 
    ///     `h_munu`: Array2, AO core Hamiltonian.
    ///     `s_munu`: Array2, AO overlap matrix.
    ///     `g_ca`: Array2, spin alpha AO coefficient matrix of |^\Gamma\Psi\rangle.
    ///     `g_cb`: Array2, spin beta AO coefficient matrix of |^\Gamma\Psi\rangle.
    ///     `l_ca`: Array2, spin alpha AO coefficient matrix of |^\Lambda\Psi\rangle.
    ///     `l_cb`: Array2, spin beta AO coefficient matrix of |^\Lambda\Psi\rangle.
    ///     `goa`: Array1, alpha occupancy vector for |^\Gamma\Psi\rangle.
    ///     `gob`: Array1, beta occupancy vector for |^\Gamma\Psi\rangle.
    ///     `loa`: Array1, alpha occupancy vector for |^\Lambda\Psi\rangle.
    ///     `lob`: Array1, beta occupancy vector for |^\Lambda\Psi\rangle.
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
        let mut iiba: [[[[Array4<f64>; 2]; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| Array4::<f64>::zeros((2 * nmo, 2 * nmo, 2 * nmo, 2 * nmo))))
            })
        });

        let combos: Vec<(usize, usize, usize, usize)> =
            (0..2).flat_map(|mi| (0..2).flat_map(move |mj|
            (0..2).flat_map(move |mk| (0..2).map(move |ml| (mi, mj, mk, ml)))
        )).collect();

        let blocks: Vec<((usize, usize, usize, usize), (Array4<f64>, Array4<f64>))> = combos.into_par_iter().map(|(ma0, maj, mb0, mbj)| {
            let blk = eri_ao2mo(eri, cx_a[ma0], xc_a[maj], cx_b[mb0], xc_b[mbj]).as_standard_layout().to_owned();
            let blkt = blk.view().permuted_axes([2, 3, 0, 1]).to_owned().as_standard_layout().to_owned();
            ((ma0, maj, mb0, mbj), (blk, blkt))
        }).collect();

        for ((ma0, maj, mb0, mbj), (blk, blkt)) in blocks {
            iiab[ma0][maj][mb0][mbj] = blk;
            iiba[mb0][mbj][ma0][maj] = blkt;
        }

        Self {vab0, vab, vba0, vba, iiab, iiba}
    }
        
    /// Build the left and right factorisations of the {}^{\Gamma\Lambda} X_{ij}^{m_k} and 
    /// {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices such that our intermediates can be computed more
    /// easily.
    /// # Arguments:
    ///     `m`: Array2, {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
    ///     `s`: Array2, AO overlap matrix S_{\mu\nu}.
    ///     `cx`: Array2, AO coefficient matrix for \Lambda. Should be renamed.
    ///     `cw`: Array2, AO coefficient matrix for \Gamma. Should be renamed.
    ///     `i`: usize, selector for m being 0 or 1.
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
///     `l_ex`: Excitation, left excitations relative to reference \Lambda in the bra.
///     `g_ex`: Excitation, right excitations relative to reference \Gamma in the ket.
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
///     `l`: usize, total excitation rank.
///     `m`: usize, number of zero-overlap orbital couplings.
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
///     `side`: Side, whether the label is from the bra (\Lambda) or ket (\Gamma).
///     `p`: usize, orbital index in MO basis.
///     `nmo`: usize, number of MOs.
fn label_to_idx(side: Side, p: usize, nmo: usize) -> usize {
    match side {
        Side::Lambda => p,        // First block.
        Side::Gamma  => nmo + p,  // Second block.
    }
}

/// All the same spin routines require a number of the same quantities. It is more efficient to
/// precompute them once here rather than inside each matrix element routine.
/// # Arguments:
///     `w`: SameSpin: same spin Wick's reference pair intermediates.
///     `l_ex`: ExcitationSpin, spin resolved excitation array for |{}^\Lambda \Psi\rangle.
///     `g_ex`: ExcitationSpin, spin resolved excitation array for |{}^\Gamma \Psi\rangle. 
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
///     `w`: SameSpin: same spin Wick's reference pair intermediates.
///     `l_ex`: ExcitationSpin, spin resolved excitation array for |{}^\Lambda \Psi\rangle.
///     `g_ex`: ExcitationSpin, spin resolved excitation array for |{}^\Gamma \Psi\rangle. 
pub fn lg_overlap(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {

    // If the total excitation rank is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the overlap element is zero.
    let l = l_ex.holes.len() + g_ex.holes.len();
    if w.m > l {return 0.0;}

    let mut acc = 0.0;

    // Iterate over all possible distributions of zeros amongst the columns.
    for bitstring in iter_m_combinations(l, w.m) {
        mix_columns(&mut scratch.det_mix, &scratch.det0, &scratch.det1, bitstring);
        let Some(d) = det_thresh(&scratch.det_mix, tol) else {continue;};
        acc += d;
    }
    w.phase * w.tilde_s_prod * acc
}

/// Calculate one electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
///     `w`: SameSpin: same spin Wick's reference pair intermediates.
///     `l_ex`: ExcitationSpin, spin resolved excitation array for |{}^\Lambda \Psi\rangle.
///     `g_ex`: ExcitationSpin, spin resolved excitation array for |{}^\Gamma \Psi\rangle.
pub fn lg_h1(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    
    // If the total excitation rank L + 1 is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the one electron matrix element is zero.
    let l = l_ex.holes.len() + g_ex.holes.len();
    if w.m > (l + 1) {return 0.0;}

    let mut acc = 0.0;

    // Iterate over all possible distributions of zeros amongst the columns.
    for bitstring in iter_m_combinations(l + 1, w.m) {
        let mi = (bitstring & 1) == 1;

        let mcol: u64 = bitstring >> 1;
        mix_columns(&mut scratch.det_mix, &scratch.det0, &scratch.det1, mcol);
        
        // First term in the sum of Eqn 26 involves the same sum over all possible bitstrings with
        // the F0 matrix multiplying the contraction determinant.
        let Some(det_det) = adjugate_transpose(&mut scratch.adjt_det, &mut scratch.invs, &mut scratch.lu, &scratch.det_mix, tol) else {continue;};
        let mut contrib = det_det * w.f0[mi as usize];

        // Remaining terms in Eqn 26 consist of the same sum over all possible bitstrings, but now
        // we iterate over columns, forming a new determinant for each with the current column
        // replaced by the F matrices.
        for b in 0..l {
            // Find correct zero index for this column and use to select F.
            let mj = ((bitstring >> (b + 1)) & 1) == 1; 
            let f = w.f(mi as usize, mj as usize);

            // Replace column of det with F matrices.
            let cb = scratch.cols[b];
            for a in 0..l {
                let ra = scratch.rows[a];
                scratch.fcol[a] = f[(ra, cb)];
            }
            let dcol = &scratch.det_mix.column(b);
            let acol = scratch.adjt_det.column(b);
            scratch.dv.assign(&scratch.fcol);
            scratch.dv -= dcol;
            let det_det_f = det_det + scratch.dv.dot(&acol);
            contrib -= det_det_f;
        }
        acc += contrib;
    }
    w.phase * w.tilde_s_prod * acc
}

/// Calculate the same-spin two electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
///     `w`: SameSpin: same spin Wick's reference pair intermediates.
///     `l_ex`: ExcitationSpin, spin resolved excitation array for |{}^\Lambda \Psi\rangle.
///     `g_ex`: ExcitationSpin, spin resolved excitation array for |{}^\Gamma \Psi\rangle.
pub fn lg_h2_same(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    
    // If the total excitation rank L + 2 is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the two electron matrix element is zero.
    let l = l_ex.holes.len() + g_ex.holes.len();
    if w.m > (l + 2) {return 0.0;}
    
    let mut acc = 0.0;

    // Iterate over all possible distributions of zeros amongst the columns.
    for bits in iter_m_combinations(l + 2, w.m) {
        let m1 = (bits & 1) == 1;
        let m2 = ((bits >> 1) & 1) == 1;
        let mcol = |k: usize| ((bits >> (k + 2)) & 1) == 1;
        let ind: u64 = bits >> 2;
        
        // Construct mixed X0, Y0, X1, Y1 determinant for the given bitstring of size L + 2.
        mix_columns(&mut scratch.det_mix, &scratch.det0, &scratch.det1, ind);
        let Some(det_det) = adjugate_transpose(&mut scratch.adjt_det, &mut scratch.invs, &mut scratch.lu, &scratch.det_mix, tol) else {continue;};

        let mut contrib = 0.0f64;

        // Equation 30 sum over all distributions of zeros with V0 matrix multiplying size L + 2 determinant.
        let q = (m1 as usize) + (m2 as usize);
        let x = w.v0[q] * det_det;
        contrib += x;

        // Equation 34 sum over all distributions of zeros and iterating over all columns replacing
        // the current column with the V matrices to form a new determinant.
        for k in 0..l {
            let mk = mcol(k);
            // Choose correct column of V based upon zero distributions.
            let vcol = w.v(m1 as usize, m2 as usize, mk as usize);

            // Calculate det(D (k --> V)), that is, determinant D with column k replaced by V using
            // the identity, det(D (k --> V)) = det(D) + (V - D(k))^T adj(D(k)) in which D(k)
            // indicates the kth column of D.
            for r in 0..l {
                    scratch.v1[r] = vcol[(scratch.rows[r], scratch.cols[k])];
            }
            let v2 = scratch.det_mix.column(k);      
            let a =  scratch.adjt_det.column(k);
            scratch.dv1.assign(&scratch.v1);
            scratch.dv1 -= &v2;
            contrib -= 2.0 * (det_det + scratch.dv1.dot(&a));
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

                let mj = mcol(j);
                
                // Inner sum iterates over all columns of the L - 1 determinant and replaces column
                // k with the appropriate column of J.
                for k2 in 0..(l - 1) {
                    let k_full = if k2 < j {k2} else {k2 + 1};
                    let mk = mcol(k_full);
                    
                    // Extract slice of J corresponding to the current distribution of zeros and
                    // get the correct minor matrix so as to align with the L - 1 dimensions.
                    let j4 = w.j(m1 as usize, m2 as usize, mk as usize, mj as usize);
                    slice4(&mut scratch.jslice_full, &j4, &scratch.rows, &scratch.cols, ri_fixed, cj_fixed);
                    minor(&mut scratch.jslice2, &scratch.jslice_full, i, j);   
                    
                    // Calculate det(D (k --> J)), that is, determinant D with column k replaced by J using
                    // the identity, det(D (k --> J)) = det(D) + (J - D(k))^T adj(D(k)) in which D(k)
                    // indicates the kth column of D.
                    let v1 = scratch.jslice2.column(k2);
                    let v2 = scratch.det_mix2.column(k2);
                    let a =  scratch.adjt_det2.column(k2);
                    scratch.dv1m.assign(&v1);
                    scratch.dv1m -= &v2;
                    contrib += 1.0 * phase * (det_det2 + scratch.dv1m.dot(&a));
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
///     `w`: SameSpin: same spin Wick's reference pair intermediates.
///     `l_ex`: ExcitationSpin, spin resolved excitation array for |{}^\Lambda \Psi\rangle.
///     `g_ex`: ExcitationSpin, spin resolved excitation array for |{}^\Gamma \Psi\rangle.
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
    scratch.rows_a.extend(scratch.rows_label_a.iter().map(|(s, _t, i)| label_to_idx(*s, *i, w.aa.nmo)));
    scratch.cols_a.clear();
    scratch.cols_a.extend(scratch.cols_label_a.iter().map(|(s, _t, i)| label_to_idx(*s, *i, w.aa.nmo)));
    scratch.rows_b.clear();
    scratch.rows_b.extend(scratch.rows_label_b.iter().map(|(s, _t, i)| label_to_idx(*s, *i, w.bb.nmo)));
    scratch.cols_b.clear();
    scratch.cols_b.extend(scratch.cols_label_b.iter().map(|(s, _t, i)| label_to_idx(*s, *i, w.bb.nmo)));
    
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
        let ma0 = (bits_a & 1) == 1;
        let ma_col = |k: usize| ((bits_a >> (k + 1)) & 1) == 1;
        let inda: u64 = bits_a >> 1;
        
        // Construct mixed X0, Y0, X1, Y1 determinant for the given bitstring of size L + 1 for
        // spin alpha.
        mix_columns(&mut scratch.deta_mix, &scratch.deta0, &scratch.deta1, inda);
        let Some(det_deta) = adjugate_transpose(&mut scratch.adjt_deta, &mut scratch.invsla, &mut scratch.lua, &scratch.deta_mix, tol) else {continue;};
        
        // Iterate over all possible distributions of beta zeros amongst the columns.
        for bits_b in iter_m_combinations(lb + 1, w.bb.m) {
            let mb0 = (bits_b & 1) == 1;
            let mb_col = |k: usize| ((bits_b >> (k + 1)) & 1) == 1;
            let indb: u64 = bits_b >> 1;
            
            // Construct mixed X0, Y0, X1, Y1 determinant for the given bitstring of size L + 1 for
            // spin beta.
            mix_columns(&mut scratch.detb_mix, &scratch.detb0, &scratch.detb1, indb);
            let Some(det_detb) = adjugate_transpose(&mut scratch.adjt_detb, &mut scratch.invslb, &mut scratch.lub, &scratch.detb_mix, tol) else {continue;};

            let mut contrib = 0.0f64;

            // Equation 30 sum over all distributions of zeros with V0_{ab} matrix multiplying the
            // alpha and beta size La and Lb determinants respectively.
            let x = w.ab.vab0[ma0 as usize][mb0 as usize] * det_deta * det_detb;
            contrib += x;

            // Equation 34 alpha sum over all distributions of zeros and iterating over all columns replacing
            // the current column with the V matrices to form a new determinant.
            for k in 0..la {
                // Choose correct column of V based upon zero distributions.
                let mak = ma_col(k);
                let vcol = &w.ab.vab(ma0 as usize, mb0 as usize, mak as usize);

                // Calculate det(D (k --> V)), that is, determinant D with column k replaced by V using
                // the identity, det(D (k --> V)) = det(D) + (V - D(k))^T adj(D(k)) in which D(k)
                // indicates the kth column of D.
                for r in 0..la {
                    scratch.v1a[r] = vcol[(scratch.rows_a[r], scratch.cols_a[k])];
                }
                let v2 = scratch.deta_mix.column(k);      
                let a = scratch.adjt_deta.column(k);
                scratch.dv1a.assign(&scratch.v1a);
                scratch.dv1a -= &v2;
                contrib -= (det_deta + scratch.dv1a.dot(&a)) * det_detb;
            }

            // Equation 34 beta sum over all distributions of zeros and iterating over all columns replacing
            // the current column with the V matrices to form a new determinant.
            for k in 0..lb {
                // Choose correct column of V based upon zero distributions.
                let mbk = mb_col(k);
                let vcol = &w.ab.vba(mb0 as usize, ma0 as usize, mbk as usize);
                
                // Calculate det(D (k --> V)), that is, determinant D with column k replaced by V using
                // the identity, det(D (k --> V)) = det(D) + (V - D(k))^T adj(D(k)) in which D(k)
                // indicates the kth column of D.
                for r in 0..lb {
                    scratch.v1b[r] = vcol[(scratch.rows_b[r], scratch.cols_b[k])];
                }
                let v2 = scratch.detb_mix.column(k);      
                let a = scratch.adjt_detb.column(k);
                scratch.dv1b.assign(&scratch.v1b);
                scratch.dv1b -= &v2;
                contrib -= (det_detb + scratch.dv1b.dot(&a)) * det_deta;
            }

            // Equation 38 beta double sum over excitation ranks (i, j) which vary the two electron 4-index tensor
            // II (no exchange, but otherwise analogous to J in same spin case) accordingly, and within each double 
            // sum is another sum over columns of determinant where column k is replaced with II matrices to form a new determinant.
            for (i, &ra) in scratch.rows_a.iter().enumerate() {
                for (j, &ca) in scratch.cols_a.iter().enumerate() {
                    let phase = if ((i + j) & 1) == 0 {1.0} else {-1.0};
                
                    // Find the (L - 1) by (L - 1) determinant removing row and column i, j.
                    minor(&mut scratch.deta_mix_minor, &scratch.deta_mix, i, j);
                    let Some(det_deta_minor_mix) = det_thresh(&scratch.deta_mix_minor, tol) else {continue;};
                    
                    // Inner sum iterates over all columns of the determinant and replaces column
                    // k with the appropriate column of II.
                    for k in 0..lb {
                        let mbk = mb_col(k);
                        let ma1 = ma_col(j);
                        
                        // Extract slice of II corresponding to the current distribution of zeros and
                        // get the correct minor matrix so as to align with the L - 1 dimensions.
                        let iib = &w.ab.iiba(mb0 as usize, mbk as usize, ma0 as usize, ma1 as usize);
                        slice4(&mut scratch.iisliceb, iib, &scratch.rows_b, &scratch.cols_b, ra, ca);
                        
                        // Calculate det(D (k --> J)), that is, determinant D with column k replaced by J using
                        // the identity, det(D (k --> J)) = det(D) + (J - D(k))^T adj(D(k)) in which D(k)
                        // indicates the kth column of D. 
                        for r in 0..lb {
                            scratch.v1b[r] = scratch.iisliceb[(r, k)];
                        }
                        let v2 = scratch.detb_mix.column(k);      
                        let a = scratch.adjt_detb.column(k);
                        scratch.dv1b.assign(&scratch.v1b);
                        scratch.dv1b -= &v2;
                        contrib += 0.5 * phase * (det_detb + scratch.dv1b.dot(&a)) * det_deta_minor_mix;
                    }
                }
            }

            // Equation 38 alpha double sum over excitation ranks (i, j) which vary the two electron 4-index tensor
            // II (no exchange, but otherwise analogous to J in same spin case) accordingly, and within each double 
            // sum is another sum over columns of determinant where column k is replaced with II matrices to form a new determinant.
            for (i, &rb) in scratch.rows_b.iter().enumerate() {
                for (j, &cb) in scratch.cols_b.iter().enumerate() {
                    let phase = if ((i + j) & 1) == 0 {1.0} else {-1.0};
                    
                    // Find the (L - 1) by (L - 1) determinant removing row and column i, j.
                    minor(&mut scratch.detb_mix_minor, &scratch.detb_mix, i, j);
                    let Some(det_detb_minor_mix) = det_thresh(&scratch.detb_mix_minor, tol) else {continue;};
                    
                    // Inner sum iterates over all columns of the L determinant and replaces column
                    // k with the appropriate column of II.
                    for k in 0..la {
                        let mak = ma_col(k);
                        let mb1 = mb_col(j);
                        
                        // Extract slice of II corresponding to the current distribution of zeros and
                        // get the correct minor matrix so as to align with the L - 1 dimensions.
                        let iia = &w.ab.iiab(ma0 as usize, mak as usize, mb0 as usize, mb1 as usize);
                        slice4(&mut scratch.iislicea, iia, &scratch.rows_a, &scratch.cols_a, rb, cb);
                        
                        // Calculate det(D (k --> J)), that is, determinant D with column k replaced by J using
                        // the identity, det(D (k --> J)) = det(D) + (J - D(k))^T adj(D(k)) in which D(k)
                        // indicates the kth column of D.
                        for r in 0..la {
                            scratch.v1a[r] = scratch.iislicea[(r, k)];
                        }
                        let v2 = scratch.deta_mix.column(k);      
                        let a = scratch.adjt_deta.column(k);
                        scratch.dv1a.assign(&scratch.v1a);
                        scratch.dv1a -= &v2;
                        contrib += 0.5 * phase * (det_deta + scratch.dv1a.dot(&a)) * det_detb_minor_mix;
                    }
                }
            }
            acc += contrib;
        }
    }

    (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * acc
}


