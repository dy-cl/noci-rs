// nonorthogonalwicks/view.rs
use std::ptr::NonNull;

use ndarray::{ArrayView2};

use super::types::{PairMeta, PairOffset, SameSpinOffset, DiffSpinOffset};

/// Storage for data which allows the Wicks objects to be viewed.
#[derive(Clone)]
pub struct WicksView {
    /// Pointer to contiguous data which contains all intermediates.
    pub(crate) slab: NonNull<f64>, 
    /// Length of storage.
    pub(crate) slab_len: usize,
    /// Number of reference determinants.
    pub(crate) nref: usize,
    /// Offset gives where in the storage each tensor for pair p begins.
    pub(crate) off: Vec<PairOffset>,
    /// Scalars that are cheap to store.
    pub(crate) meta: Vec<PairMeta>,
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
    fn idx(&self, lp: usize, gp: usize) -> usize {
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
    fn view2(&self, off_f64: usize, n: usize) -> ArrayView2<'_, f64> {
        unsafe {ArrayView2::from_shape_ptr((n, n), self.slab_ptr().add(off_f64))}
    }

    /// Read tensor slab beginning at a given offset and interpret the following `n * n` elements
    /// as a flat row-major matrix slice without constructing an ndarray view.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `off_f64`: Offset from the beginning of the tensor slab in units of `f64`.
    /// - `n`: Matrix dimension.
    /// # Returns
    /// - `&[f64]`: Slice containing the `n * n` matrix entries in row-major order.
    #[inline(always)]
    fn slice2(&self, off_f64: usize, n: usize) -> &[f64] {
        unsafe {std::slice::from_raw_parts(self.slab_ptr().add(off_f64), n * n)}
    }

    /// Read tensor slab beginning at a given offset and interpret the following `n * n * n * n`
    /// elements as a flat row-major rank-4 tensor slice without constructing an ndarray view.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `off_f64`: Offset from the beginning of the tensor slab in units of `f64`.
    /// - `n`: Tensor dimension along each axis.
    /// # Returns
    /// - `&[f64]`: Slice containing the `n^4` tensor entries in row-major order.
    #[inline(always)]
    fn slice4(&self, off_f64: usize, n: usize) -> &[f64] {
        unsafe {std::slice::from_raw_parts(self.slab_ptr().add(off_f64), n * n * n * n)}
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
    pub(crate) fn pair(&self, lp: usize, gp: usize) -> WicksPairView<'_> {
        let idx = self.idx(lp, gp);

        let aa = SameSpinView {nmo: self.meta[idx].aa.nmo, m: self.meta[idx].aa.m, tilde_s_prod: self.meta[idx].aa.tilde_s_prod, 
                               phase: self.meta[idx].aa.phase, f0f: self.meta[idx].aa.f0f, f0h: self.meta[idx].aa.f0h, v0: self.meta[idx].aa.v0, 
                               w: self, off: self.off[idx].aa};
        let bb = SameSpinView {nmo: self.meta[idx].bb.nmo, m: self.meta[idx].bb.m, tilde_s_prod: self.meta[idx].bb.tilde_s_prod, 
                               phase: self.meta[idx].bb.phase, f0f: self.meta[idx].bb.f0f, f0h: self.meta[idx].bb.f0h, v0: self.meta[idx].bb.v0, 
                               w: self, off: self.off[idx].bb};
        let ab = DiffSpinView {nmo: self.meta[idx].ab.nmo, vab0: self.meta[idx].ab.vab0, w: self, off: self.off[idx].ab};

        WicksPairView {aa, bb, ab}
    }
}

// Read only view of same-spin Wick's intermediates.  
#[derive(Clone, Copy)]
pub(crate) struct SameSpinView<'a> {
    /// Number of molecular orbitals for this spin block.
    pub(crate) nmo: usize,
    /// Number of zero-overlap orbital pairs in the biorthogonal basis for this spin block.
    pub(crate) m: usize,
    /// Product of the non-zero singular values, i.e. the reduced overlap for this spin block.
    pub(crate) tilde_s_prod: f64,
    /// Overall phase associated with this same-spin block.
    pub(crate) phase: f64,
    /// Zeroth-order Fock one-body scalar contributions for the two branch choices.
    pub(crate) f0f: [f64; 2],
    /// Zeroth-order Hamiltonian one-body scalar contributions for the two branch choices.
    pub(crate) f0h: [f64; 2],
    /// Zeroth-order two-body scalar contributions for the allowed branch combinations.
    pub(crate) v0: [f64; 3],
    /// Parent view providing access to the contiguous tensor slab.
    pub(crate) w: &'a WicksView,
    /// Offsets for all same-spin intermediates belonging to this reference pair.
    pub(crate) off: SameSpinOffset,
}

impl<'a> SameSpinView<'a> {
    /// Get tensor dimension n. 
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// # Returns
    /// - `usize`: Tensor dimension `2 * nmo`.
    pub(crate) fn n(&self) -> usize {2 * self.nmo}
    
    /// Get a view to the `X[mi]` matrix.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi`: Zero distribution selector. 
    /// # Returns
    /// - `ArrayView2<'_, f64>`: View of the `X[mi]` matrix.
    pub(crate) fn x(&self, mi: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.x[mi], self.n())
    }
    
    /// Get a view to the `Y[mi]` matrix.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi`: Zero distribution selector. 
    /// # Returns
    /// - `ArrayView2<'_, f64>`: View of the `Y[mi]` matrix.
    pub(crate) fn y(&self, mi: usize) -> ArrayView2<'_, f64> {
        self.w.view2(self.off.y[mi], self.n())
    }
    
    /// Get a slice of the transpoed Hamiltonian `F[mi][mj]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi, mj`: Zero distribution selectors.
    /// # Returns
    /// - `&'a [f64]`: Slice of the transpoed Hamiltonian `F[mi][mj]` matrix data.
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn fh_t_slice(&self, mi: usize, mj: usize) -> &'a [f64] {
        self.w.slice2(self.off.fh[mi][mj], self.n())
    }

    /// Get a slice of the transpoed Fock `F[mi][mj]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi, mj`: Zero distribution selectors.
    /// # Returns
    /// - `&'a [f64]`: Slice of the transpoed Fock `F[mi][mj]` matrix data.
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn ff_t_slice(&self, mi: usize, mj: usize) -> &'a [f64] {
        self.w.slice2(self.off.ff[mi][mj], self.n())
    }

    /// Get a slice of the transpoed `V[mi][mj][mk]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi, mj, mk`: Zero distribution selectors.
    /// # Returns
    /// - `&'a [f64]`: Slice of the transpoed `V[mi][mj][mk]` matrix data.
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn v_t_slice(&self, mi: usize, mj: usize, mk: usize) -> &'a [f64] {
        self.w.slice2(self.off.v[mi][mj][mk], self.n())
    }

    /// Get a slice of the `J[slot]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `slot`: Compressed storage slot for the requested J tensor.
    /// # Returns
    /// - `&'a [f64]`: Slice of the requested J tensor data.
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn j_slice(&self, slot: usize) -> &'a [f64] {
        self.w.slice4(self.off.j[slot], self.n())
    }
}

/// Read only view of diff-spin Wick's intermediates. 
#[derive(Clone, Copy)]
pub(crate) struct DiffSpinView<'a> {
    /// Number of molecular orbitals for this different-spin block.
    pub(crate) nmo: usize,
    /// Zeroth-order mixed-spin Vab scalar contributions for the branch combinations.
    pub(crate) vab0: [[f64; 2]; 2],
    /// Parent view providing access to the contiguous tensor slab.
    w: &'a WicksView,
    /// Offsets for all different-spin intermediates belonging to this reference pair.
    off: DiffSpinOffset,
}

impl<'a> DiffSpinView<'a> {
    /// Get tensor dimension n. 
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// # Returns
    /// - `usize`: Tensor dimension `2 * nmo`.
    pub(crate) fn n(&self) -> usize {2 * self.nmo}
    
    /// Get a slice of the transpoed `Vab[ma0][mb0][mak]` tensor.
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// - `ma0, mb0, mak`: Zero distribution selectors.
    /// # Returns
    /// - `&'a [f64]`: Slice of the transpoed `Vab[ma0][mb0][mak]` matrix data.
    #[inline(always)]
    pub fn vab_t_slice(&self, ma0: usize, mb0: usize, mak: usize) -> &'a [f64] {
        self.w.slice2(self.off.vab[ma0][mb0][mak], self.n())
    }

    /// Get a slice of the transpoed `Vba[mb0][ma0][mbk]` tensor.
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// - `mb0, ma0, mbk`: Zero distribution selectors.
    /// # Returns
    /// - `&'a [f64]`: Slice of the transpoed `Vba[mb0][ma0][mbk]` matrix data.
    #[inline(always)]
    pub fn vba_t_slice(&self, mb0: usize, ma0: usize, mbk: usize) -> &'a [f64] {
        self.w.slice2(self.off.vba[mb0][ma0][mbk], self.n())
    }

    /// Get a slice of the `IIab[ma0][maj][mb0][mbj]` tensor.
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// - `ma0, maj, mb0, mbj`: Zero distribution selectors.
    /// # Returns
    /// - `&'a [f64]`: Slice of the `IIab[ma0][maj][mb0][mbj]` tensor data.
    #[inline(always)]
    pub fn iiab_slice(&self, ma0: usize, maj: usize, mb0: usize, mbj: usize) -> &'a [f64] {
        self.w.slice4(self.off.iiab[ma0][maj][mb0][mbj], self.n())
    }
}

/// Storage for views of each type of spin pairing.
#[derive(Clone, Copy)]
pub(crate) struct WicksPairView<'a> {
    /// Same-spin alpha-alpha intermediates for the reference pair.
    pub(crate) aa: SameSpinView<'a>,
    /// Same-spin beta-beta intermediates for the reference pair.
    pub(crate) bb: SameSpinView<'a>,
    /// Different-spin alpha-beta intermediates for the reference pair.
    pub(crate) ab: DiffSpinView<'a>,
}

