// nonorthogonalwicks/view.rs
use std::ops::Deref;
use std::ptr::NonNull;

use ndarray::ArrayView2;

use super::types::{
    DiffSpinMeta, DiffSpinOffset, PairMeta, PairOffset, SameSpinMeta, SameSpinOffset,
};
use crate::noci::NOCIScalar;

/// Storage for data which allows the Wicks objects to be viewed.
#[derive(Clone)]
pub struct WicksView<T: NOCIScalar> {
    /// Pointer to contiguous data which contains all intermediates.
    pub(crate) slab: NonNull<T>,
    /// Length of storage.
    pub(crate) slab_len: usize,
    /// Number of reference determinants.
    pub(crate) nref: usize,
    /// Offset gives where in the storage each tensor for pair p begins.
    pub(crate) off: Vec<PairOffset>,
    /// Scalars that are cheap to store.
    pub(crate) meta: Vec<PairMeta<T>>,
}

// Implying that WicksView can be shared across threads.
unsafe impl<T: NOCIScalar> Sync for WicksView<T> {}
unsafe impl<T: NOCIScalar> Send for WicksView<T> {}

impl<T: NOCIScalar> WicksView<T> {
    /// Map a pair index (lp, gp) into a 1D flattened index.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `lp`: Pair index 1.
    /// - `gp`: Pair index 2.
    /// # Returns
    /// - `usize`: Flattened pair index.
    fn idx(
        &self,
        lp: usize,
        gp: usize,
    ) -> usize {
        lp * self.nref + gp
    }

    /// Get a pointer to the start of the shared tensor storage.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// # Returns
    /// - `*const T`: Pointer to the start of the shared tensor slab.
    fn slab_ptr(&self) -> *const T {
        self.slab.as_ptr() as *const T
    }

    /// Read tensor slab beginning at a given offset and interpret the following n * n elements as
    /// a n by n matrix. Lifetime elision '_ ensures that the ArrayView2 may not outlive the borrow
    /// of self (WicksView) which in turn is only valid while the remote memory storage is valid.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `off`: Offset from the beginning of the tensor slab in units of T.
    /// - `n`: Size of matrix to be read.
    /// # Returns
    /// - `ArrayView2<'_, T>`: Matrix view into the tensor slab.
    fn view2(
        &self,
        off: usize,
        n: usize,
    ) -> ArrayView2<'_, T> {
        unsafe { ArrayView2::from_shape_ptr((n, n), self.slab_ptr().add(off)) }
    }

    /// Read tensor slab beginning at a given offset and interpret the following `n * n` elements
    /// as a flat row-major matrix slice without constructing an ndarray view.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `off`: Offset from the beginning of the tensor slab in units of `T`.
    /// - `n`: Matrix dimension.
    /// # Returns
    /// - `&[T]`: Slice containing the `n * n` matrix entries in row-major order.
    #[inline(always)]
    fn slice2(
        &self,
        off: usize,
        n: usize,
    ) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.slab_ptr().add(off), n * n) }
    }

    /// Read tensor slab beginning at a given offset and interpret the following `n * n * n * n`
    /// elements as a flat row-major rank-4 tensor slice without constructing an ndarray view.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `off`: Offset from the beginning of the tensor slab in units of `T`.
    /// - `n`: Tensor dimension along each axis.
    /// # Returns
    /// - `&[T]`: Slice containing the `n^4` tensor entries in row-major order.
    #[inline(always)]
    fn slice4(
        &self,
        off: usize,
        n: usize,
    ) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.slab_ptr().add(off), n * n * n * n) }
    }

    /// Return a view for precomputed intermediates for a given lp, gp. Lifetime elision '_ ensures
    /// that the ArrayView4 may not outlive the borrow of self (WicksView) which in turn is only valid
    /// while the remote memory storage is valid.
    /// # Arguments:
    /// - `self`: View into Wick's intermediates.
    /// - `lp`: Pair index 1.
    /// - `gp`: Pair index 2.
    /// # Returns
    /// - `WicksPairView<'_, T>`: Grouped view of the same-spin and different-spin intermediates for the pair.
    pub(crate) fn pair(
        &self,
        lp: usize,
        gp: usize,
    ) -> WicksPairView<'_, T> {
        let idx = self.idx(lp, gp);
        let meta = &self.meta[idx];
        let off = &self.off[idx];

        let aa = SameSpinView {
            meta: &meta.aa,
            w: self,
            off: &off.aa,
        };
        let bb = SameSpinView {
            meta: &meta.bb,
            w: self,
            off: &off.bb,
        };
        let ab = DiffSpinView {
            meta: &meta.ab,
            w: self,
            off: &off.ab,
        };

        WicksPairView { aa, bb, ab }
    }
}

// Read only view of same-spin Wick's intermediates.
#[derive(Clone, Copy)]
pub(crate) struct SameSpinView<'a, T: NOCIScalar> {
    /// Metadata and scalar intermediates for this same-spin reference pair.
    pub(crate) meta: &'a SameSpinMeta<T>,
    /// Parent view providing access to the contiguous tensor slab.
    pub(crate) w: &'a WicksView<T>,
    /// Offsets for all same-spin intermediates belonging to this reference pair.
    pub(crate) off: &'a SameSpinOffset,
}

impl<T: NOCIScalar> Deref for SameSpinView<'_, T> {
    type Target = SameSpinMeta<T>;

    /// Borrow the same-spin metadata for transparent field access.
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// # Returns
    /// - `&SameSpinMeta<T>`: Borrowed same-spin metadata.
    fn deref(&self) -> &Self::Target {
        self.meta
    }
}

impl<'a, T: NOCIScalar> SameSpinView<'a, T> {
    /// Get tensor dimension n.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// # Returns
    /// - `usize`: Tensor dimension `nmo`.
    pub(crate) fn n(&self) -> usize {
        self.nmo
    }

    /// Get a view to the `X[mi]` matrix.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi`: Zero distribution selector.
    /// # Returns
    /// - `ArrayView2<'_, T>`: View of the `X[mi]` matrix.
    pub(crate) fn x(
        &self,
        mi: usize,
    ) -> ArrayView2<'_, T> {
        self.w.view2(self.off.x[mi], self.n())
    }

    /// Get a view to the `Y[mi]` matrix.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi`: Zero distribution selector.
    /// # Returns
    /// - `ArrayView2<'_, T>`: View of the `Y[mi]` matrix.
    pub(crate) fn y(
        &self,
        mi: usize,
    ) -> ArrayView2<'_, T> {
        self.w.view2(self.off.y[mi], self.n())
    }

    /// Get a view to the basis-space `X[mi]` contraction matrix used for RDM indices.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi`: Zero distribution selector.
    /// # Returns
    /// - `ArrayView2<'_, T>`: Basis-space X contraction matrix.
    #[cfg(feature = "nocc")]
    pub(crate) fn xrdm(
        &self,
        mi: usize,
        nbas: usize,
    ) -> ArrayView2<'_, T> {
        self.w.view2(self.off.xrdm[mi], nbas)
    }

    /// Get a view to the basis-space `Y[mi]` contraction matrix used for RDM indices.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi`: Zero distribution selector.
    /// # Returns
    /// - `ArrayView2<'_, T>`: Basis-space Y contraction matrix.
    #[cfg(feature = "nocc")]
    pub(crate) fn yrdm(
        &self,
        mi: usize,
        nbas: usize,
    ) -> ArrayView2<'_, T> {
        self.w.view2(self.off.yrdm[mi], nbas)
    }

    /// Get a slice of the transpoed Hamiltonian `F[mi][mj]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi, mj`: Zero distribution selectors.
    /// # Returns
    /// - `&[T]`: Slice of the transpoed Hamiltonian `F[mi][mj]` matrix data.
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn fh_t_slice(
        &self,
        mi: usize,
        mj: usize,
    ) -> &[T] {
        self.w.slice2(self.off.fh[mi][mj], self.n())
    }

    /// Get a slice of the transpoed Fock `F[mi][mj]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi, mj`: Zero distribution selectors.
    /// # Returns
    /// - `&[T]`: Slice of the transpoed Fock `F[mi][mj]` matrix data.
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn ff_t_slice(
        &self,
        mi: usize,
        mj: usize,
    ) -> &[T] {
        self.w.slice2(self.off.ff[mi][mj], self.n())
    }

    /// Get a slice of the transpoed `V[mi][mj][mk]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `mi, mj, mk`: Zero distribution selectors.
    /// # Returns
    /// - `&[T]`: Slice of the transpoed `V[mi][mj][mk]` matrix data.
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn v_t_slice(
        &self,
        mi: usize,
        mj: usize,
        mk: usize,
    ) -> &[T] {
        self.w.slice2(self.off.v[mi][mj][mk], self.n())
    }

    /// Get a slice of the `J[slot]` tensor.
    /// # Arguments:
    /// - `self`: View to same-spin Wick's intermediates.
    /// - `slot`: Compressed storage slot for the requested J tensor.
    /// # Returns
    /// - `&[T]`: Slice of the requested J tensor data.
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn j_slice(
        &self,
        slot: usize,
    ) -> &[T] {
        self.w.slice4(self.off.j[slot], self.n())
    }
}

/// Read only view of diff-spin Wick's intermediates.
#[derive(Clone, Copy)]
pub(crate) struct DiffSpinView<'a, T: NOCIScalar> {
    /// Metadata and scalar intermediates for this different-spin reference pair.
    pub(crate) meta: &'a DiffSpinMeta<T>,
    /// Parent view providing access to the contiguous tensor slab.
    w: &'a WicksView<T>,
    /// Offsets for all different-spin intermediates belonging to this reference pair.
    off: &'a DiffSpinOffset,
}

impl<T: NOCIScalar> Deref for DiffSpinView<'_, T> {
    type Target = DiffSpinMeta<T>;

    /// Borrow the different-spin metadata for transparent field access.
    /// # Arguments:
    /// - `self`: Different-spin Wick view.
    /// # Returns
    /// - `&DiffSpinMeta<T>`: Borrowed different-spin metadata.
    fn deref(&self) -> &Self::Target {
        self.meta
    }
}

impl<'a, T: NOCIScalar> DiffSpinView<'a, T> {
    /// Get tensor dimension n.
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// # Returns
    /// - `usize`: Tensor dimension `nmo`.
    pub(crate) fn n(&self) -> usize {
        self.nmo
    }

    /// Get a slice of the transpoed `Vab[ma0][mb0][mak]` tensor.
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// - `ma0, mb0, mak`: Zero distribution selectors.
    /// # Returns
    /// - `&[T]`: Slice of the transpoed `Vab[ma0][mb0][mak]` matrix data.
    #[inline(always)]
    pub fn vab_t_slice(
        &self,
        ma0: usize,
        mb0: usize,
        mak: usize,
    ) -> &[T] {
        self.w.slice2(self.off.vab[ma0][mb0][mak], self.n())
    }

    /// Get a slice of the transpoed `Vba[mb0][ma0][mbk]` tensor.
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// - `mb0, ma0, mbk`: Zero distribution selectors.
    /// # Returns
    /// - `&[T]`: Slice of the transpoed `Vba[mb0][ma0][mbk]` matrix data.
    #[inline(always)]
    pub fn vba_t_slice(
        &self,
        mb0: usize,
        ma0: usize,
        mbk: usize,
    ) -> &[T] {
        self.w.slice2(self.off.vba[mb0][ma0][mbk], self.n())
    }

    /// Get a slice of the `IIab[ma0][maj][mb0][mbj]` tensor.
    /// # Arguments:
    /// - `self`: View to diff-spin Wick's intermediates.
    /// - `ma0, maj, mb0, mbj`: Zero distribution selectors.
    /// # Returns
    /// - `&[T]`: Slice of the `IIab[ma0][maj][mb0][mbj]` tensor data.
    #[inline(always)]
    pub fn iiab_slice(
        &self,
        ma0: usize,
        maj: usize,
        mb0: usize,
        mbj: usize,
    ) -> &[T] {
        self.w.slice4(self.off.iiab[ma0][maj][mb0][mbj], self.n())
    }
}

/// Storage for views of each type of spin pairing.
#[derive(Clone, Copy)]
pub(crate) struct WicksPairView<'a, T: NOCIScalar> {
    /// Same-spin alpha-alpha intermediates for the reference pair.
    pub(crate) aa: SameSpinView<'a, T>,
    /// Same-spin beta-beta intermediates for the reference pair.
    pub(crate) bb: SameSpinView<'a, T>,
    /// Different-spin alpha-beta intermediates for the reference pair.
    pub(crate) ab: DiffSpinView<'a, T>,
}
