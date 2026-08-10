// nonorthogonalwicks/view.rs
// Standard library imports.
use std::ops::Deref;
use std::ptr::NonNull;

// External crate imports.
use ndarray::ArrayView2;

// Crate-root imports.
use crate::noci::NOCIScalar;

// Parent/sibling imports.
use super::types::{
    DiffSpinMeta, DiffSpinOffset, PairMeta, PairOffset, SameSpinMeta, SameSpinOffset,
};

/// Read-only address, offset and metadata view over the contiguous slab of precomputed
/// nonorthogonal Wick intermediates.
#[derive(Clone)]
pub struct WicksView<T: NOCIScalar> {
    /// Pointer to the first entry of the contiguous tensor slab.
    pub(crate) slab: NonNull<T>,
    /// Length of the tensor slab in units of `T`.
    pub(crate) slab_len: usize,
    /// Number of reference determinants.
    pub(crate) nref: usize,
    /// Per-reference-pair offsets locating each stored matrix and tensor.
    pub(crate) off: Vec<PairOffset>,
    /// Per-reference-pair scalar metadata stored outside the tensor slab.
    pub(crate) meta: Vec<PairMeta<T>>,
}

// The backing allocation is kept alive by `WicksShared`, and this view exposes only immutable slab access.
unsafe impl<T: NOCIScalar> Sync for WicksView<T> {}
unsafe impl<T: NOCIScalar> Send for WicksView<T> {}

impl<T: NOCIScalar> WicksView<T> {
    /// Map the ordered reference-pair indices (x,w) to the flattened pair index.
    /// # Arguments:
    /// - `self`: View over the stored Wick intermediates.
    /// - `lp`: Bra-reference index x.
    /// - `gp`: Ket-reference index w.
    /// # Returns
    /// - `usize`: Flattened index of the ordered reference pair (x,w).
    fn idx(
        &self,
        lp: usize,
        gp: usize,
    ) -> usize {
        lp * self.nref + gp
    }

    /// Return a pointer to the first entry of the contiguous tensor slab.
    /// # Arguments:
    /// - `self`: View over the stored Wick intermediates.
    /// # Returns
    /// - `*const T`: Pointer to the first slab entry.
    fn slab_ptr(&self) -> *const T {
        self.slab.as_ptr() as *const T
    }

    /// `Interpret n \times n contiguous slab entries beginning at off as a row-major matrix.`
    /// The returned view cannot outlive the borrow of `self`, whose lifetime is tied to the
    /// backing shared-memory or memory-mapped allocation.
    /// # Arguments:
    /// - `self`: View over the stored Wick intermediates.
    /// - `off`: Offset from the start of the tensor slab in units of `T`.
    /// - `n`: Matrix dimension.
    /// # Returns
    /// - `ArrayView2<'_, T>`: Matrix view into the tensor slab.
    fn view2(
        &self,
        off: usize,
        n: usize,
    ) -> ArrayView2<'_, T> {
        unsafe { ArrayView2::from_shape_ptr((n, n), self.slab_ptr().add(off)) }
    }

    /// `Interpret n \times n contiguous slab entries beginning at off as a flat row-major`
    /// matrix slice without constructing an ndarray view.
    /// # Arguments:
    /// - `self`: View over the stored Wick intermediates.
    /// - `off`: Offset from the start of the tensor slab in units of `T`.
    /// - `n`: Matrix dimension.
    /// # Returns
    /// - `&[T]`: Flat row-major matrix slice.
    #[inline(always)]
    fn slice2(
        &self,
        off: usize,
        n: usize,
    ) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.slab_ptr().add(off), n * n) }
    }

    /// `Interpret n^4 contiguous slab entries beginning at off as a flat row-major rank-four`
    /// tensor slice without constructing an ndarray view.
    /// # Arguments:
    /// - `self`: View over the stored Wick intermediates.
    /// - `off`: Offset from the start of the tensor slab in units of `T`.
    /// - `n`: Dimension of each tensor axis.
    /// # Returns
    /// - `&[T]`: Flat row-major rank-four tensor slice.
    #[inline(always)]
    fn slice4(
        &self,
        off: usize,
        n: usize,
    ) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.slab_ptr().add(off), n * n * n * n) }
    }

    /// Return grouped alpha-alpha, beta-beta and alpha-beta views for an ordered reference pair
    /// with bra reference x and ket reference w.
    /// # Arguments:
    /// - `self`: View over the stored Wick intermediates.
    /// - `lp`: Bra-reference index x.
    /// - `gp`: Ket-reference index w.
    /// # Returns
    /// - `WicksPairView<'_, T>`: Grouped spin-resolved views for the ordered pair (x,w).
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

/// Read-only view of the same-spin intermediates for one spin sector of an ordered reference pair.
#[derive(Clone, Copy)]
pub(crate) struct SameSpinView<'a, T: NOCIScalar> {
    /// Metadata and scalar same-spin intermediates for this reference pair and spin sector.
    pub(crate) meta: &'a SameSpinMeta<T>,
    /// Parent view providing access to the contiguous tensor slab.
    pub(crate) w: &'a WicksView<T>,
    /// Offsets locating the same-spin matrices and tensors in the slab.
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
    /// Return the molecular-orbital dimension of each stored same-spin matrix or tensor axis.
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// # Returns
    /// - `usize`: Molecular-orbital dimension `nmo`.
    pub(crate) fn n(&self) -> usize {
        self.nmo
    }

    /// `Return the X^{(m_i)} fundamental-contraction matrix.`
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// - `mi`: `Fundamental-contraction assignment m_i \in \{0,1\}.`
    /// # Returns
    /// - `ArrayView2<'_, T>`: `View of X^{(m_i)}.`
    pub(crate) fn x(
        &self,
        mi: usize,
    ) -> ArrayView2<'_, T> {
        self.w.view2(self.off.x[mi], self.n())
    }

    /// `Return the Y^{(m_i)} fundamental-contraction matrix.`
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// - `mi`: `Fundamental-contraction assignment m_i \in \{0,1\}.`
    /// # Returns
    /// - `ArrayView2<'_, T>`: `View of Y^{(m_i)}.`
    pub(crate) fn y(
        &self,
        mi: usize,
    ) -> ArrayView2<'_, T> {
        self.w.view2(self.off.y[mi], self.n())
    }

    /// `Return X^{(m_i)} as a flat row-major slice for specialised scalar kernels.`
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// - `mi`: `Fundamental-contraction assignment m_i \in \{0,1\}.`
    /// # Returns
    /// - `&[T]`: `Flat row-major X^{(m_i)} matrix.`
    pub(crate) fn x_slice(
        &self,
        mi: usize,
    ) -> &[T] {
        self.w.slice2(self.off.x[mi], self.n())
    }

    /// `Return Y^{(m_i)} as a flat row-major slice for specialised scalar kernels.`
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// - `mi`: `Fundamental-contraction assignment m_i \in \{0,1\}.`
    /// # Returns
    /// - `&[T]`: `Flat row-major Y^{(m_i)} matrix.`
    pub(crate) fn y_slice(
        &self,
        mi: usize,
    ) -> &[T] {
        self.w.slice2(self.off.y[mi], self.n())
    }

    /// `Return X^{(m_i)} represented in the external RDM basis.`
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// - `mi`: `Fundamental-contraction assignment m_i \in \{0,1\}.`
    /// - `nbas`: Dimension of the external RDM basis.
    /// # Returns
    /// - `ArrayView2<'_, T>`: `External-basis X^{(m_i)} matrix.`
    #[cfg(feature = "nocc")]
    pub(crate) fn xrdm(
        &self,
        mi: usize,
        nbas: usize,
    ) -> ArrayView2<'_, T> {
        self.w.view2(self.off.xrdm[mi], nbas)
    }

    /// `Return Y^{(m_i)} represented in the external RDM basis.`
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// - `mi`: `Fundamental-contraction assignment m_i \in \{0,1\}.`
    /// - `nbas`: Dimension of the external RDM basis.
    /// # Returns
    /// - `ArrayView2<'_, T>`: `External-basis Y^{(m_i)} matrix.`
    #[cfg(feature = "nocc")]
    pub(crate) fn yrdm(
        &self,
        mi: usize,
        nbas: usize,
    ) -> ArrayView2<'_, T> {
        self.w.view2(self.off.yrdm[mi], nbas)
    }

    /// Return the transposed one-column intermediate
    /// `\mathcal F^{(m_i,m_j)} constructed from the one-electron Hamiltonian. The stored`
    /// `[z,r]` ordering makes the replacement column with fixed z contiguous.
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// - `mi, mj`: Fundamental-contraction assignments `m_i,m_j \in \{0,1\}`.
    /// # Returns
    /// - `&[T]`: `Flat transposed \mathcal F^{(m_i,m_j)} matrix.`
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn fh_t_slice(
        &self,
        mi: usize,
        mj: usize,
    ) -> &[T] {
        self.w.slice2(self.off.fh[mi][mj], self.n())
    }

    /// Return the transposed one-column intermediate
    /// `\mathcal F^{(m_i,m_j)} constructed from the current generalised-Fock operator. The`
    /// stored `[z,r]` ordering makes the replacement column with fixed z contiguous.
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// - `mi, mj`: Fundamental-contraction assignments `m_i,m_j \in \{0,1\}`.
    /// # Returns
    /// - `&[T]`: `Flat transposed \mathcal F^{(m_i,m_j)} matrix.`
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn ff_t_slice(
        &self,
        mi: usize,
        mj: usize,
    ) -> &[T] {
        self.w.slice2(self.off.ff[mi][mj], self.n())
    }

    /// Return the transposed same-spin one-column intermediate
    /// `\mathcal V^{(m_1,m_2,m_3)}. Rust storage is ordered as v[m_1][m_3][m_2], and the`
    /// stored `[z,r]` matrix ordering makes the replacement column with fixed z contiguous.
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// - `mi`: `First assignment m_1.`
    /// - `mj`: `Third assignment m_3 in the mathematical ordering.`
    /// - `mk`: `Second assignment m_2 in the mathematical ordering.`
    /// # Returns
    /// - `&[T]`: `Flat transposed \mathcal V^{(m_1,m_2,m_3)} matrix.`
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn v_t_slice(
        &self,
        mi: usize,
        mj: usize,
        mk: usize,
    ) -> &[T] {
        self.w.slice2(self.off.v[mi][mj][mk], self.n())
    }

    /// Return one symmetry-unique same-spin
    /// `\mathcal J^{(m_1,m_2,m_3,m_4)} tensor in stored [i,j,r,c] evaluator order.`
    /// # Arguments:
    /// - `self`: Same-spin Wick view.
    /// - `slot`: `Symmetry-compressed \mathcal J storage slot.`
    /// # Returns
    /// - `&[T]`: `Flat rank-four \mathcal J tensor.`
    #[inline(always)]
    pub(in crate::nonorthogonalwicks) fn j_slice(
        &self,
        slot: usize,
    ) -> &[T] {
        self.w.slice4(self.off.j[slot], self.n())
    }
}

/// Read-only view of the different-spin intermediates for an ordered reference pair.
#[derive(Clone, Copy)]
pub(crate) struct DiffSpinView<'a, T: NOCIScalar> {
    /// Metadata and scalar different-spin intermediates for this reference pair.
    pub(crate) meta: &'a DiffSpinMeta<T>,
    /// Parent view providing access to the contiguous tensor slab.
    w: &'a WicksView<T>,
    /// Offsets locating the different-spin matrices and tensors in the slab.
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
    /// Return the molecular-orbital dimension of each stored different-spin tensor axis.
    /// # Arguments:
    /// - `self`: Different-spin Wick view.
    /// # Returns
    /// - `usize`: Molecular-orbital dimension `nmo`.
    pub(crate) fn n(&self) -> usize {
        self.nmo
    }

    /// Return the transposed alpha-spin one-column intermediate
    /// `\mathcal V^\alpha{}^{(m_{\alpha0},m_{\beta0},m_{\alpha z})} in stored [z,r] order.`
    /// # Arguments:
    /// - `self`: Different-spin Wick view.
    /// - `ma0`: `Assignment m_{\alpha0} of the alpha-spin operator contraction.`
    /// - `mb0`: `Assignment m_{\beta0} of the beta-spin scalar contraction.`
    /// - `mak`: `Assignment m_{\alpha z} of the replaced alpha-spin determinant column.`
    /// # Returns
    /// - `&[T]`: `Flat transposed \mathcal V^\alpha matrix.`
    #[inline(always)]
    pub fn vab_t_slice(
        &self,
        ma0: usize,
        mb0: usize,
        mak: usize,
    ) -> &[T] {
        self.w.slice2(self.off.vab[ma0][mb0][mak], self.n())
    }

    /// Return the transposed beta-spin one-column intermediate
    /// `\mathcal V^\beta{}^{(m_{\beta0},m_{\alpha0},m_{\beta y})} in stored [y,r] order.`
    /// # Arguments:
    /// - `self`: Different-spin Wick view.
    /// - `mb0`: `Assignment m_{\beta0} of the beta-spin operator contraction.`
    /// - `ma0`: `Assignment m_{\alpha0} of the alpha-spin scalar contraction.`
    /// - `mbk`: `Assignment m_{\beta y} of the replaced beta-spin determinant column.`
    /// # Returns
    /// - `&[T]`: `Flat transposed \mathcal V^\beta matrix.`
    #[inline(always)]
    pub fn vba_t_slice(
        &self,
        mb0: usize,
        ma0: usize,
        mbk: usize,
    ) -> &[T] {
        self.w.slice2(self.off.vba[mb0][ma0][mbk], self.n())
    }

    /// Return the different-spin two-column intermediate
    /// `\mathcal{II}^{(m_{\alpha0},m_{\alpha z},m_{\beta0},m_{\beta y})} in stored`
    /// `[r,c,i,j]` order.
    /// # Arguments:
    /// - `self`: Different-spin Wick view.
    /// - `ma0`: `Assignment m_{\alpha0}.`
    /// - `maj`: `Assignment m_{\alpha z}.`
    /// - `mb0`: `Assignment m_{\beta0}.`
    /// - `mbj`: `Assignment m_{\beta y}.`
    /// # Returns
    /// - `&[T]`: `Flat rank-four \mathcal{II} tensor.`
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

/// Grouped alpha-alpha, beta-beta and alpha-beta views for one ordered reference pair (x,w).
#[derive(Clone, Copy)]
pub(crate) struct WicksPairView<'a, T: NOCIScalar> {
    /// Same-spin alpha-alpha intermediates.
    pub(crate) aa: SameSpinView<'a, T>,
    /// Same-spin beta-beta intermediates.
    pub(crate) bb: SameSpinView<'a, T>,
    /// Different-spin alpha-beta intermediates.
    pub(crate) ab: DiffSpinView<'a, T>,
}
