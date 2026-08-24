// nonorthogonalwicks/scratch.rs
// Standard library imports.
use std::ops::{Deref, DerefMut};

// Crate-root imports.
use crate::noci::NOCIScalar;

/// Reusable index storage with a logical length independent of the retained allocation.
#[derive(Default)]
pub struct IndexVec {
    /// Retained backing storage for the indices.
    data: Vec<usize>,
    /// Number of indices currently in use.
    len: usize,
}

impl IndexVec {
    /// Return the active indices as an immutable slice.
    /// # Arguments:
    /// - `self`: Reusable index storage.
    /// # Returns
    /// - `&[usize]`: Immutable slice containing the active indices.
    #[inline(always)]
    pub fn as_slice(&self) -> &[usize] {
        &self.data[..self.len]
    }

    /// Return the active indices as a mutable slice.
    /// # Arguments:
    /// - `self`: Reusable index storage.
    /// # Returns
    /// - `&mut [usize]`: Mutable slice containing the active indices.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        &mut self.data[..self.len]
    }

    /// Ensure that the backing allocation can hold `len` indices and set the logical length.
    /// Existing allocation is retained when it is already large enough.
    /// # Arguments:
    /// - `self`: Reusable index storage to resize.
    /// - `len`: Required logical length.
    /// # Returns
    /// - `()`: Updates the active length and grows the backing storage when required.
    #[inline(always)]
    pub fn ensure(
        &mut self,
        len: usize,
    ) {
        if self.len == len {
            return;
        }
        if self.data.len() < len {
            self.data.resize(len, 0);
        }
        self.len = len;
    }
}

impl Deref for IndexVec {
    type Target = [usize];

    /// Return the active indices as an immutable slice.
    /// # Returns
    /// - `&[usize]`: Immutable slice containing the active indices.
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for IndexVec {
    /// Return the active indices as a mutable slice.
    /// # Returns
    /// - `&mut [usize]`: Mutable slice containing the active indices.
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Reusable one-dimensional work storage with a logical length independent of the retained allocation.
pub struct Vec1<T> {
    /// Retained backing storage for the work values.
    data: Vec<T>,
    /// Number of values currently in use.
    len: usize,
}

impl<T> Default for Vec1<T> {
    /// Construct empty reusable one-dimensional work storage.
    /// # Returns
    /// - `Self`: Empty storage with zero logical length.
    fn default() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
        }
    }
}

impl<T> Vec1<T> {
    /// Return the active work values as a mutable slice.
    /// # Arguments:
    /// - `self`: Reusable work vector.
    /// # Returns
    /// - `&mut [T]`: Mutable slice containing the active values.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }
}

impl<T: Clone + From<f64>> Vec1<T> {
    /// Ensure that the backing allocation can hold `len` values and set the logical length.
    /// Newly allocated entries are initialised to zero.
    /// # Arguments:
    /// - `self`: Reusable work vector to resize.
    /// - `len`: Required logical length.
    /// # Returns
    /// - `()`: Updates the active length and grows the backing storage when required.
    #[inline(always)]
    pub fn ensure(
        &mut self,
        len: usize,
    ) {
        if self.len == len {
            return;
        }
        if self.data.len() < len {
            self.data.resize(len, <T as From<f64>>::from(0.0));
        }
        self.len = len;
    }
}

/// Reusable row-major matrix work storage with a logical shape independent of the retained allocation.
pub struct Vec2<T> {
    /// Retained backing storage for the matrix entries.
    data: Vec<T>,
    /// Logical number of rows.
    nrows: usize,
    /// Logical number of columns.
    ncols: usize,
}

impl<T> Default for Vec2<T> {
    /// Construct empty reusable matrix work storage.
    /// # Returns
    /// - `Self`: `Empty storage with shape 0 \times 0.`
    fn default() -> Self {
        Self {
            data: Vec::new(),
            nrows: 0,
            ncols: 0,
        }
    }
}

impl<T> Vec2<T> {
    /// Return the active row-major matrix entries as an immutable slice.
    /// # Arguments:
    /// - `self`: Reusable matrix work storage.
    /// # Returns
    /// - `&[T]`: Immutable slice containing `nrows * ncols` active entries.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.nrows * self.ncols]
    }

    /// Return the active row-major matrix entries as a mutable slice.
    /// # Arguments:
    /// - `self`: Reusable matrix work storage.
    /// # Returns
    /// - `&mut [T]`: Mutable slice containing `nrows * ncols` active entries.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.nrows * self.ncols;
        &mut self.data[..len]
    }
}

impl<T: Clone + From<f64>> Vec2<T> {
    /// Ensure that the backing allocation can hold an `nrows` by `ncols` matrix and set the
    /// logical shape. Newly allocated entries are initialised to zero.
    /// # Arguments:
    /// - `self`: Reusable matrix work storage to resize.
    /// - `nrows`: Required number of rows.
    /// - `ncols`: Required number of columns.
    /// # Returns
    /// - `()`: Updates the logical shape and grows the backing storage when required.
    #[inline(always)]
    pub fn ensure(
        &mut self,
        nrows: usize,
        ncols: usize,
    ) {
        if self.nrows == nrows && self.ncols == ncols {
            return;
        }
        let need = nrows * ncols;
        if self.data.len() < need {
            self.data.resize(need, <T as From<f64>>::from(0.0));
        }
        self.nrows = nrows;
        self.ncols = ncols;
    }
}

/// Independent reusable workspaces for the alpha-alpha, beta-beta and different-spin evaluators.
pub struct WickScratchSpin<T: NOCIScalar> {
    /// Same-spin alpha-alpha evaluator workspace.
    pub aa: WickScratch<T>,
    /// Same-spin beta-beta evaluator workspace.
    pub bb: WickScratch<T>,
    /// Different-spin alpha-beta evaluator workspace.
    pub diff: WickScratch<T>,
}

impl<T: NOCIScalar> WickScratchSpin<T> {
    /// Construct empty split Wick work storage.
    /// # Returns
    /// - `WickScratchSpin<T>`: Default-initialised spin-resolved workspaces.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    fn default() -> Self {
        Self {
            aa: WickScratch::default(),
            bb: WickScratch::default(),
            diff: WickScratch::default(),
        }
    }

    /// Preallocate the spin-resolved workspaces to the largest contraction-determinant
    /// dimensions required by the current calculation.
    /// # Arguments:
    /// - `maxsame`: Maximum same-spin contraction-determinant dimension L.
    /// - `maxla`: `Maximum alpha-spin dimension L_\alpha for different-spin terms.`
    /// - `maxlb`: `Maximum beta-spin dimension L_\beta for different-spin terms.`
    /// # Returns
    /// - `WickScratchSpin<T>`: Split work storage with the requested capacities retained.
    #[inline]
    pub fn with_sizes(
        maxsame: usize,
        maxla: usize,
        maxlb: usize,
    ) -> Self {
        Self {
            aa: WickScratch::with_sizes(maxsame, 0, 0),
            bb: WickScratch::with_sizes(maxsame, 0, 0),
            diff: WickScratch::with_sizes(0, maxla, maxlb),
        }
    }
}

/// Reusable determinant, cofactor and numerical-factorisation storage used while evaluating
/// nonorthogonal Wick matrix elements. The same object can be resized and reused across
/// determinant pairs and excitation ranks without repeated allocation.
pub struct WickScratch<T: NOCIScalar> {
    /// `Compact row indices in V_x \cup O_w, ordered as the bra excitation particles`
    /// followed by the ket excitation holes.
    pub rows: IndexVec,
    /// `Compact column indices in O_x \cup V_w, ordered as the bra excitation holes`
    /// followed by the ket excitation particles.
    pub cols: IndexVec,
    /// Same-spin contraction-determinant dimension L for which the full workspace is prepared.
    same_rank: Option<usize>,
    /// `Different-spin dimensions (L_\alpha,L_\beta) for which the workspace is prepared.`
    diff_rank: Option<(usize, usize)>,

    /// `Endpoint contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).`
    pub det0: Vec2<T>,
    /// `Endpoint contraction determinant \mathbf D_{\mathrm{ov}}(1,\ldots,1).`
    pub det1: Vec2<T>,
    /// `Mixed contraction determinant \mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L).`
    pub det_mix: Vec2<T>,

    // Retained work vectors that are not read by the current evaluator and helper implementations.
    // They remain part of the workspace layout and are still resized by `ensure_same`.
    pub fcol: Vec1<T>,
    pub dv: Vec1<T>,
    pub v1: Vec1<T>,
    pub dv1: Vec1<T>,
    pub dv1m: Vec1<T>,

    // Retained matrix buffers that are not read by the current evaluator and helper implementations.
    // `det_mix2` remains active as the minor \mathbf D_{\mathrm{ov}}[\eta|z].
    pub jslice_full: Vec2<T>,
    pub jslice2: Vec2<T>,
    /// `Minor contraction determinant \mathbf D_{\mathrm{ov}}[\eta|z] used in the`
    /// `two-column same-spin \mathcal J contribution.`
    pub det_mix2: Vec2<T>,

    // Retained endpoint buffers that are not read by the current different-spin evaluator.
    pub deta0: Vec2<T>,
    pub deta1: Vec2<T>,
    /// Mixed alpha-spin contraction determinant
    /// `\mathbf D_{\alpha,\mathrm{ov}}(m_{\alpha1},\ldots,m_{\alpha L_\alpha}).`
    pub deta_mix: Vec2<T>,
    pub detb0: Vec2<T>,
    pub detb1: Vec2<T>,
    /// Mixed beta-spin contraction determinant
    /// `\mathbf D_{\beta,\mathrm{ov}}(m_{\beta1},\ldots,m_{\beta L_\beta}).`
    pub detb_mix: Vec2<T>,

    // Retained vectors that are not read by the current different-spin evaluator.
    pub v1a: Vec1<T>,
    pub v1b: Vec1<T>,
    pub dv1a: Vec1<T>,
    pub dv1b: Vec1<T>,

    // Retained matrices that are not read by the current different-spin evaluator.
    pub iislicea: Vec2<T>,
    pub iisliceb: Vec2<T>,
    pub deta_mix_minor: Vec2<T>,
    pub detb_mix_minor: Vec2<T>,

    /// `Cofactor matrix \operatorname{cof}[\mathbf D_{\mathrm{ov}}].`
    pub adjt_det: Vec2<T>,
    /// `Cofactor matrix \operatorname{cof}[\mathbf D_{\alpha,\mathrm{ov}}].`
    pub adjt_deta: Vec2<T>,
    /// `Cofactor matrix \operatorname{cof}[\mathbf D_{\beta,\mathrm{ov}}].`
    pub adjt_detb: Vec2<T>,
    /// `Cofactor matrix \operatorname{cof}[\mathbf D_{\mathrm{ov}}[\eta|z]].`
    pub adjt_det2: Vec2<T>,
    // Retained cofactor buffers for the unused spin-resolved minor determinants.
    pub adjt_deta_mix_minor: Vec2<T>,
    pub adjt_detb_mix_minor: Vec2<T>,

    /// Inverse-singular-value workspace used by the SVD fallback for
    /// `\mathbf D_{\mathrm{ov}}.`
    pub invs: Vec1<f64>,
    /// Inverse-singular-value workspace used by the SVD fallback for
    /// `\mathbf D_{\alpha,\mathrm{ov}}.`
    pub invsla: Vec1<f64>,
    /// Inverse-singular-value workspace used by the SVD fallback for
    /// `\mathbf D_{\beta,\mathrm{ov}}.`
    pub invslb: Vec1<f64>,
    // Retained inverse-singular-value workspaces for the unused minor kernels.
    pub invslm1: Vec1<f64>,
    pub invslam1: Vec1<f64>,
    pub invslbm1: Vec1<f64>,

    /// `LU-factorisation workspace used for \mathbf D_{\mathrm{ov}}.`
    pub lu: Vec2<T>,
    /// `LU-factorisation workspace used for \mathbf D_{\alpha,\mathrm{ov}}.`
    pub lua: Vec2<T>,
    /// `LU-factorisation workspace used for \mathbf D_{\beta,\mathrm{ov}}.`
    pub lub: Vec2<T>,
}

impl<T: NOCIScalar> Default for WickScratch<T> {
    fn default() -> Self {
        Self {
            rows: IndexVec::default(),
            cols: IndexVec::default(),
            same_rank: None,
            diff_rank: None,
            det0: Vec2::default(),
            det1: Vec2::default(),
            det_mix: Vec2::default(),
            fcol: Vec1::default(),
            dv: Vec1::default(),
            v1: Vec1::default(),
            dv1: Vec1::default(),
            dv1m: Vec1::default(),
            jslice_full: Vec2::default(),
            jslice2: Vec2::default(),
            det_mix2: Vec2::default(),
            deta0: Vec2::default(),
            deta1: Vec2::default(),
            deta_mix: Vec2::default(),
            detb0: Vec2::default(),
            detb1: Vec2::default(),
            detb_mix: Vec2::default(),
            v1a: Vec1::default(),
            v1b: Vec1::default(),
            dv1a: Vec1::default(),
            dv1b: Vec1::default(),
            iislicea: Vec2::default(),
            iisliceb: Vec2::default(),
            deta_mix_minor: Vec2::default(),
            detb_mix_minor: Vec2::default(),
            adjt_det: Vec2::default(),
            adjt_deta: Vec2::default(),
            adjt_detb: Vec2::default(),
            adjt_det2: Vec2::default(),
            adjt_deta_mix_minor: Vec2::default(),
            adjt_detb_mix_minor: Vec2::default(),
            invs: Vec1::default(),
            invsla: Vec1::default(),
            invslb: Vec1::default(),
            invslm1: Vec1::default(),
            invslam1: Vec1::default(),
            invslbm1: Vec1::default(),
            lu: Vec2::default(),
            lua: Vec2::default(),
            lub: Vec2::default(),
        }
    }
}

impl<T: NOCIScalar> WickScratch<T> {
    /// Preallocate the reusable workspace to the largest contraction-determinant dimensions
    /// required by the current calculation.
    /// # Arguments:
    /// - `maxsame`: Maximum same-spin contraction-determinant dimension L.
    /// - `maxla`: `Maximum alpha-spin dimension L_\alpha for different-spin terms.`
    /// - `maxlb`: `Maximum beta-spin dimension L_\beta for different-spin terms.`
    /// # Returns
    /// - `WickScratch<T>`: Workspace with the requested capacities retained.
    #[inline]
    pub fn with_sizes(
        maxsame: usize,
        maxla: usize,
        maxlb: usize,
    ) -> Self {
        let mut s = Self::default();
        s.ensure_same(maxsame);
        s.ensure_diff(maxla, maxlb);
        s
    }

    /// Resize the same-spin workspace for a contraction determinant of dimension L. This
    /// prepares both endpoint determinants, a mixed determinant, its cofactor matrix and the
    /// `(L-1) \times (L-1) minor required by the two-column \mathcal J contribution.`
    /// # Arguments:
    /// - `self`: Reusable Wick workspace.
    /// - `l`: Same-spin contraction-determinant dimension L.
    /// # Returns
    /// - `()`: Resizes the active and retained same-spin buffers in place.
    #[inline(always)]
    pub fn ensure_same(
        &mut self,
        l: usize,
    ) {
        if self.same_rank == Some(l) {
            return;
        }
        self.same_rank = Some(l);
        self.rows.ensure(l);
        self.cols.ensure(l);

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

    /// Resize the different-spin workspace for independent alpha- and beta-spin contraction
    /// `determinants of dimensions L_\alpha and L_\beta. This prepares the mixed determinants,`
    /// cofactor matrices and determinant-factorisation workspaces used by the factorised
    /// different-spin two-body evaluator.
    /// # Arguments:
    /// - `self`: Reusable Wick workspace.
    /// - `la`: `Alpha-spin contraction-determinant dimension L_\alpha.`
    /// - `lb`: `Beta-spin contraction-determinant dimension L_\beta.`
    /// # Returns
    /// - `()`: Resizes the active and retained different-spin buffers in place.
    #[inline(always)]
    pub fn ensure_diff(
        &mut self,
        la: usize,
        lb: usize,
    ) {
        if self.diff_rank == Some((la, lb)) {
            return;
        }
        self.diff_rank = Some((la, lb));

        self.deta0.ensure(la, la);
        self.deta1.ensure(la, la);
        self.deta_mix.ensure(la, la);
        self.adjt_deta.ensure(la, la);
        self.v1a.ensure(la);
        self.dv1a.ensure(la);
        self.iislicea.ensure(la, la);
        self.invsla.ensure(la);
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
        self.lub.ensure(6, 6);

        let lbm1 = lb.saturating_sub(1);
        self.detb_mix_minor.ensure(lbm1, lbm1);
        self.adjt_detb_mix_minor.ensure(lbm1, lbm1);
        self.invslbm1.ensure(lbm1);
    }
}
