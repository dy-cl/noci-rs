// nonorthogonalwicks/scratch.rs
use crate::noci::NOCIScalar;

pub struct Vec1<T> {
    data: Vec<T>,
    len: usize,
}

impl<T> Default for Vec1<T> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
        }
    }
}

impl<T> Vec1<T> {
    /// Get the contents of the scratch vector as a mutable slice.
    /// # Arguments:
    /// - `self`: Scratch vector.
    /// # Returns
    /// - `&mut [T]`: Mutable slice.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }
}

impl<T: Clone + From<f64>> Vec1<T> {
    /// Ensure the storage can hold at least `len` elements.
    /// # Arguments:
    /// - `self`: Scratch vector to resize.
    /// - `len`: Required logical length.
    /// # Returns
    /// - `()`: Grows storage if required and updates length.
    #[inline(always)]
    pub fn ensure(
        &mut self,
        len: usize,
    ) {
        if self.data.len() < len {
            self.data.resize(len, <T as From<f64>>::from(0.0));
        }
        self.len = len;
    }
}

pub struct Vec2<T> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T> Default for Vec2<T> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            nrows: 0,
            ncols: 0,
        }
    }
}

impl<T> Vec2<T> {
    /// Get the contents of the scratch matrix as an immutable slice.
    /// # Arguments:
    /// - `self`: Scratch matrix.
    /// # Returns
    /// - `&[T]`: Immutable slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.nrows * self.ncols]
    }

    /// Get the contents of the scratch matrix as a mutable slice.
    /// # Arguments:
    /// - `self`: Scratch matrix.
    /// # Returns
    /// - `&mut [T]`: Mutable slice.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.nrows * self.ncols;
        &mut self.data[..len]
    }
}

impl<T: Clone + From<f64>> Vec2<T> {
    /// Ensure the storage can hold at least `nrows * ncols` elements and set the shape.
    /// # Arguments:
    /// - `self`: Scratch matrix to resize.
    /// - `nrows`: Required number of rows.
    /// - `ncols`: Required number of columns.
    /// # Returns
    /// - `()`: Grows storage if required and updates the shape.
    #[inline(always)]
    pub fn ensure(
        &mut self,
        nrows: usize,
        ncols: usize,
    ) {
        let need = nrows * ncols;
        if self.data.len() < need {
            self.data.resize(need, <T as From<f64>>::from(0.0));
        }
        self.nrows = nrows;
        self.ncols = ncols;
    }
}

pub struct WickScratchSpin<T: NOCIScalar> {
    pub aa: WickScratch<T>,
    pub bb: WickScratch<T>,
    pub diff: WickScratch<T>,
}

impl<T: NOCIScalar> WickScratchSpin<T> {
    /// Construct empty split scratch storage for Wick's quantities.
    /// # Arguments:
    /// # Returns
    /// - `WickScratchSpin<T>`: Default-initialised split scratch storage.
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

    /// Pre-allocate split Wick scratch buffers to the largest sizes needed for
    /// the current calculation in order to reduce repeated allocations.
    /// # Arguments:
    /// - `maxsame`: Maximum total excitation rank required for same-spin terms.
    /// - `maxla`: Maximum total alpha-spin excitation rank required for different-spin terms.
    /// - `maxlb`: Maximum total beta-spin excitation rank required for different-spin terms.
    /// # Returns
    /// - `WickScratchSpin<T>`: Split scratch storage with the requested capacities reserved.
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

pub struct WickScratch<T: NOCIScalar> {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,

    pub det0: Vec2<T>,
    pub det1: Vec2<T>,
    pub det_mix: Vec2<T>,

    pub fcol: Vec1<T>,
    pub dv: Vec1<T>,
    pub v1: Vec1<T>,
    pub dv1: Vec1<T>,
    pub dv1m: Vec1<T>,

    pub jslice_full: Vec2<T>,
    pub jslice2: Vec2<T>,
    pub det_mix2: Vec2<T>,

    pub deta0: Vec2<T>,
    pub deta1: Vec2<T>,
    pub deta_mix: Vec2<T>,
    pub detb0: Vec2<T>,
    pub detb1: Vec2<T>,
    pub detb_mix: Vec2<T>,

    pub v1a: Vec1<T>,
    pub v1b: Vec1<T>,
    pub dv1a: Vec1<T>,
    pub dv1b: Vec1<T>,

    pub iislicea: Vec2<T>,
    pub iisliceb: Vec2<T>,
    pub deta_mix_minor: Vec2<T>,
    pub detb_mix_minor: Vec2<T>,

    pub adjt_det: Vec2<T>,
    pub adjt_deta: Vec2<T>,
    pub adjt_detb: Vec2<T>,
    pub adjt_det2: Vec2<T>,
    pub adjt_deta_mix_minor: Vec2<T>,
    pub adjt_detb_mix_minor: Vec2<T>,

    pub invs: Vec1<f64>,
    pub invsla: Vec1<f64>,
    pub invslb: Vec1<f64>,
    pub invslm1: Vec1<f64>,
    pub invslam1: Vec1<f64>,
    pub invslbm1: Vec1<f64>,

    pub lu: Vec2<T>,
    pub lua: Vec2<T>,
    pub lub: Vec2<T>,
}

impl<T: NOCIScalar> Default for WickScratch<T> {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
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
    /// Pre-allocate the Wick' scratch buffers to the largest sizes needed for the
    /// current calculation in order to reduce repeated allocations.
    /// # Arguments:
    /// - `maxsame`: Maximum total excitation rank required for same-spin terms.
    /// - `maxla`: Maximum total alpha-spin excitation rank required for different-spin terms.
    /// - `maxlb`: Maximum total beta-spin excitation rank required for different-spin terms.
    /// # Returns
    /// - `WickScratch<T>`: Scratch storage with the requested capacities reserved.
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

    /// If the previously allocated size of the scratch space is the not the same in
    /// the same spin case resize all the scratch space quantities to be correct.
    /// # Arguments:
    /// - `self`: Scratch space for Wick's quantities.
    /// - `l`: Excitation rank.
    /// # Returns
    /// - `()`: Resizes same-spin scratch storage in place.
    #[inline(always)]
    pub fn ensure_same(
        &mut self,
        l: usize,
    ) {
        self.rows.clear();
        self.cols.clear();

        if self.rows.capacity() < l {
            self.rows.reserve_exact(l - self.rows.capacity());
        }
        if self.cols.capacity() < l {
            self.cols.reserve_exact(l - self.cols.capacity());
        }

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
    pub fn ensure_diff(
        &mut self,
        la: usize,
        lb: usize,
    ) {
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
