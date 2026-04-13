// nonorthogonalwicks/scratch.rs

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

#[derive(Default)]
pub struct WickScratchSpin {
    pub aa: WickScratch,
    pub bb: WickScratch,
    pub diff: WickScratch,
}

impl WickScratchSpin {
    /// Construct empty split scratch storage for Wick's quantities.
    /// # Arguments:
    /// # Returns
    /// - `WickScratchSpin`: Default-initialised split scratch storage.
    #[inline]
    pub fn new() -> Self {Self::default()}

    /// Pre-allocate split Wick scratch buffers to the largest sizes needed for
    /// the current calculation in order to reduce repeated allocations.
    /// # Arguments:
    /// - `maxsame`: Maximum total excitation rank required for same-spin terms.
    /// - `maxla`: Maximum total alpha-spin excitation rank required for different-spin terms.
    /// - `maxlb`: Maximum total beta-spin excitation rank required for different-spin terms.
    /// # Returns
    /// - `WickScratchSpin`: Split scratch storage with the requested capacities reserved.
    #[inline]
    pub fn with_sizes(maxsame: usize, maxla: usize, maxlb: usize) -> Self {
        Self {
            aa: WickScratch::with_sizes(maxsame, 0, 0),
            bb: WickScratch::with_sizes(maxsame, 0, 0),
            diff: WickScratch::with_sizes(0, maxla, maxlb),
        }
    }
}

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

