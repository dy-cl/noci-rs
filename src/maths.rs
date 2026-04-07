/// maths.rs
use std::{cell::RefCell};

use ndarray::{Array1, Array2, ArrayView2, Array4, Axis};
use ndarray_linalg::{Eigh, UPLO};
use rayon::prelude::*;

thread_local! {static HT_SCRATCH: RefCell<Vec<f64>> = const {RefCell::new(Vec::new())}; static GT_SCRATCH: RefCell<Vec<f64>> = const {RefCell::new(Vec::new())};}

/// Return positive eigenvalue subspace of symmetric matrix S. 
/// # Arguments:
/// - `s`: Symmetric matrix.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns
/// - `(Array1<f64>, Array2<f64>)`: Positive eigenvalues and their associated eigenvectors.
pub fn positive_subspace_real(s: &Array2<f64>, tol: f64) -> (Array1<f64>, Array2<f64>) {
    let (lambdas, evecs) = s.eigh(UPLO::Lower).unwrap();
    // Filter out eigenvalues smaller than tol.
    let pos: Vec<usize> = lambdas.iter().enumerate().filter_map(|(i, &x)| if x > tol {Some(i)} else {None}).collect();
    let poslambdas = Array1::from_iter(pos.iter().map(|&i| lambdas[i]));
    
    // Filter out eigenvectors associated with negative eigenvalues.
    let mut posvecs = Array2::<f64>::zeros((s.nrows(), pos.len()));
    for (j, &i) in pos.iter().enumerate() {
        posvecs.column_mut(j).assign(&evecs.column(i));
    }

    (poslambdas, posvecs)

}

/// Loewdin symmetric orthogonalizer, computes X = S^{-1/2} and returns X = U \Lambda^{-1/2} U^T 
/// for real matrices.
/// # Arguments 
/// - `s`: Symmetric matrix, uses only the lower triangle. 
/// - `project`: Whether or not to project to non-zero positive subspace of S.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns
/// - `Array2<f64>`: Loewdin symmetric orthogonalizer.
pub fn loewdin_x_real(s: &Array2<f64>, project: bool, tol: f64) -> Array2<f64> {
    if project {
        let (vals, vecs) = positive_subspace_real(s, tol);
        let d = Array2::from_diag(&vals.mapv(|x| 1.0 / x.sqrt()));
        vecs.dot(&d).dot(&vecs.t())
    } else {
        let (lambdas, evecs) = s.eigh(UPLO::Lower).unwrap();
        let d = Array2::from_diag(&lambdas.mapv(|x| 1.0 / x.sqrt()));
        evecs.dot(&d).dot(&evecs.t())
    }
}

/// Rectangular orthogonalizer, computes X = U_+ \Lambda_+^{-1 / 2}.
/// # Arguments:
/// `s`: matrix to orthogonalize.
/// - `tol`: Tolerance for whether a number is considered zero. 
/// # Returns
/// - `Array2<f64>`: Rectangular orthogonalizer for the positive subspace of `s`.
pub fn orthogonaliser_real(s: &Array2<f64>, tol: f64) -> Array2<f64> {
    let (vals, vecs) = positive_subspace_real(s, tol);
    let d = Array2::from_diag(&vals.mapv(|x| 1.0 / x.sqrt()));
    vecs.dot(&d)
}

/// Solve the real generalized eigenproblem F C = S C e using the Loewdin orthogonalizer.
/// # Arguments
/// - `f`: Hermitian matrix, uses only the lower triangle. Often Fock matrix 
///   is f here.
/// - `s`: Hermitian matrix, uses only the lower triangle. Often AO basis overlap 
///   matrix is s here.
/// - `project`: Whether or not to project to non-zero positive subspace of S.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns
/// - `(Array1<f64>, Array2<f64>)`: Eigenvalues and eigenvectors of the generalized eigenproblem.
pub fn general_evp_real(f: &Array2<f64>, s: &Array2<f64>, project: bool, tol: f64) 
                        -> (Array1<f64>, Array2<f64>) {
    // X = S^{-1/2}
    let x = if project {
        orthogonaliser_real(s, tol)
    } else {
        loewdin_x_real(s, false, tol)
    };
    // \tilde{F} = X^T F X.
    let ft = x.t().dot(f).dot(&x);
    // \tilde{F} U = \epsilon U.
    let (epsilon, u) = ft.eigh(UPLO::Lower).unwrap();
    // C = X U. 
    let c = x.dot(&u);
    (epsilon, c)
}

/// Calculate Einstein summation of matrices `g` and `h` as \sum_{a,b} g_{b,a} h_{ab}. 
/// Assumes `g` and `h` are of identical shape.
/// # Arguments 
/// - `g`: Matrix 1. 
/// - `h`: Matrix 2.
/// # Returns
/// - `f64`: Contracted scalar.
pub fn einsum_ba_ab_real(g: &Array2<f64>, h: &Array2<f64>) -> f64 {
let n = g.nrows();
    
    // Convert ndarrays into memory ordered slice.
    let gs = g.as_slice_memory_order().unwrap();
    let hs = h.as_slice_memory_order().unwrap();

    let mut acc = 0.0;
    
    // Index of 2D tensor element in 1D is given by (a * n) + b.
    for a in 0..n {
        for b in 0..n {
            // g[b,a]. Use of get_unchecked means no out of bounds checking is performed. 
            // If index i is invalid this produces undefined behaviour rather than a panic, check this when debugging. 
            // Use of unsafe is consequently required as get_unchecked is an unsafe operation.
            let g_ba = unsafe {*gs.get_unchecked(b * n + a)};
            // h[a,b].
            let h_ab = unsafe {*hs.get_unchecked(a * n + b)};
            acc += g_ba * h_ab; 
        }
    }
    acc
}

/// Perform dot product between two vectors with unrolled loop of length 8.
/// # Arguments:
/// - `x`: Vector 1.
/// - `y`: Vector 2.
/// - `n`: Vector length.
/// # Returns
/// - `f64`: Dot product of the two vectors.
#[inline(always)]
fn dot_product_unroll8(mut x: *const f64, mut y: *const f64, n: usize) -> f64 {
    let mut i = 0usize;

    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    
    unsafe {
        // Accumulate contributions to dot product sum 8 at a time.
        while i + 8 <= n {
            let x0 = *x;       
            let y0 = *y;
            let x1 = *x.add(1); 
            let y1 = *y.add(1);
            let x2 = *x.add(2); 
            let y2 = *y.add(2);
            let x3 = *x.add(3); 
            let y3 = *y.add(3);
            let x4 = *x.add(4); 
            let y4 = *y.add(4);
            let x5 = *x.add(5); 
            let y5 = *y.add(5);
            let x6 = *x.add(6); 
            let y6 = *y.add(6);
            let x7 = *x.add(7); 
            let y7 = *y.add(7);

            s0 = x0.mul_add(y0, s0);
            s1 = x1.mul_add(y1, s1);
            s2 = x2.mul_add(y2, s2);
            s3 = x3.mul_add(y3, s3);
            s0 = x4.mul_add(y4, s0);
            s1 = x5.mul_add(y5, s1);
            s2 = x6.mul_add(y6, s2);
            s3 = x7.mul_add(y7, s3);
            
            x = x.add(8);
            y = y.add(8);
            i += 8;
        }

        let mut sum = s0 + s1 + s2 + s3;
        
        // Accumulate less than 8 remaining contributions.
        while i < n {
            sum = (*x).mul_add(*y, sum);
            x = x.add(1);
            y = y.add(1);
            i += 1;
        }
        sum
    }
}

/// Calculate Einstein summation of matrices `g` and `h` and 4D tensor `t` as 
/// \sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c, d}. Assumes `g`, `h` and `t` all 
/// have axes of equal length.
/// # Arguments
/// - `g`: Matrix 1. 
/// - `t`: 4D tensor.
/// - `h`: Matrix 2.
/// # Returns
/// - `f64`: Contracted scalar.
pub fn einsum_ba_abcd_cd_real(g: &Array2<f64>, t: &Array4<f64>, h: &Array2<f64>) -> f64 {
    let n = g.nrows();

    // Convert ndarrays into memory ordered slice.
    let gs = g.as_slice_memory_order().unwrap();
    let hs = h.as_slice_memory_order().unwrap();
    let ts = t.as_slice_memory_order().unwrap();

    let mut acc = 0.0f64;
    
    // Reuse ht and gt across calls to this function.
    HT_SCRATCH.with(|hbuf| {
        GT_SCRATCH.with(|gbuf| {
            // Transpose g[a, b] = gt[b, a] into contiguous in fastest index layouts.
            let mut ht = hbuf.borrow_mut();
            let mut gt = gbuf.borrow_mut();
            
            ht.resize(n * n, 0.0);
            gt.resize(n * n, 0.0);

            for d in 0..n {
                for c in 0..n {
                    // Use of get_unchecked means no out of bounds checking is performed. If index i is invalid 
                    // this produces undefined behaviour rather than a panic, check this when debugging. 
                    // Use of unsafe is consequently required as get_unchecked is an unsafe operation.
                    unsafe { *ht.get_unchecked_mut(c * n + d) = *hs.get_unchecked(c * n + d); }
                }
            }
            for b in 0..n {
                let b_idx = b * n;
                for a in 0..n {
                    // Use of get_unchecked means no out of bounds checking is performed. If index i is invalid 
                    // this produces undefined behaviour rather than a panic, check this when debugging. 
                    // Use of unsafe is consequently required as get_unchecked is an unsafe operation.
                    unsafe{*gt.get_unchecked_mut(a * n + b) = *gs.get_unchecked(b_idx + a);}
                }
            }

            unsafe {
                let ts_ptr = ts.as_ptr();
                let ht_ptr = ht.as_ptr();
                let gt_ptr = gt.as_ptr();
                // Index of 4D tensor element in 1D is given by (((a * n + b) * n + c) * n + d). So d varies
                // fastes, then b, then c, then a. Therefore iteration should be in order, a, c, b, d.
                // So iterate (a,b,c,d) for contiguous access.
                for a in 0..n {
                    // For a given a, the block t[a, :, :, :] starts at a * n^3. See above element indexing.
                    let ta_index = a * n * n * n;
                    // gt[b, a] (g[a, b])
                    let gt_a_ptr = gt_ptr.add(a * n);
                    for b in 0..n {
                        let g_ba_ptr = *gt_a_ptr.add(b);
                        if g_ba_ptr == 0.0 {continue;}
                        // For a given a, b, the block t[a, b, :, :] starts at a*n^3 + b*n^2. See above element indexing.
                        let tab_index = ta_index + b * n * n;
                        for c in 0..n {
                            // ht[c, d] (h[d, c]).
                            let ht_c_ptr = ht_ptr.add(c * n);
                            // For a given a, b, c, the block t[a, b, c, :] starts at a*n^3 + b*n^2 + c*n. See above element indexing.
                            // Get the vector t[a, b, c, :].
                            let tabc_vec_ptr = ts_ptr.add(tab_index + c * n);
                            
                            // Compute dot product of t[a, b, c, :] with ht[:, c] with unrolling and
                            // accumulate g_{b,a} t_{a,b,c,d} h_{d,c}. 
                            let dot = dot_product_unroll8(tabc_vec_ptr, ht_c_ptr, n);
                            acc = g_ba_ptr.mul_add(dot, acc);
                        }
                    }
                }
            }
            acc
        })
    })
}

/// Calculate a matrix vector product HC = U in parallel.
/// # Arguments  
/// - `h`: Matrix. 
/// - `c`: Vector.
/// # Returns
/// - `Array1<f64>`: Matrix-vector product.
pub fn parallel_matvec_real(h: &Array2<f64>, c: &Array1<f64>) -> Array1<f64> {
    // Iterate over rows of the matrix h in parallel, for each computing its dot product with
    // vector c, and finally collect the results into a vector.
    let result: Vec<f64> = h.axis_iter(Axis(0)).into_par_iter().map(|row| {row.iter().zip(c.iter()).map(|(&hij, &cj)| hij * cj).sum::<f64>()}).collect();
    Array1::from_vec(result)
}

/// Transform ERIs from AO to MO basis as:
///     (pq|rs) = \sum_{\mu\nu\lambda\sigma} (\mu\nu|\lambda\sigma) C_{\mu, p} C_{\nu, q} C_{\lambda, r} C_{\sigma, s}.
/// Contraction is performed one indexd at a time for O(n^5) work.
/// # Arguments:
/// - `eri`: AO basis ERIs.
/// - `c_mu_p`: MO coefficients C_{\mu, p}. 
/// - `c_nu_q`: MO coefficients C_{\nu, q}.
/// - `c_lam_r`: MO coefficients C_{\lambda, r}.
/// - `c_sigma_s`: MO coefficients C_{\sigma, s}.
/// # Returns
/// - `Array4<f64>`: ERIs transformed to the MO basis.
pub fn eri_ao2mo(eri: &Array4<f64>, c_mu_p: &Array2<f64>, c_nu_q: &Array2<f64>, c_lam_r: &Array2<f64>, c_sig_s: &Array2<f64>) -> Array4<f64> {
    let nbas = c_mu_p.nrows();
    let nmo_p = c_mu_p.ncols();
    let nmo_q = c_nu_q.ncols();
    let nmo_r = c_lam_r.ncols();
    let nmo_s = c_sig_s.ncols();

    let mut t1 = Array4::<f64>::zeros((nbas, nbas, nbas, nmo_s));
    for mu in 0..nbas {
        for nu in 0..nbas {
            for lam in 0..nbas {
                for s in 0..nmo_s {
                    let mut acc = 0.0;
                    for sig in 0..nbas {
                        acc += eri[(mu, nu, lam, sig)] * c_sig_s[(sig, s)];
                    }
                    t1[(mu, nu, lam, s)] = acc;
                }
            }
        }
    }

    let mut t2 = Array4::<f64>::zeros((nbas, nbas, nmo_r, nmo_s));
    for mu in 0..nbas {
        for nu in 0..nbas {
            for r in 0..nmo_r {
                for s in 0..nmo_s {
                    let mut acc = 0.0;
                    for lam in 0..nbas {
                        acc += t1[(mu, nu, lam, s)] * c_lam_r[(lam, r)];
                    }
                    t2[(mu, nu, r, s)] = acc;
                }
            }
        }
    }

    let mut t3 = Array4::<f64>::zeros((nbas, nmo_q, nmo_r, nmo_s));
    for mu in 0..nbas {
        for q in 0..nmo_q {
            for r in 0..nmo_r {
                for s in 0..nmo_s {
                    let mut acc = 0.0;
                    for nu in 0..nbas {
                        acc += t2[(mu, nu, r, s)] * c_nu_q[(nu, q)];
                    }
                    t3[(mu, q, r, s)] = acc;
                }
            }
        }
    }

    let mut out = Array4::<f64>::zeros((nmo_p, nmo_q, nmo_r, nmo_s));
    for p in 0..nmo_p {
        for q in 0..nmo_q {
            for r in 0..nmo_r {
                for s in 0..nmo_s {
                    let mut acc = 0.0;
                    for mu in 0..nbas {
                        acc += t3[(mu, q, r, s)] * c_mu_p[(mu, p)];
                    }
                    out[(p, q, r, s)] = acc;
                }
            }
        }
    }
    out
}

/// Build the square `l x l` contraction determinant with `x` elements in the
/// diagonal and lower triangle, and `y` elements in the upper triangle.
/// Dispatches to specialised implementations for small excitation ranks.
/// # Arguments:
/// - `d`: Matrix to write into.
/// - `l`: Excitation rank.
/// - `x`: Matrix supplying diagonal and lower-triangular elements.
/// - `y`: Matrix supplying upper-triangular elements.
/// - `rows`: Row indices into `x` and `y`.
/// - `cols`: Column indices into `x` and `y`.
/// # Returns
/// - `()`: Writes the contraction determinant into `d`.
#[inline(always)]
pub fn build_d(d: &mut [f64], l: usize, x: &ArrayView2<f64>, y: &ArrayView2<f64>, rows: &[usize], cols: &[usize]) {
    match l {
        0 => {}
        1 => build_d::build_d1(d, x, y, rows, cols),
        2 => build_d::build_d2(d, x, y, rows, cols),
        3 => build_d::build_d3(d, x, y, rows, cols),
        4 => build_d::build_d4(d, x, y, rows, cols),
        _ => build_d::build_d_gen(d, l, x, y, rows, cols),
    }
}

mod build_d {
    use ndarray::{ArrayView2};

    /// Construct contraction determinant for excitation rank 1.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `x`: X matrix elements.
    /// - `y`: Y matrix elements.
    /// - `rows`: Row indices of X or Y.
    /// - `cols`: Column indices of X or Y.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn build_d1(d: &mut [f64], x: &ArrayView2<f64>, _y: &ArrayView2<f64>, rows: &[usize], cols: &[usize]) {
        let xstr = x.strides();
        let xptr = x.as_ptr();

        unsafe {
            let r0 = *rows.get_unchecked(0) as isize;
            let c0 = *cols.get_unchecked(0) as isize;

            d[0] = *xptr.offset(r0 * xstr[0] + c0 * xstr[1]);
        }
    }

    /// Construct contraction determinant for excitation rank 2.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `x`: X matrix elements.
    /// - `y`: Y matrix elements.
    /// - `rows`: Row indices of X or Y.
    /// - `cols`: Column indices of X or Y.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn build_d2(d: &mut [f64], x: &ArrayView2<f64>, y: &ArrayView2<f64>, rows: &[usize], cols: &[usize]) {
        let xstr = x.strides();
        let ystr = y.strides();
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();

        unsafe {
            let r0 = *rows.get_unchecked(0) as isize;
            let r1 = *rows.get_unchecked(1) as isize;

            let c0 = *cols.get_unchecked(0) as isize;
            let c1 = *cols.get_unchecked(1) as isize;

            let xr0 = r0 * xstr[0];
            let xr1 = r1 * xstr[0];
            let yr0 = r0 * ystr[0];

            d[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            d[1] = *yptr.offset(yr0 + c1 * ystr[1]);

            d[2] = *xptr.offset(xr1 + c0 * xstr[1]);
            d[3] = *xptr.offset(xr1 + c1 * xstr[1]);
        }
    }

    /// Construct contraction determinant for excitation rank 3.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `x`: X matrix elements.
    /// - `y`: Y matrix elements.
    /// - `rows`: Row indices of X or Y.
    /// - `cols`: Column indices of X or Y.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn build_d3(d: &mut [f64], x: &ArrayView2<f64>, y: &ArrayView2<f64>, rows: &[usize], cols: &[usize]) {
        let xstr = x.strides();
        let ystr = y.strides();
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();

        unsafe {
            let r0 = *rows.get_unchecked(0) as isize;
            let r1 = *rows.get_unchecked(1) as isize;
            let r2 = *rows.get_unchecked(2) as isize;

            let c0 = *cols.get_unchecked(0) as isize;
            let c1 = *cols.get_unchecked(1) as isize;
            let c2 = *cols.get_unchecked(2) as isize;

            let xr0 = r0 * xstr[0];
            let xr1 = r1 * xstr[0];
            let xr2 = r2 * xstr[0];

            let yr0 = r0 * ystr[0];
            let yr1 = r1 * ystr[0];

            d[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            d[1] = *yptr.offset(yr0 + c1 * ystr[1]);
            d[2] = *yptr.offset(yr0 + c2 * ystr[1]);

            d[3] = *xptr.offset(xr1 + c0 * xstr[1]);
            d[4] = *xptr.offset(xr1 + c1 * xstr[1]);
            d[5] = *yptr.offset(yr1 + c2 * ystr[1]);

            d[6] = *xptr.offset(xr2 + c0 * xstr[1]);
            d[7] = *xptr.offset(xr2 + c1 * xstr[1]);
            d[8] = *xptr.offset(xr2 + c2 * xstr[1]);
        }
    }

    /// Construct contraction determinant for excitation rank 4.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `x`: X matrix elements.
    /// - `y`: Y matrix elements.
    /// - `rows`: Row indices of X or Y.
    /// - `cols`: Column indices of X or Y.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn build_d4(d: &mut [f64], x: &ArrayView2<f64>, y: &ArrayView2<f64>, rows: &[usize], cols: &[usize]) {
        let xstr = x.strides();
        let ystr = y.strides();
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();

        unsafe {
            let r0 = *rows.get_unchecked(0) as isize;
            let r1 = *rows.get_unchecked(1) as isize;
            let r2 = *rows.get_unchecked(2) as isize;
            let r3 = *rows.get_unchecked(3) as isize;

            let c0 = *cols.get_unchecked(0) as isize;
            let c1 = *cols.get_unchecked(1) as isize;
            let c2 = *cols.get_unchecked(2) as isize;
            let c3 = *cols.get_unchecked(3) as isize;

            let xr0 = r0 * xstr[0];
            let xr1 = r1 * xstr[0];
            let xr2 = r2 * xstr[0];
            let xr3 = r3 * xstr[0];

            let yr0 = r0 * ystr[0];
            let yr1 = r1 * ystr[0];
            let yr2 = r2 * ystr[0];

            d[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            d[1] = *yptr.offset(yr0 + c1 * ystr[1]);
            d[2] = *yptr.offset(yr0 + c2 * ystr[1]);
            d[3] = *yptr.offset(yr0 + c3 * ystr[1]);

            d[4] = *xptr.offset(xr1 + c0 * xstr[1]);
            d[5] = *xptr.offset(xr1 + c1 * xstr[1]);
            d[6] = *yptr.offset(yr1 + c2 * ystr[1]);
            d[7] = *yptr.offset(yr1 + c3 * ystr[1]);

            d[8]  = *xptr.offset(xr2 + c0 * xstr[1]);
            d[9]  = *xptr.offset(xr2 + c1 * xstr[1]);
            d[10] = *xptr.offset(xr2 + c2 * xstr[1]);
            d[11] = *yptr.offset(yr2 + c3 * ystr[1]);

            d[12] = *xptr.offset(xr3 + c0 * xstr[1]);
            d[13] = *xptr.offset(xr3 + c1 * xstr[1]);
            d[14] = *xptr.offset(xr3 + c2 * xstr[1]);
            d[15] = *xptr.offset(xr3 + c3 * xstr[1]);
        }
    }

    /// Construct contraction determinant for arbitrary excitation rank.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `l`: Excitation rank.
    /// - `x`: X matrix elements.
    /// - `y`: Y matrix elements.
    /// - `rows`: Row indices of X or Y.
    /// - `cols`: Column indices of X or Y.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn build_d_gen(d: &mut [f64], l: usize, x: &ArrayView2<f64>, y: &ArrayView2<f64>, rows: &[usize], cols: &[usize]) {
        // 2D views for X and Y are stored as base + r * strides[0] + c * strides[1] for an element at
        // (r, c) where base is the pointer to the start of the slab.
        let xstr = x.strides();
        let ystr = y.strides();
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();

        unsafe {
            // Iterate over rows of output.
            for i in 0..l {
                // Row index for output row i.
                let r = *rows.get_unchecked(i) as isize;

                let xr = r * xstr[0];
                let yr = r * ystr[0];
                let base = i * l;

                // Lower triange of matrix including diagonal gets X.
                // X[r, c] is at base + r * xstride[0] + c * xstride[1].
                for j in 0..=i {
                    let c = *cols.get_unchecked(j) as isize;
                    d[base + j] = *xptr.offset(xr + c * xstr[1]);
                }
                
                // Upper triangle gets Y.
                // Y[r, c] is at based + r * ystride[0] + c * ystride[1].
                for j in (i + 1)..l {
                    let c = *cols.get_unchecked(j) as isize;
                    d[base + j] = *yptr.offset(yr + c * ystr[1]);
                }
            }
        }
    }
}

/// Mix columns of `det1` into `det0` according to `bits`.
/// For column `c`, if bit `c` of `bits` is set then the output column is taken
/// from `det1`; otherwise it is taken from `det0`.
/// Dispatches to specialised implementations for small excitation ranks.
/// # Arguments:
/// - `d`: Matrix to write into.
/// - `det0`: Base matrix.
/// - `det1`: Mixing matrix.
/// - `l`: Excitation rank.
/// - `bits`: Bitstring selecting which columns are taken from `det1`.
/// # Returns
/// - `()`: Writes the mixed matrix into `d`.
#[inline(always)]
pub fn mix_columns(d: &mut [f64], det0: &[f64], det1: &[f64], l: usize, bits: u64) {
    match l {
        0 => {}
        1 => mix_columns::mix_columns1(d, det0, det1, bits),
        2 => mix_columns::mix_columns2(d, det0, det1, bits),
        3 => mix_columns::mix_columns3(d, det0, det1, bits),
        4 => mix_columns::mix_columns4(d, det0, det1, bits),
        _ => mix_columns::mix_columns_gen(d, det0, det1, l, bits),
    }
}

mod mix_columns {
    /// Mix columns of `det1` into `det0` according to `bits` for excitation rank 1.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `det0`: Base matrix.
    /// - `det1`: Mixing matrix.
    /// - `bits`: Bitstring.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn mix_columns1(d: &mut [f64], det0: &[f64], det1: &[f64], bits: u64) {
        let b0 = (bits & 1) != 0;
        d[0] = if b0 { det1[0] } else { det0[0] };
    }
    
    /// Mix columns of `det1` into `det0` according to `bits` for excitation rank 2.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `det0`: Base matrix.
    /// - `det1`: Mixing matrix.
    /// - `bits`: Bitstring.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn mix_columns2(d: &mut [f64], det0: &[f64], det1: &[f64], bits: u64) {
        let b0 = (bits & 1) != 0;
        let b1 = (bits & 2) != 0;

        d[0] = if b0 {det1[0]} else {det0[0]};
        d[1] = if b1 {det1[1]} else {det0[1]};

        d[2] = if b0 {det1[2]} else {det0[2]};
        d[3] = if b1 {det1[3]} else {det0[3]};
    }
    
    /// Mix columns of `det1` into `det0` according to `bits` for excitation rank 3.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `det0`: Base matrix.
    /// - `det1`: Mixing matrix.
    /// - `bits`: Bitstring.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn mix_columns3(d: &mut [f64], det0: &[f64], det1: &[f64], bits: u64) {
        let b0 = (bits & 1) != 0;
        let b1 = (bits & 2) != 0;
        let b2 = (bits & 4) != 0;

        d[0] = if b0 {det1[0]} else {det0[0]};
        d[1] = if b1 {det1[1]} else {det0[1]};
        d[2] = if b2 {det1[2]} else {det0[2]};

        d[3] = if b0 {det1[3]} else {det0[3]};
        d[4] = if b1 {det1[4]} else {det0[4]};
        d[5] = if b2 {det1[5]} else {det0[5]};

        d[6] = if b0 {det1[6]} else {det0[6]};
        d[7] = if b1 {det1[7]} else {det0[7]};
        d[8] = if b2 {det1[8]} else {det0[8]};
    }
    
    /// Mix columns of `det1` into `det0` according to `bits` for excitation rank 4.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `det0`: Base matrix.
    /// - `det1`: Mixing matrix.
    /// - `bits`: Bitstring.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn mix_columns4(d: &mut [f64], det0: &[f64], det1: &[f64], bits: u64) {
        let b0 = (bits & 1) != 0;
        let b1 = (bits & 2) != 0;
        let b2 = (bits & 4) != 0;
        let b3 = (bits & 8) != 0;

        d[0]  = if b0 {det1[0]}  else {det0[0]};
        d[1]  = if b1 {det1[1]}  else {det0[1]};
        d[2]  = if b2 {det1[2]}  else {det0[2]};
        d[3]  = if b3 {det1[3]}  else {det0[3]};

        d[4]  = if b0 {det1[4]}  else {det0[4]};
        d[5]  = if b1 {det1[5]}  else {det0[5]};
        d[6]  = if b2 {det1[6]}  else {det0[6]};
        d[7]  = if b3 {det1[7]}  else {det0[7]};

        d[8]  = if b0 {det1[8]}  else {det0[8]};
        d[9]  = if b1 {det1[9]}  else {det0[9]};
        d[10] = if b2 {det1[10]} else {det0[10]};
        d[11] = if b3 {det1[11]} else {det0[11]};

        d[12] = if b0 {det1[12]} else {det0[12]};
        d[13] = if b1 {det1[13]} else {det0[13]};
        d[14] = if b2 {det1[14]} else {det0[14]};
        d[15] = if b3 {det1[15]} else {det0[15]};
    }
    
    /// Mix columns of `det1` into `det0` according to `bits` for arbitrary excitation rank.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `det0`: Base matrix.
    /// - `det1`: Mixing matrix.
    /// - `bits`: Bitstring.
    /// - `l`: Excitation rank.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn mix_columns_gen(d: &mut [f64], det0: &[f64], det1: &[f64], l: usize, bits: u64) {
        unsafe {
            for r in 0..l {
                let base = r * l;
                for c in 0..l {
                    let k = base + c;
                    let use1 = ((bits >> c) & 1) != 0;
                    *d.get_unchecked_mut(k) = if use1 {*det1.get_unchecked(k)} else {*det0.get_unchecked(base + c)};
                }
            }
        }
    }
}

/// Construct the minor of an `n x n` matrix obtained by removing row `r_rm`
/// and column `c_rm`. Dispatches to specialised implementations for small matrix sizes.
/// # Arguments:
/// - `out`: Matrix to be written into.
/// - `m`: Base matrix.
/// - `n`: Dimension of the square matrix `m`.
/// - `r_rm`: Row index to remove.
/// - `c_rm`: Column index to remove.
/// # Returns
/// - `()`: Writes the minor into `out`.
#[inline(always)]
pub fn minor(out: &mut [f64], m: &[f64], n: usize, r_rm: usize, c_rm: usize) {
    match n {
        0 | 1 => {}
        2 => minor::minor2(out, m, r_rm, c_rm),
        3 => minor::minor3(out, m, r_rm, c_rm),
        4 => minor::minor4(out, m, r_rm, c_rm),
        _ => minor::minor_gen(out, m, n, r_rm, c_rm),
    }
}

mod minor {
    /// Calculate minor of a square matrix for excitation rank 2. 
    /// # Arguments:
    /// - `out`: Matrix to be written into.
    /// - `m`: Base matrix.
    /// - `r_rm`: Row index to remove.
    /// - `c_rm`: Column index to remove.
    /// # Returns
    /// - `()`: Writes the minor into `out`.
    #[inline(always)]
    pub(super) fn minor2(out: &mut [f64], m: &[f64], r_rm: usize, c_rm: usize) {
        let r = 1 ^ r_rm;
        let c = 1 ^ c_rm;
        out[0] = m[r * 2 + c];
    }

    /// Calculate minor of a square matrix for excitation rank 3. 
    /// # Arguments:
    /// - `out`: Matrix to be written into.
    /// - `m`: Base matrix.
    /// - `r_rm`: Row index to remove.
    /// - `c_rm`: Column index to remove.
    /// # Returns
    /// - `()`: Writes the minor into `out`.
    #[inline(always)]
    pub(super) fn minor3(out: &mut [f64], m: &[f64], r_rm: usize, c_rm: usize) {
        let mut ii = 0usize;
        for i in 0..3 {
            if i == r_rm {continue;}
            let base_i = i * 3;
            let base_o = ii * 2;
            let mut jj = 0usize;
            for j in 0..3 {
                if j == c_rm {continue;}
                out[base_o + jj] = m[base_i + j];
                jj += 1;
            }
            ii += 1;
        }
    }

    /// Calculate minor of a square matrix for excitation rank 4. 
    /// # Arguments:
    /// - `out`: Matrix to be written into.
    /// - `m`: Base matrix.
    /// - `r_rm`: Row index to remove.
    /// - `c_rm`: Column index to remove.
    /// # Returns
    /// - `()`: Writes the minor into `out`.
    #[inline(always)]
    pub(super) fn minor4(out: &mut [f64], m: &[f64], r_rm: usize, c_rm: usize) {
        let mut ii = 0usize;
        for i in 0..4 {
            if i == r_rm {continue;}
            let base_i = i * 4;
            let base_o = ii * 3;
            let mut jj = 0usize;
            for j in 0..4 {
                if j == c_rm {continue;}
                out[base_o + jj] = m[base_i + j];
                jj += 1;
            }
            ii += 1;
        }
    }

    /// Calculate minor of a square matrix for arbitrary excitation rank. 
    /// # Arguments:
    /// - `out`: Matrix to be written into.
    /// - `m`: Base matrix.
    /// - `r_rm`: Row index to remove.
    /// - `c_rm`: Column index to remove.
    /// # Returns
    /// - `()`: Writes the minor into `out`.
    #[inline(always)]
    pub(super) fn minor_gen(out: &mut [f64], m: &[f64], n: usize, r_rm: usize, c_rm: usize) {
        if n == 0 {return;}
        let mut ii = 0usize;
        for i in 0..n {
            if i == r_rm {continue;}
            let mut jj = 0usize;
            for j in 0..n {
                if j == c_rm {continue;}
                out[ii * (n - 1) + jj] = m[i * n + j];
                jj += 1;
            }
            ii += 1;
        }
    }
}

/// Compute the determinant of an `n x n` matrix using explicit formulas for
/// small sizes and a generic fallback for larger matrices.
/// # Arguments:
/// - `a`: Matrix stored in row-major order.
/// - `n`: Matrix dimension.
/// # Returns
/// - `Option<f64>`: Determinant of `a`, or `None` if evaluation fails.
#[inline(always)]
pub fn det(a: &[f64], n: usize) -> Option<f64> {
    if a.len() != n * n {
        return None;
    }

    match n {
        0 => Some(1.0),
        1 => Some(a[0]),
        2 => {
            let d = det::det2(a);
            if d.is_finite() {Some(d)} else {None}
        }
        3 => {
            let d = det::det3(a);
            if d.is_finite() {Some(d)} else {None}
        }
        4 => {
            let d = det::det4(a);
            if d.is_finite() {Some(d)} else {None}
        }
        _ => det::det_gen(a, n),
    }
}

mod det {
    use ndarray::{ArrayView2};
    use ndarray_linalg::{FactorizeInto, SVD, Determinant};

    /// Calculate the determinant of a 2 x 2 matrix.
    /// # Arguments:
    /// - `a`: Matrix to calculate the determinant of.
    /// # Returns
    /// - `f64`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det2(a: &[f64]) -> f64 {
        let a00 = a[0]; let a01 = a[1];
        let a10 = a[2]; let a11 = a[3];
        a00 * a11 - a01 * a10
    }

    /// Calculate the determinant of a 2 x 2 matrix.
    /// # Arguments:
    /// - `a00`: Matrix element (0, 0).
    /// - `a01`: Matrix element (0, 1).
    /// - `a10`: Matrix element (1, 0).
    /// - `a11`: Matrix element (1, 1).
    /// # Returns
    /// - `f64`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det2scalar(a00: f64, a01: f64, a10: f64, a11: f64) -> f64 {
        a00 * a11 - a01 * a10
    }

    /// Calculate the determinant of a 3 x 3 matrix. 
    /// # Arguments:
    /// - `a`: Matrix to calculate the determinant of.
    /// # Returns
    /// - `f64`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det3(a: &[f64]) -> f64 {
        let a00 = a[0]; let a01 = a[1]; let a02 = a[2];
        let a10 = a[3]; let a11 = a[4]; let a12 = a[5];
        let a20 = a[6]; let a21 = a[7]; let a22 = a[8];

        a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
    }

    /// Calculate the determinant of a 3 x 3 matrix.
    /// # Arguments:
    /// - `a00`: Matrix element (0, 0).
    /// - `a01`: Matrix element (0, 1).
    /// - `a02`: Matrix element (0, 2).
    /// - `a10`: Matrix element (1, 0).
    /// - `a11`: Matrix element (1, 1).
    /// - `a12`: Matrix element (1, 2).
    /// - `a20`: Matrix element (2, 0).
    /// - `a21`: Matrix element (2, 1).
    /// - `a22`: Matrix element (2, 2).
    /// # Returns
    /// - `f64`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det3scalar(a00: f64, a01: f64, a02: f64, a10: f64, a11: f64, a12: f64, a20: f64, a21: f64, a22: f64) -> f64 {
        a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
    }

    /// Calculate the determinant of a 4 x 4 matrix.
    /// # Arguments:
    /// - `a`: Matrix to calculate the determinant of.
    /// # Returns
    /// - `f64`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det4(a: &[f64]) -> f64 {
        let m00 = {
            let a11 = a[5];  let a12 = a[6];  let a13 = a[7];
            let a21 = a[9];  let a22 = a[10]; let a23 = a[11];
            let a31 = a[13]; let a32 = a[14]; let a33 = a[15];
            a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
        };

        let m01 = {
            let a10 = a[4];  let a12 = a[6];  let a13 = a[7];
            let a20 = a[8];  let a22 = a[10]; let a23 = a[11];
            let a30 = a[12]; let a32 = a[14]; let a33 = a[15];
            a10 * (a22 * a33 - a23 * a32) - a12 * (a20 * a33 - a23 * a30) + a13 * (a20 * a32 - a22 * a30)
        };

        let m02 = {
            let a10 = a[4];  let a11 = a[5];  let a13 = a[7];
            let a20 = a[8];  let a21 = a[9];  let a23 = a[11];
            let a30 = a[12]; let a31 = a[13]; let a33 = a[15];
            a10 * (a21 * a33 - a23 * a31) - a11 * (a20 * a33 - a23 * a30) + a13 * (a20 * a31 - a21 * a30)
        };

        let m03 = {
            let a10 = a[4];  let a11 = a[5];  let a12 = a[6];
            let a20 = a[8];  let a21 = a[9];  let a22 = a[10];
            let a30 = a[12]; let a31 = a[13]; let a32 = a[14];
            a10 * (a21 * a32 - a22 * a31) - a11 * (a20 * a32 - a22 * a30) + a12 * (a20 * a31 - a21 * a30)
        };

        let a00 = a[0];
        let a01 = a[1];
        let a02 = a[2];
        let a03 = a[3];

        a00 * m00 - a01 * m01 + a02 * m02 - a03 * m03
    }

    /// Compute determinant of `a` for arbitrary size using LU factorisation first and
    /// SVD as a fallback.
    /// # Arguments:
    /// - `a`: Matrix to find determinant of.
    /// - `n`: Matrix dimension.
    /// # Returns
    /// - `Option<f64>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn det_gen(a: &[f64], n: usize) -> Option<f64> {
        let av = ArrayView2::from_shape((n, n), a).ok()?;
        
        if let Ok(m) = av.to_owned().factorize_into() && let Ok(d) = m.det() && d.is_finite() {
            return Some(d);
        }

        let (u_opt, s, vt_opt) = av.svd(true, true).ok()?;
        let u = u_opt?;
        let vt = vt_opt?;

        let det_u = u.det().ok()?;
        let det_vt = vt.det().ok()?;

        let mut det = det_u * det_vt;
        for &si in s.iter() {
            det *= si;
        }

        if det.is_finite() {Some(det)} else {None}
    }
}

/// Compute the determinant and adjugate transpose of an `n x n` matrix using
/// explicit formulas for small sizes and generic LU/SVD-based methods for
/// larger matrices.
/// # Arguments:
/// - `adjt`: Scratch space for the adjugate transpose.
/// - `invs`: Scratch space for inverse singular values used by the SVD fallback.
/// - `lu`: Scratch space used by the LU-based fallback.
/// - `a`: Input matrix stored in row-major order.
/// - `n`: Matrix dimension.
/// - `thresh`: Threshold below which singular values are treated as zero in the
///   SVD fallback.
/// # Returns
/// - `Option<f64>`: Determinant of `a`, or `None` if evaluation fails.
#[inline(always)]
pub fn adjugate_transpose(adjt: &mut [f64], invs: &mut [f64], lu: &mut [f64], a: &[f64], n: usize, thresh: f64) -> Option<f64> {
    match n {
        0 => {Some(1.0)}
        1 => adjt::adjt1(adjt, a),
        2 => adjt::adjt2(adjt, a),
        3 => adjt::adjt3(adjt, a),
        4 => adjt::adjt4(adjt, a),
        _ => adjt::adjt_gen(adjt, invs, lu, a, n, thresh),
    }
}

mod adjt {
    use crate::maths::det::det2scalar;
    use crate::maths::det::det3scalar;
    use crate::maths::det::det4;

    use ndarray::{Array2, ArrayView2};
    use ndarray_linalg::{FactorizeInto, SVD, Determinant, InverseInto};

    /// Calculate determinant of 1 by 1 matrix `a` and write its adjugate transpose into `adjt`.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `a`: Input matrix.
    /// # Returns
    /// - `Option<f64>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt1(adjt: &mut [f64], a: &[f64]) -> Option<f64> {
        adjt[0] = 1.0;
        Some(a[0])
    }

    /// Calculate determinant of 2 by 2 matrix `a` and write its adjugate transpose into `adjt`.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `a`: Input matrix.
    /// # Returns
    /// - `Option<f64>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt2(adjt: &mut [f64], a: &[f64]) -> Option<f64> {
        let a00 = a[0]; let a01 = a[1];
        let a10 = a[2]; let a11 = a[3];

        let det = det2scalar(a00, a01, a10, a11);
        if !det.is_finite() {return None;}

        adjt[0] =  a11;
        adjt[1] = -a10;
        adjt[2] = -a01;
        adjt[3] =  a00;

        Some(det)
    }

    /// Calculate determinant of 3 by 3 matrix `a` and write its adjugate transpose into `adjt`.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `a`: Input matrix.
    /// # Returns
    /// - `Option<f64>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt3(adjt: &mut [f64], a: &[f64]) -> Option<f64> {
        let a00 = a[0]; let a01 = a[1]; let a02 = a[2];
        let a10 = a[3]; let a11 = a[4]; let a12 = a[5];
        let a20 = a[6]; let a21 = a[7]; let a22 = a[8];

        let det = det3scalar(a00, a01, a02, a10, a11, a12, a20, a21, a22);
        if !det.is_finite() {return None;}

        let c00 =  det2scalar(a11, a12, a21, a22);
        let c01 = -det2scalar(a10, a12, a20, a22);
        let c02 =  det2scalar(a10, a11, a20, a21);

        let c10 = -det2scalar(a01, a02, a21, a22);
        let c11 =  det2scalar(a00, a02, a20, a22);
        let c12 = -det2scalar(a00, a01, a20, a21);

        let c20 =  det2scalar(a01, a02, a11, a12);
        let c21 = -det2scalar(a00, a02, a10, a12);
        let c22 =  det2scalar(a00, a01, a10, a11);

        adjt[0] = c00; adjt[1] = c01; adjt[2] = c02;
        adjt[3] = c10; adjt[4] = c11; adjt[5] = c12;
        adjt[6] = c20; adjt[7] = c21; adjt[8] = c22;

        Some(det)
    }

    /// Calculate determinant of 4 by 4 matrix `a` and write its adjugate transpose into `adjt`.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `a`: Input matrix.
    /// # Returns
    /// - `Option<f64>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt4(adjt: &mut [f64], a: &[f64]) -> Option<f64> {
        let det = det4(a);
        if !det.is_finite() {return None;}

        for i in 0..4 {
            for j in 0..4 {
                let mut r = [0usize; 3];
                let mut c = [0usize; 3];

                let mut ri = 0usize;
                for rr in 0..4 {
                    if rr == i {continue;}
                    r[ri] = rr;
                    ri += 1;
                }

                let mut ci = 0usize;
                for cc in 0..4 {
                    if cc == j {continue;}
                    c[ci] = cc;
                    ci += 1;
                }

                let m00 = a[r[0] * 4 + c[0]];
                let m01 = a[r[0] * 4 + c[1]];
                let m02 = a[r[0] * 4 + c[2]];
                let m10 = a[r[1] * 4 + c[0]];
                let m11 = a[r[1] * 4 + c[1]];
                let m12 = a[r[1] * 4 + c[2]];
                let m20 = a[r[2] * 4 + c[0]];
                let m21 = a[r[2] * 4 + c[1]];
                let m22 = a[r[2] * 4 + c[2]];

                let minordet = det3scalar(m00, m01, m02, m10, m11, m12, m20, m21, m22);
                let sign = if ((i + j) & 1) == 0 {1.0} else {-1.0};
                adjt[i * 4 + j] = sign * minordet;
            }
        }
        Some(det)
    }
    
    /// Compute the determinant and adjugate transpose of an `n x n` matrix using
    /// LU factorisation first and SVD as a fallback.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `a`: Input matrix.
    /// # Returns
    /// - `Option<f64>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt_gen(adjt: &mut [f64], invs: &mut [f64], lu: &mut [f64], a: &[f64], n: usize, thresh: f64,) -> Option<f64> {
        if let Some(det) = adjtlu(adjt, lu, a, n) {return Some(det);}
        adjtsvd(adjt, invs, a, n, thresh)
    }

    /// Compute the determinant and adjugate transpose of an `n x n` matrix using
    /// LU factorisation.
    /// # Arguments:
    /// - `adjt`: Preallocated output scratch space.
    /// - `lu`: Preallocated scratch space for LU solve.
    /// - `a`: Matrix to find determinant and adjugate-transpose of.
    /// # Returns
    /// - `Option<f64>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjtlu(adjt: &mut [f64], lu: &mut [f64], a: &[f64], n: usize) -> Option<f64> {
        let nn = n * n;
        lu[..nn].copy_from_slice(&a[..nn]);

        let m = Array2::from_shape_vec((n, n), lu[..nn].to_vec()).ok()?;
        let f = m.factorize_into().ok()?;
        let det = f.det().ok()?;
        let inv = f.inv_into().ok()?;

        let invs = inv.as_slice()?;
        for r in 0..n {
            for c in 0..n {
                adjt[r * n + c] = det * invs[c * n + r];
            }
        }
        Some(det)
    }

    /// Compute the determinant and adjugate transpose of an `n x n` matrix using
    /// a singular value decomposition.
    /// # Arguments:
    /// - `adjt`: Preallocated output scratch space for the adjugate transpose.
    /// - `invs`: Preallocated scratch space for inverse singular values.
    /// - `a`: Input matrix stored in row-major order.
    /// - `n`: Matrix dimension.
    /// - `thresh`: Threshold below which singular values are treated as zero.
    /// # Returns
    /// - `Option<f64>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjtsvd(adjt: &mut [f64], invs: &mut [f64], a: &[f64], n: usize, thresh: f64) -> Option<f64> {
        let nn = n * n;
        if adjt.len() < nn || invs.len() < n || a.len() < nn {
            return None;
        }

        adjt[..nn].fill(0.0);
        invs[..n].fill(0.0);

        let av = ArrayView2::from_shape((n, n), a).ok()?;
        let (u_opt, s, vt_opt) = av.svd(true, true).ok()?;
        let u = u_opt?;
        let vt = vt_opt?;

        let det_u = u.det().ok()?;
        let det_vt = vt.det().ok()?;
        let mut red_det = det_u * det_vt;
        let mut det = red_det;

        let mut nzero = 0usize;
        let mut zerok = 0usize;

        for i in 0..n {
            let si = s[i];
            det *= si;
            if si.abs() > thresh {
                red_det *= si;
                invs[i] = 1.0 / si;
            } else {
                nzero += 1;
                zerok = i;
            }
        }

        if nzero == 0 {
            for i in 0..n {
                for r in 0..n {
                    let ur = u[(r, i)];
                    let scale = invs[i] * ur;
                    for c in 0..n {
                        adjt[r * n + c] += scale * vt[(i, c)];
                    }
                }
            }
            for x in &mut adjt[..nn] {
                *x *= det;
            }
        } else if nzero == 1 {
            let k = zerok;
            for r in 0..n {
                let ur = u[(r, k)];
                let scale = red_det * ur;
                for c in 0..n {
                    adjt[r * n + c] = scale * vt[(k, c)];
                }
            }
        } else {
            return Some(det);
        }

        Some(det)
    }
}

