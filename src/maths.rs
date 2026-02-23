// maths.rs
use std::cell::RefCell;

use ndarray::{Array1, Array2, ArrayView2, Array4, Axis};
use ndarray_linalg::{SVD, Eigh, UPLO, Determinant};
use rayon::prelude::*;

thread_local! {static HT_SCRATCH: RefCell<Vec<f64>> = RefCell::new(Vec::new()); static GT_SCRATCH: RefCell<Vec<f64>> = RefCell::new(Vec::new());}

/// Loewdin symmetric orthogonalizer, computes X = S^{-1/2} for real/complex matrices.
/// # Arguments 
/// `s`: Array2, Hermitian matrix, uses only the lower triangle. Often AO basis overlap 
/// matrix is s here.
/// `project`: Bool, whether or not to project to non-zero positive subspace of S.
/// `tol`: Float, tolerance for whether a number is considered zero.
pub fn loewdin_x_real(s: &Array2<f64>, project: bool, tol: f64) -> Array2<f64> {
    // S = U \Lambda U^T
    let (lambdas, evecs) = s.eigh(UPLO::Lower).unwrap();
    // \Lambda^{-1/2}
    let invsqrt: Array1<f64> = lambdas.mapv(|x| {
        if project {
            if x > tol {1.0 / x.sqrt()} else {0.0}
        } else {
            1.0 / x.sqrt()
        }
    });
    let d = Array2::from_diag(&invsqrt);
    // X = U \Lambda^{-1/2} U^T
    evecs.dot(&d).dot(&evecs.t())
}

/// Solve the real/complex generalized eigenproblem F C = S C e using the Loewdin orthogonalizer.
/// # Arguments
///     `f`: Array2, Hermitian matrix, uses only the lower triangle. Often Fock matrix 
///     is f here.
///     `s`: Array2, Hermitian matrix, uses only the lower triangle. Often AO basis overlap 
///     matrix is s here.
///     `project`: Bool, whether or not to project to non-zero positive subspace of S.
///     `tol`: Float, tolerance for whether a number is considered zero.
pub fn general_evp_real(f: &Array2<f64>, s: &Array2<f64>, project: bool, tol: f64) 
                        -> (Array1<f64>, Array2<f64>) {
    // X = S^{-1/2}
    let x = loewdin_x_real(s, project, tol); 
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
///     `g`: Array2, matrix 1. 
///     `h`: Array2, matrix 2.
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
///     `x`: [f64], vector 1.
///     `y`: [f64], vector 2.
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
///     `g`: Array2, matrix 1. 
///     `t`: Array4, 4D tensor.
///     `h`: Array2, matrix 2.
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

/// Calculate the determinant of a 2 x 2 matrix. Takes Array2.
/// # Arguments:
///     `a`: Array2, matrix to calculate the determinant of.
fn det2(a: &Array2<f64>) -> f64 {
    let a00 = a[(0, 0)]; let a01 = a[(0, 1)];
    let a10 = a[(1, 0)]; let a11 = a[(1, 1)];
    a00 * a11 - a01 * a10
}

/// Calculate the determinant of a 2 x 2 matrix. Takes 4 scalars.
/// # Arguments:
///     `a`: Array2, matrix to calculate the determinant of.
fn det2scalar(a00: f64, a01: f64, a10: f64, a11: f64) -> f64 {
    a00 * a11 - a01 * a10
}


/// Calculate the determinant of a 3 x 3 matrix. Takes Array2
/// # Arguments:
///     `a`: Array2, matrix to calculate the determinant of.
fn det3(a: &Array2<f64>) -> f64 {
    let a00 = a[(0, 0)]; let a01 = a[(0, 1)]; let a02 = a[(0, 2)];
    let a10 = a[(1, 0)]; let a11 = a[(1, 1)]; let a12 = a[(1, 2)];
    let a20 = a[(2, 0)]; let a21 = a[(2, 1)]; let a22 = a[(2, 2)];
    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
}

/// Calculate the determinant of a 3 x 3 matrix. Takes 9 scalars.
/// # Arguments:
///     `a`: Array2, matrix to calculate the determinant of.
fn det3scalar(a00: f64, a01: f64, a02: f64, a10: f64, a11: f64, a12: f64, a20: f64, a21: f64, a22: f64) -> f64 {
    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
}

/// Calculate the determinant of a 4 x 4 matrix. Takes Array2.
/// # Arguments:
///     `a`: Array2, matrix to calculate the determinant of.
fn det4(a: &Array2<f64>) -> f64 {
    let m00 = {
        let a11 = a[(1, 1)]; let a12 = a[(1, 2)]; let a13 = a[(1, 3)];
        let a21 = a[(2, 1)]; let a22 = a[(2, 2)]; let a23 = a[(2, 3)];
        let a31 = a[(3, 1)]; let a32 = a[(3, 2)]; let a33 = a[(3, 3)];
        a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
    };

    let m01 = {
        let a10 = a[(1, 0)]; let a12 = a[(1, 2)]; let a13 = a[(1, 3)];
        let a20 = a[(2, 0)]; let a22 = a[(2, 2)]; let a23 = a[(2, 3)];
        let a30 = a[(3, 0)]; let a32 = a[(3, 2)]; let a33 = a[(3, 3)];
        a10 * (a22 * a33 - a23 * a32) - a12 * (a20 * a33 - a23 * a30) + a13 * (a20 * a32 - a22 * a30)
    };

    let m02 = {
        let a10 = a[(1, 0)]; let a11 = a[(1, 1)]; let a13 = a[(1, 3)];
        let a20 = a[(2, 0)]; let a21 = a[(2, 1)]; let a23 = a[(2, 3)];
        let a30 = a[(3, 0)]; let a31 = a[(3, 1)]; let a33 = a[(3, 3)];
        a10 * (a21 * a33 - a23 * a31) - a11 * (a20 * a33 - a23 * a30) + a13 * (a20 * a31 - a21 * a30)
    };

    let m03 = {
        let a10 = a[(1, 0)]; let a11 = a[(1, 1)]; let a12 = a[(1, 2)];
        let a20 = a[(2, 0)]; let a21 = a[(2, 1)]; let a22 = a[(2, 2)];
        let a30 = a[(3, 0)]; let a31 = a[(3, 1)]; let a32 = a[(3, 2)];
        a10 * (a21 * a32 - a22 * a31) - a11 * (a20 * a32 - a22 * a30) + a12 * (a20 * a31 - a21 * a30)
    };

    let a00 = a[(0, 0)];
    let a01 = a[(0, 1)];
    let a02 = a[(0, 2)];
    let a03 = a[(0, 3)];

    a00 * m00 - a01 * m01 + a02 * m02 - a03 * m03
}

/// Calculate determinant of 1 by 1 matrix `a` and write its adjugate transpose into `adjt`.
/// # Arguments:
///     `adjt`: Array2, scratch space for writing adjugate transpose.
///     `a`: Array2, input matrix.
fn adjt1(adjt: &mut Array2<f64>, a: &Array2<f64>) -> Option<f64> {
    adjt[(0, 0)] = 1.0;
    Some(a[(0, 0)])
}

/// Calculate determinant of 2 by 2 matrix `a` and write its adjugate transpose into `adjt`.
/// # Arguments:
///     `adjt`: Array2, scratch space for writing adjugate transpose.
///     `a`: Array2, input matrix.
fn adjt2(adjt: &mut Array2<f64>, a: &Array2<f64>, thresh: f64) -> Option<f64> {
    let a00 = a[(0, 0)]; let a01 = a[(0, 1)];
    let a10 = a[(1, 0)]; let a11 = a[(1, 1)];

    let det = det2scalar(a00, a01, a10, a11);
    if !det.is_finite() || det.abs() <= thresh * thresh {return None;}

    adjt[(0, 0)] =  a11;
    adjt[(0, 1)] = -a10;
    adjt[(1, 0)] = -a01;
    adjt[(1, 1)] =  a00;

    Some(det)
}

/// Calculate determinant of 3 by 3 matrix `a` and write its adjugate transpose into `adjt`.
/// # Arguments:
///     `adjt`: Array2, scratch space for writing adjugate transpose.
///     `a`: Array2, input matrix.
fn adjt3(adjt: &mut Array2<f64>, a: &Array2<f64>, thresh: f64) -> Option<f64> {
    let a00 = a[(0, 0)]; let a01 = a[(0, 1)]; let a02 = a[(0, 2)];
    let a10 = a[(1, 0)]; let a11 = a[(1, 1)]; let a12 = a[(1, 2)];
    let a20 = a[(2, 0)]; let a21 = a[(2, 1)]; let a22 = a[(2, 2)];

    let det = det3scalar(a00, a01, a02, a10, a11, a12, a20, a21, a22);
    if !det.is_finite() || det.abs() <= thresh * thresh * thresh {return None;}

    let c00 =  det2scalar(a11, a12, a21, a22);
    let c01 = -det2scalar(a10, a12, a20, a22);
    let c02 =  det2scalar(a10, a11, a20, a21);

    let c10 = -det2scalar(a01, a02, a21, a22);
    let c11 =  det2scalar(a00, a02, a20, a22);
    let c12 = -det2scalar(a00, a01, a20, a21);

    let c20 =  det2scalar(a01, a02, a11, a12);
    let c21 = -det2scalar(a00, a02, a10, a12);
    let c22 =  det2scalar(a00, a01, a10, a11);

    adjt[(0, 0)] = c00; adjt[(0, 1)] = c01; adjt[(0, 2)] = c02;
    adjt[(1, 0)] = c10; adjt[(1, 1)] = c11; adjt[(1, 2)] = c12;
    adjt[(2, 0)] = c20; adjt[(2, 1)] = c21; adjt[(2, 2)] = c22;

    Some(det)
}

/// Calculate determinant of 4 by 4 matrix `a` and write its adjugate transpose into `adjt`.
/// # Arguments:
///     `adjt`: Array2, scratch space for writing adjugate transpose.
///     `a`: Array2, input matrix.
fn adjt4(adjt: &mut Array2<f64>, a: &Array2<f64>, thresh: f64) -> Option<f64> {
    let det = det4(a);
    if !det.is_finite() || det.abs() <= thresh.powi(4) {return None;}

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

            let m00 = a[(r[0], c[0])]; let m01 = a[(r[0], c[1])]; let m02 = a[(r[0], c[2])];
            let m10 = a[(r[1], c[0])]; let m11 = a[(r[1], c[1])]; let m12 = a[(r[1], c[2])];
            let m20 = a[(r[2], c[0])]; let m21 = a[(r[2], c[1])]; let m22 = a[(r[2], c[2])];

            let minordet = det3scalar(m00, m01, m02, m10, m11, m12, m20, m21, m22);
            let sign = if ((i + j) & 1) == 0 {1.0} else {-1.0};
            adjt[(i, j)] = sign * minordet;
        }
    }
    Some(det)
}

/// Calculate a matrix vector product HC = U in parallel.
/// # Arguments  
/// `h`: Array2, matrix. 
/// `c`: Array1, vector.
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
///     `eri`: Array4, AO basis ERIs.
///     `c_mu_p`: Array2, MO coefficients C_{\mu, p}. 
///     `c_nu_q`: Array2, MO coefficients C_{\nu, q}.
///     `c_lam_r`: Array2, MO coefficients C_{\lambda, r}.
///     `c_sigma_s`: Array2, MO coefficients C_{\sigma, s}.
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

/// Build square L x L contraction determinant with X elements in the diagonal and lower half, Y
/// elements in the upper half.
/// # Arguments:
///     `d`: Array2, matrix to write into.
///     `x`: Array2, X matrix elements.
///     `y`: Array2, Y matrix elements.
///     `rows`: [usize], row indices of X or Y.
///     `cols`: [usize], column indices of X or Y.
pub fn build_d(d: &mut Array2<f64>, x: &ArrayView2<f64>, y: &ArrayView2<f64>, rows: &[usize], cols: &[usize],) {
    let l = rows.len();
    for i in 0..l {
        let r = rows[i];
        // Diagonal and lower triangle.
        for j in 0..=i {
            let c = cols[j];
            d[(i, j)] = x[(r, c)];
        }
        // Upper triangle.
        for j in (i + 1)..l {
            let c = cols[j];
            d[(i, j)] = y[(r, c)];
        }
    }
}

/// Mix columns of det1 into det0 as prescribed by a bitstring. For each column c if bit c of the
/// bitstring is 1 the output column c is taken from det1 and det0 otherwise.
/// # Arguments:
///     `d`: Array2, matrix to write into.
///    `det0`: Array2, base matrix.
///    `det1`: Array2, mixing matrix.
///    `bits`: usize, bitstring.
pub fn mix_columns(d: &mut Array2<f64>, det0: &Array2<f64>, det1: &Array2<f64>, bits: u64) {
    let n = det0.ncols();
    for c in 0..n {
        if ((bits >> c) & 1) == 1 {
            d.column_mut(c).assign(&det1.column(c));
        } else {
            d.column_mut(c).assign(&det0.column(c));
        }
    }
}

/// Calculate minor of a square matrix, that is, the matrix obtained by removing row `r_rm` and
/// column `c_rm`.
/// # Arguments:
///     `out`: Array2, matrix to be written into.
///     `m`: Array2, base matrix.
///     `r_rm`: usize, row index to remove.
///     `c_rm`: usize, column index to remove.
pub fn minor(out: &mut Array2<f64>, m: &Array2<f64>, r_rm: usize, c_rm: usize) {
    let n = m.nrows();
    if n == 0 {return;}
    let mut ii = 0usize;
    for i in 0..n {
        if i == r_rm {continue;}
        let mut jj = 0usize;
        for j in 0..n {
            if j == c_rm {continue;}
            out[(ii, jj)] = m[(i, j)];
            jj += 1;
        }
        ii += 1;
    }
}

/// Compute determinant of `a` with fast exact formula up to n == 4 (this would need to be larger 
/// if the code is generalised to allow greater than double excitations), or fallback to SVD determinant
/// evaluation method when this proves to be troublesome.
/// # Arguments:
///     `a`: Array2, matrix to find determinant of.
///     `thresh`: f64, tolerance for singular values in SVD fallback.
pub fn det_thresh(a: &Array2<f64>, thresh: f64) -> Option<f64> {
    let n = a.nrows();
    if n != a.ncols() {return None;}
    if n == 0 {return Some(1.0);}
    if n == 1 {return Some(a[(0, 0)]);}

    let det = match n {2 => det2(a), 3 => det3(a), 4 => det4(a), _ => unreachable!()};
    if det.is_finite() {return Some(det);}

    let (u_opt, s, vt_opt) = a.svd(true, true).ok()?;
    let u = u_opt?;
    let vt = vt_opt?;

    let det_u = u.det().ok()?;
    let det_vt = vt.det().ok()?;

    let mut det = det_u * det_vt;
    for &si in s.iter() {
        if si.abs() <= thresh {return Some(0.0);}
        det *= si;
    }
    Some(det)
}

/// Compute determinant of `a` with fast exact formula up to n == 4 (this would need to be larger 
/// if the code is generalised to allow greater than double excitations) and the adjugate-transpose
/// and fallback to SVD determinant and adjugate-transpose evaluation method when this proves to be
/// troublesome.
///     `adjt`: Array2, preallocated output scratch space.
///     `invs`: Array1, preallocated scratch space for singular values of SVD.
///     `a`: Array2, matrix to find determinant and adjugate-transpose of.
///     `thresh`: f64, tolerance for singular values in SVD fallback.
pub fn adjugate_transpose(adjt: &mut Array2<f64>, invs: &mut Array1<f64>, a: &Array2<f64>, thresh: f64) -> Option<f64> {
    let n = a.nrows();
    if n != a.ncols() {return None;}
  
    let detquick = match n {
        0 => {Some(1.0)}
        1 => adjt1(adjt, a),
        2 => adjt2(adjt, a, thresh),
        3 => adjt3(adjt, a, thresh),
        4 => adjt4(adjt, a, thresh),
        _ => None,
    };
    adjt.fill(0.0);
    invs.fill(0.0);
    if detquick.is_some() {return detquick;}

    let (u_opt, s, vt_opt) = a.svd(true, true).ok()?;
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
            let alpha = invs[i];
            for r in 0..n {
                let ur = u[(r, i)];
                let scale = alpha * ur;
                for c in 0..n {
                    adjt[(r, c)] += scale * vt[(i, c)];
                }
            }
        }
        *adjt *= det;
    } else if nzero == 1 {
        let k = zerok;
        for r in 0..n {
            let ur = u[(r, k)];
            let scale = red_det * ur;
            for c in 0..n {
                adjt[(r, c)] = scale * vt[(k, c)];
            }
        }
    } else {
        return None;
    }
    Some(det)
}
