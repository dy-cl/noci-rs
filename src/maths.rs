// maths.rs
use std::cell::RefCell;

use ndarray::{Array1, Array2, Array4, Axis};
use ndarray_linalg::{Eigh, UPLO, Determinant};
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
///     `x`: Array2, X matrix elements.
///     `y`: Array2, Y matrix elements.
///     `rows`: [usize], row indices of X or Y.
///     `cols`: [usize], column indices of X or Y.
pub fn build_d(x: &Array2<f64>, y: &Array2<f64>, rows: &[usize], cols: &[usize],) -> Array2<f64> {
    let l = rows.len();
    let mut out = Array2::<f64>::zeros((l, l));
    for i in 0..l {
        let r = rows[i];
        for j in 0..l {
            let c = cols[j];
            out[(i, j)] = if i >= j {x[(r, c)]} else {y[(r, c)]};
        }
    }
    out
}

/// Mix columns of det1 into det0 as prescribed by a bitstring. For each column c if bit c of the
/// bitstring is 1 the output column c is taken from det1 and det0 otherwise.
/// # Arguments:
///    `det0`: Array2, base matrix.
///    `det1`: Array2, mixing matrix.
///    `bits`: usize, bitstring.
pub fn mix_columns(det0: &Array2<f64>, det1: &Array2<f64>, bits: u64) -> Array2<f64> {
    let n = det0.ncols();
    let mut out = det0.clone();
    for c in 0..n {
        if ((bits >> c) & 1) == 1 {
            out.column_mut(c).assign(&det1.column(c));
        }
    }
    out
}

/// Calculate minor of a square matrix, that is, the matrix obtained by removing row `r_rm` and
/// column `c_rm`.
/// # Arguments:
///     `m`: Array2, base matrix.
///     `r_rm`: usize, row index to remove.
///     `c_rm`: usize, column index to remove.
pub fn minor(m: &Array2<f64>, r_rm: usize, c_rm: usize) -> Array2<f64> {
    let n = m.nrows();
    if n == 0 {return Array2::<f64>::zeros((0, 0));}
    let mut out = Array2::<f64>::zeros((n - 1, n - 1));
    let mut ii = 0;
    for i in 0..n {
        if i == r_rm {continue;}
        let mut jj = 0;
        for j in 0..n {
            if j == c_rm {continue;}
            out[(ii, jj)] = m[(i, j)];
            jj += 1;
        }
        ii += 1;
    }
    out
}

/// Calculate determinant and transpose of adjugate matrix using cofactor expansion. 
/// For each i, j calculate the minor matrix M_{i, j} (see above), find its determinant, and
/// compute adjt_{i, j} = -1^{i + j} det(M_{i, j}). The determinant is subsequently found as 
/// det(A) = \sum_j A_{0, j} * adjt_{0, j}.
/// # Arguments:
///     `a`: Array2, input matrix.
pub fn adjugate_transpose(a: &Array2<f64>) -> Option<(f64, Array2<f64>)> {
    let n = a.nrows();

    if n == 0 {return Some((1.0, Array2::zeros((0, 0))));}
    if n == 1 {return Some((a[(0, 0)], Array2::from_elem((1, 1), 1.0)));}

    let mut adjt = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut m = Array2::<f64>::zeros((n - 1, n - 1));
            let mut ii = 0;
            for r in 0..n {
                if r == i { continue; }
                let mut jj = 0;
                for c in 0..n {
                    if c == j { continue; }
                    m[(ii, jj)] = a[(r, c)];
                    jj += 1;
                }
                ii += 1;
            }
            let detm = match n - 1 {
                0 => 1.0,
                1 => m[(0, 0)],
                _ => m.det().unwrap_or(0.0),
            };
            adjt[(i, j)] = if ((i + j) & 1) == 0 {detm} else {-detm};
        }
    }

    let mut det = 0.0;
    for j in 0..n {
        det += a[(0, j)] * adjt[(0, j)];
    }

    Some((det, adjt))
}

