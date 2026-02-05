// maths.rs
use std::cell::RefCell;

use ndarray::{Array1, Array2, Array4, Axis};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex64;
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

pub fn loewdin_x_complex(s: &Array2<Complex64>, project: bool, tol: f64) -> Array2<Complex64> {
    // S = U \Lambda U^{\dagger}
    let (lambdas, u) = s.eigh(UPLO::Lower).unwrap(); 
    // \Lambda^{-1/2}
    let invsqrt: Array1<Complex64> = lambdas.mapv(|x| { 
        if project {
            if x > tol {Complex64::new(1.0 / x.sqrt(), 0.0)} else {Complex64::new(0.0, 0.0)}
        } else {
           Complex64::new(1.0 / x.sqrt(), 0.0) 
        }
    });
    let d = Array2::from_diag(&invsqrt);
    // X = U \Lambda^{-1/2} U^{\dagger}
    let u_dag = u.t().map(|z| z.conj());
    u.dot(&d).dot(&u_dag)
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

pub fn general_evp_complex(f: &Array2<Complex64>, s: &Array2<Complex64>, project: bool, 
                           tol: f64) -> (Array1<f64>, Array2<Complex64>) {
    // X = S^{-1/2}
    let x = loewdin_x_complex(s, project, tol); 
    // \tilde{F} = X^{\dagger} F X.
    let xt = x.t().map(|z| z.conj());
    let ft = xt.dot(f).dot(&x);
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

pub fn einsum_ab_ba_real(g: &Array2<f64>, h: &Array2<f64>) -> f64 {
    let n = g.nrows();
    let gs = g.as_slice_memory_order().unwrap();
    let hs = h.as_slice_memory_order().unwrap();
    let mut acc = 0.0;
    for a in 0..n {
        let arow = a * n;
        for b in 0..n {
            let g_ab = unsafe { *gs.get_unchecked(arow + b) };
            let h_ba = unsafe { *hs.get_unchecked(b * n + a) };
            acc += g_ab * h_ba;
        }
    }
    acc
}

pub fn einsum_ab_ab_real(g: &Array2<f64>, h: &Array2<f64>) -> f64 {
    let n = g.nrows();
    let gs = g.as_slice_memory_order().unwrap();
    let hs = h.as_slice_memory_order().unwrap();
    let mut acc = 0.0;
    for i in 0..(n*n) {
        acc += unsafe { *gs.get_unchecked(i) } * unsafe { *hs.get_unchecked(i) };
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

pub fn einsum_ba_acbd_dc_real(g: &Array2<f64>, t: &Array4<f64>, h: &Array2<f64>) -> f64 {
    let n = g.nrows();

    let gs = g.as_slice_memory_order().unwrap();
    let hs = h.as_slice_memory_order().unwrap();
    let ts = t.as_slice_memory_order().unwrap();

    let mut acc = 0.0f64;

    HT_SCRATCH.with(|hbuf| {
        GT_SCRATCH.with(|gbuf| {
            let mut ht = hbuf.borrow_mut();
            let mut gt = gbuf.borrow_mut();

            ht.resize(n * n, 0.0);
            gt.resize(n * n, 0.0);

            // ht[c,d] = h[d,c]
            for c in 0..n {
                let c_idx = c * n;
                for d in 0..n {
                    unsafe {
                        //*ht.get_unchecked_mut(c_idx + d) = *hs.get_unchecked(d * n + c);
                        *ht.get_unchecked_mut(c_idx + d) = *hs.get_unchecked(c * n + d);
                    }
                }
            }

            // gt[a,b] = g[b,a]
            for b in 0..n {
                let b_idx = b * n;
                for a in 0..n {
                    unsafe {
                        *gt.get_unchecked_mut(a * n + b) = *gs.get_unchecked(b_idx + a);
                    }
                }
            }

            unsafe {
                let ts_ptr = ts.as_ptr();
                let ht_ptr = ht.as_ptr();
                let gt_ptr = gt.as_ptr();

                for a in 0..n {
                    let gt_a_ptr = gt_ptr.add(a * n); 
                    for c in 0..n {
                        let ht_c_ptr = ht_ptr.add(c * n); 
                        let tac_base = (a * n + c) * n * n; 
                        for b in 0..n {
                            let g_ba = *gt_a_ptr.add(b);
                            if g_ba == 0.0 {
                                continue;
                            }

                            let tabc_vec_ptr = ts_ptr.add(tac_base + b * n);

                            let dot = dot_product_unroll8(tabc_vec_ptr, ht_c_ptr, n);
                            acc = g_ba.mul_add(dot, acc);
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

pub fn contract_pq_rs(v_pqrs: &Array4<f64>, m_rs: &Array2<f64>) -> Array2<f64> {
    let (np, nq, nr, ns) = v_pqrs.dim();

    let mut out = Array2::<f64>::zeros((np, nq));
    for p in 0..np {
        for q in 0..nq {
            let mut acc = 0.0;
            for r in 0..nr {
                for s in 0..ns {
                    acc += v_pqrs[(p, q, r, s)] * m_rs[(r, s)];
                }
            }
            out[(p, q)] = acc;
        }
    }
    out
}

pub fn contract_pq_rs_transposed(v_pqrs: &Array4<f64>, m_sr: &Array2<f64>) -> Array2<f64> {
    let (np, nq, nr, ns) = v_pqrs.dim();

    let mut out = Array2::<f64>::zeros((np, nq));
    for p in 0..np {
        for q in 0..nq {
            let mut acc = 0.0;
            for r in 0..nr {
                for s in 0..ns {
                    acc += v_pqrs[(p, q, r, s)] * m_sr[(s, r)];
                }
            }
            out[(p, q)] = acc;
        }
    }
    out
}
