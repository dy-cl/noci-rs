// maths.rs
use std::cell::RefCell;

use ndarray::{Array1, Array2, Array4, Axis};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex64;
use rayon::prelude::*;

thread_local! {static HT_SCRATCH: RefCell<Vec<f64>> = RefCell::new(Vec::new());}

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

/// Perform dot product between two vectors with unrolled loop of length 8.
/// # Arguments:
///     `x`: [f64], vector 1.
///     `y`: [f64], vector 2.
#[inline(always)]
fn dot_product_unroll8(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let mut sum = 0.0;
    let mut i = 0usize;

    // Accumulate contributions to dot product sum 8 at a time.
    while i + 8 <= n {
        unsafe {
            sum += *x.get_unchecked(i) * *y.get_unchecked(i);
            sum += *x.get_unchecked(i + 1) * *y.get_unchecked(i + 1);
            sum += *x.get_unchecked(i + 2) * *y.get_unchecked(i + 2);
            sum += *x.get_unchecked(i + 3) * *y.get_unchecked(i + 3);
            sum += *x.get_unchecked(i + 4) * *y.get_unchecked(i + 4);
            sum += *x.get_unchecked(i + 5) * *y.get_unchecked(i + 5);
            sum += *x.get_unchecked(i + 6) * *y.get_unchecked(i + 6);
            sum += *x.get_unchecked(i + 7) * *y.get_unchecked(i + 7);
        }
        i += 8;
    }

    // Accumulate less than 8 remaining contributions.
    while i < n {
        unsafe { sum += *x.get_unchecked(i) * *y.get_unchecked(i); }
        i += 1;
    }
    sum
}

/// Calculate Einstein summation of matrices `g` and `h` and 4D tensor `t` as 
/// \sum_{a,c}\sum_{b,d} g_{b,a} t_{a,c,b,d} h_{d,c}. Assumes `g`, `h` and `t` all 
/// have axes of equal length.
/// # Arguments
///     `g`: Array2, matrix 1. 
///     `t`: Array4, 4D tensor.
///     `h`: Array2, matrix 2.
pub fn einsum_ba_acbd_dc_real(g: &Array2<f64>, t: &Array4<f64>, h: &Array2<f64>) -> f64 {
    let n = g.nrows();

    // Convert ndarrays into memory ordered slice.
    let gs = g.as_slice_memory_order().unwrap();
    let hs = h.as_slice_memory_order().unwrap();
    let ts = t.as_slice_memory_order().unwrap();

    let mut acc = 0.0f64;
    
    // Reuse ht across calls to this function.
    HT_SCRATCH.with(|buf| {
        // Transpose h[d, c] = ht[c, d] into layout which is contiguous by d as this will be fastest index.
        let mut ht = buf.borrow_mut();
        ht.resize(n * n, 0.0);
        for d in 0..n {
            let d_idx = d * n;
            for c in 0..n {
                // Use of get_unchecked means no out of bounds checking is performed. If index i is invalid 
                // this produces undefined behaviour rather than a panic, check this when debugging. 
                // Use of unsafe is consequently required as get_unchecked is an unsafe operation.
                ht[c * n + d] = unsafe{*hs.get_unchecked(d_idx + c)};
            }
        }

        // Index of 4D tensor element in 1D is given by (((a * n + c) * n + b) * n + d). So d varies
        // fastes, then b, then c, then a. Therefore iteration should be in order, a, c, b, d.
        // So iterate (a,c,b,d) for contiguous access.
        for a in 0..n {
            // For a given a, the block t[a, :, :, :] starts at a * n^3. See above element indexing.
            let ta_index = a * n * n * n;
            for c in 0..n {
                // ht[c, d] (h[d, c]).
                let ht_cd = &ht[c * n..(c + 1) * n];
                // For a given a, c, the block t[a, c, :, :] starts at a*n^3 + c*n^2. See above element indexing.
                let tac_index = ta_index + c * n * n;

                for b in 0..n {
                    let g_ba = unsafe {*gs.get_unchecked(b * n + a)};
                    if g_ba == 0.0 {continue;}
                    
                    // For a given a, c, b, the block t[a, c, b, :] starts at a*n^3 + c*n^2 + b*n. See above element indexing.
                    let tacb_index = tac_index + b * n;
                    // Get the vector t[a, c, b, :].
                    let tacb_vec = unsafe {ts.get_unchecked(tacb_index..tacb_index + n)};
                    
                    // Compute dot product of t[a, c, b, :] with ht[:, c] with unrolling.
                    let sum = dot_product_unroll8(tacb_vec, ht_cd);
                    acc += g_ba * sum;
                }
            }
        }
        acc
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


