// maths/einsum.rs 

use std::{cell::RefCell};

use ndarray::{Array2, Array4};
use num_complex::Complex64;


thread_local! {
    static HT_SCRATCH: RefCell<Vec<f64>> = const {RefCell::new(Vec::new())};
    static GT_SCRATCH: RefCell<Vec<f64>> = const {RefCell::new(Vec::new())};

    static CHT_SCRATCH: RefCell<Vec<Complex64>> = const {RefCell::new(Vec::new())};
    static CGT_SCRATCH: RefCell<Vec<Complex64>> = const {RefCell::new(Vec::new())};
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

/// Calculate Einstein summation of matrices `g` and `h` as \sum_{a,b} g_{b,a} h_{ab}.
/// Assumes `g` and `h` are of identical shape.
/// # Arguments
/// - `g`: Matrix 1.
/// - `h`: Matrix 2.
/// # Returns
/// - `Complex64`: Contracted scalar.
pub fn einsum_ba_ab_complex(g: &Array2<Complex64>, h: &Array2<Complex64>) -> Complex64 {
    let n = g.nrows();

    let gs = g.as_slice_memory_order().unwrap();
    let hs = h.as_slice_memory_order().unwrap();

    let mut acc = Complex64::new(0.0, 0.0);

    for a in 0..n {
        for b in 0..n {
            let g_ba = unsafe {*gs.get_unchecked(b * n + a)};
            let h_ab = unsafe {*hs.get_unchecked(a * n + b)};
            acc += g_ba * h_ab;
        }
    }

    acc
}

/// Calculate Einstein summation of complex matrix `g` and real matrix `h` as \sum_{a,b} g_{b,a} h_{ab}.
/// Assumes `g` and `h` are of identical shape.
/// # Arguments
/// - `g`: Complex matrix.
/// - `h`: Real matrix.
/// # Returns
/// - `Complex64`: Contracted scalar.
pub fn einsum_ba_ab_complex_real(g: &Array2<Complex64>, h: &Array2<f64>) -> Complex64 {
    let n = g.nrows();

    let gs = g.as_slice_memory_order().unwrap();
    let hs = h.as_slice_memory_order().unwrap();

    let mut acc = Complex64::new(0.0, 0.0);

    for a in 0..n {
        for b in 0..n {
            let g_ba = unsafe {*gs.get_unchecked(b * n + a)};
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
fn dot_product_unroll8_real(mut x: *const f64, mut y: *const f64, n: usize) -> f64 {
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

/// Perform dot product between two complex vectors with unrolled loop of length 8.
/// # Arguments:
/// - `x`: Vector 1.
/// - `y`: Vector 2.
/// - `n`: Vector length.
/// # Returns
/// - `Complex64`: Dot product of the two vectors.
#[inline(always)]
fn dot_product_unroll8_complex(mut x: *const Complex64, mut y: *const Complex64, n: usize) -> Complex64 {
    let mut i = 0usize;

    let mut s0 = Complex64::new(0.0, 0.0);
    let mut s1 = Complex64::new(0.0, 0.0);
    let mut s2 = Complex64::new(0.0, 0.0);
    let mut s3 = Complex64::new(0.0, 0.0);

    unsafe {
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

            s0 += x0 * y0;
            s1 += x1 * y1;
            s2 += x2 * y2;
            s3 += x3 * y3;
            s0 += x4 * y4;
            s1 += x5 * y5;
            s2 += x6 * y6;
            s3 += x7 * y7;

            x = x.add(8);
            y = y.add(8);
            i += 8;
        }

        let mut sum = s0 + s1 + s2 + s3;

        while i < n {
            sum += *x * *y;
            x = x.add(1);
            y = y.add(1);
            i += 1;
        }

        sum
    }
}

/// Perform dot product between a real vector and a complex vector with unrolled loop of length 8.
/// # Arguments:
/// - `x`: Real vector.
/// - `y`: Complex vector.
/// - `n`: Vector length.
/// # Returns
/// - `Complex64`: Dot product of the two vectors.
#[inline(always)]
fn dot_product_unroll8_real_complex(mut x: *const f64, mut y: *const Complex64, n: usize) -> Complex64 {
    let mut i = 0usize;

    let mut s0 = Complex64::new(0.0, 0.0);
    let mut s1 = Complex64::new(0.0, 0.0);
    let mut s2 = Complex64::new(0.0, 0.0);
    let mut s3 = Complex64::new(0.0, 0.0);

    unsafe {
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

            s0 += y0 * x0;
            s1 += y1 * x1;
            s2 += y2 * x2;
            s3 += y3 * x3;
            s0 += y4 * x4;
            s1 += y5 * x5;
            s2 += y6 * x6;
            s3 += y7 * x7;

            x = x.add(8);
            y = y.add(8);
            i += 8;
        }

        let mut sum = s0 + s1 + s2 + s3;

        while i < n {
            sum += *y * *x;
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
                            let dot = dot_product_unroll8_real(tabc_vec_ptr, ht_c_ptr, n);
                            acc = g_ba_ptr.mul_add(dot, acc);
                        }
                    }
                }
            }
            acc
        })
    })
}

/// Calculate Einstein summation of complex matrices `g` and `h` and complex 4D tensor `t` as
/// \sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.
/// Assumes `g`, `h` and `t` all have axes of equal length.
/// # Arguments
/// - `g`: Matrix 1.
/// - `t`: 4D tensor.
/// - `h`: Matrix 2.
/// # Returns
/// - `Complex64`: Contracted scalar.
pub fn einsum_ba_abcd_cd_complex(g: &Array2<Complex64>, t: &Array4<Complex64>, h: &Array2<Complex64>) -> Complex64 {
    let n = g.nrows();

    let gs = g.as_slice_memory_order().unwrap();
    let hs = h.as_slice_memory_order().unwrap();
    let ts = t.as_slice_memory_order().unwrap();

    let mut acc = Complex64::new(0.0, 0.0);

    CHT_SCRATCH.with(|hbuf| {
        CGT_SCRATCH.with(|gbuf| {
            let mut ht = hbuf.borrow_mut();
            let mut gt = gbuf.borrow_mut();

            ht.resize(n * n, Complex64::new(0.0, 0.0));
            gt.resize(n * n, Complex64::new(0.0, 0.0));

            for d in 0..n {
                for c in 0..n {
                    unsafe {*ht.get_unchecked_mut(c * n + d) = *hs.get_unchecked(c * n + d);}
                }
            }

            for b in 0..n {
                let b_idx = b * n;
                for a in 0..n {
                    unsafe {*gt.get_unchecked_mut(a * n + b) = *gs.get_unchecked(b_idx + a);}
                }
            }

            unsafe {
                let ts_ptr = ts.as_ptr();
                let ht_ptr = ht.as_ptr();
                let gt_ptr = gt.as_ptr();

                for a in 0..n {
                    let ta_index = a * n * n * n;
                    let gt_a_ptr = gt_ptr.add(a * n);

                    for b in 0..n {
                        let g_ba = *gt_a_ptr.add(b);
                        if g_ba.norm() == 0.0 {continue;}

                        let tab_index = ta_index + b * n * n;

                        for c in 0..n {
                            let ht_c_ptr = ht_ptr.add(c * n);
                            let tabc_vec_ptr = ts_ptr.add(tab_index + c * n);
                            let dot = dot_product_unroll8_complex(tabc_vec_ptr, ht_c_ptr, n);
                            acc += g_ba * dot;
                        }
                    }
                }
            }

            acc
        })
    })
}

/// Calculate Einstein summation of complex matrices `g` and `h` and real 4D tensor `t` as
/// \sum_{a,b}\sum_{c,d} g_{b,a} t_{a,b,c,d} h_{c,d}.
/// Assumes `g`, `h` and `t` all have axes of equal length.
/// # Arguments
/// - `g`: Matrix 1.
/// - `t`: Real 4D tensor.
/// - `h`: Matrix 2.
/// # Returns
/// - `Complex64`: Contracted scalar.
pub fn einsum_ba_abcd_cd_complex_real(g: &Array2<Complex64>, t: &Array4<f64>, h: &Array2<Complex64>) -> Complex64 {
    let n = g.nrows();

    let gs = g.as_slice_memory_order().unwrap();
    let hs = h.as_slice_memory_order().unwrap();
    let ts = t.as_slice_memory_order().unwrap();

    let mut acc = Complex64::new(0.0, 0.0);

    CHT_SCRATCH.with(|hbuf| {
        CGT_SCRATCH.with(|gbuf| {
            let mut ht = hbuf.borrow_mut();
            let mut gt = gbuf.borrow_mut();

            ht.resize(n * n, Complex64::new(0.0, 0.0));
            gt.resize(n * n, Complex64::new(0.0, 0.0));

            for d in 0..n {
                for c in 0..n {
                    unsafe {*ht.get_unchecked_mut(c * n + d) = *hs.get_unchecked(c * n + d);}
                }
            }

            for b in 0..n {
                let b_idx = b * n;
                for a in 0..n {
                    unsafe {*gt.get_unchecked_mut(a * n + b) = *gs.get_unchecked(b_idx + a);}
                }
            }

            unsafe {
                let ts_ptr = ts.as_ptr();
                let ht_ptr = ht.as_ptr();
                let gt_ptr = gt.as_ptr();

                for a in 0..n {
                    let ta_index = a * n * n * n;
                    let gt_a_ptr = gt_ptr.add(a * n);

                    for b in 0..n {
                        let g_ba = *gt_a_ptr.add(b);
                        if g_ba.norm() == 0.0 {continue;}

                        let tab_index = ta_index + b * n * n;

                        for c in 0..n {
                            let ht_c_ptr = ht_ptr.add(c * n);
                            let tabc_vec_ptr = ts_ptr.add(tab_index + c * n);
                            let dot = dot_product_unroll8_real_complex(tabc_vec_ptr, ht_c_ptr, n);
                            acc += g_ba * dot;
                        }
                    }
                }
            }

            acc
        })
    })
}
