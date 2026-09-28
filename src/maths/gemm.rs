// maths/gemm.rs
//! Register-blocked matrix products that read one operand in place.
//!
//! `C = A B` is evaluated with `A` addressed through a row-offset table and a column-offset table,
//! so any strided view of a larger tensor is used without copying it, and with `B` packed into
//! panels of register width. Tensor contractions routinely multiply a large reference tensor by a
//! small operand; copying the large operand into a matrix layout for every product costs as much
//! memory traffic as the arithmetic, so it is read where it lies and only the small operand is
//! packed.
//!
//! The tile kernel is generic over the packed type and instantiated for AVX-512 (`F64x8`) and
//! AVX2/FMA (`F64x4`), with a scalar fallback; the widest kernel the CPU supports is chosen at
//! run time. Every tile keeps most of the vector registers as accumulators: wide products use few
//! rows and many columns, and narrow ones trade columns for rows.

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::cell::RefCell;

// Parent/sibling imports.
#[cfg(target_arch = "x86_64")]
use super::simd::{F64x4, F64x8, Simd};

/// Summed indices per block, so one packed panel of `B` stays in the L1 cache.
#[cfg(target_arch = "x86_64")]
const KC: usize = 256;

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// Packed panels of the small operand, reused by every product on this thread.
    static PANELS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Compute `C = A B` with `A_{ik} = a[r_i + c_k]`, `B` a contiguous row-major `k \times n` matrix and
/// `C` a contiguous row-major `m \times n` matrix, which is overwritten, using the widest kernel the
/// CPU supports.
/// # Arguments:
/// - `a`: Data of `A`.
/// - `rows`: Offset `r_i` of every row of `A`.
/// - `cols`: Offset `c_k` of every column of `A`.
/// - `b`: Row-major `B`.
/// - `n`: Number of columns of `B` and `C`.
/// - `c`: Row-major `C`, overwritten.
/// # Returns:
/// - `()`: Writes `C`.
/// # Safety
/// - Every `r_i + c_k` must index `a`, `b` must hold `k n` elements and `c` must hold `m n`.
pub(crate) unsafe fn strided_gemm(
    a: &[f64],
    rows: &[usize],
    cols: &[usize],
    b: &[f64],
    n: usize,
    c: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: Each kernel runs only when the CPU supports its features, and the caller
        // guarantees the index ranges.
        if is_x86_feature_detected!("avx512f") {
            unsafe { strided_gemm_f64x8(a, rows, cols, b, n, c) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { strided_gemm_f64x4(a, rows, cols, b, n, c) };
            return;
        }
    }

    strided_gemm_scalar(a, rows, cols, b, n, c);
}

/// Compute `C = A B` with the AVX-512 tile kernel, choosing the tile shape from the width of `B`.
/// # Arguments:
/// - `a`: Data of `A`.
/// - `rows`: Offset of every row of `A`.
/// - `cols`: Offset of every column of `A`.
/// - `b`: Row-major `B`.
/// - `n`: Number of columns of `B` and `C`.
/// - `c`: Row-major `C`, overwritten.
/// # Returns:
/// - `()`: Writes `C`.
/// # Safety
/// - The CPU must support AVX-512F, and the index ranges must hold as for `strided_gemm`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn strided_gemm_f64x8(
    a: &[f64],
    rows: &[usize],
    cols: &[usize],
    b: &[f64],
    n: usize,
    c: &mut [f64],
) {
    // SAFETY: The caller guarantees AVX-512F and the index ranges.
    unsafe {
        if n > 16 {
            gemm_tiles::<F64x8, 8, 8, 3>(a, rows, cols, b, n, c);
        } else if n > 8 {
            gemm_tiles::<F64x8, 8, 12, 2>(a, rows, cols, b, n, c);
        } else {
            gemm_tiles::<F64x8, 8, 24, 1>(a, rows, cols, b, n, c);
        }
    }
}

/// Compute `C = A B` with the AVX2/FMA tile kernel, choosing the tile shape from the width of `B`.
/// # Arguments:
/// - `a`: Data of `A`.
/// - `rows`: Offset of every row of `A`.
/// - `cols`: Offset of every column of `A`.
/// - `b`: Row-major `B`.
/// - `n`: Number of columns of `B` and `C`.
/// - `c`: Row-major `C`, overwritten.
/// # Returns:
/// - `()`: Writes `C`.
/// # Safety
/// - The CPU must support AVX2 and FMA, and the index ranges must hold as for `strided_gemm`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn strided_gemm_f64x4(
    a: &[f64],
    rows: &[usize],
    cols: &[usize],
    b: &[f64],
    n: usize,
    c: &mut [f64],
) {
    // SAFETY: The caller guarantees AVX2, FMA and the index ranges.
    unsafe {
        if n > 8 {
            gemm_tiles::<F64x4, 4, 4, 3>(a, rows, cols, b, n, c);
        } else if n > 4 {
            gemm_tiles::<F64x4, 4, 6, 2>(a, rows, cols, b, n, c);
        } else {
            gemm_tiles::<F64x4, 4, 12, 1>(a, rows, cols, b, n, c);
        }
    }
}

/// Compute `C = A B` with scalar arithmetic, `C_{ij} = \sum_k A_{ik} B_{kj}`, one row of `A` at a
/// time.
/// # Arguments:
/// - `a`: Data of `A`.
/// - `rows`: Offset of every row of `A`.
/// - `cols`: Offset of every column of `A`.
/// - `b`: Row-major `B`.
/// - `n`: Number of columns of `B` and `C`.
/// - `c`: Row-major `C`, overwritten.
/// # Returns:
/// - `()`: Writes `C`.
/// # Panics
/// - Panics if an offset lies outside `a` or the matrices are shorter than their shapes.
fn strided_gemm_scalar(
    a: &[f64],
    rows: &[usize],
    cols: &[usize],
    b: &[f64],
    n: usize,
    c: &mut [f64],
) {
    for (&row, out) in rows.iter().zip(c.chunks_mut(n)) {
        out.fill(0.0);
        for (&col, bk) in cols.iter().zip(b.chunks(n)) {
            let x = a[row + col];
            for (o, &y) in out.iter_mut().zip(bk) {
                *o += x * y;
            }
        }
    }
}

/// Run `C = A B` over `MR \times N NV` tiles: pack `B` into panels of `N NV` columns, then sweep
/// every row block of `A` over every panel.
/// # Arguments:
/// - `a`: Data of `A`.
/// - `rows`: Offset of every row of `A`.
/// - `cols`: Offset of every column of `A`.
/// - `b`: Row-major `B`.
/// - `n`: Number of columns of `B` and `C`.
/// - `c`: Row-major `C`, overwritten.
/// # Returns:
/// - `()`: Writes `C`.
/// # Safety
/// - The CPU must support the instructions of `V`, and the index ranges must hold as for
///   `strided_gemm`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn gemm_tiles<V: Simd<N, Scalar = f64>, const N: usize, const MR: usize, const NV: usize>(
    a: &[f64],
    rows: &[usize],
    cols: &[usize],
    b: &[f64],
    n: usize,
    c: &mut [f64],
) {
    let (m, k) = (rows.len(), cols.len());
    let nr = N * NV;
    let panels = n.div_ceil(nr);

    // Sweep blocks of `KC` summed indices, so one packed panel stays in the L1 cache while every
    // row block passes over it and the later blocks accumulate into `C`.
    PANELS.with_borrow_mut(|packed| {
        for k0 in (0..k).step_by(KC) {
            let kc = KC.min(k - k0);

            // Pack this block of `B` as `[panel][kc][N NV]`, padding the last panel with zeros.
            packed.clear();
            packed.resize(panels * kc * nr, 0.0);
            for p in 0..panels {
                let width = nr.min(n - p * nr);
                for kk in 0..kc {
                    let src = (k0 + kk) * n + p * nr;
                    let dst = (p * kc + kk) * nr;
                    packed[dst..dst + width].copy_from_slice(&b[src..src + width]);
                }
            }

            // Every row block against every panel; rows past `m` repeat the first row and are
            // dropped.
            for i0 in (0..m).step_by(MR) {
                let mr = MR.min(m - i0);
                let mut offsets = [rows[i0]; MR];
                offsets[..mr].copy_from_slice(&rows[i0..i0 + mr]);
                for p in 0..panels {
                    let width = nr.min(n - p * nr);
                    // SAFETY: The caller guarantees the feature set and the index ranges.
                    unsafe {
                        gemm_tile::<V, N, MR, NV>(
                            a,
                            &offsets,
                            &cols[k0..k0 + kc],
                            &packed[p * kc * nr..(p + 1) * kc * nr],
                            (c, n, i0, p * nr),
                            (mr, width, k0 > 0),
                        );
                    }
                }
            }
        }
    });
}

/// Accumulate one `MR \times N NV` tile of `C = A B` in registers and store its valid part,
/// `C_{ij} = \sum_k A_{ik} B_{kj}`.
/// # Arguments:
/// - `a`: Data of `A`.
/// - `offsets`: Offset of each of the tile's rows of `A`.
/// - `cols`: Offset of every column of `A`.
/// - `panel`: Packed `k \times N NV` panel of `B`.
/// - `out`: Output matrix, its row length, and the first row and column of the tile.
/// - `valid`: Number of valid rows and columns of the tile, and whether to add to `C`.
/// # Returns:
/// - `()`: Writes the tile into `C`.
/// # Safety
/// - The CPU must support the instructions of `V`, and the index ranges must hold as for
///   `strided_gemm`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn gemm_tile<V: Simd<N, Scalar = f64>, const N: usize, const MR: usize, const NV: usize>(
    a: &[f64],
    offsets: &[usize; MR],
    cols: &[usize],
    panel: &[f64],
    out: (&mut [f64], usize, usize, usize),
    valid: (usize, usize, bool),
) {
    let (c, ldc, i0, j0) = out;
    let (mr, width, accumulate) = valid;
    let nr = N * NV;

    // SAFETY: Every load index lies inside `a` or `panel` by the caller's guarantee, and the
    // stores are bounded by the valid rows and columns.
    unsafe {
        let pa = a.as_ptr();
        let pb = panel.as_ptr();
        let rows = offsets.map(|offset| pa.add(offset));

        // Accumulate `MR \times NV` packed columns over the summed index, starting from the
        // valid part of `C` after the first block.
        let mut acc = [[V::zero(); NV]; MR];
        if accumulate {
            for (r, row) in acc.iter_mut().enumerate().take(mr) {
                let mut buffer = [[0.0f64; N]; NV];
                let src = (i0 + r) * ldc + j0;
                buffer.as_flattened_mut()[..width].copy_from_slice(&c[src..src + width]);
                for (x, lanes) in row.iter_mut().zip(&buffer) {
                    *x = V::load(lanes);
                }
            }
        }
        for (kk, &col) in cols.iter().enumerate() {
            let mut bv = [V::zero(); NV];
            for (v, x) in bv.iter_mut().enumerate() {
                *x = V::load(&*(pb.add(kk * nr + N * v) as *const [f64; N]));
            }
            for (row, &start) in acc.iter_mut().zip(&rows) {
                let av = V::splat(*start.add(col));
                for (x, &b) in row.iter_mut().zip(&bv) {
                    *x = V::madd(*x, av, b);
                }
            }
        }

        // Store the valid rows, going through a buffer for a partial last panel.
        for (r, row) in acc.iter().enumerate().take(mr) {
            let dst = (i0 + r) * ldc + j0;
            if width == nr {
                for (v, x) in row.iter().enumerate() {
                    x.store(&mut *(c.as_mut_ptr().add(dst + N * v) as *mut [f64; N]));
                }
            } else {
                let mut buffer = [[0.0f64; N]; NV];
                for (x, lanes) in row.iter().zip(buffer.iter_mut()) {
                    x.store(lanes);
                }
                c[dst..dst + width].copy_from_slice(&buffer.as_flattened()[..width]);
            }
        }
    }
}

/// Offsets of every element of a strided index space in row-major order,
/// `\sum_l i_l s_l`.
/// # Arguments:
/// - `layout`: Extent and stride of every axis, outermost first.
/// # Returns:
/// - `Vec<usize>`: Offset of every index tuple.
pub(crate) fn strided_offsets(layout: &[(usize, usize)]) -> Vec<usize> {
    let mut out = vec![0usize];
    for &(d, s) in layout {
        out = out
            .iter()
            .flat_map(|&o| (0..d).map(move |i| o + i * s))
            .collect();
    }
    out
}
