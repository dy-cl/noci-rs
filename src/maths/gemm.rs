// maths/gemm.rs
//! Register-blocked matrix products that read one operand in place.
//!
//! `C = A B` is evaluated with `A` addressed through a row-offset table and a column-offset table,
//! so any strided view of a larger tensor is used without copying it, and with `B` packed into
//! panels of register width. Tensor contractions routinely multiply a large reference tensor by a
//! small operand; copying the large operand into a matrix layout for every product costs as much
//! memory traffic as the arithmetic, so it is read where it lies and only the small operand is
//! packed. Every tile keeps twelve AVX2 accumulators busy: wide products use four rows by twelve
//! columns, and narrow ones trade columns for rows.

// Standard library imports.
use std::arch::x86_64::{
    _mm256_broadcast_sd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd,
};
use std::cell::RefCell;

thread_local! {
    /// Packed panels of the small operand, reused by every product on this thread.
    static PANELS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Return whether the fused multiply-add kernels can run on this CPU.
/// # Arguments:
/// - None.
/// # Returns:
/// - `bool`: Whether AVX2 and FMA are available.
pub(crate) fn strided_gemm_available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Compute `C = A B` with `A_{ik} = a[r_i + c_k]`, `B` a contiguous row-major `k \times n` matrix and
/// `C` a contiguous row-major `m \times n` matrix, which is overwritten.
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
/// - The CPU must support AVX2 and FMA, as reported by `strided_gemm_available`.
/// - Every `r_i + c_k` must index `a`, `b` must hold `k n` elements and `c` must hold `m n`.
pub(crate) unsafe fn strided_gemm(
    a: &[f64],
    rows: &[usize],
    cols: &[usize],
    b: &[f64],
    n: usize,
    c: &mut [f64],
) {
    // SAFETY: The caller guarantees AVX2 and FMA and the index ranges.
    unsafe {
        if n > 8 {
            gemm_tiles::<4, 3>(a, rows, cols, b, n, c);
        } else if n > 4 {
            gemm_tiles::<6, 2>(a, rows, cols, b, n, c);
        } else {
            gemm_tiles::<12, 1>(a, rows, cols, b, n, c);
        }
    }
}

/// Run `C = A B` over `MR \times 4NV` tiles: pack `B` into panels of `4NV` columns, then sweep
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
/// - As for `strided_gemm`.
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_tiles<const MR: usize, const NV: usize>(
    a: &[f64],
    rows: &[usize],
    cols: &[usize],
    b: &[f64],
    n: usize,
    c: &mut [f64],
) {
    let k = cols.len();
    let nr = 4 * NV;
    let panels = n.div_ceil(nr);

    // Pack `B` as `[panel][k][4NV]` into this thread's reused panel storage, padding the last
    // panel with zeros.
    PANELS.with_borrow_mut(|packed| {
        packed.clear();
        packed.resize(panels * k * nr, 0.0);
        for p in 0..panels {
            let width = nr.min(n - p * nr);
            for kk in 0..k {
                let src = &b[kk * n + p * nr..kk * n + p * nr + width];
                let dst = (p * k + kk) * nr;
                packed[dst..dst + width].copy_from_slice(src);
            }
        }
        // SAFETY: The caller guarantees the feature set and the index ranges.
        unsafe { sweep_tiles::<MR, NV>(a, rows, cols, packed, n, c) }
    });
}

/// Sweep every `MR`-row block of `A` over every packed panel of `B`.
/// # Arguments:
/// - `a`: Data of `A`.
/// - `rows`: Offset of every row of `A`.
/// - `cols`: Offset of every column of `A`.
/// - `packed`: Packed panels of `B`.
/// - `n`: Number of columns of `B` and `C`.
/// - `c`: Row-major `C`, overwritten.
/// # Returns:
/// - `()`: Writes `C`.
/// # Safety
/// - As for `strided_gemm`.
#[target_feature(enable = "avx2,fma")]
unsafe fn sweep_tiles<const MR: usize, const NV: usize>(
    a: &[f64],
    rows: &[usize],
    cols: &[usize],
    packed: &[f64],
    n: usize,
    c: &mut [f64],
) {
    let (m, k) = (rows.len(), cols.len());
    let nr = 4 * NV;
    let panels = n.div_ceil(nr);

    // Every row block against every panel; rows past `m` repeat the first row and are dropped.
    for i0 in (0..m).step_by(MR) {
        let mr = MR.min(m - i0);
        let mut offsets = [rows[i0]; MR];
        offsets[..mr].copy_from_slice(&rows[i0..i0 + mr]);
        for p in 0..panels {
            let width = nr.min(n - p * nr);
            // SAFETY: The caller guarantees the feature set and the index ranges.
            unsafe {
                gemm_tile::<MR, NV>(
                    a,
                    &offsets,
                    cols,
                    &packed[p * k * nr..(p + 1) * k * nr],
                    (c, n, i0, p * nr),
                    (mr, width),
                );
            }
        }
    }
}

/// Accumulate one `MR \times 4NV` tile of `C = A B` in registers and store its valid part,
/// `C_{ij} = \sum_k A_{ik} B_{kj}`.
/// # Arguments:
/// - `a`: Data of `A`.
/// - `offsets`: Offset of each of the tile's rows of `A`.
/// - `cols`: Offset of every column of `A`.
/// - `panel`: Packed `k \times 4NV` panel of `B`.
/// - `out`: Output matrix, its row length, and the first row and column of the tile.
/// - `valid`: Number of valid rows and columns of the tile.
/// # Returns:
/// - `()`: Writes the tile into `C`.
/// # Safety
/// - As for `strided_gemm`.
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_tile<const MR: usize, const NV: usize>(
    a: &[f64],
    offsets: &[usize; MR],
    cols: &[usize],
    panel: &[f64],
    out: (&mut [f64], usize, usize, usize),
    valid: (usize, usize),
) {
    let (c, ldc, i0, j0) = out;
    let (mr, width) = valid;
    let nr = 4 * NV;

    // SAFETY: Every load index lies inside `a` or `panel` by the caller's guarantee, and the
    // stores are bounded by the valid rows and columns.
    unsafe {
        let pa = a.as_ptr();
        let pb = panel.as_ptr();

        // Accumulate `MR \times NV` packed columns over the summed index.
        let mut acc = [[_mm256_setzero_pd(); NV]; MR];
        for (kk, &col) in cols.iter().enumerate() {
            let mut bv = [_mm256_setzero_pd(); NV];
            for (v, x) in bv.iter_mut().enumerate() {
                *x = _mm256_loadu_pd(pb.add(kk * nr + 4 * v));
            }
            for (row, &offset) in acc.iter_mut().zip(offsets) {
                let av = _mm256_broadcast_sd(&*pa.add(offset + col));
                for (x, &b) in row.iter_mut().zip(&bv) {
                    *x = _mm256_fmadd_pd(av, b, *x);
                }
            }
        }

        // Store the valid rows, going through a buffer for a partial last panel.
        for (r, row) in acc.iter().enumerate().take(mr) {
            let dst = (i0 + r) * ldc + j0;
            if width == nr {
                for (v, x) in row.iter().enumerate() {
                    _mm256_storeu_pd(c.as_mut_ptr().add(dst + 4 * v), *x);
                }
            } else {
                let mut buffer = [0.0f64; 12];
                for (v, x) in row.iter().enumerate() {
                    _mm256_storeu_pd(buffer.as_mut_ptr().add(4 * v), *x);
                }
                c[dst..dst + width].copy_from_slice(&buffer[..width]);
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
