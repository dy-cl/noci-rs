#![allow(clippy::too_many_arguments)]

// maths/wick.rs

// External crate imports.
#[cfg(feature = "nocc")]
use ndarray::Array2;
use ndarray::ArrayView2;

// Crate-root imports.
use crate::StateScalar;
#[cfg(feature = "nocc")]
use crate::noci::NOCIScalar;

/// Calculate a determinant coefficient for an occupation bitstring.
/// # Arguments:
/// - `c`: Orbital coefficient matrix in an orthonormal basis.
/// - `mask`: Occupation bitstring.
/// - `nel`: Number of occupied orbitals.
/// # Returns
/// - `T`: Determinant coefficient for the occupied rows and first `nel` columns.
#[cfg(feature = "nocc")]
pub(crate) fn det_occupied_minor<T: NOCIScalar>(
    c: &Array2<T>,
    mask: u128,
    nel: usize,
) -> T {
    // The selected rows define the occupied Slater determinant minor.
    let mut rows = Vec::with_capacity(nel);

    for p in 0..c.nrows() {
        if ((mask >> p) & 1) == 1 {
            rows.push(p);
        }
    }

    let mut minor = Vec::with_capacity(nel * nel);
    for &r in rows.iter() {
        for col in 0..nel {
            minor.push(c[(r, col)]);
        }
    }

    det(minor.as_slice(), nel).unwrap_or(<T as From<f64>>::from(0.0))
}

/// Build the square `l x l` contraction determinant with `x` elements in the
/// diagonal and lower triangle, and `y` elements in the upper triangle.
/// Dispatches small fixed ranks to the const-generic implementation and larger ranks to the
/// arbitrary-rank implementation.
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
pub fn build_d<T: Copy + 'static>(
    d: &mut [T],
    l: usize,
    x: &ArrayView2<T>,
    y: &ArrayView2<T>,
    rows: &[usize],
    cols: &[usize],
) {
    // Runtime rank dispatches to the same const-rank GNME determinant fill when available.
    match l {
        0 => build_d_const::<T, 0>(d, x, y, rows, cols),
        1 => build_d_const::<T, 1>(d, x, y, rows, cols),
        2 => build_d_const::<T, 2>(d, x, y, rows, cols),
        3 => build_d_const::<T, 3>(d, x, y, rows, cols),
        4 => build_d_const::<T, 4>(d, x, y, rows, cols),
        _ => build_d::build_d_gen(d, l, x, y, rows, cols),
    }
}

/// Construct a contraction determinant with compile-time excitation rank `L`.
/// `D_{ij} = X_{r_i c_j}` for `i >= j`, and `D_{ij} = Y_{r_i c_j}` for `i < j`.
/// # Arguments:
/// - `d`: Row-major `L x L` determinant storage to write.
/// - `x`: Lower-triangle and diagonal contraction matrix `X`.
/// - `y`: Upper-triangle contraction matrix `Y`.
/// - `rows`: Row labels `r_i`.
/// - `cols`: Column labels `c_j`.
/// # Returns
/// - `()`: Writes the contraction determinant into `d`.
#[inline(always)]
pub fn build_d_const<T: Copy + 'static, const L: usize>(
    d: &mut [T],
    x: &ArrayView2<T>,
    y: &ArrayView2<T>,
    rows: &[usize],
    cols: &[usize],
) {
    // Build the GNME contraction determinant: X fills the diagonal/lower triangle
    // and Y fills the upper triangle for the ordered labels (r_i,c_j).
    build_d::build_d_const::<T, L>(d, x, y, rows, cols);
}

mod build_d {
    // Standard library imports.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    use std::any::TypeId;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    use std::arch::x86_64::{_mm256_i64gather_pd, _mm256_set1_epi64x};
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    use std::arch::x86_64::{_mm256_maskstore_pd, _mm256_set_epi64x};

    // External crate imports.
    use ndarray::ArrayView2;

    /// Construct contraction determinant for compile-time excitation rank `L`.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `x`: X matrix elements.
    /// - `y`: Y matrix elements.
    /// - `rows`: Row indices of X or Y.
    /// - `cols`: Column indices of X or Y.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn build_d_const<T: Copy + 'static, const L: usize>(
        d: &mut [T],
        x: &ArrayView2<T>,
        y: &ArrayView2<T>,
        rows: &[usize],
        cols: &[usize],
    ) {
        let xstr = x.strides();
        let ystr = y.strides();
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();

        unsafe {
            for i in 0..L {
                let r = *rows.get_unchecked(i) as isize;
                let xr = r * xstr[0];
                let yr = r * ystr[0];
                let base = i * L;

                for j in 0..L {
                    let c = *cols.get_unchecked(j) as isize;
                    let value = if i >= j {
                        *xptr.offset(xr + c * xstr[1])
                    } else {
                        *yptr.offset(yr + c * ystr[1])
                    };
                    *d.get_unchecked_mut(base + j) = value;
                }
            }
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
    pub(super) fn build_d_gen<T: Copy + 'static>(
        d: &mut [T],
        l: usize,
        x: &ArrayView2<T>,
        y: &ArrayView2<T>,
        rows: &[usize],
        cols: &[usize],
    ) {
        // Generic rank keeps the same D_ov convention as the const path.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            unsafe {
                let xstr = x.strides();
                let ystr = y.strides();
                let xptr = x.as_ptr().cast::<f64>();
                let yptr = y.as_ptr().cast::<f64>();
                let dptr = d.as_mut_ptr().cast::<f64>();

                for (i, &row) in rows.iter().take(l).enumerate() {
                    let r = row as i64;
                    let base = i * l;
                    let mut j = 0usize;

                    while j <= i {
                        let count = (i + 1 - j).min(4);
                        let mut idx = [0i64; 4];

                        for lane in 0..count {
                            idx[lane] = r * xstr[0] as i64 + cols[j + lane] as i64 * xstr[1] as i64;
                        }
                        for lane in count..4 {
                            idx[lane] = idx[count - 1];
                        }

                        let indices = _mm256_set_epi64x(idx[3], idx[2], idx[1], idx[0]);
                        let values = _mm256_i64gather_pd(xptr, indices, 8);
                        let mask = match count {
                            1 => _mm256_set_epi64x(0, 0, 0, -1),
                            2 => _mm256_set_epi64x(0, 0, -1, -1),
                            3 => _mm256_set_epi64x(0, -1, -1, -1),
                            _ => _mm256_set1_epi64x(-1),
                        };

                        _mm256_maskstore_pd(dptr.add(base + j), mask, values);
                        j += count;
                    }

                    j = i + 1;
                    while j < l {
                        let count = (l - j).min(4);
                        let mut idx = [0i64; 4];

                        for lane in 0..count {
                            idx[lane] = r * ystr[0] as i64 + cols[j + lane] as i64 * ystr[1] as i64;
                        }
                        for lane in count..4 {
                            idx[lane] = idx[count - 1];
                        }

                        let indices = _mm256_set_epi64x(idx[3], idx[2], idx[1], idx[0]);
                        let values = _mm256_i64gather_pd(yptr, indices, 8);
                        let mask = match count {
                            1 => _mm256_set_epi64x(0, 0, 0, -1),
                            2 => _mm256_set_epi64x(0, 0, -1, -1),
                            3 => _mm256_set_epi64x(0, -1, -1, -1),
                            _ => _mm256_set1_epi64x(-1),
                        };

                        _mm256_maskstore_pd(dptr.add(base + j), mask, values);
                        j += count;
                    }
                }

                return;
            }
        }

        let xstr = x.strides();
        let ystr = y.strides();
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();

        unsafe {
            for i in 0..l {
                let r = *rows.get_unchecked(i) as isize;

                let xr = r * xstr[0];
                let yr = r * ystr[0];
                let base = i * l;

                for j in 0..=i {
                    let c = *cols.get_unchecked(j) as isize;
                    d[base + j] = *xptr.offset(xr + c * xstr[1]);
                }

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
pub fn mix_columns<T: Copy + 'static>(
    d: &mut [T],
    det0: &[T],
    det1: &[T],
    l: usize,
    bits: u64,
) {
    // Runtime rank dispatch for one mixed D_ov distribution in the zero-overlap sum.
    match l {
        0 => mix_columns_const::<T, 0>(d, det0, det1, bits),
        1 => mix_columns_const::<T, 1>(d, det0, det1, bits),
        2 => mix_columns_const::<T, 2>(d, det0, det1, bits),
        3 => mix_columns_const::<T, 3>(d, det0, det1, bits),
        4 => mix_columns_const::<T, 4>(d, det0, det1, bits),
        _ => mix_columns::mix_columns_gen(d, det0, det1, l, bits),
    }
}

/// Mix columns of `det1` into `det0` with compile-time excitation rank `L`.
/// For column `c`, if bit `c` of `bits` is set then the output column is taken
/// from `det1`; otherwise it is taken from `det0`.
/// # Arguments:
/// - `d`: Matrix to write into.
/// - `det0`: Base matrix.
/// - `det1`: Mixing matrix.
/// - `bits`: Bitstring selecting which columns are taken from `det1`.
/// # Returns
/// - `()`: Writes the mixed matrix into `d`.
#[inline(always)]
pub fn mix_columns_const<T: Copy + 'static, const L: usize>(
    d: &mut [T],
    det0: &[T],
    det1: &[T],
    bits: u64,
) {
    // This builds one mixed-distribution D_ov by choosing each column from det0 or det1.
    mix_columns::mix_columns_const::<T, L>(d, det0, det1, bits);
}

mod mix_columns {
    // Standard library imports.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    use std::any::TypeId;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    use std::arch::x86_64::{
        _mm256_blendv_pd, _mm256_castsi256_pd, _mm256_maskload_pd, _mm256_maskstore_pd,
        _mm256_set_epi64x, _mm256_set1_epi64x,
    };
    /// Mix columns of `det1` into `det0` according to `bits` for compile-time excitation rank `L`.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `det0`: Base matrix.
    /// - `det1`: Mixing matrix.
    /// - `bits`: Bitstring.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn mix_columns_const<T: Copy + 'static, const L: usize>(
        d: &mut [T],
        det0: &[T],
        det1: &[T],
        bits: u64,
    ) {
        // Column bit `c` selects whether m_c = 0 or 1 for this determinant distribution.
        unsafe {
            for i in 0..L {
                let base = i * L;

                for j in 0..L {
                    let idx = base + j;
                    let value = if ((bits >> j) & 1) != 0 {
                        *det1.get_unchecked(idx)
                    } else {
                        *det0.get_unchecked(idx)
                    };
                    *d.get_unchecked_mut(idx) = value;
                }
            }
        }
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
    pub(super) fn mix_columns_gen<T: Copy + 'static>(
        d: &mut [T],
        det0: &[T],
        det1: &[T],
        l: usize,
        bits: u64,
    ) {
        // Generic mixed determinant builder for ranks outside the fixed small kernels.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            unsafe {
                let dptr = d.as_mut_ptr().cast::<f64>();
                let p0 = det0.as_ptr().cast::<f64>();
                let p1 = det1.as_ptr().cast::<f64>();

                for r in 0..l {
                    let base = r * l;
                    let mut c = 0usize;

                    while c < l {
                        let count = (l - c).min(4);
                        let valid = match count {
                            1 => _mm256_set_epi64x(0, 0, 0, -1),
                            2 => _mm256_set_epi64x(0, 0, -1, -1),
                            3 => _mm256_set_epi64x(0, -1, -1, -1),
                            _ => _mm256_set1_epi64x(-1),
                        };

                        let mut selected = [0i64; 4];

                        for (lane, value) in selected.iter_mut().enumerate().take(count) {
                            *value = if ((bits >> (c + lane)) & 1) != 0 {
                                -1
                            } else {
                                0
                            };
                        }

                        let select = _mm256_castsi256_pd(_mm256_set_epi64x(
                            selected[3],
                            selected[2],
                            selected[1],
                            selected[0],
                        ));
                        let v0 = _mm256_maskload_pd(p0.add(base + c), valid);
                        let v1 = _mm256_maskload_pd(p1.add(base + c), valid);
                        let mixed = _mm256_blendv_pd(v0, v1, select);

                        _mm256_maskstore_pd(dptr.add(base + c), valid, mixed);
                        c += count;
                    }
                }

                return;
            }
        }

        unsafe {
            for r in 0..l {
                let base = r * l;
                for c in 0..l {
                    let k = base + c;
                    let use1 = ((bits >> c) & 1) != 0;
                    *d.get_unchecked_mut(k) = if use1 {
                        *det1.get_unchecked(k)
                    } else {
                        *det0.get_unchecked(k)
                    };
                }
            }
        }
    }
}

/// Construct the minor of an `L x L` matrix obtained by removing row `r_rm` and column `c_rm`.
/// The resulting matrix is `M^{(r_rm,c_rm)}` with dimension `(L - 1) x (L - 1)`.
/// Small fixed ranks use the const-generic implementation and larger ranks use the
/// arbitrary-rank implementation.
/// # Arguments:
/// - `out`: Row-major matrix minor to write.
/// - `m`: Input row-major matrix `M`.
/// - `l`: Matrix rank `L`.
/// - `r_rm`: Row index removed from `M`.
/// - `c_rm`: Column index removed from `M`.
/// # Returns
/// - `()`: Writes `M^{(r_rm,c_rm)}` into `out`.
#[inline(always)]
pub fn minor<T: Copy + 'static>(
    out: &mut [T],
    m: &[T],
    l: usize,
    r_rm: usize,
    c_rm: usize,
) {
    // Runtime rank dispatches to const-rank minor extraction for the small cofactor cases.
    match l {
        0 => minor_const::<T, 0>(out, m, r_rm, c_rm),
        1 => minor_const::<T, 1>(out, m, r_rm, c_rm),
        2 => minor_const::<T, 2>(out, m, r_rm, c_rm),
        3 => minor_const::<T, 3>(out, m, r_rm, c_rm),
        4 => minor_const::<T, 4>(out, m, r_rm, c_rm),
        _ => minor_mod::minor_gen(out, m, l, r_rm, c_rm),
    }
}

/// Construct a matrix minor with compile-time matrix rank `L`.
/// The result is `M^{(r_rm,c_rm)}` with dimension `(L - 1) x (L - 1)`.
/// # Arguments:
/// - `out`: Row-major matrix minor to write.
/// - `m`: Input row-major `L x L` matrix `M`.
/// - `r_rm`: Row index removed from `M`.
/// - `c_rm`: Column index removed from `M`.
/// # Returns
/// - `()`: Writes `M^{(r_rm,c_rm)}` into `out`.
#[inline(always)]
pub fn minor_const<T: Copy + 'static, const L: usize>(
    out: &mut [T],
    m: &[T],
    r_rm: usize,
    c_rm: usize,
) {
    // Const-rank path deletes row `r_rm` and column `c_rm` from one D_ov minor.
    minor_mod::minor_const::<T, L>(out, m, r_rm, c_rm);
}

mod minor_mod {
    // Standard library imports.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    use std::any::TypeId;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    use std::arch::x86_64::{
        _mm256_i64gather_pd, _mm256_maskstore_pd, _mm256_set_epi64x, _mm256_set1_epi64x,
    };
    /// Construct the matrix minor for compile-time rank `L`.
    /// For `L = 1`, the rank-zero minor contains no stored elements. For `L = 2,3,4`,
    /// this uses the same fixed-rank arithmetic as the former rank-specific kernels.
    /// # Arguments:
    /// - `out`: Row-major matrix minor to write.
    /// - `m`: Input row-major `L x L` matrix `M`.
    /// - `r_rm`: Row index removed from `M`.
    /// - `c_rm`: Column index removed from `M`.
    /// # Returns
    /// - `()`: Writes `M^{(r_rm,c_rm)}` into `out`.
    #[inline(always)]
    pub(super) fn minor_const<T: Copy + 'static, const L: usize>(
        out: &mut [T],
        m: &[T],
        r_rm: usize,
        c_rm: usize,
    ) {
        if L <= 1 {
            return;
        }

        // Copy D_ov[eta, z] into its cofactor minor D_ov[r_rm|c_rm].
        let mut ii = 0usize;
        for i in 0..L {
            if i == r_rm {
                continue;
            }

            let mut jj = 0usize;
            for j in 0..L {
                if j == c_rm {
                    continue;
                }

                unsafe {
                    let src = i * L + j;
                    let dst = ii * (L - 1) + jj;
                    *out.get_unchecked_mut(dst) = *m.get_unchecked(src);
                }

                jj += 1;
            }

            ii += 1;
        }
    }

    /// Construct the matrix minor for arbitrary matrix rank `L`.
    /// The result is the `(L - 1) x (L - 1)` matrix `M^{(r_rm,c_rm)}`.
    /// The real-valued path gathers groups of up to four retained elements with packed AVX2 operations.
    /// # Arguments:
    /// - `out`: Row-major matrix minor to write.
    /// - `m`: Input row-major `L x L` matrix `M`.
    /// - `l`: Matrix rank `L`.
    /// - `r_rm`: Row index removed from `M`.
    /// - `c_rm`: Column index removed from `M`.
    /// # Returns
    /// - `()`: Writes `M^{(r_rm,c_rm)}` into `out`.
    #[inline(always)]
    pub(super) fn minor_gen<T: Copy + 'static>(
        out: &mut [T],
        m: &[T],
        l: usize,
        r_rm: usize,
        c_rm: usize,
    ) {
        // Generic minor extraction for cofactor and second-minor determinants.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            unsafe {
                let mptr = m.as_ptr().cast::<f64>();
                let optr = out.as_mut_ptr().cast::<f64>();
                let out_l = l - 1;
                let mut retained = Vec::with_capacity(out_l);
                for j in 0..l {
                    if j != c_rm {
                        retained.push(j);
                    }
                }

                let mut ii = 0usize;
                for i in 0..l {
                    if i == r_rm {
                        continue;
                    }

                    let mut jj = 0usize;
                    while jj < out_l {
                        let count = (out_l - jj).min(4);
                        let mut idx = [0i64; 4];

                        for lane in 0..count {
                            idx[lane] = (i * l + retained[jj + lane]) as i64;
                        }
                        for lane in count..4 {
                            idx[lane] = idx[count - 1];
                        }

                        let indices = _mm256_set_epi64x(idx[3], idx[2], idx[1], idx[0]);
                        let values = _mm256_i64gather_pd(mptr, indices, 8);
                        let mask = match count {
                            1 => _mm256_set_epi64x(0, 0, 0, -1),
                            2 => _mm256_set_epi64x(0, 0, -1, -1),
                            3 => _mm256_set_epi64x(0, -1, -1, -1),
                            _ => _mm256_set1_epi64x(-1),
                        };

                        _mm256_maskstore_pd(optr.add(ii * out_l + jj), mask, values);
                        jj += count;
                    }

                    ii += 1;
                }

                return;
            }
        }

        let mut ii = 0usize;
        for i in 0..l {
            if i == r_rm {
                continue;
            }

            let mut jj = 0usize;
            for j in 0..l {
                if j == c_rm {
                    continue;
                }

                out[ii * (l - 1) + jj] = m[i * l + j];
                jj += 1;
            }
            ii += 1;
        }
    }
}

/// Construct the minor `M^{(r_rm,c_rm)}` and compute its determinant and adjugate transpose.
/// If `M` has rank `L`, the minor has rank `L - 1`, with cofactor matrix
/// `C_{ij} = (-1)^{i+j} det((M^{(r_rm,c_rm)})^{(i,j)})`.
/// Fixed-rank kernels are used for `L = 1,...,4`; arbitrary ranks use the general minor and adjugate kernels.
/// # Arguments:
/// - `adjt`: Row-major cofactor matrix of the minor to write.
/// - `minor_out`: Row-major matrix minor `M^{(r_rm,c_rm)}` to write.
/// - `invs`: Scratch inverse singular values used by the generic SVD fallback.
/// - `lu`: Scratch matrix used by the generic LU fallback.
/// - `m`: Input row-major matrix `M`.
/// - `l`: Matrix rank `L`.
/// - `r_rm`: Row index removed from `M`.
/// - `c_rm`: Column index removed from `M`.
/// - `thresh`: Threshold below which a singular value is treated as zero.
/// # Returns
/// - `Option<T>`: Determinant of `M^{(r_rm,c_rm)}`, or `None` if evaluation fails.
#[inline(always)]
pub fn minor_adjugate_transpose<T: StateScalar>(
    adjt: &mut [T],
    minor_out: &mut [T],
    invs: &mut [f64],
    lu: &mut [T],
    m: &[T],
    l: usize,
    r_rm: usize,
    c_rm: usize,
    thresh: f64,
) -> Option<T> {
    // Compute det M[r_rm|c_rm] and its cofactor matrix for second-minor Wick contractions.
    match l {
        0 => None,
        1 => minor_adjt::minor_adjt_const::<T, 1, 0>(adjt, minor_out, m, r_rm, c_rm),
        2 => minor_adjt::minor_adjt_const::<T, 2, 1>(adjt, minor_out, m, r_rm, c_rm),
        3 => minor_adjt::minor_adjt_const::<T, 3, 2>(adjt, minor_out, m, r_rm, c_rm),
        4 => minor_adjt::minor_adjt_const::<T, 4, 3>(adjt, minor_out, m, r_rm, c_rm),
        _ => minor_adjt::minor_adjt_gen(adjt, minor_out, invs, lu, m, l, r_rm, c_rm, thresh),
    }
}

mod minor_adjt {
    // Crate-root imports.
    use crate::StateScalar;

    /// Construct a fixed-rank minor and compute its determinant and adjugate transpose.
    /// `L` is the source matrix rank and `M` is the minor rank, so `M = L - 1`.
    /// # Arguments:
    /// - `adjt`: Row-major minor cofactor matrix to write.
    /// - `minor_out`: Row-major minor matrix to write.
    /// - `m`: Input row-major `L x L` matrix.
    /// - `r_rm`: Row index removed from `M`.
    /// - `c_rm`: Column index removed from `M`.
    /// # Returns
    /// - `Option<T>`: Determinant of the minor, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn minor_adjt_const<T: StateScalar, const L: usize, const M: usize>(
        adjt: &mut [T],
        minor_out: &mut [T],
        m: &[T],
        r_rm: usize,
        c_rm: usize,
    ) -> Option<T> {
        // Form M[r_rm|c_rm], then compute its determinant and cofactor matrix.
        if M == 0 {
            Some(T::from_real(1.0))
        } else {
            super::minor_const::<T, L>(minor_out, m, r_rm, c_rm);
            super::adjt_mod::adjt_const::<T, M>(adjt, minor_out)
        }
    }

    /// Construct a minor of arbitrary rank and compute its determinant and adjugate transpose.
    /// The minor is constructed explicitly before the generic adjugate evaluator is applied.
    /// # Arguments:
    /// - `adjt`: Row-major cofactor matrix of the minor to write.
    /// - `minor_out`: Row-major minor matrix to write.
    /// - `invs`: Scratch inverse singular values used by the SVD fallback.
    /// - `lu`: Scratch matrix used by the LU fallback.
    /// - `m`: Input row-major matrix `M`.
    /// - `l`: Matrix rank `L`.
    /// - `r_rm`: Row index removed from `M`.
    /// - `c_rm`: Column index removed from `M`.
    /// - `thresh`: Threshold below which a singular value is treated as zero.
    /// # Returns
    /// - `Option<T>`: Determinant of the minor, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn minor_adjt_gen<T: StateScalar>(
        adjt: &mut [T],
        minor_out: &mut [T],
        invs: &mut [f64],
        lu: &mut [T],
        m: &[T],
        l: usize,
        r_rm: usize,
        c_rm: usize,
        thresh: f64,
    ) -> Option<T> {
        // Generic second-minor path used when fixed-rank cofactor formulas do not apply.
        super::minor_mod::minor_gen(minor_out, m, l, r_rm, c_rm);
        super::adjugate_transpose(adjt, invs, lu, minor_out, l - 1, thresh)
    }
}

/// Compute the determinant of an `n x n` matrix using explicit formulas for
/// small sizes and a generic fallback for larger matrices.
/// # Arguments:
/// - `a`: Matrix stored in row-major order.
/// - `n`: Matrix dimension.
/// # Returns
/// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
#[inline(always)]
pub fn det<T: StateScalar>(
    a: &[T],
    n: usize,
) -> Option<T> {
    if a.len() != n * n {
        return None;
    }

    // Runtime rank dispatches to fixed formulas for small determinants and generic LU otherwise.
    match n {
        0 => det_const::<T, 0>(a),
        1 => det_const::<T, 1>(a),
        2 => det_const::<T, 2>(a),
        3 => det_const::<T, 3>(a),
        4 => det_const::<T, 4>(a),
        _ => det_mod::det_gen(a, n),
    }
}

/// Compute a determinant with compile-time matrix rank `N`.
/// `\det A` follows the same fixed-rank arithmetic used by `det(a, n)`.
/// # Arguments:
/// - `a`: Matrix stored in row-major order.
/// # Returns
/// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
#[inline(always)]
pub fn det_const<T: StateScalar, const N: usize>(a: &[T]) -> Option<T> {
    if a.len() != N * N {
        return None;
    }

    // Fixed ranks use explicit determinant algebra for the Wick kernels' small minors.
    match N {
        0 => Some(T::from_real(1.0)),
        1 => Some(det_mod::det_const::<T, 1>(a)),
        2 => {
            let d = det_mod::det_const::<T, 2>(a);
            if d.abs().is_finite() { Some(d) } else { None }
        }
        3 => {
            let d = det_mod::det_const::<T, 3>(a);
            if d.abs().is_finite() { Some(d) } else { None }
        }
        4 => {
            let d = det_mod::det_const::<T, 4>(a);
            if d.abs().is_finite() { Some(d) } else { None }
        }
        5 => {
            let mut lu = [T::from_real(0.0); 25];
            lu.copy_from_slice(a);
            det_lu_fixed::<T, 5, 25>(&mut lu)
        }
        6 => {
            let mut lu = [T::from_real(0.0); 36];
            lu.copy_from_slice(a);
            det_lu_fixed::<T, 6, 36>(&mut lu)
        }
        _ => det_mod::det_gen(a, N),
    }
}

/// Compute a determinant with partial-pivot LU for a fixed small matrix size.
/// The const rank lets the compiler simplify hot determinant kernels without changing memory use.
/// # Arguments:
/// - `lu`: Mutable row-major matrix storage overwritten with LU factors.
/// # Returns
/// - `Option<T>`: Determinant of `lu`, or `None` if evaluation produces non-finite values.
#[inline(always)]
fn det_lu_fixed<T: StateScalar, const N: usize, const S: usize>(lu: &mut [T; S]) -> Option<T> {
    // In-place partial-pivot LU gives the same determinant as the generic fallback for N > 4.
    let mut sign = 1.0;
    let mut k = 0usize;

    while k < N {
        let mut pivot = k;
        let mut pivot_abs = lu[k * N + k].abs();

        if !pivot_abs.is_finite() {
            return None;
        }

        let mut r = k + 1;
        while r < N {
            let abs = lu[r * N + k].abs();
            if !abs.is_finite() {
                return None;
            }
            if abs > pivot_abs {
                pivot = r;
                pivot_abs = abs;
            }
            r += 1;
        }

        if pivot_abs == 0.0 {
            return Some(T::from_real(0.0));
        }

        if pivot != k {
            let mut c = 0usize;
            while c < N {
                lu.swap(k * N + c, pivot * N + c);
                c += 1;
            }
            sign = -sign;
        }

        let pivot_value = lu[k * N + k];
        r = k + 1;
        while r < N {
            let factor = lu[r * N + k] / pivot_value;
            lu[r * N + k] = factor;

            let mut c = k + 1;
            while c < N {
                lu[r * N + c] -= factor * lu[k * N + c];
                c += 1;
            }
            r += 1;
        }

        k += 1;
    }

    let mut det = T::from_real(sign);
    let mut i = 0usize;
    while i < N {
        det *= lu[i * N + i];
        i += 1;
    }

    if det.abs().is_finite() {
        Some(det)
    } else {
        None
    }
}

mod det_mod {
    // Standard library imports.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    use std::any::TypeId;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    use std::arch::x86_64::{
        _mm_add_sd, _mm_cvtsd_f64, _mm256_castpd256_pd128, _mm256_extractf128_pd, _mm256_fmadd_pd,
        _mm256_fmsub_pd, _mm256_hadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set_pd,
    };

    // External crate imports.
    use ndarray::ArrayView2;
    use ndarray_linalg::{Determinant, SVD};

    // Crate-root imports.
    use crate::StateScalar;

    /// Calculate determinant of a fixed-rank matrix.
    /// `\det A` uses the same fixed-rank formulas as the former rank-specific kernels.
    /// # Arguments:
    /// - `a`: Matrix to calculate the determinant of.
    /// # Returns
    /// - `T`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det_const<T: StateScalar, const N: usize>(a: &[T]) -> T {
        // Explicit small-N determinant formulas are used by fixed-rank Wick minors.
        match N {
            1 => a[0],
            2 => {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    unsafe {
                        let a = std::slice::from_raw_parts(a.as_ptr().cast::<f64>(), 4);
                        let lhs0 = _mm256_set_pd(0.0, 0.0, 0.0, a[0]);
                        let rhs0 = _mm256_set_pd(0.0, 0.0, 0.0, a[3]);
                        let lhs1 = _mm256_set_pd(0.0, 0.0, 0.0, a[1]);
                        let rhs1 = _mm256_set_pd(0.0, 0.0, 0.0, a[2]);
                        let values = _mm256_fmsub_pd(lhs0, rhs0, _mm256_mul_pd(lhs1, rhs1));
                        let det = _mm_cvtsd_f64(_mm256_castpd256_pd128(values));

                        return T::from_real(det);
                    }
                }

                det2scalar(a[0], a[1], a[2], a[3])
            }
            3 => {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    unsafe {
                        let a = std::slice::from_raw_parts(a.as_ptr().cast::<f64>(), 9);

                        let x0 = _mm256_set_pd(0.0, a[3], a[5], a[4]);
                        let y0 = _mm256_set_pd(0.0, a[7], a[6], a[8]);
                        let x1 = _mm256_set_pd(0.0, a[4], a[3], a[5]);
                        let y1 = _mm256_set_pd(0.0, a[6], a[8], a[7]);
                        let cof = _mm256_fmsub_pd(x0, y0, _mm256_mul_pd(x1, y1));

                        let row = _mm256_set_pd(0.0, a[2], a[1], a[0]);
                        let products = _mm256_mul_pd(row, cof);
                        let sums = _mm256_hadd_pd(products, products);
                        let low = _mm256_castpd256_pd128(sums);
                        let high = _mm256_extractf128_pd(sums, 1);
                        let det = _mm_cvtsd_f64(_mm_add_sd(low, high));

                        return T::from_real(det);
                    }
                }

                let a00 = a[0];
                let a01 = a[1];
                let a02 = a[2];
                let a10 = a[3];
                let a11 = a[4];
                let a12 = a[5];
                let a20 = a[6];
                let a21 = a[7];
                let a22 = a[8];

                det3scalar(a00, a01, a02, a10, a11, a12, a20, a21, a22)
            }
            4 => {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    unsafe {
                        let a = std::slice::from_raw_parts(a.as_ptr().cast::<f64>(), 16);

                        let x = [a[4], a[5], a[6], a[7]];
                        let y = [a[8], a[9], a[10], a[11]];
                        let z = [a[12], a[13], a[14], a[15]];

                        let x0 = _mm256_set_pd(x[0], x[0], x[0], x[1]);
                        let x1 = _mm256_set_pd(x[1], x[1], x[2], x[2]);
                        let x2 = _mm256_set_pd(x[2], x[3], x[3], x[3]);

                        let y0 = _mm256_set_pd(y[0], y[0], y[0], y[1]);
                        let y1 = _mm256_set_pd(y[1], y[1], y[2], y[2]);
                        let y2 = _mm256_set_pd(y[2], y[3], y[3], y[3]);

                        let z0 = _mm256_set_pd(z[0], z[0], z[0], z[1]);
                        let z1 = _mm256_set_pd(z[1], z[1], z[2], z[2]);
                        let z2 = _mm256_set_pd(z[2], z[3], z[3], z[3]);

                        let m0 = _mm256_fmsub_pd(y1, z2, _mm256_mul_pd(y2, z1));
                        let m1 = _mm256_fmsub_pd(y0, z2, _mm256_mul_pd(y2, z0));
                        let m2 = _mm256_fmsub_pd(y0, z1, _mm256_mul_pd(y1, z0));

                        let minors01 = _mm256_fmsub_pd(x0, m0, _mm256_mul_pd(x1, m1));
                        let minors = _mm256_fmadd_pd(x2, m2, minors01);
                        let cof = _mm256_mul_pd(minors, _mm256_set_pd(-1.0, 1.0, -1.0, 1.0));
                        let row = _mm256_loadu_pd(a.as_ptr());
                        let products = _mm256_mul_pd(row, cof);
                        let sums = _mm256_hadd_pd(products, products);
                        let low = _mm256_castpd256_pd128(sums);
                        let high = _mm256_extractf128_pd(sums, 1);
                        let det = _mm_cvtsd_f64(_mm_add_sd(low, high));

                        return T::from_real(det);
                    }
                }

                let m00 = {
                    let a11 = a[5];
                    let a12 = a[6];
                    let a13 = a[7];
                    let a21 = a[9];
                    let a22 = a[10];
                    let a23 = a[11];
                    let a31 = a[13];
                    let a32 = a[14];
                    let a33 = a[15];
                    a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
                        + a13 * (a21 * a32 - a22 * a31)
                };

                let m01 = {
                    let a10 = a[4];
                    let a12 = a[6];
                    let a13 = a[7];
                    let a20 = a[8];
                    let a22 = a[10];
                    let a23 = a[11];
                    let a30 = a[12];
                    let a32 = a[14];
                    let a33 = a[15];
                    a10 * (a22 * a33 - a23 * a32) - a12 * (a20 * a33 - a23 * a30)
                        + a13 * (a20 * a32 - a22 * a30)
                };

                let m02 = {
                    let a10 = a[4];
                    let a11 = a[5];
                    let a13 = a[7];
                    let a20 = a[8];
                    let a21 = a[9];
                    let a23 = a[11];
                    let a30 = a[12];
                    let a31 = a[13];
                    let a33 = a[15];
                    a10 * (a21 * a33 - a23 * a31) - a11 * (a20 * a33 - a23 * a30)
                        + a13 * (a20 * a31 - a21 * a30)
                };

                let m03 = {
                    let a10 = a[4];
                    let a11 = a[5];
                    let a12 = a[6];
                    let a20 = a[8];
                    let a21 = a[9];
                    let a22 = a[10];
                    let a30 = a[12];
                    let a31 = a[13];
                    let a32 = a[14];
                    a10 * (a21 * a32 - a22 * a31) - a11 * (a20 * a32 - a22 * a30)
                        + a12 * (a20 * a31 - a21 * a30)
                };

                a[0] * m00 - a[1] * m01 + a[2] * m02 - a[3] * m03
            }
            _ => unreachable!(),
        }
    }

    /// Calculate determinant of a 2 x 2 matrix from scalar elements.
    /// # Arguments:
    /// - `a00`: Matrix element (0, 0).
    /// - `a01`: Matrix element (0, 1).
    /// - `a10`: Matrix element (1, 0).
    /// - `a11`: Matrix element (1, 1).
    /// # Returns
    /// - `T`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det2scalar<T: StateScalar>(
        a00: T,
        a01: T,
        a10: T,
        a11: T,
    ) -> T {
        // det [[a00,a01],[a10,a11]].
        a00 * a11 - a01 * a10
    }

    /// Calculate determinant of a 3 x 3 matrix from scalar elements.
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
    /// - `T`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det3scalar<T: StateScalar>(
        a00: T,
        a01: T,
        a02: T,
        a10: T,
        a11: T,
        a12: T,
        a20: T,
        a21: T,
        a22: T,
    ) -> T {
        // First-row Laplace expansion of the 3 x 3 determinant.
        a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20)
    }

    /// Compute determinant of `a` for arbitrary size using LU factorisation first and SVD as a fallback.
    /// # Arguments:
    /// - `a`: Matrix to find determinant of.
    /// - `n`: Matrix dimension.
    /// # Returns
    /// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn det_gen<T: StateScalar>(
        a: &[T],
        n: usize,
    ) -> Option<T> {
        // Ranks five and six use stack LU for hot Wick kernels; larger ranks use LAPACK.
        match n {
            5 => {
                let mut lu = [T::from_real(0.0); 25];
                lu.copy_from_slice(&a[..25]);
                if let Some(d) = super::det_lu_fixed::<T, 5, 25>(&mut lu) {
                    return Some(d);
                }
            }
            6 => {
                let mut lu = [T::from_real(0.0); 36];
                lu.copy_from_slice(&a[..36]);
                if let Some(d) = super::det_lu_fixed::<T, 6, 36>(&mut lu) {
                    return Some(d);
                }
            }
            7 => {
                let mut lu = [T::from_real(0.0); 49];
                lu.copy_from_slice(&a[..49]);
                if let Some(d) = super::det_lu_fixed::<T, 7, 49>(&mut lu) {
                    return Some(d);
                }
            }
            8 => {
                let mut lu = [T::from_real(0.0); 64];
                lu.copy_from_slice(&a[..64]);
                if let Some(d) = super::det_lu_fixed::<T, 8, 64>(&mut lu) {
                    return Some(d);
                }
            }
            _ => {
                let mut lu = a[..n * n].to_vec();
                if let Some(d) = det_lu_in_place(&mut lu, n) {
                    return Some(d);
                }
            }
        }

        let av = ArrayView2::from_shape((n, n), a).ok()?;
        let (u_opt, s, vt_opt) = av.svd(true, true).ok()?;
        let u = u_opt?;
        let vt = vt_opt?;

        let det_u = u.det().ok()?;
        let det_vt = vt.det().ok()?;

        let mut det = det_u * det_vt;
        for &si in s.iter() {
            det *= T::from_real(si);
        }

        if det.abs().is_finite() {
            Some(det)
        } else {
            None
        }
    }

    /// Compute a determinant with partial-pivot LU in caller-provided row-major storage.
    /// This avoids calling LAPACK for the many tiny determinants in Wick overlap evaluation.
    /// # Arguments:
    /// - `lu`: Mutable row-major matrix storage overwritten with LU factors.
    /// - `n`: Matrix dimension.
    /// # Returns
    /// - `Option<T>`: Determinant of `lu`, or `None` if evaluation produces non-finite values.
    #[inline(always)]
    fn det_lu_in_place<T: StateScalar>(
        lu: &mut [T],
        n: usize,
    ) -> Option<T> {
        // Partial-pivot LU accumulates det(P) times the diagonal product of U.
        let mut sign = 1.0;

        for k in 0..n {
            let mut pivot = k;
            let mut pivot_abs = lu[k * n + k].abs();

            if !pivot_abs.is_finite() {
                return None;
            }

            for r in (k + 1)..n {
                let abs = lu[r * n + k].abs();
                if !abs.is_finite() {
                    return None;
                }
                if abs > pivot_abs {
                    pivot = r;
                    pivot_abs = abs;
                }
            }

            if pivot_abs == 0.0 {
                return Some(T::from_real(0.0));
            }

            if pivot != k {
                for c in 0..n {
                    lu.swap(k * n + c, pivot * n + c);
                }
                sign = -sign;
            }

            let pivot_value = lu[k * n + k];
            for r in (k + 1)..n {
                let factor = lu[r * n + k] / pivot_value;
                lu[r * n + k] = factor;

                for c in (k + 1)..n {
                    lu[r * n + c] = -factor * lu[k * n + c];
                }
            }
        }

        let mut det = T::from_real(sign);
        for i in 0..n {
            det *= lu[i * n + i];
        }

        if det.abs().is_finite() {
            Some(det)
        } else {
            None
        }
    }
}

/// Compute the determinant and adjugate transpose of an `n x n` matrix using
/// explicit formulas for small sizes and generic LU/SVD-based methods for larger matrices.
/// # Arguments:
/// - `adjt`: Scratch space for the adjugate transpose.
/// - `invs`: Scratch space for inverse singular values used by the SVD fallback.
/// - `lu`: Scratch space used by the LU-based fallback.
/// - `a`: Input matrix stored in row-major order.
/// - `n`: Matrix dimension.
/// - `thresh`: Threshold below which singular values are treated as zero in the SVD fallback.
/// # Returns
/// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
#[inline(always)]
pub fn adjugate_transpose<T: StateScalar>(
    adjt: &mut [T],
    invs: &mut [f64],
    lu: &mut [T],
    a: &[T],
    n: usize,
    thresh: f64,
) -> Option<T> {
    // Runtime rank dispatches to the fixed cofactor formulas used by one-column replacements.
    match n {
        0 => adjugate_transpose_const::<T, 0>(adjt, invs, lu, a, thresh),
        1 => adjugate_transpose_const::<T, 1>(adjt, invs, lu, a, thresh),
        2 => adjugate_transpose_const::<T, 2>(adjt, invs, lu, a, thresh),
        3 => adjugate_transpose_const::<T, 3>(adjt, invs, lu, a, thresh),
        4 => adjugate_transpose_const::<T, 4>(adjt, invs, lu, a, thresh),
        _ => adjt_mod::adjt_gen(adjt, invs, lu, a, n, thresh),
    }
}

/// Compute `\det A` and the cofactor matrix of `A` with compile-time matrix rank `N`.
/// The cofactor matrix is stored row-major as
/// `\operatorname{cof}[A]_{rc}=(-1)^{r+c}\det A[r|c]`. This uses the same fixed-rank
/// arithmetic as `adjugate_transpose(adjt, invs, lu, a, n, thresh)`.
/// # Arguments:
/// - `adjt`: Output row-major cofactor matrix `\operatorname{cof}[A]`.
/// - `invs`: Scratch inverse singular values used by the generic fallback.
/// - `lu`: Scratch matrix used by the generic fallback.
/// - `a`: Row-major `N x N` matrix `A`.
/// - `thresh`: Threshold below which a singular value is treated as zero in the generic fallback.
/// # Returns
/// - `Option<T>`: Determinant `\det A`, or `None` if evaluation fails.
#[inline(always)]
pub fn adjugate_transpose_const<T: StateScalar, const N: usize>(
    adjt: &mut [T],
    invs: &mut [f64],
    lu: &mut [T],
    a: &[T],
    thresh: f64,
) -> Option<T> {
    // The fixed-rank adjugate path returns det A and cof[A]_{rc}=(-1)^{r+c} det A[r|c],
    // which is the cofactor form used by one-column Wick replacement determinants.
    match N {
        0 => adjt_mod::adjt_const::<T, 0>(adjt, a),
        1 => adjt_mod::adjt_const::<T, 1>(adjt, a),
        2 => adjt_mod::adjt_const::<T, 2>(adjt, a),
        3 => adjt_mod::adjt_const::<T, 3>(adjt, a),
        4 => adjt_mod::adjt_const::<T, 4>(adjt, a),
        _ => adjt_mod::adjt_gen(adjt, invs, lu, a, N, thresh),
    }
}

mod adjt_mod {
    // Standard library imports.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    use std::any::TypeId;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    use std::arch::x86_64::{
        _mm_add_sd, _mm_cvtsd_f64, _mm256_castpd256_pd128, _mm256_extractf128_pd, _mm256_fmadd_pd,
        _mm256_fmsub_pd, _mm256_hadd_pd, _mm256_loadu_pd, _mm256_maskstore_pd, _mm256_mul_pd,
        _mm256_set_epi64x, _mm256_set_pd, _mm256_storeu_pd,
    };

    // External crate imports.
    use ndarray::{Array2, ArrayView2};
    use ndarray_linalg::{Determinant, FactorizeInto, InverseInto, SVD};

    // Crate-root imports.
    use crate::StateScalar;

    // Parent/sibling imports.
    use super::det_mod::{det2scalar, det3scalar};

    /// Calculate determinant and adjugate transpose for compile-time matrix rank `N`.
    /// `\operatorname{cof}[A]_{rc}=(-1)^{r+c}\det A[r|c]`.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `a`: Input row-major `N x N` matrix.
    /// # Returns
    /// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt_const<T: StateScalar, const N: usize>(
        adjt: &mut [T],
        a: &[T],
    ) -> Option<T> {
        // Evaluate the cofactor matrix explicitly for small N; higher ranks fall back to
        // the generic adjugate implementation with the same cof[A] convention.
        match N {
            0 => Some(T::from_real(1.0)),
            1 => {
                adjt[0] = T::from_real(1.0);
                Some(a[0])
            }
            2 => {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    unsafe {
                        let a = std::slice::from_raw_parts(a.as_ptr().cast::<f64>(), 4);
                        let adjt =
                            std::slice::from_raw_parts_mut(adjt.as_mut_ptr().cast::<f64>(), 4);

                        let lhs0 = _mm256_set_pd(0.0, 0.0, 0.0, a[0]);
                        let rhs0 = _mm256_set_pd(0.0, 0.0, 0.0, a[3]);
                        let lhs1 = _mm256_set_pd(0.0, 0.0, 0.0, a[1]);
                        let rhs1 = _mm256_set_pd(0.0, 0.0, 0.0, a[2]);
                        let values = _mm256_fmsub_pd(lhs0, rhs0, _mm256_mul_pd(lhs1, rhs1));
                        let det = _mm_cvtsd_f64(_mm256_castpd256_pd128(values));

                        if !det.abs().is_finite() {
                            return None;
                        }

                        let cof = _mm256_set_pd(a[0], -a[1], -a[2], a[3]);
                        _mm256_storeu_pd(adjt.as_mut_ptr(), cof);

                        return Some(T::from_real(det));
                    }
                }

                let a00 = a[0];
                let a01 = a[1];
                let a10 = a[2];
                let a11 = a[3];

                let det = det2scalar(a00, a01, a10, a11);
                if !det.abs().is_finite() {
                    return None;
                }

                adjt[0] = a11;
                adjt[1] = T::from_real(-1.0) * a10;
                adjt[2] = T::from_real(-1.0) * a01;
                adjt[3] = a00;

                Some(det)
            }
            3 => {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    unsafe {
                        let a = std::slice::from_raw_parts(a.as_ptr().cast::<f64>(), 9);
                        let adjt =
                            std::slice::from_raw_parts_mut(adjt.as_mut_ptr().cast::<f64>(), 9);
                        let valid = _mm256_set_epi64x(0, -1, -1, -1);

                        let r0 = [a[0], a[1], a[2]];
                        let r1 = [a[3], a[4], a[5]];
                        let r2 = [a[6], a[7], a[8]];

                        let p00 = _mm256_set_pd(0.0, r1[0], r1[2], r1[1]);
                        let q00 = _mm256_set_pd(0.0, r2[1], r2[0], r2[2]);
                        let p01 = _mm256_set_pd(0.0, r1[1], r1[0], r1[2]);
                        let q01 = _mm256_set_pd(0.0, r2[0], r2[2], r2[1]);
                        let c0 = _mm256_fmsub_pd(p00, q00, _mm256_mul_pd(p01, q01));

                        _mm256_maskstore_pd(adjt.as_mut_ptr(), valid, c0);

                        let row0 = _mm256_set_pd(0.0, r0[2], r0[1], r0[0]);
                        let products = _mm256_mul_pd(row0, c0);
                        let sums = _mm256_hadd_pd(products, products);
                        let low = _mm256_castpd256_pd128(sums);
                        let high = _mm256_extractf128_pd(sums, 1);
                        let det = _mm_cvtsd_f64(_mm_add_sd(low, high));

                        if !det.abs().is_finite() {
                            return None;
                        }

                        let p10 = _mm256_set_pd(0.0, r0[0], r0[2], r0[1]);
                        let q10 = _mm256_set_pd(0.0, r2[1], r2[0], r2[2]);
                        let p11 = _mm256_set_pd(0.0, r0[1], r0[0], r0[2]);
                        let q11 = _mm256_set_pd(0.0, r2[0], r2[2], r2[1]);
                        let c1 = _mm256_fmsub_pd(p11, q11, _mm256_mul_pd(p10, q10));

                        _mm256_maskstore_pd(adjt.as_mut_ptr().add(3), valid, c1);

                        let p20 = _mm256_set_pd(0.0, r0[0], r0[2], r0[1]);
                        let q20 = _mm256_set_pd(0.0, r1[1], r1[0], r1[2]);
                        let p21 = _mm256_set_pd(0.0, r0[1], r0[0], r0[2]);
                        let q21 = _mm256_set_pd(0.0, r1[0], r1[2], r1[1]);
                        let c2 = _mm256_fmsub_pd(p20, q20, _mm256_mul_pd(p21, q21));

                        _mm256_maskstore_pd(adjt.as_mut_ptr().add(6), valid, c2);

                        return Some(T::from_real(det));
                    }
                }

                let a00 = a[0];
                let a01 = a[1];
                let a02 = a[2];
                let a10 = a[3];
                let a11 = a[4];
                let a12 = a[5];
                let a20 = a[6];
                let a21 = a[7];
                let a22 = a[8];

                let det = det3scalar(a00, a01, a02, a10, a11, a12, a20, a21, a22);
                if !det.abs().is_finite() {
                    return None;
                }

                let neg = T::from_real(-1.0);

                let c00 = det2scalar(a11, a12, a21, a22);
                let c01 = neg * det2scalar(a10, a12, a20, a22);
                let c02 = det2scalar(a10, a11, a20, a21);

                let c10 = neg * det2scalar(a01, a02, a21, a22);
                let c11 = det2scalar(a00, a02, a20, a22);
                let c12 = neg * det2scalar(a00, a01, a20, a21);

                let c20 = det2scalar(a01, a02, a11, a12);
                let c21 = neg * det2scalar(a00, a02, a10, a12);
                let c22 = det2scalar(a00, a01, a10, a11);

                adjt[0] = c00;
                adjt[1] = c01;
                adjt[2] = c02;
                adjt[3] = c10;
                adjt[4] = c11;
                adjt[5] = c12;
                adjt[6] = c20;
                adjt[7] = c21;
                adjt[8] = c22;

                Some(det)
            }
            4 => {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    unsafe {
                        let a = std::slice::from_raw_parts(a.as_ptr().cast::<f64>(), 16);
                        let adjt =
                            std::slice::from_raw_parts_mut(adjt.as_mut_ptr().cast::<f64>(), 16);

                        let r0 = [a[0], a[1], a[2], a[3]];
                        let r1 = [a[4], a[5], a[6], a[7]];
                        let r2 = [a[8], a[9], a[10], a[11]];
                        let r3 = [a[12], a[13], a[14], a[15]];

                        let x0 = _mm256_set_pd(r1[0], r1[0], r1[0], r1[1]);
                        let x1 = _mm256_set_pd(r1[1], r1[1], r1[2], r1[2]);
                        let x2 = _mm256_set_pd(r1[2], r1[3], r1[3], r1[3]);
                        let y0 = _mm256_set_pd(r2[0], r2[0], r2[0], r2[1]);
                        let y1 = _mm256_set_pd(r2[1], r2[1], r2[2], r2[2]);
                        let y2 = _mm256_set_pd(r2[2], r2[3], r2[3], r2[3]);
                        let z0 = _mm256_set_pd(r3[0], r3[0], r3[0], r3[1]);
                        let z1 = _mm256_set_pd(r3[1], r3[1], r3[2], r3[2]);
                        let z2 = _mm256_set_pd(r3[2], r3[3], r3[3], r3[3]);

                        let m0 = _mm256_fmsub_pd(y1, z2, _mm256_mul_pd(y2, z1));
                        let m1 = _mm256_fmsub_pd(y0, z2, _mm256_mul_pd(y2, z0));
                        let m2 = _mm256_fmsub_pd(y0, z1, _mm256_mul_pd(y1, z0));
                        let minors01 = _mm256_fmsub_pd(x0, m0, _mm256_mul_pd(x1, m1));
                        let minors = _mm256_fmadd_pd(x2, m2, minors01);
                        let c0 = _mm256_mul_pd(minors, _mm256_set_pd(-1.0, 1.0, -1.0, 1.0));

                        _mm256_storeu_pd(adjt.as_mut_ptr(), c0);

                        let products = _mm256_mul_pd(_mm256_loadu_pd(a.as_ptr()), c0);
                        let sums = _mm256_hadd_pd(products, products);
                        let low = _mm256_castpd256_pd128(sums);
                        let high = _mm256_extractf128_pd(sums, 1);
                        let det = _mm_cvtsd_f64(_mm_add_sd(low, high));

                        if !det.abs().is_finite() {
                            return None;
                        }

                        let x0 = _mm256_set_pd(r0[0], r0[0], r0[0], r0[1]);
                        let x1 = _mm256_set_pd(r0[1], r0[1], r0[2], r0[2]);
                        let x2 = _mm256_set_pd(r0[2], r0[3], r0[3], r0[3]);
                        let y0 = _mm256_set_pd(r2[0], r2[0], r2[0], r2[1]);
                        let y1 = _mm256_set_pd(r2[1], r2[1], r2[2], r2[2]);
                        let y2 = _mm256_set_pd(r2[2], r2[3], r2[3], r2[3]);
                        let z0 = _mm256_set_pd(r3[0], r3[0], r3[0], r3[1]);
                        let z1 = _mm256_set_pd(r3[1], r3[1], r3[2], r3[2]);
                        let z2 = _mm256_set_pd(r3[2], r3[3], r3[3], r3[3]);

                        let m0 = _mm256_fmsub_pd(y1, z2, _mm256_mul_pd(y2, z1));
                        let m1 = _mm256_fmsub_pd(y0, z2, _mm256_mul_pd(y2, z0));
                        let m2 = _mm256_fmsub_pd(y0, z1, _mm256_mul_pd(y1, z0));
                        let minors01 = _mm256_fmsub_pd(x0, m0, _mm256_mul_pd(x1, m1));
                        let minors = _mm256_fmadd_pd(x2, m2, minors01);
                        let c1 = _mm256_mul_pd(minors, _mm256_set_pd(1.0, -1.0, 1.0, -1.0));

                        _mm256_storeu_pd(adjt.as_mut_ptr().add(4), c1);

                        let x0 = _mm256_set_pd(r0[0], r0[0], r0[0], r0[1]);
                        let x1 = _mm256_set_pd(r0[1], r0[1], r0[2], r0[2]);
                        let x2 = _mm256_set_pd(r0[2], r0[3], r0[3], r0[3]);
                        let y0 = _mm256_set_pd(r1[0], r1[0], r1[0], r1[1]);
                        let y1 = _mm256_set_pd(r1[1], r1[1], r1[2], r1[2]);
                        let y2 = _mm256_set_pd(r1[2], r1[3], r1[3], r1[3]);
                        let z0 = _mm256_set_pd(r3[0], r3[0], r3[0], r3[1]);
                        let z1 = _mm256_set_pd(r3[1], r3[1], r3[2], r3[2]);
                        let z2 = _mm256_set_pd(r3[2], r3[3], r3[3], r3[3]);

                        let m0 = _mm256_fmsub_pd(y1, z2, _mm256_mul_pd(y2, z1));
                        let m1 = _mm256_fmsub_pd(y0, z2, _mm256_mul_pd(y2, z0));
                        let m2 = _mm256_fmsub_pd(y0, z1, _mm256_mul_pd(y1, z0));
                        let minors01 = _mm256_fmsub_pd(x0, m0, _mm256_mul_pd(x1, m1));
                        let minors = _mm256_fmadd_pd(x2, m2, minors01);
                        let c2 = _mm256_mul_pd(minors, _mm256_set_pd(-1.0, 1.0, -1.0, 1.0));

                        _mm256_storeu_pd(adjt.as_mut_ptr().add(8), c2);

                        let x0 = _mm256_set_pd(r0[0], r0[0], r0[0], r0[1]);
                        let x1 = _mm256_set_pd(r0[1], r0[1], r0[2], r0[2]);
                        let x2 = _mm256_set_pd(r0[2], r0[3], r0[3], r0[3]);
                        let y0 = _mm256_set_pd(r1[0], r1[0], r1[0], r1[1]);
                        let y1 = _mm256_set_pd(r1[1], r1[1], r1[2], r1[2]);
                        let y2 = _mm256_set_pd(r1[2], r1[3], r1[3], r1[3]);
                        let z0 = _mm256_set_pd(r2[0], r2[0], r2[0], r2[1]);
                        let z1 = _mm256_set_pd(r2[1], r2[1], r2[2], r2[2]);
                        let z2 = _mm256_set_pd(r2[2], r2[3], r2[3], r2[3]);

                        let m0 = _mm256_fmsub_pd(y1, z2, _mm256_mul_pd(y2, z1));
                        let m1 = _mm256_fmsub_pd(y0, z2, _mm256_mul_pd(y2, z0));
                        let m2 = _mm256_fmsub_pd(y0, z1, _mm256_mul_pd(y1, z0));
                        let minors01 = _mm256_fmsub_pd(x0, m0, _mm256_mul_pd(x1, m1));
                        let minors = _mm256_fmadd_pd(x2, m2, minors01);
                        let c3 = _mm256_mul_pd(minors, _mm256_set_pd(1.0, -1.0, 1.0, -1.0));

                        _mm256_storeu_pd(adjt.as_mut_ptr().add(12), c3);

                        return Some(T::from_real(det));
                    }
                }

                let a00 = a[0];
                let a01 = a[1];
                let a02 = a[2];
                let a03 = a[3];
                let a10 = a[4];
                let a11 = a[5];
                let a12 = a[6];
                let a13 = a[7];
                let a20 = a[8];
                let a21 = a[9];
                let a22 = a[10];
                let a23 = a[11];
                let a30 = a[12];
                let a31 = a[13];
                let a32 = a[14];
                let a33 = a[15];

                let neg = T::from_real(-1.0);

                let c00 = det3scalar(a11, a12, a13, a21, a22, a23, a31, a32, a33);
                let c01 = neg * det3scalar(a10, a12, a13, a20, a22, a23, a30, a32, a33);
                let c02 = det3scalar(a10, a11, a13, a20, a21, a23, a30, a31, a33);
                let c03 = neg * det3scalar(a10, a11, a12, a20, a21, a22, a30, a31, a32);

                let det = a00 * c00 + a01 * c01 + a02 * c02 + a03 * c03;
                if !det.abs().is_finite() {
                    return None;
                }

                let c10 = neg * det3scalar(a01, a02, a03, a21, a22, a23, a31, a32, a33);
                let c11 = det3scalar(a00, a02, a03, a20, a22, a23, a30, a32, a33);
                let c12 = neg * det3scalar(a00, a01, a03, a20, a21, a23, a30, a31, a33);
                let c13 = det3scalar(a00, a01, a02, a20, a21, a22, a30, a31, a32);

                let c20 = det3scalar(a01, a02, a03, a11, a12, a13, a31, a32, a33);
                let c21 = neg * det3scalar(a00, a02, a03, a10, a12, a13, a30, a32, a33);
                let c22 = det3scalar(a00, a01, a03, a10, a11, a13, a30, a31, a33);
                let c23 = neg * det3scalar(a00, a01, a02, a10, a11, a12, a30, a31, a32);

                let c30 = neg * det3scalar(a01, a02, a03, a11, a12, a13, a21, a22, a23);
                let c31 = det3scalar(a00, a02, a03, a10, a12, a13, a20, a22, a23);
                let c32 = neg * det3scalar(a00, a01, a03, a10, a11, a13, a20, a21, a23);
                let c33 = det3scalar(a00, a01, a02, a10, a11, a12, a20, a21, a22);

                adjt[0] = c00;
                adjt[1] = c01;
                adjt[2] = c02;
                adjt[3] = c03;
                adjt[4] = c10;
                adjt[5] = c11;
                adjt[6] = c12;
                adjt[7] = c13;
                adjt[8] = c20;
                adjt[9] = c21;
                adjt[10] = c22;
                adjt[11] = c23;
                adjt[12] = c30;
                adjt[13] = c31;
                adjt[14] = c32;
                adjt[15] = c33;

                Some(det)
            }
            _ => unreachable!(),
        }
    }

    /// Compute determinant and adjugate transpose using LU first and SVD as fallback.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `invs`: Scratch space for inverse singular values.
    /// - `lu`: Scratch space for LU factorisation.
    /// - `a`: Input matrix.
    /// - `n`: Matrix dimension.
    /// - `thresh`: Threshold below which singular values are treated as zero.
    /// # Returns
    /// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt_gen<T: StateScalar>(
        adjt: &mut [T],
        invs: &mut [f64],
        lu: &mut [T],
        a: &[T],
        n: usize,
        thresh: f64,
    ) -> Option<T> {
        // Prefer LU for speed, falling back to SVD when LU cannot supply stable cofactors.
        if let Some(det) = adjtlu(adjt, lu, a, n) {
            return Some(det);
        }
        adjtsvd(adjt, invs, a, n, thresh)
    }

    /// Compute determinant and adjugate transpose using LU factorisation.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `lu`: Scratch space for LU factorisation.
    /// - `a`: Input matrix.
    /// - `n`: Matrix dimension.
    /// # Returns
    /// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjtlu<T: StateScalar>(
        adjt: &mut [T],
        lu: &mut [T],
        a: &[T],
        n: usize,
    ) -> Option<T> {
        // adj(A)^T is stored as det(A) A^{-T}, matching the cofactor convention.
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

    /// Compute determinant and adjugate transpose using singular value decomposition.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `invs`: Scratch space for inverse singular values.
    /// - `a`: Input matrix.
    /// - `n`: Matrix dimension.
    /// - `thresh`: Threshold below which singular values are treated as zero.
    /// # Returns
    /// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjtsvd<T: StateScalar>(
        adjt: &mut [T],
        invs: &mut [f64],
        a: &[T],
        n: usize,
        thresh: f64,
    ) -> Option<T> {
        // SVD fallback forms cofactors from det(A) A^{-T} while filtering small singular values.
        let nn = n * n;
        if adjt.len() < nn || invs.len() < n || a.len() < nn {
            return None;
        }

        adjt[..nn].fill(T::from_real(0.0));
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
            det *= T::from_real(si);

            if si.abs() > thresh {
                red_det *= T::from_real(si);
                invs[i] = 1.0 / si;
            } else {
                nzero += 1;
                zerok = i;
            }
        }

        if nzero == 0 {
            for i in 0..n {
                let inv_si = T::from_real(invs[i]);
                for r in 0..n {
                    let ur = u[(r, i)].conj();
                    let scale = inv_si * ur;
                    for c in 0..n {
                        adjt[r * n + c] += scale * vt[(i, c)].conj();
                    }
                }
            }

            for x in &mut adjt[..nn] {
                *x *= det;
            }
        } else if nzero == 1 {
            let k = zerok;
            for r in 0..n {
                let ur = u[(r, k)].conj();
                let scale = red_det * ur;
                for c in 0..n {
                    adjt[r * n + c] = scale * vt[(k, c)].conj();
                }
            }
        } else {
            return Some(det);
        }

        Some(det)
    }
}
