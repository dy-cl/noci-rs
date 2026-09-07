// maths/wick.rs

// External crate imports.
#[cfg(feature = "nocc")]
use ndarray::Array2;
use ndarray::{ArrayView2, ArrayViewMut2};
use ndarray_linalg::{Determinant, FactorizeInto, InverseInto, SVD};

// Crate-root imports.
#[cfg(target_arch = "x86_64")]
use crate::maths::Simd;
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
    let mut rows = Vec::with_capacity(nel);

    for p in 0..c.nrows() {
        if ((mask >> p) & 1) == 1 {
            rows.push(p);
        }
    }

    let mut matrix = Vec::with_capacity(nel * nel);

    for &row in &rows {
        for col in 0..nel {
            matrix.push(c[(row, col)]);
        }
    }

    det_dynamic(matrix.as_slice(), nel).unwrap_or_else(|| T::from_real(0.0))
}

/// Construct the rank-`L` contraction determinant.
/// `D_{ij} = X_{r_i c_j}` for `i >= j`, and `D_{ij} = Y_{r_i c_j}` for `i < j`.
/// # Arguments:
/// - `d`: Row-major `L x L` determinant storage.
/// - `x`: Lower-triangle and diagonal contraction matrix `X`.
/// - `y`: Upper-triangle contraction matrix `Y`.
/// - `rows`: Row labels `r_i`.
/// - `cols`: Column labels `c_j`.
/// # Returns
/// - `()`: Writes the contraction determinant into `d`.
#[inline(always)]
pub fn build_d_const<T: Copy, const L: usize>(
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
            let row = *rows.get_unchecked(i) as isize;
            let xr = row * xstr[0];
            let yr = row * ystr[0];
            let base = i * L;

            for j in 0..=i {
                let col = *cols.get_unchecked(j) as isize;
                *d.get_unchecked_mut(base + j) = *xptr.offset(xr + col * xstr[1]);
            }

            for j in (i + 1)..L {
                let col = *cols.get_unchecked(j) as isize;
                *d.get_unchecked_mut(base + j) = *yptr.offset(yr + col * ystr[1]);
            }
        }
    }
}

/// Construct a runtime-rank contraction determinant.
/// `D_{ij} = X_{r_i c_j}` for `i >= j`, and `D_{ij} = Y_{r_i c_j}` for `i < j`.
/// # Arguments:
/// - `d`: Row-major determinant storage.
/// - `l`: Runtime determinant rank.
/// - `x`: Lower-triangle and diagonal contraction matrix `X`.
/// - `y`: Upper-triangle contraction matrix `Y`.
/// - `rows`: Row labels.
/// - `cols`: Column labels.
/// # Returns
/// - `()`: Writes the contraction determinant into `d`.
#[inline(always)]
pub fn build_d_dynamic<T: Copy>(
    d: &mut [T],
    l: usize,
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
        for i in 0..l {
            let row = *rows.get_unchecked(i) as isize;
            let xr = row * xstr[0];
            let yr = row * ystr[0];
            let base = i * l;

            for j in 0..=i {
                let col = *cols.get_unchecked(j) as isize;
                *d.get_unchecked_mut(base + j) = *xptr.offset(xr + col * xstr[1]);
            }

            for j in (i + 1)..l {
                let col = *cols.get_unchecked(j) as isize;
                *d.get_unchecked_mut(base + j) = *yptr.offset(yr + col * ystr[1]);
            }
        }
    }
}

/// Select each column of a rank-`L` matrix from `det0` or `det1`.
/// Bit `c` selects the source of column `c`.
/// # Arguments:
/// - `d`: Mixed row-major matrix to write.
/// - `det0`: Matrix supplying columns whose bits are zero.
/// - `det1`: Matrix supplying columns whose bits are one.
/// - `bits`: Packed column-selection bitstring.
/// # Returns
/// - `()`: Writes the mixed matrix into `d`.
#[inline(always)]
pub fn mix_columns_const<T: Copy, const L: usize>(
    d: &mut [T],
    det0: &[T],
    det1: &[T],
    bits: u64,
) {
    unsafe {
        for row in 0..L {
            let base = row * L;

            for col in 0..L {
                let index = base + col;
                *d.get_unchecked_mut(index) = if ((bits >> col) & 1) != 0 {
                    *det1.get_unchecked(index)
                } else {
                    *det0.get_unchecked(index)
                };
            }
        }
    }
}

/// Select each column of a runtime-rank matrix from `det0` or `det1`.
/// Bit `c` selects the source of column `c`.
/// # Arguments:
/// - `d`: Mixed row-major matrix to write.
/// - `det0`: Matrix supplying columns whose bits are zero.
/// - `det1`: Matrix supplying columns whose bits are one.
/// - `l`: Runtime matrix rank.
/// - `bits`: Packed column-selection bitstring.
/// # Returns
/// - `()`: Writes the mixed matrix into `d`.
#[inline(always)]
pub fn mix_columns_dynamic<T: Copy>(
    d: &mut [T],
    det0: &[T],
    det1: &[T],
    l: usize,
    bits: u64,
) {
    unsafe {
        for row in 0..l {
            let base = row * l;

            for col in 0..l {
                let index = base + col;
                *d.get_unchecked_mut(index) = if ((bits >> col) & 1) != 0 {
                    *det1.get_unchecked(index)
                } else {
                    *det0.get_unchecked(index)
                };
            }
        }
    }
}

/// Construct the first minor obtained by deleting one row and one column from an `L x L` matrix.
/// # Arguments:
/// - `out`: Row-major `(L - 1) x (L - 1)` minor storage.
/// - `matrix`: Input row-major `L x L` matrix.
/// - `removed_row`: Removed row.
/// - `removed_col`: Removed column.
/// # Returns
/// - `()`: Writes the first minor into `out`.
#[inline(always)]
pub fn minor_const<T: Copy, const L: usize>(
    out: &mut [T],
    matrix: &[T],
    removed_row: usize,
    removed_col: usize,
) {
    if L <= 1 {
        return;
    }

    let mut minor_row = 0usize;

    for row in 0..L {
        if row == removed_row {
            continue;
        }

        let mut minor_col = 0usize;

        for col in 0..L {
            if col == removed_col {
                continue;
            }

            unsafe {
                *out.get_unchecked_mut(minor_row * (L - 1) + minor_col) =
                    *matrix.get_unchecked(row * L + col);
            }

            minor_col += 1;
        }

        minor_row += 1;
    }
}

/// Construct the first minor obtained by deleting one row and one column from a runtime-rank
/// square matrix.
/// # Arguments:
/// - `out`: Row-major minor storage.
/// - `matrix`: Input row-major square matrix.
/// - `l`: Runtime matrix rank.
/// - `removed_row`: Removed row.
/// - `removed_col`: Removed column.
/// # Returns
/// - `()`: Writes the first minor into `out`.
#[inline(always)]
pub fn minor_dynamic<T: Copy>(
    out: &mut [T],
    matrix: &[T],
    l: usize,
    removed_row: usize,
    removed_col: usize,
) {
    if l <= 1 {
        return;
    }

    let mut minor_row = 0usize;

    for row in 0..l {
        if row == removed_row {
            continue;
        }

        let mut minor_col = 0usize;

        for col in 0..l {
            if col == removed_col {
                continue;
            }

            unsafe {
                *out.get_unchecked_mut(minor_row * (l - 1) + minor_col) =
                    *matrix.get_unchecked(row * l + col);
            }

            minor_col += 1;
        }

        minor_row += 1;
    }
}

/// Construct the second minor obtained by deleting two rows and two columns from an `L x L`
/// matrix.
/// # Arguments:
/// - `out`: Row-major `(L - 2) x (L - 2)` second-minor storage.
/// - `matrix`: Input row-major `L x L` matrix.
/// - `row0`: First removed row.
/// - `row1`: Second removed row.
/// - `col0`: First removed column.
/// - `col1`: Second removed column.
/// # Returns
/// - `()`: Writes the second minor into `out`.
#[inline(always)]
pub fn second_minor_const<T: Copy, const L: usize>(
    out: &mut [T],
    matrix: &[T],
    row0: usize,
    row1: usize,
    col0: usize,
    col1: usize,
) {
    if L <= 2 {
        return;
    }

    let mut minor_row = 0usize;

    for row in 0..L {
        if row == row0 || row == row1 {
            continue;
        }

        let mut minor_col = 0usize;

        for col in 0..L {
            if col == col0 || col == col1 {
                continue;
            }

            unsafe {
                *out.get_unchecked_mut(minor_row * (L - 2) + minor_col) =
                    *matrix.get_unchecked(row * L + col);
            }

            minor_col += 1;
        }

        minor_row += 1;
    }
}

/// Compute `det(A)` for a compile-time `N x N` matrix using the Faddeev-LeVerrier recurrence.
/// # Arguments:
/// - `matrix`: Row-major matrix entries.
/// # Returns
/// - `T`: Determinant of `A`.
#[inline(always)]
pub fn det_const<T: NOCIScalar, const N: usize, const D: usize>(matrix: &[T]) -> T {
    if N == 0 {
        return T::from_real(1.0);
    }

    let zero = T::from_real(0.0);
    let one = T::from_real(1.0);
    let mut b = [zero; D];
    let mut product = [zero; D];

    for i in 0..N {
        b[i * N + i] = one;
    }

    for k in 1..=N {
        product.fill(zero);

        for row in 0..N {
            for col in 0..N {
                let mut value = zero;

                for inner in 0..N {
                    value += matrix[row * N + inner] * b[inner * N + col];
                }

                product[row * N + col] = value;
            }
        }

        let mut trace = zero;

        for i in 0..N {
            trace += product[i * N + i];
        }

        let coefficient = T::from_real(-1.0 / k as f64) * trace;

        if k == N {
            return if (N & 1) == 0 {
                coefficient
            } else {
                -coefficient
            };
        }

        b.copy_from_slice(&product);

        for i in 0..N {
            b[i * N + i] += coefficient;
        }
    }

    unreachable!()
}

/// Compute independent determinants of packed compile-time `N x N` matrices using the
/// Faddeev-LeVerrier recurrence.
/// # Arguments:
/// - `matrix`: Row-major packed matrix entries.
/// # Returns
/// - `V`: Packed determinants, one independent determinant per SIMD lane.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn det_simd_const<V: Simd<LANES>, const LANES: usize, const N: usize, const D: usize>(
    matrix: &[V]
) -> V {
    if N == 0 {
        return V::one();
    }

    let zero = V::zero();
    let one = V::one();
    let mut b = [zero; D];
    let mut product = [zero; D];

    for i in 0..N {
        b[i * N + i] = one;
    }

    for k in 1..=N {
        product.fill(zero);

        for row in 0..N {
            for col in 0..N {
                let mut value = zero;

                for inner in 0..N {
                    value = V::madd(value, matrix[row * N + inner], b[inner * N + col]);
                }

                product[row * N + col] = value;
            }
        }

        let mut trace = zero;

        for i in 0..N {
            trace = V::add(trace, product[i * N + i]);
        }

        let coefficient = V::scale_real(trace, -1.0 / k as f64);

        if k == N {
            return if (N & 1) == 0 {
                coefficient
            } else {
                V::sub(zero, coefficient)
            };
        }

        b.copy_from_slice(&product);

        for i in 0..N {
            let index = i * N + i;
            b[index] = V::add(b[index], coefficient);
        }
    }

    unreachable!()
}

/// Compute `det(A)` and the cofactor matrix of a compile-time `N x N` matrix using the
/// Faddeev-LeVerrier recurrence.
/// The output convention is
/// `cof[A]_{rc} = (-1)^{r+c} det A[r|c] = adj(A)_{cr}`.
/// # Arguments:
/// - `cof`: Row-major cofactor matrix to write.
/// - `matrix`: Row-major matrix entries.
/// # Returns
/// - `T`: Determinant of `A`.
#[inline(always)]
pub fn adjugate_transpose_const<T: NOCIScalar, const N: usize, const D: usize>(
    cof: &mut [T],
    matrix: &[T],
) -> T {
    if N == 0 {
        return T::from_real(1.0);
    }

    let zero = T::from_real(0.0);
    let one = T::from_real(1.0);
    let mut b = [zero; D];
    let mut product = [zero; D];

    for i in 0..N {
        b[i * N + i] = one;
    }

    if N == 1 {
        cof[0] = one;
    }

    for k in 1..=N {
        product.fill(zero);

        for row in 0..N {
            for col in 0..N {
                let mut value = zero;

                for inner in 0..N {
                    value += matrix[row * N + inner] * b[inner * N + col];
                }

                product[row * N + col] = value;
            }
        }

        let mut trace = zero;

        for i in 0..N {
            trace += product[i * N + i];
        }

        let coefficient = T::from_real(-1.0 / k as f64) * trace;

        if k == N {
            return if (N & 1) == 0 {
                coefficient
            } else {
                -coefficient
            };
        }

        b.copy_from_slice(&product);

        for i in 0..N {
            b[i * N + i] += coefficient;
        }

        if k + 1 == N {
            let sign = if ((N - 1) & 1) == 0 { 1.0 } else { -1.0 };
            let sign = T::from_real(sign);

            for row in 0..N {
                for col in 0..N {
                    cof[row * N + col] = sign * b[col * N + row];
                }
            }
        }
    }

    unreachable!()
}

/// Compute independent determinants and cofactor matrices of packed compile-time `N x N` matrices
/// using the Faddeev-LeVerrier recurrence.
/// # Arguments:
/// - `cof`: Row-major packed cofactor matrix to write.
/// - `matrix`: Row-major packed matrix entries.
/// # Returns
/// - `V`: Packed determinants, one independent determinant per SIMD lane.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn adjugate_transpose_simd_const<
    V: Simd<LANES>,
    const LANES: usize,
    const N: usize,
    const D: usize,
>(
    cof: &mut [V],
    matrix: &[V],
) -> V {
    if N == 0 {
        return V::one();
    }

    let zero = V::zero();
    let one = V::one();
    let mut b = [zero; D];
    let mut product = [zero; D];

    for i in 0..N {
        b[i * N + i] = one;
    }

    if N == 1 {
        cof[0] = one;
    }

    for k in 1..=N {
        product.fill(zero);

        for row in 0..N {
            for col in 0..N {
                let mut value = zero;

                for inner in 0..N {
                    value = V::madd(value, matrix[row * N + inner], b[inner * N + col]);
                }

                product[row * N + col] = value;
            }
        }

        let mut trace = zero;

        for i in 0..N {
            trace = V::add(trace, product[i * N + i]);
        }

        let coefficient = V::scale_real(trace, -1.0 / k as f64);

        if k == N {
            return if (N & 1) == 0 {
                coefficient
            } else {
                V::sub(zero, coefficient)
            };
        }

        b.copy_from_slice(&product);

        for i in 0..N {
            let index = i * N + i;
            b[index] = V::add(b[index], coefficient);
        }

        if k + 1 == N {
            let sign = if ((N - 1) & 1) == 0 { 1.0 } else { -1.0 };

            for row in 0..N {
                for col in 0..N {
                    cof[row * N + col] = V::scale_real(b[col * N + row], sign);
                }
            }
        }
    }

    unreachable!()
}

/// Compute a determinant for a runtime-rank square matrix using partial-pivot LU and SVD fallback.
/// # Arguments:
/// - `matrix`: Input row-major square matrix.
/// - `n`: Runtime matrix rank.
/// # Returns
/// - `Option<T>`: Determinant when evaluation succeeds.
pub fn det_dynamic<T: NOCIScalar>(
    matrix: &[T],
    n: usize,
) -> Option<T> {
    let nn = n.checked_mul(n)?;

    if matrix.len() < nn {
        return None;
    }

    if n == 0 {
        return Some(T::from_real(1.0));
    }

    let mut lu = matrix[..nn].to_vec();

    if let Some(determinant) = det_lu_in_place(&mut lu, n) {
        return Some(determinant);
    }

    let view = ArrayView2::from_shape((n, n), &matrix[..nn]).ok()?;
    let (u, singular, vt) = view.svd(true, true).ok()?;
    let mut determinant = u?.det().ok()? * vt?.det().ok()?;

    for &value in &singular {
        determinant *= T::from_real(value);
    }

    if determinant.abs().is_finite() {
        Some(determinant)
    } else {
        None
    }
}

/// Compute a determinant using partial-pivot LU in caller-owned row-major storage.
/// # Arguments:
/// - `lu`: Runtime-rank matrix overwritten by its LU factors.
/// - `n`: Runtime matrix rank.
/// # Returns
/// - `Option<T>`: Determinant, zero for a singular matrix, or `None` for non-finite arithmetic.
fn det_lu_in_place<T: NOCIScalar>(
    lu: &mut [T],
    n: usize,
) -> Option<T> {
    let mut sign = 1.0;

    for k in 0..n {
        let mut pivot = k;
        let mut pivot_abs = lu[k * n + k].abs();

        if !pivot_abs.is_finite() {
            return None;
        }

        for row in (k + 1)..n {
            let value = lu[row * n + k].abs();

            if !value.is_finite() {
                return None;
            }

            if value > pivot_abs {
                pivot = row;
                pivot_abs = value;
            }
        }

        if pivot_abs == 0.0 {
            return Some(T::from_real(0.0));
        }

        if pivot != k {
            for col in 0..n {
                lu.swap(k * n + col, pivot * n + col);
            }

            sign = -sign;
        }

        let pivot_value = lu[k * n + k];

        for row in (k + 1)..n {
            let factor = lu[row * n + k] / pivot_value;
            lu[row * n + k] = factor;

            for col in (k + 1)..n {
                let pivot_entry = lu[k * n + col];
                lu[row * n + col] -= factor * pivot_entry;
            }
        }
    }

    let mut determinant = T::from_real(sign);

    for i in 0..n {
        determinant *= lu[i * n + i];
    }

    if determinant.abs().is_finite() {
        Some(determinant)
    } else {
        None
    }
}

/// Compute a runtime-rank determinant and cofactor matrix using LU and SVD fallback.
/// The output convention is
/// `cof[A]_{rc} = (-1)^{r+c} det A[r|c] = adj(A)_{cr}`.
/// # Arguments:
/// - `cof`: Row-major cofactor matrix to write.
/// - `invs`: Scratch inverse singular values.
/// - `lu`: Scratch matrix used by the LU path.
/// - `matrix`: Input row-major square matrix.
/// - `n`: Runtime matrix rank.
/// - `tol`: Threshold below which singular values are treated as zero.
/// # Returns
/// - `Option<T>`: Determinant when evaluation succeeds.
pub fn adjugate_transpose_dynamic<T: NOCIScalar>(
    cof: &mut [T],
    invs: &mut [f64],
    lu: &mut [T],
    matrix: &[T],
    n: usize,
    tol: f64,
) -> Option<T> {
    let nn = n.checked_mul(n)?;

    if cof.len() < nn || invs.len() < n || lu.len() < nn || matrix.len() < nn {
        return None;
    }

    if n == 0 {
        return Some(T::from_real(1.0));
    }

    lu[..nn].copy_from_slice(&matrix[..nn]);
    let lu_view = ArrayViewMut2::from_shape((n, n), &mut lu[..nn]).ok()?;

    if let Ok(factorisation) = lu_view.to_owned().factorize_into()
        && let (Ok(determinant), Ok(inverse)) = (factorisation.det(), factorisation.inv_into())
    {
        let inverse = inverse.as_slice()?;

        for row in 0..n {
            for col in 0..n {
                cof[row * n + col] = determinant * inverse[col * n + row];
            }
        }

        return Some(determinant);
    }

    cof[..nn].fill(T::from_real(0.0));
    invs[..n].fill(0.0);
    let view = ArrayView2::from_shape((n, n), &matrix[..nn]).ok()?;
    let (u, singular, vt) = view.svd(true, true).ok()?;
    let u = u?;
    let vt = vt?;
    let mut reduced_determinant = u.det().ok()? * vt.det().ok()?;
    let mut determinant = reduced_determinant;
    let mut nzero = 0usize;
    let mut zero_index = 0usize;

    for i in 0..n {
        let value = singular[i];
        determinant *= T::from_real(value);

        if value.abs() > tol {
            reduced_determinant *= T::from_real(value);
            invs[i] = 1.0 / value;
        } else {
            nzero += 1;
            zero_index = i;
        }
    }

    if nzero == 0 {
        for i in 0..n {
            let inverse = T::from_real(invs[i]);

            for row in 0..n {
                let scale = inverse * u[(row, i)].conj();

                for col in 0..n {
                    cof[row * n + col] += scale * vt[(i, col)].conj();
                }
            }
        }

        for value in &mut cof[..nn] {
            *value *= determinant;
        }
    } else if nzero == 1 {
        for row in 0..n {
            let scale = reduced_determinant * u[(row, zero_index)].conj();

            for col in 0..n {
                cof[row * n + col] = scale * vt[(zero_index, col)].conj();
            }
        }
    }

    Some(determinant)
}
