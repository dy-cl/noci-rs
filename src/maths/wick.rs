// maths/wick.rs
//! Structural matrix operations are generic over `T: Copy` and therefore work for scalar and
//! packed entries without separate SIMD wrappers. Arithmetic reductions use distinct scalar and
//! packed implementations because they require different arithmetic traits and numerical paths.

// External crate imports.
#[cfg(feature = "nocc")]
use ndarray::Array2;
use ndarray::{ArrayView2, ArrayViewMut2};
use ndarray_linalg::{Determinant, FactorizeInto, InverseInto, SVD};

// Crate-root imports.
#[cfg(target_arch = "x86_64")]
use crate::maths::Simd;
use crate::noci::NOCIScalar;

/// Calculate a runtime-rank determinant coefficient for an occupation bitstring.
/// NOCC supplies both the occupation mask and electron count at runtime, so this composed helper
/// has no fixed-rank or packed caller.
/// # Arguments:
/// - `c`: Orbital coefficient matrix in an orthonormal basis.
/// - `mask`: Occupation bitstring.
/// - `nel`: Number of occupied orbitals.
/// # Returns
/// - `T`: Determinant coefficient for the occupied rows and first `nel` columns.
#[cfg(feature = "nocc")]
pub(crate) fn det_occupied_minor_dynamic<T: NOCIScalar>(
    c: &Array2<T>,
    mask: u128,
    nel: usize,
) -> T {
    // Decode the occupied rows in orbital order so the determinant has the same orientation as
    // the occupation bitstring.
    let mut rows = Vec::with_capacity(nel);

    for p in 0..c.nrows() {
        if ((mask >> p) & 1) == 1 {
            rows.push(p);
        }
    }

    // Materialise the occupied-row, leading-column square minor in row-major order.
    let mut matrix = Vec::with_capacity(nel * nel);
    for &row in &rows {
        for col in 0..nel {
            matrix.push(c[(row, col)]);
        }
    }

    // NOCC supplies `nel` at runtime, so this composed helper deliberately uses the dynamic
    // determinant path rather than introducing a rank dispatcher here.
    det_dynamic(matrix.as_slice(), nel).unwrap_or_else(|| T::from_real(0.0))
}

/// Construct the rank-`L` contraction determinant.
/// `D_{ij} = X_{r_i c_j}` for `i >= j`, and `D_{ij} = Y_{r_i c_j}` for `i < j`.
/// Entry selection only copies `T`, so the same implementation accepts scalar or packed values.
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
    // Cache ndarray strides and base pointers once; orbital labels then map directly to matrix
    // offsets inside the fixed-rank loops.
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

            // The diagonal and lower triangle use `X` contractions.
            for j in 0..=i {
                let col = *cols.get_unchecked(j) as isize;
                *d.get_unchecked_mut(base + j) = *xptr.offset(xr + col * xstr[1]);
            }

            // The strict upper triangle uses `Y` contractions.
            for j in (i + 1)..L {
                let col = *cols.get_unchecked(j) as isize;
                *d.get_unchecked_mut(base + j) = *yptr.offset(yr + col * ystr[1]);
            }
        }
    }
}

/// Construct a runtime-rank contraction determinant.
/// `D_{ij} = X_{r_i c_j}` for `i >= j`, and `D_{ij} = Y_{r_i c_j}` for `i < j`.
/// Entry selection only copies `T`, so the same implementation accepts scalar or packed values.
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
    // The runtime path uses the same fill convention as `build_d_const`, but carries `l` as the
    // matrix stride because ranks above `MAXEXCIT` are not monomorphised.
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

            // Copy the diagonal and lower triangle from `X`.
            for j in 0..=i {
                let col = *cols.get_unchecked(j) as isize;
                *d.get_unchecked_mut(base + j) = *xptr.offset(xr + col * xstr[1]);
            }

            // Copy the strict upper triangle from `Y`.
            for j in (i + 1)..l {
                let col = *cols.get_unchecked(j) as isize;
                *d.get_unchecked_mut(base + j) = *yptr.offset(yr + col * ystr[1]);
            }
        }
    }
}

/// Select each column of a rank-`L` matrix from `det0` or `det1`.
/// Bit `c` selects the source of column `c`.
/// Column selection only copies `T`, so the same implementation accepts scalar or packed values.
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
        // Visit in row-major order so each selection bit controls one complete source column.
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
/// Column selection only copies `T`, so the same implementation accepts scalar or packed values.
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
        // This is the runtime-rank form of the same column-wise selection performed above.
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
/// Minor construction only copies `T`, so the same implementation accepts scalar or packed values.
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
    // The first minor of a scalar matrix has no stored entries when its source order is zero or
    // one; callers evaluate the empty determinant separately.
    if L <= 1 {
        return;
    }

    // Compact retained rows and columns into a contiguous `(L - 1) x (L - 1)` matrix.
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
/// Minor construction only copies `T`, so the same implementation accepts scalar or packed values.
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
    // Preserve the fixed-rank empty-minor convention for a runtime source order.
    if l <= 1 {
        return;
    }

    // Source coordinates use stride `l`; output coordinates count only retained entries.
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
/// Minor construction only copies `T`, so the same implementation accepts scalar or packed values.
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
    // Orders zero, one and two have an empty second minor, whose determinant is handled as one by
    // the determinant evaluator.
    if L <= 2 {
        return;
    }

    // `minor_row` and `minor_col` compact the retained entries while source coordinates continue
    // to use the original `L x L` stride.
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

/// Construct the second minor obtained by deleting two rows and two columns from a runtime-rank
/// square matrix.
/// Minor construction only copies `T`, so the same implementation accepts scalar or packed values.
/// # Arguments:
/// - `out`: Row-major second-minor storage.
/// - `matrix`: Input row-major square matrix.
/// - `l`: Runtime matrix rank.
/// - `row0`: First removed row.
/// - `row1`: Second removed row.
/// - `col0`: First removed column.
/// - `col1`: Second removed column.
/// # Returns
/// - `()`: Writes the second minor into `out`.
#[inline(always)]
pub fn second_minor_dynamic<T: Copy>(
    out: &mut [T],
    matrix: &[T],
    l: usize,
    row0: usize,
    row1: usize,
    col0: usize,
    col1: usize,
) {
    // Runtime ranks below three likewise have an empty second minor.
    if l <= 2 {
        return;
    }

    // Compact retained rows and columns while indexing the input with its runtime stride.
    let mut minor_row = 0usize;
    for row in 0..l {
        if row == row0 || row == row1 {
            continue;
        }

        let mut minor_col = 0usize;
        for col in 0..l {
            if col == col0 || col == col1 {
                continue;
            }

            unsafe {
                *out.get_unchecked_mut(minor_row * (l - 2) + minor_col) =
                    *matrix.get_unchecked(row * l + col);
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
    // The empty determinant is the multiplicative identity.
    if N == 0 {
        return T::from_real(1.0);
    }

    let zero = T::from_real(0.0);
    let one = T::from_real(1.0);
    let mut b = [zero; D];
    let mut product = [zero; D];

    // Start the recurrence with `B_0 = I`.
    for i in 0..N {
        b[i * N + i] = one;
    }

    for k in 1..=N {
        // Form `A B_{k-1}`. Const ranks let LLVM unroll these matrix products without pivot or
        // matrix-order branches.
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

        // `c_k = -tr(A B_{k-1}) / k` is the next characteristic-polynomial coefficient.
        let mut trace = zero;

        for i in 0..N {
            trace += product[i * N + i];
        }

        let coefficient = T::from_real(-1.0 / k as f64) * trace;

        // The final coefficient differs from `det(A)` by `(-1)^N`.
        if k == N {
            return if (N & 1) == 0 {
                coefficient
            } else {
                -coefficient
            };
        }

        // Advance with `B_k = A B_{k-1} + c_k I`.
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
    // Every lane follows the scalar recurrence independently, including the empty determinant.
    if N == 0 {
        return V::one();
    }

    let zero = V::zero();
    let one = V::one();
    let mut b = [zero; D];
    let mut product = [zero; D];

    // Start every packed recurrence with `B_0 = I`.
    for i in 0..N {
        b[i * N + i] = one;
    }

    for k in 1..=N {
        // Form packed `A B_{k-1}` with fused lane-local products and no shared pivot decision.
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

        // Compute packed `c_k = -tr(A B_{k-1}) / k`.
        let mut trace = zero;

        for i in 0..N {
            trace = V::add(trace, product[i * N + i]);
        }

        let coefficient = V::scale_real(trace, -1.0 / k as f64);

        // Convert the final characteristic coefficient to one determinant per lane.
        if k == N {
            return if (N & 1) == 0 {
                coefficient
            } else {
                V::sub(zero, coefficient)
            };
        }

        // Advance every lane with `B_k = A B_{k-1} + c_k I`.
        b.copy_from_slice(&product);

        for i in 0..N {
            let index = i * N + i;
            b[index] = V::add(b[index], coefficient);
        }
    }

    unreachable!()
}

/// Compute a determinant for a runtime-rank square matrix using partial-pivot LU and SVD fallback.
/// This path evaluates one scalar matrix because packed lanes cannot share pivot decisions.
/// # Arguments:
/// - `matrix`: Input row-major square matrix.
/// - `n`: Runtime matrix rank.
/// # Returns
/// - `Option<T>`: Determinant when evaluation succeeds.
pub fn det_dynamic<T: NOCIScalar>(
    matrix: &[T],
    n: usize,
) -> Option<T> {
    // Validate the runtime shape before slicing caller storage.
    let nn = n.checked_mul(n)?;

    if matrix.len() < nn {
        return None;
    }

    // Preserve the empty-determinant identity used by fixed kernels and minor evaluation.
    if n == 0 {
        return Some(T::from_real(1.0));
    }

    // Partial-pivot LU is the normal runtime path and avoids the higher cost of an SVD.
    let mut lu = matrix[..nn].to_vec();

    if let Some(determinant) = det_lu_in_place(&mut lu, n) {
        return Some(determinant);
    }

    // Singular or non-finite LU arithmetic falls back to a rank-independent SVD determinant.
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
        // Select the largest finite entry in this column to control numerical growth.
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

        // An exactly zero pivot makes the determinant zero without requiring division.
        if pivot_abs == 0.0 {
            return Some(T::from_real(0.0));
        }

        // Move the selected pivot row into place and record the determinant sign change.
        if pivot != k {
            for col in 0..n {
                lu.swap(k * n + col, pivot * n + col);
            }

            sign = -sign;
        }

        // Eliminate entries below the pivot while storing the multipliers in the lower triangle.
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

    // The determinant is the signed product of the upper-triangular diagonal.
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
    // The order-zero determinant is one and has no stored cofactor entries.
    if N == 0 {
        return T::from_real(1.0);
    }

    let zero = T::from_real(0.0);
    let one = T::from_real(1.0);
    let mut b = [zero; D];
    let mut product = [zero; D];

    // Initialise `B_0 = I`; the penultimate recurrence matrix yields the adjugate.
    for i in 0..N {
        b[i * N + i] = one;
    }

    // Deleting the only row and column leaves the empty determinant.
    if N == 1 {
        cof[0] = one;
    }

    for k in 1..=N {
        // Form `A B_{k-1}` with the same recurrence used by `det_const`.
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

        // Obtain the next characteristic-polynomial coefficient from the trace.
        let mut trace = zero;

        for i in 0..N {
            trace += product[i * N + i];
        }

        let coefficient = T::from_real(-1.0 / k as f64) * trace;

        // The last coefficient gives the determinant after applying `(-1)^N`.
        if k == N {
            return if (N & 1) == 0 {
                coefficient
            } else {
                -coefficient
            };
        }

        // Advance `B_k = A B_{k-1} + c_k I`.
        b.copy_from_slice(&product);

        for i in 0..N {
            b[i * N + i] += coefficient;
        }

        // `(-1)^(N-1) B_{N-1}` is `adj(A)`; transpose while writing the repository's cofactor
        // convention `cof[A]_{rc} = adj(A)_{cr}`.
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
    // Each packed order-zero determinant is one and has no cofactor entries.
    if N == 0 {
        return V::one();
    }

    let zero = V::zero();
    let one = V::one();
    let mut b = [zero; D];
    let mut product = [zero; D];

    // Initialise packed `B_0 = I`; all lanes use the scalar recurrence independently.
    for i in 0..N {
        b[i * N + i] = one;
    }

    // The cofactor of each packed `1 x 1` matrix is the empty determinant.
    if N == 1 {
        cof[0] = one;
    }

    for k in 1..=N {
        // Form packed `A B_{k-1}` with the same structure as the scalar fixed kernel.
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

        // Obtain one characteristic-polynomial coefficient per lane from the packed trace.
        let mut trace = zero;

        for i in 0..N {
            trace = V::add(trace, product[i * N + i]);
        }

        let coefficient = V::scale_real(trace, -1.0 / k as f64);

        // Convert the last coefficients to packed determinants.
        if k == N {
            return if (N & 1) == 0 {
                coefficient
            } else {
                V::sub(zero, coefficient)
            };
        }

        // Advance every lane with `B_k = A B_{k-1} + c_k I`.
        b.copy_from_slice(&product);

        for i in 0..N {
            let index = i * N + i;
            b[index] = V::add(b[index], coefficient);
        }

        // Transpose `(-1)^(N-1) B_{N-1}` into the packed cofactor convention.
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

/// Compute a runtime-rank determinant and cofactor matrix using LU and SVD fallback.
/// This path evaluates one scalar matrix because packed lanes cannot share pivot decisions.
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
    // Validate all caller-owned runtime scratch before creating matrix views.
    let nn = n.checked_mul(n)?;

    if cof.len() < nn || invs.len() < n || lu.len() < nn || matrix.len() < nn {
        return None;
    }

    // The order-zero determinant is one and has no cofactor entries.
    if n == 0 {
        return Some(T::from_real(1.0));
    }

    // For a nonsingular matrix, `cof(A) = det(A) A^{-T}` gives determinant and cofactors from one
    // LU factorisation.
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

    // If LU inversion fails, use the SVD so singular matrices still receive polynomially correct
    // first cofactors.
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

    // Separate nonzero singular values from the null space while accumulating determinant factors.
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
        // Reconstruct `A^{-T}` from all singular triplets, then multiply by `det(A)`.
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
        // A one-dimensional null space has a nonzero rank-`N - 1` cofactor matrix formed from the
        // omitted singular triplet and the product of retained singular values.
        for row in 0..n {
            let scale = reduced_determinant * u[(row, zero_index)].conj();

            for col in 0..n {
                cof[row * n + col] = scale * vt[(zero_index, col)].conj();
            }
        }
    }
    // Two or more zero singular values imply every first cofactor is zero; the initial fill already
    // represents that result.

    Some(determinant)
}
