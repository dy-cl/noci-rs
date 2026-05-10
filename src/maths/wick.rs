// maths/wick.rs

use ndarray::ArrayView2;
use crate::StateScalar;

/// Build the square `l x l` contraction determinant with `x` elements in the
/// diagonal and lower triangle, and `y` elements in the upper triangle.
/// Dispatches to specialised implementations for small excitation ranks.
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
pub fn build_d<T: Copy>(d: &mut [T], l: usize, x: &ArrayView2<T>, y: &ArrayView2<T>, rows: &[usize], cols: &[usize]) {
    match l {
        0 => {}
        1 => build_d::build_d1(d, x, y, rows, cols),
        2 => build_d::build_d2(d, x, y, rows, cols),
        3 => build_d::build_d3(d, x, y, rows, cols),
        4 => build_d::build_d4(d, x, y, rows, cols),
        _ => build_d::build_d_gen(d, l, x, y, rows, cols),
    }
}

mod build_d {
    use ndarray::ArrayView2;

    /// Construct contraction determinant for excitation rank 1.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `x`: X matrix elements.
    /// - `y`: Y matrix elements.
    /// - `rows`: Row indices of X or Y.
    /// - `cols`: Column indices of X or Y.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn build_d1<T: Copy>(d: &mut [T], x: &ArrayView2<T>, _y: &ArrayView2<T>, rows: &[usize], cols: &[usize]) {
        let xstr = x.strides();
        let xptr = x.as_ptr();

        unsafe {
            let r0 = *rows.get_unchecked(0) as isize;
            let c0 = *cols.get_unchecked(0) as isize;

            d[0] = *xptr.offset(r0 * xstr[0] + c0 * xstr[1]);
        }
    }

    /// Construct contraction determinant for excitation rank 2.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `x`: X matrix elements.
    /// - `y`: Y matrix elements.
    /// - `rows`: Row indices of X or Y.
    /// - `cols`: Column indices of X or Y.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn build_d2<T: Copy>(d: &mut [T], x: &ArrayView2<T>, y: &ArrayView2<T>, rows: &[usize], cols: &[usize]) {
        let xstr = x.strides();
        let ystr = y.strides();
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();

        unsafe {
            let r0 = *rows.get_unchecked(0) as isize;
            let r1 = *rows.get_unchecked(1) as isize;

            let c0 = *cols.get_unchecked(0) as isize;
            let c1 = *cols.get_unchecked(1) as isize;

            let xr0 = r0 * xstr[0];
            let xr1 = r1 * xstr[0];
            let yr0 = r0 * ystr[0];

            d[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            d[1] = *yptr.offset(yr0 + c1 * ystr[1]);

            d[2] = *xptr.offset(xr1 + c0 * xstr[1]);
            d[3] = *xptr.offset(xr1 + c1 * xstr[1]);
        }
    }

    /// Construct contraction determinant for excitation rank 3.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `x`: X matrix elements.
    /// - `y`: Y matrix elements.
    /// - `rows`: Row indices of X or Y.
    /// - `cols`: Column indices of X or Y.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn build_d3<T: Copy>(d: &mut [T], x: &ArrayView2<T>, y: &ArrayView2<T>, rows: &[usize], cols: &[usize]) {
        let xstr = x.strides();
        let ystr = y.strides();
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();

        unsafe {
            let r0 = *rows.get_unchecked(0) as isize;
            let r1 = *rows.get_unchecked(1) as isize;
            let r2 = *rows.get_unchecked(2) as isize;

            let c0 = *cols.get_unchecked(0) as isize;
            let c1 = *cols.get_unchecked(1) as isize;
            let c2 = *cols.get_unchecked(2) as isize;

            let xr0 = r0 * xstr[0];
            let xr1 = r1 * xstr[0];
            let xr2 = r2 * xstr[0];

            let yr0 = r0 * ystr[0];
            let yr1 = r1 * ystr[0];

            d[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            d[1] = *yptr.offset(yr0 + c1 * ystr[1]);
            d[2] = *yptr.offset(yr0 + c2 * ystr[1]);

            d[3] = *xptr.offset(xr1 + c0 * xstr[1]);
            d[4] = *xptr.offset(xr1 + c1 * xstr[1]);
            d[5] = *yptr.offset(yr1 + c2 * ystr[1]);

            d[6] = *xptr.offset(xr2 + c0 * xstr[1]);
            d[7] = *xptr.offset(xr2 + c1 * xstr[1]);
            d[8] = *xptr.offset(xr2 + c2 * xstr[1]);
        }
    }

    /// Construct contraction determinant for excitation rank 4.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `x`: X matrix elements.
    /// - `y`: Y matrix elements.
    /// - `rows`: Row indices of X or Y.
    /// - `cols`: Column indices of X or Y.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn build_d4<T: Copy>(d: &mut [T], x: &ArrayView2<T>, y: &ArrayView2<T>, rows: &[usize], cols: &[usize]) {
        let xstr = x.strides();
        let ystr = y.strides();
        let xptr = x.as_ptr();
        let yptr = y.as_ptr();

        unsafe {
            let r0 = *rows.get_unchecked(0) as isize;
            let r1 = *rows.get_unchecked(1) as isize;
            let r2 = *rows.get_unchecked(2) as isize;
            let r3 = *rows.get_unchecked(3) as isize;

            let c0 = *cols.get_unchecked(0) as isize;
            let c1 = *cols.get_unchecked(1) as isize;
            let c2 = *cols.get_unchecked(2) as isize;
            let c3 = *cols.get_unchecked(3) as isize;

            let xr0 = r0 * xstr[0];
            let xr1 = r1 * xstr[0];
            let xr2 = r2 * xstr[0];
            let xr3 = r3 * xstr[0];

            let yr0 = r0 * ystr[0];
            let yr1 = r1 * ystr[0];
            let yr2 = r2 * ystr[0];

            d[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            d[1] = *yptr.offset(yr0 + c1 * ystr[1]);
            d[2] = *yptr.offset(yr0 + c2 * ystr[1]);
            d[3] = *yptr.offset(yr0 + c3 * ystr[1]);

            d[4] = *xptr.offset(xr1 + c0 * xstr[1]);
            d[5] = *xptr.offset(xr1 + c1 * xstr[1]);
            d[6] = *yptr.offset(yr1 + c2 * ystr[1]);
            d[7] = *yptr.offset(yr1 + c3 * ystr[1]);

            d[8] = *xptr.offset(xr2 + c0 * xstr[1]);
            d[9] = *xptr.offset(xr2 + c1 * xstr[1]);
            d[10] = *xptr.offset(xr2 + c2 * xstr[1]);
            d[11] = *yptr.offset(yr2 + c3 * ystr[1]);

            d[12] = *xptr.offset(xr3 + c0 * xstr[1]);
            d[13] = *xptr.offset(xr3 + c1 * xstr[1]);
            d[14] = *xptr.offset(xr3 + c2 * xstr[1]);
            d[15] = *xptr.offset(xr3 + c3 * xstr[1]);
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
    pub(super) fn build_d_gen<T: Copy>(d: &mut [T], l: usize, x: &ArrayView2<T>, y: &ArrayView2<T>, rows: &[usize], cols: &[usize]) {
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
pub fn mix_columns<T: Copy>(d: &mut [T], det0: &[T], det1: &[T], l: usize, bits: u64) {
    match l {
        0 => {}
        1 => mix_columns::mix_columns1(d, det0, det1, bits),
        2 => mix_columns::mix_columns2(d, det0, det1, bits),
        3 => mix_columns::mix_columns3(d, det0, det1, bits),
        4 => mix_columns::mix_columns4(d, det0, det1, bits),
        _ => mix_columns::mix_columns_gen(d, det0, det1, l, bits),
    }
}

mod mix_columns {
    /// Mix columns of `det1` into `det0` according to `bits` for excitation rank 1.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `det0`: Base matrix.
    /// - `det1`: Mixing matrix.
    /// - `bits`: Bitstring.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn mix_columns1<T: Copy>(d: &mut [T], det0: &[T], det1: &[T], bits: u64) {
        let b0 = (bits & 1) != 0;
        d[0] = if b0 {det1[0]} else {det0[0]};
    }

    /// Mix columns of `det1` into `det0` according to `bits` for excitation rank 2.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `det0`: Base matrix.
    /// - `det1`: Mixing matrix.
    /// - `bits`: Bitstring.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn mix_columns2<T: Copy>(d: &mut [T], det0: &[T], det1: &[T], bits: u64) {
        let b0 = (bits & 1) != 0;
        let b1 = (bits & 2) != 0;

        d[0] = if b0 {det1[0]} else {det0[0]};
        d[1] = if b1 {det1[1]} else {det0[1]};

        d[2] = if b0 {det1[2]} else {det0[2]};
        d[3] = if b1 {det1[3]} else {det0[3]};
    }

    /// Mix columns of `det1` into `det0` according to `bits` for excitation rank 3.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `det0`: Base matrix.
    /// - `det1`: Mixing matrix.
    /// - `bits`: Bitstring.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn mix_columns3<T: Copy>(d: &mut [T], det0: &[T], det1: &[T], bits: u64) {
        let b0 = (bits & 1) != 0;
        let b1 = (bits & 2) != 0;
        let b2 = (bits & 4) != 0;

        d[0] = if b0 {det1[0]} else {det0[0]};
        d[1] = if b1 {det1[1]} else {det0[1]};
        d[2] = if b2 {det1[2]} else {det0[2]};

        d[3] = if b0 {det1[3]} else {det0[3]};
        d[4] = if b1 {det1[4]} else {det0[4]};
        d[5] = if b2 {det1[5]} else {det0[5]};

        d[6] = if b0 {det1[6]} else {det0[6]};
        d[7] = if b1 {det1[7]} else {det0[7]};
        d[8] = if b2 {det1[8]} else {det0[8]};
    }

    /// Mix columns of `det1` into `det0` according to `bits` for excitation rank 4.
    /// # Arguments:
    /// - `d`: Matrix to write into.
    /// - `det0`: Base matrix.
    /// - `det1`: Mixing matrix.
    /// - `bits`: Bitstring.
    /// # Returns
    /// - `()`: Writes the contraction determinant into `d`.
    #[inline(always)]
    pub(super) fn mix_columns4<T: Copy>(d: &mut [T], det0: &[T], det1: &[T], bits: u64) {
        let b0 = (bits & 1) != 0;
        let b1 = (bits & 2) != 0;
        let b2 = (bits & 4) != 0;
        let b3 = (bits & 8) != 0;

        d[0] = if b0 {det1[0]} else {det0[0]};
        d[1] = if b1 {det1[1]} else {det0[1]};
        d[2] = if b2 {det1[2]} else {det0[2]};
        d[3] = if b3 {det1[3]} else {det0[3]};

        d[4] = if b0 {det1[4]} else {det0[4]};
        d[5] = if b1 {det1[5]} else {det0[5]};
        d[6] = if b2 {det1[6]} else {det0[6]};
        d[7] = if b3 {det1[7]} else {det0[7]};

        d[8] = if b0 {det1[8]} else {det0[8]};
        d[9] = if b1 {det1[9]} else {det0[9]};
        d[10] = if b2 {det1[10]} else {det0[10]};
        d[11] = if b3 {det1[11]} else {det0[11]};

        d[12] = if b0 {det1[12]} else {det0[12]};
        d[13] = if b1 {det1[13]} else {det0[13]};
        d[14] = if b2 {det1[14]} else {det0[14]};
        d[15] = if b3 {det1[15]} else {det0[15]};
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
    pub(super) fn mix_columns_gen<T: Copy>(d: &mut [T], det0: &[T], det1: &[T], l: usize, bits: u64) {
        unsafe {
            for r in 0..l {
                let base = r * l;
                for c in 0..l {
                    let k = base + c;
                    let use1 = ((bits >> c) & 1) != 0;
                    *d.get_unchecked_mut(k) = if use1 {*det1.get_unchecked(k)} else {*det0.get_unchecked(k)};
                }
            }
        }
    }
}

/// Construct the minor of an `n x n` matrix obtained by removing row `r_rm`
/// and column `c_rm`. Dispatches to specialised implementations for small matrix sizes.
/// # Arguments:
/// - `out`: Matrix to be written into.
/// - `m`: Base matrix.
/// - `n`: Dimension of the square matrix `m`.
/// - `r_rm`: Row index to remove.
/// - `c_rm`: Column index to remove.
/// # Returns
/// - `()`: Writes the minor into `out`.
#[inline(always)]
pub fn minor<T: Copy>(out: &mut [T], m: &[T], n: usize, r_rm: usize, c_rm: usize) {
    match n {
        0 | 1 => {}
        2 => minor::minor2(out, m, r_rm, c_rm),
        3 => minor::minor3(out, m, r_rm, c_rm),
        4 => minor::minor4(out, m, r_rm, c_rm),
        _ => minor::minor_gen(out, m, n, r_rm, c_rm),
    }
}

mod minor {
    /// Calculate minor of a square matrix for excitation rank 2.
    /// # Arguments:
    /// - `out`: Matrix to be written into.
    /// - `m`: Base matrix.
    /// - `r_rm`: Row index to remove.
    /// - `c_rm`: Column index to remove.
    /// # Returns
    /// - `()`: Writes the minor into `out`.
    #[inline(always)]
    pub(super) fn minor2<T: Copy>(out: &mut [T], m: &[T], r_rm: usize, c_rm: usize) {
        let r = 1 ^ r_rm;
        let c = 1 ^ c_rm;
        out[0] = m[r * 2 + c];
    }

    /// Calculate minor of a square matrix for excitation rank 3.
    /// # Arguments:
    /// - `out`: Matrix to be written into.
    /// - `m`: Base matrix.
    /// - `r_rm`: Row index to remove.
    /// - `c_rm`: Column index to remove.
    /// # Returns
    /// - `()`: Writes the minor into `out`.
    #[inline(always)]
    pub(super) fn minor3<T: Copy>(out: &mut [T], m: &[T], r_rm: usize, c_rm: usize) {
        let mut ii = 0usize;
        for i in 0..3 {
            if i == r_rm {continue;}
            let base_i = i * 3;
            let base_o = ii * 2;
            let mut jj = 0usize;
            for j in 0..3 {
                if j == c_rm {continue;}
                out[base_o + jj] = m[base_i + j];
                jj += 1;
            }
            ii += 1;
        }
    }

    /// Calculate minor of a square matrix for excitation rank 4.
    /// # Arguments:
    /// - `out`: Matrix to be written into.
    /// - `m`: Base matrix.
    /// - `r_rm`: Row index to remove.
    /// - `c_rm`: Column index to remove.
    /// # Returns
    /// - `()`: Writes the minor into `out`.
    #[inline(always)]
    pub(super) fn minor4<T: Copy>(out: &mut [T], m: &[T], r_rm: usize, c_rm: usize) {
        let mut ii = 0usize;
        for i in 0..4 {
            if i == r_rm {continue;}
            let base_i = i * 4;
            let base_o = ii * 3;
            let mut jj = 0usize;
            for j in 0..4 {
                if j == c_rm {continue;}
                out[base_o + jj] = m[base_i + j];
                jj += 1;
            }
            ii += 1;
        }
    }

    /// Calculate minor of a square matrix for arbitrary excitation rank.
    /// # Arguments:
    /// - `out`: Matrix to be written into.
    /// - `m`: Base matrix.
    /// - `r_rm`: Row index to remove.
    /// - `c_rm`: Column index to remove.
    /// # Returns
    /// - `()`: Writes the minor into `out`.
    #[inline(always)]
    pub(super) fn minor_gen<T: Copy>(out: &mut [T], m: &[T], n: usize, r_rm: usize, c_rm: usize) {
        if n == 0 {return;}
        let mut ii = 0usize;
        for i in 0..n {
            if i == r_rm {continue;}
            let mut jj = 0usize;
            for j in 0..n {
                if j == c_rm {continue;}
                out[ii * (n - 1) + jj] = m[i * n + j];
                jj += 1;
            }
            ii += 1;
        }
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
pub fn det<T: StateScalar>(a: &[T], n: usize) -> Option<T> {
    if a.len() != n * n {
        return None;
    }

    match n {
        0 => Some(T::from_real(1.0)),
        1 => Some(a[0]),
        2 => {
            let d = det_mod::det2(a);
            if d.abs().is_finite() {Some(d)} else {None}
        }
        3 => {
            let d = det_mod::det3(a);
            if d.abs().is_finite() {Some(d)} else {None}
        }
        4 => {
            let d = det_mod::det4(a);
            if d.abs().is_finite() {Some(d)} else {None}
        }
        _ => det_mod::det_gen(a, n),
    }
}

mod det_mod {
    use ndarray::ArrayView2;
    use ndarray_linalg::{Determinant, FactorizeInto, SVD};
    use crate::StateScalar;

    /// Calculate determinant of a 2 x 2 matrix.
    /// # Arguments:
    /// - `a`: Matrix to calculate the determinant of.
    /// # Returns
    /// - `T`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det2<T: StateScalar>(a: &[T]) -> T {
        det2scalar(a[0], a[1], a[2], a[3])
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
    pub(super) fn det2scalar<T: StateScalar>(a00: T, a01: T, a10: T, a11: T) -> T {
        a00 * a11 - a01 * a10
    }

    /// Calculate determinant of a 3 x 3 matrix.
    /// # Arguments:
    /// - `a`: Matrix to calculate the determinant of.
    /// # Returns
    /// - `T`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det3<T: StateScalar>(a: &[T]) -> T {
        let a00 = a[0]; let a01 = a[1]; let a02 = a[2];
        let a10 = a[3]; let a11 = a[4]; let a12 = a[5];
        let a20 = a[6]; let a21 = a[7]; let a22 = a[8];

        det3scalar(a00, a01, a02, a10, a11, a12, a20, a21, a22)
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
    pub(super) fn det3scalar<T: StateScalar>(a00: T, a01: T, a02: T, a10: T, a11: T, a12: T, a20: T, a21: T, a22: T) -> T {
        a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
    }

    /// Calculate determinant of a 4 x 4 matrix.
    /// # Arguments:
    /// - `a`: Matrix to calculate the determinant of.
    /// # Returns
    /// - `T`: Determinant of the matrix.
    #[inline(always)]
    pub(super) fn det4<T: StateScalar>(a: &[T]) -> T {
        let m00 = {
            let a11 = a[5];  let a12 = a[6];  let a13 = a[7];
            let a21 = a[9];  let a22 = a[10]; let a23 = a[11];
            let a31 = a[13]; let a32 = a[14]; let a33 = a[15];
            a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
        };

        let m01 = {
            let a10 = a[4];  let a12 = a[6];  let a13 = a[7];
            let a20 = a[8];  let a22 = a[10]; let a23 = a[11];
            let a30 = a[12]; let a32 = a[14]; let a33 = a[15];
            a10 * (a22 * a33 - a23 * a32) - a12 * (a20 * a33 - a23 * a30) + a13 * (a20 * a32 - a22 * a30)
        };

        let m02 = {
            let a10 = a[4];  let a11 = a[5];  let a13 = a[7];
            let a20 = a[8];  let a21 = a[9];  let a23 = a[11];
            let a30 = a[12]; let a31 = a[13]; let a33 = a[15];
            a10 * (a21 * a33 - a23 * a31) - a11 * (a20 * a33 - a23 * a30) + a13 * (a20 * a31 - a21 * a30)
        };

        let m03 = {
            let a10 = a[4];  let a11 = a[5];  let a12 = a[6];
            let a20 = a[8];  let a21 = a[9];  let a22 = a[10];
            let a30 = a[12]; let a31 = a[13]; let a32 = a[14];
            a10 * (a21 * a32 - a22 * a31) - a11 * (a20 * a32 - a22 * a30) + a12 * (a20 * a31 - a21 * a30)
        };

        a[0] * m00 - a[1] * m01 + a[2] * m02 - a[3] * m03
    }

    /// Compute determinant of `a` for arbitrary size using LU factorisation first and SVD as a fallback.
    /// # Arguments:
    /// - `a`: Matrix to find determinant of.
    /// - `n`: Matrix dimension.
    /// # Returns
    /// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn det_gen<T: StateScalar>(a: &[T], n: usize) -> Option<T> {
        let av = ArrayView2::from_shape((n, n), a).ok()?;

        if let Ok(m) = av.to_owned().factorize_into() {
            if let Ok(d) = m.det() {
                if d.abs().is_finite() {return Some(d);}
            }
        }

        let (u_opt, s, vt_opt) = av.svd(true, true).ok()?;
        let u = u_opt?;
        let vt = vt_opt?;

        let det_u = u.det().ok()?;
        let det_vt = vt.det().ok()?;

        let mut det = det_u * det_vt;
        for &si in s.iter() {
            det *= T::from_real(si);
        }

        if det.abs().is_finite() {Some(det)} else {None}
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
pub fn adjugate_transpose<T: StateScalar>(adjt: &mut [T], invs: &mut [f64], lu: &mut [T], a: &[T], n: usize, thresh: f64) -> Option<T> {
    match n {
        0 => Some(T::from_real(1.0)),
        1 => adjt_mod::adjt1(adjt, a),
        2 => adjt_mod::adjt2(adjt, a),
        3 => adjt_mod::adjt3(adjt, a),
        4 => adjt_mod::adjt4(adjt, a),
        _ => adjt_mod::adjt_gen(adjt, invs, lu, a, n, thresh),
    }
}

mod adjt_mod {
    use ndarray::{Array2, ArrayView2};
    use ndarray_linalg::{Determinant, FactorizeInto, InverseInto, SVD};
    use crate::StateScalar;
    use super::det_mod::{det2scalar, det3scalar, det4};

    /// Calculate determinant of 1 by 1 matrix `a` and write its adjugate transpose into `adjt`.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `a`: Input matrix.
    /// # Returns
    /// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt1<T: StateScalar>(adjt: &mut [T], a: &[T]) -> Option<T> {
        adjt[0] = T::from_real(1.0);
        Some(a[0])
    }

    /// Calculate determinant of 2 by 2 matrix `a` and write its adjugate transpose into `adjt`.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `a`: Input matrix.
    /// # Returns
    /// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt2<T: StateScalar>(adjt: &mut [T], a: &[T]) -> Option<T> {
        let a00 = a[0]; let a01 = a[1];
        let a10 = a[2]; let a11 = a[3];

        let det = det2scalar(a00, a01, a10, a11);
        if !det.abs().is_finite() {return None;}

        adjt[0] = a11;
        adjt[1] = T::from_real(-1.0) * a10;
        adjt[2] = T::from_real(-1.0) * a01;
        adjt[3] = a00;

        Some(det)
    }

    /// Calculate determinant of 3 by 3 matrix `a` and write its adjugate transpose into `adjt`.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `a`: Input matrix.
    /// # Returns
    /// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt3<T: StateScalar>(adjt: &mut [T], a: &[T]) -> Option<T> {
        let a00 = a[0]; let a01 = a[1]; let a02 = a[2];
        let a10 = a[3]; let a11 = a[4]; let a12 = a[5];
        let a20 = a[6]; let a21 = a[7]; let a22 = a[8];

        let det = det3scalar(a00, a01, a02, a10, a11, a12, a20, a21, a22);
        if !det.abs().is_finite() {return None;}

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

        adjt[0] = c00; adjt[1] = c01; adjt[2] = c02;
        adjt[3] = c10; adjt[4] = c11; adjt[5] = c12;
        adjt[6] = c20; adjt[7] = c21; adjt[8] = c22;

        Some(det)
    }

    /// Calculate determinant of 4 by 4 matrix `a` and write its adjugate transpose into `adjt`.
    /// # Arguments:
    /// - `adjt`: Scratch space for writing adjugate transpose.
    /// - `a`: Input matrix.
    /// # Returns
    /// - `Option<T>`: Determinant of `a`, or `None` if evaluation fails.
    #[inline(always)]
    pub(super) fn adjt4<T: StateScalar>(adjt: &mut [T], a: &[T]) -> Option<T> {
        let det = det4(a);
        if !det.abs().is_finite() {return None;}

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

                let m00 = a[r[0] * 4 + c[0]];
                let m01 = a[r[0] * 4 + c[1]];
                let m02 = a[r[0] * 4 + c[2]];
                let m10 = a[r[1] * 4 + c[0]];
                let m11 = a[r[1] * 4 + c[1]];
                let m12 = a[r[1] * 4 + c[2]];
                let m20 = a[r[2] * 4 + c[0]];
                let m21 = a[r[2] * 4 + c[1]];
                let m22 = a[r[2] * 4 + c[2]];

                let minordet = det3scalar(m00, m01, m02, m10, m11, m12, m20, m21, m22);
                let sign = if ((i + j) & 1) == 0 {T::from_real(1.0)} else {T::from_real(-1.0)};
                adjt[i * 4 + j] = sign * minordet;
            }
        }

        Some(det)
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
    pub(super) fn adjt_gen<T: StateScalar>(adjt: &mut [T], invs: &mut [f64], lu: &mut [T], a: &[T], n: usize, thresh: f64) -> Option<T> {
        if let Some(det) = adjtlu(adjt, lu, a, n) {return Some(det);}
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
    pub(super) fn adjtlu<T: StateScalar>(adjt: &mut [T], lu: &mut [T], a: &[T], n: usize) -> Option<T> {
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
    pub(super) fn adjtsvd<T: StateScalar>(adjt: &mut [T], invs: &mut [f64], a: &[T], n: usize, thresh: f64) -> Option<T> {
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
