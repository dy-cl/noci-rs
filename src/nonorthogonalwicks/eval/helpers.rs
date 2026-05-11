// nonorthogonalwicks/eval/helpers.rs
use ndarray::Array2;
use ndarray_linalg::Determinant;

use crate::ExcitationSpin;
use crate::noci::NOCIScalar;
use crate::time_call;

use super::super::layout::{idx, idx4};
use super::super::scratch::{Vec1, Vec2, WickScratch};
use super::super::view::{SameSpinView, WicksPairView};

/// Calculate determinant of a row-major square matrix slice.
/// # Arguments:
/// - `a`: Row-major matrix data.
/// - `n`: Matrix dimension.
/// # Returns
/// - `Option<T>`: Determinant if the matrix could be formed and factorised.
#[inline(always)]
pub(super) fn det_slice<T: NOCIScalar>(a: &[T], n: usize) -> Option<T> {
    if n == 0 {return Some(<T as From<f64>>::from(1.0));}
    let m = Array2::<T>::from_shape_vec((n, n), a[..n * n].to_vec()).ok()?;
    m.det().ok()
}

/// Calculate determinant of a row-major square matrix slice, returning zero if factorisation fails.
/// # Arguments:
/// - `a`: Row-major matrix data.
/// - `n`: Matrix dimension.
/// # Returns
/// - `T`: Determinant or zero if singular/factorisation failed.
#[inline(always)]
pub(super) fn det_or_zero<T: NOCIScalar>(a: &[T], n: usize) -> T {
    det_slice(a, n).unwrap_or(<T as From<f64>>::from(0.0))
}

/// # Arguments:
/// - `mi, mj, mk, ml`: Zero distribution selectors.
/// # Returns
/// - `(usize, bool)`: Compressed slot index and whether swapped access is required.
pub(super) fn jslot(mi: usize, mj: usize, mk: usize, ml: usize,) -> (usize, bool) {
    match (mi, mj, mk, ml) {
        (0, 0, 0, 0) => (0, false),
        (0, 0, 0, 1) => (1, false),
        (0, 1, 0, 0) => (1, true),
        (0, 0, 1, 0) => (2, false),
        (1, 0, 0, 0) => (2, true),
        (0, 0, 1, 1) => (3, false),
        (1, 1, 0, 0) => (3, true),
        (0, 1, 0, 1) => (4, false),
        (0, 1, 1, 0) => (5, false),
        (1, 0, 0, 1) => (5, true),
        (0, 1, 1, 1) => (6, false),
        (1, 1, 0, 1) => (6, true),
        (1, 0, 1, 0) => (7, false),
        (1, 0, 1, 1) => (8, false),
        (1, 1, 1, 0) => (8, true),
        (1, 1, 1, 1) => (9, false),
        _ => unreachable!(),
    }
}

/// Read one entry from the required minor-column of same-spin `J`.
/// # Arguments:
/// - `jsl`: Flat row-major storage of `J`.
/// - `n`: Tensor dimension.
/// - `rows`: Full contraction-determinant row labels.
/// - `cols`: Full contraction-determinant column labels.
/// - `i_rm`: Removed row index in the full `l x l` slice.
/// - `j_rm`: Removed column index in the full `l x l` slice.
/// - `r_minor`: Row index in the `(l - 1) x (l - 1)` minor.
/// - `k_minor`: Column index in the `(l - 1) x (l - 1)` minor.
/// - `i_fixed`: Fixed tensor index for the packed first axis.
/// - `j_fixed`: Fixed tensor index for the packed second axis.
/// - `swap`: Whether to use the swapped logical access.
/// # Returns
/// - `T`: Requested entry of the minor-column.
#[inline(always)]
pub(super) fn j_replacement<T: NOCIScalar>(jsl: &[T], n: usize, rows: &[usize], cols: &[usize], i_rm: usize, j_rm: usize, r_minor: usize, k_minor: usize, i_fixed: usize, j_fixed: usize, swap: bool) -> T {
    let r_full = minor_to_full(r_minor, i_rm);
    let k_full = minor_to_full(k_minor, j_rm);
    let rr = rows[r_full];
    let cc = cols[k_full];
    if !swap {
        jsl[idx4(n, i_fixed, j_fixed, rr, cc)]
    } else {
        jsl[idx4(n, rr, cc, i_fixed, j_fixed)]
    }
}

/// Read one entry from the required replacement column of different-spin `IIab`.
/// # Arguments:
/// - `iisl`: Flat row-major storage of `IIab`.
/// - `n`: Tensor dimension.
/// - `rows`: Full contraction-determinant row labels.
/// - `cols`: Full contraction-determinant column labels.
/// - `r_full`: Row index in the full determinant column being corrected.
/// - `k_full`: Column index in the full determinant column being corrected.
/// - `i_fixed`: First fixed tensor index.
/// - `j_fixed`: Second fixed tensor index.
/// - `ijrc`: Whether to read the tensor as `t[i, j, r, c]` instead of `t[r, c, i, j]`.
/// # Returns
/// - `T`: Requested entry of the replacement column.
#[inline(always)]
pub(super) fn ii_replacement<T: NOCIScalar>(iisl: &[T], n: usize, rows: &[usize], cols: &[usize], r_full: usize, k_full: usize, i_fixed: usize, j_fixed: usize, ijrc: bool) -> T {
    let rr = rows[r_full];
    let cc = cols[k_full];
    if ijrc {
        iisl[idx4(n, i_fixed, j_fixed, rr, cc)]
    } else {
        iisl[idx4(n, rr, cc, i_fixed, j_fixed)]
    }
}

/// Map an index in a minor matrix back to the corresponding index in the full matrix.
/// # Arguments:
/// - `midx`: Index in the minor.
/// - `removed`: Removed row or column index in the full matrix.
/// # Returns
/// - `usize`: Corresponding index in the full matrix.
#[inline(always)]
pub(super) fn minor_to_full(midx: usize, removed: usize) -> usize {
    if midx < removed {midx} else {midx + 1}
}

/// Extract bit `k` from a distribution bitstring. Each possible bitstring assigns `m` zero-overlap
/// orbital pairs to a given contraction. The bits are also used to select whether a given
/// contraction uses the `0` or `1` branch of intermediates.
/// # Arguments:
/// - `bits`: Bitstring encoding a zero distribution.
/// - `k`: Index of the bit to extract.
/// # Returns
/// - `usize`: `0` or `1` depending on the selected branch for contraction `k`.
#[inline(always)]
pub(super) fn bit(bits: u64, k: usize) -> usize {
    ((bits >> k) & 1) as usize
}

/// Calculate determinant correction obtained by replacing one column of a determinant.
/// If one column of determinant `D` is replaced by a new column, the determinant update correction `C` may be
/// written as C = \sum_r (N_r - O_r) * A_r, where N_r is the new entry in row `r` of replacement
/// column, O_r is the old entry and A_r is the cofactor of row `r` in that column.
/// # Arguments:
/// - `n`: Dimension of the determinant.
/// - `old`: Row-major storage of the original determinant.
/// - `cof`: Row-major storage of the adjugate-transpose / cofactor matrix.
/// - `col`: Column index to replace.
/// - `new_at`: Closure returning the new value for row `r` in the replacement column.
/// # Returns
/// - `T`: Cofactor contraction for the chosen column replacement.
#[inline(always)]
pub(super) fn column_replacement_correction<T: NOCIScalar>(n: usize, old: &[T], cof: &[T], col: usize, mut new_at: impl FnMut(usize) -> T) -> T {
    let mut correction = <T as From<f64>>::from(0.0);
    for r in 0..n {
        let i = idx(n, r, col);
        correction += (new_at(r) - old[i]) * cof[i];
    }
    correction
}

/// Form same spin mixed contraction determinants for all allowed bitstrings.
/// For a same-spin matrix element with excitation rank `l` we require a sum over all possible ways
/// to distribute `m` zero-overlap orbital pairs across contractions. In determinant form this is a
/// sum over mixed determinants obtained by choosing columns from the `det0` and `det1` branch.
/// # Arguments:
/// - `w`: Same-spin view containing the number of zero-overlap pairs `m`.
/// - `l`: Excitation rank.
/// - `pbits`: Number of leading bits reserved for operator-specific selectors
/// before the determinant-column bits begin.
/// - `scratch`: Wick's scratch space containing `det0`, `det1`, and `det_mix`.
/// - `f`: Closure applied once for each mixed determinant.
/// # Returns
/// - `()`: Writes the mixed determinant into `scratch.det_mix` and calls `f`.
#[inline(always)]
pub(super) fn mix_dets_same<T: NOCIScalar>(w: &SameSpinView<'_, T>, l: usize, pbits: usize, scratch: &mut WickScratch<T>, mut f: impl FnMut(u64, &mut WickScratch<T>)) {
    let mut prev_cbits: Option<u64> = None;
    for_each_m_combination(l + pbits, w.m, |bits| {
        let cbits = bits >> pbits;
        match prev_cbits {
            None => {
                mix_columns(scratch.det_mix.as_mut_slice(), scratch.det0.as_slice(), scratch.det1.as_slice(), l, cbits);
            }
            Some(prev) => {
                let mut changed = prev ^ cbits;
                while changed != 0 {
                    let col = changed.trailing_zeros() as usize;
                    if bit(cbits, col) == 0 {
                        let src = scratch.det0.as_slice();
                        let dst = scratch.det_mix.as_mut_slice();
                        for r in 0..l {
                            let i = idx(l, r, col);
                            dst[i] = src[i];
                        }
                    } else {
                        let src = scratch.det1.as_slice();
                        let dst = scratch.det_mix.as_mut_slice();
                        for r in 0..l {
                            let i = idx(l, r, col);
                            dst[i] = src[i];
                        }
                    }
                    changed &= changed - 1;
                }
            }
        }
        prev_cbits = Some(cbits);
        f(bits, scratch);
    });
}

/// Get determinant and adjugate transpose of each same-spin determinant corresponding to an allowed
/// bitstring. Mixes determinants for each bitstring and calls determinant routines.
/// # Arguments:
/// - `w`: Same-spin Wick's view.
/// - `l`: Determinant dimension.
/// - `pbits`: Number of non-column selector bits at the front of the bitstring.
/// - `scratch`: Scratch space for mixed determinant, inverse workspace, and cofactors.
/// - `tol`: Singularity threshold.
/// - `f`: Closure receiving the packed bitstring, scratch space, and determinant value.
/// # Returns
/// - `()`: Calls `f` only for nonsingular mixed determinants.
#[inline(always)]
pub(super) fn get_det_adjt_same<T: NOCIScalar>(w: &SameSpinView<'_, T>, l: usize, pbits: usize, scratch: &mut WickScratch<T>, tol: f64, mut f: impl FnMut(u64, &mut WickScratch<T>, T)) {
    time_call!(crate::timers::nonorthogonalwicks::add_get_det_adjt_same, {
        mix_dets_same(w, l, pbits, scratch, |bits, scratch| {
            if let Some(det_det) = adjugate_transpose_generic(scratch.adjt_det.as_mut_slice(), scratch.det_mix.as_slice(), l, tol) {
                f(bits, scratch, det_det);
            }
        });
    })
}

/// Get determinant and adjugate transpose of each different-spin determinant corresponding to an allowed
/// bitstring. Mixes determinants for each bitstring and calls determinant routines.
/// # Arguments:
/// - `w`: Diff-spin pair Wick's view.
/// - `l`: Determinant dimension.
/// - `pbits`: Number of non-column selector bits at the front of the bitstring.
/// - `scratch`: Scratch space for mixed determinant, inverse workspace, and cofactors.
/// - `tol`: Singularity threshold.
/// - `f`: Closure receiving the packed bitstring, scratch space, and determinant value.
/// # Returns
/// - `()`: Calls `f` only for nonsingular mixed determinants.
#[inline(always)]
pub(super) fn get_det_adjt_diff<T: NOCIScalar>(w: &WicksPairView<'_, T>, la: usize, lb: usize, scratch: &mut WickScratch<T>, deta0: &[T], deta1: &[T], detb0: &[T], detb1: &[T], tol: f64, mut f: impl FnMut(u64, u64, &mut WickScratch<T>, T, T)) {
    time_call!(crate::timers::nonorthogonalwicks::add_get_det_adjt_diff, {
        for_each_m_combination(la + 1, w.aa.m, |bits_a| {
            let inda = bits_a >> 1;
            mix_columns(scratch.deta_mix.as_mut_slice(), deta0, deta1, la, inda);
            if let Some(det_a) = adjugate_transpose_generic(scratch.adjt_deta.as_mut_slice(), scratch.deta_mix.as_slice(), la, tol) {
                for_each_m_combination(lb + 1, w.bb.m, |bits_b| {
                    let indb = bits_b >> 1;
                    mix_columns(scratch.detb_mix.as_mut_slice(), detb0, detb1, lb, indb);
                    if let Some(det_b) = adjugate_transpose_generic(scratch.adjt_detb.as_mut_slice(), scratch.detb_mix.as_slice(), lb, tol) {
                        f(bits_a, bits_b, scratch, det_a, det_b);
                    }
                });
            }
        });
    })
}

/// Form the `l - 1` by `l - 1` minor of a determinant and find its adjugate transpose.
/// # Arguments:
/// - `full`: Row-major storage of the determinant.
/// - `l`: Dimension of the determinant.
/// - `i`: Row to remove.
/// - `j`: Column to remove.
/// - `minorb`: Scratch storage for the minor determinant.
/// - `adjtb`: Scratch storage for the adjugate-transpose of the minor.
/// - `invsb`: Scratch storage for inverse-related data.
/// - `lun`: Scratch storage for the LU factorization.
/// - `tol`: Singularity threshold.
/// - `f`: Closure receiving the minor dimension, minor entries, cofactors, and determinant.
/// # Returns
/// - `()`: Calls `f` only if the minor determinant is nonsingular.
#[inline(always)]
pub(super) fn minor_adjt<T: NOCIScalar>(full: &[T], l: usize, i: usize, j: usize, minorb: &mut Vec2<T>, adjtb: &mut Vec2<T>, _invsb: &mut Vec1<f64>, 
                                        _lub: &mut Vec2<T>, tol: f64, mut f: impl FnMut(usize, &[T], &[T], T)) {
    let lm1 = l.saturating_sub(1);
    minor_generic(minorb.as_mut_slice(), full, l, i, j);
    if let Some(det_minor) = adjugate_transpose_generic(adjtb.as_mut_slice(), minorb.as_slice(), lm1, tol) {
        f(lm1, minorb.as_slice(), adjtb.as_slice(), det_minor);
    }
}

/// Construct the row and column indices used for a contraction determinant.
/// Indices are written in the concatenated orbital space `[Lambda orbitals; Gamma orbitals]`,
/// so any Gamma index is offset by `nmo`.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra (`Lambda`) determinant.
/// - `g_ex`: Excitation defining the ket (`Gamma`) determinant.
/// - `nmo`: Number of molecular orbitals in a single determinant.
/// - `rows`: Output row indices.
/// - `cols`: Output column indices.
/// # Returns
/// - `()`: Writes the contraction determinant indices into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices_gen(l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, nmo: usize, rows: &mut Vec<usize>, cols: &mut Vec<usize>) {
    time_call!(crate::timers::nonorthogonalwicks::add_construct_determinant_indices_gen, {
        let nl = l_ex.holes.len();
        let ng = g_ex.holes.len();
        rows.clear();
        cols.clear();
        let need = nl + ng;
        if rows.capacity() < need {rows.reserve_exact(need - rows.capacity());}
        if cols.capacity() < need {cols.reserve_exact(need - cols.capacity());}

        if nl == 0 && ng == 0 {
            return;
        }

        if nl > 0 && ng == 0 {
            for &a in &l_ex.parts {rows.push(a);}
            for &i in &l_ex.holes {cols.push(i);}
            return;
        }

        if nl == 0 && ng > 0 {
            for &i in &g_ex.holes {rows.push(nmo + i);}
            for &a in &g_ex.parts {cols.push(nmo + a);}
            return;
        }

        for &a in &l_ex.parts {rows.push(a);}
        for &i in &g_ex.holes {rows.push(nmo + i);}
        for &i in &l_ex.holes {cols.push(i);}
        for &a in &g_ex.parts {cols.push(nmo + a);}
    })
}

/// Call `f` for every bitstring of length `l` containing exactly `m` set bits.
/// These bitstrings enumerate all allowed zero-distribution patterns.
/// # Arguments:
/// - `l`: Bitstring length.
/// - `m`: Number of set bits required.
/// - `f`: Callback applied to each valid bitstring.
/// # Returns
/// - `()`: Calls `f` once for each valid combination.
#[inline(always)]
pub(super) fn for_each_m_combination(l: usize, m: usize, mut f: impl FnMut(u64)) {
    if m > l {return;}
    if m == 0 {f(0); return;}
    if m == l {f((1u64 << l) - 1); return;}

    let limit = 1u64 << l;
    let mut x = (1u64 << m) - 1;
    while x < limit {
        f(x);
        let c = x & x.wrapping_neg();
        let r = x + c;
        x = (((r ^ x) >> 2) / c) | r;
    }
}

/// Copy columns from det0 or det1 into output according to bitstring.
/// # Arguments:
/// - `out`: Output mixed determinant.
/// - `det0`: Branch-zero determinant.
/// - `det1`: Branch-one determinant.
/// - `n`: Determinant dimension.
/// - `bits`: Bitstring selecting branch-one columns.
/// # Returns
/// - `()`: Writes mixed determinant.
#[inline(always)]
fn mix_columns<T: NOCIScalar>(out: &mut [T], det0: &[T], det1: &[T], n: usize, bits: u64) {
    for c in 0..n {
        let src = if bit(bits, c) == 0 {det0} else {det1};
        for r in 0..n {
            out[idx(n, r, c)] = src[idx(n, r, c)];
        }
    }
}

/// Form minor by deleting row i and column j from full matrix.
/// # Arguments:
/// - `out`: Output minor storage.
/// - `full`: Full determinant storage.
/// - `n`: Dimension of full determinant.
/// - `i`: Row to remove.
/// - `j`: Column to remove.
/// # Returns
/// - `()`: Writes minor.
#[inline(always)]
fn minor_generic<T: NOCIScalar>(out: &mut [T], full: &[T], n: usize, i: usize, j: usize) {
    let mut rr = 0;
    for r in 0..n {
        if r == i {continue;}
        let mut cc = 0;
        for c in 0..n {
            if c == j {continue;}
            out[idx(n - 1, rr, cc)] = full[idx(n, r, c)];
            cc += 1;
        }
        rr += 1;
    }
}

/// Compute determinant and adjugate transpose/cofactor matrix.
/// # Arguments:
/// - `adjt`: Output adjugate-transpose/cofactor matrix.
/// - `full`: Full determinant storage.
/// - `n`: Matrix dimension.
/// - `tol`: Singularity threshold.
/// # Returns
/// - `Option<T>`: Determinant if nonsingular.
#[inline(always)]
fn adjugate_transpose_generic<T: NOCIScalar>(adjt: &mut [T], full: &[T], n: usize, tol: f64) -> Option<T> {
    let detv = det_slice(full, n)?;
    if detv.abs() <= tol {return None;}

    if n == 0 {
        return Some(<T as From<f64>>::from(1.0));
    }

    if n == 1 {
        adjt[0] = <T as From<f64>>::from(1.0);
        return Some(detv);
    }

    let mut minor = vec![<T as From<f64>>::from(0.0); (n - 1) * (n - 1)];
    for r in 0..n {
        for c in 0..n {
            minor_generic(&mut minor, full, n, r, c);
            let md = det_or_zero(&minor, n - 1);
            let sign = if ((r + c) & 1) == 0 {1.0} else {-1.0};
            adjt[idx(n, r, c)] = <T as From<f64>>::from(sign) * md;
        }
    }

    Some(detv)
}
