// nonorthogonalwicks/eval/helpers.rs

#[cfg(feature = "nocc")]
use ndarray::{Array2, ArrayView2, s};

use crate::ExcitationSpin;
#[cfg(feature = "nocc")]
use crate::maths::adjoint;
use crate::maths::{det, minor as build_minor, mix_columns};
use crate::noci::NOCIScalar;
use crate::time_call;

use super::super::layout::{idx, idx4};
use super::super::scratch::{IndexVec, Vec2, WickScratch};
use super::super::view::{SameSpinView, WicksPairView};

/// Row and column layout for determinant replacement operations.
#[derive(Clone, Copy)]
pub(super) struct ReplacementLayout<'a> {
    /// Tensor dimension.
    pub n: usize,
    /// Full contraction-determinant row labels.
    pub rows: &'a [usize],
    /// Full contraction-determinant column labels.
    pub cols: &'a [usize],
}

/// Row and column index in a determinant-like matrix.
#[derive(Clone, Copy)]
pub(super) struct DetIndex {
    /// Row index.
    pub row: usize,
    /// Column index.
    pub col: usize,
}

/// Branch-zero and branch-one determinant slices.
#[derive(Clone, Copy)]
pub(super) struct DetBranches<'a, T> {
    /// Branch-zero determinant.
    pub zero: &'a [T],
    /// Branch-one determinant.
    pub one: &'a [T],
}

/// Minor obtained by deleting one row and one column.
#[derive(Clone, Copy)]
pub(super) struct Minor {
    /// Dimension of the full determinant.
    pub l: usize,
    /// Row to remove.
    pub row: usize,
    /// Column to remove.
    pub col: usize,
}

/// Calculate determinant of a row-major square matrix slice.
/// # Arguments:
/// - `a`: Row-major matrix data.
/// - `n`: Matrix dimension.
/// # Returns
/// - `Option<T>`: Determinant if the matrix could be formed and factorised.
#[inline(always)]
pub(super) fn det_slice<T: NOCIScalar>(
    a: &[T],
    n: usize,
) -> Option<T> {
    det(a, n)
}

/// Calculate determinant of a row-major square matrix slice, returning zero if factorisation fails.
/// # Arguments:
/// - `a`: Row-major matrix data.
/// - `n`: Matrix dimension.
/// # Returns
/// - `T`: Determinant or zero if singular/factorisation failed.
#[inline(always)]
pub(super) fn det_or_zero<T: NOCIScalar>(
    a: &[T],
    n: usize,
) -> T {
    det_slice(a, n).unwrap_or(<T as From<f64>>::from(0.0))
}

/// Extend a Wick contraction determinant with RDM-basis rows and columns.
/// # Arguments:
/// - `d`: Wick contraction determinant indexed in `[left determinant MOs; right determinant MOs]`.
/// - `l_c`: Left determinant orbital coefficients in the RDM basis.
/// - `g_c`: Right determinant orbital coefficients in the RDM basis.
/// - `nmo`: Number of determinant orbitals in one spin block.
/// # Returns:
/// - `Array2<T>`: Extended Wick contraction determinant whose original determinant-orbital block
///   is unchanged, and whose final rows and columns open RDM-basis operators.
#[inline(always)]
#[cfg(feature = "nocc")]
pub(super) fn extend_rdm_d<T: NOCIScalar>(
    d: &ArrayView2<'_, T>,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    nmo: usize,
) -> Array2<T> {
    let npair = 2 * nmo;
    let nrdm = l_c.nrows();
    let mut out = Array2::<T>::zeros((npair + nrdm, npair + nrdm));

    out.slice_mut(s![0..npair, 0..npair]).assign(d);

    let left_rows = d.slice(s![0..nmo, ..]).to_owned();
    let right_cols = d.slice(s![.., nmo..npair]).to_owned();
    let g_dag = adjoint(g_c);

    let rdm_rows = l_c.dot(&left_rows);
    let rdm_cols = right_cols.dot(&g_dag);
    let rdm_block = rdm_rows.slice(s![.., nmo..npair]).dot(&g_dag);

    out.slice_mut(s![npair..npair + nrdm, 0..npair])
        .assign(&rdm_rows);
    out.slice_mut(s![0..npair, npair..npair + nrdm])
        .assign(&rdm_cols);
    out.slice_mut(s![npair..npair + nrdm, npair..npair + nrdm])
        .assign(&rdm_block);

    out
}

/// # Arguments:
/// - `mi, mj, mk, ml`: Zero distribution selectors.
/// # Returns
/// - `(usize, bool)`: Compressed slot index and whether swapped access is required.
pub(super) fn jslot(
    mi: usize,
    mj: usize,
    mk: usize,
    ml: usize,
) -> (usize, bool) {
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
/// - `layout`: Tensor dimension and determinant row/column labels.
/// - `removed`: Removed row and column indices in the full `l x l` slice.
/// - `minor`: Row and column indices in the `(l - 1) x (l - 1)` minor.
/// - `fixed`: Fixed tensor indices for the packed first and second axes.
/// - `swap`: Whether to use the swapped logical access.
/// # Returns
/// - `T`: Requested entry of the minor-column.
#[inline(always)]
pub(super) fn j_replacement<T: NOCIScalar>(
    jsl: &[T],
    layout: ReplacementLayout<'_>,
    removed: DetIndex,
    minor: DetIndex,
    fixed: DetIndex,
    swap: bool,
) -> T {
    let r_full = minor_to_full(minor.row, removed.row);
    let k_full = minor_to_full(minor.col, removed.col);
    let rr = layout.rows[r_full];
    let cc = layout.cols[k_full];

    if !swap {
        jsl[idx4(layout.n, fixed.row, fixed.col, rr, cc)]
    } else {
        jsl[idx4(layout.n, rr, cc, fixed.row, fixed.col)]
    }
}

/// Read one entry from the required replacement column of different-spin `IIab`.
/// # Arguments:
/// - `iisl`: Flat row-major storage of `IIab`.
/// - `layout`: Tensor dimension and determinant row/column labels.
/// - `entry`: Row and column indices in the full determinant column being corrected.
/// - `fixed`: Fixed tensor indices.
/// - `ijrc`: Whether to read the tensor as `t[i, j, r, c]` instead of `t[r, c, i, j]`.
/// # Returns
/// - `T`: Requested entry of the replacement column.
#[inline(always)]
pub(super) fn ii_replacement<T: NOCIScalar>(
    iisl: &[T],
    layout: ReplacementLayout<'_>,
    entry: DetIndex,
    fixed: DetIndex,
    ijrc: bool,
) -> T {
    let rr = layout.rows[entry.row];
    let cc = layout.cols[entry.col];

    if ijrc {
        iisl[idx4(layout.n, fixed.row, fixed.col, rr, cc)]
    } else {
        iisl[idx4(layout.n, rr, cc, fixed.row, fixed.col)]
    }
}

/// Map an index in a minor matrix back to the corresponding index in the full matrix.
/// # Arguments:
/// - `midx`: Index in the minor.
/// - `removed`: Removed row or column index in the full matrix.
/// # Returns
/// - `usize`: Corresponding index in the full matrix.
#[inline(always)]
pub(super) fn minor_to_full(
    midx: usize,
    removed: usize,
) -> usize {
    if midx < removed { midx } else { midx + 1 }
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
pub(super) fn bit(
    bits: u64,
    k: usize,
) -> usize {
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
pub(super) fn column_replacement_correction<T: NOCIScalar>(
    n: usize,
    old: &[T],
    cof: &[T],
    col: usize,
    mut new_at: impl FnMut(usize) -> T,
) -> T {
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
///   before the determinant-column bits begin.
/// - `scratch`: Wick's scratch space containing `det0`, `det1`, and `det_mix`.
/// - `f`: Closure applied once for each mixed determinant.
/// # Returns
/// - `()`: Writes the mixed determinant into `scratch.det_mix` and calls `f`.
#[inline(always)]
pub(super) fn mix_dets_same<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    pbits: usize,
    scratch: &mut WickScratch<T>,
    mut f: impl FnMut(u64, &mut WickScratch<T>),
) {
    let mut prev_cbits: Option<u64> = None;
    for_each_m_combination(l + pbits, w.m, |bits| {
        let cbits = bits >> pbits;
        match prev_cbits {
            None => {
                mix_columns(
                    scratch.det_mix.as_mut_slice(),
                    scratch.det0.as_slice(),
                    scratch.det1.as_slice(),
                    l,
                    cbits,
                );
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
pub(super) fn get_det_adjt_same<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    pbits: usize,
    scratch: &mut WickScratch<T>,
    tol: f64,
    mut f: impl FnMut(u64, &mut WickScratch<T>, T),
) {
    time_call!(crate::timers::nonorthogonalwicks::add_get_det_adjt_same, {
        mix_dets_same(w, l, pbits, scratch, |bits, scratch| {
            if let Some(det_det) = adjugate_transpose_generic(
                scratch.adjt_det.as_mut_slice(),
                scratch.det_mix.as_slice(),
                l,
                tol,
            ) {
                f(bits, scratch, det_det);
            }
        });
    })
}

/// Get determinant and adjugate transpose of each different-spin determinant corresponding to an allowed
/// bitstring. Mixes determinants for each bitstring and calls determinant routines.
/// # Arguments:
/// - `w`: Diff-spin pair Wick's view.
/// - `rank`: Alpha and beta determinant dimensions.
/// - `scratch`: Scratch space for mixed determinant, inverse workspace, and cofactors.
/// - `deta`: Branch-zero and branch-one alpha determinants.
/// - `detb`: Branch-zero and branch-one beta determinants.
/// - `tol`: Singularity threshold.
/// - `f`: Closure receiving the packed alpha/beta bitstrings, scratch space, and determinant values.
/// # Returns
/// - `()`: Calls `f` only for nonsingular mixed determinants.
#[inline(always)]
pub(super) fn get_det_adjt_diff<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    rank: (usize, usize),
    scratch: &mut WickScratch<T>,
    deta: DetBranches<'_, T>,
    detb: DetBranches<'_, T>,
    tol: f64,
    mut f: impl FnMut(u64, u64, &mut WickScratch<T>, T, T),
) {
    let (la, lb) = rank;

    time_call!(crate::timers::nonorthogonalwicks::add_get_det_adjt_diff, {
        for_each_m_combination(la + 1, w.aa.m, |bits_a| {
            let inda = bits_a >> 1;
            mix_columns(
                scratch.deta_mix.as_mut_slice(),
                deta.zero,
                deta.one,
                la,
                inda,
            );
            if let Some(det_a) = adjugate_transpose_generic(
                scratch.adjt_deta.as_mut_slice(),
                scratch.deta_mix.as_slice(),
                la,
                tol,
            ) {
                for_each_m_combination(lb + 1, w.bb.m, |bits_b| {
                    let indb = bits_b >> 1;
                    mix_columns(
                        scratch.detb_mix.as_mut_slice(),
                        detb.zero,
                        detb.one,
                        lb,
                        indb,
                    );
                    if let Some(det_b) = adjugate_transpose_generic(
                        scratch.adjt_detb.as_mut_slice(),
                        scratch.detb_mix.as_slice(),
                        lb,
                        tol,
                    ) {
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
/// - `minor`: Dimension and removed row/column of the minor.
/// - `minorb`: Scratch storage for the minor determinant.
/// - `adjtb`: Scratch storage for the adjugate-transpose of the minor.
/// - `tol`: Singularity threshold.
/// - `f`: Closure receiving the minor dimension, minor entries, cofactors, and determinant.
/// # Returns
/// - `()`: Calls `f` only if the minor determinant is nonsingular.
#[inline(always)]
pub(super) fn minor_adjt<T: NOCIScalar>(
    full: &[T],
    minor: Minor,
    minorb: &mut Vec2<T>,
    adjtb: &mut Vec2<T>,
    tol: f64,
    mut f: impl FnMut(usize, &[T], &[T], T),
) {
    let lm1 = minor.l.saturating_sub(1);
    build_minor(minorb.as_mut_slice(), full, minor.l, minor.row, minor.col);
    if let Some(det_minor) =
        adjugate_transpose_generic(adjtb.as_mut_slice(), minorb.as_slice(), lm1, tol)
    {
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
pub(super) fn construct_determinant_indices_gen(
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    nmo: usize,
    rows: &mut IndexVec,
    cols: &mut IndexVec,
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_construct_determinant_indices_gen,
        {
            let nl = l_ex.holes.len();
            let ng = g_ex.holes.len();
            let need = nl + ng;
            rows.ensure(need);
            cols.ensure(need);
            let rows = rows.as_mut_slice();
            let cols = cols.as_mut_slice();

            rows[..nl].copy_from_slice(&l_ex.parts);
            cols[..nl].copy_from_slice(&l_ex.holes);

            for (row, &hole) in rows[nl..].iter_mut().zip(&g_ex.holes) {
                *row = nmo + hole;
            }
            for (col, &part) in cols[nl..].iter_mut().zip(&g_ex.parts) {
                *col = nmo + part;
            }
        }
    )
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
pub(super) fn for_each_m_combination(
    l: usize,
    m: usize,
    mut f: impl FnMut(u64),
) {
    if m > l {
        return;
    }
    if m == 0 {
        f(0);
        return;
    }
    if m == l {
        f((1u64 << l) - 1);
        return;
    }

    let limit = 1u64 << l;
    let mut x = (1u64 << m) - 1;
    while x < limit {
        f(x);
        let c = x & x.wrapping_neg();
        let r = x + c;
        x = (((r ^ x) >> 2) / c) | r;
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
fn adjugate_transpose_generic<T: NOCIScalar>(
    adjt: &mut [T],
    full: &[T],
    n: usize,
    tol: f64,
) -> Option<T> {
    let detv = det_slice(full, n)?;
    if detv.abs() <= tol {
        return None;
    }

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
            build_minor(&mut minor, full, n, r, c);
            let md = det_or_zero(&minor, n - 1);
            let sign = if ((r + c) & 1) == 0 { 1.0 } else { -1.0 };
            adjt[idx(n, r, c)] = <T as From<f64>>::from(sign) * md;
        }
    }

    Some(detv)
}
