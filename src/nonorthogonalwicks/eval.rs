// nonorthogonalwicks/eval.rs
use crate::{ExcitationSpin};
use super::scratch::{Vec1, Vec2, WickScratch};
use super::view::{SameSpinView, WicksPairView};

use crate::maths::{minor, adjugate_transpose, det, mix_columns, build_d};
use crate::time_call;
use crate::timers::nonorthogonalwicks as wick_timers;
use super::layout::{idx, idx4};

/// # Arguments:
/// - `mi, mj, mk, ml`: Zero distribution selectors.
/// # Returns
/// - `(usize, bool)`: Compressed slot index and whether swapped access is required.
fn jslot(mi: usize, mj: usize, mk: usize, ml: usize,) -> (usize, bool) {
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
/// - `f64`: Requested entry of the minor-column.
#[inline(always)]
fn j_replacement(jsl: &[f64], n: usize, rows: &[usize], cols: &[usize], i_rm: usize, j_rm: usize, r_minor: usize, k_minor: usize, i_fixed: usize, j_fixed: usize, swap: bool) -> f64 {
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
/// - `f64`: Requested entry of the replacement column.
#[inline(always)]
fn ii_replacement(iisl: &[f64], n: usize, rows: &[usize], cols: &[usize], r_full: usize, k_full: usize, i_fixed: usize, j_fixed: usize, ijrc: bool) -> f64 {
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
fn minor_to_full(midx: usize, removed: usize) -> usize {
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
fn bit(bits: u64, k: usize) -> usize {
    ((bits >> k) & 1) as usize
}

/// Calculate determinant correction obtained by replacing one column of a determinant. If one
/// column of determinant `D` is replaced by a new column, the determinant update correction `C` may be
/// written as C = \sum_r (N_r - O_r) * A_r, where N_r is the new entry in row `r` of replacement
/// column, O_r is the old entry and A_r is the cofactor of row `r` in that column. 
/// # Arguments:
/// - `n`: Dimension of the determinant.
/// - `old`: Row-major storage of the original determinant.
/// - `cof`: Row-major storage of the adjugate-transpose / cofactor matrix.
/// - `col`: Column index to replace.
/// - `new_at`: Closure returning the new value for row `r` in the replacement column.
/// # Returns
/// - `f64`: Cofactor contraction for the chosen column replacement.
#[inline(always)]
fn column_replacement_correction(n: usize, old: &[f64], cof: &[f64], col: usize, mut new_at: impl FnMut(usize) -> f64) -> f64 {
    let mut correction = 0.0;
    for r in 0..n {
        let i = idx(n, r, col);
        correction += (new_at(r) - old[i]) * cof[i];
    }
    correction
}

/// Form same spin mixed contraction determinants for all allowed bitstrings. For a same-spin matrix element 
/// with excitation rank `l` we require a sum over all possible ways to distribute `m` zero-overlap 
/// orbital pairs across contractions. In determinant form this is a sum over mixed determinants 
/// obtained by choosing columns from the `det0` and `det1` branch.
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
fn mix_dets_same(w: &SameSpinView<'_>, l: usize, pbits: usize, scratch: &mut WickScratch, mut f: impl FnMut(u64, &mut WickScratch),) {
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
/// - `scratch`: Wick's scratch space containing `det0`, `det1`, and `det_mix`.
/// - `scratch`: Scratch space for mixed determinant, inverse workspace, and cofactors.
/// - `tol`: Singularity threshold.
/// - `f`: Closure receiving the packed bitstring, scratch space, and determinant value.
/// # Returns
/// - `()`: Calls `f` only for nonsingular mixed determinants.
#[inline(always)]
fn get_det_adjt_same(w: &SameSpinView<'_>, l: usize, pbits: usize, scratch: &mut WickScratch, tol: f64, mut f: impl FnMut(u64, &mut WickScratch, f64)) {
    time_call!(wick_timers::add_get_det_adjt_same, {
        mix_dets_same(w, l, pbits, scratch, |bits, scratch| {
            if let Some(det_det) = adjugate_transpose(scratch.adjt_det.as_mut_slice(), scratch.invs.as_mut_slice(), scratch.lu.as_mut_slice(), scratch.det_mix.as_slice(), l, tol) {
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
/// - `scratch`: Wick's scratch space containing `det0`, `det1`, and `det_mix`.
/// - `scratch`: Scratch space for mixed determinant, inverse workspace, and cofactors.
/// - `tol`: Singularity threshold.
/// - `f`: Closure receiving the packed bitstring, scratch space, and determinant value.
/// # Returns
/// - `()`: Calls `f` only for nonsingular mixed determinants.
#[inline(always)]
fn get_det_adjt_diff(w: &WicksPairView<'_>, la: usize, lb: usize, scratch: &mut WickScratch, deta0: &[f64], deta1: &[f64], 
                     detb0: &[f64], detb1: &[f64], tol: f64, mut f: impl FnMut(u64, u64, &mut WickScratch, f64, f64)) {
    time_call!(wick_timers::add_get_det_adjt_diff, {
        for_each_m_combination(la + 1, w.aa.m, |bits_a| {
            let inda = bits_a >> 1;
            mix_columns(scratch.deta_mix.as_mut_slice(), deta0, deta1, la, inda);

            if let Some(det_a) = adjugate_transpose(scratch.adjt_deta.as_mut_slice(), scratch.invsla.as_mut_slice(), scratch.lua.as_mut_slice(), scratch.deta_mix.as_slice(), la, tol) {
                for_each_m_combination(lb + 1, w.bb.m, |bits_b| {
                    let indb = bits_b >> 1;
                    mix_columns(scratch.detb_mix.as_mut_slice(), detb0, detb1, lb, indb);

                    if let Some(det_b) = adjugate_transpose(scratch.adjt_detb.as_mut_slice(), scratch.invslb.as_mut_slice(), scratch.lub.as_mut_slice(), scratch.detb_mix.as_slice(), lb, tol) {
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
fn minor_adjt(full: &[f64], l: usize, i: usize, j: usize, minorb: &mut Vec2, adjtb: &mut Vec2, invsb: &mut Vec1, lub: &mut Vec2, tol: f64, mut f: impl FnMut(usize, &[f64], &[f64], f64)) {
    let lm1 = l.saturating_sub(1);
    minor(minorb.as_mut_slice(), full, l, i, j);

    if let Some(det_minor) = adjugate_transpose(adjtb.as_mut_slice(), invsb.as_mut_slice(), lub.as_mut_slice(), minorb.as_slice(), lm1, tol) {
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
fn construct_determinant_indices(l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, nmo: usize, rows: &mut Vec<usize>, cols: &mut Vec<usize>) {
    time_call!(wick_timers::add_construct_determinant_indices, {
        let nl = l_ex.holes.len();
        let ng = g_ex.holes.len();

        rows.clear();
        cols.clear();

        let need = nl + ng;
        if rows.capacity() < need {
            rows.reserve_exact(need - rows.capacity());
        }
        if cols.capacity() < need {
            cols.reserve_exact(need - cols.capacity());
        }

        if nl == 0 && ng == 0 {
            return;
        }

        if nl > 0 && ng == 0 {
            for &a in &l_ex.parts {
                rows.push(a);
            }
            for &i in &l_ex.holes {
                cols.push(i);
            }
            return;
        }

        if nl == 0 && ng > 0 {
            for &i in &g_ex.holes {
                rows.push(nmo + i);
            }
            for &a in &g_ex.parts {
                cols.push(nmo + a);
            }
            return;
        }

        for &a in &l_ex.parts {
            rows.push(a);
        }
        for &i in &g_ex.holes {
            rows.push(nmo + i);
        }
        for &i in &l_ex.holes {
            cols.push(i);
        }
        for &a in &g_ex.parts {
            cols.push(nmo + a);
        }
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
fn for_each_m_combination(l: usize, m: usize, mut f: impl FnMut(u64)) {
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

/// All the same spin routines require a number of the same quantities. It is more efficient to
/// precompute them once here rather than inside each matrix element routine.
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle. 
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `()`: Prepares the shared same-spin scratch quantities in place.
pub fn prepare_same(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) {
    time_call!(wick_timers::add_prepare_same, {
        let l = l_ex.holes.len() + g_ex.holes.len();
        scratch.ensure_same(l);

        construct_determinant_indices(l_ex, g_ex, w.nmo, &mut scratch.rows, &mut scratch.cols);

        let x0 = w.x(0);
        let y0 = w.y(0);
        let x1 = w.x(1);
        let y1 = w.y(1);

        build_d(scratch.det0.as_mut_slice(), l, &x0, &y0, scratch.rows.as_slice(), scratch.cols.as_slice());
        build_d(scratch.det1.as_mut_slice(), l, &x1, &y1, scratch.rows.as_slice(), scratch.cols.as_slice());
    })
}

/// Calculate overlap matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. Utilises
/// a sum over possible ways to distribute zeros across the columns of the L by L determinant.
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle. 
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element.
pub fn lg_overlap(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) -> f64 {
    time_call!(wick_timers::add_lg_overlap, {
        let l = l_ex.holes.len() + g_ex.holes.len();

        if w.m == 0 {
            return w.phase * w.tilde_s_prod * det(scratch.det0.as_slice(), l).unwrap_or(0.0);
        }
        if w.m == l {
            return w.phase * w.tilde_s_prod * det(scratch.det1.as_slice(), l).unwrap_or(0.0);
        }

        let mut acc = 0.0;
        mix_dets_same(w, l, 0, scratch, |_, scratch| {
            if let Some(d) = det(scratch.det_mix.as_slice(), l) {
                acc += d;
            }
        });
        w.phase * w.tilde_s_prod * acc
    })
}

#[derive(Clone, Copy)]
enum OneBody {
    /// Select Hamiltonian one-body intermediates.
    H1,
    /// Select Fock one-body intermediates.
    Fock,
}

/// Read the scalar zeroth-order one-body contribution for the chosen branch.
/// # Arguments:
/// - `w`: Same-spin Wick's view.
/// - `ob`: Selects Hamiltonian or Fock intermediates.
/// - `mi`: Branch selector for the operator contraction.
/// # Returns
/// - `f64`: Zeroth-order one-body scalar contribution.
#[inline(always)]
fn one_body_scalar(w: &SameSpinView<'_>, ob: OneBody, mi: usize) -> f64 {
    match ob {OneBody::H1 => w.f0h[mi], OneBody::Fock => w.f0f[mi]}
}

/// Calculate one electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: One-electron Hamiltonian matrix element.
#[inline(always)]
pub(crate) fn lg_h1(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    time_call!(wick_timers::add_lg_h1, {
        lg_one_body(w, l_ex, g_ex, scratch, tol, OneBody::H1)
    })
}

/// Calculate one electron Fock matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: One-electron Fock matrix element.
#[inline(always)]
pub(crate) fn lg_f(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64,) -> f64 {
    lg_one_body(w, l_ex, g_ex, scratch, tol, OneBody::Fock)
}

/// Calculate one body matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: One-electron Hamiltonian matrix element.
#[inline(always)]
fn lg_one_body(w: &SameSpinView<'_>, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64, ob: OneBody,) -> f64 {
    let l = l_ex.holes.len() + g_ex.holes.len();

    let mut acc = 0.0;
    let n = w.n();

    get_det_adjt_same(w, l, 1, scratch, tol, |bits, scratch, det_det| {
        let mi = bit(bits, 0);
        let mut contrib = det_det * one_body_scalar(w, ob, mi);

        let f0 = match ob {
            OneBody::H1 => w.fh_t_slice(mi, 0),
            OneBody::Fock => w.ff_t_slice(mi, 0),
        };
        let f1 = match ob {
            OneBody::H1 => w.fh_t_slice(mi, 1),
            OneBody::Fock => w.ff_t_slice(mi, 1),
        };

        for b in 0..l {
            let mj = bit(bits, b + 1);
            let cb = scratch.cols[b];
            let fsl = if mj == 0 {f0} else {f1};
            let base = cb * n;

            let corr = column_replacement_correction(l, scratch.det_mix.as_slice(), scratch.adjt_det.as_slice(), b, |r| fsl[base + scratch.rows[r]]);
            contrib -= det_det + corr;
        }
        acc += contrib;
    });
    w.phase * w.tilde_s_prod * acc
}

/// Calculate the same-spin two electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Same-spin two-electron Hamiltonian matrix element.
#[inline(always)]
pub(crate) fn lg_h2_same(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    time_call!(wick_timers::add_lg_h2_same, {
        let l = l_ex.holes.len() + g_ex.holes.len();
        let mut acc = 0.0;
        let n = w.n();

        get_det_adjt_same(w, l, 2, scratch, tol, |bits, scratch, det_det| {
            let m1 = bit(bits, 0);
            let m2 = bit(bits, 1);

            let mut contrib = w.v0[m1 + m2] * det_det;

            let v0 = w.v_t_slice(m1, m2, 0);
            let v1 = w.v_t_slice(m1, m2, 1);

            for k in 0..l {
                let mk = bit(bits, k + 2);
                let ck = scratch.cols[k];
                let vsl = if mk == 0 {v0} else {v1};
                let base = ck * n;

                let corr = column_replacement_correction(l, scratch.det_mix.as_slice(), scratch.adjt_det.as_slice(), k, |r| vsl[base + scratch.rows[r]]);
                contrib -= 2.0 * (det_det + corr);
            }

            for i in 0..l {
                for j in 0..l {
                    let phase = if ((i + j) & 1) == 0 {1.0} else {-1.0};
                    let ri_fixed = scratch.rows[i];
                    let cj_fixed = scratch.cols[j];
                    let mj = bit(bits, j + 2);

                    minor_adjt(scratch.det_mix.as_slice(), l, i, j, &mut scratch.det_mix2, &mut scratch.adjt_det2, &mut scratch.invslm1, 
                              &mut scratch.lu, tol, |lm1, det_minor, cof_minor, det_det2| {

                        for k2 in 0..lm1 {
                            let k_full = if k2 < j {k2} else {k2 + 1};
                            let mk = bit(bits, k_full + 2);
                            let (slot, swap) = jslot(m1, m2, mk, mj);

                            let jsl = w.j_slice(slot);

                            let corr = column_replacement_correction(lm1, det_minor, cof_minor, k2, |r| {
                                j_replacement(jsl, n, scratch.rows.as_slice(), scratch.cols.as_slice(), i, j, r, k2, ri_fixed, cj_fixed, swap)
                            });

                            contrib += phase * (det_det2 + corr);
                            }
                        },
                    );
                }
            }
            acc += contrib;
        });
        w.phase * w.tilde_s_prod * acc
    })
}

/// Calculate the different-spin two electron Hamiltonian matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. 
/// # Arguments:
/// - `w`: Same-spin and different-spin Wick's reference pair intermediates.
/// - `l_ex_a`: Spin alpha excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex_a`: Spin alpha excitation array for |{}^\Gamma \Psi\rangle.
/// - `l_ex_b`: Spin beta excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex_b`: Spin beta excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Different-spin two-electron Hamiltonian matrix element.
#[inline(always)]
pub(crate) fn lg_h2_diff(w: &WicksPairView, l_ex_a: &ExcitationSpin, g_ex_a: &ExcitationSpin, l_ex_b: &ExcitationSpin, g_ex_b: &ExcitationSpin,
                  diff: &mut WickScratch, a: &WickScratch, b: &WickScratch, tol: f64) -> f64 {
    time_call!(wick_timers::add_lg_h2_diff, {
        let la = l_ex_a.holes.len() + g_ex_a.holes.len();
        let lb = l_ex_b.holes.len() + g_ex_b.holes.len();

        diff.ensure_diff(la, lb);

        let rows_a = &a.rows[..la];
        let cols_a = &a.cols[..la];
        let deta0  = &a.det0.as_slice()[..la * la];
        let deta1  = &a.det1.as_slice()[..la * la];

        let rows_b = &b.rows[..lb];
        let cols_b = &b.cols[..lb];
        let detb0  = &b.det0.as_slice()[..lb * lb];
        let detb1  = &b.det1.as_slice()[..lb * lb];

        let mut acc = 0.0;
        let n = w.ab.n();

        get_det_adjt_diff(w, la, lb, diff, deta0, deta1, detb0, detb1, tol, |bits_a, bits_b, scratch, det_deta, det_detb| {
            let ma0 = bit(bits_a, 0);
            let mb0 = bit(bits_b, 0);
            let mut contrib = w.ab.vab0[ma0][mb0] * det_deta * det_detb;

            let na = w.ab.n();
            let vab0 = w.ab.vab_t_slice(ma0, mb0, 0);
            let vab1 = w.ab.vab_t_slice(ma0, mb0, 1);

            for (k, &ck) in cols_a.iter().enumerate().take(la) {
                let mak = bit(bits_a, k + 1);
                let vsl = if mak == 0 {vab0} else {vab1};
                let base = ck * na;

                let corr = column_replacement_correction(la, scratch.deta_mix.as_slice(), scratch.adjt_deta.as_slice(), k, |r| vsl[base + rows_a[r]]);
                contrib -= (det_deta + corr) * det_detb;
            }

            let nb = w.ab.n();
            let vba0 = w.ab.vba_t_slice(mb0, ma0, 0);
            let vba1 = w.ab.vba_t_slice(mb0, ma0, 1);

            for (k, &ck) in cols_b.iter().enumerate().take(lb) {
                let mbk = bit(bits_b, k + 1);
                let vsl = if mbk == 0 {vba0} else {vba1};
                let base = ck * nb;

                let corr = column_replacement_correction(lb, scratch.detb_mix.as_slice(), scratch.adjt_detb.as_slice(), k, |r| vsl[base + rows_b[r]]);
                contrib -= (det_detb + corr) * det_deta;
            }

            for (i, &ra) in rows_a.iter().enumerate() {
                for (j, &ca) in cols_a.iter().enumerate() {
                    let cofa = scratch.adjt_deta.as_slice()[idx(la, i, j)];
                    let ma1 = bit(bits_a, j + 1);

                    for k in 0..lb {
                        let mbk = bit(bits_b, k + 1);
                        let iisl = w.ab.iiab_slice(ma0, ma1, mb0, mbk);

                        let corr = column_replacement_correction(lb, scratch.detb_mix.as_slice(), scratch.adjt_detb.as_slice(), k, |r| {
                            ii_replacement(iisl, n, rows_b, cols_b, r, k, ra, ca, true)
                        });
                        contrib += 0.5 * cofa * (det_detb + corr);
                    }
                }
            }

            for (i, &rb) in rows_b.iter().enumerate() {
                for (j, &cb) in cols_b.iter().enumerate() {
                    let cofb = scratch.adjt_detb.as_slice()[idx(lb, i, j)];
                    let mb1 = bit(bits_b, j + 1);

                    for k in 0..la {
                        let mak = bit(bits_a, k + 1);
                        let iisl = w.ab.iiab_slice(ma0, mak, mb0, mb1);

                        let corr = column_replacement_correction(la, scratch.deta_mix.as_slice(), scratch.adjt_deta.as_slice(), k, |r| {
                            ii_replacement(iisl, n, rows_a, cols_a, r, k, rb, cb, false)
                        });
                        contrib += 0.5 * cofb * (det_deta + corr);
                    }
                }
            }
            acc += contrib;
        });
        (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * acc
    })
}

