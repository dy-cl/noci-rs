// nonorthogonalwicks/eval/h2diff.rs
use crate::ExcitationSpin;
use crate::maths::adjugate_transpose;
use crate::time_call;
use super::helpers::{bit, column_replacement_correction, get_det_adjt_diff, ii_replacement};
use super::super::layout::{idx, idx4};
use super::super::scratch::WickScratch;
use super::super::view::WicksPairView;

/// Calculate the different-spin two-electron Hamiltonian matrix element between two determinants
/// |{}^\Lambda \Psi\rangle and |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's
/// theorem prescription. Dispatches to the zero-overlap fast path when `w.aa.m == 0 && w.bb.m == 0`
/// and otherwise to the full generic path.
/// # Arguments:
/// - `w`: Same-spin and different-spin Wick's reference pair intermediates.
/// - `l_ex_a`: Spin-alpha excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex_a`: Spin-alpha excitation array for |{}^\Gamma \Psi\rangle.
/// - `l_ex_b`: Spin-beta excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex_b`: Spin-beta excitation array for |{}^\Gamma \Psi\rangle.
/// - `diff`: Different-spin scratch space for mixed determinants, cofactors, and work buffers.
/// - `a`: Prepared same-spin alpha scratch space.
/// - `b`: Prepared same-spin beta scratch space.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Different-spin two-electron Hamiltonian matrix element.
#[inline(always)]
pub(crate) fn lg_h2_diff(w: &WicksPairView, l_ex_a: &ExcitationSpin, g_ex_a: &ExcitationSpin, l_ex_b: &ExcitationSpin, g_ex_b: &ExcitationSpin,
                  diff: &mut WickScratch, a: &WickScratch, b: &WickScratch, tol: f64) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_h2_diff, {
        if w.aa.m == 0 && w.bb.m == 0 {
            lg_h2_diff_m0(w, l_ex_a, g_ex_a, l_ex_b, g_ex_b, diff, a, b, tol)
        }
        else {
            lg_h2_diff_gen(w, l_ex_a, g_ex_a, l_ex_b, g_ex_b, diff, a, b, tol)
        }
    })
}

/// Calculate the different-spin two-electron Hamiltonian matrix element for the zero-overlap case
/// `w.aa.m == 0 && w.bb.m == 0`. For small excitation ranks, dispatches further to specialized
/// `(la, lb) = (1, 1)` and `(2, 2)` kernels, and otherwise falls back to the general `m = 0` path.
/// # Arguments:
/// - `w`: Same-spin and different-spin Wick's reference pair intermediates with zero-overlap counts zero.
/// - `l_ex_a`: Spin-alpha excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex_a`: Spin-alpha excitation array for |{}^\Gamma \Psi\rangle.
/// - `l_ex_b`: Spin-beta excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex_b`: Spin-beta excitation array for |{}^\Gamma \Psi\rangle.
/// - `diff`: Different-spin scratch space for cofactors and work buffers.
/// - `a`: Prepared same-spin alpha scratch space.
/// - `b`: Prepared same-spin beta scratch space.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Different-spin two-electron Hamiltonian matrix element in the `m = 0` case.
#[inline(always)]
fn lg_h2_diff_m0(w: &WicksPairView, l_ex_a: &ExcitationSpin, g_ex_a: &ExcitationSpin, l_ex_b: &ExcitationSpin, g_ex_b: &ExcitationSpin,
                 diff: &mut WickScratch, a: &WickScratch, b: &WickScratch, tol: f64) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_h2_diff_m0, {
        let la = l_ex_a.holes.len() + g_ex_a.holes.len();
        let lb = l_ex_b.holes.len() + g_ex_b.holes.len();

        match (la, lb) {
            (0, 0) => (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * w.ab.vab0[0][0],
            (1, 1) => lg_h2_diff_m0_11(w, a, b),
            (1, 3) => lg_h2_diff_m0_13(w, diff, a, b, tol),
            (2, 2) => lg_h2_diff_m0_22(w, a, b),
            (3, 1) => lg_h2_diff_m0_31(w, diff, a, b, tol),
            _ => lg_h2_diff_m0_gen(w, l_ex_a, g_ex_a, l_ex_b, g_ex_b, diff, a, b, tol),
        }
    })
}

/// Calculate the different-spin two-electron Hamiltonian matrix element for the specialized
/// `(la, lb) = (1, 1)`, `m = 0` case.
/// # Arguments:
/// - `w`: Same-spin and different-spin Wick's reference pair intermediates with zero-overlap counts zero.
/// - `a`: Prepared same-spin alpha scratch space with `la = 1`.
/// - `b`: Prepared same-spin beta scratch space with `lb = 1`.
/// # Returns
/// - `f64`: Different-spin two-electron Hamiltonian matrix element for `(la, lb) = (1, 1)`.
#[inline(always)]
fn lg_h2_diff_m0_11(w: &WicksPairView, a: &WickScratch, b: &WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_h2_diff_m0_11, {
        let n = w.ab.n();

        let ra = a.rows[0];
        let ca = a.cols[0];
        let rb = b.rows[0];
        let cb = b.cols[0];

        let deta = a.det0.as_slice()[0];
        let detb = b.det0.as_slice()[0];

        let vab = w.ab.vab_t_slice(0, 0, 0);
        let vba = w.ab.vba_t_slice(0, 0, 0);
        let iisl = w.ab.iiab_slice(0, 0, 0, 0);

        let term = w.ab.vab0[0][0] * deta * detb
            - vab[ca * n + ra] * detb
            - vba[cb * n + rb] * deta
            + iisl[idx4(n, ra, ca, rb, cb)];

        (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * term
    })
}

/// Calculate the different-spin two-electron Hamiltonian matrix element for the specialized
/// `(la, lb) = (1, 3)`, `m = 0` case.
/// # Arguments:
/// - `w`: Same-spin and different-spin Wick's reference pair intermediates with zero-overlap counts zero.
/// - `a`: Prepared same-spin alpha scratch space with `la = 1`.
/// - `b`: Prepared same-spin beta scratch space with `lb = 3`.
/// # Returns
/// - `f64`: Different-spin two-electron Hamiltonian matrix element for `(la, lb) = (1, 3)`.
#[inline(always)]
fn lg_h2_diff_m0_13(w: &WicksPairView, diff: &mut WickScratch, a: &WickScratch, b: &WickScratch, tol: f64) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_h2_diff_m0_13, {
        diff.ensure_diff(1, 3);

        let n = w.ab.n();
        let ra = a.rows[0];
        let ca = a.cols[0];
        let deta = a.det0.as_slice()[0];

        let rows_b = &b.rows[..3];
        let cols_b = &b.cols[..3];
        let detb0 = &b.det0.as_slice()[..9];

        if let Some(detb) = adjugate_transpose(diff.adjt_detb.as_mut_slice(), diff.invslb.as_mut_slice(), diff.lub.as_mut_slice(), detb0, 3, tol) {
            let cofb = diff.adjt_detb.as_slice();

            let r0 = rows_b[0];
            let r1 = rows_b[1];
            let r2 = rows_b[2];
            let c0 = cols_b[0];
            let c1 = cols_b[1];
            let c2 = cols_b[2];

            let vab = w.ab.vab_t_slice(0, 0, 0);
            let vba = w.ab.vba_t_slice(0, 0, 0);
            let iisl = w.ab.iiab_slice(0, 0, 0, 0);

            let vba_term =
                cofb[idx(3, 0, 0)] * vba[c0 * n + r0] + cofb[idx(3, 1, 0)] * vba[c0 * n + r1] + cofb[idx(3, 2, 0)] * vba[c0 * n + r2]
              + cofb[idx(3, 0, 1)] * vba[c1 * n + r0] + cofb[idx(3, 1, 1)] * vba[c1 * n + r1] + cofb[idx(3, 2, 1)] * vba[c1 * n + r2]
              + cofb[idx(3, 0, 2)] * vba[c2 * n + r0] + cofb[idx(3, 1, 2)] * vba[c2 * n + r1] + cofb[idx(3, 2, 2)] * vba[c2 * n + r2];

            let ii_term =
                cofb[idx(3, 0, 0)] * iisl[idx4(n, ra, ca, r0, c0)] + cofb[idx(3, 1, 0)] * iisl[idx4(n, ra, ca, r1, c0)] + cofb[idx(3, 2, 0)] * iisl[idx4(n, ra, ca, r2, c0)]
              + cofb[idx(3, 0, 1)] * iisl[idx4(n, ra, ca, r0, c1)] + cofb[idx(3, 1, 1)] * iisl[idx4(n, ra, ca, r1, c1)] + cofb[idx(3, 2, 1)] * iisl[idx4(n, ra, ca, r2, c1)]
              + cofb[idx(3, 0, 2)] * iisl[idx4(n, ra, ca, r0, c2)] + cofb[idx(3, 1, 2)] * iisl[idx4(n, ra, ca, r1, c2)] + cofb[idx(3, 2, 2)] * iisl[idx4(n, ra, ca, r2, c2)];

            let contrib = w.ab.vab0[0][0] * deta * detb - vab[ca * n + ra] * detb - deta * vba_term + ii_term;
            (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * contrib
        } else {
            0.0
        }
    })
}

/// Calculate the different-spin two-electron Hamiltonian matrix element for the specialized
/// `(la, lb) = (2, 2)`, `m = 0` case. 
/// # Arguments:
/// - `w`: Same-spin and different-spin Wick's reference pair intermediates with zero-overlap counts zero.
/// - `a`: Prepared same-spin alpha scratch space with `la = 2`.
/// - `b`: Prepared same-spin beta scratch space with `lb = 2`.
/// # Returns
/// - `f64`: Different-spin two-electron Hamiltonian matrix element for `(la, lb) = (2, 2)`.
#[inline(always)]
fn lg_h2_diff_m0_22(w: &WicksPairView, a: &WickScratch, b: &WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_h2_diff_m0_22, {
        let n = w.ab.n();

        let rows_a = &a.rows[..2];
        let cols_a = &a.cols[..2];
        let rows_b = &b.rows[..2];
        let cols_b = &b.cols[..2];

        let da = a.det0.as_slice();
        let db = b.det0.as_slice();

        let a00 = da[0];
        let a01 = da[1];
        let a10 = da[2];
        let a11 = da[3];
        let deta = a00 * a11 - a01 * a10;

        let b00 = db[0];
        let b01 = db[1];
        let b10 = db[2];
        let b11 = db[3];
        let detb = b00 * b11 - b01 * b10;

        let r0a = rows_a[0];
        let r1a = rows_a[1];
        let c0a = cols_a[0];
        let c1a = cols_a[1];

        let r0b = rows_b[0];
        let r1b = rows_b[1];
        let c0b = cols_b[0];
        let c1b = cols_b[1];

        let vab = w.ab.vab_t_slice(0, 0, 0);
        let vba = w.ab.vba_t_slice(0, 0, 0);
        let iisl = w.ab.iiab_slice(0, 0, 0, 0);

        let au0 = vab[c0a * n + r0a];
        let au1 = vab[c0a * n + r1a];
        let av0 = vab[c1a * n + r0a];
        let av1 = vab[c1a * n + r1a];

        let deta_c0 = au0 * a11 - a01 * au1;
        let deta_c1 = a00 * av1 - av0 * a10;

        let bu0 = vba[c0b * n + r0b];
        let bu1 = vba[c0b * n + r1b];
        let bv0 = vba[c1b * n + r0b];
        let bv1 = vba[c1b * n + r1b];

        let detb_c0 = bu0 * b11 - b01 * bu1;
        let detb_c1 = b00 * bv1 - bv0 * b10;

        let mut contrib = w.ab.vab0[0][0] * deta * detb - (deta_c0 + deta_c1) * detb - (detb_c0 + detb_c1) * deta;

        let cofa = [a11, -a10, -a01, a00];
        let cofb = [b11, -b10, -b01, b00];

        for i in 0..2 {
            let ra = rows_a[i];
            for j in 0..2 {
                let ca = cols_a[j];
                let cof = cofa[idx(2, i, j)];

                let x00 = iisl[idx4(n, ra, ca, r0b, c0b)];
                let x10 = iisl[idx4(n, ra, ca, r1b, c0b)];
                let x01 = iisl[idx4(n, ra, ca, r0b, c1b)];
                let x11 = iisl[idx4(n, ra, ca, r1b, c1b)];

                contrib += 0.5 * cof * (x00 * b11 - b01 * x10);
                contrib += 0.5 * cof * (b00 * x11 - x01 * b10);
            }
        }

        for i in 0..2 {
            let rb = rows_b[i];
            for j in 0..2 {
                let cb = cols_b[j];
                let cof = cofb[idx(2, i, j)];

                let x00 = iisl[idx4(n, r0a, c0a, rb, cb)];
                let x10 = iisl[idx4(n, r1a, c0a, rb, cb)];
                let x01 = iisl[idx4(n, r0a, c1a, rb, cb)];
                let x11 = iisl[idx4(n, r1a, c1a, rb, cb)];

                contrib += 0.5 * cof * (x00 * a11 - a01 * x10);
                contrib += 0.5 * cof * (a00 * x11 - x01 * a10);
            }
        }

        (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * contrib
    })
}

/// Calculate the different-spin two-electron Hamiltonian matrix element for the specialized
/// `(la, lb) = (3, 1)`, `m = 0` case.
/// # Arguments:
/// - `w`: Same-spin and different-spin Wick's reference pair intermediates with zero-overlap counts zero.
/// - `a`: Prepared same-spin alpha scratch space with `la = 3`.
/// - `b`: Prepared same-spin beta scratch space with `lb = 1`.
/// # Returns
/// - `f64`: Different-spin two-electron Hamiltonian matrix element for `(la, lb) = (3, 1)`.
#[inline(always)]
fn lg_h2_diff_m0_31(w: &WicksPairView, diff: &mut WickScratch, a: &WickScratch, b: &WickScratch, tol: f64) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_h2_diff_m0_31, {
        diff.ensure_diff(3, 1);

        let n = w.ab.n();
        let rb = b.rows[0];
        let cb = b.cols[0];
        let detb = b.det0.as_slice()[0];

        let rows_a = &a.rows[..3];
        let cols_a = &a.cols[..3];
        let deta0 = &a.det0.as_slice()[..9];

        if let Some(deta) = adjugate_transpose(diff.adjt_deta.as_mut_slice(), diff.invsla.as_mut_slice(), diff.lua.as_mut_slice(), deta0, 3, tol) {
            let cofa = diff.adjt_deta.as_slice();

            let r0 = rows_a[0];
            let r1 = rows_a[1];
            let r2 = rows_a[2];
            let c0 = cols_a[0];
            let c1 = cols_a[1];
            let c2 = cols_a[2];

            let vab = w.ab.vab_t_slice(0, 0, 0);
            let vba = w.ab.vba_t_slice(0, 0, 0);
            let iisl = w.ab.iiab_slice(0, 0, 0, 0);

            let vab_term =
                cofa[idx(3, 0, 0)] * vab[c0 * n + r0] + cofa[idx(3, 1, 0)] * vab[c0 * n + r1] + cofa[idx(3, 2, 0)] * vab[c0 * n + r2]
              + cofa[idx(3, 0, 1)] * vab[c1 * n + r0] + cofa[idx(3, 1, 1)] * vab[c1 * n + r1] + cofa[idx(3, 2, 1)] * vab[c1 * n + r2]
              + cofa[idx(3, 0, 2)] * vab[c2 * n + r0] + cofa[idx(3, 1, 2)] * vab[c2 * n + r1] + cofa[idx(3, 2, 2)] * vab[c2 * n + r2];

            let ii_term =
                cofa[idx(3, 0, 0)] * iisl[idx4(n, r0, c0, rb, cb)] + cofa[idx(3, 1, 0)] * iisl[idx4(n, r1, c0, rb, cb)] + cofa[idx(3, 2, 0)] * iisl[idx4(n, r2, c0, rb, cb)]
              + cofa[idx(3, 0, 1)] * iisl[idx4(n, r0, c1, rb, cb)] + cofa[idx(3, 1, 1)] * iisl[idx4(n, r1, c1, rb, cb)] + cofa[idx(3, 2, 1)] * iisl[idx4(n, r2, c1, rb, cb)]
              + cofa[idx(3, 0, 2)] * iisl[idx4(n, r0, c2, rb, cb)] + cofa[idx(3, 1, 2)] * iisl[idx4(n, r1, c2, rb, cb)] + cofa[idx(3, 2, 2)] * iisl[idx4(n, r2, c2, rb, cb)];

            let contrib = w.ab.vab0[0][0] * deta * detb - detb * vab_term - vba[cb * n + rb] * deta + ii_term;
            (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * contrib
        } else {
            0.0
        }
    })
}

/// Calculate the different-spin two-electron Hamiltonian matrix element for the general `m = 0`
/// case with arbitrary excitation ranks `la` and `lb`. 
/// # Arguments:
/// - `w`: Same-spin and different-spin Wick's reference pair intermediates with zero-overlap counts zero.
/// - `l_ex_a`: Spin-alpha excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex_a`: Spin-alpha excitation array for |{}^\Gamma \Psi\rangle.
/// - `l_ex_b`: Spin-beta excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex_b`: Spin-beta excitation array for |{}^\Gamma \Psi\rangle.
/// - `diff`: Different-spin scratch space for mixed determinants, cofactors, and work buffers.
/// - `a`: Prepared same-spin alpha scratch space.
/// - `b`: Prepared same-spin beta scratch space.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Different-spin two-electron Hamiltonian matrix element for the general `m = 0` path.
#[inline(always)]
fn lg_h2_diff_m0_gen(w: &WicksPairView, l_ex_a: &ExcitationSpin, g_ex_a: &ExcitationSpin, l_ex_b: &ExcitationSpin, g_ex_b: &ExcitationSpin,
                 diff: &mut WickScratch, a: &WickScratch, b: &WickScratch, tol: f64) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_h2_diff_m0_gen, {
        let la = l_ex_a.holes.len() + g_ex_a.holes.len();
        let lb = l_ex_b.holes.len() + g_ex_b.holes.len();

        diff.ensure_diff(la, lb);

        let rows_a = &a.rows[..la];
        let cols_a = &a.cols[..la];
        let deta0 = &a.det0.as_slice()[..la * la];

        let rows_b = &b.rows[..lb];
        let cols_b = &b.cols[..lb];
        let detb0 = &b.det0.as_slice()[..lb * lb];

        let mut acc = 0.0;
        let n = w.ab.n();

        if let Some(det_deta) = adjugate_transpose(diff.adjt_deta.as_mut_slice(), diff.invsla.as_mut_slice(), diff.lua.as_mut_slice(), deta0, la, tol)
            && let Some(det_detb) = adjugate_transpose(diff.adjt_detb.as_mut_slice(), diff.invslb.as_mut_slice(), diff.lub.as_mut_slice(), detb0, lb, tol) {
            let mut contrib = w.ab.vab0[0][0] * det_deta * det_detb;

            let vab = w.ab.vab_t_slice(0, 0, 0);
            for (k, &ck) in cols_a.iter().enumerate().take(la) {
                let base = ck * n;
                let corr = column_replacement_correction(la, deta0, diff.adjt_deta.as_slice(), k, |r| vab[base + rows_a[r]]);
                contrib -= (det_deta + corr) * det_detb;
            }

            let vba = w.ab.vba_t_slice(0, 0, 0);
            for (k, &ck) in cols_b.iter().enumerate().take(lb) {
                let base = ck * n;
                let corr = column_replacement_correction(lb, detb0, diff.adjt_detb.as_slice(), k, |r| vba[base + rows_b[r]]);
                contrib -= (det_detb + corr) * det_deta;
            }

            let iisl = w.ab.iiab_slice(0, 0, 0, 0);

            for (i, &ra) in rows_a.iter().enumerate() {
                for (j, &ca) in cols_a.iter().enumerate() {
                    let cofa = diff.adjt_deta.as_slice()[idx(la, i, j)];

                    for k in 0..lb {
                        let corr = column_replacement_correction(lb, detb0, diff.adjt_detb.as_slice(), k, |r| {
                            ii_replacement(iisl, n, rows_b, cols_b, r, k, ra, ca, true)
                        });
                        contrib += 0.5 * cofa * (det_detb + corr);
                    }
                }
            }

            for (i, &rb) in rows_b.iter().enumerate() {
                for (j, &cb) in cols_b.iter().enumerate() {
                    let cofb = diff.adjt_detb.as_slice()[idx(lb, i, j)];

                    for k in 0..la {
                        let corr = column_replacement_correction(la, deta0, diff.adjt_deta.as_slice(), k, |r| {
                            ii_replacement(iisl, n, rows_a, cols_a, r, k, rb, cb, false)
                        });
                        contrib += 0.5 * cofb * (det_deta + corr);
                    }
                }
            }

            acc += contrib;
        }

        (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * acc
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
fn lg_h2_diff_gen(w: &WicksPairView, l_ex_a: &ExcitationSpin, g_ex_a: &ExcitationSpin, l_ex_b: &ExcitationSpin, g_ex_b: &ExcitationSpin,
                  diff: &mut WickScratch, a: &WickScratch, b: &WickScratch, tol: f64) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_h2_diff_gen, {
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


