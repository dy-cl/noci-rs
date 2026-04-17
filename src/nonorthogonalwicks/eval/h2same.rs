// nonorthogonalwicks/eval/h2same.rs
use crate::ExcitationSpin;
use crate::maths::adjugate_transpose;
use crate::time_call;
use crate::timers::nonorthogonalwicks as wick_timers;
use super::helpers::{bit, column_replacement_correction, get_det_adjt_same, j_replacement, jslot, minor_adjt};
use super::super::layout::idx4;
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;

/// Calculate the same-spin two-electron Hamiltonian matrix element between two determinants
/// |{}^\Lambda \Psi\rangle and |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's
/// theorem prescription. Dispatches to the zero-overlap fast path when `w.m == 0` and otherwise
/// to the full generic path.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Same-spin two-electron Hamiltonian matrix element.
#[inline(always)]
pub(crate) fn lg_h2_same(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    time_call!(wick_timers::add_lg_h2_same, {
        if w.m == 0 {
            lg_h2_same_m0(w, l_ex, g_ex, scratch, tol)
        } else {
            lg_h2_same_gen(w, l_ex, g_ex, scratch, tol)
        }
    })
}

/// Calculate the same-spin two-electron Hamiltonian matrix element for the zero-overlap case
/// `w.m == 0`. For small excitation rank, dispatches further to specialized `l = 1` and `l = 2`
/// kernels, and otherwise falls back to the general `m = 0` path.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Same-spin two-electron Hamiltonian matrix element in the `m = 0` case.
#[inline(always)]
fn lg_h2_same_m0(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    time_call!(wick_timers::add_lg_h2_same_m0, {
        let l = l_ex.holes.len() + g_ex.holes.len();
        match l {
            0 => w.phase * w.tilde_s_prod * w.v0[0],
            1 => lg_h2_same_m0_l1(w, scratch),
            2 => lg_h2_same_m0_l2(w, scratch),
            _ => lg_h2_same_m0_gen(w, l_ex, g_ex, scratch, tol),
        }
    })
}

/// Calculate the specialized `l = 1`, `m = 0` same-spin two-electron Hamiltonian matrix element.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `l = 1` contraction determinant and indices.
/// # Returns
/// - `f64`: Same-spin two-electron Hamiltonian matrix element for `l = 1`.
#[inline(always)]
fn lg_h2_same_m0_l1(w: &SameSpinView, scratch: &mut WickScratch) -> f64 {
    time_call!(wick_timers::add_lg_h2_same_m0_l1, {
        let n = w.n();
        let r0 = scratch.rows[0];
        let c0 = scratch.cols[0];
        let det0 = scratch.det0.as_slice();
        let det = det0[0];
        let vsl = w.v_t_slice(0, 0, 0);
        let repl = vsl[c0 * n + r0];

        w.phase * w.tilde_s_prod * (w.v0[0] * det - 2.0 * repl)
    })
}

/// Calculate the specialized `l = 2`, `m = 0` same-spin two-electron Hamiltonian matrix element.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `l = 2` contraction determinant and indices.
/// # Returns
/// - `f64`: Same-spin two-electron Hamiltonian matrix element for `l = 2`.
#[inline(always)]
fn lg_h2_same_m0_l2(w: &SameSpinView, scratch: &mut WickScratch) -> f64 {
    time_call!(wick_timers::add_lg_h2_same_m0_l2, {
        let n = w.n();
        let d = scratch.det0.as_slice();
        let a00 = d[0];
        let a01 = d[1];
        let a10 = d[2];
        let a11 = d[3];
        let det = a00 * a11 - a01 * a10;

        let r0 = scratch.rows[0];
        let r1 = scratch.rows[1];
        let c0 = scratch.cols[0];
        let c1 = scratch.cols[1];

        let vsl = w.v_t_slice(0, 0, 0);

        let u0 = vsl[c0 * n + r0];
        let u1 = vsl[c0 * n + r1];
        let v0 = vsl[c1 * n + r0];
        let v1 = vsl[c1 * n + r1];

        let det_c0 = u0 * a11 - a01 * u1;
        let det_c1 = a00 * v1 - v0 * a10;

        let jsl = w.j_slice(0);
        let jterm =
            jsl[idx4(n, r0, c0, r1, c1)]
          - jsl[idx4(n, r0, c1, r1, c0)]
          - jsl[idx4(n, r1, c0, r0, c1)]
          + jsl[idx4(n, r1, c1, r0, c0)];

        w.phase * w.tilde_s_prod * (w.v0[0] * det - 2.0 * (det_c0 + det_c1) + jterm)
    })
}

/// Calculate the same-spin two-electron Hamiltonian matrix element for the general `m = 0` case
/// with arbitrary excitation rank `l`. 
/// determinant routines.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Same-spin two-electron Hamiltonian matrix element for the general `m = 0` path.
#[inline(always)]
fn lg_h2_same_m0_gen(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    time_call!(wick_timers::add_lg_h2_same_m0_gen, {
        let l = l_ex.holes.len() + g_ex.holes.len();
        let mut acc = 0.0;
        let n = w.n();
        let det0 = &scratch.det0.as_slice()[..l * l];

        if let Some(det_det) = adjugate_transpose(scratch.adjt_det.as_mut_slice(), scratch.invs.as_mut_slice(), scratch.lu.as_mut_slice(), det0, l, tol) {
            let mut contrib = w.v0[0] * det_det;
            let vsl = w.v_t_slice(0, 0, 0);

            for k in 0..l {
                let ck = scratch.cols[k];
                let base = ck * n;
                let corr = column_replacement_correction(l, det0, scratch.adjt_det.as_slice(), k, |r| vsl[base + scratch.rows[r]]);
                contrib -= 2.0 * (det_det + corr);
            }

            let jsl = w.j_slice(0);

            for i in 0..l {
                for j in 0..l {
                    let phase = if ((i + j) & 1) == 0 {1.0} else {-1.0};
                    let ri_fixed = scratch.rows[i];
                    let cj_fixed = scratch.cols[j];

                    minor_adjt(det0, l, i, j, &mut scratch.det_mix2, &mut scratch.adjt_det2, &mut scratch.invslm1, &mut scratch.lu, tol, |lm1, det_minor, cof_minor, det_det2| {
                        for k2 in 0..lm1 {
                            let corr = column_replacement_correction(lm1, det_minor, cof_minor, k2, |r| {
                                j_replacement(jsl, n, scratch.rows.as_slice(), scratch.cols.as_slice(), i, j, r, k2, ri_fixed, cj_fixed, false)
                            });
                            contrib += phase * (det_det2 + corr);
                        }
                    });
                }
            }

            acc += contrib;
        }

        w.phase * w.tilde_s_prod * acc
    })
}

/// Calculate the same-spin two-electron Hamiltonian matrix element between two determinants
/// |{}^\Lambda \Psi\rangle and |{}^\Gamma \Psi\rangle using the full generic extended
/// non-orthogonal Wick's theorem prescription. This path evaluates the sum over allowed
/// zero-distribution bitstrings and is used when `w.m > 0`.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `f64`: Same-spin two-electron Hamiltonian matrix element.
#[inline(always)]
fn lg_h2_same_gen(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64) -> f64 {
    time_call!(wick_timers::add_lg_h2_same_gen, {
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


