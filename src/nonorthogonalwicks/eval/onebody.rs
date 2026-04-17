// nonorthogonalwicks/eval/onebody.rs
use crate::ExcitationSpin;
use crate::maths::adjugate_transpose;
use crate::time_call;
use crate::timers::nonorthogonalwicks as wick_timers;
use super::helpers::{bit, column_replacement_correction, get_det_adjt_same};
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;

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

/// Calculate one-body matrix elements between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription.
/// Dispatches to the zero-overlap fast path when `w.m == 0` and otherwise to the full
/// generic path.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// - `ob`: Selects Hamiltonian or Fock one-body intermediates.
/// # Returns
/// - `f64`: One-body matrix element.
#[inline(always)]
fn lg_one_body(w: &SameSpinView<'_>, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64, ob: OneBody) -> f64 {
    if w.m == 0 {
        lg_one_body_m0(w, l_ex, g_ex, scratch, tol, ob)
    }
    else {
        lg_one_body_gen(w, l_ex, g_ex, scratch, tol, ob)
    }
}

/// Calculate one-body matrix elements for the zero-overlap case `w.m == 0`.
/// For small excitation rank, dispatches further to specialized `l = 1` and `l = 2`
/// kernels, and otherwise falls back to the general `m = 0` path.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// - `ob`: Selects Hamiltonian or Fock one-body intermediates.
/// # Returns
/// - `f64`: One-body matrix element in the `m = 0` case.
#[inline(always)]
fn lg_one_body_m0(w: &SameSpinView<'_>, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64, ob: OneBody) -> f64 {
    time_call!(wick_timers::add_lg_one_body_m0, {
        let l = l_ex.holes.len() + g_ex.holes.len();
        match l {
            0 => w.phase * w.tilde_s_prod * one_body_scalar(w, ob, 0),
            1 => lg_one_body_m0_l1(w, scratch, ob),
            2 => lg_one_body_m0_l2(w, scratch, ob),
            _ => lg_one_body_m0_gen(w, l_ex, g_ex, scratch, tol, ob),
        }
    })
}

/// Calculate the specialized `l = 1`, `m = 0` one-body matrix element.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `l = 1` contraction determinant and indices.
/// - `ob`: Selects Hamiltonian or Fock one-body intermediates.
/// # Returns
/// - `f64`: One-body matrix element for `l = 1`.
#[inline(always)]
fn lg_one_body_m0_l1(w: &SameSpinView<'_>, scratch: &mut WickScratch, ob: OneBody) -> f64 {
    time_call!(wick_timers::add_lg_one_body_m0_l1, {
        let n = w.n();
        let det0 = scratch.det0.as_slice();
        let det = det0[0];
        let r0 = scratch.rows[0];
        let c0 = scratch.cols[0];
        let fsl = match ob {OneBody::H1 => w.fh_t_slice(0, 0), OneBody::Fock => w.ff_t_slice(0, 0)};
        let repl = fsl[c0 * n + r0];

        w.phase * w.tilde_s_prod * (det * one_body_scalar(w, ob, 0) - repl)
    })
}

/// Calculate the specialized `l = 2`, `m = 0` one-body matrix element.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `l = 2` contraction determinant and indices.
/// - `ob`: Selects Hamiltonian or Fock one-body intermediates.
/// # Returns
/// - `f64`: One-body matrix element for `l = 2`.
#[inline(always)]
fn lg_one_body_m0_l2(w: &SameSpinView<'_>, scratch: &mut WickScratch, ob: OneBody) -> f64 {
    time_call!(wick_timers::add_lg_one_body_m0_l2, {
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

        let fsl = match ob {OneBody::H1 => w.fh_t_slice(0, 0), OneBody::Fock => w.ff_t_slice(0, 0)};

        let u0 = fsl[c0 * n + r0];
        let u1 = fsl[c0 * n + r1];
        let v0 = fsl[c1 * n + r0];
        let v1 = fsl[c1 * n + r1];

        let det_c0 = u0 * a11 - a01 * u1;
        let det_c1 = a00 * v1 - v0 * a10;

        w.phase * w.tilde_s_prod * (det * one_body_scalar(w, ob, 0) - det_c0 - det_c1)
    })
}

/// Calculate the one-body matrix element for the general `m = 0` case with arbitrary excitation
/// rank `l`. 
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// - `ob`: Selects Hamiltonian or Fock one-body intermediates.
/// # Returns
/// - `f64`: One-body matrix element for the general `m = 0` path.
#[inline(always)]
fn lg_one_body_m0_gen(w: &SameSpinView<'_>, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64, ob: OneBody) -> f64 {
    time_call!(wick_timers::add_lg_one_body_m0_gen, {
        let l = l_ex.holes.len() + g_ex.holes.len();
        let mut acc = 0.0;
        let n = w.n();
        let det0 = &scratch.det0.as_slice()[..l * l];

        if let Some(det_det) = adjugate_transpose(scratch.adjt_det.as_mut_slice(), scratch.invs.as_mut_slice(), scratch.lu.as_mut_slice(), det0, l, tol) {
            let mut contrib = det_det * one_body_scalar(w, ob, 0);
            let fsl = match ob {OneBody::H1 => w.fh_t_slice(0, 0), OneBody::Fock => w.ff_t_slice(0, 0)};

            for b in 0..l {
                let cb = scratch.cols[b];
                let base = cb * n;
                let corr = column_replacement_correction(l, det0, scratch.adjt_det.as_slice(), b, |r| fsl[base + scratch.rows[r]]);
                contrib -= det_det + corr;
            }
            acc += contrib;
        }

        w.phase * w.tilde_s_prod * acc
    })
}

/// Calculate one-body matrix elements between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the full generic extended non-orthogonal Wick's theorem
/// prescription. This path evaluates the sum over allowed zero-distribution bitstrings and is
/// used when `w.m > 0`.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// - `ob`: Selects Hamiltonian or Fock one-body intermediates.
/// # Returns
/// - `f64`: One-body matrix element.
#[inline(always)]
fn lg_one_body_gen(w: &SameSpinView<'_>, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch, tol: f64, ob: OneBody,) -> f64 {
    time_call!(wick_timers::add_lg_one_body_gen, {
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
    })
}
