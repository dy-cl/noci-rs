// nonorthogonalwicks/eval/overlap.rs
use crate::ExcitationSpin;
use crate::maths::det;
use crate::time_call;
use super::helpers::mix_dets_same;
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;

/// Calculate overlap matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription.
/// Dispatches to zero-overlap and fully-zeroed fast paths where possible.
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element.
#[inline(always)]
pub fn lg_overlap(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap, {
        let l = l_ex.holes.len() + g_ex.holes.len();

        if w.m > l {
            0.0
        }
        else if w.m == 0 {
            lg_overlap_m0(w, l, scratch)
        }
        else if w.m == l {
            lg_overlap_ml(w, l, scratch)
        }
        else {
            lg_overlap_gen(w, l, scratch)
        }
    })
}

/// Calculate overlap matrix elements for the zero-overlap case `w.m == 0`.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `l`: Total excitation rank entering the determinant.
/// - `scratch`: Scratch space containing the prepared `det0`.
/// # Returns
/// - `f64`: Overlap matrix element in the `m = 0` case.
#[inline(always)]
fn lg_overlap_m0(w: &SameSpinView<'_>, l: usize, scratch: &mut WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0, {
        match l {
            0 => w.phase * w.tilde_s_prod,
            1 => lg_overlap_m0_l1(w, scratch),
            2 => lg_overlap_m0_l2(w, scratch),
            3 => lg_overlap_m0_l3(w, scratch),
            _ => w.phase * w.tilde_s_prod * det(scratch.det0.as_slice(), l).unwrap_or(0.0),
        }
    })
}

/// Calculate the specialized `l = 1`, `m = 0` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `det0`.
/// # Returns
/// - `f64`: Overlap matrix element for `l = 1`.
#[inline(always)]
fn lg_overlap_m0_l1(w: &SameSpinView<'_>, scratch: &mut WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0_l1, {
        let d = scratch.det0.as_slice();
        w.phase * w.tilde_s_prod * d[0]
    })
}

/// Calculate the specialized `l = 2`, `m = 0` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `det0`.
/// # Returns
/// - `f64`: Overlap matrix element for `l = 2`.
#[inline(always)]
fn lg_overlap_m0_l2(w: &SameSpinView<'_>, scratch: &mut WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0_l2, {
        let d = scratch.det0.as_slice();
        w.phase * w.tilde_s_prod * (d[0] * d[3] - d[1] * d[2])
    })
}

/// Calculate the specialized `l = 3`, `m = 0` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `det0`.
/// # Returns
/// - `f64`: Overlap matrix element for `l = 3`.
#[inline(always)]
fn lg_overlap_m0_l3(w: &SameSpinView<'_>, scratch: &mut WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0_l3, {
        let d = scratch.det0.as_slice();
        w.phase * w.tilde_s_prod * (d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6]) + d[2] * (d[3] * d[7] - d[4] * d[6]))
    })
}

/// Calculate overlap matrix elements when all determinant columns are zero-replacement columns,
/// i.e. `w.m == l`.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l`: Total excitation rank entering the determinant.
/// - `scratch`: Scratch space containing the prepared `det1`.
/// # Returns
/// - `f64`: Overlap matrix element in the `m = l` case.
#[inline(always)]
fn lg_overlap_ml(w: &SameSpinView<'_>, l: usize, scratch: &mut WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml, {
        match l {
            0 => w.phase * w.tilde_s_prod,
            1 => lg_overlap_ml_l1(w, scratch),
            2 => lg_overlap_ml_l2(w, scratch),
            3 => lg_overlap_ml_l3(w, scratch),
            _ => w.phase * w.tilde_s_prod * det(scratch.det1.as_slice(), l).unwrap_or(0.0),
        }
    })
}

/// Calculate the specialized `l = 1`, `m = l` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `scratch`: Scratch space containing the prepared `det1`.
/// # Returns
/// - `f64`: Overlap matrix element for `l = 1`.
#[inline(always)]
fn lg_overlap_ml_l1(w: &SameSpinView<'_>, scratch: &mut WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml_l1, {
        let d = scratch.det1.as_slice();
        w.phase * w.tilde_s_prod * d[0]
    })
}

/// Calculate the specialized `l = 2`, `m = l` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `scratch`: Scratch space containing the prepared `det1`.
/// # Returns
/// - `f64`: Overlap matrix element for `l = 2`.
#[inline(always)]
fn lg_overlap_ml_l2(w: &SameSpinView<'_>, scratch: &mut WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml_l2, {
        let d = scratch.det1.as_slice();
        w.phase * w.tilde_s_prod * (d[0] * d[3] - d[1] * d[2])
    })
}

/// Calculate the specialized `l = 3`, `m = l` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `scratch`: Scratch space containing the prepared `det1`.
/// # Returns
/// - `f64`: Overlap matrix element for `l = 3`.
#[inline(always)]
fn lg_overlap_ml_l3(w: &SameSpinView<'_>, scratch: &mut WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml_l3, {
        let d = scratch.det1.as_slice();
        w.phase * w.tilde_s_prod * (d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6]) + d[2] * (d[3] * d[7] - d[4] * d[6]))
    })
}

/// Calculate overlap matrix elements for the general `0 < w.m < l` case.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l`: Total excitation rank entering the determinant.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element for the general mixed-column path.
#[inline(always)]
fn lg_overlap_gen(w: &SameSpinView<'_>, l: usize, scratch: &mut WickScratch) -> f64 {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_gen, {
        let mut acc = 0.0;
        mix_dets_same(w, l, 0, scratch, |_, scratch| {
            let d = scratch.det_mix.as_slice();
            let contrib = match l {
                1 => d[0],
                2 => d[0] * d[3] - d[1] * d[2],
                3 => d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6]) + d[2] * (d[3] * d[7] - d[4] * d[6]),
                _ => det(d, l).unwrap_or(0.0),
            };
            acc += contrib;
        });
        w.phase * w.tilde_s_prod * acc
    })
}
