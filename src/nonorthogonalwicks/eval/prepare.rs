// nonorthogonalwicks/eval/prepare.rs 
use crate::ExcitationSpin;
use crate::maths::build_d;
use crate::time_call;
use crate::timers::nonorthogonalwicks as wick_timers;
use super::helpers::construct_determinant_indices;
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;

/// Prepare shared same-spin scratch quantities used by overlap and Hamiltonian evaluations.
/// Dispatches to the zero-overlap fast path when `w.m == 0` and otherwise to the full generic path.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `()`: Prepares the required same-spin scratch quantities in place.
pub fn prepare_same(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) {
    time_call!(wick_timers::add_prepare_same, {
        if w.m == 0 {
            prepare_same_m0(w, l_ex, g_ex, scratch)
        }
        else {
            prepare_same_gen(w, l_ex, g_ex, scratch)
        }
    })
}

/// Prepare shared same-spin scratch quantities for the zero-overlap case `m = 0`.
/// Only the `det0` branch is required, so the second branch determinant is not built.
/// For small excitation rank, dispatches further to specialized `l = 1` and `l = 2` kernels.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `()`: Prepares the zero-overlap same-spin scratch quantities in place.
#[inline(always)]
fn prepare_same_m0(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) {
    time_call!(wick_timers::add_prepare_same_m0, {
        let l = l_ex.holes.len() + g_ex.holes.len();
        scratch.ensure_same(l);

        construct_determinant_indices(l_ex, g_ex, w.nmo, &mut scratch.rows, &mut scratch.cols);

        match l {
            0 => {}
            1 => prepare_same_m0_l1(w, scratch),
            2 => prepare_same_m0_l2(w, scratch),
            _ => {
                let x0 = w.x(0);
                let y0 = w.y(0);
                build_d(scratch.det0.as_mut_slice(), l, &x0, &y0, scratch.rows.as_slice(), scratch.cols.as_slice());
            }
        }
    })
}

/// Prepare the `l = 1`, `m = 0` same-spin contraction determinant directly.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space whose `rows`, `cols`, and `det0` buffers have already been prepared.
/// # Returns
/// - `()`: Writes the one-by-one contraction determinant into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0_l1(w: &SameSpinView, scratch: &mut WickScratch) {
    time_call!(wick_timers::add_prepare_same_m0_l1, {
        let x0 = w.x(0);
        let xstr = x0.strides();
        let xptr = x0.as_ptr();
        let r0 = scratch.rows[0] as isize;
        let c0 = scratch.cols[0] as isize;
        let det0 = scratch.det0.as_mut_slice();

        unsafe {det0[0] = *xptr.offset(r0 * xstr[0] + c0 * xstr[1]);}
    })
}

/// Prepare the `l = 2`, `m = 0` same-spin contraction determinant directly.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space whose `rows`, `cols`, and `det0` buffers have already been prepared.
/// # Returns
/// - `()`: Writes the two-by-two contraction determinant into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0_l2(w: &SameSpinView, scratch: &mut WickScratch) {
    time_call!(wick_timers::add_prepare_same_m0_l2, {
        let x0 = w.x(0);
        let y0 = w.y(0);
        let xstr = x0.strides();
        let ystr = y0.strides();
        let xptr = x0.as_ptr();
        let yptr = y0.as_ptr();

        let r0 = scratch.rows[0] as isize;
        let r1 = scratch.rows[1] as isize;
        let c0 = scratch.cols[0] as isize;
        let c1 = scratch.cols[1] as isize;

        let xr0 = r0 * xstr[0];
        let xr1 = r1 * xstr[0];
        let yr0 = r0 * ystr[0];

        let det0 = scratch.det0.as_mut_slice();

        unsafe {
            det0[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            det0[1] = *yptr.offset(yr0 + c1 * ystr[1]);
            det0[2] = *xptr.offset(xr1 + c0 * xstr[1]);
            det0[3] = *xptr.offset(xr1 + c1 * xstr[1]);
        }
    })
}

/// Builds both determinant branches `det0` and `det1`, which are later mixed according to the
/// allowed bitstrings of the nonorthogonal Wick expansion.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `()`: Prepares the generic same-spin scratch quantities in place.
pub fn prepare_same_gen(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) {
    time_call!(wick_timers::add_prepare_same_gen, {
        let l = l_ex.holes.len() + g_ex.holes.len();
        scratch.ensure_same(l);

        construct_determinant_indices(l_ex, g_ex, w.nmo, &mut scratch.rows, &mut scratch.cols);

        let x0 = w.x(0);
        let y0 = w.y(0);
        build_d(scratch.det0.as_mut_slice(), l, &x0, &y0, scratch.rows.as_slice(), scratch.cols.as_slice());

        let x1 = w.x(1);
        let y1 = w.y(1);
        build_d(scratch.det1.as_mut_slice(), l, &x1, &y1, scratch.rows.as_slice(), scratch.cols.as_slice());
    })
}



