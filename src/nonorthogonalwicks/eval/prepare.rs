// nonorthogonalwicks/eval/prepare.rs 
use crate::ExcitationSpin;
use crate::maths::build_d;
use crate::time_call;
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
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same, {
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
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0, {
        let l = l_ex.holes.len() + g_ex.holes.len();

        match l {
            0 => {
                scratch.ensure_same(0);
                scratch.rows.clear();
                scratch.cols.clear();}
            1 => {
                scratch.ensure_same(1);
                construct_determinant_indices_l1(l_ex, g_ex, w.nmo, &mut scratch.rows, &mut scratch.cols);
                prepare_same_m0_l1(w, scratch);
            }
            2 => {
                scratch.ensure_same(2);
                construct_determinant_indices_l2(l_ex, g_ex, w.nmo, &mut scratch.rows, &mut scratch.cols);
                prepare_same_m0_l2(w, scratch);
            }
            3 => {
                scratch.ensure_same(3);
                construct_determinant_indices_l3(l_ex, g_ex, w.nmo, &mut scratch.rows, &mut scratch.cols);
                prepare_same_m0_l3(w, scratch);
            }
            4 => {
                scratch.ensure_same(4);
                construct_determinant_indices_l4(l_ex, g_ex, w.nmo, &mut scratch.rows, &mut scratch.cols);
                prepare_same_m0_l4(w, scratch);
            }
            _ => {
                scratch.ensure_same(l);
                construct_determinant_indices(l_ex, g_ex, w.nmo, &mut scratch.rows, &mut scratch.cols);

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
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l1, {
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
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l2, {
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

/// Prepare the `l = 3`, `m = 0` same-spin contraction determinant directly.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space whose `rows`, `cols`, and `det0` buffers have already been prepared.
/// # Returns
/// - `()`: Writes the three-by-three contraction determinant into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0_l3(w: &SameSpinView, scratch: &mut WickScratch) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l3, {
        let x0 = w.x(0);
        let y0 = w.y(0);
        let xstr = x0.strides();
        let ystr = y0.strides();
        let xptr = x0.as_ptr();
        let yptr = y0.as_ptr();

        let r0 = scratch.rows[0] as isize;
        let r1 = scratch.rows[1] as isize;
        let r2 = scratch.rows[2] as isize;
        let c0 = scratch.cols[0] as isize;
        let c1 = scratch.cols[1] as isize;
        let c2 = scratch.cols[2] as isize;

        let xr0 = r0 * xstr[0];
        let xr1 = r1 * xstr[0];
        let xr2 = r2 * xstr[0];
        let yr0 = r0 * ystr[0];
        let yr1 = r1 * ystr[0];

        let det0 = scratch.det0.as_mut_slice();

        unsafe {
            det0[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            det0[1] = *yptr.offset(yr0 + c1 * ystr[1]);
            det0[2] = *yptr.offset(yr0 + c2 * ystr[1]);

            det0[3] = *xptr.offset(xr1 + c0 * xstr[1]);
            det0[4] = *xptr.offset(xr1 + c1 * xstr[1]);
            det0[5] = *yptr.offset(yr1 + c2 * ystr[1]);

            det0[6] = *xptr.offset(xr2 + c0 * xstr[1]);
            det0[7] = *xptr.offset(xr2 + c1 * xstr[1]);
            det0[8] = *xptr.offset(xr2 + c2 * xstr[1]);
        }
    })
}

/// Prepare the `l = 4`, `m = 0` same-spin contraction determinant directly.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space whose `rows`, `cols`, and `det0` buffers have already been prepared.
/// # Returns
/// - `()`: Writes the four-by-four contraction determinant into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0_l4(w: &SameSpinView, scratch: &mut WickScratch) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l4, {
        let x0 = w.x(0);
        let y0 = w.y(0);
        let xstr = x0.strides();
        let ystr = y0.strides();
        let xptr = x0.as_ptr();
        let yptr = y0.as_ptr();

        let r0 = scratch.rows[0] as isize;
        let r1 = scratch.rows[1] as isize;
        let r2 = scratch.rows[2] as isize;
        let r3 = scratch.rows[3] as isize;
        let c0 = scratch.cols[0] as isize;
        let c1 = scratch.cols[1] as isize;
        let c2 = scratch.cols[2] as isize;
        let c3 = scratch.cols[3] as isize;

        let xr0 = r0 * xstr[0];
        let xr1 = r1 * xstr[0];
        let xr2 = r2 * xstr[0];
        let xr3 = r3 * xstr[0];
        let yr0 = r0 * ystr[0];
        let yr1 = r1 * ystr[0];
        let yr2 = r2 * ystr[0];

        let det0 = scratch.det0.as_mut_slice();

        unsafe {
            det0[0]  = *xptr.offset(xr0 + c0 * xstr[1]);
            det0[1]  = *yptr.offset(yr0 + c1 * ystr[1]);
            det0[2]  = *yptr.offset(yr0 + c2 * ystr[1]);
            det0[3]  = *yptr.offset(yr0 + c3 * ystr[1]);

            det0[4]  = *xptr.offset(xr1 + c0 * xstr[1]);
            det0[5]  = *xptr.offset(xr1 + c1 * xstr[1]);
            det0[6]  = *yptr.offset(yr1 + c2 * ystr[1]);
            det0[7]  = *yptr.offset(yr1 + c3 * ystr[1]);

            det0[8]  = *xptr.offset(xr2 + c0 * xstr[1]);
            det0[9]  = *xptr.offset(xr2 + c1 * xstr[1]);
            det0[10] = *xptr.offset(xr2 + c2 * xstr[1]);
            det0[11] = *yptr.offset(yr2 + c3 * ystr[1]);

            det0[12] = *xptr.offset(xr3 + c0 * xstr[1]);
            det0[13] = *xptr.offset(xr3 + c1 * xstr[1]);
            det0[14] = *xptr.offset(xr3 + c2 * xstr[1]);
            det0[15] = *xptr.offset(xr3 + c3 * xstr[1]);
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
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_gen, {
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

/// Construct the row and column indices used for a rank-1 contraction determinant.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra (`Lambda`) determinant.
/// - `g_ex`: Excitation defining the ket (`Gamma`) determinant.
/// - `nmo`: Number of molecular orbitals in a single determinant.
/// - `rows`: Output row indices.
/// - `cols`: Output column indices.
/// # Returns
/// - `()`: Writes the rank-1 contraction determinant indices into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices_l1(l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, nmo: usize, rows: &mut Vec<usize>, cols: &mut Vec<usize>) {
    time_call!(crate::timers::nonorthogonalwicks::add_construct_determinant_indices_l1, {
        let nl = l_ex.holes.len();

        rows.clear();
        cols.clear();

        if rows.capacity() < 1 {rows.reserve_exact(1 - rows.capacity());}
        if cols.capacity() < 1 {cols.reserve_exact(1 - cols.capacity());}

        unsafe {
            if nl == 1 {
                rows.push(*l_ex.parts.get_unchecked(0));
                cols.push(*l_ex.holes.get_unchecked(0));
            }
            else {
                rows.push(nmo + *g_ex.holes.get_unchecked(0));
                cols.push(nmo + *g_ex.parts.get_unchecked(0));
            }
        }
    })
}

/// Construct the row and column indices used for a rank-2 contraction determinant.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra (`Lambda`) determinant.
/// - `g_ex`: Excitation defining the ket (`Gamma`) determinant.
/// - `nmo`: Number of molecular orbitals in a single determinant.
/// - `rows`: Output row indices.
/// - `cols`: Output column indices.
/// # Returns
/// - `()`: Writes the rank-2 contraction determinant indices into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices_l2(l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, nmo: usize, rows: &mut Vec<usize>, cols: &mut Vec<usize>) {
    time_call!(crate::timers::nonorthogonalwicks::add_construct_determinant_indices_l2, {
        let nl = l_ex.holes.len();
        let ng = g_ex.holes.len();

        rows.clear();
        cols.clear();

        if rows.capacity() < 2 {rows.reserve_exact(2 - rows.capacity());}
        if cols.capacity() < 2 {cols.reserve_exact(2 - cols.capacity());}

        unsafe {
            if nl == 2 {
                rows.push(*l_ex.parts.get_unchecked(0));
                rows.push(*l_ex.parts.get_unchecked(1));
                cols.push(*l_ex.holes.get_unchecked(0));
                cols.push(*l_ex.holes.get_unchecked(1));
            }
            else if ng == 2 {
                rows.push(nmo + *g_ex.holes.get_unchecked(0));
                rows.push(nmo + *g_ex.holes.get_unchecked(1));
                cols.push(nmo + *g_ex.parts.get_unchecked(0));
                cols.push(nmo + *g_ex.parts.get_unchecked(1));
            }
            else {
                rows.push(*l_ex.parts.get_unchecked(0));
                rows.push(nmo + *g_ex.holes.get_unchecked(0));
                cols.push(*l_ex.holes.get_unchecked(0));
                cols.push(nmo + *g_ex.parts.get_unchecked(0));
            }
        }
    })
}

/// Construct the row and column indices used for a rank-3 contraction determinant.
/// Preserves the exact ordering used by `construct_determinant_indices`, but avoids the generic
/// branching and loops for the hot `l = 3` path. Indices are written in the concatenated orbital
/// space `[Lambda orbitals; Gamma orbitals]`, so any Gamma index is offset by `nmo`.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra (`Lambda`) determinant.
/// - `g_ex`: Excitation defining the ket (`Gamma`) determinant.
/// - `nmo`: Number of molecular orbitals in a single determinant.
/// - `rows`: Output row indices.
/// - `cols`: Output column indices.
/// # Returns
/// - `()`: Writes the rank-3 contraction determinant indices into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices_l3(l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, nmo: usize, rows: &mut Vec<usize>, cols: &mut Vec<usize>) {
    time_call!(crate::timers::nonorthogonalwicks::add_construct_determinant_indices_l3, {
        let nl = l_ex.holes.len();
        let ng = g_ex.holes.len();

        rows.clear();
        cols.clear();

        if rows.capacity() < 3 {rows.reserve_exact(3 - rows.capacity());}
        if cols.capacity() < 3 {cols.reserve_exact(3 - cols.capacity());}

        unsafe {
            match (nl, ng) {
                (3, 0) => {
                    rows.push(*l_ex.parts.get_unchecked(0));
                    rows.push(*l_ex.parts.get_unchecked(1));
                    rows.push(*l_ex.parts.get_unchecked(2));

                    cols.push(*l_ex.holes.get_unchecked(0));
                    cols.push(*l_ex.holes.get_unchecked(1));
                    cols.push(*l_ex.holes.get_unchecked(2));
                }
                (2, 1) => {
                    rows.push(*l_ex.parts.get_unchecked(0));
                    rows.push(*l_ex.parts.get_unchecked(1));
                    rows.push(nmo + *g_ex.holes.get_unchecked(0));

                    cols.push(*l_ex.holes.get_unchecked(0));
                    cols.push(*l_ex.holes.get_unchecked(1));
                    cols.push(nmo + *g_ex.parts.get_unchecked(0));
                }
                (1, 2) => {
                    rows.push(*l_ex.parts.get_unchecked(0));
                    rows.push(nmo + *g_ex.holes.get_unchecked(0));
                    rows.push(nmo + *g_ex.holes.get_unchecked(1));

                    cols.push(*l_ex.holes.get_unchecked(0));
                    cols.push(nmo + *g_ex.parts.get_unchecked(0));
                    cols.push(nmo + *g_ex.parts.get_unchecked(1));
                }
                (0, 3) => {
                    rows.push(nmo + *g_ex.holes.get_unchecked(0));
                    rows.push(nmo + *g_ex.holes.get_unchecked(1));
                    rows.push(nmo + *g_ex.holes.get_unchecked(2));

                    cols.push(nmo + *g_ex.parts.get_unchecked(0));
                    cols.push(nmo + *g_ex.parts.get_unchecked(1));
                    cols.push(nmo + *g_ex.parts.get_unchecked(2));
                }
                _ => unreachable!(),
            }
        }
    })
}

/// Construct the row and column indices used for a rank-4 contraction determinant.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra (`Lambda`) determinant.
/// - `g_ex`: Excitation defining the ket (`Gamma`) determinant.
/// - `nmo`: Number of molecular orbitals in a single determinant.
/// - `rows`: Output row indices.
/// - `cols`: Output column indices.
/// # Returns
/// - `()`: Writes the rank-4 contraction determinant indices into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices_l4(l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, nmo: usize, rows: &mut Vec<usize>, cols: &mut Vec<usize>) {
    time_call!(crate::timers::nonorthogonalwicks::add_construct_determinant_indices_l4, {
        let nl = l_ex.holes.len();
        let ng = g_ex.holes.len();

        rows.clear();
        cols.clear();

        if rows.capacity() < 4 {rows.reserve_exact(4 - rows.capacity());}
        if cols.capacity() < 4 {cols.reserve_exact(4 - cols.capacity());}

        unsafe {
            match (nl, ng) {
                (4, 0) => {
                    rows.push(*l_ex.parts.get_unchecked(0));
                    rows.push(*l_ex.parts.get_unchecked(1));
                    rows.push(*l_ex.parts.get_unchecked(2));
                    rows.push(*l_ex.parts.get_unchecked(3));

                    cols.push(*l_ex.holes.get_unchecked(0));
                    cols.push(*l_ex.holes.get_unchecked(1));
                    cols.push(*l_ex.holes.get_unchecked(2));
                    cols.push(*l_ex.holes.get_unchecked(3));
                }
                (3, 1) => {
                    rows.push(*l_ex.parts.get_unchecked(0));
                    rows.push(*l_ex.parts.get_unchecked(1));
                    rows.push(*l_ex.parts.get_unchecked(2));
                    rows.push(nmo + *g_ex.holes.get_unchecked(0));

                    cols.push(*l_ex.holes.get_unchecked(0));
                    cols.push(*l_ex.holes.get_unchecked(1));
                    cols.push(*l_ex.holes.get_unchecked(2));
                    cols.push(nmo + *g_ex.parts.get_unchecked(0));
                }
                (2, 2) => {
                    rows.push(*l_ex.parts.get_unchecked(0));
                    rows.push(*l_ex.parts.get_unchecked(1));
                    rows.push(nmo + *g_ex.holes.get_unchecked(0));
                    rows.push(nmo + *g_ex.holes.get_unchecked(1));

                    cols.push(*l_ex.holes.get_unchecked(0));
                    cols.push(*l_ex.holes.get_unchecked(1));
                    cols.push(nmo + *g_ex.parts.get_unchecked(0));
                    cols.push(nmo + *g_ex.parts.get_unchecked(1));
                }
                (1, 3) => {
                    rows.push(*l_ex.parts.get_unchecked(0));
                    rows.push(nmo + *g_ex.holes.get_unchecked(0));
                    rows.push(nmo + *g_ex.holes.get_unchecked(1));
                    rows.push(nmo + *g_ex.holes.get_unchecked(2));

                    cols.push(*l_ex.holes.get_unchecked(0));
                    cols.push(nmo + *g_ex.parts.get_unchecked(0));
                    cols.push(nmo + *g_ex.parts.get_unchecked(1));
                    cols.push(nmo + *g_ex.parts.get_unchecked(2));
                }
                (0, 4) => {
                    rows.push(nmo + *g_ex.holes.get_unchecked(0));
                    rows.push(nmo + *g_ex.holes.get_unchecked(1));
                    rows.push(nmo + *g_ex.holes.get_unchecked(2));
                    rows.push(nmo + *g_ex.holes.get_unchecked(3));

                    cols.push(nmo + *g_ex.parts.get_unchecked(0));
                    cols.push(nmo + *g_ex.parts.get_unchecked(1));
                    cols.push(nmo + *g_ex.parts.get_unchecked(2));
                    cols.push(nmo + *g_ex.parts.get_unchecked(3));
                }
                _ => unreachable!(),
            }
        }
    })
}
