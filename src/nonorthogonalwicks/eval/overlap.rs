// nonorthogonalwicks/eval/overlap.rs
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::mix_dets_same;
use super::prepare::prepare_same;
use crate::ExcitationSpin;
use crate::maths::{det, det_lu_l5, det_lu_l6};
use crate::noci::NOCIScalar;
use crate::time_call;

/// Calculate overlap matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription.
/// Dispatches to zero-overlap and fully-zeroed fast paths where possible.
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `T`: Overlap matrix element.
#[inline(always)]
pub fn lg_overlap<T: NOCIScalar>(
    w: &SameSpinView<T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap, {
        let l = l_ex.holes.len() + g_ex.holes.len();

        if w.m > l {
            <T as From<f64>>::from(0.0)
        } else if w.m == 0 {
            lg_overlap_m0(w, l, scratch)
        } else if w.m == l {
            lg_overlap_ml(w, l, scratch)
        } else {
            lg_overlap_gen(w, l, scratch)
        }
    })
}

/// Calculate a same-spin overlap, dispatching to the overlap-only direct path when applicable.
/// This is the Wick-layer overlap API used by factor-table construction:
/// S_{\Lambda\Gamma} = \phi\,\tilde S \det D_{\Lambda\Gamma}.
/// Keeping the `m = 0` specialization here avoids leaking Wick evaluation details into NOCI
/// overlap application code, while retaining the generic prepared-scratch path for other cases.
/// # Arguments:
/// - `w`: Same-spin Wick intermediates for the ordered parent pair.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space used by the generic prepared path.
/// # Returns
/// - `f64`: Same-spin overlap excluding determinant excitation phases.
#[inline(always)]
pub(crate) fn lg_overlap_same_f64(
    w: &SameSpinView<'_, f64>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<f64>,
) -> f64 {
    let l = l_ex.holes.len() + g_ex.holes.len();

    if w.m > l {
        return 0.0;
    }

    if w.m == 0 && l <= 6 {
        return lg_overlap_m0_direct_f64(w, l_ex, g_ex);
    }

    prepare_same(w, l_ex, g_ex, scratch);
    lg_overlap(w, l_ex, g_ex, scratch)
}

/// Calculate a same-spin overlap directly for the `m = 0` Wick path.
/// This uses the same row and column ordering as `prepare_same`, but builds only the
/// determinant needed for the overlap:
/// S_{\Lambda\Gamma} = \phi\,\tilde S \det D^{(0)}_{\Lambda\Gamma}.
/// The direct path avoids the separate scratch preparation and overlap dispatch that are needed
/// by Hamiltonian/Fock routines but redundant for overlap-only factor table construction.
/// # Arguments:
/// - `w`: Same-spin Wick intermediates with `m = 0`.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// # Returns
/// - `f64`: Same-spin overlap excluding determinant excitation phases.
#[inline(always)]
pub(crate) fn lg_overlap_m0_direct_f64(
    w: &SameSpinView<'_, f64>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
) -> f64 {
    let nl = l_ex.holes.len();
    let ng = g_ex.holes.len();
    let l = nl + ng;
    let pref = w.phase * w.tilde_s_prod;

    if l == 0 {
        return pref;
    }

    let n = w.n();
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    let x0 = w.x_slice(0);
    let y0 = w.y_slice(0);
    let mut rows = [0usize; 6];
    let mut cols = [0usize; 6];

    for k in 0..nl {
        rows[k] = l_ex.parts[k] - nocc;
        cols[k] = l_ex.holes[k];
    }

    for k in 0..ng {
        let i = nl + k;
        rows[i] = nvirt + g_ex.holes[k];
        cols[i] = g_ex.parts[k];
    }

    match l {
        1 => pref * x0[rows[0] * n + cols[0]],
        2 => {
            let d0 = x0[rows[0] * n + cols[0]];
            let d1 = y0[rows[0] * n + cols[1]];
            let d2 = x0[rows[1] * n + cols[0]];
            let d3 = x0[rows[1] * n + cols[1]];

            pref * (d0 * d3 - d1 * d2)
        }
        3 => {
            let d0 = x0[rows[0] * n + cols[0]];
            let d1 = y0[rows[0] * n + cols[1]];
            let d2 = y0[rows[0] * n + cols[2]];
            let d3 = x0[rows[1] * n + cols[0]];
            let d4 = x0[rows[1] * n + cols[1]];
            let d5 = y0[rows[1] * n + cols[2]];
            let d6 = x0[rows[2] * n + cols[0]];
            let d7 = x0[rows[2] * n + cols[1]];
            let d8 = x0[rows[2] * n + cols[2]];

            pref * (d0 * (d4 * d8 - d5 * d7) - d1 * (d3 * d8 - d5 * d6)
                + d2 * (d3 * d7 - d4 * d6))
        }
        5 => {
            let mut lu = [0.0; 25];

            for i in 0..5 {
                let row = rows[i] * n;

                for j in 0..5 {
                    lu[i * 5 + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            pref * det_lu_l5(&mut lu).unwrap_or(0.0)
        }
        6 => {
            let mut lu = [0.0; 36];

            for i in 0..6 {
                let row = rows[i] * n;

                for j in 0..6 {
                    lu[i * 6 + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            pref * det_lu_l6(&mut lu).unwrap_or(0.0)
        }
        _ => {
            let mut d = [0.0; 36];

            for i in 0..l {
                let row = rows[i] * n;

                for j in 0..l {
                    d[i * l + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            pref * det(&d[..l * l], l).unwrap_or(0.0)
        }
    }
}

/// Calculate overlap matrix elements for the zero-overlap case `w.m == 0`.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `l`: Total excitation rank entering the determinant.
/// - `scratch`: Scratch space containing the prepared `det0`.
/// # Returns
/// - `T`: Overlap matrix element in the `m = 0` case.
#[inline(always)]
fn lg_overlap_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0, {
        match l {
            0 => w.phase * <T as From<f64>>::from(w.tilde_s_prod),
            1 => lg_overlap_m0_l1(w, scratch),
            2 => lg_overlap_m0_l2(w, scratch),
            3 => lg_overlap_m0_l3(w, scratch),
            5 => lg_overlap_m0_l5(w, scratch),
            6 => lg_overlap_m0_l6(w, scratch),
            _ => {
                w.phase
                    * <T as From<f64>>::from(w.tilde_s_prod)
                    * det(scratch.det0.as_slice(), l).unwrap_or(<T as From<f64>>::from(0.0))
            }
        }
    })
}

/// Calculate the specialized `l = 1`, `m = 0` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `det0`.
/// # Returns
/// - `T`: Overlap matrix element for `l = 1`.
#[inline(always)]
fn lg_overlap_m0_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0_l1, {
        let d = scratch.det0.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * d[0]
    })
}

/// Calculate the specialized `l = 2`, `m = 0` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `det0`.
/// # Returns
/// - `T`: Overlap matrix element for `l = 2`.
#[inline(always)]
fn lg_overlap_m0_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0_l2, {
        let d = scratch.det0.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * (d[0] * d[3] - d[1] * d[2])
    })
}

/// Calculate the specialized `l = 3`, `m = 0` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `det0`.
/// # Returns
/// - `T`: Overlap matrix element for `l = 3`.
#[inline(always)]
fn lg_overlap_m0_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_m0_l3, {
        let d = scratch.det0.as_slice();
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * (d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
                + d[2] * (d[3] * d[7] - d[4] * d[6]))
    })
}

/// Calculate the specialized `l = 5`, `m = 0` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `det0`.
/// # Returns
/// - `T`: Overlap matrix element for `l = 5`.
#[inline(always)]
fn lg_overlap_m0_l5<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    let mut lu = [T::from_real(0.0); 25];
    lu.copy_from_slice(&scratch.det0.as_slice()[..25]);
    let det = det_lu_l5(&mut lu).unwrap_or(<T as From<f64>>::from(0.0));
    w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det
}

/// Calculate the specialized `l = 6`, `m = 0` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `scratch`: Scratch space containing the prepared `det0`.
/// # Returns
/// - `T`: Overlap matrix element for `l = 6`.
#[inline(always)]
fn lg_overlap_m0_l6<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    let mut lu = [T::from_real(0.0); 36];
    lu.copy_from_slice(&scratch.det0.as_slice()[..36]);
    let det = det_lu_l6(&mut lu).unwrap_or(<T as From<f64>>::from(0.0));
    w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det
}

/// Calculate overlap matrix elements when all determinant columns are zero-replacement columns,
/// i.e. `w.m == l`.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l`: Total excitation rank entering the determinant.
/// - `scratch`: Scratch space containing the prepared `det1`.
/// # Returns
/// - `T`: Overlap matrix element in the `m = l` case.
#[inline(always)]
fn lg_overlap_ml<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml, {
        match l {
            0 => w.phase * <T as From<f64>>::from(w.tilde_s_prod),
            1 => lg_overlap_ml_l1(w, scratch),
            2 => lg_overlap_ml_l2(w, scratch),
            3 => lg_overlap_ml_l3(w, scratch),
            _ => {
                w.phase
                    * <T as From<f64>>::from(w.tilde_s_prod)
                    * det(scratch.det1.as_slice(), l).unwrap_or(<T as From<f64>>::from(0.0))
            }
        }
    })
}

/// Calculate the specialized `l = 1`, `m = l` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `scratch`: Scratch space containing the prepared `det1`.
/// # Returns
/// - `T`: Overlap matrix element for `l = 1`.
#[inline(always)]
fn lg_overlap_ml_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml_l1, {
        let d = scratch.det1.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * d[0]
    })
}

/// Calculate the specialized `l = 2`, `m = l` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `scratch`: Scratch space containing the prepared `det1`.
/// # Returns
/// - `T`: Overlap matrix element for `l = 2`.
#[inline(always)]
fn lg_overlap_ml_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml_l2, {
        let d = scratch.det1.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * (d[0] * d[3] - d[1] * d[2])
    })
}

/// Calculate the specialized `l = 3`, `m = l` overlap.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `scratch`: Scratch space containing the prepared `det1`.
/// # Returns
/// - `T`: Overlap matrix element for `l = 3`.
#[inline(always)]
fn lg_overlap_ml_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_ml_l3, {
        let d = scratch.det1.as_slice();
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * (d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
                + d[2] * (d[3] * d[7] - d[4] * d[6]))
    })
}

/// Calculate overlap matrix elements for the general `0 < w.m < l` case.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l`: Total excitation rank entering the determinant.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `T`: Overlap matrix element for the general mixed-column path.
#[inline(always)]
fn lg_overlap_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_overlap_gen, {
        let mut acc = <T as From<f64>>::from(0.0);
        mix_dets_same(w, l, 0, scratch, |_, scratch| {
            let d = scratch.det_mix.as_slice();
            let contrib = match l {
                1 => d[0],
                2 => d[0] * d[3] - d[1] * d[2],
                3 => {
                    d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
                        + d[2] * (d[3] * d[7] - d[4] * d[6])
                }
                _ => det(d, l).unwrap_or(<T as From<f64>>::from(0.0)),
            };
            acc += contrib;
        });
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
    })
}
