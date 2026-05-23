// nonorthogonalwicks/eval/rdm1.rs

use ndarray::Array2;

use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{construct_determinant_indices_gen, det_slice, for_each_m_combination};
use crate::ExcitationSpin;
use crate::maths::{build_d, mix_columns};
use crate::noci::NOCIScalar;
use crate::time_call;

/// Calculate the spin-block one-body RDM matrix element between two determinants
/// |{}^\Lambda \Psi\rangle and |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's
/// theorem prescription:
/// {}^{\Lambda\Gamma}D_\sigma{}^p_q
///     = \langle{}^\Lambda\Psi_\sigma|
///       a^\dagger_{p\sigma} a_{q\sigma}
///       |{}^\Gamma\Psi_\sigma\rangle.
/// Dispatches to the zero-overlap fast path when `w.m == 0` and otherwise to the full generic
/// zero-distribution path.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `Array2<T>`: Spin-block one-body RDM matrix element.
#[inline(always)]
pub(crate) fn lg_rdm1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array2<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_rdm1, {
        if w.m == 0 {
            lg_rdm1_m0(w, l_ex, g_ex, scratch, tol)
        } else {
            lg_rdm1_gen(w, l_ex, g_ex, scratch, tol)
        }
    })
}

/// Calculate the spin-block one-body RDM matrix element for the zero-overlap case `w.m == 0`.
/// This forms the determinant obtained by adding the external contraction from
/// a^\dagger_p a_q to the usual excitation contraction determinant.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `Array2<T>`: Spin-block one-body RDM matrix element in the `m = 0` case.
#[inline(always)]
fn lg_rdm1_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array2<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_rdm1_m0, {
        let n = w.nmo;
        let l = l_ex.holes.len() + g_ex.holes.len();
        let dim = l + 1;
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);
        let x0 = w.x(0);
        let y0 = w.y(0);

        let mut out = Array2::<T>::zeros((n, n));
        let mut rows_base = Vec::with_capacity(l);
        let mut cols_base = Vec::with_capacity(l);
        let mut rows = Vec::with_capacity(dim);
        let mut cols = Vec::with_capacity(dim);
        let mut det0 = vec![zero; dim * dim];

        construct_determinant_indices_gen(l_ex, g_ex, w.nmo, &mut rows_base, &mut cols_base);

        for p in 0..n {
            for q in 0..n {
                rows.clear();
                cols.clear();

                rows.push(p);
                rows.extend_from_slice(rows_base.as_slice());

                cols.push(w.nmo + q);
                cols.extend_from_slice(cols_base.as_slice());

                build_d(&mut det0, dim, &x0, &y0, rows.as_slice(), cols.as_slice());

                if let Some(d) = det_slice(det0.as_slice(), dim)
                    && d.abs() > tol
                {
                    out[(p, q)] = pref * d;
                }
            }
        }

        let _ = scratch;
        out
    })
}

/// Calculate the spin-block one-body RDM matrix element for the generic zero-overlap case
/// `w.m > 0`. This evaluates the sum over all allowed zero-distribution bitstrings for the
/// determinant obtained by adding the external contraction from a^\dagger_p a_q to the usual
/// excitation contraction determinant.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `Array2<T>`: Spin-block one-body RDM matrix element.
#[inline(always)]
fn lg_rdm1_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array2<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_rdm1_gen, {
        let n = w.nmo;
        let l = l_ex.holes.len() + g_ex.holes.len();
        let dim = l + 1;
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);
        let x0 = w.x(0);
        let y0 = w.y(0);
        let x1 = w.x(1);
        let y1 = w.y(1);

        let mut out = Array2::<T>::zeros((n, n));
        let mut rows_base = Vec::with_capacity(l);
        let mut cols_base = Vec::with_capacity(l);
        let mut rows = Vec::with_capacity(dim);
        let mut cols = Vec::with_capacity(dim);
        let mut det0 = vec![zero; dim * dim];
        let mut det1 = vec![zero; dim * dim];
        let mut detm = vec![zero; dim * dim];

        construct_determinant_indices_gen(l_ex, g_ex, w.nmo, &mut rows_base, &mut cols_base);

        for p in 0..n {
            for q in 0..n {
                rows.clear();
                cols.clear();

                rows.push(p);
                rows.extend_from_slice(rows_base.as_slice());

                cols.push(w.nmo + q);
                cols.extend_from_slice(cols_base.as_slice());

                build_d(&mut det0, dim, &x0, &y0, rows.as_slice(), cols.as_slice());
                build_d(&mut det1, dim, &x1, &y1, rows.as_slice(), cols.as_slice());

                let mut acc = zero;

                for_each_m_combination(dim, w.m, |bits| {
                    mix_columns(
                        detm.as_mut_slice(),
                        det0.as_slice(),
                        det1.as_slice(),
                        dim,
                        bits,
                    );

                    if let Some(d) = det_slice(detm.as_slice(), dim)
                        && d.abs() > tol
                    {
                        acc += d;
                    }
                });

                out[(p, q)] = pref * acc;
            }
        }

        let _ = scratch;
        out
    })
}
