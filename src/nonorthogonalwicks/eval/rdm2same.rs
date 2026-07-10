// nonorthogonalwicks/eval/rdm2same.rs

use ndarray::{Array2, Array4};

use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;

use super::helpers::{det_slice, extend_rdm_d, for_each_m_combination};
use super::prepare::construct_determinant_indices_gen;

use crate::ExcitationSpin;
use crate::noci::NOCIScalar;

use crate::maths::{build_d, mix_columns};
use crate::time_call;

/// Calculate the same-spin two-body RDM matrix element between two determinants
/// |{}^\Lambda \Psi\rangle and |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's
/// theorem prescription. Dispatches to the zero-overlap fast path when `w.m == 0` and otherwise
/// to the full generic zero-distribution path.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `l_c`: Left determinant orbital coefficients in the RDM basis.
/// - `g_c`: Right determinant orbital coefficients in the RDM basis.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `Array4<T>`: Same-spin two-body RDM matrix element.
#[inline(always)]
pub(crate) fn lg_rdm2_same<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array4<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_rdm2_same, {
        if w.m == 0 {
            lg_rdm2_same_m0(w, l_ex, g_ex, l_c, g_c, scratch, tol)
        } else {
            lg_rdm2_same_gen(w, l_ex, g_ex, l_c, g_c, scratch, tol)
        }
    })
}

/// Calculate the same-spin two-body RDM matrix element for the zero-overlap case
/// `w.m == 0`. This forms the determinant obtained by adding the two external
/// contractions from a^\dagger_p a^\dagger_q a_s a_r to the usual excitation
/// contraction determinant.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `l_c`: Left determinant orbital coefficients in the RDM basis.
/// - `g_c`: Right determinant orbital coefficients in the RDM basis.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `Array4<T>`: Same-spin two-body RDM matrix element in the `m = 0` case.
#[inline(always)]
fn lg_rdm2_same_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array4<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_rdm2_same_m0, {
        let n = l_c.nrows();
        let l = l_ex.holes.len() + g_ex.holes.len();
        let dim = l + 2;
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);
        let x0 = w.x(0);
        let y0 = w.y(0);
        let x0rdm = w.xrdm(0, n);
        let y0rdm = w.yrdm(0, n);
        let x0p = extend_rdm_d(w, &x0, &x0rdm, l_c, g_c);
        let y0p = extend_rdm_d(w, &y0, &y0rdm, l_c, g_c);
        let x0p = x0p.view();
        let y0p = y0p.view();

        let mut out = Array4::<T>::zeros((n, n, n, n));
        let mut rows_base = Vec::with_capacity(l);
        let mut cols_base = Vec::with_capacity(l);
        let mut rows = Vec::with_capacity(dim);
        let mut cols = Vec::with_capacity(dim);
        let mut det0 = vec![zero; dim * dim];

        construct_determinant_indices_gen(l_ex, g_ex, w, &mut rows_base, &mut cols_base);

        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        rows.clear();
                        cols.clear();

                        rows.push(w.nmo + p);
                        rows.push(w.nmo + q);
                        rows.extend_from_slice(rows_base.as_slice());

                        cols.push(w.nmo + r);
                        cols.push(w.nmo + s);
                        cols.extend_from_slice(cols_base.as_slice());

                        build_d(&mut det0, dim, &x0p, &y0p, rows.as_slice(), cols.as_slice());

                        if let Some(d) = det_slice(det0.as_slice(), dim)
                            && d.abs() > tol
                        {
                            out[(p, q, r, s)] = pref * d;
                        }
                    }
                }
            }
        }

        let _ = scratch;

        out
    })
}

/// Calculate the same-spin two-body RDM matrix element for the generic zero-overlap
/// case `w.m > 0`. This evaluates the sum over all allowed zero-distribution bitstrings
/// for the determinant obtained by adding the two external contractions from
/// a^\dagger_p a^\dagger_q a_s a_r to the usual excitation contraction determinant.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `l_ex`: Spin-resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin-resolved excitation array for |{}^\Gamma \Psi\rangle.
/// - `l_c`: Left determinant orbital coefficients in the RDM basis.
/// - `g_c`: Right determinant orbital coefficients in the RDM basis.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `Array4<T>`: Same-spin two-body RDM matrix element.
#[inline(always)]
fn lg_rdm2_same_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array4<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_rdm2_same_gen, {
        let n = l_c.nrows();
        let l = l_ex.holes.len() + g_ex.holes.len();
        let dim = l + 2;
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);
        let x0 = w.x(0);
        let y0 = w.y(0);
        let x1 = w.x(1);
        let y1 = w.y(1);
        let x0rdm = w.xrdm(0, n);
        let y0rdm = w.yrdm(0, n);
        let x1rdm = w.xrdm(1, n);
        let y1rdm = w.yrdm(1, n);
        let x0p = extend_rdm_d(w, &x0, &x0rdm, l_c, g_c);
        let y0p = extend_rdm_d(w, &y0, &y0rdm, l_c, g_c);
        let x1p = extend_rdm_d(w, &x1, &x1rdm, l_c, g_c);
        let y1p = extend_rdm_d(w, &y1, &y1rdm, l_c, g_c);
        let x0p = x0p.view();
        let y0p = y0p.view();
        let x1p = x1p.view();
        let y1p = y1p.view();

        let mut out = Array4::<T>::zeros((n, n, n, n));
        let mut rows_base = Vec::with_capacity(l);
        let mut cols_base = Vec::with_capacity(l);
        let mut rows = Vec::with_capacity(dim);
        let mut cols = Vec::with_capacity(dim);
        let mut det0 = vec![zero; dim * dim];
        let mut det1 = vec![zero; dim * dim];
        let mut detm = vec![zero; dim * dim];

        construct_determinant_indices_gen(l_ex, g_ex, w, &mut rows_base, &mut cols_base);

        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        rows.clear();
                        cols.clear();

                        rows.push(w.nmo + p);
                        rows.push(w.nmo + q);
                        rows.extend_from_slice(rows_base.as_slice());

                        cols.push(w.nmo + r);
                        cols.push(w.nmo + s);
                        cols.extend_from_slice(cols_base.as_slice());

                        build_d(&mut det0, dim, &x0p, &y0p, rows.as_slice(), cols.as_slice());
                        build_d(&mut det1, dim, &x1p, &y1p, rows.as_slice(), cols.as_slice());

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

                        out[(p, q, r, s)] = pref * acc;
                    }
                }
            }
        }

        let _ = scratch;

        out
    })
}
