// nonorthogonalwicks/eval/rdmksame.rs

use ndarray::Array2;

use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{
    construct_determinant_indices_gen, det_slice, extend_rdm_d, for_each_m_combination,
};
use super::overlap::lg_overlap;
use super::rdm1::lg_rdm1;
use crate::ExcitationSpin;
use crate::maths::{build_d, det, mix_columns};
use crate::noci::NOCIScalar;

/// Calculate a same-spin RDM element using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `ex`: Left and right spin-resolved excitation arrays.
/// - `coeff`: Left and right determinant orbital coefficients in the RDM basis.
/// - `indices`: Creation and annihilation indices in the RDM basis.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns:
/// - `T`: Same-spin RDM element for the supplied creation and annihilation indices.
#[inline(always)]
pub(crate) fn lg_rdm_same_element<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    indices: (&[usize], &[usize]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    let (l_ex, g_ex) = ex;
    let (l_c, g_c) = coeff;
    let (ps, qs) = indices;
    let k = ps.len();
    let s = lg_overlap(w, l_ex, g_ex, scratch);

    if k == 0 {
        return s;
    }

    if s.abs() > tol {
        let g1 = lg_rdm1(w, l_ex, g_ex, l_c, g_c, scratch, tol);
        let zero = <T as From<f64>>::from(0.0);
        let mut d = vec![zero; k * k];

        for i in 0..k {
            for j in 0..k {
                d[i * k + j] = g1[(ps[i], qs[j])];
            }
        }

        let mut v = det(d.as_slice(), k).unwrap_or(zero);
        for _ in 1..k {
            v /= s;
        }
        return v;
    }

    if w.m == 0 {
        lg_rdm_same_element_m0(w, ex, coeff, indices, scratch, tol)
    } else {
        lg_rdm_same_element_gen(w, ex, coeff, indices, scratch, tol)
    }
}

/// Calculate a same-spin RDM element for the zero-overlap case `w.m == 0`.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates with `m = 0`.
/// - `ex`: Left and right spin-resolved excitation arrays.
/// - `coeff`: Left and right determinant orbital coefficients in the RDM basis.
/// - `indices`: Creation and annihilation indices in the RDM basis.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns:
/// - `T`: Same-spin RDM element in the `m = 0` case.
#[inline(always)]
fn lg_rdm_same_element_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    indices: (&[usize], &[usize]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    let (l_ex, g_ex) = ex;
    let (l_c, g_c) = coeff;
    let (ps, qs) = indices;
    let k = ps.len();
    let l = l_ex.holes.len() + g_ex.holes.len();
    let dim = l + k;
    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
    let zero = <T as From<f64>>::from(0.0);

    let x0 = w.x(0);
    let y0 = w.y(0);
    let x0p = extend_rdm_d(&x0, l_c, g_c, w.nmo);
    let y0p = extend_rdm_d(&y0, l_c, g_c, w.nmo);
    let x0p = x0p.view();
    let y0p = y0p.view();

    let mut rows_base = Vec::with_capacity(l);
    let mut cols_base = Vec::with_capacity(l);
    let mut rows = Vec::with_capacity(dim);
    let mut cols = Vec::with_capacity(dim);
    let mut det0 = vec![zero; dim * dim];

    construct_determinant_indices_gen(l_ex, g_ex, w.nmo, &mut rows_base, &mut cols_base);

    for &p in ps {
        rows.push(2 * w.nmo + p);
    }
    rows.extend_from_slice(rows_base.as_slice());

    for &q in qs {
        cols.push(2 * w.nmo + q);
    }
    cols.extend_from_slice(cols_base.as_slice());

    build_d(&mut det0, dim, &x0p, &y0p, rows.as_slice(), cols.as_slice());

    let _ = scratch;

    if let Some(d) = det_slice(det0.as_slice(), dim)
        && d.abs() > tol
    {
        pref * d
    } else {
        zero
    }
}

/// Calculate a same-spin RDM element for the generic zero-overlap case `w.m > 0`.
/// # Arguments:
/// - `w`: Same-spin Wick's reference pair intermediates.
/// - `ex`: Left and right spin-resolved excitation arrays.
/// - `coeff`: Left and right determinant orbital coefficients in the RDM basis.
/// - `indices`: Creation and annihilation indices in the RDM basis.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns:
/// - `T`: Same-spin RDM element in the generic zero-overlap case.
#[inline(always)]
fn lg_rdm_same_element_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    indices: (&[usize], &[usize]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    let (l_ex, g_ex) = ex;
    let (l_c, g_c) = coeff;
    let (ps, qs) = indices;
    let k = ps.len();
    let l = l_ex.holes.len() + g_ex.holes.len();
    let dim = l + k;
    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
    let zero = <T as From<f64>>::from(0.0);

    let x0 = w.x(0);
    let y0 = w.y(0);
    let x1 = w.x(1);
    let y1 = w.y(1);

    let x0p = extend_rdm_d(&x0, l_c, g_c, w.nmo);
    let y0p = extend_rdm_d(&y0, l_c, g_c, w.nmo);
    let x1p = extend_rdm_d(&x1, l_c, g_c, w.nmo);
    let y1p = extend_rdm_d(&y1, l_c, g_c, w.nmo);

    let x0p = x0p.view();
    let y0p = y0p.view();
    let x1p = x1p.view();
    let y1p = y1p.view();

    let mut rows_base = Vec::with_capacity(l);
    let mut cols_base = Vec::with_capacity(l);
    let mut rows = Vec::with_capacity(dim);
    let mut cols = Vec::with_capacity(dim);
    let mut det0 = vec![zero; dim * dim];
    let mut det1 = vec![zero; dim * dim];
    let mut detm = vec![zero; dim * dim];

    construct_determinant_indices_gen(l_ex, g_ex, w.nmo, &mut rows_base, &mut cols_base);

    for &p in ps {
        rows.push(2 * w.nmo + p);
    }
    rows.extend_from_slice(rows_base.as_slice());

    for &q in qs {
        cols.push(2 * w.nmo + q);
    }
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

    let _ = scratch;

    pref * acc
}
