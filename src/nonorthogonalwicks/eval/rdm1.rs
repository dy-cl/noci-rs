// nonorthogonalwicks/eval/rdm1.rs

use ndarray::Array2;

use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;

use super::helpers::{det_slice, extend_rdm_d, for_each_m_combination};
use super::prepare::construct_determinant_indices;

use crate::ExcitationSpin;
use crate::noci::NOCIScalar;

use crate::maths::{build_d, mix_columns};
use crate::time_call;

/// Evaluate the unnormalised spin-resolved one-body transition density matrix between excited
/// determinants generated from the reference pair \langle{}^x\Psi| and |{}^w\Psi\rangle:
/// {}^{xw}\gamma_\sigma{}^p_q = \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat a^\dagger_{p\sigma}\hat a_{q\sigma}
/// |{}^w\Psi_{j\cdots}^{b\cdots}\rangle
/// = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+1}\\m_1+\cdots+m_{L+1}=m}}
/// \det\mathbf D_{\mathrm{RDM}}^{pq}(m_1,\ldots,m_{L+1}).
/// The external creation-annihilation pair augments the overlap contraction determinant from dimension
/// L to L + 1. The first assignment belongs to this external pair and the remaining L assignments
/// belong to the excitation columns. Each m_i is zero or one, and the matrix element vanishes when
/// m > L + 1. The implementation stores the orbital-pairing phase separately from the product of
/// non-zero singular values forming {}^{xw}\tilde S.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.
/// - `g_ex`: Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.
/// - `l_c`: Bra-reference molecular-orbital coefficients in the external RDM basis.
/// - `g_c`: Ket-reference molecular-orbital coefficients in the external RDM basis.
/// - `scratch`: Scratch storage retained for the common Wick evaluator interface.
/// - `tol`: Numerical threshold applied to determinant contributions.
/// # Returns
/// - `Array2<T>`: Unnormalised spin-resolved one-body transition density matrix.
#[inline(always)]
pub(crate) fn xw_rdm1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array2<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_rdm1, {
        // For m = 0 only \mathbf D_{\mathrm{RDM}}^{pq}(0,\ldots,0) contributes. Otherwise,
        // sum every distribution satisfying \sum_{i=1}^{L+1}m_i = m.
        if w.m == 0 {
            xw_rdm1_m0(w, l_ex, g_ex, l_c, g_c, scratch, tol)
        } else {
            xw_rdm1_gen(w, l_ex, g_ex, l_c, g_c, scratch, tol)
        }
    })
}

/// Evaluate the one-body transition density matrix when m = 0, so every contraction uses m_i = 0:
/// {}^{xw}\gamma_\sigma{}^p_q = {}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{pq}(0,\ldots,0).
/// The augmented determinant has dimension L + 1 and contains the external RDM pair followed by the
/// L excitation pairs used in \mathbf D_{\mathrm{ov}}.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `l_c`: Bra-reference molecular-orbital coefficients in the external RDM basis.
/// - `g_c`: Ket-reference molecular-orbital coefficients in the external RDM basis.
/// - `scratch`: Scratch storage retained for the common Wick evaluator interface.
/// - `tol`: Numerical threshold applied to determinant contributions.
/// # Returns
/// - `Array2<T>`: One-body transition density matrix for m = 0.
#[inline(always)]
fn xw_rdm1_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array2<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_rdm1_m0, {
        // The external RDM basis has dimension n, while the augmented contraction determinant has
        // dimension L + 1 for L = L_x + L_w.
        let n = l_c.nrows();
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
        let dim = l + 1;
        // Recover the full reduced-overlap prefactor \phi^{xw}{}^{xw}\tilde S from the separately
        // stored orbital-pairing phase and product of non-zero singular values.
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);
        // Extend X^{(0)} and Y^{(0)} to include the external RDM basis and all cross contractions
        // between the RDM labels and the compact excitation row and column spaces.
        let x0 = w.x(0);
        let y0 = w.y(0);
        let x0p = extend_rdm_d(w, &x0, &w.xrdm(0, n), l_c, g_c);
        let y0p = extend_rdm_d(w, &y0, &w.yrdm(0, n), l_c, g_c);
        let x0p = x0p.view();
        let y0p = y0p.view();

        // Allocate the output matrix, the common excitation labels and the augmented determinant.
        let mut out = Array2::<T>::zeros((n, n));
        let mut rows_base = Vec::with_capacity(l);
        let mut cols_base = Vec::with_capacity(l);
        let mut rows = Vec::with_capacity(dim);
        let mut cols = Vec::with_capacity(dim);
        let mut det0 = vec![zero; dim * dim];

        // Construct the L excitation labels in the same V_x \cup O_w row ordering and
        // O_x \cup V_w column ordering used by \mathbf D_{\mathrm{ov}}.
        rows_base.resize(l, 0);
        cols_base.resize(l, 0);
        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            rows_base.as_mut_slice(),
            cols_base.as_mut_slice(),
        );

        for p in 0..n {
            for q in 0..n {
                rows.clear();
                cols.clear();

                // Prepend the external RDM labels so the first determinant column corresponds to
                // the contraction containing \hat a^\dagger_{p\sigma}\hat a_{q\sigma}; append the
                // L excitation labels without changing their relative ordering.
                rows.push(w.nmo + p);
                rows.extend_from_slice(rows_base.as_slice());

                cols.push(w.nmo + q);
                cols.extend_from_slice(cols_base.as_slice());

                // Build \mathbf D_{\mathrm{RDM}}^{pq}(0,\ldots,0), with X^{(0)} on and below
                // the diagonal and Y^{(0)} above the diagonal.
                build_d(&mut det0, dim, &x0p, &y0p, rows.as_slice(), cols.as_slice());

                // {}^{xw}\gamma_\sigma{}^p_q = \phi^{xw}{}^{xw}\tilde S
                // \det\mathbf D_{\mathrm{RDM}}^{pq}(0,\ldots,0). Numerically negligible
                // determinant contributions are left as zero.
                if let Some(d) = det_slice(det0.as_slice(), dim)
                    && d.abs() > tol
                {
                    out[(p, q)] = pref * d;
                }
            }
        }

        // This evaluator currently uses local determinant storage but retains the common scratch argument.
        let _ = scratch;

        out
    })
}

/// Evaluate the one-body transition density matrix when m > 0 by summing every allowed distribution:
/// {}^{xw}\gamma_\sigma{}^p_q = {}^{xw}\tilde S
/// \sum_{\substack{m_1,\ldots,m_{L+1}\\m_1+\cdots+m_{L+1}=m}}
/// \det\mathbf D_{\mathrm{RDM}}^{pq}(m_1,\ldots,m_{L+1}), \qquad m_i \in \{0,1\}.
/// The first assignment belongs to the external RDM pair, while each remaining assignment selects the
/// corresponding excitation column from the m_i = 0 or m_i = 1 fundamental contractions.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates containing one or more zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `l_c`: Bra-reference molecular-orbital coefficients in the external RDM basis.
/// - `g_c`: Ket-reference molecular-orbital coefficients in the external RDM basis.
/// - `scratch`: Scratch storage retained for the common Wick evaluator interface.
/// - `tol`: Numerical threshold applied to determinant contributions.
/// # Returns
/// - `Array2<T>`: One-body transition density matrix summed over all allowed distributions.
#[inline(always)]
fn xw_rdm1_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array2<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_rdm1_gen, {
        // The external RDM basis has dimension n, while the augmented contraction determinant has
        // dimension L + 1 for L = L_x + L_w.
        let n = l_c.nrows();
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
        let dim = l + 1;
        // Recover the full reduced-overlap prefactor \phi^{xw}{}^{xw}\tilde S.
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);
        // Extend both fundamental-contraction choices to include the external RDM basis and the
        // cross contractions connecting it to the excitation row and column spaces.
        let x0 = w.x(0);
        let y0 = w.y(0);
        let x1 = w.x(1);
        let y1 = w.y(1);
        let x0p = extend_rdm_d(w, &x0, &w.xrdm(0, n), l_c, g_c);
        let y0p = extend_rdm_d(w, &y0, &w.yrdm(0, n), l_c, g_c);
        let x1p = extend_rdm_d(w, &x1, &w.xrdm(1, n), l_c, g_c);
        let y1p = extend_rdm_d(w, &y1, &w.yrdm(1, n), l_c, g_c);
        let x0p = x0p.view();
        let y0p = y0p.view();
        let x1p = x1p.view();
        let y1p = y1p.view();

        // Store the all-m_i = 0 and all-m_i = 1 endpoint determinants and one mixed determinant.
        let mut out = Array2::<T>::zeros((n, n));
        let mut rows_base = Vec::with_capacity(l);
        let mut cols_base = Vec::with_capacity(l);
        let mut rows = Vec::with_capacity(dim);
        let mut cols = Vec::with_capacity(dim);
        let mut det0 = vec![zero; dim * dim];
        let mut det1 = vec![zero; dim * dim];
        let mut detm = vec![zero; dim * dim];

        // Construct the L excitation labels in the ordering used by \mathbf D_{\mathrm{ov}}.
        rows_base.resize(l, 0);
        cols_base.resize(l, 0);
        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            rows_base.as_mut_slice(),
            cols_base.as_mut_slice(),
        );

        for p in 0..n {
            for q in 0..n {
                rows.clear();
                cols.clear();

                // Prepend the external RDM labels so column zero carries m_1; columns 1,\ldots,L
                // retain the excitation ordering and carry m_2,\ldots,m_{L+1}.
                rows.push(w.nmo + p);
                rows.extend_from_slice(rows_base.as_slice());

                cols.push(w.nmo + q);
                cols.extend_from_slice(cols_base.as_slice());

                // Construct the endpoint determinants \mathbf D_{\mathrm{RDM}}^{pq}(0,\ldots,0)
                // and \mathbf D_{\mathrm{RDM}}^{pq}(1,\ldots,1).
                build_d(&mut det0, dim, &x0p, &y0p, rows.as_slice(), cols.as_slice());
                build_d(&mut det1, dim, &x1p, &y1p, rows.as_slice(), cols.as_slice());

                let mut acc = zero;

                // Enumerate every distribution satisfying \sum_{i=1}^{L+1}m_i = m. For each
                // distribution, select column i from `det0` when m_i = 0 and from `det1` when m_i = 1.
                for_each_m_combination(dim, w.m, |bits| {
                    mix_columns(
                        detm.as_mut_slice(),
                        det0.as_slice(),
                        det1.as_slice(),
                        dim,
                        bits,
                    );

                    // Add \det\mathbf D_{\mathrm{RDM}}^{pq}(m_1,\ldots,m_{L+1}) to the
                    // constrained sum when it is numerically significant.
                    if let Some(d) = det_slice(detm.as_slice(), dim)
                        && d.abs() > tol
                    {
                        acc += d;
                    }
                });

                // Multiply the constrained determinant sum by the reduced-overlap prefactor.
                out[(p, q)] = pref * acc;
            }
        }

        // This evaluator currently uses local determinant storage but retains the common scratch argument.
        let _ = scratch;

        out
    })
}
