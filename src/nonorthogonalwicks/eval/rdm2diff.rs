// nonorthogonalwicks/eval/rdm2same.rs

// External crate imports.
use ndarray::{Array2, Array4};

// Crate-root imports.
use crate::ExcitationSpin;
use crate::maths::{build_d, mix_columns};
use crate::noci::NOCIScalar;
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{det_slice, extend_rdm_d, for_each_m_combination};
use super::prepare::construct_determinant_indices;

/// Evaluate the unnormalised same-spin two-body transition density matrix between excited
/// `determinants generated from the reference pair \langle{}^x\Psi| and |{}^w\Psi\rangle:`
/// `{}^{xw}\Gamma_{\sigma\sigma}{}^{pq}_{rs}`
/// `= \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat a^\dagger_{p\sigma}\hat a^\dagger_{q\sigma}`
/// `\hat a_{s\sigma}\hat a_{r\sigma}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+2}\\m_1+\cdots+m_{L+2}=m}}`
/// `\det\mathbf D_{\mathrm{RDM}}^{pqrs}(m_1,\ldots,m_{L+2}).`
/// The two external creation-annihilation pairs augment the overlap contraction determinant from
/// dimension L to L + 2. The first two assignments belong to the external RDM columns and the
/// `remaining L assignments belong to the excitation columns. Each m_i is zero or one, and the`
/// matrix element vanishes when m > L + 2. Expansion of the determinant generates the direct and
/// exchange contraction patterns with their fermionic signs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: `Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.`
/// - `g_ex`: `Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.`
/// - `l_c`: Bra-reference molecular-orbital coefficients in the external RDM basis.
/// - `g_c`: Ket-reference molecular-orbital coefficients in the external RDM basis.
/// - `scratch`: Scratch storage retained for the common Wick evaluator interface.
/// - `tol`: Numerical threshold applied to determinant contributions.
/// # Returns
/// - `Array4<T>`: Unnormalised same-spin two-body transition density matrix.
#[inline(always)]
pub(crate) fn xw_rdm2_same<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array4<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_rdm2_same, {
        // For m = 0 only \mathbf D_{\mathrm{RDM}}^{pqrs}(0,\ldots,0) contributes. Otherwise,
        // sum every distribution satisfying \sum_{i=1}^{L+2}m_i = m.
        if w.m == 0 {
            xw_rdm2_same_m0(w, l_ex, g_ex, l_c, g_c, scratch, tol)
        } else {
            xw_rdm2_same_gen(w, l_ex, g_ex, l_c, g_c, scratch, tol)
        }
    })
}

/// Evaluate the same-spin two-body transition density matrix when m = 0, so every contraction
/// `uses m_i = 0:`
/// `{}^{xw}\Gamma_{\sigma\sigma}{}^{pq}_{rs}`
/// `= {}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{pqrs}(0,\ldots,0).`
/// The augmented determinant has dimension L + 2 and contains the two external RDM pairs followed
/// `by the L excitation pairs used in \mathbf D_{\mathrm{ov}}.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `l_c`: Bra-reference molecular-orbital coefficients in the external RDM basis.
/// - `g_c`: Ket-reference molecular-orbital coefficients in the external RDM basis.
/// - `scratch`: Scratch storage retained for the common Wick evaluator interface.
/// - `tol`: Numerical threshold applied to determinant contributions.
/// # Returns
/// - `Array4<T>`: Same-spin two-body transition density matrix for m = 0.
#[inline(always)]
fn xw_rdm2_same_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array4<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_rdm2_same_m0, {
        // The external RDM basis has dimension n, while the augmented contraction determinant has
        // dimension L + 2 for L = L_x + L_w.
        let n = l_c.nrows();
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
        let dim = l + 2;
        // Recover the full reduced-overlap prefactor from the separately stored orbital-pairing
        // phase and product of non-zero singular values.
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);
        // Extend X^{(0)} and Y^{(0)} to include the external RDM basis and all cross contractions
        // between the RDM labels and the compact excitation row and column spaces.
        let x0 = w.x(0);
        let y0 = w.y(0);
        let x0rdm = w.xrdm(0, n);
        let y0rdm = w.yrdm(0, n);
        let x0p = extend_rdm_d(w, &x0, &x0rdm, l_c, g_c);
        let y0p = extend_rdm_d(w, &y0, &y0rdm, l_c, g_c);
        let x0p = x0p.view();
        let y0p = y0p.view();

        // Allocate the output tensor, the common excitation labels and the augmented determinant.
        let mut out = Array4::<T>::zeros((n, n, n, n));
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
                for r in 0..n {
                    for s in 0..n {
                        rows.clear();
                        cols.clear();

                        // Prepend the two external RDM row labels p,q and column labels r,s, then
                        // append the L excitation labels without changing their relative ordering.
                        rows.push(w.nmo + p);
                        rows.push(w.nmo + q);
                        rows.extend_from_slice(rows_base.as_slice());

                        cols.push(w.nmo + r);
                        cols.push(w.nmo + s);
                        cols.extend_from_slice(cols_base.as_slice());

                        // Build \mathbf D_{\mathrm{RDM}}^{pqrs}(0,\ldots,0), with X^{(0)} on and
                        // below the diagonal and Y^{(0)} above it. Its leading 2 \times 2 block
                        // generates the direct-minus-exchange contraction of the external operators.
                        build_d(&mut det0, dim, &x0p, &y0p, rows.as_slice(), cols.as_slice());

                        // {}^{xw}\Gamma_{\sigma\sigma}{}^{pq}_{rs}
                        // = {}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{pqrs}(0,\ldots,0).
                        // Numerically negligible determinant contributions are left as zero.
                        if let Some(d) = det_slice(det0.as_slice(), dim)
                            && d.abs() > tol
                        {
                            out[(p, q, r, s)] = pref * d;
                        }
                    }
                }
            }
        }

        // This evaluator currently uses local determinant storage but retains the common scratch argument.
        let _ = scratch;

        out
    })
}

/// Evaluate the same-spin two-body transition density matrix when m > 0 by summing every allowed
/// distribution:
/// `{}^{xw}\Gamma_{\sigma\sigma}{}^{pq}_{rs}`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+2}\\m_1+\cdots+m_{L+2}=m}}`
/// `\det\mathbf D_{\mathrm{RDM}}^{pqrs}(m_1,\ldots,m_{L+2}), \qquad m_i \in \{0,1\}.`
/// The first two assignments belong to the external RDM columns, while each remaining assignment
/// `selects the corresponding excitation column from the m_i = 0 or m_i = 1 fundamental contractions.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates containing one or more zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `l_c`: Bra-reference molecular-orbital coefficients in the external RDM basis.
/// - `g_c`: Ket-reference molecular-orbital coefficients in the external RDM basis.
/// - `scratch`: Scratch storage retained for the common Wick evaluator interface.
/// - `tol`: Numerical threshold applied to determinant contributions.
/// # Returns
/// - `Array4<T>`: Same-spin two-body transition density matrix summed over all allowed distributions.
#[inline(always)]
fn xw_rdm2_same_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Array4<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_rdm2_same_gen, {
        // The external RDM basis has dimension n, while the augmented contraction determinant has
        // dimension L + 2 for L = L_x + L_w.
        let n = l_c.nrows();
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
        let dim = l + 2;
        // Recover the full reduced-overlap prefactor from the separately stored orbital-pairing
        // phase and product of non-zero singular values.
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);
        // Extend both fundamental-contraction choices to include the external RDM basis and the
        // cross contractions connecting it to the excitation row and column spaces.
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

        // Store the all-m_i = 0 and all-m_i = 1 endpoint determinants and one mixed determinant.
        let mut out = Array4::<T>::zeros((n, n, n, n));
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
                for r in 0..n {
                    for s in 0..n {
                        rows.clear();
                        cols.clear();

                        // Prepend the two external RDM row labels so columns zero and one carry
                        // m_1 and m_2; the appended excitation columns carry m_3,\ldots,m_{L+2}.
                        rows.push(w.nmo + p);
                        rows.push(w.nmo + q);
                        rows.extend_from_slice(rows_base.as_slice());

                        cols.push(w.nmo + r);
                        cols.push(w.nmo + s);
                        cols.extend_from_slice(cols_base.as_slice());

                        // Construct the endpoint determinants
                        // \mathbf D_{\mathrm{RDM}}^{pqrs}(0,\ldots,0) and
                        // \mathbf D_{\mathrm{RDM}}^{pqrs}(1,\ldots,1).
                        build_d(&mut det0, dim, &x0p, &y0p, rows.as_slice(), cols.as_slice());
                        build_d(&mut det1, dim, &x1p, &y1p, rows.as_slice(), cols.as_slice());

                        let mut acc = zero;

                        // Enumerate every distribution satisfying \sum_{i=1}^{L+2}m_i = m. For
                        // each distribution, select column i from `det0` for m_i = 0 and from
                        // `det1` for m_i = 1.
                        for_each_m_combination(dim, w.m, |bits| {
                            mix_columns(
                                detm.as_mut_slice(),
                                det0.as_slice(),
                                det1.as_slice(),
                                dim,
                                bits,
                            );

                            // Add \det\mathbf D_{\mathrm{RDM}}^{pqrs}(m_1,\ldots,m_{L+2})
                            // to the constrained sum when it is numerically significant.
                            if let Some(d) = det_slice(detm.as_slice(), dim)
                                && d.abs() > tol
                            {
                                acc += d;
                            }
                        });

                        // Multiply the constrained determinant sum by the reduced-overlap prefactor.
                        out[(p, q, r, s)] = pref * acc;
                    }
                }
            }
        }

        // This evaluator currently uses local determinant storage but retains the common scratch argument.
        let _ = scratch;

        out
    })
}
