// nonorthogonalwicks/eval/rdmksame.rs

use ndarray::Array2;

use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;

use super::helpers::{det_slice, extend_rdm_d, for_each_m_combination};
use super::overlap::xw_overlap;
use super::prepare::construct_determinant_indices;
use super::rdm1::xw_rdm1;

use crate::ExcitationSpin;
use crate::noci::NOCIScalar;

use crate::maths::{build_d, det, mix_columns};

/// Evaluate an unnormalised arbitrary-rank same-spin transition density-matrix element between
/// excited determinants generated from the reference pair \langle{}^x\Psi| and |{}^w\Psi\rangle:
/// {}^{xw}\Gamma_\sigma{}^{p_1\cdots p_k}_{q_1\cdots q_k}
/// = \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat a^\dagger_{p_1\sigma}\cdots
/// \hat a^\dagger_{p_k\sigma}\hat a_{q_k\sigma}\cdots\hat a_{q_1\sigma}
/// |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.
/// For nonzero excited-determinant overlap S, the generalised Wick theorem gives:
/// {}^{xw}\Gamma_\sigma{}^{p_1\cdots p_k}_{q_1\cdots q_k}
/// = \det[{}^{xw}\gamma_\sigma{}^{p_i}_{q_j}]_{ij}/S^{k-1}.
/// When S is numerically zero, the element is evaluated directly from the augmented contraction
/// determinant of dimension L + k:
/// {}^{xw}\Gamma_\sigma{}^{p_1\cdots p_k}_{q_1\cdots q_k}
/// = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+k}\\m_1+\cdots+m_{L+k}=m}}
/// \det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(m_1,\ldots,m_{L+k}).
/// The first k assignments belong to the external pairs (p_i,q_i), while the remaining L assignments
/// belong to the excitation columns. Each m_i is zero or one, and the element vanishes when m > L + k.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `coeff`: Bra- and ket-reference molecular-orbital coefficients in the external RDM basis.
/// - `indices`: Creation indices \mathbf p and annihilation indices \mathbf q in the external RDM basis;
///   both slices must have the same length k.
/// - `scratch`: Scratch storage used by the overlap and one-body evaluators.
/// - `tol`: Numerical threshold used to select the factorised or direct evaluation and to discard
///   negligible determinant contributions.
/// # Returns:
/// - `T`: Unnormalised same-spin rank-k transition density-matrix element.
#[inline(always)]
pub(crate) fn xw_rdm_same_element<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    indices: (&[usize], &[usize]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    // Unpack the bra and ket excitations, external-basis coefficients and rank-k RDM indices.
    let (l_ex, g_ex) = ex;
    let (l_c, g_c) = coeff;
    let (ps, qs) = indices;
    let k = ps.len();
    // S = \langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle.
    let s = xw_overlap(w, l_ex, g_ex, scratch);

    // The rank-zero transition density is the overlap itself.
    if k == 0 {
        return s;
    }

    // For S \neq 0, form the rank-k transition density from the one-body transition density:
    // {}^{xw}\Gamma_\sigma{}^{p_1\cdots p_k}_{q_1\cdots q_k}
    // = \det[{}^{xw}\gamma_\sigma{}^{p_i}_{q_j}]_{ij}/S^{k-1}.
    if s.abs() > tol {
        let g1 = xw_rdm1(w, l_ex, g_ex, l_c, g_c, scratch, tol);
        let zero = <T as From<f64>>::from(0.0);
        let mut d = vec![zero; k * k];

        // Construct the k \times k matrix with elements
        // d_{ij} = {}^{xw}\gamma_\sigma{}^{p_i}_{q_j}.
        for i in 0..k {
            for j in 0..k {
                d[i * k + j] = g1[(ps[i], qs[j])];
            }
        }

        // The determinant contains the antisymmetrised sum over all pairings of the external
        // creation and annihilation operators.
        let mut v = det(d.as_slice(), k).unwrap_or(zero);

        // Since `g1` is unnormalised, divide by S^{k-1} to recover the unnormalised rank-k element.
        for _ in 1..k {
            v /= s;
        }

        return v;
    }

    // When the excited-determinant overlap is numerically zero, division by S is unavailable.
    // Evaluate the augmented contraction determinant directly, selecting the m = 0 or m > 0
    // reference-pair path according to the number of zero-overlap orbital pairs.
    if w.m == 0 {
        xw_rdm_same_element_m0(w, ex, coeff, indices, scratch, tol)
    } else {
        xw_rdm_same_element_gen(w, ex, coeff, indices, scratch, tol)
    }
}

/// Evaluate an arbitrary-rank same-spin transition density-matrix element directly when the
/// reference pair has no zero-overlap orbital pairs:
/// {}^{xw}\Gamma_\sigma{}^{p_1\cdots p_k}_{q_1\cdots q_k}
/// = {}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0).
/// This path is used when the excited-determinant overlap is numerically zero even though m = 0.
/// The augmented determinant has dimension L + k and contains the k external pairs followed by
/// the L excitation pairs used in \mathbf D_{\mathrm{ov}}.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `coeff`: Bra- and ket-reference molecular-orbital coefficients in the external RDM basis.
/// - `indices`: Creation indices \mathbf p and annihilation indices \mathbf q, each of length k.
/// - `scratch`: Scratch storage retained for the common Wick evaluator interface.
/// - `tol`: Numerical threshold applied to the augmented determinant contribution.
/// # Returns:
/// - `T`: Unnormalised same-spin rank-k transition density-matrix element for m = 0.
#[inline(always)]
fn xw_rdm_same_element_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    indices: (&[usize], &[usize]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    // Determine the external rank k, excitation rank L and augmented determinant dimension L + k.
    let (l_ex, g_ex) = ex;
    let (l_c, g_c) = coeff;
    let (ps, qs) = indices;
    let k = ps.len();
    let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
    let dim = l + k;
    // Recover the full reduced-overlap prefactor from the separately stored orbital-pairing
    // phase and product of non-zero singular values.
    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
    let zero = <T as From<f64>>::from(0.0);
    let n = l_c.nrows();

    // Extend X^{(0)} and Y^{(0)} to include the external RDM basis and all cross contractions
    // between the external labels and the compact excitation row and column spaces.
    let x0 = w.x(0);
    let y0 = w.y(0);
    let x0rdm = w.xrdm(0, n);
    let y0rdm = w.yrdm(0, n);
    let x0p = extend_rdm_d(w, &x0, &x0rdm, l_c, g_c);
    let y0p = extend_rdm_d(w, &y0, &y0rdm, l_c, g_c);
    let x0p = x0p.view();
    let y0p = y0p.view();

    // Allocate the excitation labels, augmented row and column labels and determinant storage.
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

    // Prepend the external creation labels p_1,\ldots,p_k to the augmented determinant rows.
    for &p in ps {
        rows.push(w.nmo + p);
    }

    rows.extend_from_slice(rows_base.as_slice());

    // Prepend the corresponding annihilation labels q_1,\ldots,q_k to the augmented determinant columns.
    for &q in qs {
        cols.push(w.nmo + q);
    }

    cols.extend_from_slice(cols_base.as_slice());

    // Build \mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0), with X^{(0)} on and
    // below the diagonal and Y^{(0)} above it. Its determinant generates the antisymmetrised
    // sum over all complete contractions of the external and excitation operators.
    build_d(&mut det0, dim, &x0p, &y0p, rows.as_slice(), cols.as_slice());

    // This direct evaluator currently uses local determinant storage but retains the common scratch argument.
    let _ = scratch;

    // {}^{xw}\Gamma_\sigma{}^{p_1\cdots p_k}_{q_1\cdots q_k}
    // = {}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0).
    if let Some(d) = det_slice(det0.as_slice(), dim)
        && d.abs() > tol
    {
        pref * d
    } else {
        zero
    }
}

/// Evaluate an arbitrary-rank same-spin transition density-matrix element directly when the
/// reference pair contains one or more zero-overlap orbital pairs:
/// {}^{xw}\Gamma_\sigma{}^{p_1\cdots p_k}_{q_1\cdots q_k}
/// = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+k}\\m_1+\cdots+m_{L+k}=m}}
/// \det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(m_1,\ldots,m_{L+k}),
/// \qquad m_i \in \{0,1\}.
/// The first k assignments belong to the external pairs (p_i,q_i), while the remaining L
/// assignments select the excitation columns from the m_i = 0 or m_i = 1 fundamental contractions.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates containing one or more zero-overlap orbital pairs.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `coeff`: Bra- and ket-reference molecular-orbital coefficients in the external RDM basis.
/// - `indices`: Creation indices \mathbf p and annihilation indices \mathbf q, each of length k.
/// - `scratch`: Scratch storage retained for the common Wick evaluator interface.
/// - `tol`: Numerical threshold applied to determinant contributions.
/// # Returns:
/// - `T`: Unnormalised same-spin rank-k transition density-matrix element summed over all allowed distributions.
#[inline(always)]
fn xw_rdm_same_element_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    indices: (&[usize], &[usize]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    // Determine the external rank k, excitation rank L and augmented determinant dimension L + k.
    let (l_ex, g_ex) = ex;
    let (l_c, g_c) = coeff;
    let (ps, qs) = indices;
    let k = ps.len();
    let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
    let dim = l + k;
    // Recover the full reduced-overlap prefactor from the separately stored orbital-pairing
    // phase and product of non-zero singular values.
    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
    let zero = <T as From<f64>>::from(0.0);
    let n = l_c.nrows();

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

    // Allocate the excitation labels, augmented labels and endpoint and mixed determinants.
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

    // Prepend the external creation labels so determinant columns 0,\ldots,k-1 carry
    // m_1,\ldots,m_k; the appended excitation columns carry m_{k+1},\ldots,m_{L+k}.
    for &p in ps {
        rows.push(w.nmo + p);
    }

    rows.extend_from_slice(rows_base.as_slice());

    // Prepend the corresponding external annihilation labels in the same pair ordering.
    for &q in qs {
        cols.push(w.nmo + q);
    }

    cols.extend_from_slice(cols_base.as_slice());

    // Construct the endpoint determinants \mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)
    // and \mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(1,\ldots,1).
    build_d(&mut det0, dim, &x0p, &y0p, rows.as_slice(), cols.as_slice());
    build_d(&mut det1, dim, &x1p, &y1p, rows.as_slice(), cols.as_slice());

    let mut acc = zero;

    // Enumerate every distribution satisfying \sum_{i=1}^{L+k}m_i = m. For each distribution,
    // select column i from `det0` for m_i = 0 and from `det1` for m_i = 1.
    for_each_m_combination(dim, w.m, |bits| {
        mix_columns(
            detm.as_mut_slice(),
            det0.as_slice(),
            det1.as_slice(),
            dim,
            bits,
        );

        // Add \det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(m_1,\ldots,m_{L+k})
        // to the constrained sum when it is numerically significant.
        if let Some(d) = det_slice(detm.as_slice(), dim)
            && d.abs() > tol
        {
            acc += d;
        }
    });

    // This direct evaluator currently uses local determinant storage but retains the common scratch argument.
    let _ = scratch;

    // Multiply the constrained determinant sum by the reduced-overlap prefactor.
    pref * acc
}
