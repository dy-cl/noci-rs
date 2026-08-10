// nonorthogonalwicks/eval/h2same.rs

use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{DetIndex, Minor, ReplacementLayout};
use crate::ExcitationSpin;
use crate::noci::NOCIScalar;
use crate::time_call;

use super::super::layout::{idx, idx4};
use super::helpers::{
    bit, column_replacement_correction, column_replacement_det, get_det_adjt_same, j_replacement,
    jslot, minor_adjt,
};
use crate::maths::adjugate_transpose;

/// Evaluate the same-spin two-body matrix element between excited determinants generated from the
/// reference pair \langle{}^x\Psi| and |{}^w\Psi\rangle:
/// \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat v|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
/// = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+2}\\m_1+\cdots+m_{L+2}=m}}
/// [{}^xV_0^{(m_1,m_2)}\det\mathbf D_{\mathrm{ov}}
/// - 2\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal V_z}
/// + \sum_{z<y}\sum_{\eta<\xi}\phi_{\eta\xi}^{zy}\mathcal J_{\eta z,\xi y}
///   \det\mathbf D_{\mathrm{ov}}[\eta,\xi|z,y]].
///   The first two assignments belong to the operator contractions and the remaining L assignments
///   belong to the columns of \mathbf D_{\mathrm{ov}}. Each m_i is zero or one, and the assignments
///   carried by \mathcal V and \mathcal J are selected from the same distribution. The implementation
///   stores the orbital-pairing phase separately from the product of non-zero singular values.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.
/// - `g_ex`: Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.
/// - `scratch`: Scratch storage for contraction determinants, cofactors, minors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: Same-spin two-body matrix element.
#[inline(always)]
pub(crate) fn xw_h2_same<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_same, {
        // For m = 0 only \mathbf D_{\mathrm{ov}}(0,\ldots,0) contributes. For m > 0,
        // enumerate the distributions satisfying \sum_{i=1}^{L+2}m_i = m.
        if w.m == 0 {
            xw_h2_same_m0(w, l_ex, g_ex, scratch, tol)
        } else {
            xw_h2_same_gen(w, l_ex, g_ex, scratch, tol)
        }
    })
}

/// Evaluate the same-spin two-body matrix element when m = 0, so every contraction uses m_i = 0.
/// Fixed-rank kernels are used for L = 1,\ldots,4, while all other excitation ranks use the
/// general cofactor form. For L = 0 only the scalar intermediate {}^xV_0^{(0,0)} contributes.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Prepared m_i = 0 contraction determinant and scratch work arrays.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: Same-spin two-body matrix element for m = 0.
#[inline(always)]
fn xw_h2_same_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_same_m0, {
        // Determine the total excitation rank L = L_x + L_w.
        let l = l_ex.holes.len() + g_ex.holes.len();
        // Dispatch to direct fixed-rank forms of C_1 + C_2 + C_3.
        match l {
            // For L = 0, C_1 = {}^xV_0^{(0,0)} and C_2 = C_3 = 0.
            0 => w.phase * <T as From<f64>>::from(w.tilde_s_prod) * w.v0[0],
            1 => xw_h2_same_m0_l1(w, scratch),
            2 => xw_h2_same_m0_l2(w, scratch),
            3 => xw_h2_same_m0_l3(w, scratch, tol),
            4 => xw_h2_same_m0_l4(w, scratch, tol),
            _ => xw_h2_same_m0_gen(w, l_ex, g_ex, scratch, tol),
        }
    })
}

/// Evaluate the fixed-rank L = 1 same-spin two-body matrix element for m = 0.
/// The scalar and one-column terms contribute, while the two-column \mathcal J term is absent.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-one contraction determinant and its row and column labels.
/// # Returns
/// - `T`: Same-spin two-body matrix element for L = 1 and m = 0.
#[inline(always)]
fn xw_h2_same_m0_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_same_m0_l1, {
        // For L = 1, \mathbf D_{\mathrm{ov}} = [D_{00}] and no pair of excitation columns
        // exists for the two-column contribution C_3.
        let n = w.n();
        let r0 = scratch.rows[0];
        let c0 = scratch.cols[0];
        let det0 = scratch.det0.as_slice();
        let det = det0[0];
        let vsl = w.v_t_slice(0, 0, 0);
        // \det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal V_0} = \mathcal V_{r_0c_0}.
        let repl = vsl[c0 * n + r0];

        // H = {}^{xw}\tilde S[{}^xV_0^{(0,0)}\det\mathbf D_{\mathrm{ov}}
        // - 2\det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal V_0}].
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * (w.v0[0] * det - <T as From<f64>>::from(2.0) * repl)
    })
}

/// Evaluate the fixed-rank L = 2 same-spin two-body matrix element for m = 0.
/// The scalar, two one-column replacements and rank-two \mathcal J contribution are evaluated directly.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-two contraction determinant and its row and column labels.
/// # Returns
/// - `T`: Same-spin two-body matrix element for L = 2 and m = 0.
#[inline(always)]
fn xw_h2_same_m0_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_same_m0_l2, {
        // Evaluate \det\mathbf D_{\mathrm{ov}} = D_{00}D_{11} - D_{01}D_{10}.
        let n = w.n();
        let d = scratch.det0.as_slice();
        let a00 = d[0];
        let a01 = d[1];
        let a10 = d[2];
        let a11 = d[3];
        let det = a00 * a11 - a01 * a10;

        let r0 = scratch.rows[0];
        let r1 = scratch.rows[1];
        let c0 = scratch.cols[0];
        let c1 = scratch.cols[1];

        let vsl = w.v_t_slice(0, 0, 0);

        // Form \det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal V_z} for z = 0,1.
        let u0 = vsl[c0 * n + r0];
        let u1 = vsl[c0 * n + r1];
        let v0 = vsl[c1 * n + r0];
        let v1 = vsl[c1 * n + r1];

        let det_c0 = u0 * a11 - a01 * u1;
        let det_c1 = a00 * v1 - v0 * a10;

        // For L = 2 the two-column contribution reduces to
        // C_3 = 2(\mathcal J_{r_0c_0,r_1c_1} - \mathcal J_{r_0c_1,r_1c_0}).
        let jsl = w.j_slice(0);
        let direct = jsl[idx4(n, r0, c0, r1, c1)];
        let exchange = jsl[idx4(n, r0, c1, r1, c0)];
        let jterm = <T as From<f64>>::from(2.0) * (direct - exchange);

        // H = {}^{xw}\tilde S[{}^xV_0^{(0,0)}\det\mathbf D_{\mathrm{ov}}
        // - 2\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal V_z} + C_3].
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * (w.v0[0] * det - <T as From<f64>>::from(2.0) * (det_c0 + det_c1) + jterm)
    })
}

/// Evaluate the fixed-rank L = 3 same-spin two-body matrix element for m = 0.
/// The \mathcal V term is contracted with the cofactor matrix of \mathbf D_{\mathrm{ov}}, while
/// each \mathcal J term is contracted directly with the corresponding second minor.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-three contraction determinant and scratch storage for its cofactors.
/// - `tol`: Numerical tolerance used when evaluating the determinant and its cofactors.
/// # Returns
/// - `T`: Same-spin two-body matrix element for L = 3 and m = 0.
#[inline(always)]
fn xw_h2_same_m0_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_same_m0_l3, {
        // Select the rank-three contraction determinant and its row and column labels.
        let n = w.n();
        let rows = scratch.rows.as_slice();
        let cols = scratch.cols.as_slice();
        let det0 = &scratch.det0.as_slice()[..9];

        // Evaluate \det\mathbf D_{\mathrm{ov}} and
        // \operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}.
        if let Some(det) = adjugate_transpose(
            scratch.adjt_det.as_mut_slice(),
            scratch.invs.as_mut_slice(),
            scratch.lu.as_mut_slice(),
            det0,
            3,
            tol,
        ) {
            let cof = scratch.adjt_det.as_slice();
            let vsl = w.v_t_slice(0, 0, 0);

            let r0 = rows[0];
            let r1 = rows[1];
            let r2 = rows[2];
            let c0 = cols[0];
            let c1 = cols[1];
            let c2 = cols[2];

            // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal V_z}
            // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal V_{\eta z}.
            let vterm = cof[idx(3, 0, 0)] * vsl[c0 * n + r0]
                + cof[idx(3, 1, 0)] * vsl[c0 * n + r1]
                + cof[idx(3, 2, 0)] * vsl[c0 * n + r2]
                + cof[idx(3, 0, 1)] * vsl[c1 * n + r0]
                + cof[idx(3, 1, 1)] * vsl[c1 * n + r1]
                + cof[idx(3, 2, 1)] * vsl[c1 * n + r2]
                + cof[idx(3, 0, 2)] * vsl[c2 * n + r0]
                + cof[idx(3, 1, 2)] * vsl[c2 * n + r1]
                + cof[idx(3, 2, 2)] * vsl[c2 * n + r2];

            // C_3 = 2\sum_{\eta<\xi}\sum_{z<y}
            // (-1)^{\eta+\xi+z+y}
            // \det\mathbf D_{\mathrm{ov}}[\eta,\xi|z,y]
            // (\mathcal J_{\eta z,\xi y}-\mathcal J_{\eta y,\xi z}).
            // For L = 3 each second minor is 1 x 1, so all nine distinct
            // row-pair/column-pair contributions can be evaluated directly.
            let jsl = w.j_slice(0);
            let n2 = n * n;

            // Precompute the fixed first-pair offsets in the flattened rank-four
            // \mathcal J tensor so each contraction requires only additions.
            let base00 = (r0 * n + c0) * n2;
            let base01 = (r0 * n + c1) * n2;
            let base02 = (r0 * n + c2) * n2;
            let base10 = (r1 * n + c0) * n2;
            let base11 = (r1 * n + c1) * n2;
            let base12 = (r1 * n + c2) * n2;

            let r1n = r1 * n;
            let r2n = r2 * n;

            let mut jterm = <T as From<f64>>::from(0.0);

            // (\eta,\xi) = (0,1).
            jterm += det0[8] * (jsl[base00 + r1n + c1] - jsl[base01 + r1n + c0]);
            jterm -= det0[7] * (jsl[base00 + r1n + c2] - jsl[base02 + r1n + c0]);
            jterm += det0[6] * (jsl[base01 + r1n + c2] - jsl[base02 + r1n + c1]);

            // (\eta,\xi) = (0,2).
            jterm -= det0[5] * (jsl[base00 + r2n + c1] - jsl[base01 + r2n + c0]);
            jterm += det0[4] * (jsl[base00 + r2n + c2] - jsl[base02 + r2n + c0]);
            jterm -= det0[3] * (jsl[base01 + r2n + c2] - jsl[base02 + r2n + c1]);

            // (\eta,\xi) = (1,2).
            jterm += det0[2] * (jsl[base10 + r2n + c1] - jsl[base11 + r2n + c0]);
            jterm -= det0[1] * (jsl[base10 + r2n + c2] - jsl[base12 + r2n + c0]);
            jterm += det0[0] * (jsl[base11 + r2n + c2] - jsl[base12 + r2n + c1]);

            let jterm = <T as From<f64>>::from(2.0) * jterm;

            // H = {}^{xw}\tilde S[{}^xV_0^{(0,0)}\det\mathbf D_{\mathrm{ov}}
            // - 2\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal V_z} + C_3].
            w.phase
                * <T as From<f64>>::from(w.tilde_s_prod)
                * (w.v0[0] * det - <T as From<f64>>::from(2.0) * vterm + jterm)
        } else {
            <T as From<f64>>::from(0.0)
        }
    })
}

/// Evaluate the fixed-rank L = 4 same-spin two-body matrix element for m = 0.
/// The \mathcal V term is contracted with the cofactor matrix of \mathbf D_{\mathrm{ov}}, while
/// each \mathcal J term is contracted directly with the corresponding second minor.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-four contraction determinant and scratch storage for its cofactors and minors.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: Same-spin two-body matrix element for L = 4 and m = 0.
#[inline(always)]
fn xw_h2_same_m0_l4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_same_m0_l4, {
        // Select the rank-four contraction determinant and its row and column labels.
        let n = w.n();
        let rows = scratch.rows.as_slice();
        let cols = scratch.cols.as_slice();
        let det0 = &scratch.det0.as_slice()[..16];

        // Evaluate \det\mathbf D_{\mathrm{ov}} and
        // \operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}.
        if let Some(det) = adjugate_transpose(
            scratch.adjt_det.as_mut_slice(),
            scratch.invs.as_mut_slice(),
            scratch.lu.as_mut_slice(),
            det0,
            4,
            tol,
        ) {
            let cof = scratch.adjt_det.as_slice();
            let vsl = w.v_t_slice(0, 0, 0);

            let r0 = rows[0];
            let r1 = rows[1];
            let r2 = rows[2];
            let r3 = rows[3];
            let c0 = cols[0];
            let c1 = cols[1];
            let c2 = cols[2];
            let c3 = cols[3];

            // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal V_z}
            // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal V_{\eta z}.
            let vterm = cof[idx(4, 0, 0)] * vsl[c0 * n + r0]
                + cof[idx(4, 1, 0)] * vsl[c0 * n + r1]
                + cof[idx(4, 2, 0)] * vsl[c0 * n + r2]
                + cof[idx(4, 3, 0)] * vsl[c0 * n + r3]
                + cof[idx(4, 0, 1)] * vsl[c1 * n + r0]
                + cof[idx(4, 1, 1)] * vsl[c1 * n + r1]
                + cof[idx(4, 2, 1)] * vsl[c1 * n + r2]
                + cof[idx(4, 3, 1)] * vsl[c1 * n + r3]
                + cof[idx(4, 0, 2)] * vsl[c2 * n + r0]
                + cof[idx(4, 1, 2)] * vsl[c2 * n + r1]
                + cof[idx(4, 2, 2)] * vsl[c2 * n + r2]
                + cof[idx(4, 3, 2)] * vsl[c2 * n + r3]
                + cof[idx(4, 0, 3)] * vsl[c3 * n + r0]
                + cof[idx(4, 1, 3)] * vsl[c3 * n + r1]
                + cof[idx(4, 2, 3)] * vsl[c3 * n + r2]
                + cof[idx(4, 3, 3)] * vsl[c3 * n + r3];

            // C_3 = 2\sum_{\eta<\xi}\sum_{z<y}
            // (-1)^{\eta+\xi+z+y}
            // \det\mathbf D_{\mathrm{ov}}[\eta,\xi|z,y]
            // (\mathcal J_{\eta z,\xi y}-\mathcal J_{\eta y,\xi z}).
            // For L = 4 each second minor is 2 x 2. Evaluating the 36 distinct
            // row-pair/column-pair combinations directly avoids constructing sixteen
            // rank-three first minors and their nine-element cofactor matrices.
            let jsl = w.j_slice(0);
            let n2 = n * n;

            // Precompute the fixed first-pair offsets in the flattened rank-four
            // \mathcal J tensor so each contraction requires only additions.
            let base00 = (r0 * n + c0) * n2;
            let base01 = (r0 * n + c1) * n2;
            let base02 = (r0 * n + c2) * n2;
            let base03 = (r0 * n + c3) * n2;

            let base10 = (r1 * n + c0) * n2;
            let base11 = (r1 * n + c1) * n2;
            let base12 = (r1 * n + c2) * n2;
            let base13 = (r1 * n + c3) * n2;

            let base20 = (r2 * n + c0) * n2;
            let base21 = (r2 * n + c1) * n2;
            let base22 = (r2 * n + c2) * n2;
            let base23 = (r2 * n + c3) * n2;

            let r1n = r1 * n;
            let r2n = r2 * n;
            let r3n = r3 * n;

            let mut jterm = <T as From<f64>>::from(0.0);

            // (\eta,\xi) = (0,1).
            jterm += (det0[10] * det0[15] - det0[11] * det0[14])
                * (jsl[base00 + r1n + c1] - jsl[base01 + r1n + c0]);
            jterm -= (det0[9] * det0[15] - det0[11] * det0[13])
                * (jsl[base00 + r1n + c2] - jsl[base02 + r1n + c0]);
            jterm += (det0[9] * det0[14] - det0[10] * det0[13])
                * (jsl[base00 + r1n + c3] - jsl[base03 + r1n + c0]);
            jterm += (det0[8] * det0[15] - det0[11] * det0[12])
                * (jsl[base01 + r1n + c2] - jsl[base02 + r1n + c1]);
            jterm -= (det0[8] * det0[14] - det0[10] * det0[12])
                * (jsl[base01 + r1n + c3] - jsl[base03 + r1n + c1]);
            jterm += (det0[8] * det0[13] - det0[9] * det0[12])
                * (jsl[base02 + r1n + c3] - jsl[base03 + r1n + c2]);

            // (\eta,\xi) = (0,2).
            jterm -= (det0[6] * det0[15] - det0[7] * det0[14])
                * (jsl[base00 + r2n + c1] - jsl[base01 + r2n + c0]);
            jterm += (det0[5] * det0[15] - det0[7] * det0[13])
                * (jsl[base00 + r2n + c2] - jsl[base02 + r2n + c0]);
            jterm -= (det0[5] * det0[14] - det0[6] * det0[13])
                * (jsl[base00 + r2n + c3] - jsl[base03 + r2n + c0]);
            jterm -= (det0[4] * det0[15] - det0[7] * det0[12])
                * (jsl[base01 + r2n + c2] - jsl[base02 + r2n + c1]);
            jterm += (det0[4] * det0[14] - det0[6] * det0[12])
                * (jsl[base01 + r2n + c3] - jsl[base03 + r2n + c1]);
            jterm -= (det0[4] * det0[13] - det0[5] * det0[12])
                * (jsl[base02 + r2n + c3] - jsl[base03 + r2n + c2]);

            // (\eta,\xi) = (0,3).
            jterm += (det0[6] * det0[11] - det0[7] * det0[10])
                * (jsl[base00 + r3n + c1] - jsl[base01 + r3n + c0]);
            jterm -= (det0[5] * det0[11] - det0[7] * det0[9])
                * (jsl[base00 + r3n + c2] - jsl[base02 + r3n + c0]);
            jterm += (det0[5] * det0[10] - det0[6] * det0[9])
                * (jsl[base00 + r3n + c3] - jsl[base03 + r3n + c0]);
            jterm += (det0[4] * det0[11] - det0[7] * det0[8])
                * (jsl[base01 + r3n + c2] - jsl[base02 + r3n + c1]);
            jterm -= (det0[4] * det0[10] - det0[6] * det0[8])
                * (jsl[base01 + r3n + c3] - jsl[base03 + r3n + c1]);
            jterm += (det0[4] * det0[9] - det0[5] * det0[8])
                * (jsl[base02 + r3n + c3] - jsl[base03 + r3n + c2]);

            // (\eta,\xi) = (1,2).
            jterm += (det0[2] * det0[15] - det0[3] * det0[14])
                * (jsl[base10 + r2n + c1] - jsl[base11 + r2n + c0]);
            jterm -= (det0[1] * det0[15] - det0[3] * det0[13])
                * (jsl[base10 + r2n + c2] - jsl[base12 + r2n + c0]);
            jterm += (det0[1] * det0[14] - det0[2] * det0[13])
                * (jsl[base10 + r2n + c3] - jsl[base13 + r2n + c0]);
            jterm += (det0[0] * det0[15] - det0[3] * det0[12])
                * (jsl[base11 + r2n + c2] - jsl[base12 + r2n + c1]);
            jterm -= (det0[0] * det0[14] - det0[2] * det0[12])
                * (jsl[base11 + r2n + c3] - jsl[base13 + r2n + c1]);
            jterm += (det0[0] * det0[13] - det0[1] * det0[12])
                * (jsl[base12 + r2n + c3] - jsl[base13 + r2n + c2]);

            // (\eta,\xi) = (1,3).
            jterm -= (det0[2] * det0[11] - det0[3] * det0[10])
                * (jsl[base10 + r3n + c1] - jsl[base11 + r3n + c0]);
            jterm += (det0[1] * det0[11] - det0[3] * det0[9])
                * (jsl[base10 + r3n + c2] - jsl[base12 + r3n + c0]);
            jterm -= (det0[1] * det0[10] - det0[2] * det0[9])
                * (jsl[base10 + r3n + c3] - jsl[base13 + r3n + c0]);
            jterm -= (det0[0] * det0[11] - det0[3] * det0[8])
                * (jsl[base11 + r3n + c2] - jsl[base12 + r3n + c1]);
            jterm += (det0[0] * det0[10] - det0[2] * det0[8])
                * (jsl[base11 + r3n + c3] - jsl[base13 + r3n + c1]);
            jterm -= (det0[0] * det0[9] - det0[1] * det0[8])
                * (jsl[base12 + r3n + c3] - jsl[base13 + r3n + c2]);

            // (\eta,\xi) = (2,3).
            jterm += (det0[2] * det0[7] - det0[3] * det0[6])
                * (jsl[base20 + r3n + c1] - jsl[base21 + r3n + c0]);
            jterm -= (det0[1] * det0[7] - det0[3] * det0[5])
                * (jsl[base20 + r3n + c2] - jsl[base22 + r3n + c0]);
            jterm += (det0[1] * det0[6] - det0[2] * det0[5])
                * (jsl[base20 + r3n + c3] - jsl[base23 + r3n + c0]);
            jterm += (det0[0] * det0[7] - det0[3] * det0[4])
                * (jsl[base21 + r3n + c2] - jsl[base22 + r3n + c1]);
            jterm -= (det0[0] * det0[6] - det0[2] * det0[4])
                * (jsl[base21 + r3n + c3] - jsl[base23 + r3n + c1]);
            jterm += (det0[0] * det0[5] - det0[1] * det0[4])
                * (jsl[base22 + r3n + c3] - jsl[base23 + r3n + c2]);

            let jterm = <T as From<f64>>::from(2.0) * jterm;

            // H = {}^{xw}\tilde S[{}^xV_0^{(0,0)}\det\mathbf D_{\mathrm{ov}}
            // - 2\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal V_z} + C_3].
            w.phase
                * <T as From<f64>>::from(w.tilde_s_prod)
                * (w.v0[0] * det - <T as From<f64>>::from(2.0) * vterm + jterm)
        } else {
            <T as From<f64>>::from(0.0)
        }
    })
}

/// Evaluate the same-spin two-body matrix element for arbitrary L when m = 0.
/// The scalar term uses \det\mathbf D_{\mathrm{ov}}, each \mathcal V term replaces one column,
/// and each \mathcal J term is evaluated from a minor with one further column replacement.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Prepared contraction determinant and scratch storage for cofactors and minors.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: Same-spin two-body matrix element for arbitrary L and m = 0.
#[inline(always)]
fn xw_h2_same_m0_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_same_m0_gen, {
        // Determine L = L_x + L_w and select \mathbf D_{\mathrm{ov}}(0,\ldots,0).
        let l = l_ex.holes.len() + g_ex.holes.len();
        let mut acc = <T as From<f64>>::from(0.0);
        let n = w.n();
        let det0 = &scratch.det0.as_slice()[..l * l];

        // Evaluate \det\mathbf D_{\mathrm{ov}} and its cofactor matrix.
        if let Some(det_det) = adjugate_transpose(
            scratch.adjt_det.as_mut_slice(),
            scratch.invs.as_mut_slice(),
            scratch.lu.as_mut_slice(),
            det0,
            l,
            tol,
        ) {
            // C_1 = {}^xV_0^{(0,0)}\det\mathbf D_{\mathrm{ov}}.
            let mut contrib = w.v0[0] * det_det;
            let vsl = w.v_t_slice(0, 0, 0);

            // C_2 = -2\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal V_z}.
            // `det_det + corr` is the determinant after replacing column z.
            for k in 0..l {
                let ck = scratch.cols[k];
                let base = ck * n;
                let corr =
                    column_replacement_correction(l, det0, scratch.adjt_det.as_slice(), k, |r| {
                        vsl[base + scratch.rows[r]]
                    });
                contrib -= <T as From<f64>>::from(2.0) * (det_det + corr);
            }

            // C_3 = \sum_{z<y}\sum_{\eta<\xi}\phi_{\eta\xi}^{zy}
            // \mathcal J_{\eta z,\xi y}\det\mathbf D_{\mathrm{ov}}[\eta,\xi|z,y].
            let jsl = w.j_slice(0);
            let layout = ReplacementLayout {
                n,
                rows: scratch.rows.as_slice(),
                cols: scratch.cols.as_slice(),
            };

            for i in 0..l {
                for j in 0..l {
                    let phase = if ((i + j) & 1) == 0 {
                        <T as From<f64>>::from(1.0)
                    } else {
                        <T as From<f64>>::from(-1.0)
                    };
                    let ri_fixed = scratch.rows[i];
                    let cj_fixed = scratch.cols[j];

                    // Remove row \eta = i and column z = j, then replace each remaining
                    // column y by \mathcal J_{\eta z,\xi y}; the outer phase is (-1)^{\eta+z}.
                    minor_adjt(
                        det0,
                        Minor { l, row: i, col: j },
                        &mut scratch.det_mix2,
                        &mut scratch.adjt_det2,
                        tol,
                        |lm1, _det_minor, cof_minor, _det_det2| {
                            for k2 in 0..lm1 {
                                let det_repl = column_replacement_det(lm1, cof_minor, k2, |r| {
                                    j_replacement(
                                        jsl,
                                        layout,
                                        DetIndex { row: i, col: j },
                                        DetIndex { row: r, col: k2 },
                                        DetIndex {
                                            row: ri_fixed,
                                            col: cj_fixed,
                                        },
                                        false,
                                    )
                                });
                                contrib += phase * det_repl;
                            }
                        },
                    );
                }
            }

            acc += contrib;
        }

        // Apply the orbital-pairing phase to the singular-value product to recover
        // {}^{xw}\tilde S(C_1 + C_2 + C_3).
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
    })
}

/// Evaluate the same-spin two-body matrix element when m > 0 by summing every allowed distribution:
/// m_1 + \cdots + m_{L+2} = m, \qquad m_i \in \{0,1\}.
/// The first two assignments select the two operator contractions in V_0, \mathcal V and \mathcal J,
/// while the remaining assignments select the columns of \mathbf D_{\mathrm{ov}}.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for mixed contraction determinants, cofactors, minors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: Same-spin two-body matrix element summed over all allowed distributions.
#[inline(always)]
fn xw_h2_same_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_same_gen, {
        // Determine L = L_x + L_w.
        let l = l_ex.holes.len() + g_ex.holes.len();
        let mut acc = <T as From<f64>>::from(0.0);
        let n = w.n();

        // Enumerate all distributions over the two operator contractions and L determinant
        // columns, construct the corresponding mixed \mathbf D_{\mathrm{ov}}, and evaluate its cofactors.
        get_det_adjt_same(w, l, 2, scratch, tol, |bits, scratch, det_det| {
            // The first two bits select m_1 and m_2; bit k + 2 selects the assignment of column k.
            let m1 = bit(bits, 0);
            let m2 = bit(bits, 1);

            // C_1 = {}^xV_0^{(m_1,m_2)}\det\mathbf D_{\mathrm{ov}}. The scalar storage is indexed
            // by m_1 + m_2, with `v0[1]` containing the combined (0,1) and (1,0) assignments.
            let mut contrib = w.v0[m1 + m2] * det_det;

            let v0 = w.v_t_slice(m1, m2, 0);
            let v1 = w.v_t_slice(m1, m2, 1);

            // C_2 = -2\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal V_z}.
            // Select \mathcal V^{(m_1,m_2,m_z)} for the assignment of each replaced column.
            for k in 0..l {
                let mk = bit(bits, k + 2);
                let ck = scratch.cols[k];
                let vsl = if mk == 0 { v0 } else { v1 };
                let base = ck * n;

                let corr = column_replacement_correction(
                    l,
                    scratch.det_mix.as_slice(),
                    scratch.adjt_det.as_slice(),
                    k,
                    |r| vsl[base + scratch.rows[r]],
                );
                contrib -= <T as From<f64>>::from(2.0) * (det_det + corr);
            }

            // C_3 = \sum_{z<y}\sum_{\eta<\xi}\phi_{\eta\xi}^{zy}
            // \mathcal J_{\eta z,\xi y}\det\mathbf D_{\mathrm{ov}}[\eta,\xi|z,y].
            let layout = ReplacementLayout {
                n,
                rows: scratch.rows.as_slice(),
                cols: scratch.cols.as_slice(),
            };

            for i in 0..l {
                for j in 0..l {
                    let phase = if ((i + j) & 1) == 0 {
                        <T as From<f64>>::from(1.0)
                    } else {
                        <T as From<f64>>::from(-1.0)
                    };
                    let ri_fixed = scratch.rows[i];
                    let cj_fixed = scratch.cols[j];
                    let mj = bit(bits, j + 2);

                    // Remove row \eta = i and column z = j. For each remaining column y,
                    // select the symmetry-unique \mathcal J branch for its four m_i assignments.
                    minor_adjt(
                        scratch.det_mix.as_slice(),
                        Minor { l, row: i, col: j },
                        &mut scratch.det_mix2,
                        &mut scratch.adjt_det2,
                        tol,
                        |lm1, _det_minor, cof_minor, _det_det2| {
                            for k2 in 0..lm1 {
                                let k_full = if k2 < j { k2 } else { k2 + 1 };
                                let mk = bit(bits, k_full + 2);
                                // Pair-exchange symmetry maps the requested \mathcal J distribution to
                                // its stored branch; `swap` records whether the excitation pairs are interchanged.
                                let (slot, swap) = jslot(m1, m2, mk, mj);

                                let jsl = w.j_slice(slot);

                                let det_repl = column_replacement_det(lm1, cof_minor, k2, |r| {
                                    j_replacement(
                                        jsl,
                                        layout,
                                        DetIndex { row: i, col: j },
                                        DetIndex { row: r, col: k2 },
                                        DetIndex {
                                            row: ri_fixed,
                                            col: cj_fixed,
                                        },
                                        swap,
                                    )
                                });

                                contrib += phase * det_repl;
                            }
                        },
                    );
                }
            }
            acc += contrib;
        });
        // Apply the orbital-pairing phase to the singular-value product and multiply the
        // constrained sum \sum_{\{m_i\}}(C_1 + C_2 + C_3).
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
    })
}
