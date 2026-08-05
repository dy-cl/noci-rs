// nonorthogonalwicks/eval/h2diff.rs
use super::super::scratch::WickScratch;
use super::super::view::WicksPairView;
use super::helpers::{DetBranches, DetIndex, ReplacementLayout};
use crate::Excitation;
use crate::noci::NOCIScalar;
use crate::time_call;

use super::super::layout::{idx, idx4};
use super::helpers::{bit, column_replacement_det, get_det_adjt_diff, ii_replacement};
use crate::maths::adjugate_transpose;

/// Evaluate the different-spin two-body matrix element between excited determinants generated from
/// the reference pair \langle{}^x\Psi| and |{}^w\Psi\rangle. The alpha- and beta-spin contraction determinants factorise:
/// \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat v_{\alpha\beta}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
/// = {}^{xw}\tilde S_\alpha{}^{xw}\tilde S_\beta
/// \sum_{\substack{m_{\alpha0}+\sum_zm_{\alpha z}=m_\alpha\\m_{\beta0}+\sum_ym_{\beta y}=m_\beta}}
/// [{}^xV_{\alpha\beta,0}^{(m_{\alpha0},m_{\beta0})}\det\mathbf D_{\alpha,\mathrm{ov}}\det\mathbf D_{\beta,\mathrm{ov}}
/// - \sum_z\det\mathbf D_{\alpha,\mathrm{ov}}^{z\rightarrow\mathcal V^\alpha_z}\det\mathbf D_{\beta,\mathrm{ov}}
/// - \sum_y\det\mathbf D_{\alpha,\mathrm{ov}}\det\mathbf D_{\beta,\mathrm{ov}}^{y\rightarrow\mathcal V^\beta_y}
/// + \sum_{z,y,\eta,\xi}\operatorname{cof}[\mathbf D_{\alpha,\mathrm{ov}}]_{\eta z}
///   \mathcal{II}_{\eta z,\xi y}^{(m_{\alpha0},m_{\alpha z},m_{\beta0},m_{\beta y})}
///   \operatorname{cof}[\mathbf D_{\beta,\mathrm{ov}}]_{\xi y}].
///   Each m_{\sigma0} and m_{\sigma z} is zero or one. The implementation applies the orbital-pairing
///   phases separately from the reduced overlap products. No exchange term occurs between the spin spaces.
/// # Arguments:
/// - `w`: Same-spin and different-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.
/// - `g_ex`: Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.
/// - `diff`: Scratch storage for mixed contraction determinants, cofactors and work buffers.
/// - `a`: Prepared alpha-spin contraction determinants and their row and column labels.
/// - `b`: Prepared beta-spin contraction determinants and their row and column labels.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: Different-spin two-body matrix element.
#[inline(always)]
pub(crate) fn xw_h2_diff<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    l_ex: &Excitation,
    g_ex: &Excitation,
    diff: &mut WickScratch<T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_diff, {
        // For m_\alpha = m_\beta = 0, only the all-m_i = 0 contraction determinants contribute.
        // Otherwise, sum the independent distributions satisfying \sum_i m_{\alpha i} = m_\alpha
        // and \sum_i m_{\beta i} = m_\beta.
        if w.aa.m == 0 && w.bb.m == 0 {
            xw_h2_diff_m0(w, l_ex, g_ex, diff, a, b, tol)
        } else {
            xw_h2_diff_gen(w, l_ex, g_ex, diff, a, b, tol)
        }
    })
}

/// Evaluate the different-spin two-body matrix element when m_\alpha = m_\beta = 0. Both spin spaces
/// use only \mathbf D_{\sigma,\mathrm{ov}}(0,\ldots,0) and the m_i = 0 intermediates. Fixed-rank kernels
/// are used for (L_\alpha,L_\beta) = (1,1), (1,2), (1,3), (2,1), (2,2) and (3,1); all other
/// excitation ranks use the general cofactor form.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs in either spin space.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `diff`: Scratch storage for cofactors and work buffers.
/// - `a`: Prepared alpha-spin contraction determinant.
/// - `b`: Prepared beta-spin contraction determinant.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: Different-spin two-body matrix element for m_\alpha = m_\beta = 0.
#[inline(always)]
fn xw_h2_diff_m0<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    l_ex: &Excitation,
    g_ex: &Excitation,
    diff: &mut WickScratch<T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_diff_m0, {
        // Determine L_\alpha = L_{x,\alpha} + L_{w,\alpha} and L_\beta = L_{x,\beta} + L_{w,\beta}.
        let l_ex_a = &l_ex.alpha;
        let g_ex_a = &g_ex.alpha;
        let l_ex_b = &l_ex.beta;
        let g_ex_b = &g_ex.beta;

        let la = l_ex_a.holes.len() + g_ex_a.holes.len();
        let lb = l_ex_b.holes.len() + g_ex_b.holes.len();

        // Dispatch to direct fixed-rank forms of C_0 + C_\alpha + C_\beta + C_{\alpha\beta}.
        match (la, lb) {
            (0, 0) => {
                // H_{\alpha\beta} = \phi_\alpha{}^{xw}\tilde S_\alpha
                // \phi_\beta{}^{xw}\tilde S_\beta V_{\alpha\beta,0}^{(0,0)}.
                (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
                    * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod))
                    * w.ab.vab0[0][0]
            }
            (1, 1) => xw_h2_diff_m0_11(w, a, b),
            (1, 2) => xw_h2_diff_m0_12(w, a, b),
            (1, 3) => xw_h2_diff_m0_13(w, diff, a, b, tol),
            (2, 1) => xw_h2_diff_m0_21(w, a, b),
            (2, 2) => xw_h2_diff_m0_22(w, a, b),
            (3, 1) => xw_h2_diff_m0_31(w, diff, a, b, tol),
            _ => xw_h2_diff_m0_gen(w, l_ex, g_ex, diff, a, b, tol),
        }
    })
}

/// Evaluate the fixed-rank (L_\alpha,L_\beta) = (1,1) matrix element for m_\alpha = m_\beta = 0.
/// All four terms of the different-spin expansion reduce to individual determinant entries.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `a`: Prepared rank-one alpha-spin contraction determinant.
/// - `b`: Prepared rank-one beta-spin contraction determinant.
/// # Returns
/// - `T`: Different-spin two-body matrix element for (L_\alpha,L_\beta) = (1,1).
#[inline(always)]
fn xw_h2_diff_m0_11<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_diff_m0_11, {
        // Read \det\mathbf D_{\alpha,\mathrm{ov}}, \det\mathbf D_{\beta,\mathrm{ov}} and the m_i = 0 intermediates.
        let n = w.ab.n();

        let ra = a.rows[0];
        let ca = a.cols[0];
        let rb = b.rows[0];
        let cb = b.cols[0];

        let deta = a.det0.as_slice()[0];
        let detb = b.det0.as_slice()[0];

        let vab = w.ab.vab_t_slice(0, 0, 0);
        let vba = w.ab.vba_t_slice(0, 0, 0);
        let iisl = w.ab.iiab_slice(0, 0, 0, 0);

        // C_0 = V_{\alpha\beta,0}\det\mathbf D_{\alpha,\mathrm{ov}}\det\mathbf D_{\beta,\mathrm{ov}}.
        // C_\alpha = -\det\mathbf D_{\alpha,\mathrm{ov}}^{0\rightarrow\mathcal V^\alpha_0}\det\mathbf D_{\beta,\mathrm{ov}},
        // C_\beta = -\det\mathbf D_{\alpha,\mathrm{ov}}\det\mathbf D_{\beta,\mathrm{ov}}^{0\rightarrow\mathcal V^\beta_0},
        // C_{\alpha\beta} = \operatorname{cof}[\mathbf D_\alpha]_{00}\mathcal{II}_{00,00}
        // \operatorname{cof}[\mathbf D_\beta]_{00} = \mathcal{II}_{00,00}.
        let term =
            w.ab.vab0[0][0] * deta * detb - vab[ca * n + ra] * detb - vba[cb * n + rb] * deta
                + iisl[idx4(n, ra, ca, rb, cb)];

        // Multiply C_0 + C_\alpha + C_\beta + C_{\alpha\beta} by
        // \phi_\alpha{}^{xw}\tilde S_\alpha\phi_\beta{}^{xw}\tilde S_\beta.
        (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
            * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod))
            * term
    })
}

/// Evaluate the fixed-rank (L_\alpha,L_\beta) = (1,2) matrix element for m_\alpha = m_\beta = 0.
/// The alpha-spin determinant is scalar, while the beta-spin \mathcal V^\beta and \mathcal{II}
/// contributions are evaluated using the explicit cofactors of its rank-two contraction determinant.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `a`: Prepared rank-one alpha-spin contraction determinant.
/// - `b`: Prepared rank-two beta-spin contraction determinant.
/// # Returns
/// - `T`: Different-spin two-body matrix element for (L_\alpha,L_\beta) = (1,2).
#[inline(always)]
fn xw_h2_diff_m0_12<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_diff_m0_12, {
        // Read the rank-one \mathbf D_{\alpha,\mathrm{ov}} and rank-two \mathbf D_{\beta,\mathrm{ov}}.
        let n = w.ab.n();

        let ra = a.rows[0];
        let ca = a.cols[0];
        let deta = a.det0.as_slice()[0];

        let rows_b = &b.rows[..2];
        let cols_b = &b.cols[..2];
        let db = b.det0.as_slice();

        let b00 = db[0];
        let b01 = db[1];
        let b10 = db[2];
        let b11 = db[3];
        let detb = b00 * b11 - b01 * b10;

        let r0b = rows_b[0];
        let r1b = rows_b[1];
        let c0b = cols_b[0];
        let c1b = cols_b[1];

        let vab = w.ab.vab_t_slice(0, 0, 0);
        let vba = w.ab.vba_t_slice(0, 0, 0);
        let iisl = w.ab.iiab_slice(0, 0, 0, 0);

        // \det\mathbf D_{\alpha,\mathrm{ov}}^{0\rightarrow\mathcal V^\alpha_0}
        // = \mathcal V^\alpha_{r_\alpha c_\alpha}.
        let det_a = vab[ca * n + ra];

        // Form \det\mathbf D_{\beta,\mathrm{ov}}^{z\rightarrow\mathcal V^\beta_z} for z = 0,1.
        // C_\beta = -\det\mathbf D_{\alpha,\mathrm{ov}}
        // \sum_y\det\mathbf D_{\beta,\mathrm{ov}}^{y\rightarrow\mathcal V^\beta_y}.
        let bu0 = vba[c0b * n + r0b];
        let bu1 = vba[c0b * n + r1b];
        let bv0 = vba[c1b * n + r0b];
        let bv1 = vba[c1b * n + r1b];
        let detb_c0 = bu0 * b11 - b01 * bu1;
        let detb_c1 = b00 * bv1 - bv0 * b10;

        // C_{\alpha\beta} = \sum_{y,\xi}\mathcal{II}_{r_\alpha c_\alpha,\xi y}
        // \operatorname{cof}[\mathbf D_{\beta,\mathrm{ov}}]_{\xi y}.
        let n2 = n * n;
        let abase = (ra * n + ca) * n2;
        let b00_idx = r0b * n + c0b;
        let b01_idx = r0b * n + c1b;
        let b10_idx = r1b * n + c0b;
        let b11_idx = r1b * n + c1b;
        let ii =
            b11 * iisl[abase + b00_idx] - b10 * iisl[abase + b01_idx] - b01 * iisl[abase + b10_idx]
                + b00 * iisl[abase + b11_idx];

        // C_0 = V_{\alpha\beta,0}\det\mathbf D_\alpha\det\mathbf D_\beta,
        // C_\alpha = -\det\mathbf D_\alpha^{0\rightarrow\mathcal V^\alpha_0}\det\mathbf D_\beta,
        // C_\beta = -\det\mathbf D_\alpha\sum_y\det\mathbf D_\beta^{y\rightarrow\mathcal V^\beta_y}.
        let contrib =
            w.ab.vab0[0][0] * deta * detb - det_a * detb - deta * (detb_c0 + detb_c1) + ii;

        // H_{\alpha\beta} = \phi_\alpha{}^{xw}\tilde S_\alpha
        // \phi_\beta{}^{xw}\tilde S_\beta(C_0 + C_\alpha + C_\beta + C_{\alpha\beta}).
        (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
            * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod))
            * contrib
    })
}

/// Evaluate the fixed-rank (L_\alpha,L_\beta) = (1,3) matrix element for m_\alpha = m_\beta = 0.
/// The beta-spin \mathcal V^\beta and \mathcal{II} terms are contracted with the cofactor matrix of
/// \mathbf D_{\beta,\mathrm{ov}}.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `diff`: Scratch storage for the beta-spin adjugate-transpose and factorisation work arrays.
/// - `a`: Prepared rank-one alpha-spin contraction determinant.
/// - `b`: Prepared rank-three beta-spin contraction determinant.
/// - `tol`: Numerical tolerance used when evaluating the beta-spin determinant and cofactors.
/// # Returns
/// - `T`: Different-spin two-body matrix element for (L_\alpha,L_\beta) = (1,3).
#[inline(always)]
fn xw_h2_diff_m0_13<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    diff: &mut WickScratch<T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_diff_m0_13, {
        // Store \det\mathbf D_{\beta,\mathrm{ov}} and \operatorname{cof}[\mathbf D_{\beta,\mathrm{ov}}]_{\xi y}.
        diff.ensure_diff(1, 3);

        let n = w.ab.n();
        let ra = a.rows[0];
        let ca = a.cols[0];
        let deta = a.det0.as_slice()[0];

        let rows_b = &b.rows[..3];
        let cols_b = &b.cols[..3];
        let detb0 = &b.det0.as_slice()[..9];

        // Evaluate \det\mathbf D_{\beta,\mathrm{ov}} and its cofactor matrix.
        if let Some(detb) = adjugate_transpose(
            diff.adjt_detb.as_mut_slice(),
            diff.invslb.as_mut_slice(),
            diff.lub.as_mut_slice(),
            detb0,
            3,
            tol,
        ) {
            let cofb = diff.adjt_detb.as_slice();

            let r0 = rows_b[0];
            let r1 = rows_b[1];
            let r2 = rows_b[2];
            let c0 = cols_b[0];
            let c1 = cols_b[1];
            let c2 = cols_b[2];

            // C_\alpha = -\sum_z\det\mathbf D_{\alpha,\mathrm{ov}}^{z\rightarrow\mathcal V^\alpha_z}
            // \det\mathbf D_{\beta,\mathrm{ov}}.
            let vab = w.ab.vab_t_slice(0, 0, 0);
            // C_\beta = -\det\mathbf D_{\alpha,\mathrm{ov}}
            // \sum_y\det\mathbf D_{\beta,\mathrm{ov}}^{y\rightarrow\mathcal V^\beta_y}.
            let vba = w.ab.vba_t_slice(0, 0, 0);
            // C_{\alpha\beta} = \sum_{z,y,\eta,\xi}\operatorname{cof}[\mathbf D_\alpha]_{\eta z}
            // \mathcal{II}_{\eta z,\xi y}\operatorname{cof}[\mathbf D_\beta]_{\xi y}.
            let iisl = w.ab.iiab_slice(0, 0, 0, 0);

            // C_\beta = -\det\mathbf D_{\alpha,\mathrm{ov}}
            // \sum_{y,\xi}\mathcal V^\beta_{\xi y}\operatorname{cof}[\mathbf D_{\beta,\mathrm{ov}}]_{\xi y}.
            let vba_term = cofb[idx(3, 0, 0)] * vba[c0 * n + r0]
                + cofb[idx(3, 1, 0)] * vba[c0 * n + r1]
                + cofb[idx(3, 2, 0)] * vba[c0 * n + r2]
                + cofb[idx(3, 0, 1)] * vba[c1 * n + r0]
                + cofb[idx(3, 1, 1)] * vba[c1 * n + r1]
                + cofb[idx(3, 2, 1)] * vba[c1 * n + r2]
                + cofb[idx(3, 0, 2)] * vba[c2 * n + r0]
                + cofb[idx(3, 1, 2)] * vba[c2 * n + r1]
                + cofb[idx(3, 2, 2)] * vba[c2 * n + r2];

            // C_{\alpha\beta} = \sum_{y,\xi}\mathcal{II}_{r_\alpha c_\alpha,\xi y}
            // \operatorname{cof}[\mathbf D_{\beta,\mathrm{ov}}]_{\xi y}.
            let ii_term = cofb[idx(3, 0, 0)] * iisl[idx4(n, ra, ca, r0, c0)]
                + cofb[idx(3, 1, 0)] * iisl[idx4(n, ra, ca, r1, c0)]
                + cofb[idx(3, 2, 0)] * iisl[idx4(n, ra, ca, r2, c0)]
                + cofb[idx(3, 0, 1)] * iisl[idx4(n, ra, ca, r0, c1)]
                + cofb[idx(3, 1, 1)] * iisl[idx4(n, ra, ca, r1, c1)]
                + cofb[idx(3, 2, 1)] * iisl[idx4(n, ra, ca, r2, c1)]
                + cofb[idx(3, 0, 2)] * iisl[idx4(n, ra, ca, r0, c2)]
                + cofb[idx(3, 1, 2)] * iisl[idx4(n, ra, ca, r1, c2)]
                + cofb[idx(3, 2, 2)] * iisl[idx4(n, ra, ca, r2, c2)];

            // C_0 + C_\alpha + C_\beta + C_{\alpha\beta}.
            let contrib =
                w.ab.vab0[0][0] * deta * detb - vab[ca * n + ra] * detb - deta * vba_term + ii_term;
            // H_{\alpha\beta} = \phi_\alpha{}^{xw}\tilde S_\alpha
            // \phi_\beta{}^{xw}\tilde S_\beta(C_0 + C_\alpha + C_\beta + C_{\alpha\beta}).
            (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
                * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod))
                * contrib
        } else {
            <T as From<f64>>::from(0.0)
        }
    })
}

/// Evaluate the fixed-rank (L_\alpha,L_\beta) = (2,1) matrix element for m_\alpha = m_\beta = 0.
/// The beta-spin determinant is scalar, while the alpha-spin \mathcal V^\alpha and \mathcal{II}
/// contributions are evaluated using the explicit cofactors of its rank-two contraction determinant.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `a`: Prepared rank-two alpha-spin contraction determinant.
/// - `b`: Prepared rank-one beta-spin contraction determinant.
/// # Returns
/// - `T`: Different-spin two-body matrix element for (L_\alpha,L_\beta) = (2,1).
#[inline(always)]
fn xw_h2_diff_m0_21<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_diff_m0_21, {
        // Read the rank-two \mathbf D_{\alpha,\mathrm{ov}} and rank-one \mathbf D_{\beta,\mathrm{ov}}.
        let n = w.ab.n();

        let rows_a = &a.rows[..2];
        let cols_a = &a.cols[..2];
        let da = a.det0.as_slice();

        let a00 = da[0];
        let a01 = da[1];
        let a10 = da[2];
        let a11 = da[3];
        let deta = a00 * a11 - a01 * a10;

        let rb = b.rows[0];
        let cb = b.cols[0];
        let detb = b.det0.as_slice()[0];

        let r0a = rows_a[0];
        let r1a = rows_a[1];
        let c0a = cols_a[0];
        let c1a = cols_a[1];

        let vab = w.ab.vab_t_slice(0, 0, 0);
        let vba = w.ab.vba_t_slice(0, 0, 0);
        let iisl = w.ab.iiab_slice(0, 0, 0, 0);

        // Form \det\mathbf D_{\alpha,\mathrm{ov}}^{z\rightarrow\mathcal V^\alpha_z} for z = 0,1.
        // C_\alpha = -\sum_z\det\mathbf D_{\alpha,\mathrm{ov}}^{z\rightarrow\mathcal V^\alpha_z}
        // \det\mathbf D_{\beta,\mathrm{ov}}.
        let au0 = vab[c0a * n + r0a];
        let au1 = vab[c0a * n + r1a];
        let av0 = vab[c1a * n + r0a];
        let av1 = vab[c1a * n + r1a];
        let deta_c0 = au0 * a11 - a01 * au1;
        let deta_c1 = a00 * av1 - av0 * a10;

        // \det\mathbf D_{\beta,\mathrm{ov}}^{0\rightarrow\mathcal V^\beta_0}
        // = \mathcal V^\beta_{r_\beta c_\beta}.
        let det_b = vba[cb * n + rb];

        // C_{\alpha\beta} = \sum_{z,\eta}\operatorname{cof}[\mathbf D_{\alpha,\mathrm{ov}}]_{\eta z}
        // \mathcal{II}_{\eta z,r_\beta c_\beta}.
        let n2 = n * n;
        let bidx = rb * n + cb;
        let a00_base = (r0a * n + c0a) * n2;
        let a01_base = (r0a * n + c1a) * n2;
        let a10_base = (r1a * n + c0a) * n2;
        let a11_base = (r1a * n + c1a) * n2;
        let ii =
            a11 * iisl[a00_base + bidx] - a10 * iisl[a01_base + bidx] - a01 * iisl[a10_base + bidx]
                + a00 * iisl[a11_base + bidx];

        // C_0 = V_{\alpha\beta,0}\det\mathbf D_\alpha\det\mathbf D_\beta,
        // C_\alpha = -\sum_z\det\mathbf D_\alpha^{z\rightarrow\mathcal V^\alpha_z}\det\mathbf D_\beta,
        // C_\beta = -\det\mathbf D_\alpha\det\mathbf D_\beta^{0\rightarrow\mathcal V^\beta_0}.
        let contrib =
            w.ab.vab0[0][0] * deta * detb - (deta_c0 + deta_c1) * detb - det_b * deta + ii;

        // H_{\alpha\beta} = \phi_\alpha{}^{xw}\tilde S_\alpha
        // \phi_\beta{}^{xw}\tilde S_\beta(C_0 + C_\alpha + C_\beta + C_{\alpha\beta}).
        (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
            * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod))
            * contrib
    })
}

/// Evaluate the fixed-rank (L_\alpha,L_\beta) = (2,2) matrix element for m_\alpha = m_\beta = 0.
/// The one-column terms use the explicit rank-two cofactors, while the \mathcal{II} term contracts
/// the cofactor matrices from both spin spaces.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `a`: Prepared rank-two alpha-spin contraction determinant.
/// - `b`: Prepared rank-two beta-spin contraction determinant.
/// # Returns
/// - `T`: Different-spin two-body matrix element for (L_\alpha,L_\beta) = (2,2).
#[inline(always)]
fn xw_h2_diff_m0_22<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_diff_m0_22, {
        // Read the two rank-two contraction determinants and their row and column labels.
        let n = w.ab.n();

        let rows_a = &a.rows[..2];
        let cols_a = &a.cols[..2];
        let rows_b = &b.rows[..2];
        let cols_b = &b.cols[..2];

        // Evaluate \det\mathbf D_{\alpha,\mathrm{ov}} and \det\mathbf D_{\beta,\mathrm{ov}} explicitly.
        let da = a.det0.as_slice();
        let db = b.det0.as_slice();

        let a00 = da[0];
        let a01 = da[1];
        let a10 = da[2];
        let a11 = da[3];
        let deta = a00 * a11 - a01 * a10;

        let b00 = db[0];
        let b01 = db[1];
        let b10 = db[2];
        let b11 = db[3];
        let detb = b00 * b11 - b01 * b10;

        let r0a = rows_a[0];
        let r1a = rows_a[1];
        let c0a = cols_a[0];
        let c1a = cols_a[1];

        let r0b = rows_b[0];
        let r1b = rows_b[1];
        let c0b = cols_b[0];
        let c1b = cols_b[1];

        let vab = w.ab.vab_t_slice(0, 0, 0);
        let vba = w.ab.vba_t_slice(0, 0, 0);
        let iisl = w.ab.iiab_slice(0, 0, 0, 0);

        // C_\alpha = -\sum_z\det\mathbf D_{\alpha,\mathrm{ov}}^{z\rightarrow\mathcal V^\alpha_z}
        // \det\mathbf D_{\beta,\mathrm{ov}}.
        let au0 = vab[c0a * n + r0a];
        let au1 = vab[c0a * n + r1a];
        let av0 = vab[c1a * n + r0a];
        let av1 = vab[c1a * n + r1a];

        let deta_c0 = au0 * a11 - a01 * au1;
        let deta_c1 = a00 * av1 - av0 * a10;

        // C_\beta = -\det\mathbf D_{\alpha,\mathrm{ov}}
        // \sum_y\det\mathbf D_{\beta,\mathrm{ov}}^{y\rightarrow\mathcal V^\beta_y}.
        let bu0 = vba[c0b * n + r0b];
        let bu1 = vba[c0b * n + r1b];
        let bv0 = vba[c1b * n + r0b];
        let bv1 = vba[c1b * n + r1b];

        let detb_c0 = bu0 * b11 - b01 * bu1;
        let detb_c1 = b00 * bv1 - bv0 * b10;

        // Add C_0 = V_{\alpha\beta,0}\det\mathbf D_\alpha\det\mathbf D_\beta and the two
        // one-column contributions C_\alpha and C_\beta.
        let mut contrib =
            w.ab.vab0[0][0] * deta * detb - (deta_c0 + deta_c1) * detb - (detb_c0 + detb_c1) * deta;

        // C_{\alpha\beta} = \sum_{z,y,\eta,\xi}\operatorname{cof}[\mathbf D_\alpha]_{\eta z}
        // \mathcal{II}_{\eta z,\xi y}\operatorname{cof}[\mathbf D_\beta]_{\xi y}.
        // First contract \mathcal{II} with the beta-spin cofactors for each alpha index pair.
        let n2 = n * n;
        let a00_base = (r0a * n + c0a) * n2;
        let a01_base = (r0a * n + c1a) * n2;
        let a10_base = (r1a * n + c0a) * n2;
        let a11_base = (r1a * n + c1a) * n2;

        let b00_idx = r0b * n + c0b;
        let b01_idx = r0b * n + c1b;
        let b10_idx = r1b * n + c0b;
        let b11_idx = r1b * n + c1b;

        let ii00 = b11 * iisl[a00_base + b00_idx]
            - b10 * iisl[a00_base + b01_idx]
            - b01 * iisl[a00_base + b10_idx]
            + b00 * iisl[a00_base + b11_idx];
        let ii01 = b11 * iisl[a01_base + b00_idx]
            - b10 * iisl[a01_base + b01_idx]
            - b01 * iisl[a01_base + b10_idx]
            + b00 * iisl[a01_base + b11_idx];
        let ii10 = b11 * iisl[a10_base + b00_idx]
            - b10 * iisl[a10_base + b01_idx]
            - b01 * iisl[a10_base + b10_idx]
            + b00 * iisl[a10_base + b11_idx];
        let ii11 = b11 * iisl[a11_base + b00_idx]
            - b10 * iisl[a11_base + b01_idx]
            - b01 * iisl[a11_base + b10_idx]
            + b00 * iisl[a11_base + b11_idx];

        // Complete C_{\alpha\beta} by contracting the resulting quantities with the alpha-spin cofactors.
        contrib += a11 * ii00 - a10 * ii01 - a01 * ii10 + a00 * ii11;

        // H_{\alpha\beta} = \phi_\alpha{}^{xw}\tilde S_\alpha
        // \phi_\beta{}^{xw}\tilde S_\beta(C_0 + C_\alpha + C_\beta + C_{\alpha\beta}).
        (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
            * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod))
            * contrib
    })
}

/// Evaluate the fixed-rank (L_\alpha,L_\beta) = (3,1) matrix element for m_\alpha = m_\beta = 0.
/// The alpha-spin \mathcal V^\alpha and \mathcal{II} terms are contracted with the cofactor matrix of
/// \mathbf D_{\alpha,\mathrm{ov}}.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `diff`: Scratch storage for the alpha-spin adjugate-transpose and factorisation work arrays.
/// - `a`: Prepared rank-three alpha-spin contraction determinant.
/// - `b`: Prepared rank-one beta-spin contraction determinant.
/// - `tol`: Numerical tolerance used when evaluating the alpha-spin determinant and cofactors.
/// # Returns
/// - `T`: Different-spin two-body matrix element for (L_\alpha,L_\beta) = (3,1).
#[inline(always)]
fn xw_h2_diff_m0_31<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    diff: &mut WickScratch<T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_diff_m0_31, {
        // Store \det\mathbf D_{\alpha,\mathrm{ov}} and \operatorname{cof}[\mathbf D_{\alpha,\mathrm{ov}}]_{\eta z}.
        diff.ensure_diff(3, 1);

        let n = w.ab.n();
        let rb = b.rows[0];
        let cb = b.cols[0];
        let detb = b.det0.as_slice()[0];

        let rows_a = &a.rows[..3];
        let cols_a = &a.cols[..3];
        let deta0 = &a.det0.as_slice()[..9];

        // Evaluate \det\mathbf D_{\alpha,\mathrm{ov}} and its cofactor matrix.
        if let Some(deta) = adjugate_transpose(
            diff.adjt_deta.as_mut_slice(),
            diff.invsla.as_mut_slice(),
            diff.lua.as_mut_slice(),
            deta0,
            3,
            tol,
        ) {
            let cofa = diff.adjt_deta.as_slice();

            let r0 = rows_a[0];
            let r1 = rows_a[1];
            let r2 = rows_a[2];
            let c0 = cols_a[0];
            let c1 = cols_a[1];
            let c2 = cols_a[2];

            let vab = w.ab.vab_t_slice(0, 0, 0);
            let vba = w.ab.vba_t_slice(0, 0, 0);
            let iisl = w.ab.iiab_slice(0, 0, 0, 0);

            // C_\alpha = -\det\mathbf D_{\beta,\mathrm{ov}}
            // \sum_{z,\eta}\operatorname{cof}[\mathbf D_{\alpha,\mathrm{ov}}]_{\eta z}\mathcal V^\alpha_{\eta z}.
            let vab_term = cofa[idx(3, 0, 0)] * vab[c0 * n + r0]
                + cofa[idx(3, 1, 0)] * vab[c0 * n + r1]
                + cofa[idx(3, 2, 0)] * vab[c0 * n + r2]
                + cofa[idx(3, 0, 1)] * vab[c1 * n + r0]
                + cofa[idx(3, 1, 1)] * vab[c1 * n + r1]
                + cofa[idx(3, 2, 1)] * vab[c1 * n + r2]
                + cofa[idx(3, 0, 2)] * vab[c2 * n + r0]
                + cofa[idx(3, 1, 2)] * vab[c2 * n + r1]
                + cofa[idx(3, 2, 2)] * vab[c2 * n + r2];

            // C_{\alpha\beta} = \sum_{z,\eta}\operatorname{cof}[\mathbf D_{\alpha,\mathrm{ov}}]_{\eta z}
            // \mathcal{II}_{\eta z,r_\beta c_\beta}.
            let ii_term = cofa[idx(3, 0, 0)] * iisl[idx4(n, r0, c0, rb, cb)]
                + cofa[idx(3, 1, 0)] * iisl[idx4(n, r1, c0, rb, cb)]
                + cofa[idx(3, 2, 0)] * iisl[idx4(n, r2, c0, rb, cb)]
                + cofa[idx(3, 0, 1)] * iisl[idx4(n, r0, c1, rb, cb)]
                + cofa[idx(3, 1, 1)] * iisl[idx4(n, r1, c1, rb, cb)]
                + cofa[idx(3, 2, 1)] * iisl[idx4(n, r2, c1, rb, cb)]
                + cofa[idx(3, 0, 2)] * iisl[idx4(n, r0, c2, rb, cb)]
                + cofa[idx(3, 1, 2)] * iisl[idx4(n, r1, c2, rb, cb)]
                + cofa[idx(3, 2, 2)] * iisl[idx4(n, r2, c2, rb, cb)];

            // C_0 + C_\alpha + C_\beta + C_{\alpha\beta}.
            let contrib =
                w.ab.vab0[0][0] * deta * detb - detb * vab_term - vba[cb * n + rb] * deta + ii_term;
            // H_{\alpha\beta} = \phi_\alpha{}^{xw}\tilde S_\alpha
            // \phi_\beta{}^{xw}\tilde S_\beta(C_0 + C_\alpha + C_\beta + C_{\alpha\beta}).
            (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
                * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod))
                * contrib
        } else {
            <T as From<f64>>::from(0.0)
        }
    })
}

/// Evaluate the different-spin two-body matrix element for arbitrary L_\alpha and L_\beta when
/// m_\alpha = m_\beta = 0. The scalar term uses both contraction determinants, each \mathcal V term
/// replaces one column of the corresponding determinant, and the \mathcal{II} term contracts the
/// cofactor matrices from the two spin spaces.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `diff`: Scratch storage for both cofactor matrices and factorisation work arrays.
/// - `a`: Prepared alpha-spin contraction determinant.
/// - `b`: Prepared beta-spin contraction determinant.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: Different-spin two-body matrix element for arbitrary spin-resolved excitation ranks.
#[inline(always)]
fn xw_h2_diff_m0_gen<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    l_ex: &Excitation,
    g_ex: &Excitation,
    diff: &mut WickScratch<T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_diff_m0_gen, {
        // Determine L_\alpha and L_\beta and select \mathbf D_{\alpha,\mathrm{ov}}(0,\ldots,0)
        // and \mathbf D_{\beta,\mathrm{ov}}(0,\ldots,0).
        let l_ex_a = &l_ex.alpha;
        let g_ex_a = &g_ex.alpha;
        let l_ex_b = &l_ex.beta;
        let g_ex_b = &g_ex.beta;

        let la = l_ex_a.holes.len() + g_ex_a.holes.len();
        let lb = l_ex_b.holes.len() + g_ex_b.holes.len();

        diff.ensure_diff(la, lb);

        let rows_a = &a.rows[..la];
        let cols_a = &a.cols[..la];
        let deta0 = &a.det0.as_slice()[..la * la];

        let rows_b = &b.rows[..lb];
        let cols_b = &b.cols[..lb];
        let detb0 = &b.det0.as_slice()[..lb * lb];

        let mut acc = <T as From<f64>>::from(0.0);
        let n = w.ab.n();

        // Evaluate \det\mathbf D_{\alpha,\mathrm{ov}}, \det\mathbf D_{\beta,\mathrm{ov}} and both cofactor matrices.
        if let Some(det_deta) = adjugate_transpose(
            diff.adjt_deta.as_mut_slice(),
            diff.invsla.as_mut_slice(),
            diff.lua.as_mut_slice(),
            deta0,
            la,
            tol,
        ) && let Some(det_detb) = adjugate_transpose(
            diff.adjt_detb.as_mut_slice(),
            diff.invslb.as_mut_slice(),
            diff.lub.as_mut_slice(),
            detb0,
            lb,
            tol,
        ) {
            // C_0 = V_{\alpha\beta,0}\det\mathbf D_{\alpha,\mathrm{ov}}\det\mathbf D_{\beta,\mathrm{ov}}.
            let mut contrib = w.ab.vab0[0][0] * det_deta * det_detb;

            // C_\alpha = -\sum_z\det\mathbf D_{\alpha,\mathrm{ov}}^{z\rightarrow\mathcal V^\alpha_z}
            // \det\mathbf D_{\beta,\mathrm{ov}}.
            let vab = w.ab.vab_t_slice(0, 0, 0);
            for (k, &ck) in cols_a.iter().enumerate().take(la) {
                let base = ck * n;
                let det_repl = column_replacement_det(la, diff.adjt_deta.as_slice(), k, |r| {
                    vab[base + rows_a[r]]
                });
                contrib -= det_repl * det_detb;
            }

            // C_\beta = -\det\mathbf D_{\alpha,\mathrm{ov}}
            // \sum_y\det\mathbf D_{\beta,\mathrm{ov}}^{y\rightarrow\mathcal V^\beta_y}.
            let vba = w.ab.vba_t_slice(0, 0, 0);
            for (k, &ck) in cols_b.iter().enumerate().take(lb) {
                let base = ck * n;
                let det_repl = column_replacement_det(lb, diff.adjt_detb.as_slice(), k, |r| {
                    vba[base + rows_b[r]]
                });
                contrib -= det_repl * det_deta;
            }

            // C_{\alpha\beta} = \sum_{z,y,\eta,\xi}\operatorname{cof}[\mathbf D_\alpha]_{\eta z}
            // \mathcal{II}_{\eta z,\xi y}\operatorname{cof}[\mathbf D_\beta]_{\xi y}.
            let iisl = w.ab.iiab_slice(0, 0, 0, 0);

            let layout_b = ReplacementLayout {
                n,
                rows: rows_b,
                cols: cols_b,
            };

            // Fix an alpha-spin cofactor and form the beta-spin replacement determinant whose
            // replacement column is \mathcal{II}_{\eta z,\xi y}; their product gives each term in C_{\alpha\beta}.
            for (i, &ra) in rows_a.iter().enumerate() {
                for (j, &ca) in cols_a.iter().enumerate() {
                    let cofa = diff.adjt_deta.as_slice()[idx(la, i, j)];

                    for k in 0..lb {
                        let det_repl = column_replacement_det(
                            lb,
                            diff.adjt_detb.as_slice(),
                            k,
                            |r| {
                                ii_replacement(
                                    iisl,
                                    layout_b,
                                    DetIndex { row: r, col: k },
                                    DetIndex { row: ra, col: ca },
                                    true,
                                )
                            },
                        );
                        contrib += cofa * det_repl;
                    }
                }
            }

            acc += contrib;
        }

        // H_{\alpha\beta} = \phi_\alpha{}^{xw}\tilde S_\alpha
        // \phi_\beta{}^{xw}\tilde S_\beta(C_0 + C_\alpha + C_\beta + C_{\alpha\beta}).
        (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
            * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod))
            * acc
    })
}

/// Evaluate the different-spin two-body matrix element when either spin space contains one or more
/// zero-overlap orbital pairs. The alpha- and beta-spin assignments are enumerated independently:
/// m_{\alpha0} + \sum_zm_{\alpha z} = m_\alpha,
/// m_{\beta0} + \sum_ym_{\beta y} = m_\beta.
/// The first assignment in each spin space selects the scalar/operator contraction, while each remaining
/// assignment selects the m_i = 0 or m_i = 1 column of the corresponding contraction determinant.
/// # Arguments:
/// - `w`: Same-spin and different-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `diff`: Scratch storage for mixed contraction determinants, cofactors and work buffers.
/// - `a`: Prepared alpha-spin m_i = 0 and m_i = 1 contraction determinants.
/// - `b`: Prepared beta-spin m_i = 0 and m_i = 1 contraction determinants.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: Different-spin two-body matrix element summed over all allowed spin-resolved distributions.
#[inline(always)]
fn xw_h2_diff_gen<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    l_ex: &Excitation,
    g_ex: &Excitation,
    diff: &mut WickScratch<T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h2_diff_gen, {
        // Determine L_\alpha and L_\beta and select the all-m_i = 0 and all-m_i = 1
        // contraction determinants for each spin space.
        let l_ex_a = &l_ex.alpha;
        let g_ex_a = &g_ex.alpha;
        let l_ex_b = &l_ex.beta;
        let g_ex_b = &g_ex.beta;

        let la = l_ex_a.holes.len() + g_ex_a.holes.len();
        let lb = l_ex_b.holes.len() + g_ex_b.holes.len();

        diff.ensure_diff(la, lb);

        let rows_a = &a.rows[..la];
        let cols_a = &a.cols[..la];
        let deta0 = &a.det0.as_slice()[..la * la];
        let deta1 = &a.det1.as_slice()[..la * la];

        let rows_b = &b.rows[..lb];
        let cols_b = &b.cols[..lb];
        let detb0 = &b.det0.as_slice()[..lb * lb];
        let detb1 = &b.det1.as_slice()[..lb * lb];

        let mut acc = <T as From<f64>>::from(0.0);
        let n = w.ab.n();

        // Enumerate m_{\alpha0} + \sum_zm_{\alpha z} = m_\alpha and
        // m_{\beta0} + \sum_ym_{\beta y} = m_\beta, constructing both mixed determinants and cofactors.
        get_det_adjt_diff(
            w,
            (la, lb),
            diff,
            DetBranches {
                zero: deta0,
                one: deta1,
            },
            DetBranches {
                zero: detb0,
                one: detb1,
            },
            tol,
            |bits_a, bits_b, scratch, det_deta, det_detb| {
                // The leading bits select m_{\alpha0} and m_{\beta0}; bit k + 1 selects the
                // assignment of column k in the corresponding contraction determinant.
                let ma0 = bit(bits_a, 0);
                let mb0 = bit(bits_b, 0);

                // C_0 = V_{\alpha\beta,0}^{(m_{\alpha0},m_{\beta0})}
                // \det\mathbf D_{\alpha,\mathrm{ov}}\det\mathbf D_{\beta,\mathrm{ov}}.
                let mut contrib = w.ab.vab0[ma0][mb0] * det_deta * det_detb;

                // C_\alpha = -\sum_z\det\mathbf D_{\alpha,\mathrm{ov}}^{z\rightarrow
                // \mathcal V_z^{\alpha,(m_{\alpha0},m_{\beta0},m_{\alpha z})}}
                // \det\mathbf D_{\beta,\mathrm{ov}}.
                let na = w.ab.n();
                let vab0 = w.ab.vab_t_slice(ma0, mb0, 0);
                let vab1 = w.ab.vab_t_slice(ma0, mb0, 1);

                for (k, &ck) in cols_a.iter().enumerate().take(la) {
                    let mak = bit(bits_a, k + 1);
                    let vsl = if mak == 0 { vab0 } else { vab1 };
                    let base = ck * na;

                    let det_repl = column_replacement_det(
                        la,
                        scratch.adjt_deta.as_slice(),
                        k,
                        |r| vsl[base + rows_a[r]],
                    );
                    contrib -= det_repl * det_detb;
                }

                // C_\beta = -\det\mathbf D_{\alpha,\mathrm{ov}}\sum_y
                // \det\mathbf D_{\beta,\mathrm{ov}}^{y\rightarrow
                // \mathcal V_y^{\beta,(m_{\beta0},m_{\alpha0},m_{\beta y})}}.
                let nb = w.ab.n();
                let vba0 = w.ab.vba_t_slice(mb0, ma0, 0);
                let vba1 = w.ab.vba_t_slice(mb0, ma0, 1);

                for (k, &ck) in cols_b.iter().enumerate().take(lb) {
                    let mbk = bit(bits_b, k + 1);
                    let vsl = if mbk == 0 { vba0 } else { vba1 };
                    let base = ck * nb;

                    let det_repl = column_replacement_det(
                        lb,
                        scratch.adjt_detb.as_slice(),
                        k,
                        |r| vsl[base + rows_b[r]],
                    );
                    contrib -= det_repl * det_deta;
                }

                // C_{\alpha\beta} = \sum_{z,y,\eta,\xi}\operatorname{cof}[\mathbf D_\alpha]_{\eta z}
                // \mathcal{II}_{\eta z,\xi y}^{(m_{\alpha0},m_{\alpha z},m_{\beta0},m_{\beta y})}
                // \operatorname{cof}[\mathbf D_\beta]_{\xi y}.
                let layout_b = ReplacementLayout {
                    n,
                    rows: rows_b,
                    cols: cols_b,
                };

                // Fix an alpha-spin cofactor and form the beta-spin replacement determinant whose
                // replacement column is the matching \mathcal{II} intermediate.
                for (i, &ra) in rows_a.iter().enumerate() {
                    for (j, &ca) in cols_a.iter().enumerate() {
                        let cofa = scratch.adjt_deta.as_slice()[idx(la, i, j)];
                        let ma1 = bit(bits_a, j + 1);

                        for k in 0..lb {
                            let mbk = bit(bits_b, k + 1);
                            let iisl = w.ab.iiab_slice(ma0, ma1, mb0, mbk);

                            let det_repl = column_replacement_det(
                                lb,
                                scratch.adjt_detb.as_slice(),
                                k,
                                |r| {
                                    ii_replacement(
                                        iisl,
                                        layout_b,
                                        DetIndex { row: r, col: k },
                                        DetIndex { row: ra, col: ca },
                                        true,
                                    )
                                },
                            );
                            contrib += cofa * det_repl;
                        }
                    }
                }

                acc += contrib;
            },
        );

        // H_{\alpha\beta} = \phi_\alpha{}^{xw}\tilde S_\alpha\phi_\beta{}^{xw}\tilde S_\beta
        // \sum_{\{m_{\alpha i}\},\{m_{\beta i}\}}(C_0 + C_\alpha + C_\beta + C_{\alpha\beta}).
        (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
            * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod))
            * acc
    })
}
