// nonorthogonalwicks/eval/onebody.rs
// Crate-root imports.
use crate::ExcitationSpin;
use crate::maths::adjugate_transpose;
use crate::noci::NOCIScalar;
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{bit, column_replacement_correction, get_det_adjt_same};

#[derive(Clone, Copy)]
enum OneBody {
    /// `Use the {}^x F_0^{(m_i)} and {}^{\chi_r\chi_z}\mathcal F_{rz}^{(m_i,m_j)}`
    /// `intermediates constructed from the one-electron Hamiltonian \hat h.`
    H1,
    /// `Use the corresponding intermediates constructed from the current generalised-Fock operator \hat F.`
    Fock,
}

/// `Read the scalar one-body intermediate {}^x F_0^{(m_i)} for the selected operator and`
/// `zero-overlap assignment m_i of the operator contraction.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `ob`: Selects intermediates constructed from the one-electron Hamiltonian or generalised Fock operator.
/// - `mi`: `Operator-contraction assignment m_i \in \{0,1\}.`
/// # Returns
/// - `T`: `Scalar intermediate {}^x F_0^{(m_i)}.`
#[inline(always)]
fn one_body_scalar<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ob: OneBody,
    mi: usize,
) -> T {
    // Read {}^x F_0^{(m_i)} from the intermediates formed with \hat h or \hat F.
    match ob {
        OneBody::H1 => w.f0h[mi],
        OneBody::Fock => w.f0f[mi],
    }
}

/// Evaluate the one-electron Hamiltonian matrix element between excited determinants generated from
/// `\langle{}^x\Psi| and |{}^w\Psi\rangle. The {}^x F_0^{(m_i)} and`
/// `{}^{\chi_r\chi_z}\mathcal F_{rz}^{(m_i,m_j)} intermediates are those constructed from \hat h.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: `Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.`
/// - `g_ex`: `Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.`
/// - `scratch`: Scratch storage for contraction determinants, cofactors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: One-electron Hamiltonian matrix element.
#[inline(always)]
pub(crate) fn xw_h1<T: NOCIScalar>(
    w: &SameSpinView<T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_h1, {
        xw_one_body(w, l_ex, g_ex, scratch, tol, OneBody::H1)
    })
}

/// Evaluate the generalised-Fock matrix element between excited determinants generated from
/// `\langle{}^x\Psi| and |{}^w\Psi\rangle. The {}^x F_0^{(m_i)} and`
/// `{}^{\chi_r\chi_z}\mathcal F_{rz}^{(m_i,m_j)} intermediates are those constructed from \hat F.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: `Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.`
/// - `g_ex`: `Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.`
/// - `scratch`: Scratch storage for contraction determinants, cofactors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `T`: Generalised-Fock matrix element.
#[inline(always)]
pub(crate) fn xw_f<T: NOCIScalar>(
    w: &SameSpinView<T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f, {
        xw_one_body(w, l_ex, g_ex, scratch, tol, OneBody::Fock)
    })
}

/// Evaluate a one-body matrix element between excited determinants generated from the reference pair
/// `\langle{}^x\Psi| and |{}^w\Psi\rangle:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat f|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+1}\\m_1+\cdots+m_{L+1}=m}}`
/// `[{}^x F_0^{(m_1)}\det\mathbf D_{\mathrm{ov}}(m_2,\ldots,m_{L+1})`
/// `- \sum_{z=1}^{L}\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}`
///   `(m_1,\ldots,m_{L+1})].`
///   `Each m_i is zero or one. The first assignment belongs to the operator contraction and the`
///   `remaining L assignments belong to the columns of \mathbf D_{\mathrm{ov}}. The implementation`
///   stores the orbital-pairing phase separately from the product of non-zero singular values.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: `Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.`
/// - `g_ex`: `Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.`
/// - `scratch`: Scratch storage for contraction determinants, cofactors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// - `ob`: Selects the one-electron Hamiltonian or generalised-Fock intermediates.
/// # Returns
/// - `T`: One-body matrix element.
#[inline(always)]
fn xw_one_body<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
    ob: OneBody,
) -> T {
    // For m = 0 only the all-m_i = 0 contraction determinant contributes. Otherwise,
    // sum the distributions satisfying \sum_{i=1}^{L+1}m_i = m.
    if w.m == 0 {
        xw_one_body_m0(w, l_ex, g_ex, scratch, tol, ob)
    } else {
        xw_one_body_gen(w, l_ex, g_ex, scratch, tol, ob)
    }
}

/// `Evaluate the one-body matrix element when m = 0, so every contraction uses m_i = 0.`
/// `Specialised kernels are used for L = 1,2,3, while all other excitation ranks use the general`
/// `cofactor form. For L = 0 only the scalar intermediate {}^x F_0^{(0)} contributes.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: `Prepared m_i = 0 contraction determinant and scratch work arrays.`
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// - `ob`: Selects the one-electron Hamiltonian or generalised-Fock intermediates.
/// # Returns
/// - `T`: One-body matrix element for m = 0.
#[inline(always)]
fn xw_one_body_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
    ob: OneBody,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_one_body_m0, {
        // Determine the total excitation rank L = L_x + L_w.
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
        // Dispatch to direct fixed-rank forms of
        // {}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}} - \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}.
        match l {
            // For L = 0, \det\mathbf D_{\mathrm{ov}} = 1 and there are no replacement columns.
            0 => w.phase * <T as From<f64>>::from(w.tilde_s_prod) * one_body_scalar(w, ob, 0),
            1 => xw_one_body_m0_l1(w, scratch, ob),
            2 => xw_one_body_m0_l2(w, scratch, ob),
            3 => xw_one_body_m0_l3(w, scratch, tol, ob),
            4 => xw_one_body_m0_l4(w, scratch, tol, ob),
            _ => xw_one_body_m0_gen(w, l_ex, g_ex, scratch, tol, ob),
        }
    })
}

/// `Evaluate the fixed-rank L = 1 one-body matrix element for m = 0.`
/// The scalar term and the sole one-column replacement are evaluated directly.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-one contraction determinant and its row and column labels.
/// - `ob`: Selects the one-electron Hamiltonian or generalised-Fock intermediates.
/// # Returns
/// - `T`: `One-body matrix element for L = 1 and m = 0.`
#[inline(always)]
fn xw_one_body_m0_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    ob: OneBody,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_one_body_m0_l1, {
        // For L = 1, \mathbf D_{\mathrm{ov}} = [D_{00}] and
        // \det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0} = \mathcal F_{r_0c_0}^{(0,0)}.
        let n = w.n();
        let det0 = scratch.det0.as_slice();
        let det = det0[0];
        let r0 = scratch.rows[0];
        let c0 = scratch.cols[0];
        // Select {}^{\chi_{r_0}\chi_{c_0}}\mathcal F_{r_0c_0}^{(0,0)}
        // constructed from the one-electron Hamiltonian or generalised Fock operator.
        let fsl = match ob {
            OneBody::H1 => w.fh_t_slice(0, 0),
            OneBody::Fock => w.ff_t_slice(0, 0),
        };
        let repl = fsl[c0 * n + r0];

        // \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat f|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
        // = {}^{xw}\tilde S[{}^x F_0^{(0)}D_{00} - \mathcal F_{r_0c_0}^{(0,0)}].
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * (det * one_body_scalar(w, ob, 0) - repl)
    })
}

/// `Evaluate the fixed-rank L = 2 one-body matrix element for m = 0.`
/// The scalar term and both one-column replacements are evaluated directly.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-two contraction determinant and its row and column labels.
/// - `ob`: Selects the one-electron Hamiltonian or generalised-Fock intermediates.
/// # Returns
/// - `T`: `One-body matrix element for L = 2 and m = 0.`
#[inline(always)]
fn xw_one_body_m0_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    ob: OneBody,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_one_body_m0_l2, {
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

        // Select the m_1 = m_z = 0 one-column intermediate.
        let fsl = match ob {
            OneBody::H1 => w.fh_t_slice(0, 0),
            OneBody::Fock => w.ff_t_slice(0, 0),
        };

        // Form the two replacement columns:
        // (\mathcal F_{r_0c_0}^{(0,0)},\mathcal F_{r_1c_0}^{(0,0)})^T and
        // (\mathcal F_{r_0c_1}^{(0,0)},\mathcal F_{r_1c_1}^{(0,0)})^T.
        let u0 = fsl[c0 * n + r0];
        let u1 = fsl[c0 * n + r1];
        let v0 = fsl[c1 * n + r0];
        let v1 = fsl[c1 * n + r1];

        // Evaluate \det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0} and
        // \det\mathbf D_{\mathrm{ov}}^{1\rightarrow\mathcal F_1}.
        let det_c0 = u0 * a11 - a01 * u1;
        let det_c1 = a00 * v1 - v0 * a10;

        // \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat f|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
        // = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}} - \det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0}
        // - \det\mathbf D_{\mathrm{ov}}^{1\rightarrow\mathcal F_1}].
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * (det * one_body_scalar(w, ob, 0) - det_c0 - det_c1)
    })
}

/// `Evaluate the fixed-rank L = 3 one-body matrix element for m = 0.`
/// `The sum of column-replacement determinants is evaluated by contracting the \mathcal F entries`
/// `with \operatorname{cof}[\mathbf D_{\mathrm{ov}}].`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-three contraction determinant and scratch storage for its cofactors.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate-transpose matrix.
/// - `ob`: Selects the one-electron Hamiltonian or generalised-Fock intermediates.
/// # Returns
/// - `T`: `One-body matrix element for L = 3 and m = 0.`
#[inline(always)]
fn xw_one_body_m0_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
    ob: OneBody,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_one_body_m0_l3, {
        // Select the rank-three contraction determinant \mathbf D_{\mathrm{ov}}(0,0,0).
        let n = w.n();
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
            let rows = scratch.rows.as_slice();
            let cols = scratch.cols.as_slice();

            let r0 = rows[0];
            let r1 = rows[1];
            let r2 = rows[2];
            let c0 = cols[0];
            let c1 = cols[1];
            let c2 = cols[2];

            // Select the m_1 = m_z = 0 one-column intermediate.
            let fsl = match ob {
                OneBody::H1 => w.fh_t_slice(0, 0),
                OneBody::Fock => w.ff_t_slice(0, 0),
            };

            // Read \mathcal F_{\eta z}^{(0,0)} for every row \eta and column z.
            let f00 = fsl[c0 * n + r0];
            let f10 = fsl[c0 * n + r1];
            let f20 = fsl[c0 * n + r2];
            let f01 = fsl[c1 * n + r0];
            let f11 = fsl[c1 * n + r1];
            let f21 = fsl[c1 * n + r2];
            let f02 = fsl[c2 * n + r0];
            let f12 = fsl[c2 * n + r1];
            let f22 = fsl[c2 * n + r2];

            // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}
            // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}.
            let repl = cof[0] * f00
                + cof[3] * f10
                + cof[6] * f20
                + cof[1] * f01
                + cof[4] * f11
                + cof[7] * f21
                + cof[2] * f02
                + cof[5] * f12
                + cof[8] * f22;

            // \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat f|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
            // = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}
            // - \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}].
            w.phase
                * <T as From<f64>>::from(w.tilde_s_prod)
                * (det * one_body_scalar(w, ob, 0) - repl)
        } else {
            <T as From<f64>>::from(0.0)
        }
    })
}

/// `Evaluate the fixed-rank L = 4 one-body matrix element for m = 0.`
/// `The sum of column-replacement determinants is evaluated by contracting the \mathcal F entries`
/// `with \operatorname{cof}[\mathbf D_{\mathrm{ov}}].`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-four contraction determinant and scratch storage for its cofactors.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate-transpose matrix.
/// - `ob`: Selects the one-electron Hamiltonian or generalised-Fock intermediates.
/// # Returns
/// - `T`: `One-body matrix element for L = 4 and m = 0.`
#[inline(always)]
fn xw_one_body_m0_l4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
    ob: OneBody,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_one_body_m0_l4, {
        // Select the rank-four contraction determinant \mathbf D_{\mathrm{ov}}(0,0,0,0).
        let n = w.n();
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
            let rows = scratch.rows.as_slice();
            let cols = scratch.cols.as_slice();

            let r0 = rows[0];
            let r1 = rows[1];
            let r2 = rows[2];
            let r3 = rows[3];
            let c0 = cols[0];
            let c1 = cols[1];
            let c2 = cols[2];
            let c3 = cols[3];

            // Select the m_1 = m_z = 0 one-column intermediate.
            let fsl = match ob {
                OneBody::H1 => w.fh_t_slice(0, 0),
                OneBody::Fock => w.ff_t_slice(0, 0),
            };

            // Read \mathcal F_{\eta z}^{(0,0)} for every row \eta and column z.
            let f00 = fsl[c0 * n + r0];
            let f10 = fsl[c0 * n + r1];
            let f20 = fsl[c0 * n + r2];
            let f30 = fsl[c0 * n + r3];
            let f01 = fsl[c1 * n + r0];
            let f11 = fsl[c1 * n + r1];
            let f21 = fsl[c1 * n + r2];
            let f31 = fsl[c1 * n + r3];
            let f02 = fsl[c2 * n + r0];
            let f12 = fsl[c2 * n + r1];
            let f22 = fsl[c2 * n + r2];
            let f32 = fsl[c2 * n + r3];
            let f03 = fsl[c3 * n + r0];
            let f13 = fsl[c3 * n + r1];
            let f23 = fsl[c3 * n + r2];
            let f33 = fsl[c3 * n + r3];

            // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}
            // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}.
            let repl = cof[0] * f00
                + cof[4] * f10
                + cof[8] * f20
                + cof[12] * f30
                + cof[1] * f01
                + cof[5] * f11
                + cof[9] * f21
                + cof[13] * f31
                + cof[2] * f02
                + cof[6] * f12
                + cof[10] * f22
                + cof[14] * f32
                + cof[3] * f03
                + cof[7] * f13
                + cof[11] * f23
                + cof[15] * f33;

            // \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat f|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
            // = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}
            // - \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}].
            w.phase
                * <T as From<f64>>::from(w.tilde_s_prod)
                * (det * one_body_scalar(w, ob, 0) - repl)
        } else {
            <T as From<f64>>::from(0.0)
        }
    })
}

/// Evaluate the one-body matrix element for arbitrary L when m = 0:
/// `{}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{z=1}^{L}\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}].`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Prepared contraction determinant and scratch storage for its cofactors.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate-transpose matrix.
/// - `ob`: Selects the one-electron Hamiltonian or generalised-Fock intermediates.
/// # Returns
/// - `T`: One-body matrix element for arbitrary L and m = 0.
#[inline(always)]
fn xw_one_body_m0_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
    ob: OneBody,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_one_body_m0_gen, {
        // Determine L = L_x + L_w and select \mathbf D_{\mathrm{ov}}(0,\ldots,0).
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
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
            // Start with {}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}.
            let mut contrib = det_det * one_body_scalar(w, ob, 0);
            let fsl = match ob {
                OneBody::H1 => w.fh_t_slice(0, 0),
                OneBody::Fock => w.ff_t_slice(0, 0),
            };

            // Subtract \det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z} for each column z.
            for b in 0..l {
                let cb = scratch.cols[b];
                let base = cb * n;
                // `corr` is the determinant correction, so `det_det + corr` is the determinant
                // obtained by replacing column b with \mathcal F_b^{(0,0)}.
                let corr =
                    column_replacement_correction(l, det0, scratch.adjt_det.as_slice(), b, |r| {
                        fsl[base + scratch.rows[r]]
                    });
                contrib -= det_det + corr;
            }
            acc += contrib;
        }

        // Apply the orbital-pairing phase to the product of non-zero singular values.
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
    })
}

/// Evaluate the one-body matrix element when m > 0 by summing every allowed distribution:
/// `m_1 + \cdots + m_{L+1} = m, \qquad m_i \in \{0,1\}.`
/// `The first assignment selects {}^x F_0^{(m_1)} and the operator side of each`
/// `\mathcal F^{(m_1,m_j)} column; the remaining assignments select the columns of`
/// `\mathbf D_{\mathrm{ov}}.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for mixed contraction determinants, cofactors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// - `ob`: Selects the one-electron Hamiltonian or generalised-Fock intermediates.
/// # Returns
/// - `T`: One-body matrix element summed over all allowed distributions.
#[inline(always)]
fn xw_one_body_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
    ob: OneBody,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_one_body_gen, {
        // Determine L = L_x + L_w.
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;

        let mut acc = <T as From<f64>>::from(0.0);
        let n = w.n();

        // Enumerate all distributions over the operator contraction and L determinant columns,
        // construct the corresponding mixed \mathbf D_{\mathrm{ov}}, and evaluate its cofactors.
        get_det_adjt_same(w, l, 1, scratch, tol, |bits, scratch, det_det| {
            // Bit zero is m_1, the assignment of the operator contraction.
            let mi = bit(bits, 0);
            // Start with {}^x F_0^{(m_1)}\det\mathbf D_{\mathrm{ov}}.
            let mut contrib = det_det * one_body_scalar(w, ob, mi);

            // Select \mathcal F^{(m_1,0)} and \mathcal F^{(m_1,1)}. The assignment
            // of each replaced determinant column chooses between these two slices.
            let f0 = match ob {
                OneBody::H1 => w.fh_t_slice(mi, 0),
                OneBody::Fock => w.ff_t_slice(mi, 0),
            };
            let f1 = match ob {
                OneBody::H1 => w.fh_t_slice(mi, 1),
                OneBody::Fock => w.ff_t_slice(mi, 1),
            };

            // Subtract every \det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}.
            for b in 0..l {
                // Bit b + 1 is the zero-overlap assignment of determinant column b.
                let mj = bit(bits, b + 1);
                let cb = scratch.cols[b];
                let fsl = if mj == 0 { f0 } else { f1 };
                let base = cb * n;

                // `det_det + corr` is the mixed contraction determinant with column b
                // replaced by \mathcal F_b^{(m_1,m_{b+2})}.
                let corr = column_replacement_correction(
                    l,
                    scratch.det_mix.as_slice(),
                    scratch.adjt_det.as_slice(),
                    b,
                    |r| fsl[base + scratch.rows[r]],
                );
                contrib -= det_det + corr;
            }
            acc += contrib;
        });
        // Apply the orbital-pairing phase to the product of non-zero singular values and
        // multiply the constrained sum over m_1,\ldots,m_{L+1}.
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
    })
}
