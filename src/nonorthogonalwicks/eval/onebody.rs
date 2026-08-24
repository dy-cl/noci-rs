// nonorthogonalwicks/eval/onebody.rs
// Crate-root imports.
use crate::ExcitationSpin;
use crate::maths::{adjugate_transpose, adjugate_transpose_const};
use crate::noci::NOCIScalar;
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{bit, column_replacement_correction, get_det_adjt_same};

/// `Read the scalar generalised-Fock intermediate {}^x F_0^{(m_i)} for the zero-overlap`
/// assignment `m_i` of the operator contraction.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `mi`: `Operator-contraction assignment m_i \in \{0,1\}.`
/// # Returns:
/// - `T`: `Scalar intermediate {}^x F_0^{(m_i)}.`
#[inline(always)]
fn one_body_scalar<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    mi: usize,
) -> T {
    // This is the scalar intermediate F_0^{(m_i)} from the one-body GNME expansion.
    w.f0f[mi]
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
        // Evaluate the same-spin one-body GNME expression through the m = 0 or mixed
        // zero-overlap distribution path selected in `xw_one_body`.
        xw_one_body(w, l_ex, g_ex, scratch, tol)
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
/// # Returns:
/// - `T`: One-body matrix element.
#[inline(always)]
fn xw_one_body<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    // For m = 0 only the all-m_i = 0 contraction determinant contributes. Otherwise,
    // sum the distributions satisfying \sum_{i=1}^{L+1}m_i = m.
    if w.m == 0 {
        xw_one_body_m0(w, l_ex, g_ex, scratch, tol)
    } else {
        xw_one_body_gen(w, l_ex, g_ex, scratch, tol)
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
/// # Returns:
/// - `T`: One-body matrix element for m = 0.
#[inline(always)]
fn xw_one_body_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_one_body_m0, {
        // Determine the total excitation rank L = L_x + L_w.
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
        // Dispatch to direct fixed-rank forms of
        // {}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}
        // - \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}.
        match l {
            // For L = 0, \det\mathbf D_{\mathrm{ov}} = 1 and there are no replacement columns.
            0 => w.phase * <T as From<f64>>::from(w.tilde_s_prod) * one_body_scalar(w, 0),
            1 => xw_one_body_m0_const::<T, 1>(w, scratch, tol),
            2 => xw_one_body_m0_const::<T, 2>(w, scratch, tol),
            3 => xw_one_body_m0_const::<T, 3>(w, scratch, tol),
            4 => xw_one_body_m0_const::<T, 4>(w, scratch, tol),
            _ => xw_one_body_m0_gen(w, l_ex, g_ex, scratch, tol),
        }
    })
}

/// `Evaluate the fixed-rank L one-body matrix element for m = 0.`
/// `The sum of column-replacement determinants is evaluated by contracting the \mathcal F entries`
/// `with \operatorname{cof}[\mathbf D_{\mathrm{ov}}]:`
/// `\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}`
/// `= \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-L contraction determinant and scratch storage for its cofactors.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate-transpose matrix.
/// # Returns:
/// - `T`: `One-body matrix element for fixed L and m = 0.`
#[inline(always)]
fn xw_one_body_m0_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_one_body_m0_const,
        {
            let n = w.n();
            let det0 = &scratch.det0.as_slice()[..L * L];
            scratch.adjt_det.ensure(L, L);

            // Evaluate det D_ov and cof[D_ov]_{eta z}=(-1)^{eta+z} det D_ov[eta|z]
            // for the m_i = 0 overlap contraction determinant.
            if let Some(det) = adjugate_transpose_const::<T, L>(
                scratch.adjt_det.as_mut_slice(),
                scratch.invs.as_mut_slice(),
                scratch.lu.as_mut_slice(),
                det0,
                tol,
            ) {
                let cof = scratch.adjt_det.as_slice();
                let rows = scratch.rows.as_slice();
                let cols = scratch.cols.as_slice();
                let fsl = w.ff_t_slice(0, 0);
                let mut repl = <T as From<f64>>::from(0.0);

                // Laplace expansion of the inserted one-body row gives
                // sum_z det D_ov^{z -> F_z} = sum_{eta,z} cof[D_ov]_{eta z} F_{eta z}.
                for z in 0..L {
                    let base = cols[z] * n;

                    for eta in 0..L {
                        repl += cof[eta * L + z] * fsl[base + rows[eta]];
                    }
                }

                // <x Psi_i...^a...|f|w Psi_j...^b...>
                // = S_tilde [F_0 det D_ov - sum_{eta,z} cof[D_ov]_{eta z} F_{eta z}].
                w.phase
                    * <T as From<f64>>::from(w.tilde_s_prod)
                    * (det * one_body_scalar(w, 0) - repl)
            } else {
                <T as From<f64>>::from(0.0)
            }
        }
    )
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
/// # Returns:
/// - `T`: One-body matrix element for arbitrary L and m = 0.
#[inline(always)]
fn xw_one_body_m0_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
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
            let mut contrib = det_det * one_body_scalar(w, 0);
            let fsl = w.ff_t_slice(0, 0);

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
/// # Returns:
/// - `T`: One-body matrix element summed over all allowed distributions.
#[inline(always)]
fn xw_one_body_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
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
            let mut contrib = det_det * one_body_scalar(w, mi);

            // Select \mathcal F^{(m_1,0)} and \mathcal F^{(m_1,1)}. The assignment
            // of each replaced determinant column chooses between these two slices.
            let f0 = w.ff_t_slice(mi, 0);
            let f1 = w.ff_t_slice(mi, 1);

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
