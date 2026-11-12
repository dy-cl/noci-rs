// elements/nonorthogonalwicks/eval/onebody.rs

// Crate-root imports.
use crate::ExcitationSpin;
use crate::NOCIScalar;
use crate::determinant::SpinStringData;
use crate::maths::{
    Simd, adjugate_transpose_const, adjugate_transpose_dynamic, build_d_dynamic, det_dynamic,
};
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::batch::{Packet, SpinFactor};
use super::dispatch::{dispatch_pair_ranks, dispatch_spin_ranks};
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, mix_dets_same,
};
use super::prepare::construct_determinant_indices;

/// The same-spin one-body factor family: the overlap together with the generalised-Fock matrix
/// element `\langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat f|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`.
/// This type selects `SpinFactor<2>` for both pair batches and factorised Fock applications.
/// Its tolerance configures the runtime determinant/cofactor fallback; the fixed-rank packed
/// evaluator needs no tolerance. Keeping it here lets both paths use the same row scaffold.
pub(crate) struct OneBodyFactor {
    /// Numerical tolerance used by scalar determinant and adjugate evaluation.
    pub(crate) tol: f64,
}

impl SpinFactor<2> for OneBodyFactor {
    /// Evaluate one same-spin overlap and generalised-Fock pair through the prepared scalar path.
    /// # Arguments:
    /// - `self`: One-body factor family and its tolerance.
    /// - `w`: Same-spin reference-pair Wick intermediates.
    /// - `pair`: Bra and ket determinants, oriented as the reference pair `(x,w)`.
    /// - `scratch`: Reusable same-spin Wick workspace.
    /// # Returns
    /// - `[T; 2]`: Same-spin overlap and generalised-Fock factors, excluding the excitation phases.
    fn values<T: NOCIScalar>(
        &self,
        w: &SameSpinView<'_, T>,
        pair: (&SpinStringData, &SpinStringData),
        scratch: &mut WickScratch<T>,
    ) -> [T; 2] {
        let (x, wdet) = pair;
        let (s, f) = xw_onebody_prepared(w, &x.excitation, &wdet.excitation, scratch, self.tol);
        [s, f]
    }

    /// Evaluate one packet of fixed-rank same-spin overlap and generalised-Fock factors.
    /// Lane-wise, `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}` and
    /// `F = {}^{xw}\tilde S[F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
    /// `-\sum_{\eta z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}]`.
    /// # Arguments:
    /// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
    /// - `packet`: Contraction labels of every lane.
    /// # Returns
    /// - `[[T; LANES]; 2]`: Overlap and generalised-Fock factors of every lane, excluding the
    ///   excitation phases.
    /// # Safety
    /// - Cached labels must match `RX` and `RW`, and the caller must establish CPU support for `V`.
    #[inline(always)]
    unsafe fn packed<
        T: NOCIScalar,
        V: Simd<LANES, Scalar = T>,
        const LANES: usize,
        const RX: usize,
        const RW: usize,
        const L: usize,
        const D: usize,
        const XFIX: bool,
    >(
        w: &SameSpinView<'_, T>,
        packet: &Packet<'_, LANES>,
    ) -> [[T; LANES]; 2] {
        time_call!(
            crate::timers::nonorthogonalwicks::add_xw_onebody_m0_const,
            {
                // Build packed `\mathbf D_{\mathrm{ov}}(0,\ldots,0)` with `X^{(0)}` on and below the
                // diagonal and `Y^{(0)}` above it.
                let x0 = w.x_slice(0);
                let y0 = w.y_slice(0);
                let fsl = w.ff_t_slice(0, 0);
                let mut d = [V::zero(); D];
                let mut cof = [V::zero(); D];
                for eta in 0..L {
                    for z in 0..L {
                        let matrix = if eta >= z { x0 } else { y0 };
                        d[eta * L + z] =
                            unsafe { packet.load::<T, V, RX, XFIX>(matrix, eta, z, false) };
                    }
                }

                // Reuse `\operatorname{cof}[\mathbf D_{\mathrm{ov}}]` for every one-body replacement:
                // `F = \tilde S(F_0\det\mathbf D_{\mathrm{ov}}`
                // `-\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z})`.
                let determinant = adjugate_transpose_const::<V, LANES, L, D>(&mut cof, &d);
                let mut replacement = V::zero();

                for eta in 0..L {
                    for z in 0..L {
                        let entry = unsafe { packet.load::<T, V, RX, XFIX>(fsl, eta, z, true) };
                        replacement = V::madd(replacement, cof[eta * L + z], entry);
                    }
                }

                // Restore `{}^{xw}\tilde S` to both matrix elements. Its orbital-pairing phase
                // is stored in `w.phase`; the row writer later supplies the excitation phases.
                let fock = V::sub(V::mul(determinant, V::splat(w.f0f[0])), replacement);
                let pref = V::splat(w.phase * T::from_real(w.tilde_s_prod));

                let mut values = [[T::from_real(0.0); LANES]; 2];
                V::store(V::mul(pref, determinant), &mut values[0]);
                V::store(V::mul(pref, fock), &mut values[1]);

                values
            }
        )
    }
}

/// Evaluate the prepared overlap and generalised-Fock matrix element.
/// The overlap is
/// `S = {}^{xw}\tilde S\sum_{\sum_i m_i=m}\det\mathbf D_{\mathrm{ov}}`, and the one-body factor is
/// `F = {}^{xw}\tilde S\sum_{\sum_i m_i=m}`
/// `[F_0^{(m_1)}\det\mathbf D_{\mathrm{ov}}`
/// `-\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\boldsymbol{\mathcal F}_z}]`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Reusable Wick workspace.
/// - `tol`: Numerical tolerance used by runtime cofactor evaluation.
/// # Returns
/// - `(T, T)`: Same-spin overlap and generalised-Fock matrix element.
#[inline(always)]
pub(crate) fn xw_onebody_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_onebody, {
        // A spin sector with near-zero orbital pairs is the sum of its branches.
        if w.nbranches() > 1 {
            let zero = <T as From<f64>>::from(0.0);
            return (0..w.nbranches()).fold((zero, zero), |(s, f), k| {
                let (sk, fk) = xw_onebody_prepared(&w.branch(k), x_ex, w_ex, scratch, tol);
                (s + sk, f + fk)
            });
        }

        // With `m = 0`, one determinant and its cofactors suffice. Nonzero nullity requires the
        // constrained sum over binary fundamental-contraction assignments.
        if w.m == 0 {
            xw_onebody_m0_prepared(w, x_ex, w_ex, scratch, tol)
        } else {
            xw_onebody_gen_prepared(w, x_ex, w_ex, scratch, tol)
        }
    })
}

/// Prepare and evaluate the overlap and generalised-Fock matrix element together when `m = 0`.
/// `Every contraction uses m_i = 0, so the total excitation rank L = L_x + L_w determines one`
/// `contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0). Generated ranks use the packed`
/// `kernel with one lane; arbitrary ranks construct the generic determinant once and then apply`
/// `the general cofactor form. For L = 0, the overlap is {}^{xw}\tilde S and only`
/// `{}^x F_0^{(0)} contributes to F.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for determinant labels and generic work arrays.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `m = 0`.
#[inline(always)]
fn xw_onebody_m0_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_onebody_m0, {
        // `L = R_x + R_w` chooses empty, generated fixed-rank, or runtime-rank evaluation.
        let rx = x_ex.holes.count_ones() as usize;
        let rw = w_ex.holes.count_ones() as usize;

        if rx == 0 && rw == 0 {
            let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
            (pref, pref * w.f0f[0])
        } else {
            // Fixed ranks evaluate the packed kernel with one lane, the ket as its varying
            // determinant.
            dispatch_spin_ranks!(
                (rx, rw),
                |RX, RW, L, D| {
                    let (fixed, varying) = (x_ex.cache(), [w_ex.cache()]);
                    let packet = Packet::new(w, &fixed, &varying);
                    let [[s], [f]] = unsafe {
                        OneBodyFactor::packed::<T, T::Packed1, 1, RX, RW, L, D, true>(w, &packet)
                    };
                    (s, f)
                },
                xw_onebody_m0_gen_prepared(w, x_ex, w_ex, scratch, rx + rw, tol),
            )
        }
    })
}

/// Prepare and evaluate the generic-rank overlap and generalised-Fock matrix element for `m = 0`.
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}},`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{z = 1}^{L}\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}].`
/// `The determinant labels and \mathbf D_{\mathrm{ov}}(0,\ldots,0) are prepared once before the`
/// `cofactor evaluation. If the one-body adjugate path rejects the determinant, only the overlap`
/// `determinant is evaluated separately, preserving the numerical convention of the existing evaluator.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the contraction determinant and its cofactors.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate-transpose matrix.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for arbitrary `L` and `m = 0`.
#[inline(always)]
fn xw_onebody_m0_gen_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    l: usize,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_onebody_m0_gen, {
        // Build `\mathbf D_{\text{ov}}(0,\ldots,0)` once for both S and F.
        scratch.ensure_same(l);

        construct_determinant_indices(
            x_ex,
            w_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let x0 = w.x(0);
        let y0 = w.y(0);
        build_d_dynamic(
            scratch.det0.as_mut_slice(),
            l,
            &x0,
            &y0,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

        let zero = <T as From<f64>>::from(0.0);
        let n = w.n();
        let det0 = &scratch.det0.as_slice()[..l * l];
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

        // `F = \tilde S[F_0^{(0)}\det D - \sum_z\det D^{z\rightarrow\mathcal F_z}]`.
        // Reuse `\operatorname{cof}(D)` for each replacement column.
        if let Some(det_det) = adjugate_transpose_dynamic(
            scratch.adjt_det.as_mut_slice(),
            scratch.invs.as_mut_slice(),
            scratch.lu.as_mut_slice(),
            det0,
            l,
            tol,
        ) {
            let mut contrib = det_det * w.f0f[0];
            let fsl = w.ff_t_slice(0, 0);

            // `w.f0f[0]` is `{}^xF_0^{(0)}`. Column z of the replacement intermediate
            // is contiguous in `fsl`, which stores `\mathcal F_{\eta z}` transposed.
            for b in 0..l {
                let cb = scratch.cols[b];
                let base = cb * n;
                let corr =
                    column_replacement_correction(l, det0, scratch.adjt_det.as_slice(), b, |r| {
                        fsl[base + scratch.rows[r]]
                    });

                // The correction is `\det D^{z\rightarrow\mathcal F_z} - \det D`.
                contrib -= det_det + corr;
            }

            (pref * det_det, pref * contrib)
        } else {
            let overlap = pref * det_dynamic(det0, l).unwrap_or(zero);
            (overlap, zero)
        }
    })
}

/// Prepare and evaluate the overlap and generalised-Fock matrix element when `m > 0` by summing
/// the allowed one-body distributions
/// `m_1 + \cdots + m_{L+1} = m, \qquad m_i \in \{0,1\}.`
/// `The determinant labels and endpoint contraction determinants`
/// `\mathbf D_{\mathrm{ov}}(0,\ldots,0)` and `\mathbf D_{\mathrm{ov}}(1,\ldots,1)` are built once.
/// `The first assignment selects {}^x F_0^{(m_1)} and the operator side of each`
/// `\mathcal F^{(m_1,m_j)} column; the remaining assignments select the columns of`
/// `\mathbf D_{\mathrm{ov}}. Terms with m_1 = 0 also satisfy the overlap constraint`
/// `m_2+\cdots+m_{L+1} = m and are accumulated into the overlap without a second distribution loop.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for endpoint and mixed contraction determinants, cofactors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` summed over all distributions.
#[inline(always)]
fn xw_onebody_gen_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_onebody_gen, {
        // For `m > 0`, sum the one-body expression over
        // `m_1 + \cdots + m_{L+1} = m`, with each `m_i \in \{0,1\}`.
        let l = x_ex.holes.count_ones() as usize + w_ex.holes.count_ones() as usize;
        scratch.ensure_same(l);

        // Encode the overlap determinant's annihilation rows and creation columns.
        construct_determinant_indices(
            x_ex,
            w_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        // Build endpoint determinants so mixed zero-overlap assignments can copy columns cheaply.
        let x0 = w.x(0);
        let y0 = w.y(0);
        build_d_dynamic(
            scratch.det0.as_mut_slice(),
            l,
            &x0,
            &y0,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

        let x1 = w.x(1);
        let y1 = w.y(1);
        build_d_dynamic(
            scratch.det1.as_mut_slice(),
            l,
            &x1,
            &y1,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

        // Accumulate overlap and one-body matrix elements over all valid `m_i` distributions.
        let zero = <T as From<f64>>::from(0.0);
        let n = w.n();
        let mut overlap_acc = zero;
        let mut fock_acc = zero;

        mix_dets_same(w, l, 1, scratch, |bits, scratch| {
            // Bit 0 is the operator assignment `m_1`; bits 1..L select the columns of D.
            let mi = bit(bits, 0);

            // Use the adjugate to evaluate every one-column Laplace replacement together.
            if let Some(det_det) = adjugate_transpose_generic(
                scratch.adjt_det.as_mut_slice(),
                scratch.det_mix.as_slice(),
                l,
                tol,
            ) {
                if mi == 0 {
                    // Only these terms also satisfy the overlap sum over L columns.
                    overlap_acc += det_det;
                }

                let mut contrib = det_det * w.f0f[mi];
                let f0 = w.ff_t_slice(mi, 0);
                let f1 = w.ff_t_slice(mi, 1);

                // Subtract `\sum_z\det D^{z\rightarrow\boldsymbol{\mathcal F}_z}`.
                // `ff_t_slice(mi,mj)` stores `\mathcal F_{\eta z}` transposed, so each
                // creation-column index selects one contiguous replacement column.
                for b in 0..l {
                    let mj = bit(bits, b + 1);
                    let cb = scratch.cols[b];
                    let fsl = if mj == 0 { f0 } else { f1 };
                    let base = cb * n;
                    let corr = column_replacement_correction(
                        l,
                        scratch.det_mix.as_slice(),
                        scratch.adjt_det.as_slice(),
                        b,
                        |r| fsl[base + scratch.rows[r]],
                    );

                    contrib -= det_det + corr;
                }

                fock_acc += contrib;
            } else if mi == 0 {
                // Preserve the overlap determinant if the adjugate evaluation fails.
                overlap_acc += det_dynamic(scratch.det_mix.as_slice(), l).unwrap_or(zero);
            }
        });

        // Restore `{}^{xw}\tilde S`, including its orbital-pairing phase, to both results.
        // Excitation phases are applied by the calling row or determinant-pair evaluator.
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        (pref * overlap_acc, pref * fock_acc)
    })
}
