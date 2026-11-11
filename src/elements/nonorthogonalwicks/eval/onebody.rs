// elements/nonorthogonalwicks/eval/onebody.rs

// Crate-root imports.
use crate::ExcitationSpin;
use crate::NOCIScalar;
use crate::determinant::SpinStringData;
#[cfg(target_arch = "x86_64")]
use crate::maths::{Simd, adjugate_transpose_simd_const};
use crate::maths::{
    adjugate_transpose_const, adjugate_transpose_dynamic, build_d_const, build_d_dynamic,
    det_dynamic,
};
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
#[cfg(target_arch = "x86_64")]
use super::batch::Packet;
use super::batch::SpinFactor;
use super::dispatch::{dispatch_pair_ranks, dispatch_spin_ranks};
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, mix_dets_same,
};
use super::prepare::construct_determinant_indices;

/// The same-spin one-body factor family: the overlap together with the generalised-Fock matrix
/// element `\langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat f|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`.
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
    #[cfg(target_arch = "x86_64")]
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
                let determinant = adjugate_transpose_simd_const::<V, LANES, L, D>(&mut cof, &d);
                let mut replacement = V::zero();
                for eta in 0..L {
                    for z in 0..L {
                        let entry = unsafe { packet.load::<T, V, RX, XFIX>(fsl, eta, z, true) };
                        replacement = V::madd(replacement, cof[eta * L + z], entry);
                    }
                }

                // Apply the reduced-overlap phase and singular-value product after the packed
                // determinant work.
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
/// `contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0). Fixed-rank prepared kernels are`
/// `used for L = 1,\ldots,4; arbitrary ranks construct the generic determinant once and then apply`
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
            dispatch_spin_ranks!(
                (rx, rw),
                |RX, RW, L, D| xw_onebody_m0_prepared_const::<T, RX, RW, L, D>(
                    w, x_ex, w_ex, scratch, tol,
                ),
                xw_onebody_m0_gen_prepared(w, x_ex, w_ex, scratch, rx + rw, tol),
            )
        }
    })
}

/// Prepare and evaluate the fixed-rank `L` overlap and generalised-Fock matrix element for `m = 0`.
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}.`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}].`
/// `For L = 1 and \mathbf D_{\mathrm{ov}} = [D_{00}], this reduces to`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}D_{00} - \mathcal F_{r_0c_0}^{(0,0)}].`
/// `For L = 2, the replacement contribution is`
/// `\det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0}`
/// `+ \det\mathbf D_{\mathrm{ov}}^{1\rightarrow\mathcal F_1}.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving determinant labels, determinant entries and cofactors.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for fixed `L` and `m = 0`.
#[inline(always)]
fn xw_onebody_m0_prepared_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_onebody_m0_const,
        {
            // Allocate the rank-`L` determinant, label, and cofactor storage.
            scratch.ensure_same(L);

            // Map bra labels into `V_x x O_x` and ket labels into `O_w x V_w`.
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;
            let rows = scratch.rows.as_mut_slice();
            let cols = scratch.cols.as_mut_slice();
            let mut x_holes = x_ex.holes;
            let mut x_parts = x_ex.parts;
            for i in 0..RX {
                cols[i] = x_holes.trailing_zeros() as usize;
                rows[i] = x_parts.trailing_zeros() as usize - nocc;
                x_holes &= x_holes - 1;
                x_parts &= x_parts - 1;
            }

            let mut w_holes = w_ex.holes;
            let mut w_parts = w_ex.parts;
            for i in 0..RW {
                rows[RX + i] = nvirt + w_holes.trailing_zeros() as usize;
                cols[RX + i] = w_parts.trailing_zeros() as usize;
                w_holes &= w_holes - 1;
                w_parts &= w_parts - 1;
            }

            // For `m = 0`, every column uses the `m_i = 0` fundamental contractions:
            // `D_{\eta z} = X^{(0)}_{r_\eta c_z}` for `\eta \geq z`, otherwise
            // `D_{\eta z} = Y^{(0)}_{r_\eta c_z}`.
            let x0 = w.x(0);
            let y0 = w.y(0);
            build_d_const::<T, L>(
                scratch.det0.as_mut_slice(),
                &x0,
                &y0,
                scratch.rows.as_slice(),
                scratch.cols.as_slice(),
            );

            let det0 = &scratch.det0.as_slice()[..D];
            let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

            // Evaluate `\det\mathbf D_{\mathrm{ov}}` and
            // `\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z} = (-1)^{\eta+z}`
            // `\det\mathbf D_{\mathrm{ov}}[\eta|z]`.
            // for the overlap part and the one-body column replacements.
            let det = adjugate_transpose_const::<T, L, D>(
                &mut scratch.adjt_det.as_mut_slice()[..D],
                det0,
            );
            if det.abs() > tol {
                let n = w.n();
                let fsl = w.ff_t_slice(0, 0);
                let rows = scratch.rows.as_slice();
                let cols = scratch.cols.as_slice();
                let cof = scratch.adjt_det.as_slice();
                let mut repl = <T as From<f64>>::from(0.0);

                // Laplace expansion of the inserted one-body row gives
                // `\sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\boldsymbol{\mathcal F}_z}`
                // ` = \sum_{\eta z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}`
                // `\mathcal F_{\eta z}`.
                for z in 0..L {
                    let base = cols[z] * n;

                    for eta in 0..L {
                        repl += cof[eta * L + z] * fsl[base + rows[eta]];
                    }
                }

                // Return `\tilde S\det\mathbf D_{\mathrm{ov}}` and
                // `\tilde S(F_0\det\mathbf D_{\mathrm{ov}}`
                // `- \sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z})`.
                (pref * det, pref * (det * w.f0f[0] - repl))
            } else {
                (<T as From<f64>>::from(0.0), <T as From<f64>>::from(0.0))
            }
        }
    )
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
        // Generic `m = 0` path: build `\mathbf D_{\mathrm{ov}}` once, then use
        // `\operatorname{cof}[\mathbf D_{\mathrm{ov}}]` for every
        // one-column F replacement determinant.
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

            for b in 0..l {
                let cb = scratch.cols[b];
                let base = cb * n;
                let corr =
                    column_replacement_correction(l, det0, scratch.adjt_det.as_slice(), b, |r| {
                        fsl[base + scratch.rows[r]]
                    });
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
        // General zero-overlap path: prepare the all-zero and all-one endpoint determinants,
        // then sum the GNME one-body expression over every allowed mixed distribution.
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
            let mi = bit(bits, 0);

            // Use the adjugate to evaluate every one-column Laplace replacement together.
            if let Some(det_det) = adjugate_transpose_generic(
                scratch.adjt_det.as_mut_slice(),
                scratch.det_mix.as_slice(),
                l,
                tol,
            ) {
                if mi == 0 {
                    overlap_acc += det_det;
                }

                let mut contrib = det_det * w.f0f[mi];
                let f0 = w.ff_t_slice(mi, 0);
                let f1 = w.ff_t_slice(mi, 1);

                // Replace each excitation column by its precomputed one-body intermediate.
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

        // Restore the reduced-overlap and excitation-phase prefactor common to both results.
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        (pref * overlap_acc, pref * fock_acc)
    })
}
