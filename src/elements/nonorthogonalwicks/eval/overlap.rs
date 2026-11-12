// elements/nonorthogonalwicks/eval/overlap.rs

// Crate-root imports.
use crate::ExcitationSpin;
use crate::NOCIScalar;
use crate::determinant::SpinStringData;
use crate::maths::{Simd, det_const, det_dynamic};
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::batch::{Packet, SpinFactor};
use super::dispatch::{dispatch_pair_ranks, dispatch_spin_ranks};
use super::helpers::mix_dets_same;
use super::prepare::{construct_determinant_indices, prepare_same};

/// The same-spin overlap factor family,
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`.
pub(crate) struct OverlapFactor;

impl SpinFactor<1> for OverlapFactor {
    /// Evaluate one same-spin overlap through the prepared scalar path.
    /// # Arguments:
    /// - `self`: Overlap factor family.
    /// - `w`: Same-spin reference-pair Wick intermediates.
    /// - `pair`: Bra and ket determinants, oriented as the reference pair `(x,w)`.
    /// - `scratch`: Reusable same-spin Wick workspace.
    /// # Returns
    /// - `[T; 1]`: Same-spin overlap, excluding the excitation phases.
    fn values<T: NOCIScalar>(
        &self,
        w: &SameSpinView<'_, T>,
        pair: (&SpinStringData, &SpinStringData),
        scratch: &mut WickScratch<T>,
    ) -> [T; 1] {
        let (x, wdet) = pair;
        [xw_overlap_prepared(
            w,
            &x.excitation,
            &wdet.excitation,
            scratch,
        )]
    }

    /// Evaluate one packet of fixed-rank same-spin overlaps. Each lane evaluates
    /// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`, with `X^{(0)}` contractions
    /// on and below the diagonal and `Y^{(0)}` contractions above it.
    /// # Arguments:
    /// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
    /// - `packet`: Contraction labels of every lane.
    /// # Returns
    /// - `[[T; LANES]; 1]`: Overlap of every lane, excluding the excitation phases.
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
    ) -> [[T; LANES]; 1] {
        // Build packed `\mathbf D_{\mathrm{ov}}(0,\ldots,0)`: lower entries use `X^{(0)}` and
        // upper entries use `Y^{(0)}`.
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let mut d = [V::zero(); D];
        for eta in 0..L {
            for z in 0..L {
                let matrix = if eta >= z { x0 } else { y0 };
                d[eta * L + z] = unsafe { packet.load::<T, V, RX, XFIX>(matrix, eta, z, false) };
            }
        }

        // `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}`; `w.phase` carries the orbital-pairing
        // sign.
        let determinant = det_const::<V, LANES, L, D>(&d);
        let pref = V::splat(w.phase * T::from_real(w.tilde_s_prod));
        let mut overlap = [T::from_real(0.0); LANES];
        V::store(V::mul(determinant, pref), &mut overlap);
        [overlap]
    }
}

/// Evaluate the same-spin overlap between excited determinants generated from the reference pair
/// `\langle{}^x\Psi| and |{}^w\Psi\rangle:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L = m}}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L).`
/// `Each m_i is zero or one. The lower triangle of \mathbf D_{\mathrm{ov}}, including its diagonal,`
/// `contains X^{(m_i)} contractions, while its upper triangle contains Y^{(m_i)} contractions.`
/// The implementation stores the orbital-pairing phase separately from the product of non-zero
/// `singular values forming {}^{xw}\tilde S.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: `Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.`
/// - `g_ex`: `Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.`
/// - `scratch`: Prepared contraction determinants and work storage.
/// # Returns
/// - `T`: Same-spin overlap matrix element.
#[inline(always)]
pub fn xw_overlap<T: NOCIScalar>(
    w: &SameSpinView<T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap, {
        // The contraction determinant has dimension `L = L_x + L_w`.
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;

        // A nonzero term requires one contraction for every zero-overlap orbital pair. The
        // constrained sum contains only the all-zero distribution for m = 0, only the all-one
        // distribution for m = L, and every allowed mixed distribution for 0 < m < L.
        if w.m > l {
            <T as From<f64>>::from(0.0)
        } else if w.m == 0 {
            xw_overlap_m0(w, l, scratch)
        } else if w.m == l {
            xw_overlap_ml(w, l, scratch)
        } else {
            xw_overlap_gen(w, l, scratch)
        }
    })
}

/// Evaluate the same-spin overlap
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L = m}}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L).`
/// For `m = 0`, the overlap-only path evaluates the single determinant without preparing
/// reusable Hamiltonian scratch data. Other cases use `prepare_same` followed by the general
/// overlap evaluator.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage used by the prepared evaluation path.
/// # Returns
/// - `T`: Same-spin overlap excluding excitation phases applied outside the Wick evaluation.
#[inline(always)]
pub(crate) fn xw_overlap_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) -> T {
    // A spin sector with near-zero orbital pairs is the sum of its branches.
    if w.nbranches() > 1 {
        return (0..w.nbranches()).fold(<T as From<f64>>::from(0.0), |acc, k| {
            acc + xw_overlap_prepared(&w.branch(k), l_ex, g_ex, scratch)
        });
    }

    // Determine the contraction-determinant dimension `L = L_x + L_w`.
    let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;

    // No distribution satisfying `\sum_i m_i = m` exists when `m > L`.
    if w.m > l {
        return <T as From<f64>>::from(0.0);
    }

    // For `m = 0`, construct and evaluate `\mathbf D_{\mathrm{ov}}(0,\ldots,0)` directly without
    // populating the reusable scratch representation.
    if w.m == 0 {
        return xw_overlap_m0_direct(w, l_ex, g_ex);
    }

    // Prepare the all-`m_i = 0` and, where required, all-`m_i = 1` contraction determinants
    // before applying the standard overlap evaluation.
    prepare_same(w, l_ex, g_ex, scratch);
    xw_overlap(w, l_ex, g_ex, scratch)
}

/// Evaluate the same-spin overlap directly when `m = 0`:
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// The row labels are the x-reference particles followed by the w-reference holes, while the column
/// labels are the x-reference holes followed by the w-reference particles. The determinant contains
/// `X^{(0)}` on and below the diagonal and `Y^{(0)}` above the diagonal.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// # Returns
/// - `T`: Same-spin overlap excluding excitation phases applied outside the Wick evaluation.
#[inline(always)]
pub(crate) fn xw_overlap_m0_direct<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
) -> T {
    // Split `L` into the bra- and ket-reference excitation ranks and form `{}^{xw}\tilde S`
    // from the separately stored orbital-pairing phase and non-zero singular-value product.
    let rx = l_ex.holes.count_ones() as usize;
    let rw = g_ex.holes.count_ones() as usize;
    let l = rx + rw;
    let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

    // With no excitation pairs, the determinant is the empty determinant with value one.
    if l == 0 {
        return pref;
    }

    // Fixed ranks evaluate the packed kernel with one lane, the ket as its varying determinant.
    dispatch_spin_ranks!(
        (rx, rw),
        |RX, RW, L, D| {
            let (fixed, varying) = (l_ex.cache(), [g_ex.cache()]);
            let packet = Packet::new(w, &fixed, &varying);
            unsafe {
                OverlapFactor::packed::<T, T::Packed1, 1, RX, RW, L, D, true>(w, &packet)[0][0]
            }
        },
        xw_overlap_m0_direct_gen(w, l_ex, g_ex, l),
    )
}

/// Evaluate one direct all-`m_i = 0` overlap for ranks outside the const-dispatch table:
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `l`: Total excitation rank `L = RX + RW`.
/// # Returns
/// - `T`: Same-spin overlap contribution.
#[inline(always)]
fn xw_overlap_m0_direct_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    l: usize,
) -> T {
    // Construct runtime labels for `\mathbf D_{\mathrm{ov}}` when `(R_x,R_w)` lies outside the
    // generated const-rank table.
    let n = w.n();
    let x0 = w.x_slice(0);
    let y0 = w.y_slice(0);
    let zero = <T as From<f64>>::from(0.0);
    let mut rows = vec![0usize; l];
    let mut cols = vec![0usize; l];
    let mut d = vec![zero; l * l];
    construct_determinant_indices(x_ex, w_ex, w, &mut rows, &mut cols);

    // Fill `D_{ij}` from `X^{(0)}` on/below the diagonal and `Y^{(0)}` above it.
    for i in 0..l {
        let row = rows[i] * n;
        for j in 0..l {
            d[i * l + j] = if i >= j {
                x0[row + cols[j]]
            } else {
                y0[row + cols[j]]
            };
        }
    }

    // Multiply the runtime determinant by the reduced reference overlap `{}^{xw}\tilde S`.
    w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det_dynamic(&d, l).unwrap_or(zero)
}

/// Evaluate the same-spin overlap when m = 0:
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// For `L = 0`, the overlap is the reduced reference overlap `{}^{xw}\tilde S`.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l`: `Total excitation rank L = L_x + L_w.`
/// - `scratch`: `Prepared \mathbf D_{\mathrm{ov}}(0,\ldots,0).`
/// # Returns
/// - `T`: Same-spin overlap matrix element for m = 0.
#[inline(always)]
fn xw_overlap_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap_m0, {
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * det_dynamic(scratch.det0.as_slice(), l).unwrap_or(<T as From<f64>>::from(0.0))
    })
}

/// Evaluate the same-spin overlap when m = L. The only allowed distribution is
/// `(m_1,\ldots,m_L) = (1,\ldots,1), so:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,\ldots,1)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l`: `Total excitation rank L = L_x + L_w, equal to m in this path.`
/// - `scratch`: `Prepared \mathbf D_{\mathrm{ov}}(1,\ldots,1).`
/// # Returns
/// - `T`: Same-spin overlap matrix element for m = L.
#[inline(always)]
fn xw_overlap_ml<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap_ml, {
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * det_dynamic(scratch.det1.as_slice(), l).unwrap_or(<T as From<f64>>::from(0.0))
    })
}

/// Evaluate the same-spin overlap for 0 < m < L by summing every allowed distribution:
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L = m}}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L), \qquad m_i \in \{0,1\}.`
/// `Each distribution selects every column of \mathbf D_{\mathrm{ov}} from the corresponding`
/// `all-m_i = 0 or all-m_i = 1 contraction determinant.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l`: `Total excitation rank L = L_x + L_w.`
/// - `scratch`: `Prepared all-m_i = 0 and all-m_i = 1 determinants and mixed-determinant storage.`
/// # Returns
/// - `T`: `Same-spin overlap summed over all \binom{L}{m} allowed distributions.`
#[inline(always)]
fn xw_overlap_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap_gen, {
        let mut acc = <T as From<f64>>::from(0.0);

        // Enumerate the `\binom{L}{m}` distributions satisfying `\sum_i m_i = m` and construct
        // each `\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L)` by selecting its columns from `det0` or
        // `det1`.
        mix_dets_same(w, l, 0, scratch, |_, scratch| {
            let d = scratch.det_mix.as_slice();
            acc += det_dynamic(d, l).unwrap_or(<T as From<f64>>::from(0.0));
        });

        // Apply the orbital-pairing phase to the product of non-zero singular values to recover
        // `{}^{xw}\tilde S\sum_{\{m_i\}}\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L)`.
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
    })
}
