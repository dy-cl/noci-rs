// nonorthogonalwicks/eval/rdm.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

// External crate imports.
use ndarray::Array2;
#[cfg(target_arch = "x86_64")]
use num_complex::Complex64;

// Crate-root imports.
use crate::ExcitationSpin;
use crate::config::MAXEXCIT;
#[cfg(target_arch = "x86_64")]
use crate::maths::{C64x4, C64x8, F64x4, F64x8, Simd, det_simd_const};
use crate::maths::{det_const, det_dynamic, mix_columns_dynamic};
use crate::noci::NOCIScalar;
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::dispatch::{
    dispatch_overlap_ranks, dispatch_overlap_scalar_ranks, dispatch_pair_ranks, dispatch_rdm_ranks,
    dispatch_rdm_scalar_ranks,
};
use super::helpers::{extend_rdm_d, for_each_m_combination};
use super::overlap::xw_overlap_prepared;
use super::prepare::construct_determinant_indices;

/// Evaluate a batch of unnormalised same-spin rank-`K` transition-density elements.
/// Every request evaluates
/// `{}^{xw}\tilde S\sum_{\sum_i m_i = m}`
/// `\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(m_1,\ldots,m_{L+K})`.
/// For `m = 0`, supported scalar types and ranks use the widest available fixed-rank SIMD kernel;
/// other requests use the scalar const-generic or arbitrary-rank path. The fundamental
/// contractions are transformed to the external RDM basis once for the complete batch.
/// A rank-zero batch is filled by the prepared overlap evaluator without an external-basis transform.
/// A batch is zero when `K > N_\sigma`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `coeff`: Bra- and ket-reference orbital coefficients in the external RDM basis.
/// - `requests`: Creation and annihilation index arrays in output order.
/// - `scratch`: Reusable determinant storage for scalar evaluation.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// - `out`: Same-spin RDM elements in request order.
/// # Returns
/// - `()`: Writes the evaluated requests into `out`.
pub(crate) fn xw_rdmk_same_prepared_batched<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    requests: &[([usize; K], [usize; K])],
    scratch: &mut WickScratch<T>,
    tol: f64,
    out: &mut [T],
) {
    if requests.is_empty() || out.is_empty() {
        return;
    }
    if K > w.nocc {
        for value in out.iter_mut().take(requests.len()) {
            *value = <T as From<f64>>::from(0.0);
        }
        return;
    }
    if K == 0 {
        let overlap = xw_overlap_prepared(w, ex.0, ex.1, scratch);
        for value in out.iter_mut().take(requests.len()) {
            *value = overlap;
        }
        return;
    }

    // Build the `m_i = 0` fundamental contractions once for the whole batch. The external RDM
    // basis adds `n_{\mathrm{RDM}}` rows and columns to the compact `n_{\mathrm{mo}}` determinant.
    let (l_c, g_c) = coeff;
    let nrdm = l_c.nrows();
    let ext_n = w.nmo + nrdm;
    let x0 = w.x(0);
    let y0 = w.y(0);
    let x0rdm = w.xrdm(0, nrdm);
    let y0rdm = w.yrdm(0, nrdm);
    let x0p = extend_rdm_d(w, &x0, &x0rdm, l_c, g_c).into_raw_vec();
    let y0p = extend_rdm_d(w, &y0, &y0rdm, l_c, g_c).into_raw_vec();

    #[cfg(target_arch = "x86_64")]
    if w.m == 0 && TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            // SAFETY: The `TypeId` check proves every generic value has its `f64` instantiation
            // for the duration of the SIMD helper call.
            let w_f64 = &*std::ptr::from_ref(w).cast::<SameSpinView<'_, f64>>();
            let x0_f64 = std::slice::from_raw_parts(x0p.as_ptr().cast::<f64>(), x0p.len());
            let y0_f64 = std::slice::from_raw_parts(y0p.as_ptr().cast::<f64>(), y0p.len());
            let out_f64 = std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len());
            if try_xw_rdmk_same_prepared_f64_simd(
                w_f64,
                ex,
                (x0_f64, y0_f64, ext_n),
                requests,
                tol,
                out_f64,
            ) {
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    if w.m == 0 && TypeId::of::<T>() == TypeId::of::<Complex64>() {
        unsafe {
            // SAFETY: The `TypeId` check proves every generic value has its `Complex64`
            // instantiation for the duration of the SIMD helper call.
            let w_c64 = &*std::ptr::from_ref(w).cast::<SameSpinView<'_, Complex64>>();
            let x0_c64 = std::slice::from_raw_parts(x0p.as_ptr().cast::<Complex64>(), x0p.len());
            let y0_c64 = std::slice::from_raw_parts(y0p.as_ptr().cast::<Complex64>(), y0p.len());
            let out_c64 =
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<Complex64>(), out.len());
            if try_xw_rdmk_same_prepared_c64_simd(
                w_c64,
                ex,
                (x0_c64, y0_c64, ext_n),
                requests,
                tol,
                out_c64,
            ) {
                return;
            }
        }
    }

    // If no fixed-width kernel accepts the batch, construct the optional `m_i = 1` contractions
    // and evaluate every request through the scalar or const-generic determinant path.
    let one = if w.m == 0 {
        None
    } else {
        let x1 = w.x(1);
        let y1 = w.y(1);
        let x1rdm = w.xrdm(1, nrdm);
        let y1rdm = w.yrdm(1, nrdm);
        Some((
            extend_rdm_d(w, &x1, &x1rdm, l_c, g_c).into_raw_vec(),
            extend_rdm_d(w, &y1, &y1rdm, l_c, g_c).into_raw_vec(),
        ))
    };
    let fundamental = (
        x0p.as_slice(),
        y0p.as_slice(),
        one.as_ref()
            .map(|(x1p, y1p)| (x1p.as_slice(), y1p.as_slice())),
        ext_n,
    );
    xw_rdmk_same_prepared_scalar_batch(w, ex, fundamental, requests, scratch, tol, out);
}

/// Packed RDM packet dispatcher behind one target-feature entry point.
#[cfg(target_arch = "x86_64")]
type RdmSimdPacket<T, const K: usize, const N: usize> = unsafe fn(
    &SameSpinView<'_, T>,
    (&ExcitationSpin, &ExcitationSpin),
    (&[T], &[T], usize),
    &[([usize; K], [usize; K]); N],
    f64,
    &mut [T; N],
);

/// Try to evaluate a real same-spin rank-`K` RDM batch with fixed-rank SIMD kernels.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates with `m = 0`.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions and row dimension.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: Real RDM outputs.
/// # Returns
/// - `bool`: Whether an available SIMD path evaluated the complete batch.
/// # Safety
/// - `w` must contain no zero-overlap orbital pairs.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_rdmk_same_prepared_f64_simd<const K: usize>(
    w: &SameSpinView<'_, f64>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[f64], &[f64], usize),
    requests: &[([usize; K], [usize; K])],
    tol: f64,
    out: &mut [f64],
) -> bool {
    // Generated packets cover external rank `K <= 4` and cached excitation ranks up to
    // `MAXEXCIT`; preserve scalar evaluation outside that region.
    let rx = ex.0.holes.count_ones() as usize;
    let rw = ex.1.holes.count_ones() as usize;
    if K > 4 || rx > MAXEXCIT || rw > MAXEXCIT {
        return false;
    }

    // Prefer the widest supported packet for the augmented `D_{RDM}` determinant.
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            xw_rdmk_same_prepared_simd_batch::<f64, K, 8>(
                w,
                ex,
                fundamental,
                requests,
                tol,
                out,
                xw_rdmk_same_m0_prepared_f64x8,
            );
        }
        return true;
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            xw_rdmk_same_prepared_simd_batch::<f64, K, 4>(
                w,
                ex,
                fundamental,
                requests,
                tol,
                out,
                xw_rdmk_same_m0_prepared_f64x4,
            );
        }
        return true;
    }
    false
}

/// Try to evaluate a complex same-spin rank-`K` RDM batch with fixed-rank SIMD kernels.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates with `m = 0`.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions and row dimension.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: Complex RDM outputs.
/// # Returns
/// - `bool`: Whether an available SIMD path evaluated the complete batch.
/// # Safety
/// - `w` must contain no zero-overlap orbital pairs.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_rdmk_same_prepared_c64_simd<const K: usize>(
    w: &SameSpinView<'_, Complex64>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[Complex64], &[Complex64], usize),
    requests: &[([usize; K], [usize; K])],
    tol: f64,
    out: &mut [Complex64],
) -> bool {
    // Complex packets use the same generated rank region as the real evaluator.
    let rx = ex.0.holes.count_ones() as usize;
    let rw = ex.1.holes.count_ones() as usize;
    if K > 4 || rx > MAXEXCIT || rw > MAXEXCIT {
        return false;
    }

    // Prefer the widest supported complex packet for the augmented `D_{RDM}` determinant.
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            xw_rdmk_same_prepared_simd_batch::<Complex64, K, 8>(
                w,
                ex,
                fundamental,
                requests,
                tol,
                out,
                xw_rdmk_same_m0_prepared_c64x8,
            );
        }
        return true;
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            xw_rdmk_same_prepared_simd_batch::<Complex64, K, 4>(
                w,
                ex,
                fundamental,
                requests,
                tol,
                out,
                xw_rdmk_same_m0_prepared_c64x4,
            );
        }
        return true;
    }
    false
}

/// Evaluate a same-spin RDM batch with fixed-width packed packets.
/// # Arguments:
/// - `w`: Same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions and row dimension.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: RDM outputs.
/// - `kernel`: Packed packet dispatcher.
/// # Returns
/// - `()`: Writes every RDM request.
/// # Safety
/// - `kernel` must support the current CPU and use `N` lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xw_rdmk_same_prepared_simd_batch<T: NOCIScalar, const K: usize, const N: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K])],
    tol: f64,
    out: &mut [T],
    kernel: RdmSimdPacket<T, K, N>,
) {
    // Process complete `N`-lane packets. Final packet repeats one valid request in unused lanes,
    // but writes only genuine requests back to `out`.
    let count = requests.len().min(out.len());
    let mut start = 0usize;
    while start < count {
        let lanes = (count - start).min(N);
        let mut packet = [unsafe { *requests.get_unchecked(start) }; N];
        for lane in 1..lanes {
            packet[lane] = unsafe { *requests.get_unchecked(start + lane) };
        }
        let mut values = [T::from_real(0.0); N];
        unsafe {
            kernel(w, ex, fundamental, &packet, tol, &mut values);
        }
        for lane in 0..lanes {
            unsafe {
                *out.get_unchecked_mut(start + lane) = values[lane];
            }
        }
        start += lanes;
    }
}

/// Evaluate packed fixed-rank same-spin RDM determinants.
/// Each lane evaluates
/// `\Gamma^{\mathbf p}_{\mathbf q} = {}^{xw}\tilde S`
/// `\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)`, where the external RDM
/// contractions precede the excitation contractions in the augmented determinant.
/// # Arguments:
/// - `w`: Same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions and row dimension.
/// - `requests`: External RDM indices in lane order.
/// - `tol`: Numerical threshold.
/// - `out`: RDM outputs in lane order.
/// # Returns
/// - `()`: Writes `LANES` transition-density values.
/// # Safety
/// - Indices must be valid and caller must establish `V` CPU support.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xw_rdmk_same_m0_prepared_simd_const<
    T: NOCIScalar,
    V: Simd<LANES, Scalar = T>,
    const LANES: usize,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const DD: usize,
>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K]); LANES],
    tol: f64,
    out: &mut [T; LANES],
) {
    // External creation/annihilation labels occupy first `K` rows and columns and vary by lane.
    // Excitation labels occupy remaining `L = R_x + R_w` positions and are shared by all lanes.
    let (x0, y0, ext_n) = fundamental;
    let mut rows = [[0usize; D]; LANES];
    let mut cols = [[0usize; D]; LANES];
    for lane in 0..LANES {
        for i in 0..K {
            rows[lane][i] = w.nmo + requests[lane].0[i];
            cols[lane][i] = w.nmo + requests[lane].1[i];
        }
    }

    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    let mut x_holes = ex.0.holes;
    let mut x_parts = ex.0.parts;
    for i in 0..RX {
        let col = x_holes.trailing_zeros() as usize;
        let row = x_parts.trailing_zeros() as usize - nocc;
        for lane in 0..LANES {
            cols[lane][K + i] = col;
            rows[lane][K + i] = row;
        }
        x_holes &= x_holes - 1;
        x_parts &= x_parts - 1;
    }
    let mut w_holes = ex.1.holes;
    let mut w_parts = ex.1.parts;
    for i in 0..(L - RX) {
        let row = nvirt + w_holes.trailing_zeros() as usize;
        let col = w_parts.trailing_zeros() as usize;
        for lane in 0..LANES {
            rows[lane][K + RX + i] = row;
            cols[lane][K + RX + i] = col;
        }
        w_holes &= w_holes - 1;
        w_parts &= w_parts - 1;
    }

    let zero = V::zero();
    let mut determinant = [zero; DD];

    // Build packed augmented determinant `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
    // Entries use `X^{(0)}` where the row creator stands left of the column annihilator in the
    // operator string, and `Y^{(0)}` otherwise. The external string is creators first,
    // `a^\dagger_{p_1}\cdots a^\dagger_{p_K} a_{q_K}\cdots a_{q_1}`, so its whole `K x K` block uses
    // `X^{(0)}`; the remaining entries take `X^{(0)}` on and below the diagonal.
    for i in 0..D {
        for j in 0..D {
            let matrix = if i >= j || j < K { x0 } else { y0 };
            let mut values = [T::from_real(0.0); LANES];
            for lane in 0..LANES {
                let index = rows[lane][i] * ext_n + cols[lane][j];
                values[lane] = unsafe { *matrix.get_unchecked(index) };
            }
            determinant[i * D + j] = V::load(&values);
        }
    }

    // Multiply each determinant by `{}^{xw}\tilde S`, then apply tolerance lane by lane.
    let value = det_simd_const::<V, LANES, D, DD>(&determinant);
    let pref = V::splat(w.phase * T::from_real(w.tilde_s_prod));
    let mut lanes = [T::from_real(0.0); LANES];
    V::store(V::mul(pref, value), &mut lanes);
    for lane in 0..LANES {
        out[lane] = if lanes[lane].abs() > tol {
            lanes[lane]
        } else {
            T::from_real(0.0)
        };
    }
}

/// Dispatch four real RDM values to a fixed-rank AVX2/FMA kernel.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: Real RDM outputs.
/// # Returns
/// - `()`: Writes four values.
/// # Safety
/// - The current CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_rdmk_same_m0_prepared_f64x4<const K: usize>(
    w: &SameSpinView<'_, f64>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[f64], &[f64], usize),
    requests: &[([usize; K], [usize; K]); 4],
    tol: f64,
    out: &mut [f64; 4],
) {
    let rx = ex.0.holes.count_ones() as usize;
    let rw = ex.1.holes.count_ones() as usize;
    dispatch_rdm_ranks!(
        K,
        (rx, rw),
        |K, RX, RW, L, D, DD| unsafe {
            xw_rdmk_same_m0_prepared_f64x4_const::<K, RX, RW, L, D, DD>(
                w,
                ex,
                fundamental,
                &*std::ptr::from_ref(requests).cast::<[([usize; K], [usize; K]); 4]>(),
                tol,
                out,
            )
        },
        (),
    )
}

/// Dispatch eight real RDM values to a fixed-rank AVX-512F kernel.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: Real RDM outputs.
/// # Returns
/// - `()`: Writes eight values.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_rdmk_same_m0_prepared_f64x8<const K: usize>(
    w: &SameSpinView<'_, f64>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[f64], &[f64], usize),
    requests: &[([usize; K], [usize; K]); 8],
    tol: f64,
    out: &mut [f64; 8],
) {
    let rx = ex.0.holes.count_ones() as usize;
    let rw = ex.1.holes.count_ones() as usize;
    dispatch_rdm_ranks!(
        K,
        (rx, rw),
        |K, RX, RW, L, D, DD| unsafe {
            xw_rdmk_same_m0_prepared_f64x8_const::<K, RX, RW, L, D, DD>(
                w,
                ex,
                fundamental,
                &*std::ptr::from_ref(requests).cast::<[([usize; K], [usize; K]); 8]>(),
                tol,
                out,
            )
        },
        (),
    )
}

/// Dispatch four complex RDM values to a fixed-rank AVX2/FMA kernel.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: Complex RDM outputs.
/// # Returns
/// - `()`: Writes four values.
/// # Safety
/// - The current CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_rdmk_same_m0_prepared_c64x4<const K: usize>(
    w: &SameSpinView<'_, Complex64>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[Complex64], &[Complex64], usize),
    requests: &[([usize; K], [usize; K]); 4],
    tol: f64,
    out: &mut [Complex64; 4],
) {
    let rx = ex.0.holes.count_ones() as usize;
    let rw = ex.1.holes.count_ones() as usize;
    dispatch_rdm_ranks!(
        K,
        (rx, rw),
        |K, RX, RW, L, D, DD| unsafe {
            xw_rdmk_same_m0_prepared_c64x4_const::<K, RX, RW, L, D, DD>(
                w,
                ex,
                fundamental,
                &*std::ptr::from_ref(requests).cast::<[([usize; K], [usize; K]); 4]>(),
                tol,
                out,
            )
        },
        (),
    )
}

/// Dispatch eight complex RDM values to a fixed-rank AVX-512F kernel.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: Complex RDM outputs.
/// # Returns
/// - `()`: Writes eight values.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_rdmk_same_m0_prepared_c64x8<const K: usize>(
    w: &SameSpinView<'_, Complex64>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[Complex64], &[Complex64], usize),
    requests: &[([usize; K], [usize; K]); 8],
    tol: f64,
    out: &mut [Complex64; 8],
) {
    let rx = ex.0.holes.count_ones() as usize;
    let rw = ex.1.holes.count_ones() as usize;
    dispatch_rdm_ranks!(
        K,
        (rx, rw),
        |K, RX, RW, L, D, DD| unsafe {
            xw_rdmk_same_m0_prepared_c64x8_const::<K, RX, RW, L, D, DD>(
                w,
                ex,
                fundamental,
                &*std::ptr::from_ref(requests).cast::<[([usize; K], [usize; K]); 8]>(),
                tol,
                out,
            )
        },
        (),
    )
}

/// Evaluate four real fixed-rank RDM values with AVX2/FMA.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: Real RDM outputs.
/// # Returns
/// - `()`: Writes four values.
/// # Safety
/// - The current CPU must support AVX2 and FMA; indices must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_rdmk_same_m0_prepared_f64x4_const<
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const DD: usize,
>(
    w: &SameSpinView<'_, f64>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[f64], &[f64], usize),
    requests: &[([usize; K], [usize; K]); 4],
    tol: f64,
    out: &mut [f64; 4],
) {
    unsafe {
        xw_rdmk_same_m0_prepared_simd_const::<f64, F64x4, 4, K, RX, RW, L, D, DD>(
            w,
            ex,
            fundamental,
            requests,
            tol,
            out,
        );
    }
}

/// Evaluate eight real fixed-rank RDM values with AVX-512F.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: Real RDM outputs.
/// # Returns
/// - `()`: Writes eight values.
/// # Safety
/// - The current CPU must support AVX-512F; indices must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_rdmk_same_m0_prepared_f64x8_const<
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const DD: usize,
>(
    w: &SameSpinView<'_, f64>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[f64], &[f64], usize),
    requests: &[([usize; K], [usize; K]); 8],
    tol: f64,
    out: &mut [f64; 8],
) {
    unsafe {
        xw_rdmk_same_m0_prepared_simd_const::<f64, F64x8, 8, K, RX, RW, L, D, DD>(
            w,
            ex,
            fundamental,
            requests,
            tol,
            out,
        );
    }
}

/// Evaluate four complex fixed-rank RDM values with AVX2/FMA.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: Complex RDM outputs.
/// # Returns
/// - `()`: Writes four values.
/// # Safety
/// - The current CPU must support AVX2 and FMA; indices must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_rdmk_same_m0_prepared_c64x4_const<
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const DD: usize,
>(
    w: &SameSpinView<'_, Complex64>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[Complex64], &[Complex64], usize),
    requests: &[([usize; K], [usize; K]); 4],
    tol: f64,
    out: &mut [Complex64; 4],
) {
    unsafe {
        xw_rdmk_same_m0_prepared_simd_const::<Complex64, C64x4, 4, K, RX, RW, L, D, DD>(
            w,
            ex,
            fundamental,
            requests,
            tol,
            out,
        );
    }
}

/// Evaluate eight complex fixed-rank RDM values with AVX-512F.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: Complex RDM outputs.
/// # Returns
/// - `()`: Writes eight values.
/// # Safety
/// - The current CPU must support AVX-512F; indices must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_rdmk_same_m0_prepared_c64x8_const<
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const DD: usize,
>(
    w: &SameSpinView<'_, Complex64>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[Complex64], &[Complex64], usize),
    requests: &[([usize; K], [usize; K]); 8],
    tol: f64,
    out: &mut [Complex64; 8],
) {
    unsafe {
        xw_rdmk_same_m0_prepared_simd_const::<Complex64, C64x8, 8, K, RX, RW, L, D, DD>(
            w,
            ex,
            fundamental,
            requests,
            tol,
            out,
        );
    }
}

/// Evaluate a same-spin rank-`K` RDM request batch through the scalar prepared path.
/// Each request is the constrained determinant sum
/// `{}^{xw}\tilde S\sum_{\sum_i m_i = m}\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, optional `m_i = 1` branches, and matrix rank.
/// - `requests`: Creation and annihilation index arrays in output order.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// - `out`: Same-spin RDM elements in request order.
/// # Returns
/// - `()`: Writes the evaluated requests into `out`.
#[allow(clippy::type_complexity)]
fn xw_rdmk_same_prepared_scalar_batch<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], Option<(&[T], &[T])>, usize),
    requests: &[([usize; K], [usize; K])],
    scratch: &mut WickScratch<T>,
    tol: f64,
    out: &mut [T],
) {
    // Scalar requests share transformed fundamental contractions but build distinct external-index
    // rows and columns in `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
    for (request, value) in requests.iter().zip(out.iter_mut()) {
        *value = xw_rdmk_same_prepared_scalar_value(w, ex, fundamental, request, scratch, tol);
    }
}

/// Evaluate one same-spin rank-`K` RDM request through the scalar prepared path.
/// The `m = 0` branch dispatches `(K,RX,RW,L,D)` to a const-generic determinant; the general
/// branch sums all binary contraction-column assignments satisfying `\sum_i m_i = m`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, optional `m_i = 1` branches, and matrix rank.
/// - `request`: Const-sized creation and annihilation index arrays.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// # Returns
/// - `T`: Unnormalised same-spin rank-`K` transition-density element.
#[allow(clippy::type_complexity)]
#[inline(always)]
fn xw_rdmk_same_prepared_scalar_value<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], Option<(&[T], &[T])>, usize),
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    let (x_ex, w_ex) = ex;
    let rx = x_ex.holes.count_ones() as usize;
    let rw = w_ex.holes.count_ones() as usize;
    let l = rx + rw;

    // Augmented determinant has `K + L` contractions, so no assignment can absorb `m > K + L`
    // zero-overlap pairs.
    if w.m > l + K {
        return <T as From<f64>>::from(0.0);
    }

    // `m = 0` needs one determinant; `m > 0` sums all binary column assignments with
    // `\sum_i m_i = m`.
    if w.m == 0 {
        xw_rdmk_same_m0_prepared(w, ex, fundamental, request, scratch, tol)
    } else {
        xw_rdmk_same_gen_prepared(w, ex, fundamental, request, scratch, tol)
    }
}

/// Evaluate one same-spin rank-`K` RDM element when every contraction carries `m_i = 0`:
/// `{}^{xw}\Gamma_\sigma{}^{\mathbf p}_{\mathbf q}`
/// ` = {}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)`.
/// Supported `(K,RX,RW)` tuples dispatch to the const-generic determinant of dimension
/// `D = K + RX + RW`; arbitrary ranks use the scalar generic fallback.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, unused branch marker, and matrix rank.
/// - `request`: Const-sized creation and annihilation index arrays.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to the determinant contribution.
/// # Returns
/// - `T`: Unnormalised same-spin rank-`K` transition-density element for `m = 0`.
#[allow(clippy::type_complexity)]
#[inline(always)]
fn xw_rdmk_same_m0_prepared<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], Option<(&[T], &[T])>, usize),
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_prepared,
        {
            // Dispatch by `(K,R_x,R_w)` so generated dimensions use stack-allocated determinants;
            // arbitrary ranks preserve the same augmented determinant in runtime storage.
            let rx = ex.0.holes.count_ones() as usize;
            let rw = ex.1.holes.count_ones() as usize;
            dispatch_rdm_scalar_ranks!(
                K,
                (rx, rw),
                |K, RX, RW, L, D, DD| xw_rdmk_same_m0_prepared_const::<T, K, RX, RW, L, D, DD>(
                    w,
                    ex,
                    (fundamental.0, fundamental.1, fundamental.3),
                    // SAFETY: `dispatch_rdm_scalar_ranks!` selects this arm only when the caller's const
                    // `K` equals the arm-local literal `K`, so the two array-reference layouts are
                    // identical.
                    unsafe { &*std::ptr::from_ref(request).cast::<([usize; K], [usize; K])>() },
                    scratch,
                    tol,
                ),
                xw_rdmk_same_m0_gen_prepared(
                    w,
                    ex,
                    (fundamental.0, fundamental.1, fundamental.3),
                    request,
                    scratch,
                    tol,
                ),
            )
        }
    )
}

/// Evaluate one fixed-rank same-spin rank-`K` RDM determinant for `m = 0`.
/// The augmented matrix has dimension `D = K + L`, with `X^{(0)}` on and below the diagonal and in
/// the external `K x K` block, and `Y^{(0)}` elsewhere above it, so the external operators form
/// the creators-first string `a^\dagger_{p_1}\cdots a^\dagger_{p_K} a_{q_K}\cdots a_{q_1}`. Its
/// first `K` labels are external RDM indices; its remaining labels use the
/// contraction-space ordering `V_x \cup O_w` by `O_x \cup V_w` for ranks `(RX,RW)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, and their row dimension.
/// - `request`: Const-sized creation and annihilation index arrays.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to the determinant contribution.
/// # Returns
/// - `T`: `{}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
#[inline(always)]
fn xw_rdmk_same_m0_prepared_const<
    T: NOCIScalar,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const DD: usize,
>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_prepared_const,
        {
            // Place external RDM labels first, followed by bra and ket excitation labels. This is
            // the operator ordering defining `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
            scratch.ensure_same(D);
            let rows = scratch.rows.as_mut_slice();
            let cols = scratch.cols.as_mut_slice();
            for i in 0..K {
                rows[i] = w.nmo + request.0[i];
                cols[i] = w.nmo + request.1[i];
            }

            let (x_ex, w_ex) = ex;
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;
            let mut x_holes = x_ex.holes;
            let mut x_parts = x_ex.parts;
            for i in 0..RX {
                cols[K + i] = x_holes.trailing_zeros() as usize;
                rows[K + i] = x_parts.trailing_zeros() as usize - nocc;
                x_holes &= x_holes - 1;
                x_parts &= x_parts - 1;
            }
            let mut w_holes = w_ex.holes;
            let mut w_parts = w_ex.parts;
            for i in 0..RW {
                rows[K + RX + i] = nvirt + w_holes.trailing_zeros() as usize;
                cols[K + RX + i] = w_parts.trailing_zeros() as usize;
                w_holes &= w_holes - 1;
                w_parts &= w_parts - 1;
            }

            let (x0, y0, ext_n) = fundamental;
            let d = &mut scratch.det0.as_mut_slice()[..DD];

            // Build augmented `m_i = 0` determinant `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
            // Entries use `X^{(0)}` where the row creator stands left of the column annihilator in the
            // operator string, and `Y^{(0)}` otherwise. The external string is creators first,
            // `a^\dagger_{p_1}\cdots a^\dagger_{p_K} a_{q_K}\cdots a_{q_1}`, so its whole `K x K` block uses
            // `X^{(0)}`; the remaining entries take `X^{(0)}` on and below the diagonal.
            for i in 0..D {
                let row = rows[i] * ext_n;
                for j in 0..D {
                    d[i * D + j] = if i >= j || j < K {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            // Apply reduced reference overlap and numerical threshold to determinant contribution.
            let zero = <T as From<f64>>::from(0.0);
            let value = det_const::<T, D, DD>(d);
            if value.abs() > tol {
                w.phase * <T as From<f64>>::from(w.tilde_s_prod) * value
            } else {
                zero
            }
        }
    )
}

/// Evaluate one same-spin rank-`K` `m = 0` element outside the const-dispatch table.
/// This computes `{}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}` with runtime
/// augmented dimension `D = K + RX + RW` and the same fill convention: `X` on and below the
/// diagonal and in the creators-first external block, `Y` elsewhere above it.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, and their row dimension.
/// - `request`: Const-sized creation and annihilation index arrays.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to the determinant contribution.
/// # Returns
/// - `T`: Unnormalised same-spin rank-`K` transition-density element for `m = 0`.
#[inline(always)]
fn xw_rdmk_same_m0_gen_prepared<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_gen_prepared,
        {
            // Runtime-rank path uses same external-first label ordering as fixed-rank evaluator.
            let l = ex.0.holes.count_ones() as usize + ex.1.holes.count_ones() as usize;
            let d_rank = K + l;
            scratch.ensure_same(d_rank);
            let rows = scratch.rows.as_mut_slice();
            let cols = scratch.cols.as_mut_slice();
            for i in 0..K {
                rows[i] = w.nmo + request.0[i];
                cols[i] = w.nmo + request.1[i];
            }
            construct_determinant_indices(ex.0, ex.1, w, &mut rows[K..], &mut cols[K..]);

            let (x0, y0, ext_n) = fundamental;
            let d = scratch.det0.as_mut_slice();

            // Construct `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)` at runtime, with
            // the creators-first external block entirely in `X^{(0)}`.
            for i in 0..d_rank {
                let row = rows[i] * ext_n;
                for j in 0..d_rank {
                    d[i * d_rank + j] = if i >= j || j < K {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            // Multiply accepted determinant by `{}^{xw}\tilde S`.
            let zero = <T as From<f64>>::from(0.0);
            if let Some(value) = det_dynamic(d, d_rank)
                && value.abs() > tol
            {
                w.phase * <T as From<f64>>::from(w.tilde_s_prod) * value
            } else {
                zero
            }
        }
    )
}

/// Evaluate one same-spin rank-`K` RDM element for `m > 0`.
/// For augmented rank `D = K + L`, this sums every determinant obtained by selecting column `i`
/// from the `m_i = 0` or `m_i = 1` endpoint according to each binary assignment satisfying
/// `m_1 + \cdots + m_D = m`, then multiplies by `{}^{xw}\tilde S`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates containing zero-overlap orbital pairs.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended endpoint contractions and their row dimension.
/// - `request`: Const-sized creation and annihilation index arrays.
/// - `scratch`: Reusable endpoint and mixed-determinant storage.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// # Returns
/// - `T`: Constrained determinant sum multiplied by the reduced reference overlap.
#[allow(clippy::type_complexity)]
#[inline(always)]
fn xw_rdmk_same_gen_prepared<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], Option<(&[T], &[T])>, usize),
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_gen_prepared,
        {
            // Build external-first labels for augmented rank `D = K + L`.
            let l = ex.0.holes.count_ones() as usize + ex.1.holes.count_ones() as usize;
            let d_rank = K + l;
            scratch.ensure_same(d_rank);
            let rows = scratch.rows.as_mut_slice();
            let cols = scratch.cols.as_mut_slice();
            for i in 0..K {
                rows[i] = w.nmo + request.0[i];
                cols[i] = w.nmo + request.1[i];
            }
            construct_determinant_indices(ex.0, ex.1, w, &mut rows[K..], &mut cols[K..]);

            let (x0, y0, one, ext_n) = fundamental;
            let (x1, y1) = one.unwrap_or((x0, y0));

            // Prepare all-`m_i=0` and all-`m_i=1` endpoint determinants. Each allowed distribution
            // later selects whole columns from these endpoints. The creators-first external
            // block lies entirely in `X`.
            for i in 0..d_rank {
                let row = rows[i] * ext_n;
                for j in 0..d_rank {
                    let index = i * d_rank + j;
                    if i >= j || j < K {
                        scratch.det0.as_mut_slice()[index] = x0[row + cols[j]];
                        scratch.det1.as_mut_slice()[index] = x1[row + cols[j]];
                    } else {
                        scratch.det0.as_mut_slice()[index] = y0[row + cols[j]];
                        scratch.det1.as_mut_slice()[index] = y1[row + cols[j]];
                    }
                }
            }

            let zero = <T as From<f64>>::from(0.0);
            let mut acc = zero;

            // Sum `\det\mathbf D_{\mathrm{RDM}}(m_1,\ldots,m_D)` over every binary assignment
            // satisfying `\sum_i m_i = m`.
            for_each_m_combination(d_rank, w.m, |bits| {
                mix_columns_dynamic(
                    scratch.det_mix.as_mut_slice(),
                    scratch.det0.as_slice(),
                    scratch.det1.as_slice(),
                    d_rank,
                    bits,
                );
                if let Some(value) = det_dynamic(scratch.det_mix.as_slice(), d_rank)
                    && value.abs() > tol
                {
                    acc += value;
                }
            });
            w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
        }
    )
}
