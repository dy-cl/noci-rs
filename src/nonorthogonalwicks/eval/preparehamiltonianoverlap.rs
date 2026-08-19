#![allow(clippy::too_many_arguments)]

// nonorthogonalwicks/eval/preparehamiltonianoverlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256d, __m512d, _mm256_add_pd, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_fnmadd_pd,
    _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    _mm256_sub_pd, _mm512_add_pd, _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_fnmadd_pd,
    _mm512_loadu_pd, _mm512_mul_pd, _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd,
    _mm512_sub_pd,
};

// Crate-root imports.
use crate::maths::{adjugate_transpose, det};
use crate::noci::NOCIScalar;
use crate::{DetState, Excitation, ExcitationCache, ReducedTwoSpinDetState};

// Parent/sibling imports.
use super::super::scratch::WickScratchSpin;
use super::super::view::WicksPairView;
use super::helpers::{DetBranches, DetIndex, Minor, ReplacementLayout};
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, column_replacement_det,
    get_det_adjt_diff, ii_replacement, j_replacement, jslot, minor_adjt, mix_dets_same,
};
use super::prepare::prepare_same;

/// Evaluate the Hamiltonian and overlap matrix elements between two determinants generated from
/// one ordered pair of nonorthogonal references.
/// For `m_\alpha = m_\beta = 0`, the fixed path is used when every individual spin excitation
/// rank is at most four and `L_\alpha + L_\beta <= 6`, where
/// `L_\sigma = L_{x,\sigma} + L_{w,\sigma}`. This covers every pair for which each
/// determinant has total excitation order at most three, and also higher-order pairs that
/// satisfy the same fixed-rank conditions.
/// Other `m = 0` cases use the generic fused cofactor kernel. Cases with
/// `m_\alpha > 0` or `m_\beta > 0` use the generic fused distribution kernel, so this evaluator
/// imposes no rank cutoff beyond the underlying excitation representation.
/// The fixed path evaluates `S`, `H_1`, `H_{2,\alpha\alpha}`, `H_{2,\beta\beta}` and
/// `H_{2,\alpha\beta}` together so determinants, cofactors and second minors are reused.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Full bra excitation used by the generic fallback.
/// - `w_ex`: Full ket excitation used by the generic fallback.
/// - `x_cache`: Predecoded bra excitation ranks and the first four orbital labels per spin.
/// - `w_cache`: Predecoded ket excitation ranks and the first four orbital labels per spin.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `scratch`: Reusable Wick workspace for generic-rank and nonzero-`m` evaluation.
/// - `tol`: Numerical tolerance used by generic determinant and adjugate evaluation.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
pub(crate) fn xw_hamiltonian_overlap_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &Excitation,
    w_ex: &Excitation,
    x_cache: &ExcitationCache,
    w_cache: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
    scratch: &mut WickScratchSpin<T>,
    tol: f64,
) -> (T, T) {
    if w.aa.m == 0 && w.bb.m == 0 {
        let fixed = x_cache.alpha.rank <= 4
            && x_cache.beta.rank <= 4
            && w_cache.alpha.rank <= 4
            && w_cache.beta.rank <= 4;

        if fixed {
            // The fixed path reads only predecoded ranks and orbital labels.
            // Raw excitation masks are touched only by the arbitrary-rank fallback below.
            let la = usize::from(x_cache.alpha.rank) + usize::from(w_cache.alpha.rank);
            let lb = usize::from(x_cache.beta.rank) + usize::from(w_cache.beta.rank);

            if la + lb <= 6 {
                return xw_hamiltonian_overlap_m0_prepared(
                    w,
                    la,
                    lb,
                    x_cache,
                    w_cache,
                    excitation_phase,
                    enuc,
                );
            }
        }

        return xw_hamiltonian_overlap_m0_gen_prepared(
            w,
            x_ex,
            w_ex,
            excitation_phase,
            enuc,
            scratch,
            tol,
        );
    }

    xw_hamiltonian_overlap_gen_prepared(w, x_ex, w_ex, excitation_phase, enuc, scratch, tol)
}

/// Evaluate batched Hamiltonian and overlap matrix elements for one ordered reference pair.
/// Every request supplied to this routine already belongs to that reference pair. Requests are
/// streamed through the 28 fixed `(L_\alpha, L_\beta)` bins when
/// `m_\alpha = m_\beta = 0`. The widest supported real SIMD kernel is selected internally,
/// incomplete bins are padded with one valid request, and unsupported requests use the existing
/// prepared scalar evaluator.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `basis`: Determinant basis used only by generic fallback evaluation.
/// - `reduced_basis`: Compact two-spin metadata keyed by global determinant index.
/// - `requests`: Tuples `(output, a, b)` containing output position and determinant indices.
/// - `enuc`: Nuclear repulsion energy.
/// - `scratch`: Reusable Wick workspace for scalar generic-rank evaluation.
/// - `tol`: Numerical tolerance used by generic determinant and adjugate evaluation.
/// - `out`: Hamiltonian and overlap results aligned with the original request order.
/// # Returns:
/// - `()`: Writes every matrix element in `requests` into `out`.
pub(crate) fn xw_hamiltonian_overlap_prepared_batched(
    w: &WicksPairView<'_, f64>,
    basis: &[DetState<f64>],
    reduced_basis: &[ReducedTwoSpinDetState],
    requests: &[(usize, usize, usize)],
    enuc: f64,
    scratch: &mut WickScratchSpin<f64>,
    tol: f64,
    out: &mut [(f64, f64)],
) {
    #[cfg(target_arch = "x86_64")]
    if w.aa.m == 0 && w.bb.m == 0 {
        if std::arch::is_x86_feature_detected!("avx512f") {
            let mut x_bins = [[ExcitationCache::default(); 8]; 28];
            let mut w_bins = [[ExcitationCache::default(); 8]; 28];
            let mut phases = [[1.0f64; 8]; 28];
            let mut outputs = [[0usize; 8]; 28];
            let mut counts = [0usize; 28];

            for &(output, a, b) in requests {
                let x_det = &reduced_basis[a];
                let w_det = &reduced_basis[b];
                let x_cache = x_det.excitation_cache;
                let w_cache = w_det.excitation_cache;
                let fixed = x_cache.alpha.rank <= 4
                    && x_cache.beta.rank <= 4
                    && w_cache.alpha.rank <= 4
                    && w_cache.beta.rank <= 4;
                let la = usize::from(x_cache.alpha.rank) + usize::from(w_cache.alpha.rank);
                let lb = usize::from(x_cache.beta.rank) + usize::from(w_cache.beta.rank);

                if fixed && la + lb <= 6 {
                    let bin = la * (15 - la) / 2 + lb;
                    let count = counts[bin];

                    x_bins[bin][count] = x_cache;
                    w_bins[bin][count] = w_cache;
                    phases[bin][count] = x_det.phase * w_det.phase;
                    outputs[bin][count] = output;
                    counts[bin] += 1;

                    if counts[bin] == 8 {
                        let mut h = [0.0f64; 8];
                        let mut s = [0.0f64; 8];

                        unsafe {
                            xw_hamiltonian_overlap_m0_prepared_f64x8(
                                w,
                                la,
                                lb,
                                &x_bins[bin],
                                &w_bins[bin],
                                &phases[bin],
                                enuc,
                                &mut h,
                                &mut s,
                            );
                        }

                        for lane in 0..8 {
                            out[outputs[bin][lane]] = (h[lane], s[lane]);
                        }
                        counts[bin] = 0;
                    }
                } else {
                    let x_state = &basis[a];
                    let w_state = &basis[b];

                    out[output] = xw_hamiltonian_overlap_prepared(
                        w,
                        &x_state.excitation,
                        &w_state.excitation,
                        &x_cache,
                        &w_cache,
                        x_det.phase * w_det.phase,
                        enuc,
                        scratch,
                        tol,
                    );
                }
            }

            for la in 0..=6 {
                for lb in 0..=(6 - la) {
                    let bin = la * (15 - la) / 2 + lb;
                    let count = counts[bin];
                    if count == 0 {
                        continue;
                    }

                    let fill_x = x_bins[bin][0];
                    let fill_w = w_bins[bin][0];
                    let fill_phase = phases[bin][0];

                    for lane in count..8 {
                        x_bins[bin][lane] = fill_x;
                        w_bins[bin][lane] = fill_w;
                        phases[bin][lane] = fill_phase;
                    }

                    let mut h = [0.0f64; 8];
                    let mut s = [0.0f64; 8];

                    unsafe {
                        xw_hamiltonian_overlap_m0_prepared_f64x8(
                            w,
                            la,
                            lb,
                            &x_bins[bin],
                            &w_bins[bin],
                            &phases[bin],
                            enuc,
                            &mut h,
                            &mut s,
                        );
                    }

                    for lane in 0..count {
                        out[outputs[bin][lane]] = (h[lane], s[lane]);
                    }
                }
            }
            return;
        }

        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            let mut x_bins = [[ExcitationCache::default(); 4]; 28];
            let mut w_bins = [[ExcitationCache::default(); 4]; 28];
            let mut phases = [[1.0f64; 4]; 28];
            let mut outputs = [[0usize; 4]; 28];
            let mut counts = [0usize; 28];

            for &(output, a, b) in requests {
                let x_det = &reduced_basis[a];
                let w_det = &reduced_basis[b];
                let x_cache = x_det.excitation_cache;
                let w_cache = w_det.excitation_cache;
                let fixed = x_cache.alpha.rank <= 4
                    && x_cache.beta.rank <= 4
                    && w_cache.alpha.rank <= 4
                    && w_cache.beta.rank <= 4;
                let la = usize::from(x_cache.alpha.rank) + usize::from(w_cache.alpha.rank);
                let lb = usize::from(x_cache.beta.rank) + usize::from(w_cache.beta.rank);

                if fixed && la + lb <= 6 {
                    let bin = la * (15 - la) / 2 + lb;
                    let count = counts[bin];

                    x_bins[bin][count] = x_cache;
                    w_bins[bin][count] = w_cache;
                    phases[bin][count] = x_det.phase * w_det.phase;
                    outputs[bin][count] = output;
                    counts[bin] += 1;

                    if counts[bin] == 4 {
                        let mut h = [0.0f64; 4];
                        let mut s = [0.0f64; 4];

                        unsafe {
                            xw_hamiltonian_overlap_m0_prepared_f64x4(
                                w,
                                la,
                                lb,
                                &x_bins[bin],
                                &w_bins[bin],
                                &phases[bin],
                                enuc,
                                &mut h,
                                &mut s,
                            );
                        }

                        for lane in 0..4 {
                            out[outputs[bin][lane]] = (h[lane], s[lane]);
                        }
                        counts[bin] = 0;
                    }
                } else {
                    let x_state = &basis[a];
                    let w_state = &basis[b];

                    out[output] = xw_hamiltonian_overlap_prepared(
                        w,
                        &x_state.excitation,
                        &w_state.excitation,
                        &x_cache,
                        &w_cache,
                        x_det.phase * w_det.phase,
                        enuc,
                        scratch,
                        tol,
                    );
                }
            }

            for la in 0..=6 {
                for lb in 0..=(6 - la) {
                    let bin = la * (15 - la) / 2 + lb;
                    let count = counts[bin];
                    if count == 0 {
                        continue;
                    }

                    let fill_x = x_bins[bin][0];
                    let fill_w = w_bins[bin][0];
                    let fill_phase = phases[bin][0];

                    for lane in count..4 {
                        x_bins[bin][lane] = fill_x;
                        w_bins[bin][lane] = fill_w;
                        phases[bin][lane] = fill_phase;
                    }

                    let mut h = [0.0f64; 4];
                    let mut s = [0.0f64; 4];

                    unsafe {
                        xw_hamiltonian_overlap_m0_prepared_f64x4(
                            w,
                            la,
                            lb,
                            &x_bins[bin],
                            &w_bins[bin],
                            &phases[bin],
                            enuc,
                            &mut h,
                            &mut s,
                        );
                    }

                    for lane in 0..count {
                        out[outputs[bin][lane]] = (h[lane], s[lane]);
                    }
                }
            }
            return;
        }
    }

    for &(output, a, b) in requests {
        let x_det = &reduced_basis[a];
        let w_det = &reduced_basis[b];
        let x_state = &basis[a];
        let w_state = &basis[b];

        out[output] = xw_hamiltonian_overlap_prepared(
            w,
            &x_state.excitation,
            &w_state.excitation,
            &x_det.excitation_cache,
            &w_det.excitation_cache,
            x_det.phase * w_det.phase,
            enuc,
            scratch,
            tol,
        );
    }
}

/// Dispatch an `m_\alpha = m_\beta = 0` Hamiltonian and overlap matrix element to a fixed
/// contraction-rank kernel.
/// The specialised region contains all `(L_\alpha, L_\beta)` pairs with
/// `L_\alpha + L_\beta <= 6`. The caller has already established that each individual
/// predecoded spin excitation has rank at most four.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `la`: Total alpha-spin contraction rank `L_\alpha = L_{x,\alpha} + L_{w,\alpha}`.
/// - `lb`: Total beta-spin contraction rank `L_\beta = L_{x,\beta} + L_{w,\beta}`.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    la: usize,
    lb: usize,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    match (la, lb) {
        (0, 0) => xw_hamiltonian_overlap_m0_00_prepared(w, excitation_phase, enuc),
        (0, 1) => xw_hamiltonian_overlap_m0_01_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (0, 2) => xw_hamiltonian_overlap_m0_02_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (0, 3) => xw_hamiltonian_overlap_m0_03_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (0, 4) => xw_hamiltonian_overlap_m0_04_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (0, 5) => xw_hamiltonian_overlap_m0_05_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (0, 6) => xw_hamiltonian_overlap_m0_06_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (1, 0) => xw_hamiltonian_overlap_m0_10_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (1, 1) => xw_hamiltonian_overlap_m0_11_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (1, 2) => xw_hamiltonian_overlap_m0_12_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (1, 3) => xw_hamiltonian_overlap_m0_13_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (1, 4) => xw_hamiltonian_overlap_m0_14_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (1, 5) => xw_hamiltonian_overlap_m0_15_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (2, 0) => xw_hamiltonian_overlap_m0_20_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (2, 1) => xw_hamiltonian_overlap_m0_21_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (2, 2) => xw_hamiltonian_overlap_m0_22_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (2, 3) => xw_hamiltonian_overlap_m0_23_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (2, 4) => xw_hamiltonian_overlap_m0_24_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (3, 0) => xw_hamiltonian_overlap_m0_30_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (3, 1) => xw_hamiltonian_overlap_m0_31_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (3, 2) => xw_hamiltonian_overlap_m0_32_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (3, 3) => xw_hamiltonian_overlap_m0_33_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (4, 0) => xw_hamiltonian_overlap_m0_40_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (4, 1) => xw_hamiltonian_overlap_m0_41_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (4, 2) => xw_hamiltonian_overlap_m0_42_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (5, 0) => xw_hamiltonian_overlap_m0_50_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (5, 1) => xw_hamiltonian_overlap_m0_51_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        (6, 0) => xw_hamiltonian_overlap_m0_60_prepared(w, x_ex, w_ex, excitation_phase, enuc),
        _ => unreachable!(),
    }
}

/// Dispatch 4 independent real `m_\alpha = m_\beta = 0` matrix elements to a fixed-rank
/// AVX2/FMA kernel.
/// Every SIMD lane uses the same ordered reference pair and `(L_\alpha, L_\beta)`, while the
/// orbital labels and excitation phases may differ between lanes.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `la`: Alpha-spin contraction rank shared by every lane.
/// - `lb`: Beta-spin contraction rank shared by every lane.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, individual predecoded spin
///   ranks no larger than four, `L_\alpha + L_\beta <= 6` and output slices of length at
///   least four.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn xw_hamiltonian_overlap_m0_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    la: usize,
    lb: usize,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        match (la, lb) {
            (0, 0) => xw_hamiltonian_overlap_m0_00_prepared_f64x4(w, excitation_phase, enuc, h, s),
            (0, 1) => xw_hamiltonian_overlap_m0_01_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 2) => xw_hamiltonian_overlap_m0_02_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 3) => xw_hamiltonian_overlap_m0_03_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 4) => xw_hamiltonian_overlap_m0_04_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 5) => xw_hamiltonian_overlap_m0_05_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 6) => xw_hamiltonian_overlap_m0_06_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 0) => xw_hamiltonian_overlap_m0_10_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 1) => xw_hamiltonian_overlap_m0_11_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 2) => xw_hamiltonian_overlap_m0_12_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 3) => xw_hamiltonian_overlap_m0_13_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 4) => xw_hamiltonian_overlap_m0_14_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 5) => xw_hamiltonian_overlap_m0_15_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 0) => xw_hamiltonian_overlap_m0_20_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 1) => xw_hamiltonian_overlap_m0_21_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 2) => xw_hamiltonian_overlap_m0_22_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 3) => xw_hamiltonian_overlap_m0_23_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 4) => xw_hamiltonian_overlap_m0_24_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 0) => xw_hamiltonian_overlap_m0_30_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 1) => xw_hamiltonian_overlap_m0_31_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 2) => xw_hamiltonian_overlap_m0_32_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 3) => xw_hamiltonian_overlap_m0_33_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (4, 0) => xw_hamiltonian_overlap_m0_40_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (4, 1) => xw_hamiltonian_overlap_m0_41_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (4, 2) => xw_hamiltonian_overlap_m0_42_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (5, 0) => xw_hamiltonian_overlap_m0_50_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (5, 1) => xw_hamiltonian_overlap_m0_51_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (6, 0) => xw_hamiltonian_overlap_m0_60_prepared_f64x4(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            _ => unreachable!(),
        }
    }
}

/// Dispatch 8 independent real `m_\alpha = m_\beta = 0` matrix elements to a fixed-rank
/// AVX-512 kernel.
/// Every SIMD lane uses the same ordered reference pair and `(L_\alpha, L_\beta)`, while the
/// orbital labels and excitation phases may differ between lanes.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `la`: Alpha-spin contraction rank shared by every lane.
/// - `lb`: Beta-spin contraction rank shared by every lane.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, individual predecoded spin
///   ranks no larger than four, `L_\alpha + L_\beta <= 6` and output slices of length at
///   least eight.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn xw_hamiltonian_overlap_m0_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    la: usize,
    lb: usize,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        match (la, lb) {
            (0, 0) => xw_hamiltonian_overlap_m0_00_prepared_f64x8(w, excitation_phase, enuc, h, s),
            (0, 1) => xw_hamiltonian_overlap_m0_01_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 2) => xw_hamiltonian_overlap_m0_02_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 3) => xw_hamiltonian_overlap_m0_03_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 4) => xw_hamiltonian_overlap_m0_04_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 5) => xw_hamiltonian_overlap_m0_05_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 6) => xw_hamiltonian_overlap_m0_06_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 0) => xw_hamiltonian_overlap_m0_10_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 1) => xw_hamiltonian_overlap_m0_11_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 2) => xw_hamiltonian_overlap_m0_12_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 3) => xw_hamiltonian_overlap_m0_13_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 4) => xw_hamiltonian_overlap_m0_14_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 5) => xw_hamiltonian_overlap_m0_15_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 0) => xw_hamiltonian_overlap_m0_20_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 1) => xw_hamiltonian_overlap_m0_21_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 2) => xw_hamiltonian_overlap_m0_22_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 3) => xw_hamiltonian_overlap_m0_23_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 4) => xw_hamiltonian_overlap_m0_24_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 0) => xw_hamiltonian_overlap_m0_30_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 1) => xw_hamiltonian_overlap_m0_31_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 2) => xw_hamiltonian_overlap_m0_32_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 3) => xw_hamiltonian_overlap_m0_33_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (4, 0) => xw_hamiltonian_overlap_m0_40_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (4, 1) => xw_hamiltonian_overlap_m0_41_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (4, 2) => xw_hamiltonian_overlap_m0_42_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (5, 0) => xw_hamiltonian_overlap_m0_50_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (5, 1) => xw_hamiltonian_overlap_m0_51_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (6, 0) => xw_hamiltonian_overlap_m0_60_prepared_f64x8(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            _ => unreachable!(),
        }
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (0, 0)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_00_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);
    let det_a = <T as From<f64>>::from(1.0);
    let j_a = zero;
    let replacement_a = zero;
    let det_b = <T as From<f64>>::from(1.0);
    let j_b = zero;
    let replacement_b = zero;
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_00_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm256_set1_pd(1.0);
        let j_a = _mm256_setzero_pd();
        let replacement_a = _mm256_setzero_pd();
        let det_b = _mm256_set1_pd(1.0);
        let j_b = _mm256_setzero_pd();
        let replacement_b = _mm256_setzero_pd();
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_00_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm512_set1_pd(1.0);
        let j_a = _mm512_setzero_pd();
        let replacement_a = _mm512_setzero_pd();
        let det_b = _mm512_set1_pd(1.0);
        let j_b = _mm512_setzero_pd();
        let replacement_b = _mm512_setzero_pd();
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (0, 1)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_01_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);
    let det_a = <T as From<f64>>::from(1.0);
    let j_a = zero;
    let replacement_a = zero;

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 1];
    let mut cols_b = [0usize; 1];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..1 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 1];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_b[i] * n_b;
        for j in 0..1 {
            d_b[i * 1 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_b = d_b[0];
    let j_b = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..1 {
        let base = cols_b[z] * n_b;
        for eta in 0..1 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += value;
        }
    }
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_01_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm256_set1_pd(1.0);
        let j_a = _mm256_setzero_pd();
        let replacement_a = _mm256_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 4];
        let mut cols_b = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm256_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_add_pd(replacement_b, values);
            }
        }
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_01_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm512_set1_pd(1.0);
        let j_a = _mm512_setzero_pd();
        let replacement_a = _mm512_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 8];
        let mut cols_b = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm512_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_add_pd(replacement_b, values);
            }
        }
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (0, 2)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_02_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);
    let det_a = <T as From<f64>>::from(1.0);
    let j_a = zero;
    let replacement_a = zero;

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 2];
    let mut cols_b = [0usize; 2];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..2 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 4];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..2 {
        let row = rows_b[i] * n_b;
        for j in 0..2 {
            d_b[i * 2 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 2`, all four cofactors are existing entries of \mathbf D and require no determinant
    // evaluation.
    let cof_b = [d_b[3], -d_b[2], -d_b[1], d_b[0]];
    let det_b = d_b[0] * cof_b[0] + d_b[1] * cof_b[1];
    let jsl_b = w.bb.j_slice(0);
    let r0 = rows_b[0];
    let r1 = rows_b[1];
    let c0 = cols_b[0];
    let c1 = cols_b[1];
    let direct = jsl_b[(((r0 * n_b + c0) * n_b + r1) * n_b) + c1];
    let exchange = jsl_b[(((r0 * n_b + c1) * n_b + r1) * n_b) + c0];
    let j_b = direct - exchange;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..2 {
        let base = cols_b[z] * n_b;
        for eta in 0..2 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 2 + z] * value;
        }
    }
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 2)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_02_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm256_set1_pd(1.0);
        let j_a = _mm256_setzero_pd();
        let replacement_a = _mm256_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 2]; 4];
        let mut cols_b = [[0usize; 2]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 4] = [_mm256_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 2 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_b: [__m256d; 4] = [
            d_b[3],
            _mm256_sub_pd(_mm256_setzero_pd(), d_b[2]),
            _mm256_sub_pd(_mm256_setzero_pd(), d_b[1]),
            d_b[0],
        ];
        let det_b = _mm256_fmsub_pd(d_b[0], d_b[3], _mm256_mul_pd(d_b[1], d_b[2]));
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut direct_lane = [0.0f64; 4];
        let mut exchange_lane = [0.0f64; 4];
        for lane in 0..4 {
            let r0 = rows_b[lane][0];
            let r1 = rows_b[lane][1];
            let c0 = cols_b[lane][0];
            let c1 = cols_b[lane][1];
            direct_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c0) * n_b + r1) * n_b) + c1);
            exchange_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c1) * n_b + r1) * n_b) + c0);
        }
        let j_b = _mm256_sub_pd(
            _mm256_loadu_pd(direct_lane.as_ptr()),
            _mm256_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 2 + z], values, replacement_b);
            }
        }
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 2)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_02_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm512_set1_pd(1.0);
        let j_a = _mm512_setzero_pd();
        let replacement_a = _mm512_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 2]; 8];
        let mut cols_b = [[0usize; 2]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 4] = [_mm512_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 2 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_b: [__m512d; 4] = [
            d_b[3],
            _mm512_sub_pd(_mm512_setzero_pd(), d_b[2]),
            _mm512_sub_pd(_mm512_setzero_pd(), d_b[1]),
            d_b[0],
        ];
        let det_b = _mm512_fmsub_pd(d_b[0], d_b[3], _mm512_mul_pd(d_b[1], d_b[2]));
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut direct_lane = [0.0f64; 8];
        let mut exchange_lane = [0.0f64; 8];
        for lane in 0..8 {
            let r0 = rows_b[lane][0];
            let r1 = rows_b[lane][1];
            let c0 = cols_b[lane][0];
            let c1 = cols_b[lane][1];
            direct_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c0) * n_b + r1) * n_b) + c1);
            exchange_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c1) * n_b + r1) * n_b) + c0);
        }
        let j_b = _mm512_sub_pd(
            _mm512_loadu_pd(direct_lane.as_ptr()),
            _mm512_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 2 + z], values, replacement_b);
            }
        }
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (0, 3)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_03_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);
    let det_a = <T as From<f64>>::from(1.0);
    let j_a = zero;
    let replacement_a = zero;

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 3];
    let mut cols_b = [0usize; 3];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..3 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 9];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..3 {
        let row = rows_b[i] * n_b;
        for j in 0..3 {
            d_b[i * 3 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `9` distinct second minors for `L = 3`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_b = [zero; 9];
    let jsl_b = w.bb.j_slice(0);
    let mut j_b = zero;
    for eta in 0..3 {
        for xi in (eta + 1)..3 {
            let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..3 {
                for y in (z + 1)..3 {
                    let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 1];
                    let mut ii = 0usize;
                    for r in 0..3 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..3 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 1 + jj] = d_b[r * 3 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0];
                    second_b[row_pair * 3 + col_pair] = second;
                    let r_eta = rows_b[eta];
                    let r_xi = rows_b[xi];
                    let c_z = cols_b[z];
                    let c_y = cols_b[y];
                    let direct = jsl_b[(((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y];
                    let exchange = jsl_b[(((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b += term;
                    } else {
                        j_b -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_b = [zero; 9];
    for eta in 0..3 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..3 {
            let mut value = zero;
            for c in 0..3 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_b[r * 3 + c] * second_b[row_pair * 3 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_b = d_b[0] * cof_b[0];
    for z in 1..3 {
        det_b += d_b[z] * cof_b[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..3 {
        let base = cols_b[z] * n_b;
        for eta in 0..3 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 3 + z] * value;
        }
    }
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 3)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_03_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm256_set1_pd(1.0);
        let j_a = _mm256_setzero_pd();
        let replacement_a = _mm256_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 3]; 4];
        let mut cols_b = [[0usize; 3]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 3 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm256_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 1] = [_mm256_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_b[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_b[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm256_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm256_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm256_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_b[r * 3 + c], second_b[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
        for z in 1..3 {
            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 3 + z], values, replacement_b);
            }
        }
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 3)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_03_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm512_set1_pd(1.0);
        let j_a = _mm512_setzero_pd();
        let replacement_a = _mm512_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 3]; 8];
        let mut cols_b = [[0usize; 3]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 3 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm512_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 1] = [_mm512_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_b[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_b[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm512_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm512_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm512_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_b[r * 3 + c], second_b[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
        for z in 1..3 {
            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 3 + z], values, replacement_b);
            }
        }
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (0, 4)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_04_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);
    let det_a = <T as From<f64>>::from(1.0);
    let j_a = zero;
    let replacement_a = zero;

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 4];
    let mut cols_b = [0usize; 4];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..4 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 16];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..4 {
        let row = rows_b[i] * n_b;
        for j in 0..4 {
            d_b[i * 4 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `36` distinct second minors for `L = 4`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_b = [zero; 36];
    let jsl_b = w.bb.j_slice(0);
    let mut j_b = zero;
    for eta in 0..4 {
        for xi in (eta + 1)..4 {
            let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..4 {
                for y in (z + 1)..4 {
                    let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 4];
                    let mut ii = 0usize;
                    for r in 0..4 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..4 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 2 + jj] = d_b[r * 4 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0] * minor[3] - minor[1] * minor[2];
                    second_b[row_pair * 6 + col_pair] = second;
                    let r_eta = rows_b[eta];
                    let r_xi = rows_b[xi];
                    let c_z = cols_b[z];
                    let c_y = cols_b[y];
                    let direct = jsl_b[(((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y];
                    let exchange = jsl_b[(((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b += term;
                    } else {
                        j_b -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_b = [zero; 16];
    for eta in 0..4 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..4 {
            let mut value = zero;
            for c in 0..4 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_b[r * 4 + c] * second_b[row_pair * 6 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_b[eta * 4 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_b = d_b[0] * cof_b[0];
    for z in 1..4 {
        det_b += d_b[z] * cof_b[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..4 {
        let base = cols_b[z] * n_b;
        for eta in 0..4 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 4 + z] * value;
        }
    }
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 4)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_04_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm256_set1_pd(1.0);
        let j_a = _mm256_setzero_pd();
        let replacement_a = _mm256_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 4]; 4];
        let mut cols_b = [[0usize; 4]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 16] = [_mm256_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 4 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m256d; 36] = [_mm256_setzero_pd(); 36];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm256_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 4] = [_mm256_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_b[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm256_fmsub_pd(minor[0], minor[3], _mm256_mul_pd(minor[1], minor[2]));
                        second_b[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm256_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm256_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m256d; 16] = [_mm256_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm256_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_b[r * 4 + c], second_b[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_b[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
        for z in 1..4 {
            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 4 + z], values, replacement_b);
            }
        }
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 4)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_04_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm512_set1_pd(1.0);
        let j_a = _mm512_setzero_pd();
        let replacement_a = _mm512_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 4]; 8];
        let mut cols_b = [[0usize; 4]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 16] = [_mm512_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 4 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m512d; 36] = [_mm512_setzero_pd(); 36];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm512_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 4] = [_mm512_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_b[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm512_fmsub_pd(minor[0], minor[3], _mm512_mul_pd(minor[1], minor[2]));
                        second_b[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm512_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm512_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m512d; 16] = [_mm512_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm512_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_b[r * 4 + c], second_b[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_b[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
        for z in 1..4 {
            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 4 + z], values, replacement_b);
            }
        }
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (0, 5)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_05_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);
    let det_a = <T as From<f64>>::from(1.0);
    let j_a = zero;
    let replacement_a = zero;

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 5];
    let mut cols_b = [0usize; 5];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..5 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 25];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..5 {
        let row = rows_b[i] * n_b;
        for j in 0..5 {
            d_b[i * 5 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // `L = 5` requires all `100` second minors because every one is multiplied by an independent
    // `\mathcal J` tensor coefficient.
    // Compute the `100` distinct `2 x 2` minors once first, then reuse them in every larger second
    // minor; this is the minimum minor-evaluation count for this compound-minor DAG.
    let mut minor2_b = [zero; 100];
    for r0 in 0..5 {
        for r1 in (r0 + 1)..5 {
            let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
            for c0 in 0..5 {
                for c1 in (c0 + 1)..5 {
                    let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                    minor2_b[row_pair * 10 + col_pair] =
                        d_b[r0 * 5 + c0] * d_b[r1 * 5 + c1] - d_b[r0 * 5 + c1] * d_b[r1 * 5 + c0];
                }
            }
        }
    }
    let mut second_b = [zero; 100];
    let jsl_b = w.bb.j_slice(0);
    let mut j_b = zero;
    for eta in 0..5 {
        for xi in (eta + 1)..5 {
            let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..5 {
                for y in (z + 1)..5 {
                    let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                    let mut retained_rows = [0usize; 3];
                    let mut retained_cols = [0usize; 3];
                    let mut nr = 0usize;
                    let mut nc = 0usize;
                    for r in 0..5 {
                        if r != eta && r != xi {
                            retained_rows[nr] = r;
                            nr += 1;
                        }
                    }
                    for c in 0..5 {
                        if c != z && c != y {
                            retained_cols[nc] = c;
                            nc += 1;
                        }
                    }

                    // Expand the `3 x 3` second minor through its first retained row using three
                    // precomputed `2 x 2` minors.
                    let r0 = retained_rows[0];
                    let r1 = retained_rows[1];
                    let r2 = retained_rows[2];
                    let c0 = retained_cols[0];
                    let c1 = retained_cols[1];
                    let c2 = retained_cols[2];
                    let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                    let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                    let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                    let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                    let second = d_b[r0 * 5 + c0] * minor2_b[rp12 * 10 + cp12]
                        - d_b[r0 * 5 + c1] * minor2_b[rp12 * 10 + cp02]
                        + d_b[r0 * 5 + c2] * minor2_b[rp12 * 10 + cp01];
                    second_b[row_pair * 10 + col_pair] = second;
                    let r_eta = rows_b[eta];
                    let r_xi = rows_b[xi];
                    let c_z = cols_b[z];
                    let c_y = cols_b[y];
                    let direct = jsl_b[(((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y];
                    let exchange = jsl_b[(((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b += term;
                    } else {
                        j_b -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_b = [zero; 25];
    for eta in 0..5 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..5 {
            let mut value = zero;
            for c in 0..5 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_b[r * 5 + c] * second_b[row_pair * 10 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_b[eta * 5 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_b = d_b[0] * cof_b[0];
    for z in 1..5 {
        det_b += d_b[z] * cof_b[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..5 {
        let base = cols_b[z] * n_b;
        for eta in 0..5 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 5 + z] * value;
        }
    }
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 5)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_05_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm256_set1_pd(1.0);
        let j_a = _mm256_setzero_pd();
        let replacement_a = _mm256_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 5]; 4];
        let mut cols_b = [[0usize; 5]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..5 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 25] = [_mm256_setzero_pd(); 25];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..5 {
            for j in 0..5 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 5 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // `L = 5` requires all `100` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_b: [__m256d; 100] = [_mm256_setzero_pd(); 100];
        for r0 in 0..5 {
            for r1 in (r0 + 1)..5 {
                let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..5 {
                    for c1 in (c0 + 1)..5 {
                        let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_b[row_pair * 10 + col_pair] = _mm256_fmsub_pd(
                            d_b[r0 * 5 + c0],
                            d_b[r1 * 5 + c1],
                            _mm256_mul_pd(d_b[r0 * 5 + c1], d_b[r1 * 5 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_b: [__m256d; 100] = [_mm256_setzero_pd(); 100];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm256_setzero_pd();
        for eta in 0..5 {
            for xi in (eta + 1)..5 {
                let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..5 {
                    for y in (z + 1)..5 {
                        let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 3];
                        let mut retained_cols = [0usize; 3];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..5 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..5 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                        let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let first = _mm256_fmsub_pd(
                            d_b[r0 * 5 + c0],
                            minor2_b[rp12 * 10 + cp12],
                            _mm256_mul_pd(d_b[r0 * 5 + c1], minor2_b[rp12 * 10 + cp02]),
                        );
                        let second =
                            _mm256_fmadd_pd(d_b[r0 * 5 + c2], minor2_b[rp12 * 10 + cp01], first);
                        second_b[row_pair * 10 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm256_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm256_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m256d; 25] = [_mm256_setzero_pd(); 25];
        for eta in 0..5 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..5 {
                let mut value = _mm256_setzero_pd();
                for c in 0..5 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_b[r * 5 + c], second_b[row_pair * 10 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_b[eta * 5 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
        for z in 1..5 {
            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..5 {
            for eta in 0..5 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 5 + z], values, replacement_b);
            }
        }
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 5)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_05_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm512_set1_pd(1.0);
        let j_a = _mm512_setzero_pd();
        let replacement_a = _mm512_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 5]; 8];
        let mut cols_b = [[0usize; 5]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..5 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 25] = [_mm512_setzero_pd(); 25];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..5 {
            for j in 0..5 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 5 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // `L = 5` requires all `100` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_b: [__m512d; 100] = [_mm512_setzero_pd(); 100];
        for r0 in 0..5 {
            for r1 in (r0 + 1)..5 {
                let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..5 {
                    for c1 in (c0 + 1)..5 {
                        let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_b[row_pair * 10 + col_pair] = _mm512_fmsub_pd(
                            d_b[r0 * 5 + c0],
                            d_b[r1 * 5 + c1],
                            _mm512_mul_pd(d_b[r0 * 5 + c1], d_b[r1 * 5 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_b: [__m512d; 100] = [_mm512_setzero_pd(); 100];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm512_setzero_pd();
        for eta in 0..5 {
            for xi in (eta + 1)..5 {
                let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..5 {
                    for y in (z + 1)..5 {
                        let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 3];
                        let mut retained_cols = [0usize; 3];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..5 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..5 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                        let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let first = _mm512_fmsub_pd(
                            d_b[r0 * 5 + c0],
                            minor2_b[rp12 * 10 + cp12],
                            _mm512_mul_pd(d_b[r0 * 5 + c1], minor2_b[rp12 * 10 + cp02]),
                        );
                        let second =
                            _mm512_fmadd_pd(d_b[r0 * 5 + c2], minor2_b[rp12 * 10 + cp01], first);
                        second_b[row_pair * 10 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm512_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm512_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m512d; 25] = [_mm512_setzero_pd(); 25];
        for eta in 0..5 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..5 {
                let mut value = _mm512_setzero_pd();
                for c in 0..5 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_b[r * 5 + c], second_b[row_pair * 10 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_b[eta * 5 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
        for z in 1..5 {
            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..5 {
            for eta in 0..5 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 5 + z], values, replacement_b);
            }
        }
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (0, 6)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_06_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);
    let det_a = <T as From<f64>>::from(1.0);
    let j_a = zero;
    let replacement_a = zero;

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 6];
    let mut cols_b = [0usize; 6];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..6 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 36];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..6 {
        let row = rows_b[i] * n_b;
        for j in 0..6 {
            d_b[i * 6 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // `L = 6` requires all `225` second minors because every one is multiplied by an independent
    // `\mathcal J` tensor coefficient.
    // Compute the `225` distinct `2 x 2` minors once first, then reuse them in every larger second
    // minor; this is the minimum minor-evaluation count for this compound-minor DAG.
    let mut minor2_b = [zero; 225];
    for r0 in 0..6 {
        for r1 in (r0 + 1)..6 {
            let row_pair = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
            for c0 in 0..6 {
                for c1 in (c0 + 1)..6 {
                    let col_pair = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                    minor2_b[row_pair * 15 + col_pair] =
                        d_b[r0 * 6 + c0] * d_b[r1 * 6 + c1] - d_b[r0 * 6 + c1] * d_b[r1 * 6 + c0];
                }
            }
        }
    }
    let mut second_b = [zero; 225];
    let jsl_b = w.bb.j_slice(0);
    let mut j_b = zero;
    for eta in 0..6 {
        for xi in (eta + 1)..6 {
            let row_pair = eta * (12 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..6 {
                for y in (z + 1)..6 {
                    let col_pair = z * (12 - z - 1) / 2 + (y - z - 1);
                    let mut retained_rows = [0usize; 4];
                    let mut retained_cols = [0usize; 4];
                    let mut nr = 0usize;
                    let mut nc = 0usize;
                    for r in 0..6 {
                        if r != eta && r != xi {
                            retained_rows[nr] = r;
                            nr += 1;
                        }
                    }
                    for c in 0..6 {
                        if c != z && c != y {
                            retained_cols[nc] = c;
                            nc += 1;
                        }
                    }

                    // Partition the retained `4 x 4` determinant into two row pairs. Its six
                    // Laplace products use only globally precomputed `2 x 2` minors.
                    let r0 = retained_rows[0];
                    let r1 = retained_rows[1];
                    let r2 = retained_rows[2];
                    let r3 = retained_rows[3];
                    let c0 = retained_cols[0];
                    let c1 = retained_cols[1];
                    let c2 = retained_cols[2];
                    let c3 = retained_cols[3];
                    let rp01 = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
                    let rp23 = r2 * (12 - r2 - 1) / 2 + (r3 - r2 - 1);
                    let cp01 = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                    let cp02 = c0 * (12 - c0 - 1) / 2 + (c2 - c0 - 1);
                    let cp03 = c0 * (12 - c0 - 1) / 2 + (c3 - c0 - 1);
                    let cp12 = c1 * (12 - c1 - 1) / 2 + (c2 - c1 - 1);
                    let cp13 = c1 * (12 - c1 - 1) / 2 + (c3 - c1 - 1);
                    let cp23 = c2 * (12 - c2 - 1) / 2 + (c3 - c2 - 1);
                    let p01 = minor2_b[rp01 * 15 + cp01];
                    let p02 = minor2_b[rp01 * 15 + cp02];
                    let p03 = minor2_b[rp01 * 15 + cp03];
                    let p12 = minor2_b[rp01 * 15 + cp12];
                    let p13 = minor2_b[rp01 * 15 + cp13];
                    let p23 = minor2_b[rp01 * 15 + cp23];
                    let q01 = minor2_b[rp23 * 15 + cp01];
                    let q02 = minor2_b[rp23 * 15 + cp02];
                    let q03 = minor2_b[rp23 * 15 + cp03];
                    let q12 = minor2_b[rp23 * 15 + cp12];
                    let q13 = minor2_b[rp23 * 15 + cp13];
                    let q23 = minor2_b[rp23 * 15 + cp23];
                    let second =
                        p01 * q23 - p02 * q13 + p03 * q12 + p12 * q03 - p13 * q02 + p23 * q01;
                    second_b[row_pair * 15 + col_pair] = second;
                    let r_eta = rows_b[eta];
                    let r_xi = rows_b[xi];
                    let c_z = cols_b[z];
                    let c_y = cols_b[y];
                    let direct = jsl_b[(((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y];
                    let exchange = jsl_b[(((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b += term;
                    } else {
                        j_b -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_b = [zero; 36];
    for eta in 0..6 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..6 {
            let mut value = zero;
            for c in 0..6 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (12 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (12 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_b[r * 6 + c] * second_b[row_pair * 15 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_b[eta * 6 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_b = d_b[0] * cof_b[0];
    for z in 1..6 {
        det_b += d_b[z] * cof_b[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..6 {
        let base = cols_b[z] * n_b;
        for eta in 0..6 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 6 + z] * value;
        }
    }
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 6)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_06_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm256_set1_pd(1.0);
        let j_a = _mm256_setzero_pd();
        let replacement_a = _mm256_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 6]; 4];
        let mut cols_b = [[0usize; 6]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..6 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 36] = [_mm256_setzero_pd(); 36];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..6 {
            for j in 0..6 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 6 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // `L = 6` requires all `225` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_b: [__m256d; 225] = [_mm256_setzero_pd(); 225];
        for r0 in 0..6 {
            for r1 in (r0 + 1)..6 {
                let row_pair = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..6 {
                    for c1 in (c0 + 1)..6 {
                        let col_pair = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_b[row_pair * 15 + col_pair] = _mm256_fmsub_pd(
                            d_b[r0 * 6 + c0],
                            d_b[r1 * 6 + c1],
                            _mm256_mul_pd(d_b[r0 * 6 + c1], d_b[r1 * 6 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_b: [__m256d; 225] = [_mm256_setzero_pd(); 225];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm256_setzero_pd();
        for eta in 0..6 {
            for xi in (eta + 1)..6 {
                let row_pair = eta * (12 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..6 {
                    for y in (z + 1)..6 {
                        let col_pair = z * (12 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 4];
                        let mut retained_cols = [0usize; 4];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..6 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..6 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let r3 = retained_rows[3];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let c3 = retained_cols[3];
                        let rp01 = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
                        let rp23 = r2 * (12 - r2 - 1) / 2 + (r3 - r2 - 1);
                        let cp01 = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (12 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp03 = c0 * (12 - c0 - 1) / 2 + (c3 - c0 - 1);
                        let cp12 = c1 * (12 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let cp13 = c1 * (12 - c1 - 1) / 2 + (c3 - c1 - 1);
                        let cp23 = c2 * (12 - c2 - 1) / 2 + (c3 - c2 - 1);
                        let p01 = minor2_b[rp01 * 15 + cp01];
                        let p02 = minor2_b[rp01 * 15 + cp02];
                        let p03 = minor2_b[rp01 * 15 + cp03];
                        let p12 = minor2_b[rp01 * 15 + cp12];
                        let p13 = minor2_b[rp01 * 15 + cp13];
                        let p23 = minor2_b[rp01 * 15 + cp23];
                        let q01 = minor2_b[rp23 * 15 + cp01];
                        let q02 = minor2_b[rp23 * 15 + cp02];
                        let q03 = minor2_b[rp23 * 15 + cp03];
                        let q12 = minor2_b[rp23 * 15 + cp12];
                        let q13 = minor2_b[rp23 * 15 + cp13];
                        let q23 = minor2_b[rp23 * 15 + cp23];
                        let first = _mm256_fmsub_pd(p01, q23, _mm256_mul_pd(p02, q13));
                        let second01 = _mm256_fmadd_pd(p03, q12, first);
                        let second02 = _mm256_fmadd_pd(p12, q03, second01);
                        let second03 = _mm256_fnmadd_pd(p13, q02, second02);
                        let second = _mm256_fmadd_pd(p23, q01, second03);
                        second_b[row_pair * 15 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm256_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm256_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m256d; 36] = [_mm256_setzero_pd(); 36];
        for eta in 0..6 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..6 {
                let mut value = _mm256_setzero_pd();
                for c in 0..6 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (12 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (12 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_b[r * 6 + c], second_b[row_pair * 15 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_b[eta * 6 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
        for z in 1..6 {
            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..6 {
            for eta in 0..6 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 6 + z], values, replacement_b);
            }
        }
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (0, 6)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_06_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        let det_a = _mm512_set1_pd(1.0);
        let j_a = _mm512_setzero_pd();
        let replacement_a = _mm512_setzero_pd();

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 6]; 8];
        let mut cols_b = [[0usize; 6]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..6 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 36] = [_mm512_setzero_pd(); 36];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..6 {
            for j in 0..6 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 6 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // `L = 6` requires all `225` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_b: [__m512d; 225] = [_mm512_setzero_pd(); 225];
        for r0 in 0..6 {
            for r1 in (r0 + 1)..6 {
                let row_pair = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..6 {
                    for c1 in (c0 + 1)..6 {
                        let col_pair = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_b[row_pair * 15 + col_pair] = _mm512_fmsub_pd(
                            d_b[r0 * 6 + c0],
                            d_b[r1 * 6 + c1],
                            _mm512_mul_pd(d_b[r0 * 6 + c1], d_b[r1 * 6 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_b: [__m512d; 225] = [_mm512_setzero_pd(); 225];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm512_setzero_pd();
        for eta in 0..6 {
            for xi in (eta + 1)..6 {
                let row_pair = eta * (12 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..6 {
                    for y in (z + 1)..6 {
                        let col_pair = z * (12 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 4];
                        let mut retained_cols = [0usize; 4];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..6 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..6 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let r3 = retained_rows[3];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let c3 = retained_cols[3];
                        let rp01 = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
                        let rp23 = r2 * (12 - r2 - 1) / 2 + (r3 - r2 - 1);
                        let cp01 = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (12 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp03 = c0 * (12 - c0 - 1) / 2 + (c3 - c0 - 1);
                        let cp12 = c1 * (12 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let cp13 = c1 * (12 - c1 - 1) / 2 + (c3 - c1 - 1);
                        let cp23 = c2 * (12 - c2 - 1) / 2 + (c3 - c2 - 1);
                        let p01 = minor2_b[rp01 * 15 + cp01];
                        let p02 = minor2_b[rp01 * 15 + cp02];
                        let p03 = minor2_b[rp01 * 15 + cp03];
                        let p12 = minor2_b[rp01 * 15 + cp12];
                        let p13 = minor2_b[rp01 * 15 + cp13];
                        let p23 = minor2_b[rp01 * 15 + cp23];
                        let q01 = minor2_b[rp23 * 15 + cp01];
                        let q02 = minor2_b[rp23 * 15 + cp02];
                        let q03 = minor2_b[rp23 * 15 + cp03];
                        let q12 = minor2_b[rp23 * 15 + cp12];
                        let q13 = minor2_b[rp23 * 15 + cp13];
                        let q23 = minor2_b[rp23 * 15 + cp23];
                        let first = _mm512_fmsub_pd(p01, q23, _mm512_mul_pd(p02, q13));
                        let second01 = _mm512_fmadd_pd(p03, q12, first);
                        let second02 = _mm512_fmadd_pd(p12, q03, second01);
                        let second03 = _mm512_fnmadd_pd(p13, q02, second02);
                        let second = _mm512_fmadd_pd(p23, q01, second03);
                        second_b[row_pair * 15 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm512_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm512_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m512d; 36] = [_mm512_setzero_pd(); 36];
        for eta in 0..6 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..6 {
                let mut value = _mm512_setzero_pd();
                for c in 0..6 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (12 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (12 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_b[r * 6 + c], second_b[row_pair * 15 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_b[eta * 6 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
        for z in 1..6 {
            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..6 {
            for eta in 0..6 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 6 + z], values, replacement_b);
            }
        }
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (1, 0)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_10_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 1];
    let mut cols_a = [0usize; 1];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..1 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 1];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_a[i] * n_a;
        for j in 0..1 {
            d_a[i * 1 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_a = d_a[0];
    let j_a = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..1 {
        let base = cols_a[z] * n_a;
        for eta in 0..1 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += value;
        }
    }
    let det_b = <T as From<f64>>::from(1.0);
    let j_b = zero;
    let replacement_b = zero;
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_10_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 4];
        let mut cols_a = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm256_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_add_pd(replacement_a, values);
            }
        }
        let det_b = _mm256_set1_pd(1.0);
        let j_b = _mm256_setzero_pd();
        let replacement_b = _mm256_setzero_pd();
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_10_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 8];
        let mut cols_a = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm512_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_add_pd(replacement_a, values);
            }
        }
        let det_b = _mm512_set1_pd(1.0);
        let j_b = _mm512_setzero_pd();
        let replacement_b = _mm512_setzero_pd();
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (1, 1)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_11_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 1];
    let mut cols_a = [0usize; 1];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..1 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 1];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_a[i] * n_a;
        for j in 0..1 {
            d_a[i * 1 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_a = d_a[0];
    let j_a = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..1 {
        let base = cols_a[z] * n_a;
        for eta in 0..1 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 1];
    let mut cols_b = [0usize; 1];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..1 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 1];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_b[i] * n_b;
        for j in 0..1 {
            d_b[i * 1 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_b = d_b[0];
    let j_b = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..1 {
        let base = cols_b[z] * n_b;
        for eta in 0..1 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let ii_term = iisl[(((rows_a[0] * n + cols_a[0]) * n + rows_b[0]) * n) + cols_b[0]];

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_11_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 4];
        let mut cols_a = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm256_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_add_pd(replacement_a, values);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 4];
        let mut cols_b = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm256_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_add_pd(replacement_b, values);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut lane_values = [0.0f64; 4];
        for lane in 0..4 {
            lane_values[lane] = *iisl.get_unchecked(
                (((rows_a[lane][0] * n + cols_a[lane][0]) * n + rows_b[lane][0]) * n)
                    + cols_b[lane][0],
            );
        }
        let ii_term = _mm256_loadu_pd(lane_values.as_ptr());

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_11_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 8];
        let mut cols_a = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm512_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_add_pd(replacement_a, values);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 8];
        let mut cols_b = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm512_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_add_pd(replacement_b, values);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut lane_values = [0.0f64; 8];
        for lane in 0..8 {
            lane_values[lane] = *iisl.get_unchecked(
                (((rows_a[lane][0] * n + cols_a[lane][0]) * n + rows_b[lane][0]) * n)
                    + cols_b[lane][0],
            );
        }
        let ii_term = _mm512_loadu_pd(lane_values.as_ptr());

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (1, 2)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_12_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 1];
    let mut cols_a = [0usize; 1];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..1 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 1];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_a[i] * n_a;
        for j in 0..1 {
            d_a[i * 1 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_a = d_a[0];
    let j_a = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..1 {
        let base = cols_a[z] * n_a;
        for eta in 0..1 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 2];
    let mut cols_b = [0usize; 2];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..2 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 4];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..2 {
        let row = rows_b[i] * n_b;
        for j in 0..2 {
            d_b[i * 2 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 2`, all four cofactors are existing entries of \mathbf D and require no determinant
    // evaluation.
    let cof_b = [d_b[3], -d_b[2], -d_b[1], d_b[0]];
    let det_b = d_b[0] * cof_b[0] + d_b[1] * cof_b[1];
    let jsl_b = w.bb.j_slice(0);
    let r0 = rows_b[0];
    let r1 = rows_b[1];
    let c0 = cols_b[0];
    let c1 = cols_b[1];
    let direct = jsl_b[(((r0 * n_b + c0) * n_b + r1) * n_b) + c1];
    let exchange = jsl_b[(((r0 * n_b + c1) * n_b + r1) * n_b) + c0];
    let j_b = direct - exchange;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..2 {
        let base = cols_b[z] * n_b;
        for eta in 0..2 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 2 + z] * value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    let base_a = (rows_a[0] * n + cols_a[0]) * n * n;
    for y in 0..2 {
        for xi in 0..2 {
            let value = iisl[base_a + rows_b[xi] * n + cols_b[y]];
            ii_term += cof_b[xi * 2 + y] * value;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 2)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_12_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 4];
        let mut cols_a = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm256_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_add_pd(replacement_a, values);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 2]; 4];
        let mut cols_b = [[0usize; 2]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 4] = [_mm256_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 2 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_b: [__m256d; 4] = [
            d_b[3],
            _mm256_sub_pd(_mm256_setzero_pd(), d_b[2]),
            _mm256_sub_pd(_mm256_setzero_pd(), d_b[1]),
            d_b[0],
        ];
        let det_b = _mm256_fmsub_pd(d_b[0], d_b[3], _mm256_mul_pd(d_b[1], d_b[2]));
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut direct_lane = [0.0f64; 4];
        let mut exchange_lane = [0.0f64; 4];
        for lane in 0..4 {
            let r0 = rows_b[lane][0];
            let r1 = rows_b[lane][1];
            let c0 = cols_b[lane][0];
            let c1 = cols_b[lane][1];
            direct_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c0) * n_b + r1) * n_b) + c1);
            exchange_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c1) * n_b + r1) * n_b) + c0);
        }
        let j_b = _mm256_sub_pd(
            _mm256_loadu_pd(direct_lane.as_ptr()),
            _mm256_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 2 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for y in 0..2 {
            for xi in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = (((rows_a[lane][0] * n + cols_a[lane][0]) * n + rows_b[lane][xi])
                        * n)
                        + cols_b[lane][y];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm256_fmadd_pd(
                    cof_b[xi * 2 + y],
                    _mm256_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 2)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_12_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 8];
        let mut cols_a = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm512_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_add_pd(replacement_a, values);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 2]; 8];
        let mut cols_b = [[0usize; 2]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 4] = [_mm512_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 2 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_b: [__m512d; 4] = [
            d_b[3],
            _mm512_sub_pd(_mm512_setzero_pd(), d_b[2]),
            _mm512_sub_pd(_mm512_setzero_pd(), d_b[1]),
            d_b[0],
        ];
        let det_b = _mm512_fmsub_pd(d_b[0], d_b[3], _mm512_mul_pd(d_b[1], d_b[2]));
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut direct_lane = [0.0f64; 8];
        let mut exchange_lane = [0.0f64; 8];
        for lane in 0..8 {
            let r0 = rows_b[lane][0];
            let r1 = rows_b[lane][1];
            let c0 = cols_b[lane][0];
            let c1 = cols_b[lane][1];
            direct_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c0) * n_b + r1) * n_b) + c1);
            exchange_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c1) * n_b + r1) * n_b) + c0);
        }
        let j_b = _mm512_sub_pd(
            _mm512_loadu_pd(direct_lane.as_ptr()),
            _mm512_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 2 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for y in 0..2 {
            for xi in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = (((rows_a[lane][0] * n + cols_a[lane][0]) * n + rows_b[lane][xi])
                        * n)
                        + cols_b[lane][y];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm512_fmadd_pd(
                    cof_b[xi * 2 + y],
                    _mm512_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (1, 3)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_13_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 1];
    let mut cols_a = [0usize; 1];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..1 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 1];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_a[i] * n_a;
        for j in 0..1 {
            d_a[i * 1 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_a = d_a[0];
    let j_a = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..1 {
        let base = cols_a[z] * n_a;
        for eta in 0..1 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 3];
    let mut cols_b = [0usize; 3];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..3 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 9];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..3 {
        let row = rows_b[i] * n_b;
        for j in 0..3 {
            d_b[i * 3 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `9` distinct second minors for `L = 3`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_b = [zero; 9];
    let jsl_b = w.bb.j_slice(0);
    let mut j_b = zero;
    for eta in 0..3 {
        for xi in (eta + 1)..3 {
            let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..3 {
                for y in (z + 1)..3 {
                    let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 1];
                    let mut ii = 0usize;
                    for r in 0..3 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..3 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 1 + jj] = d_b[r * 3 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0];
                    second_b[row_pair * 3 + col_pair] = second;
                    let r_eta = rows_b[eta];
                    let r_xi = rows_b[xi];
                    let c_z = cols_b[z];
                    let c_y = cols_b[y];
                    let direct = jsl_b[(((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y];
                    let exchange = jsl_b[(((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b += term;
                    } else {
                        j_b -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_b = [zero; 9];
    for eta in 0..3 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..3 {
            let mut value = zero;
            for c in 0..3 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_b[r * 3 + c] * second_b[row_pair * 3 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_b = d_b[0] * cof_b[0];
    for z in 1..3 {
        det_b += d_b[z] * cof_b[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..3 {
        let base = cols_b[z] * n_b;
        for eta in 0..3 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 3 + z] * value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    let base_a = (rows_a[0] * n + cols_a[0]) * n * n;
    for y in 0..3 {
        for xi in 0..3 {
            let value = iisl[base_a + rows_b[xi] * n + cols_b[y]];
            ii_term += cof_b[xi * 3 + y] * value;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 3)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_13_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 4];
        let mut cols_a = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm256_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_add_pd(replacement_a, values);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 3]; 4];
        let mut cols_b = [[0usize; 3]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 3 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm256_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 1] = [_mm256_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_b[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_b[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm256_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm256_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm256_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_b[r * 3 + c], second_b[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
        for z in 1..3 {
            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 3 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for y in 0..3 {
            for xi in 0..3 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = (((rows_a[lane][0] * n + cols_a[lane][0]) * n + rows_b[lane][xi])
                        * n)
                        + cols_b[lane][y];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm256_fmadd_pd(
                    cof_b[xi * 3 + y],
                    _mm256_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 3)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_13_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 8];
        let mut cols_a = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm512_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_add_pd(replacement_a, values);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 3]; 8];
        let mut cols_b = [[0usize; 3]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 3 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm512_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 1] = [_mm512_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_b[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_b[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm512_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm512_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm512_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_b[r * 3 + c], second_b[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
        for z in 1..3 {
            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 3 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for y in 0..3 {
            for xi in 0..3 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = (((rows_a[lane][0] * n + cols_a[lane][0]) * n + rows_b[lane][xi])
                        * n)
                        + cols_b[lane][y];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm512_fmadd_pd(
                    cof_b[xi * 3 + y],
                    _mm512_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (1, 4)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_14_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 1];
    let mut cols_a = [0usize; 1];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..1 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 1];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_a[i] * n_a;
        for j in 0..1 {
            d_a[i * 1 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_a = d_a[0];
    let j_a = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..1 {
        let base = cols_a[z] * n_a;
        for eta in 0..1 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 4];
    let mut cols_b = [0usize; 4];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..4 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 16];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..4 {
        let row = rows_b[i] * n_b;
        for j in 0..4 {
            d_b[i * 4 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `36` distinct second minors for `L = 4`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_b = [zero; 36];
    let jsl_b = w.bb.j_slice(0);
    let mut j_b = zero;
    for eta in 0..4 {
        for xi in (eta + 1)..4 {
            let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..4 {
                for y in (z + 1)..4 {
                    let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 4];
                    let mut ii = 0usize;
                    for r in 0..4 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..4 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 2 + jj] = d_b[r * 4 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0] * minor[3] - minor[1] * minor[2];
                    second_b[row_pair * 6 + col_pair] = second;
                    let r_eta = rows_b[eta];
                    let r_xi = rows_b[xi];
                    let c_z = cols_b[z];
                    let c_y = cols_b[y];
                    let direct = jsl_b[(((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y];
                    let exchange = jsl_b[(((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b += term;
                    } else {
                        j_b -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_b = [zero; 16];
    for eta in 0..4 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..4 {
            let mut value = zero;
            for c in 0..4 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_b[r * 4 + c] * second_b[row_pair * 6 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_b[eta * 4 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_b = d_b[0] * cof_b[0];
    for z in 1..4 {
        det_b += d_b[z] * cof_b[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..4 {
        let base = cols_b[z] * n_b;
        for eta in 0..4 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 4 + z] * value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    let base_a = (rows_a[0] * n + cols_a[0]) * n * n;
    for y in 0..4 {
        for xi in 0..4 {
            let value = iisl[base_a + rows_b[xi] * n + cols_b[y]];
            ii_term += cof_b[xi * 4 + y] * value;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 4)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_14_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 4];
        let mut cols_a = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm256_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_add_pd(replacement_a, values);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 4]; 4];
        let mut cols_b = [[0usize; 4]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 16] = [_mm256_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 4 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m256d; 36] = [_mm256_setzero_pd(); 36];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm256_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 4] = [_mm256_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_b[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm256_fmsub_pd(minor[0], minor[3], _mm256_mul_pd(minor[1], minor[2]));
                        second_b[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm256_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm256_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m256d; 16] = [_mm256_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm256_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_b[r * 4 + c], second_b[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_b[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
        for z in 1..4 {
            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 4 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for y in 0..4 {
            for xi in 0..4 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = (((rows_a[lane][0] * n + cols_a[lane][0]) * n + rows_b[lane][xi])
                        * n)
                        + cols_b[lane][y];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm256_fmadd_pd(
                    cof_b[xi * 4 + y],
                    _mm256_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 4)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_14_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 8];
        let mut cols_a = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm512_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_add_pd(replacement_a, values);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 4]; 8];
        let mut cols_b = [[0usize; 4]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 16] = [_mm512_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 4 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m512d; 36] = [_mm512_setzero_pd(); 36];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm512_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 4] = [_mm512_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_b[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm512_fmsub_pd(minor[0], minor[3], _mm512_mul_pd(minor[1], minor[2]));
                        second_b[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm512_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm512_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m512d; 16] = [_mm512_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm512_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_b[r * 4 + c], second_b[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_b[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
        for z in 1..4 {
            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 4 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for y in 0..4 {
            for xi in 0..4 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = (((rows_a[lane][0] * n + cols_a[lane][0]) * n + rows_b[lane][xi])
                        * n)
                        + cols_b[lane][y];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm512_fmadd_pd(
                    cof_b[xi * 4 + y],
                    _mm512_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (1, 5)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_15_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 1];
    let mut cols_a = [0usize; 1];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..1 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 1];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_a[i] * n_a;
        for j in 0..1 {
            d_a[i * 1 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_a = d_a[0];
    let j_a = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..1 {
        let base = cols_a[z] * n_a;
        for eta in 0..1 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 5];
    let mut cols_b = [0usize; 5];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..5 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 25];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..5 {
        let row = rows_b[i] * n_b;
        for j in 0..5 {
            d_b[i * 5 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // `L = 5` requires all `100` second minors because every one is multiplied by an independent
    // `\mathcal J` tensor coefficient.
    // Compute the `100` distinct `2 x 2` minors once first, then reuse them in every larger second
    // minor; this is the minimum minor-evaluation count for this compound-minor DAG.
    let mut minor2_b = [zero; 100];
    for r0 in 0..5 {
        for r1 in (r0 + 1)..5 {
            let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
            for c0 in 0..5 {
                for c1 in (c0 + 1)..5 {
                    let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                    minor2_b[row_pair * 10 + col_pair] =
                        d_b[r0 * 5 + c0] * d_b[r1 * 5 + c1] - d_b[r0 * 5 + c1] * d_b[r1 * 5 + c0];
                }
            }
        }
    }
    let mut second_b = [zero; 100];
    let jsl_b = w.bb.j_slice(0);
    let mut j_b = zero;
    for eta in 0..5 {
        for xi in (eta + 1)..5 {
            let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..5 {
                for y in (z + 1)..5 {
                    let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                    let mut retained_rows = [0usize; 3];
                    let mut retained_cols = [0usize; 3];
                    let mut nr = 0usize;
                    let mut nc = 0usize;
                    for r in 0..5 {
                        if r != eta && r != xi {
                            retained_rows[nr] = r;
                            nr += 1;
                        }
                    }
                    for c in 0..5 {
                        if c != z && c != y {
                            retained_cols[nc] = c;
                            nc += 1;
                        }
                    }

                    // Expand the `3 x 3` second minor through its first retained row using three
                    // precomputed `2 x 2` minors.
                    let r0 = retained_rows[0];
                    let r1 = retained_rows[1];
                    let r2 = retained_rows[2];
                    let c0 = retained_cols[0];
                    let c1 = retained_cols[1];
                    let c2 = retained_cols[2];
                    let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                    let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                    let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                    let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                    let second = d_b[r0 * 5 + c0] * minor2_b[rp12 * 10 + cp12]
                        - d_b[r0 * 5 + c1] * minor2_b[rp12 * 10 + cp02]
                        + d_b[r0 * 5 + c2] * minor2_b[rp12 * 10 + cp01];
                    second_b[row_pair * 10 + col_pair] = second;
                    let r_eta = rows_b[eta];
                    let r_xi = rows_b[xi];
                    let c_z = cols_b[z];
                    let c_y = cols_b[y];
                    let direct = jsl_b[(((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y];
                    let exchange = jsl_b[(((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b += term;
                    } else {
                        j_b -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_b = [zero; 25];
    for eta in 0..5 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..5 {
            let mut value = zero;
            for c in 0..5 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_b[r * 5 + c] * second_b[row_pair * 10 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_b[eta * 5 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_b = d_b[0] * cof_b[0];
    for z in 1..5 {
        det_b += d_b[z] * cof_b[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..5 {
        let base = cols_b[z] * n_b;
        for eta in 0..5 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 5 + z] * value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    let base_a = (rows_a[0] * n + cols_a[0]) * n * n;
    for y in 0..5 {
        for xi in 0..5 {
            let value = iisl[base_a + rows_b[xi] * n + cols_b[y]];
            ii_term += cof_b[xi * 5 + y] * value;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 5)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_15_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 4];
        let mut cols_a = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm256_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_add_pd(replacement_a, values);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 5]; 4];
        let mut cols_b = [[0usize; 5]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..5 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 25] = [_mm256_setzero_pd(); 25];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..5 {
            for j in 0..5 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 5 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // `L = 5` requires all `100` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_b: [__m256d; 100] = [_mm256_setzero_pd(); 100];
        for r0 in 0..5 {
            for r1 in (r0 + 1)..5 {
                let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..5 {
                    for c1 in (c0 + 1)..5 {
                        let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_b[row_pair * 10 + col_pair] = _mm256_fmsub_pd(
                            d_b[r0 * 5 + c0],
                            d_b[r1 * 5 + c1],
                            _mm256_mul_pd(d_b[r0 * 5 + c1], d_b[r1 * 5 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_b: [__m256d; 100] = [_mm256_setzero_pd(); 100];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm256_setzero_pd();
        for eta in 0..5 {
            for xi in (eta + 1)..5 {
                let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..5 {
                    for y in (z + 1)..5 {
                        let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 3];
                        let mut retained_cols = [0usize; 3];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..5 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..5 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                        let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let first = _mm256_fmsub_pd(
                            d_b[r0 * 5 + c0],
                            minor2_b[rp12 * 10 + cp12],
                            _mm256_mul_pd(d_b[r0 * 5 + c1], minor2_b[rp12 * 10 + cp02]),
                        );
                        let second =
                            _mm256_fmadd_pd(d_b[r0 * 5 + c2], minor2_b[rp12 * 10 + cp01], first);
                        second_b[row_pair * 10 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm256_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm256_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m256d; 25] = [_mm256_setzero_pd(); 25];
        for eta in 0..5 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..5 {
                let mut value = _mm256_setzero_pd();
                for c in 0..5 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_b[r * 5 + c], second_b[row_pair * 10 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_b[eta * 5 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
        for z in 1..5 {
            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..5 {
            for eta in 0..5 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 5 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for y in 0..5 {
            for xi in 0..5 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = (((rows_a[lane][0] * n + cols_a[lane][0]) * n + rows_b[lane][xi])
                        * n)
                        + cols_b[lane][y];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm256_fmadd_pd(
                    cof_b[xi * 5 + y],
                    _mm256_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (1, 5)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_15_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 1]; 8];
        let mut cols_a = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_a = d_a[0];
        let j_a = _mm512_setzero_pd();
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_add_pd(replacement_a, values);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 5]; 8];
        let mut cols_b = [[0usize; 5]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..5 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 25] = [_mm512_setzero_pd(); 25];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..5 {
            for j in 0..5 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 5 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // `L = 5` requires all `100` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_b: [__m512d; 100] = [_mm512_setzero_pd(); 100];
        for r0 in 0..5 {
            for r1 in (r0 + 1)..5 {
                let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..5 {
                    for c1 in (c0 + 1)..5 {
                        let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_b[row_pair * 10 + col_pair] = _mm512_fmsub_pd(
                            d_b[r0 * 5 + c0],
                            d_b[r1 * 5 + c1],
                            _mm512_mul_pd(d_b[r0 * 5 + c1], d_b[r1 * 5 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_b: [__m512d; 100] = [_mm512_setzero_pd(); 100];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm512_setzero_pd();
        for eta in 0..5 {
            for xi in (eta + 1)..5 {
                let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..5 {
                    for y in (z + 1)..5 {
                        let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 3];
                        let mut retained_cols = [0usize; 3];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..5 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..5 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                        let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let first = _mm512_fmsub_pd(
                            d_b[r0 * 5 + c0],
                            minor2_b[rp12 * 10 + cp12],
                            _mm512_mul_pd(d_b[r0 * 5 + c1], minor2_b[rp12 * 10 + cp02]),
                        );
                        let second =
                            _mm512_fmadd_pd(d_b[r0 * 5 + c2], minor2_b[rp12 * 10 + cp01], first);
                        second_b[row_pair * 10 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm512_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm512_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m512d; 25] = [_mm512_setzero_pd(); 25];
        for eta in 0..5 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..5 {
                let mut value = _mm512_setzero_pd();
                for c in 0..5 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_b[r * 5 + c], second_b[row_pair * 10 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_b[eta * 5 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
        for z in 1..5 {
            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..5 {
            for eta in 0..5 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 5 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for y in 0..5 {
            for xi in 0..5 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = (((rows_a[lane][0] * n + cols_a[lane][0]) * n + rows_b[lane][xi])
                        * n)
                        + cols_b[lane][y];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm512_fmadd_pd(
                    cof_b[xi * 5 + y],
                    _mm512_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (2, 0)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_20_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 2];
    let mut cols_a = [0usize; 2];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..2 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 4];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..2 {
        let row = rows_a[i] * n_a;
        for j in 0..2 {
            d_a[i * 2 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 2`, all four cofactors are existing entries of \mathbf D and require no determinant
    // evaluation.
    let cof_a = [d_a[3], -d_a[2], -d_a[1], d_a[0]];
    let det_a = d_a[0] * cof_a[0] + d_a[1] * cof_a[1];
    let jsl_a = w.aa.j_slice(0);
    let r0 = rows_a[0];
    let r1 = rows_a[1];
    let c0 = cols_a[0];
    let c1 = cols_a[1];
    let direct = jsl_a[(((r0 * n_a + c0) * n_a + r1) * n_a) + c1];
    let exchange = jsl_a[(((r0 * n_a + c1) * n_a + r1) * n_a) + c0];
    let j_a = direct - exchange;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..2 {
        let base = cols_a[z] * n_a;
        for eta in 0..2 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 2 + z] * value;
        }
    }
    let det_b = <T as From<f64>>::from(1.0);
    let j_b = zero;
    let replacement_b = zero;
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (2, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_20_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 2]; 4];
        let mut cols_a = [[0usize; 2]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 4] = [_mm256_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 2 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_a: [__m256d; 4] = [
            d_a[3],
            _mm256_sub_pd(_mm256_setzero_pd(), d_a[2]),
            _mm256_sub_pd(_mm256_setzero_pd(), d_a[1]),
            d_a[0],
        ];
        let det_a = _mm256_fmsub_pd(d_a[0], d_a[3], _mm256_mul_pd(d_a[1], d_a[2]));
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut direct_lane = [0.0f64; 4];
        let mut exchange_lane = [0.0f64; 4];
        for lane in 0..4 {
            let r0 = rows_a[lane][0];
            let r1 = rows_a[lane][1];
            let c0 = cols_a[lane][0];
            let c1 = cols_a[lane][1];
            direct_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c0) * n_a + r1) * n_a) + c1);
            exchange_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c1) * n_a + r1) * n_a) + c0);
        }
        let j_a = _mm256_sub_pd(
            _mm256_loadu_pd(direct_lane.as_ptr()),
            _mm256_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 2 + z], values, replacement_a);
            }
        }
        let det_b = _mm256_set1_pd(1.0);
        let j_b = _mm256_setzero_pd();
        let replacement_b = _mm256_setzero_pd();
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (2, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_20_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 2]; 8];
        let mut cols_a = [[0usize; 2]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 4] = [_mm512_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 2 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_a: [__m512d; 4] = [
            d_a[3],
            _mm512_sub_pd(_mm512_setzero_pd(), d_a[2]),
            _mm512_sub_pd(_mm512_setzero_pd(), d_a[1]),
            d_a[0],
        ];
        let det_a = _mm512_fmsub_pd(d_a[0], d_a[3], _mm512_mul_pd(d_a[1], d_a[2]));
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut direct_lane = [0.0f64; 8];
        let mut exchange_lane = [0.0f64; 8];
        for lane in 0..8 {
            let r0 = rows_a[lane][0];
            let r1 = rows_a[lane][1];
            let c0 = cols_a[lane][0];
            let c1 = cols_a[lane][1];
            direct_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c0) * n_a + r1) * n_a) + c1);
            exchange_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c1) * n_a + r1) * n_a) + c0);
        }
        let j_a = _mm512_sub_pd(
            _mm512_loadu_pd(direct_lane.as_ptr()),
            _mm512_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 2 + z], values, replacement_a);
            }
        }
        let det_b = _mm512_set1_pd(1.0);
        let j_b = _mm512_setzero_pd();
        let replacement_b = _mm512_setzero_pd();
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (2, 1)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_21_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 2];
    let mut cols_a = [0usize; 2];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..2 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 4];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..2 {
        let row = rows_a[i] * n_a;
        for j in 0..2 {
            d_a[i * 2 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 2`, all four cofactors are existing entries of \mathbf D and require no determinant
    // evaluation.
    let cof_a = [d_a[3], -d_a[2], -d_a[1], d_a[0]];
    let det_a = d_a[0] * cof_a[0] + d_a[1] * cof_a[1];
    let jsl_a = w.aa.j_slice(0);
    let r0 = rows_a[0];
    let r1 = rows_a[1];
    let c0 = cols_a[0];
    let c1 = cols_a[1];
    let direct = jsl_a[(((r0 * n_a + c0) * n_a + r1) * n_a) + c1];
    let exchange = jsl_a[(((r0 * n_a + c1) * n_a + r1) * n_a) + c0];
    let j_a = direct - exchange;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..2 {
        let base = cols_a[z] * n_a;
        for eta in 0..2 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 2 + z] * value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 1];
    let mut cols_b = [0usize; 1];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..1 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 1];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_b[i] * n_b;
        for j in 0..1 {
            d_b[i * 1 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_b = d_b[0];
    let j_b = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..1 {
        let base = cols_b[z] * n_b;
        for eta in 0..1 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    let suffix_b = rows_b[0] * n + cols_b[0];
    for z in 0..2 {
        for eta in 0..2 {
            let base_a = (rows_a[eta] * n + cols_a[z]) * n * n;
            let value = iisl[base_a + suffix_b];
            ii_term += cof_a[eta * 2 + z] * value;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (2, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_21_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 2]; 4];
        let mut cols_a = [[0usize; 2]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 4] = [_mm256_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 2 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_a: [__m256d; 4] = [
            d_a[3],
            _mm256_sub_pd(_mm256_setzero_pd(), d_a[2]),
            _mm256_sub_pd(_mm256_setzero_pd(), d_a[1]),
            d_a[0],
        ];
        let det_a = _mm256_fmsub_pd(d_a[0], d_a[3], _mm256_mul_pd(d_a[1], d_a[2]));
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut direct_lane = [0.0f64; 4];
        let mut exchange_lane = [0.0f64; 4];
        for lane in 0..4 {
            let r0 = rows_a[lane][0];
            let r1 = rows_a[lane][1];
            let c0 = cols_a[lane][0];
            let c1 = cols_a[lane][1];
            direct_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c0) * n_a + r1) * n_a) + c1);
            exchange_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c1) * n_a + r1) * n_a) + c0);
        }
        let j_a = _mm256_sub_pd(
            _mm256_loadu_pd(direct_lane.as_ptr()),
            _mm256_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 2 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 4];
        let mut cols_b = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm256_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_add_pd(replacement_b, values);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n + rows_b[lane][0])
                        * n)
                        + cols_b[lane][0];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm256_fmadd_pd(
                    cof_a[eta * 2 + z],
                    _mm256_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (2, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_21_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 2]; 8];
        let mut cols_a = [[0usize; 2]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 4] = [_mm512_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 2 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_a: [__m512d; 4] = [
            d_a[3],
            _mm512_sub_pd(_mm512_setzero_pd(), d_a[2]),
            _mm512_sub_pd(_mm512_setzero_pd(), d_a[1]),
            d_a[0],
        ];
        let det_a = _mm512_fmsub_pd(d_a[0], d_a[3], _mm512_mul_pd(d_a[1], d_a[2]));
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut direct_lane = [0.0f64; 8];
        let mut exchange_lane = [0.0f64; 8];
        for lane in 0..8 {
            let r0 = rows_a[lane][0];
            let r1 = rows_a[lane][1];
            let c0 = cols_a[lane][0];
            let c1 = cols_a[lane][1];
            direct_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c0) * n_a + r1) * n_a) + c1);
            exchange_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c1) * n_a + r1) * n_a) + c0);
        }
        let j_a = _mm512_sub_pd(
            _mm512_loadu_pd(direct_lane.as_ptr()),
            _mm512_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 2 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 8];
        let mut cols_b = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm512_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_add_pd(replacement_b, values);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n + rows_b[lane][0])
                        * n)
                        + cols_b[lane][0];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm512_fmadd_pd(
                    cof_a[eta * 2 + z],
                    _mm512_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (2, 2)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_22_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 2];
    let mut cols_a = [0usize; 2];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..2 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 4];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..2 {
        let row = rows_a[i] * n_a;
        for j in 0..2 {
            d_a[i * 2 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 2`, all four cofactors are existing entries of \mathbf D and require no determinant
    // evaluation.
    let cof_a = [d_a[3], -d_a[2], -d_a[1], d_a[0]];
    let det_a = d_a[0] * cof_a[0] + d_a[1] * cof_a[1];
    let jsl_a = w.aa.j_slice(0);
    let r0 = rows_a[0];
    let r1 = rows_a[1];
    let c0 = cols_a[0];
    let c1 = cols_a[1];
    let direct = jsl_a[(((r0 * n_a + c0) * n_a + r1) * n_a) + c1];
    let exchange = jsl_a[(((r0 * n_a + c1) * n_a + r1) * n_a) + c0];
    let j_a = direct - exchange;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..2 {
        let base = cols_a[z] * n_a;
        for eta in 0..2 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 2 + z] * value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 2];
    let mut cols_b = [0usize; 2];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..2 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 4];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..2 {
        let row = rows_b[i] * n_b;
        for j in 0..2 {
            d_b[i * 2 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 2`, all four cofactors are existing entries of \mathbf D and require no determinant
    // evaluation.
    let cof_b = [d_b[3], -d_b[2], -d_b[1], d_b[0]];
    let det_b = d_b[0] * cof_b[0] + d_b[1] * cof_b[1];
    let jsl_b = w.bb.j_slice(0);
    let r0 = rows_b[0];
    let r1 = rows_b[1];
    let c0 = cols_b[0];
    let c1 = cols_b[1];
    let direct = jsl_b[(((r0 * n_b + c0) * n_b + r1) * n_b) + c1];
    let exchange = jsl_b[(((r0 * n_b + c1) * n_b + r1) * n_b) + c0];
    let j_b = direct - exchange;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..2 {
        let base = cols_b[z] * n_b;
        for eta in 0..2 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 2 + z] * value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    for z in 0..2 {
        for eta in 0..2 {
            let base_a = (rows_a[eta] * n + cols_a[z]) * n * n;
            let mut inner = zero;
            for y in 0..2 {
                for xi in 0..2 {
                    let value = iisl[base_a + rows_b[xi] * n + cols_b[y]];
                    inner += cof_b[xi * 2 + y] * value;
                }
            }
            ii_term += cof_a[eta * 2 + z] * inner;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (2, 2)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_22_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 2]; 4];
        let mut cols_a = [[0usize; 2]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 4] = [_mm256_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 2 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_a: [__m256d; 4] = [
            d_a[3],
            _mm256_sub_pd(_mm256_setzero_pd(), d_a[2]),
            _mm256_sub_pd(_mm256_setzero_pd(), d_a[1]),
            d_a[0],
        ];
        let det_a = _mm256_fmsub_pd(d_a[0], d_a[3], _mm256_mul_pd(d_a[1], d_a[2]));
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut direct_lane = [0.0f64; 4];
        let mut exchange_lane = [0.0f64; 4];
        for lane in 0..4 {
            let r0 = rows_a[lane][0];
            let r1 = rows_a[lane][1];
            let c0 = cols_a[lane][0];
            let c1 = cols_a[lane][1];
            direct_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c0) * n_a + r1) * n_a) + c1);
            exchange_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c1) * n_a + r1) * n_a) + c0);
        }
        let j_a = _mm256_sub_pd(
            _mm256_loadu_pd(direct_lane.as_ptr()),
            _mm256_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 2 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 2]; 4];
        let mut cols_b = [[0usize; 2]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 4] = [_mm256_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 2 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_b: [__m256d; 4] = [
            d_b[3],
            _mm256_sub_pd(_mm256_setzero_pd(), d_b[2]),
            _mm256_sub_pd(_mm256_setzero_pd(), d_b[1]),
            d_b[0],
        ];
        let det_b = _mm256_fmsub_pd(d_b[0], d_b[3], _mm256_mul_pd(d_b[1], d_b[2]));
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut direct_lane = [0.0f64; 4];
        let mut exchange_lane = [0.0f64; 4];
        for lane in 0..4 {
            let r0 = rows_b[lane][0];
            let r1 = rows_b[lane][1];
            let c0 = cols_b[lane][0];
            let c1 = cols_b[lane][1];
            direct_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c0) * n_b + r1) * n_b) + c1);
            exchange_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c1) * n_b + r1) * n_b) + c0);
        }
        let j_b = _mm256_sub_pd(
            _mm256_loadu_pd(direct_lane.as_ptr()),
            _mm256_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 2 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut inner = _mm256_setzero_pd();
                for y in 0..2 {
                    for xi in 0..2 {
                        let mut lane_values = [0.0f64; 4];
                        for lane in 0..4 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm256_fmadd_pd(
                            cof_b[xi * 2 + y],
                            _mm256_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm256_fmadd_pd(cof_a[eta * 2 + z], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (2, 2)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_22_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 2]; 8];
        let mut cols_a = [[0usize; 2]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 4] = [_mm512_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 2 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_a: [__m512d; 4] = [
            d_a[3],
            _mm512_sub_pd(_mm512_setzero_pd(), d_a[2]),
            _mm512_sub_pd(_mm512_setzero_pd(), d_a[1]),
            d_a[0],
        ];
        let det_a = _mm512_fmsub_pd(d_a[0], d_a[3], _mm512_mul_pd(d_a[1], d_a[2]));
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut direct_lane = [0.0f64; 8];
        let mut exchange_lane = [0.0f64; 8];
        for lane in 0..8 {
            let r0 = rows_a[lane][0];
            let r1 = rows_a[lane][1];
            let c0 = cols_a[lane][0];
            let c1 = cols_a[lane][1];
            direct_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c0) * n_a + r1) * n_a) + c1);
            exchange_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c1) * n_a + r1) * n_a) + c0);
        }
        let j_a = _mm512_sub_pd(
            _mm512_loadu_pd(direct_lane.as_ptr()),
            _mm512_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 2 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 2]; 8];
        let mut cols_b = [[0usize; 2]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 4] = [_mm512_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 2 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_b: [__m512d; 4] = [
            d_b[3],
            _mm512_sub_pd(_mm512_setzero_pd(), d_b[2]),
            _mm512_sub_pd(_mm512_setzero_pd(), d_b[1]),
            d_b[0],
        ];
        let det_b = _mm512_fmsub_pd(d_b[0], d_b[3], _mm512_mul_pd(d_b[1], d_b[2]));
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut direct_lane = [0.0f64; 8];
        let mut exchange_lane = [0.0f64; 8];
        for lane in 0..8 {
            let r0 = rows_b[lane][0];
            let r1 = rows_b[lane][1];
            let c0 = cols_b[lane][0];
            let c1 = cols_b[lane][1];
            direct_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c0) * n_b + r1) * n_b) + c1);
            exchange_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c1) * n_b + r1) * n_b) + c0);
        }
        let j_b = _mm512_sub_pd(
            _mm512_loadu_pd(direct_lane.as_ptr()),
            _mm512_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 2 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut inner = _mm512_setzero_pd();
                for y in 0..2 {
                    for xi in 0..2 {
                        let mut lane_values = [0.0f64; 8];
                        for lane in 0..8 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm512_fmadd_pd(
                            cof_b[xi * 2 + y],
                            _mm512_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm512_fmadd_pd(cof_a[eta * 2 + z], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (2, 3)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_23_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 2];
    let mut cols_a = [0usize; 2];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..2 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 4];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..2 {
        let row = rows_a[i] * n_a;
        for j in 0..2 {
            d_a[i * 2 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 2`, all four cofactors are existing entries of \mathbf D and require no determinant
    // evaluation.
    let cof_a = [d_a[3], -d_a[2], -d_a[1], d_a[0]];
    let det_a = d_a[0] * cof_a[0] + d_a[1] * cof_a[1];
    let jsl_a = w.aa.j_slice(0);
    let r0 = rows_a[0];
    let r1 = rows_a[1];
    let c0 = cols_a[0];
    let c1 = cols_a[1];
    let direct = jsl_a[(((r0 * n_a + c0) * n_a + r1) * n_a) + c1];
    let exchange = jsl_a[(((r0 * n_a + c1) * n_a + r1) * n_a) + c0];
    let j_a = direct - exchange;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..2 {
        let base = cols_a[z] * n_a;
        for eta in 0..2 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 2 + z] * value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 3];
    let mut cols_b = [0usize; 3];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..3 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 9];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..3 {
        let row = rows_b[i] * n_b;
        for j in 0..3 {
            d_b[i * 3 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `9` distinct second minors for `L = 3`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_b = [zero; 9];
    let jsl_b = w.bb.j_slice(0);
    let mut j_b = zero;
    for eta in 0..3 {
        for xi in (eta + 1)..3 {
            let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..3 {
                for y in (z + 1)..3 {
                    let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 1];
                    let mut ii = 0usize;
                    for r in 0..3 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..3 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 1 + jj] = d_b[r * 3 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0];
                    second_b[row_pair * 3 + col_pair] = second;
                    let r_eta = rows_b[eta];
                    let r_xi = rows_b[xi];
                    let c_z = cols_b[z];
                    let c_y = cols_b[y];
                    let direct = jsl_b[(((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y];
                    let exchange = jsl_b[(((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b += term;
                    } else {
                        j_b -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_b = [zero; 9];
    for eta in 0..3 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..3 {
            let mut value = zero;
            for c in 0..3 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_b[r * 3 + c] * second_b[row_pair * 3 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_b = d_b[0] * cof_b[0];
    for z in 1..3 {
        det_b += d_b[z] * cof_b[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..3 {
        let base = cols_b[z] * n_b;
        for eta in 0..3 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 3 + z] * value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    for z in 0..2 {
        for eta in 0..2 {
            let base_a = (rows_a[eta] * n + cols_a[z]) * n * n;
            let mut inner = zero;
            for y in 0..3 {
                for xi in 0..3 {
                    let value = iisl[base_a + rows_b[xi] * n + cols_b[y]];
                    inner += cof_b[xi * 3 + y] * value;
                }
            }
            ii_term += cof_a[eta * 2 + z] * inner;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (2, 3)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_23_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 2]; 4];
        let mut cols_a = [[0usize; 2]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 4] = [_mm256_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 2 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_a: [__m256d; 4] = [
            d_a[3],
            _mm256_sub_pd(_mm256_setzero_pd(), d_a[2]),
            _mm256_sub_pd(_mm256_setzero_pd(), d_a[1]),
            d_a[0],
        ];
        let det_a = _mm256_fmsub_pd(d_a[0], d_a[3], _mm256_mul_pd(d_a[1], d_a[2]));
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut direct_lane = [0.0f64; 4];
        let mut exchange_lane = [0.0f64; 4];
        for lane in 0..4 {
            let r0 = rows_a[lane][0];
            let r1 = rows_a[lane][1];
            let c0 = cols_a[lane][0];
            let c1 = cols_a[lane][1];
            direct_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c0) * n_a + r1) * n_a) + c1);
            exchange_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c1) * n_a + r1) * n_a) + c0);
        }
        let j_a = _mm256_sub_pd(
            _mm256_loadu_pd(direct_lane.as_ptr()),
            _mm256_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 2 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 3]; 4];
        let mut cols_b = [[0usize; 3]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 3 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm256_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 1] = [_mm256_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_b[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_b[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm256_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm256_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm256_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_b[r * 3 + c], second_b[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
        for z in 1..3 {
            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 3 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut inner = _mm256_setzero_pd();
                for y in 0..3 {
                    for xi in 0..3 {
                        let mut lane_values = [0.0f64; 4];
                        for lane in 0..4 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm256_fmadd_pd(
                            cof_b[xi * 3 + y],
                            _mm256_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm256_fmadd_pd(cof_a[eta * 2 + z], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (2, 3)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_23_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 2]; 8];
        let mut cols_a = [[0usize; 2]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 4] = [_mm512_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 2 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_a: [__m512d; 4] = [
            d_a[3],
            _mm512_sub_pd(_mm512_setzero_pd(), d_a[2]),
            _mm512_sub_pd(_mm512_setzero_pd(), d_a[1]),
            d_a[0],
        ];
        let det_a = _mm512_fmsub_pd(d_a[0], d_a[3], _mm512_mul_pd(d_a[1], d_a[2]));
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut direct_lane = [0.0f64; 8];
        let mut exchange_lane = [0.0f64; 8];
        for lane in 0..8 {
            let r0 = rows_a[lane][0];
            let r1 = rows_a[lane][1];
            let c0 = cols_a[lane][0];
            let c1 = cols_a[lane][1];
            direct_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c0) * n_a + r1) * n_a) + c1);
            exchange_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c1) * n_a + r1) * n_a) + c0);
        }
        let j_a = _mm512_sub_pd(
            _mm512_loadu_pd(direct_lane.as_ptr()),
            _mm512_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 2 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 3]; 8];
        let mut cols_b = [[0usize; 3]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 3 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm512_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 1] = [_mm512_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_b[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_b[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm512_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm512_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm512_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_b[r * 3 + c], second_b[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
        for z in 1..3 {
            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 3 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut inner = _mm512_setzero_pd();
                for y in 0..3 {
                    for xi in 0..3 {
                        let mut lane_values = [0.0f64; 8];
                        for lane in 0..8 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm512_fmadd_pd(
                            cof_b[xi * 3 + y],
                            _mm512_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm512_fmadd_pd(cof_a[eta * 2 + z], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (2, 4)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_24_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 2];
    let mut cols_a = [0usize; 2];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..2 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 4];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..2 {
        let row = rows_a[i] * n_a;
        for j in 0..2 {
            d_a[i * 2 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // For `L = 2`, all four cofactors are existing entries of \mathbf D and require no determinant
    // evaluation.
    let cof_a = [d_a[3], -d_a[2], -d_a[1], d_a[0]];
    let det_a = d_a[0] * cof_a[0] + d_a[1] * cof_a[1];
    let jsl_a = w.aa.j_slice(0);
    let r0 = rows_a[0];
    let r1 = rows_a[1];
    let c0 = cols_a[0];
    let c1 = cols_a[1];
    let direct = jsl_a[(((r0 * n_a + c0) * n_a + r1) * n_a) + c1];
    let exchange = jsl_a[(((r0 * n_a + c1) * n_a + r1) * n_a) + c0];
    let j_a = direct - exchange;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..2 {
        let base = cols_a[z] * n_a;
        for eta in 0..2 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 2 + z] * value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 4];
    let mut cols_b = [0usize; 4];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..4 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 16];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..4 {
        let row = rows_b[i] * n_b;
        for j in 0..4 {
            d_b[i * 4 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `36` distinct second minors for `L = 4`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_b = [zero; 36];
    let jsl_b = w.bb.j_slice(0);
    let mut j_b = zero;
    for eta in 0..4 {
        for xi in (eta + 1)..4 {
            let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..4 {
                for y in (z + 1)..4 {
                    let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 4];
                    let mut ii = 0usize;
                    for r in 0..4 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..4 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 2 + jj] = d_b[r * 4 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0] * minor[3] - minor[1] * minor[2];
                    second_b[row_pair * 6 + col_pair] = second;
                    let r_eta = rows_b[eta];
                    let r_xi = rows_b[xi];
                    let c_z = cols_b[z];
                    let c_y = cols_b[y];
                    let direct = jsl_b[(((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y];
                    let exchange = jsl_b[(((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b += term;
                    } else {
                        j_b -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_b = [zero; 16];
    for eta in 0..4 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..4 {
            let mut value = zero;
            for c in 0..4 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_b[r * 4 + c] * second_b[row_pair * 6 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_b[eta * 4 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_b = d_b[0] * cof_b[0];
    for z in 1..4 {
        det_b += d_b[z] * cof_b[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..4 {
        let base = cols_b[z] * n_b;
        for eta in 0..4 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 4 + z] * value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    for z in 0..2 {
        for eta in 0..2 {
            let base_a = (rows_a[eta] * n + cols_a[z]) * n * n;
            let mut inner = zero;
            for y in 0..4 {
                for xi in 0..4 {
                    let value = iisl[base_a + rows_b[xi] * n + cols_b[y]];
                    inner += cof_b[xi * 4 + y] * value;
                }
            }
            ii_term += cof_a[eta * 2 + z] * inner;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (2, 4)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_24_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 2]; 4];
        let mut cols_a = [[0usize; 2]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 4] = [_mm256_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 2 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_a: [__m256d; 4] = [
            d_a[3],
            _mm256_sub_pd(_mm256_setzero_pd(), d_a[2]),
            _mm256_sub_pd(_mm256_setzero_pd(), d_a[1]),
            d_a[0],
        ];
        let det_a = _mm256_fmsub_pd(d_a[0], d_a[3], _mm256_mul_pd(d_a[1], d_a[2]));
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut direct_lane = [0.0f64; 4];
        let mut exchange_lane = [0.0f64; 4];
        for lane in 0..4 {
            let r0 = rows_a[lane][0];
            let r1 = rows_a[lane][1];
            let c0 = cols_a[lane][0];
            let c1 = cols_a[lane][1];
            direct_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c0) * n_a + r1) * n_a) + c1);
            exchange_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c1) * n_a + r1) * n_a) + c0);
        }
        let j_a = _mm256_sub_pd(
            _mm256_loadu_pd(direct_lane.as_ptr()),
            _mm256_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 2 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 4]; 4];
        let mut cols_b = [[0usize; 4]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 16] = [_mm256_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 4 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m256d; 36] = [_mm256_setzero_pd(); 36];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm256_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 4] = [_mm256_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_b[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm256_fmsub_pd(minor[0], minor[3], _mm256_mul_pd(minor[1], minor[2]));
                        second_b[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm256_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm256_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m256d; 16] = [_mm256_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm256_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_b[r * 4 + c], second_b[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_b[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
        for z in 1..4 {
            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 4 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut inner = _mm256_setzero_pd();
                for y in 0..4 {
                    for xi in 0..4 {
                        let mut lane_values = [0.0f64; 4];
                        for lane in 0..4 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm256_fmadd_pd(
                            cof_b[xi * 4 + y],
                            _mm256_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm256_fmadd_pd(cof_a[eta * 2 + z], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (2, 4)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_24_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 2]; 8];
        let mut cols_a = [[0usize; 2]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 4] = [_mm512_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 2 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_a: [__m512d; 4] = [
            d_a[3],
            _mm512_sub_pd(_mm512_setzero_pd(), d_a[2]),
            _mm512_sub_pd(_mm512_setzero_pd(), d_a[1]),
            d_a[0],
        ];
        let det_a = _mm512_fmsub_pd(d_a[0], d_a[3], _mm512_mul_pd(d_a[1], d_a[2]));
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut direct_lane = [0.0f64; 8];
        let mut exchange_lane = [0.0f64; 8];
        for lane in 0..8 {
            let r0 = rows_a[lane][0];
            let r1 = rows_a[lane][1];
            let c0 = cols_a[lane][0];
            let c1 = cols_a[lane][1];
            direct_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c0) * n_a + r1) * n_a) + c1);
            exchange_lane[lane] = *jsl_a.get_unchecked((((r0 * n_a + c1) * n_a + r1) * n_a) + c0);
        }
        let j_a = _mm512_sub_pd(
            _mm512_loadu_pd(direct_lane.as_ptr()),
            _mm512_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 2 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 4]; 8];
        let mut cols_b = [[0usize; 4]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 16] = [_mm512_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 4 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m512d; 36] = [_mm512_setzero_pd(); 36];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm512_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 4] = [_mm512_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_b[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm512_fmsub_pd(minor[0], minor[3], _mm512_mul_pd(minor[1], minor[2]));
                        second_b[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm512_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm512_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m512d; 16] = [_mm512_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm512_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_b[r * 4 + c], second_b[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_b[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
        for z in 1..4 {
            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 4 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut inner = _mm512_setzero_pd();
                for y in 0..4 {
                    for xi in 0..4 {
                        let mut lane_values = [0.0f64; 8];
                        for lane in 0..8 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm512_fmadd_pd(
                            cof_b[xi * 4 + y],
                            _mm512_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm512_fmadd_pd(cof_a[eta * 2 + z], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (3, 0)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_30_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 3];
    let mut cols_a = [0usize; 3];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..3 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 9];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..3 {
        let row = rows_a[i] * n_a;
        for j in 0..3 {
            d_a[i * 3 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `9` distinct second minors for `L = 3`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_a = [zero; 9];
    let jsl_a = w.aa.j_slice(0);
    let mut j_a = zero;
    for eta in 0..3 {
        for xi in (eta + 1)..3 {
            let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..3 {
                for y in (z + 1)..3 {
                    let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 1];
                    let mut ii = 0usize;
                    for r in 0..3 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..3 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 1 + jj] = d_a[r * 3 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0];
                    second_a[row_pair * 3 + col_pair] = second;
                    let r_eta = rows_a[eta];
                    let r_xi = rows_a[xi];
                    let c_z = cols_a[z];
                    let c_y = cols_a[y];
                    let direct = jsl_a[(((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y];
                    let exchange = jsl_a[(((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a += term;
                    } else {
                        j_a -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_a = [zero; 9];
    for eta in 0..3 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..3 {
            let mut value = zero;
            for c in 0..3 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_a[r * 3 + c] * second_a[row_pair * 3 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_a = d_a[0] * cof_a[0];
    for z in 1..3 {
        det_a += d_a[z] * cof_a[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..3 {
        let base = cols_a[z] * n_a;
        for eta in 0..3 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 3 + z] * value;
        }
    }
    let det_b = <T as From<f64>>::from(1.0);
    let j_b = zero;
    let replacement_b = zero;
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (3, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_30_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 3]; 4];
        let mut cols_a = [[0usize; 3]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 3 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm256_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 1] = [_mm256_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_a[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_a[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm256_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm256_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm256_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_a[r * 3 + c], second_a[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
        for z in 1..3 {
            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 3 + z], values, replacement_a);
            }
        }
        let det_b = _mm256_set1_pd(1.0);
        let j_b = _mm256_setzero_pd();
        let replacement_b = _mm256_setzero_pd();
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (3, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_30_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 3]; 8];
        let mut cols_a = [[0usize; 3]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 3 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm512_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 1] = [_mm512_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_a[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_a[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm512_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm512_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm512_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_a[r * 3 + c], second_a[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
        for z in 1..3 {
            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 3 + z], values, replacement_a);
            }
        }
        let det_b = _mm512_set1_pd(1.0);
        let j_b = _mm512_setzero_pd();
        let replacement_b = _mm512_setzero_pd();
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (3, 1)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_31_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 3];
    let mut cols_a = [0usize; 3];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..3 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 9];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..3 {
        let row = rows_a[i] * n_a;
        for j in 0..3 {
            d_a[i * 3 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `9` distinct second minors for `L = 3`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_a = [zero; 9];
    let jsl_a = w.aa.j_slice(0);
    let mut j_a = zero;
    for eta in 0..3 {
        for xi in (eta + 1)..3 {
            let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..3 {
                for y in (z + 1)..3 {
                    let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 1];
                    let mut ii = 0usize;
                    for r in 0..3 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..3 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 1 + jj] = d_a[r * 3 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0];
                    second_a[row_pair * 3 + col_pair] = second;
                    let r_eta = rows_a[eta];
                    let r_xi = rows_a[xi];
                    let c_z = cols_a[z];
                    let c_y = cols_a[y];
                    let direct = jsl_a[(((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y];
                    let exchange = jsl_a[(((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a += term;
                    } else {
                        j_a -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_a = [zero; 9];
    for eta in 0..3 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..3 {
            let mut value = zero;
            for c in 0..3 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_a[r * 3 + c] * second_a[row_pair * 3 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_a = d_a[0] * cof_a[0];
    for z in 1..3 {
        det_a += d_a[z] * cof_a[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..3 {
        let base = cols_a[z] * n_a;
        for eta in 0..3 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 3 + z] * value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 1];
    let mut cols_b = [0usize; 1];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..1 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 1];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_b[i] * n_b;
        for j in 0..1 {
            d_b[i * 1 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_b = d_b[0];
    let j_b = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..1 {
        let base = cols_b[z] * n_b;
        for eta in 0..1 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    let suffix_b = rows_b[0] * n + cols_b[0];
    for z in 0..3 {
        for eta in 0..3 {
            let base_a = (rows_a[eta] * n + cols_a[z]) * n * n;
            let value = iisl[base_a + suffix_b];
            ii_term += cof_a[eta * 3 + z] * value;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (3, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_31_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 3]; 4];
        let mut cols_a = [[0usize; 3]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 3 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm256_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 1] = [_mm256_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_a[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_a[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm256_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm256_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm256_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_a[r * 3 + c], second_a[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
        for z in 1..3 {
            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 3 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 4];
        let mut cols_b = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm256_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_add_pd(replacement_b, values);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n + rows_b[lane][0])
                        * n)
                        + cols_b[lane][0];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm256_fmadd_pd(
                    cof_a[eta * 3 + z],
                    _mm256_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (3, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_31_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 3]; 8];
        let mut cols_a = [[0usize; 3]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 3 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm512_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 1] = [_mm512_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_a[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_a[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm512_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm512_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm512_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_a[r * 3 + c], second_a[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
        for z in 1..3 {
            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 3 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 8];
        let mut cols_b = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm512_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_add_pd(replacement_b, values);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n + rows_b[lane][0])
                        * n)
                        + cols_b[lane][0];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm512_fmadd_pd(
                    cof_a[eta * 3 + z],
                    _mm512_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (3, 2)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_32_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 3];
    let mut cols_a = [0usize; 3];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..3 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 9];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..3 {
        let row = rows_a[i] * n_a;
        for j in 0..3 {
            d_a[i * 3 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `9` distinct second minors for `L = 3`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_a = [zero; 9];
    let jsl_a = w.aa.j_slice(0);
    let mut j_a = zero;
    for eta in 0..3 {
        for xi in (eta + 1)..3 {
            let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..3 {
                for y in (z + 1)..3 {
                    let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 1];
                    let mut ii = 0usize;
                    for r in 0..3 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..3 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 1 + jj] = d_a[r * 3 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0];
                    second_a[row_pair * 3 + col_pair] = second;
                    let r_eta = rows_a[eta];
                    let r_xi = rows_a[xi];
                    let c_z = cols_a[z];
                    let c_y = cols_a[y];
                    let direct = jsl_a[(((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y];
                    let exchange = jsl_a[(((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a += term;
                    } else {
                        j_a -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_a = [zero; 9];
    for eta in 0..3 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..3 {
            let mut value = zero;
            for c in 0..3 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_a[r * 3 + c] * second_a[row_pair * 3 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_a = d_a[0] * cof_a[0];
    for z in 1..3 {
        det_a += d_a[z] * cof_a[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..3 {
        let base = cols_a[z] * n_a;
        for eta in 0..3 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 3 + z] * value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 2];
    let mut cols_b = [0usize; 2];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..2 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 4];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..2 {
        let row = rows_b[i] * n_b;
        for j in 0..2 {
            d_b[i * 2 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 2`, all four cofactors are existing entries of \mathbf D and require no determinant
    // evaluation.
    let cof_b = [d_b[3], -d_b[2], -d_b[1], d_b[0]];
    let det_b = d_b[0] * cof_b[0] + d_b[1] * cof_b[1];
    let jsl_b = w.bb.j_slice(0);
    let r0 = rows_b[0];
    let r1 = rows_b[1];
    let c0 = cols_b[0];
    let c1 = cols_b[1];
    let direct = jsl_b[(((r0 * n_b + c0) * n_b + r1) * n_b) + c1];
    let exchange = jsl_b[(((r0 * n_b + c1) * n_b + r1) * n_b) + c0];
    let j_b = direct - exchange;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..2 {
        let base = cols_b[z] * n_b;
        for eta in 0..2 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 2 + z] * value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    for y in 0..2 {
        for xi in 0..2 {
            let suffix_b = rows_b[xi] * n + cols_b[y];
            let mut inner = zero;
            for z in 0..3 {
                for eta in 0..3 {
                    let base_a = (rows_a[eta] * n + cols_a[z]) * n * n;
                    inner += cof_a[eta * 3 + z] * iisl[base_a + suffix_b];
                }
            }
            ii_term += cof_b[xi * 2 + y] * inner;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (3, 2)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_32_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 3]; 4];
        let mut cols_a = [[0usize; 3]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 3 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm256_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 1] = [_mm256_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_a[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_a[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm256_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm256_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm256_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_a[r * 3 + c], second_a[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
        for z in 1..3 {
            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 3 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 2]; 4];
        let mut cols_b = [[0usize; 2]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 4] = [_mm256_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 2 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_b: [__m256d; 4] = [
            d_b[3],
            _mm256_sub_pd(_mm256_setzero_pd(), d_b[2]),
            _mm256_sub_pd(_mm256_setzero_pd(), d_b[1]),
            d_b[0],
        ];
        let det_b = _mm256_fmsub_pd(d_b[0], d_b[3], _mm256_mul_pd(d_b[1], d_b[2]));
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut direct_lane = [0.0f64; 4];
        let mut exchange_lane = [0.0f64; 4];
        for lane in 0..4 {
            let r0 = rows_b[lane][0];
            let r1 = rows_b[lane][1];
            let c0 = cols_b[lane][0];
            let c1 = cols_b[lane][1];
            direct_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c0) * n_b + r1) * n_b) + c1);
            exchange_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c1) * n_b + r1) * n_b) + c0);
        }
        let j_b = _mm256_sub_pd(
            _mm256_loadu_pd(direct_lane.as_ptr()),
            _mm256_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 2 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for y in 0..2 {
            for xi in 0..2 {
                let mut inner = _mm256_setzero_pd();
                for z in 0..3 {
                    for eta in 0..3 {
                        let mut lane_values = [0.0f64; 4];
                        for lane in 0..4 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm256_fmadd_pd(
                            cof_a[eta * 3 + z],
                            _mm256_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm256_fmadd_pd(cof_b[xi * 2 + y], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (3, 2)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_32_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 3]; 8];
        let mut cols_a = [[0usize; 3]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 3 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm512_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 1] = [_mm512_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_a[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_a[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm512_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm512_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm512_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_a[r * 3 + c], second_a[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
        for z in 1..3 {
            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 3 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 2]; 8];
        let mut cols_b = [[0usize; 2]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 4] = [_mm512_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 2 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_b: [__m512d; 4] = [
            d_b[3],
            _mm512_sub_pd(_mm512_setzero_pd(), d_b[2]),
            _mm512_sub_pd(_mm512_setzero_pd(), d_b[1]),
            d_b[0],
        ];
        let det_b = _mm512_fmsub_pd(d_b[0], d_b[3], _mm512_mul_pd(d_b[1], d_b[2]));
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut direct_lane = [0.0f64; 8];
        let mut exchange_lane = [0.0f64; 8];
        for lane in 0..8 {
            let r0 = rows_b[lane][0];
            let r1 = rows_b[lane][1];
            let c0 = cols_b[lane][0];
            let c1 = cols_b[lane][1];
            direct_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c0) * n_b + r1) * n_b) + c1);
            exchange_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c1) * n_b + r1) * n_b) + c0);
        }
        let j_b = _mm512_sub_pd(
            _mm512_loadu_pd(direct_lane.as_ptr()),
            _mm512_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 2 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for y in 0..2 {
            for xi in 0..2 {
                let mut inner = _mm512_setzero_pd();
                for z in 0..3 {
                    for eta in 0..3 {
                        let mut lane_values = [0.0f64; 8];
                        for lane in 0..8 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm512_fmadd_pd(
                            cof_a[eta * 3 + z],
                            _mm512_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm512_fmadd_pd(cof_b[xi * 2 + y], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (3, 3)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_33_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 3];
    let mut cols_a = [0usize; 3];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..3 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 9];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..3 {
        let row = rows_a[i] * n_a;
        for j in 0..3 {
            d_a[i * 3 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `9` distinct second minors for `L = 3`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_a = [zero; 9];
    let jsl_a = w.aa.j_slice(0);
    let mut j_a = zero;
    for eta in 0..3 {
        for xi in (eta + 1)..3 {
            let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..3 {
                for y in (z + 1)..3 {
                    let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 1];
                    let mut ii = 0usize;
                    for r in 0..3 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..3 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 1 + jj] = d_a[r * 3 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0];
                    second_a[row_pair * 3 + col_pair] = second;
                    let r_eta = rows_a[eta];
                    let r_xi = rows_a[xi];
                    let c_z = cols_a[z];
                    let c_y = cols_a[y];
                    let direct = jsl_a[(((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y];
                    let exchange = jsl_a[(((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a += term;
                    } else {
                        j_a -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_a = [zero; 9];
    for eta in 0..3 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..3 {
            let mut value = zero;
            for c in 0..3 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_a[r * 3 + c] * second_a[row_pair * 3 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_a = d_a[0] * cof_a[0];
    for z in 1..3 {
        det_a += d_a[z] * cof_a[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..3 {
        let base = cols_a[z] * n_a;
        for eta in 0..3 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 3 + z] * value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 3];
    let mut cols_b = [0usize; 3];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..3 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 9];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..3 {
        let row = rows_b[i] * n_b;
        for j in 0..3 {
            d_b[i * 3 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `9` distinct second minors for `L = 3`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_b = [zero; 9];
    let jsl_b = w.bb.j_slice(0);
    let mut j_b = zero;
    for eta in 0..3 {
        for xi in (eta + 1)..3 {
            let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..3 {
                for y in (z + 1)..3 {
                    let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 1];
                    let mut ii = 0usize;
                    for r in 0..3 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..3 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 1 + jj] = d_b[r * 3 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0];
                    second_b[row_pair * 3 + col_pair] = second;
                    let r_eta = rows_b[eta];
                    let r_xi = rows_b[xi];
                    let c_z = cols_b[z];
                    let c_y = cols_b[y];
                    let direct = jsl_b[(((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y];
                    let exchange = jsl_b[(((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b += term;
                    } else {
                        j_b -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_b = [zero; 9];
    for eta in 0..3 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..3 {
            let mut value = zero;
            for c in 0..3 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_b[r * 3 + c] * second_b[row_pair * 3 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_b = d_b[0] * cof_b[0];
    for z in 1..3 {
        det_b += d_b[z] * cof_b[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..3 {
        let base = cols_b[z] * n_b;
        for eta in 0..3 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 3 + z] * value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    for z in 0..3 {
        for eta in 0..3 {
            let base_a = (rows_a[eta] * n + cols_a[z]) * n * n;
            let mut inner = zero;
            for y in 0..3 {
                for xi in 0..3 {
                    let value = iisl[base_a + rows_b[xi] * n + cols_b[y]];
                    inner += cof_b[xi * 3 + y] * value;
                }
            }
            ii_term += cof_a[eta * 3 + z] * inner;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (3, 3)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_33_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 3]; 4];
        let mut cols_a = [[0usize; 3]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 3 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm256_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 1] = [_mm256_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_a[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_a[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm256_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm256_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm256_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_a[r * 3 + c], second_a[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
        for z in 1..3 {
            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 3 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 3]; 4];
        let mut cols_b = [[0usize; 3]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 3 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm256_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 1] = [_mm256_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_b[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_b[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm256_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm256_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m256d; 9] = [_mm256_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm256_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_b[r * 3 + c], second_b[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
        for z in 1..3 {
            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 3 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut inner = _mm256_setzero_pd();
                for y in 0..3 {
                    for xi in 0..3 {
                        let mut lane_values = [0.0f64; 4];
                        for lane in 0..4 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm256_fmadd_pd(
                            cof_b[xi * 3 + y],
                            _mm256_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm256_fmadd_pd(cof_a[eta * 3 + z], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (3, 3)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_33_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 3]; 8];
        let mut cols_a = [[0usize; 3]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 3 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm512_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 1] = [_mm512_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_a[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_a[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm512_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm512_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm512_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_a[r * 3 + c], second_a[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_a[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
        for z in 1..3 {
            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 3 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 3]; 8];
        let mut cols_b = [[0usize; 3]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..3 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..3 {
            for j in 0..3 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 3 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 3` same-spin term contains `9` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut j_b = _mm512_setzero_pd();
        for eta in 0..3 {
            for xi in (eta + 1)..3 {
                let row_pair = eta * (6 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..3 {
                    for y in (z + 1)..3 {
                        let col_pair = z * (6 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 1] = [_mm512_setzero_pd(); 1];
                        let mut ii = 0usize;
                        for r in 0..3 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..3 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 1 + jj] = d_b[r * 3 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second = minor[0];
                        second_b[row_pair * 3 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_b[lane][eta];
                            let r_xi = rows_b[lane][xi];
                            let c_z = cols_b[lane][z];
                            let c_y = cols_b[lane][y];
                            direct_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_z) * n_b + r_xi) * n_b) + c_y);
                            exchange_lane[lane] = *jsl_b
                                .get_unchecked((((r_eta * n_b + c_y) * n_b + r_xi) * n_b) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_b = _mm512_fmadd_pd(second, jdiff, j_b);
                        } else {
                            j_b = _mm512_fnmadd_pd(second, jdiff, j_b);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_b: [__m512d; 9] = [_mm512_setzero_pd(); 9];
        for eta in 0..3 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..3 {
                let mut value = _mm512_setzero_pd();
                for c in 0..3 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (6 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (6 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_b[r * 3 + c], second_b[row_pair * 3 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_b[eta * 3 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
        for z in 1..3 {
            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
        }
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 3 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for z in 0..3 {
            for eta in 0..3 {
                let mut inner = _mm512_setzero_pd();
                for y in 0..3 {
                    for xi in 0..3 {
                        let mut lane_values = [0.0f64; 8];
                        for lane in 0..8 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm512_fmadd_pd(
                            cof_b[xi * 3 + y],
                            _mm512_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm512_fmadd_pd(cof_a[eta * 3 + z], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (4, 0)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_40_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 4];
    let mut cols_a = [0usize; 4];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..4 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 16];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..4 {
        let row = rows_a[i] * n_a;
        for j in 0..4 {
            d_a[i * 4 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `36` distinct second minors for `L = 4`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_a = [zero; 36];
    let jsl_a = w.aa.j_slice(0);
    let mut j_a = zero;
    for eta in 0..4 {
        for xi in (eta + 1)..4 {
            let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..4 {
                for y in (z + 1)..4 {
                    let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 4];
                    let mut ii = 0usize;
                    for r in 0..4 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..4 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 2 + jj] = d_a[r * 4 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0] * minor[3] - minor[1] * minor[2];
                    second_a[row_pair * 6 + col_pair] = second;
                    let r_eta = rows_a[eta];
                    let r_xi = rows_a[xi];
                    let c_z = cols_a[z];
                    let c_y = cols_a[y];
                    let direct = jsl_a[(((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y];
                    let exchange = jsl_a[(((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a += term;
                    } else {
                        j_a -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_a = [zero; 16];
    for eta in 0..4 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..4 {
            let mut value = zero;
            for c in 0..4 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_a[r * 4 + c] * second_a[row_pair * 6 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_a[eta * 4 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_a = d_a[0] * cof_a[0];
    for z in 1..4 {
        det_a += d_a[z] * cof_a[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..4 {
        let base = cols_a[z] * n_a;
        for eta in 0..4 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 4 + z] * value;
        }
    }
    let det_b = <T as From<f64>>::from(1.0);
    let j_b = zero;
    let replacement_b = zero;
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (4, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_40_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 4]; 4];
        let mut cols_a = [[0usize; 4]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 16] = [_mm256_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 4 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m256d; 36] = [_mm256_setzero_pd(); 36];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm256_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 4] = [_mm256_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_a[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm256_fmsub_pd(minor[0], minor[3], _mm256_mul_pd(minor[1], minor[2]));
                        second_a[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm256_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm256_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m256d; 16] = [_mm256_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm256_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_a[r * 4 + c], second_a[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_a[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
        for z in 1..4 {
            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 4 + z], values, replacement_a);
            }
        }
        let det_b = _mm256_set1_pd(1.0);
        let j_b = _mm256_setzero_pd();
        let replacement_b = _mm256_setzero_pd();
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (4, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_40_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 4]; 8];
        let mut cols_a = [[0usize; 4]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 16] = [_mm512_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 4 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m512d; 36] = [_mm512_setzero_pd(); 36];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm512_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 4] = [_mm512_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_a[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm512_fmsub_pd(minor[0], minor[3], _mm512_mul_pd(minor[1], minor[2]));
                        second_a[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm512_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm512_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m512d; 16] = [_mm512_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm512_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_a[r * 4 + c], second_a[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_a[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
        for z in 1..4 {
            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 4 + z], values, replacement_a);
            }
        }
        let det_b = _mm512_set1_pd(1.0);
        let j_b = _mm512_setzero_pd();
        let replacement_b = _mm512_setzero_pd();
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (4, 1)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_41_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 4];
    let mut cols_a = [0usize; 4];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..4 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 16];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..4 {
        let row = rows_a[i] * n_a;
        for j in 0..4 {
            d_a[i * 4 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `36` distinct second minors for `L = 4`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_a = [zero; 36];
    let jsl_a = w.aa.j_slice(0);
    let mut j_a = zero;
    for eta in 0..4 {
        for xi in (eta + 1)..4 {
            let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..4 {
                for y in (z + 1)..4 {
                    let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 4];
                    let mut ii = 0usize;
                    for r in 0..4 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..4 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 2 + jj] = d_a[r * 4 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0] * minor[3] - minor[1] * minor[2];
                    second_a[row_pair * 6 + col_pair] = second;
                    let r_eta = rows_a[eta];
                    let r_xi = rows_a[xi];
                    let c_z = cols_a[z];
                    let c_y = cols_a[y];
                    let direct = jsl_a[(((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y];
                    let exchange = jsl_a[(((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a += term;
                    } else {
                        j_a -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_a = [zero; 16];
    for eta in 0..4 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..4 {
            let mut value = zero;
            for c in 0..4 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_a[r * 4 + c] * second_a[row_pair * 6 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_a[eta * 4 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_a = d_a[0] * cof_a[0];
    for z in 1..4 {
        det_a += d_a[z] * cof_a[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..4 {
        let base = cols_a[z] * n_a;
        for eta in 0..4 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 4 + z] * value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 1];
    let mut cols_b = [0usize; 1];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..1 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 1];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_b[i] * n_b;
        for j in 0..1 {
            d_b[i * 1 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_b = d_b[0];
    let j_b = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..1 {
        let base = cols_b[z] * n_b;
        for eta in 0..1 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    let suffix_b = rows_b[0] * n + cols_b[0];
    for z in 0..4 {
        for eta in 0..4 {
            let base_a = (rows_a[eta] * n + cols_a[z]) * n * n;
            let value = iisl[base_a + suffix_b];
            ii_term += cof_a[eta * 4 + z] * value;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (4, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_41_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 4]; 4];
        let mut cols_a = [[0usize; 4]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 16] = [_mm256_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 4 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m256d; 36] = [_mm256_setzero_pd(); 36];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm256_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 4] = [_mm256_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_a[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm256_fmsub_pd(minor[0], minor[3], _mm256_mul_pd(minor[1], minor[2]));
                        second_a[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm256_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm256_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m256d; 16] = [_mm256_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm256_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_a[r * 4 + c], second_a[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_a[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
        for z in 1..4 {
            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 4 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 4];
        let mut cols_b = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm256_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_add_pd(replacement_b, values);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n + rows_b[lane][0])
                        * n)
                        + cols_b[lane][0];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm256_fmadd_pd(
                    cof_a[eta * 4 + z],
                    _mm256_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (4, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_41_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 4]; 8];
        let mut cols_a = [[0usize; 4]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 16] = [_mm512_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 4 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m512d; 36] = [_mm512_setzero_pd(); 36];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm512_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 4] = [_mm512_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_a[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm512_fmsub_pd(minor[0], minor[3], _mm512_mul_pd(minor[1], minor[2]));
                        second_a[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm512_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm512_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m512d; 16] = [_mm512_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm512_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_a[r * 4 + c], second_a[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_a[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
        for z in 1..4 {
            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 4 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 8];
        let mut cols_b = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm512_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_add_pd(replacement_b, values);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n + rows_b[lane][0])
                        * n)
                        + cols_b[lane][0];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm512_fmadd_pd(
                    cof_a[eta * 4 + z],
                    _mm512_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (4, 2)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_42_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 4];
    let mut cols_a = [0usize; 4];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..4 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 16];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..4 {
        let row = rows_a[i] * n_a;
        for j in 0..4 {
            d_a[i * 4 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // The same-spin two-body contraction contains `36` distinct second minors for `L = 4`.
    // Each multiplies an independently stored \mathcal J tensor coefficient, so every required
    // second minor is evaluated once and then reused to form the full cofactor matrix.
    let mut second_a = [zero; 36];
    let jsl_a = w.aa.j_slice(0);
    let mut j_a = zero;
    for eta in 0..4 {
        for xi in (eta + 1)..4 {
            let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..4 {
                for y in (z + 1)..4 {
                    let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                    let mut minor = [zero; 4];
                    let mut ii = 0usize;
                    for r in 0..4 {
                        if r == eta || r == xi {
                            continue;
                        }
                        let mut jj = 0usize;
                        for c in 0..4 {
                            if c == z || c == y {
                                continue;
                            }
                            minor[ii * 2 + jj] = d_a[r * 4 + c];
                            jj += 1;
                        }
                        ii += 1;
                    }
                    let second = minor[0] * minor[3] - minor[1] * minor[2];
                    second_a[row_pair * 6 + col_pair] = second;
                    let r_eta = rows_a[eta];
                    let r_xi = rows_a[xi];
                    let c_z = cols_a[z];
                    let c_y = cols_a[y];
                    let direct = jsl_a[(((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y];
                    let exchange = jsl_a[(((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a += term;
                    } else {
                        j_a -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_a = [zero; 16];
    for eta in 0..4 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..4 {
            let mut value = zero;
            for c in 0..4 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_a[r * 4 + c] * second_a[row_pair * 6 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_a[eta * 4 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_a = d_a[0] * cof_a[0];
    for z in 1..4 {
        det_a += d_a[z] * cof_a[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..4 {
        let base = cols_a[z] * n_a;
        for eta in 0..4 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 4 + z] * value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 2];
    let mut cols_b = [0usize; 2];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..2 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 4];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..2 {
        let row = rows_b[i] * n_b;
        for j in 0..2 {
            d_b[i * 2 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 2`, all four cofactors are existing entries of \mathbf D and require no determinant
    // evaluation.
    let cof_b = [d_b[3], -d_b[2], -d_b[1], d_b[0]];
    let det_b = d_b[0] * cof_b[0] + d_b[1] * cof_b[1];
    let jsl_b = w.bb.j_slice(0);
    let r0 = rows_b[0];
    let r1 = rows_b[1];
    let c0 = cols_b[0];
    let c1 = cols_b[1];
    let direct = jsl_b[(((r0 * n_b + c0) * n_b + r1) * n_b) + c1];
    let exchange = jsl_b[(((r0 * n_b + c1) * n_b + r1) * n_b) + c0];
    let j_b = direct - exchange;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..2 {
        let base = cols_b[z] * n_b;
        for eta in 0..2 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += cof_b[eta * 2 + z] * value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    for y in 0..2 {
        for xi in 0..2 {
            let suffix_b = rows_b[xi] * n + cols_b[y];
            let mut inner = zero;
            for z in 0..4 {
                for eta in 0..4 {
                    let base_a = (rows_a[eta] * n + cols_a[z]) * n * n;
                    inner += cof_a[eta * 4 + z] * iisl[base_a + suffix_b];
                }
            }
            ii_term += cof_b[xi * 2 + y] * inner;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (4, 2)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_42_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 4]; 4];
        let mut cols_a = [[0usize; 4]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 16] = [_mm256_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 4 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m256d; 36] = [_mm256_setzero_pd(); 36];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm256_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m256d; 4] = [_mm256_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_a[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm256_fmsub_pd(minor[0], minor[3], _mm256_mul_pd(minor[1], minor[2]));
                        second_a[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm256_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm256_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m256d; 16] = [_mm256_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm256_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_a[r * 4 + c], second_a[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_a[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
        for z in 1..4 {
            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 4 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 2]; 4];
        let mut cols_b = [[0usize; 2]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 4] = [_mm256_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 2 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_b: [__m256d; 4] = [
            d_b[3],
            _mm256_sub_pd(_mm256_setzero_pd(), d_b[2]),
            _mm256_sub_pd(_mm256_setzero_pd(), d_b[1]),
            d_b[0],
        ];
        let det_b = _mm256_fmsub_pd(d_b[0], d_b[3], _mm256_mul_pd(d_b[1], d_b[2]));
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut direct_lane = [0.0f64; 4];
        let mut exchange_lane = [0.0f64; 4];
        for lane in 0..4 {
            let r0 = rows_b[lane][0];
            let r1 = rows_b[lane][1];
            let c0 = cols_b[lane][0];
            let c1 = cols_b[lane][1];
            direct_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c0) * n_b + r1) * n_b) + c1);
            exchange_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c1) * n_b + r1) * n_b) + c0);
        }
        let j_b = _mm256_sub_pd(
            _mm256_loadu_pd(direct_lane.as_ptr()),
            _mm256_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_fmadd_pd(cof_b[eta * 2 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for y in 0..2 {
            for xi in 0..2 {
                let mut inner = _mm256_setzero_pd();
                for z in 0..4 {
                    for eta in 0..4 {
                        let mut lane_values = [0.0f64; 4];
                        for lane in 0..4 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm256_fmadd_pd(
                            cof_a[eta * 4 + z],
                            _mm256_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm256_fmadd_pd(cof_b[xi * 2 + y], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (4, 2)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_42_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 4]; 8];
        let mut cols_a = [[0usize; 4]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..4 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 16] = [_mm512_setzero_pd(); 16];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..4 {
            for j in 0..4 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 4 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The `L = 4` same-spin term contains `36` independently weighted second minors; each is
        // evaluated once and reused by every cofactor contraction.
        let mut second_a: [__m512d; 36] = [_mm512_setzero_pd(); 36];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm512_setzero_pd();
        for eta in 0..4 {
            for xi in (eta + 1)..4 {
                let row_pair = eta * (8 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..4 {
                    for y in (z + 1)..4 {
                        let col_pair = z * (8 - z - 1) / 2 + (y - z - 1);
                        let mut minor: [__m512d; 4] = [_mm512_setzero_pd(); 4];
                        let mut ii = 0usize;
                        for r in 0..4 {
                            if r == eta || r == xi {
                                continue;
                            }
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == z || c == y {
                                    continue;
                                }
                                minor[ii * 2 + jj] = d_a[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let second =
                            _mm512_fmsub_pd(minor[0], minor[3], _mm512_mul_pd(minor[1], minor[2]));
                        second_a[row_pair * 6 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm512_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm512_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m512d; 16] = [_mm512_setzero_pd(); 16];
        for eta in 0..4 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..4 {
                let mut value = _mm512_setzero_pd();
                for c in 0..4 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (8 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (8 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_a[r * 4 + c], second_a[row_pair * 6 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_a[eta * 4 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
        for z in 1..4 {
            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..4 {
            for eta in 0..4 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 4 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 2]; 8];
        let mut cols_b = [[0usize; 2]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..2 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 4] = [_mm512_setzero_pd(); 4];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..2 {
            for j in 0..2 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 2 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // Rank two requires one determinant FMSUB; all cofactors are existing determinant entries
        // with sign changes only.
        let cof_b: [__m512d; 4] = [
            d_b[3],
            _mm512_sub_pd(_mm512_setzero_pd(), d_b[2]),
            _mm512_sub_pd(_mm512_setzero_pd(), d_b[1]),
            d_b[0],
        ];
        let det_b = _mm512_fmsub_pd(d_b[0], d_b[3], _mm512_mul_pd(d_b[1], d_b[2]));
        let jsl_b_t = w.bb.j_slice(0);
        let jsl_b = std::slice::from_raw_parts(jsl_b_t.as_ptr().cast::<f64>(), jsl_b_t.len());
        let mut direct_lane = [0.0f64; 8];
        let mut exchange_lane = [0.0f64; 8];
        for lane in 0..8 {
            let r0 = rows_b[lane][0];
            let r1 = rows_b[lane][1];
            let c0 = cols_b[lane][0];
            let c1 = cols_b[lane][1];
            direct_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c0) * n_b + r1) * n_b) + c1);
            exchange_lane[lane] = *jsl_b.get_unchecked((((r0 * n_b + c1) * n_b + r1) * n_b) + c0);
        }
        let j_b = _mm512_sub_pd(
            _mm512_loadu_pd(direct_lane.as_ptr()),
            _mm512_loadu_pd(exchange_lane.as_ptr()),
        );
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..2 {
            for eta in 0..2 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_fmadd_pd(cof_b[eta * 2 + z], values, replacement_b);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for y in 0..2 {
            for xi in 0..2 {
                let mut inner = _mm512_setzero_pd();
                for z in 0..4 {
                    for eta in 0..4 {
                        let mut lane_values = [0.0f64; 8];
                        for lane in 0..8 {
                            let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n
                                + rows_b[lane][xi])
                                * n)
                                + cols_b[lane][y];
                            lane_values[lane] = *iisl.get_unchecked(index);
                        }
                        inner = _mm512_fmadd_pd(
                            cof_a[eta * 4 + z],
                            _mm512_loadu_pd(lane_values.as_ptr()),
                            inner,
                        );
                    }
                }
                ii_term = _mm512_fmadd_pd(cof_b[xi * 2 + y], inner, ii_term);
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (5, 0)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_50_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 5];
    let mut cols_a = [0usize; 5];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..5 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 25];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..5 {
        let row = rows_a[i] * n_a;
        for j in 0..5 {
            d_a[i * 5 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // `L = 5` requires all `100` second minors because every one is multiplied by an independent
    // `\mathcal J` tensor coefficient.
    // Compute the `100` distinct `2 x 2` minors once first, then reuse them in every larger second
    // minor; this is the minimum minor-evaluation count for this compound-minor DAG.
    let mut minor2_a = [zero; 100];
    for r0 in 0..5 {
        for r1 in (r0 + 1)..5 {
            let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
            for c0 in 0..5 {
                for c1 in (c0 + 1)..5 {
                    let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                    minor2_a[row_pair * 10 + col_pair] =
                        d_a[r0 * 5 + c0] * d_a[r1 * 5 + c1] - d_a[r0 * 5 + c1] * d_a[r1 * 5 + c0];
                }
            }
        }
    }
    let mut second_a = [zero; 100];
    let jsl_a = w.aa.j_slice(0);
    let mut j_a = zero;
    for eta in 0..5 {
        for xi in (eta + 1)..5 {
            let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..5 {
                for y in (z + 1)..5 {
                    let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                    let mut retained_rows = [0usize; 3];
                    let mut retained_cols = [0usize; 3];
                    let mut nr = 0usize;
                    let mut nc = 0usize;
                    for r in 0..5 {
                        if r != eta && r != xi {
                            retained_rows[nr] = r;
                            nr += 1;
                        }
                    }
                    for c in 0..5 {
                        if c != z && c != y {
                            retained_cols[nc] = c;
                            nc += 1;
                        }
                    }

                    // Expand the `3 x 3` second minor through its first retained row using three
                    // precomputed `2 x 2` minors.
                    let r0 = retained_rows[0];
                    let r1 = retained_rows[1];
                    let r2 = retained_rows[2];
                    let c0 = retained_cols[0];
                    let c1 = retained_cols[1];
                    let c2 = retained_cols[2];
                    let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                    let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                    let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                    let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                    let second = d_a[r0 * 5 + c0] * minor2_a[rp12 * 10 + cp12]
                        - d_a[r0 * 5 + c1] * minor2_a[rp12 * 10 + cp02]
                        + d_a[r0 * 5 + c2] * minor2_a[rp12 * 10 + cp01];
                    second_a[row_pair * 10 + col_pair] = second;
                    let r_eta = rows_a[eta];
                    let r_xi = rows_a[xi];
                    let c_z = cols_a[z];
                    let c_y = cols_a[y];
                    let direct = jsl_a[(((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y];
                    let exchange = jsl_a[(((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a += term;
                    } else {
                        j_a -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_a = [zero; 25];
    for eta in 0..5 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..5 {
            let mut value = zero;
            for c in 0..5 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_a[r * 5 + c] * second_a[row_pair * 10 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_a[eta * 5 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_a = d_a[0] * cof_a[0];
    for z in 1..5 {
        det_a += d_a[z] * cof_a[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..5 {
        let base = cols_a[z] * n_a;
        for eta in 0..5 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 5 + z] * value;
        }
    }
    let det_b = <T as From<f64>>::from(1.0);
    let j_b = zero;
    let replacement_b = zero;
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (5, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_50_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 5]; 4];
        let mut cols_a = [[0usize; 5]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..5 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 25] = [_mm256_setzero_pd(); 25];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..5 {
            for j in 0..5 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 5 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // `L = 5` requires all `100` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_a: [__m256d; 100] = [_mm256_setzero_pd(); 100];
        for r0 in 0..5 {
            for r1 in (r0 + 1)..5 {
                let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..5 {
                    for c1 in (c0 + 1)..5 {
                        let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_a[row_pair * 10 + col_pair] = _mm256_fmsub_pd(
                            d_a[r0 * 5 + c0],
                            d_a[r1 * 5 + c1],
                            _mm256_mul_pd(d_a[r0 * 5 + c1], d_a[r1 * 5 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_a: [__m256d; 100] = [_mm256_setzero_pd(); 100];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm256_setzero_pd();
        for eta in 0..5 {
            for xi in (eta + 1)..5 {
                let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..5 {
                    for y in (z + 1)..5 {
                        let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 3];
                        let mut retained_cols = [0usize; 3];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..5 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..5 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                        let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let first = _mm256_fmsub_pd(
                            d_a[r0 * 5 + c0],
                            minor2_a[rp12 * 10 + cp12],
                            _mm256_mul_pd(d_a[r0 * 5 + c1], minor2_a[rp12 * 10 + cp02]),
                        );
                        let second =
                            _mm256_fmadd_pd(d_a[r0 * 5 + c2], minor2_a[rp12 * 10 + cp01], first);
                        second_a[row_pair * 10 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm256_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm256_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m256d; 25] = [_mm256_setzero_pd(); 25];
        for eta in 0..5 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..5 {
                let mut value = _mm256_setzero_pd();
                for c in 0..5 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_a[r * 5 + c], second_a[row_pair * 10 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_a[eta * 5 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
        for z in 1..5 {
            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..5 {
            for eta in 0..5 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 5 + z], values, replacement_a);
            }
        }
        let det_b = _mm256_set1_pd(1.0);
        let j_b = _mm256_setzero_pd();
        let replacement_b = _mm256_setzero_pd();
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (5, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_50_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 5]; 8];
        let mut cols_a = [[0usize; 5]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..5 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 25] = [_mm512_setzero_pd(); 25];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..5 {
            for j in 0..5 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 5 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // `L = 5` requires all `100` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_a: [__m512d; 100] = [_mm512_setzero_pd(); 100];
        for r0 in 0..5 {
            for r1 in (r0 + 1)..5 {
                let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..5 {
                    for c1 in (c0 + 1)..5 {
                        let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_a[row_pair * 10 + col_pair] = _mm512_fmsub_pd(
                            d_a[r0 * 5 + c0],
                            d_a[r1 * 5 + c1],
                            _mm512_mul_pd(d_a[r0 * 5 + c1], d_a[r1 * 5 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_a: [__m512d; 100] = [_mm512_setzero_pd(); 100];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm512_setzero_pd();
        for eta in 0..5 {
            for xi in (eta + 1)..5 {
                let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..5 {
                    for y in (z + 1)..5 {
                        let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 3];
                        let mut retained_cols = [0usize; 3];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..5 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..5 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                        let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let first = _mm512_fmsub_pd(
                            d_a[r0 * 5 + c0],
                            minor2_a[rp12 * 10 + cp12],
                            _mm512_mul_pd(d_a[r0 * 5 + c1], minor2_a[rp12 * 10 + cp02]),
                        );
                        let second =
                            _mm512_fmadd_pd(d_a[r0 * 5 + c2], minor2_a[rp12 * 10 + cp01], first);
                        second_a[row_pair * 10 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm512_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm512_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m512d; 25] = [_mm512_setzero_pd(); 25];
        for eta in 0..5 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..5 {
                let mut value = _mm512_setzero_pd();
                for c in 0..5 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_a[r * 5 + c], second_a[row_pair * 10 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_a[eta * 5 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
        for z in 1..5 {
            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..5 {
            for eta in 0..5 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 5 + z], values, replacement_a);
            }
        }
        let det_b = _mm512_set1_pd(1.0);
        let j_b = _mm512_setzero_pd();
        let replacement_b = _mm512_setzero_pd();
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (5, 1)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_51_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 5];
    let mut cols_a = [0usize; 5];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..5 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 25];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..5 {
        let row = rows_a[i] * n_a;
        for j in 0..5 {
            d_a[i * 5 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // `L = 5` requires all `100` second minors because every one is multiplied by an independent
    // `\mathcal J` tensor coefficient.
    // Compute the `100` distinct `2 x 2` minors once first, then reuse them in every larger second
    // minor; this is the minimum minor-evaluation count for this compound-minor DAG.
    let mut minor2_a = [zero; 100];
    for r0 in 0..5 {
        for r1 in (r0 + 1)..5 {
            let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
            for c0 in 0..5 {
                for c1 in (c0 + 1)..5 {
                    let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                    minor2_a[row_pair * 10 + col_pair] =
                        d_a[r0 * 5 + c0] * d_a[r1 * 5 + c1] - d_a[r0 * 5 + c1] * d_a[r1 * 5 + c0];
                }
            }
        }
    }
    let mut second_a = [zero; 100];
    let jsl_a = w.aa.j_slice(0);
    let mut j_a = zero;
    for eta in 0..5 {
        for xi in (eta + 1)..5 {
            let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..5 {
                for y in (z + 1)..5 {
                    let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                    let mut retained_rows = [0usize; 3];
                    let mut retained_cols = [0usize; 3];
                    let mut nr = 0usize;
                    let mut nc = 0usize;
                    for r in 0..5 {
                        if r != eta && r != xi {
                            retained_rows[nr] = r;
                            nr += 1;
                        }
                    }
                    for c in 0..5 {
                        if c != z && c != y {
                            retained_cols[nc] = c;
                            nc += 1;
                        }
                    }

                    // Expand the `3 x 3` second minor through its first retained row using three
                    // precomputed `2 x 2` minors.
                    let r0 = retained_rows[0];
                    let r1 = retained_rows[1];
                    let r2 = retained_rows[2];
                    let c0 = retained_cols[0];
                    let c1 = retained_cols[1];
                    let c2 = retained_cols[2];
                    let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                    let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                    let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                    let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                    let second = d_a[r0 * 5 + c0] * minor2_a[rp12 * 10 + cp12]
                        - d_a[r0 * 5 + c1] * minor2_a[rp12 * 10 + cp02]
                        + d_a[r0 * 5 + c2] * minor2_a[rp12 * 10 + cp01];
                    second_a[row_pair * 10 + col_pair] = second;
                    let r_eta = rows_a[eta];
                    let r_xi = rows_a[xi];
                    let c_z = cols_a[z];
                    let c_y = cols_a[y];
                    let direct = jsl_a[(((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y];
                    let exchange = jsl_a[(((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a += term;
                    } else {
                        j_a -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_a = [zero; 25];
    for eta in 0..5 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..5 {
            let mut value = zero;
            for c in 0..5 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_a[r * 5 + c] * second_a[row_pair * 10 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_a[eta * 5 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_a = d_a[0] * cof_a[0];
    for z in 1..5 {
        det_a += d_a[z] * cof_a[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..5 {
        let base = cols_a[z] * n_a;
        for eta in 0..5 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 5 + z] * value;
        }
    }

    // Construct the b-spin contraction labels directly from the cached excitation metadata.
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    let x_rank_b = usize::from(x_ex.beta.rank);
    let x_indices_b = &x_ex.beta.indices;
    let w_indices_b = &w_ex.beta.indices;
    let mut rows_b = [0usize; 1];
    let mut cols_b = [0usize; 1];
    for i in 0..x_rank_b {
        rows_b[i] = usize::from(x_indices_b[4 + i]) - nocc_b;
        cols_b[i] = usize::from(x_indices_b[i]);
    }
    for i in x_rank_b..1 {
        let k = i - x_rank_b;
        rows_b[i] = nvirt_b + usize::from(w_indices_b[k]);
        cols_b[i] = usize::from(w_indices_b[4 + k]);
    }
    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; 1];

    // Load each entry of \mathbf D_b exactly once from the fundamental contractions.
    for i in 0..1 {
        let row = rows_b[i] * n_b;
        for j in 0..1 {
            d_b[i * 1 + j] = if i >= j {
                x0_b[row + cols_b[j]]
            } else {
                y0_b[row + cols_b[j]]
            };
        }
    }

    // For `L = 1`, the only cofactor is the empty determinant with value one.
    let det_b = d_b[0];
    let j_b = zero;

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_b = w.bb.fh_t_slice(0, 0);
    let vv_b = w.bb.v_t_slice(0, 0, 0);
    let vm_b = w.ab.vba_t_slice(0, 0, 0);
    let mut replacement_b = zero;
    for z in 0..1 {
        let base = cols_b[z] * n_b;
        for eta in 0..1 {
            let index = base + rows_b[eta];
            let value = fh_b[index] + vv_b[index] + vm_b[index];
            replacement_b += value;
        }
    }

    // Contract \mathcal{II} in the orientation that adds only `min(L_\alpha^2, L_\beta^2)` outer
    // cofactor multiplications.
    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let mut ii_term = zero;
    let suffix_b = rows_b[0] * n + cols_b[0];
    for z in 0..5 {
        for eta in 0..5 {
            let base_a = (rows_a[eta] * n + cols_a[z]) * n * n;
            let value = iisl[base_a + suffix_b];
            ii_term += cof_a[eta * 5 + z] * value;
        }
    }

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (5, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_51_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 5]; 4];
        let mut cols_a = [[0usize; 5]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..5 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 25] = [_mm256_setzero_pd(); 25];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..5 {
            for j in 0..5 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 5 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // `L = 5` requires all `100` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_a: [__m256d; 100] = [_mm256_setzero_pd(); 100];
        for r0 in 0..5 {
            for r1 in (r0 + 1)..5 {
                let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..5 {
                    for c1 in (c0 + 1)..5 {
                        let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_a[row_pair * 10 + col_pair] = _mm256_fmsub_pd(
                            d_a[r0 * 5 + c0],
                            d_a[r1 * 5 + c1],
                            _mm256_mul_pd(d_a[r0 * 5 + c1], d_a[r1 * 5 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_a: [__m256d; 100] = [_mm256_setzero_pd(); 100];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm256_setzero_pd();
        for eta in 0..5 {
            for xi in (eta + 1)..5 {
                let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..5 {
                    for y in (z + 1)..5 {
                        let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 3];
                        let mut retained_cols = [0usize; 3];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..5 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..5 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                        let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let first = _mm256_fmsub_pd(
                            d_a[r0 * 5 + c0],
                            minor2_a[rp12 * 10 + cp12],
                            _mm256_mul_pd(d_a[r0 * 5 + c1], minor2_a[rp12 * 10 + cp02]),
                        );
                        let second =
                            _mm256_fmadd_pd(d_a[r0 * 5 + c2], minor2_a[rp12 * 10 + cp01], first);
                        second_a[row_pair * 10 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm256_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm256_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m256d; 25] = [_mm256_setzero_pd(); 25];
        for eta in 0..5 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..5 {
                let mut value = _mm256_setzero_pd();
                for c in 0..5 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_a[r * 5 + c], second_a[row_pair * 10 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_a[eta * 5 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
        for z in 1..5 {
            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..5 {
            for eta in 0..5 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 5 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 4];
        let mut cols_b = [[0usize; 1]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m256d; 1] = [_mm256_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm256_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm256_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm256_add_pd(replacement_b, values);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm256_setzero_pd();
        for z in 0..5 {
            for eta in 0..5 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n + rows_b[lane][0])
                        * n)
                        + cols_b[lane][0];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm256_fmadd_pd(
                    cof_a[eta * 5 + z],
                    _mm256_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (5, 1)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_51_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 5]; 8];
        let mut cols_a = [[0usize; 5]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..5 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 25] = [_mm512_setzero_pd(); 25];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..5 {
            for j in 0..5 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 5 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // `L = 5` requires all `100` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_a: [__m512d; 100] = [_mm512_setzero_pd(); 100];
        for r0 in 0..5 {
            for r1 in (r0 + 1)..5 {
                let row_pair = r0 * (10 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..5 {
                    for c1 in (c0 + 1)..5 {
                        let col_pair = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_a[row_pair * 10 + col_pair] = _mm512_fmsub_pd(
                            d_a[r0 * 5 + c0],
                            d_a[r1 * 5 + c1],
                            _mm512_mul_pd(d_a[r0 * 5 + c1], d_a[r1 * 5 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_a: [__m512d; 100] = [_mm512_setzero_pd(); 100];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm512_setzero_pd();
        for eta in 0..5 {
            for xi in (eta + 1)..5 {
                let row_pair = eta * (10 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..5 {
                    for y in (z + 1)..5 {
                        let col_pair = z * (10 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 3];
                        let mut retained_cols = [0usize; 3];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..5 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..5 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let rp12 = r1 * (10 - r1 - 1) / 2 + (r2 - r1 - 1);
                        let cp01 = c0 * (10 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (10 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp12 = c1 * (10 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let first = _mm512_fmsub_pd(
                            d_a[r0 * 5 + c0],
                            minor2_a[rp12 * 10 + cp12],
                            _mm512_mul_pd(d_a[r0 * 5 + c1], minor2_a[rp12 * 10 + cp02]),
                        );
                        let second =
                            _mm512_fmadd_pd(d_a[r0 * 5 + c2], minor2_a[rp12 * 10 + cp01], first);
                        second_a[row_pair * 10 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm512_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm512_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m512d; 25] = [_mm512_setzero_pd(); 25];
        for eta in 0..5 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..5 {
                let mut value = _mm512_setzero_pd();
                for c in 0..5 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (10 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (10 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_a[r * 5 + c], second_a[row_pair * 10 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_a[eta * 5 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
        for z in 1..5 {
            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..5 {
            for eta in 0..5 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 5 + z], values, replacement_a);
            }
        }

        // Construct the b-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_b = w.bb.nocc;
        let nvirt_b = w.bb.nmo - nocc_b;
        let mut rows_b = [[0usize; 1]; 8];
        let mut cols_b = [[0usize; 1]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
            let x_indices = &x_ex.get_unchecked(lane).beta.indices;
            let w_indices = &w_ex.get_unchecked(lane).beta.indices;
            for i in 0..x_rank {
                rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc_b;
                cols_b[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..1 {
                let k = i - x_rank;
                rows_b[lane][i] = nvirt_b + usize::from(w_indices[k]);
                cols_b[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_b = w.bb.n();
        let x0_b_t = w.bb.x_slice(0);
        let y0_b_t = w.bb.y_slice(0);
        let x0_b = std::slice::from_raw_parts(x0_b_t.as_ptr().cast::<f64>(), x0_b_t.len());
        let y0_b = std::slice::from_raw_parts(y0_b_t.as_ptr().cast::<f64>(), y0_b_t.len());
        let mut d_b: [__m512d; 1] = [_mm512_setzero_pd(); 1];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..1 {
            for j in 0..1 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                    values[lane] = if i >= j {
                        *x0_b.get_unchecked(index)
                    } else {
                        *y0_b.get_unchecked(index)
                    };
                }
                d_b[i * 1 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // The rank-one cofactor is exactly one, so the replacement contraction needs no cofactor
        // multiplication.
        let det_b = d_b[0];
        let j_b = _mm512_setzero_pd();
        let fh_b_t = w.bb.fh_t_slice(0, 0);
        let vv_b_t = w.bb.v_t_slice(0, 0, 0);
        let vm_b_t = w.ab.vba_t_slice(0, 0, 0);
        let fh_b = std::slice::from_raw_parts(fh_b_t.as_ptr().cast::<f64>(), fh_b_t.len());
        let vv_b = std::slice::from_raw_parts(vv_b_t.as_ptr().cast::<f64>(), vv_b_t.len());
        let vm_b = std::slice::from_raw_parts(vm_b_t.as_ptr().cast::<f64>(), vm_b_t.len());
        let mut replacement_b = _mm512_setzero_pd();
        for z in 0..1 {
            for eta in 0..1 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                    lane_values[lane] = *fh_b.get_unchecked(index)
                        + *vv_b.get_unchecked(index)
                        + *vm_b.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_b = _mm512_add_pd(replacement_b, values);
            }
        }

        // Contract \mathcal{II} with the smaller cofactor space on the outside to minimise vector
        // multiplications.
        let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
        let iisl = std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
        let n = w.ab.n();
        let mut ii_term = _mm512_setzero_pd();
        for z in 0..5 {
            for eta in 0..5 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = (((rows_a[lane][eta] * n + cols_a[lane][z]) * n + rows_b[lane][0])
                        * n)
                        + cols_b[lane][0];
                    lane_values[lane] = *iisl.get_unchecked(index);
                }
                ii_term = _mm512_fmadd_pd(
                    cof_a[eta * 5 + z],
                    _mm512_loadu_pd(lane_values.as_ptr()),
                    ii_term,
                );
            }
        }

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta) = (6, 0)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_60_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the a-spin contraction labels directly from the cached excitation metadata.
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    let x_rank_a = usize::from(x_ex.alpha.rank);
    let x_indices_a = &x_ex.alpha.indices;
    let w_indices_a = &w_ex.alpha.indices;
    let mut rows_a = [0usize; 6];
    let mut cols_a = [0usize; 6];
    for i in 0..x_rank_a {
        rows_a[i] = usize::from(x_indices_a[4 + i]) - nocc_a;
        cols_a[i] = usize::from(x_indices_a[i]);
    }
    for i in x_rank_a..6 {
        let k = i - x_rank_a;
        rows_a[i] = nvirt_a + usize::from(w_indices_a[k]);
        cols_a[i] = usize::from(w_indices_a[4 + k]);
    }
    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; 36];

    // Load each entry of \mathbf D_a exactly once from the fundamental contractions.
    for i in 0..6 {
        let row = rows_a[i] * n_a;
        for j in 0..6 {
            d_a[i * 6 + j] = if i >= j {
                x0_a[row + cols_a[j]]
            } else {
                y0_a[row + cols_a[j]]
            };
        }
    }

    // `L = 6` requires all `225` second minors because every one is multiplied by an independent
    // `\mathcal J` tensor coefficient.
    // Compute the `225` distinct `2 x 2` minors once first, then reuse them in every larger second
    // minor; this is the minimum minor-evaluation count for this compound-minor DAG.
    let mut minor2_a = [zero; 225];
    for r0 in 0..6 {
        for r1 in (r0 + 1)..6 {
            let row_pair = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
            for c0 in 0..6 {
                for c1 in (c0 + 1)..6 {
                    let col_pair = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                    minor2_a[row_pair * 15 + col_pair] =
                        d_a[r0 * 6 + c0] * d_a[r1 * 6 + c1] - d_a[r0 * 6 + c1] * d_a[r1 * 6 + c0];
                }
            }
        }
    }
    let mut second_a = [zero; 225];
    let jsl_a = w.aa.j_slice(0);
    let mut j_a = zero;
    for eta in 0..6 {
        for xi in (eta + 1)..6 {
            let row_pair = eta * (12 - eta - 1) / 2 + (xi - eta - 1);
            for z in 0..6 {
                for y in (z + 1)..6 {
                    let col_pair = z * (12 - z - 1) / 2 + (y - z - 1);
                    let mut retained_rows = [0usize; 4];
                    let mut retained_cols = [0usize; 4];
                    let mut nr = 0usize;
                    let mut nc = 0usize;
                    for r in 0..6 {
                        if r != eta && r != xi {
                            retained_rows[nr] = r;
                            nr += 1;
                        }
                    }
                    for c in 0..6 {
                        if c != z && c != y {
                            retained_cols[nc] = c;
                            nc += 1;
                        }
                    }

                    // Partition the retained `4 x 4` determinant into two row pairs. Its six
                    // Laplace products use only globally precomputed `2 x 2` minors.
                    let r0 = retained_rows[0];
                    let r1 = retained_rows[1];
                    let r2 = retained_rows[2];
                    let r3 = retained_rows[3];
                    let c0 = retained_cols[0];
                    let c1 = retained_cols[1];
                    let c2 = retained_cols[2];
                    let c3 = retained_cols[3];
                    let rp01 = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
                    let rp23 = r2 * (12 - r2 - 1) / 2 + (r3 - r2 - 1);
                    let cp01 = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                    let cp02 = c0 * (12 - c0 - 1) / 2 + (c2 - c0 - 1);
                    let cp03 = c0 * (12 - c0 - 1) / 2 + (c3 - c0 - 1);
                    let cp12 = c1 * (12 - c1 - 1) / 2 + (c2 - c1 - 1);
                    let cp13 = c1 * (12 - c1 - 1) / 2 + (c3 - c1 - 1);
                    let cp23 = c2 * (12 - c2 - 1) / 2 + (c3 - c2 - 1);
                    let p01 = minor2_a[rp01 * 15 + cp01];
                    let p02 = minor2_a[rp01 * 15 + cp02];
                    let p03 = minor2_a[rp01 * 15 + cp03];
                    let p12 = minor2_a[rp01 * 15 + cp12];
                    let p13 = minor2_a[rp01 * 15 + cp13];
                    let p23 = minor2_a[rp01 * 15 + cp23];
                    let q01 = minor2_a[rp23 * 15 + cp01];
                    let q02 = minor2_a[rp23 * 15 + cp02];
                    let q03 = minor2_a[rp23 * 15 + cp03];
                    let q12 = minor2_a[rp23 * 15 + cp12];
                    let q13 = minor2_a[rp23 * 15 + cp13];
                    let q23 = minor2_a[rp23 * 15 + cp23];
                    let second =
                        p01 * q23 - p02 * q13 + p03 * q12 + p12 * q03 - p13 * q02 + p23 * q01;
                    second_a[row_pair * 15 + col_pair] = second;
                    let r_eta = rows_a[eta];
                    let r_xi = rows_a[xi];
                    let c_z = cols_a[z];
                    let c_y = cols_a[y];
                    let direct = jsl_a[(((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y];
                    let exchange = jsl_a[(((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z];
                    let term = second * (direct - exchange);
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a += term;
                    } else {
                        j_a -= term;
                    }
                }
            }
        }
    }

    // Reconstruct every first cofactor from the already evaluated second minors; no minor
    // determinant is repeated for the overlap or Hamiltonian contractions.
    let mut cof_a = [zero; 36];
    for eta in 0..6 {
        let r = if eta == 0 { 1usize } else { 0usize };
        let r_minor = if r < eta { r } else { r - 1 };
        for z in 0..6 {
            let mut value = zero;
            for c in 0..6 {
                if c == z {
                    continue;
                }
                let c_minor = if c < z { c } else { c - 1 };
                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                let row_pair = row0 * (12 - row0 - 1) / 2 + (row1 - row0 - 1);
                let col_pair = col0 * (12 - col0 - 1) / 2 + (col1 - col0 - 1);
                let term = d_a[r * 6 + c] * second_a[row_pair * 15 + col_pair];
                if ((r_minor + c_minor) & 1) == 0 {
                    value += term;
                } else {
                    value -= term;
                }
            }
            cof_a[eta * 6 + z] = if ((eta + z) & 1) == 0 { value } else { -value };
        }
    }
    let mut det_a = d_a[0] * cof_a[0];
    for z in 1..6 {
        det_a += d_a[z] * cof_a[z];
    }

    // Contract each cofactor once with the sum of the one-electron, same-spin and mixed-spin
    // one-column intermediates.
    let fh_a = w.aa.fh_t_slice(0, 0);
    let vv_a = w.aa.v_t_slice(0, 0, 0);
    let vm_a = w.ab.vab_t_slice(0, 0, 0);
    let mut replacement_a = zero;
    for z in 0..6 {
        let base = cols_a[z] * n_a;
        for eta in 0..6 {
            let index = base + rows_a[eta];
            let value = fh_a[index] + vv_a[index] + vm_a[index];
            replacement_a += cof_a[eta * 6 + z] * value;
        }
    }
    let det_b = <T as From<f64>>::from(1.0);
    let j_b = zero;
    let replacement_b = zero;
    let ii_term = zero;

    // Assemble the complete electronic and nuclear Hamiltonian before applying the common overlap
    // prefactor.
    let det_ab = det_a * det_b;
    let g0 = <T as From<f64>>::from(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = g0 * det_ab;
    core -= det_b * replacement_a;
    core -= det_a * replacement_b;
    core += j_a * det_b;
    core += j_b * det_a;
    core += ii_term;

    // Apply the two reduced reference-overlap factors and the determinant excitation phase once.
    let pref = <T as From<f64>>::from(excitation_phase)
        * w.aa.phase
        * <T as From<f64>>::from(w.aa.tilde_s_prod)
        * w.bb.phase
        * <T as From<f64>>::from(w.bb.tilde_s_prod);
    (pref * core, pref * det_ab)
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta) = (6, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 4 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 4 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 4 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 4 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid predecoded
///   excitation labels and output slices of length at least 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_60_prepared_f64x4<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 6]; 4];
        let mut cols_a = [[0usize; 6]; 4];
        for lane in 0..4 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..6 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m256d; 36] = [_mm256_setzero_pd(); 36];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..6 {
            for j in 0..6 {
                let mut values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 6 + j] = _mm256_loadu_pd(values.as_ptr());
            }
        }

        // `L = 6` requires all `225` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_a: [__m256d; 225] = [_mm256_setzero_pd(); 225];
        for r0 in 0..6 {
            for r1 in (r0 + 1)..6 {
                let row_pair = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..6 {
                    for c1 in (c0 + 1)..6 {
                        let col_pair = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_a[row_pair * 15 + col_pair] = _mm256_fmsub_pd(
                            d_a[r0 * 6 + c0],
                            d_a[r1 * 6 + c1],
                            _mm256_mul_pd(d_a[r0 * 6 + c1], d_a[r1 * 6 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_a: [__m256d; 225] = [_mm256_setzero_pd(); 225];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm256_setzero_pd();
        for eta in 0..6 {
            for xi in (eta + 1)..6 {
                let row_pair = eta * (12 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..6 {
                    for y in (z + 1)..6 {
                        let col_pair = z * (12 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 4];
                        let mut retained_cols = [0usize; 4];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..6 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..6 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let r3 = retained_rows[3];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let c3 = retained_cols[3];
                        let rp01 = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
                        let rp23 = r2 * (12 - r2 - 1) / 2 + (r3 - r2 - 1);
                        let cp01 = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (12 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp03 = c0 * (12 - c0 - 1) / 2 + (c3 - c0 - 1);
                        let cp12 = c1 * (12 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let cp13 = c1 * (12 - c1 - 1) / 2 + (c3 - c1 - 1);
                        let cp23 = c2 * (12 - c2 - 1) / 2 + (c3 - c2 - 1);
                        let p01 = minor2_a[rp01 * 15 + cp01];
                        let p02 = minor2_a[rp01 * 15 + cp02];
                        let p03 = minor2_a[rp01 * 15 + cp03];
                        let p12 = minor2_a[rp01 * 15 + cp12];
                        let p13 = minor2_a[rp01 * 15 + cp13];
                        let p23 = minor2_a[rp01 * 15 + cp23];
                        let q01 = minor2_a[rp23 * 15 + cp01];
                        let q02 = minor2_a[rp23 * 15 + cp02];
                        let q03 = minor2_a[rp23 * 15 + cp03];
                        let q12 = minor2_a[rp23 * 15 + cp12];
                        let q13 = minor2_a[rp23 * 15 + cp13];
                        let q23 = minor2_a[rp23 * 15 + cp23];
                        let first = _mm256_fmsub_pd(p01, q23, _mm256_mul_pd(p02, q13));
                        let second01 = _mm256_fmadd_pd(p03, q12, first);
                        let second02 = _mm256_fmadd_pd(p12, q03, second01);
                        let second03 = _mm256_fnmadd_pd(p13, q02, second02);
                        let second = _mm256_fmadd_pd(p23, q01, second03);
                        second_a[row_pair * 15 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 4];
                        let mut exchange_lane = [0.0f64; 4];
                        for lane in 0..4 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm256_sub_pd(
                            _mm256_loadu_pd(direct_lane.as_ptr()),
                            _mm256_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm256_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm256_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m256d; 36] = [_mm256_setzero_pd(); 36];
        for eta in 0..6 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..6 {
                let mut value = _mm256_setzero_pd();
                for c in 0..6 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (12 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (12 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm256_mul_pd(d_a[r * 6 + c], second_a[row_pair * 15 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm256_add_pd(value, term)
                    } else {
                        _mm256_sub_pd(value, term)
                    };
                }
                cof_a[eta * 6 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm256_sub_pd(_mm256_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
        for z in 1..6 {
            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm256_setzero_pd();
        for z in 0..6 {
            for eta in 0..6 {
                let mut lane_values = [0.0f64; 4];
                for lane in 0..4 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm256_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm256_fmadd_pd(cof_a[eta * 6 + z], values, replacement_a);
            }
        }
        let det_b = _mm256_set1_pd(1.0);
        let j_b = _mm256_setzero_pd();
        let replacement_b = _mm256_setzero_pd();
        let ii_term = _mm256_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm256_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm256_mul_pd(det_a, det_b);
        let mut core = _mm256_mul_pd(g0, det_ab);
        core = _mm256_fnmadd_pd(det_b, replacement_a, core);
        core = _mm256_fnmadd_pd(det_a, replacement_b, core);
        core = _mm256_fmadd_pd(j_a, det_b, core);
        core = _mm256_fmadd_pd(j_b, det_a, core);
        core = _mm256_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm256_mul_pd(
            _mm256_loadu_pd(excitation_phase.as_ptr()),
            _mm256_set1_pd(ref_pref),
        );
        let h_v = _mm256_mul_pd(core, pref);
        let s_v = _mm256_mul_pd(det_ab, pref);
        _mm256_storeu_pd(h.as_mut_ptr(), h_v);
        _mm256_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta) = (6, 0)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered reference pair with `T = f64`.
/// - `x_ex`: 8 predecoded bra excitations in SIMD-lane order.
/// - `w_ex`: 8 predecoded ket excitations in SIMD-lane order.
/// - `excitation_phase`: 8 excitation phases in SIMD-lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian output slice in SIMD-lane order.
/// - `s`: Overlap output slice in SIMD-lane order.
/// # Returns:
/// - `()`: Writes 8 Hamiltonian and overlap matrix elements.
/// # Safety:
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid predecoded
///   excitation labels and output slices of length at least 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_60_prepared_f64x8<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    unsafe {
        // Construct the a-spin contraction labels for every SIMD lane without touching full
        // determinant states.
        let nocc_a = w.aa.nocc;
        let nvirt_a = w.aa.nmo - nocc_a;
        let mut rows_a = [[0usize; 6]; 8];
        let mut cols_a = [[0usize; 6]; 8];
        for lane in 0..8 {
            let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
            let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
            let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
            for i in 0..x_rank {
                rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc_a;
                cols_a[lane][i] = usize::from(x_indices[i]);
            }
            for i in x_rank..6 {
                let k = i - x_rank;
                rows_a[lane][i] = nvirt_a + usize::from(w_indices[k]);
                cols_a[lane][i] = usize::from(w_indices[4 + k]);
            }
        }
        let n_a = w.aa.n();
        let x0_a_t = w.aa.x_slice(0);
        let y0_a_t = w.aa.y_slice(0);
        let x0_a = std::slice::from_raw_parts(x0_a_t.as_ptr().cast::<f64>(), x0_a_t.len());
        let y0_a = std::slice::from_raw_parts(y0_a_t.as_ptr().cast::<f64>(), y0_a_t.len());
        let mut d_a: [__m512d; 36] = [_mm512_setzero_pd(); 36];

        // Gather one contraction-determinant entry across independent pairs, then perform all
        // determinant algebra vertically across SIMD lanes.
        for i in 0..6 {
            for j in 0..6 {
                let mut values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                    values[lane] = if i >= j {
                        *x0_a.get_unchecked(index)
                    } else {
                        *y0_a.get_unchecked(index)
                    };
                }
                d_a[i * 6 + j] = _mm512_loadu_pd(values.as_ptr());
            }
        }

        // `L = 6` requires all `225` second minors. Compute every distinct `2 x 2` minor once, then
        // reuse it in the larger-minor DAG and the cofactor reconstruction.
        let mut minor2_a: [__m512d; 225] = [_mm512_setzero_pd(); 225];
        for r0 in 0..6 {
            for r1 in (r0 + 1)..6 {
                let row_pair = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
                for c0 in 0..6 {
                    for c1 in (c0 + 1)..6 {
                        let col_pair = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                        minor2_a[row_pair * 15 + col_pair] = _mm512_fmsub_pd(
                            d_a[r0 * 6 + c0],
                            d_a[r1 * 6 + c1],
                            _mm512_mul_pd(d_a[r0 * 6 + c1], d_a[r1 * 6 + c0]),
                        );
                    }
                }
            }
        }
        let mut second_a: [__m512d; 225] = [_mm512_setzero_pd(); 225];
        let jsl_a_t = w.aa.j_slice(0);
        let jsl_a = std::slice::from_raw_parts(jsl_a_t.as_ptr().cast::<f64>(), jsl_a_t.len());
        let mut j_a = _mm512_setzero_pd();
        for eta in 0..6 {
            for xi in (eta + 1)..6 {
                let row_pair = eta * (12 - eta - 1) / 2 + (xi - eta - 1);
                for z in 0..6 {
                    for y in (z + 1)..6 {
                        let col_pair = z * (12 - z - 1) / 2 + (y - z - 1);
                        let mut retained_rows = [0usize; 4];
                        let mut retained_cols = [0usize; 4];
                        let mut nr = 0usize;
                        let mut nc = 0usize;
                        for r in 0..6 {
                            if r != eta && r != xi {
                                retained_rows[nr] = r;
                                nr += 1;
                            }
                        }
                        for c in 0..6 {
                            if c != z && c != y {
                                retained_cols[nc] = c;
                                nc += 1;
                            }
                        }
                        let r0 = retained_rows[0];
                        let r1 = retained_rows[1];
                        let r2 = retained_rows[2];
                        let r3 = retained_rows[3];
                        let c0 = retained_cols[0];
                        let c1 = retained_cols[1];
                        let c2 = retained_cols[2];
                        let c3 = retained_cols[3];
                        let rp01 = r0 * (12 - r0 - 1) / 2 + (r1 - r0 - 1);
                        let rp23 = r2 * (12 - r2 - 1) / 2 + (r3 - r2 - 1);
                        let cp01 = c0 * (12 - c0 - 1) / 2 + (c1 - c0 - 1);
                        let cp02 = c0 * (12 - c0 - 1) / 2 + (c2 - c0 - 1);
                        let cp03 = c0 * (12 - c0 - 1) / 2 + (c3 - c0 - 1);
                        let cp12 = c1 * (12 - c1 - 1) / 2 + (c2 - c1 - 1);
                        let cp13 = c1 * (12 - c1 - 1) / 2 + (c3 - c1 - 1);
                        let cp23 = c2 * (12 - c2 - 1) / 2 + (c3 - c2 - 1);
                        let p01 = minor2_a[rp01 * 15 + cp01];
                        let p02 = minor2_a[rp01 * 15 + cp02];
                        let p03 = minor2_a[rp01 * 15 + cp03];
                        let p12 = minor2_a[rp01 * 15 + cp12];
                        let p13 = minor2_a[rp01 * 15 + cp13];
                        let p23 = minor2_a[rp01 * 15 + cp23];
                        let q01 = minor2_a[rp23 * 15 + cp01];
                        let q02 = minor2_a[rp23 * 15 + cp02];
                        let q03 = minor2_a[rp23 * 15 + cp03];
                        let q12 = minor2_a[rp23 * 15 + cp12];
                        let q13 = minor2_a[rp23 * 15 + cp13];
                        let q23 = minor2_a[rp23 * 15 + cp23];
                        let first = _mm512_fmsub_pd(p01, q23, _mm512_mul_pd(p02, q13));
                        let second01 = _mm512_fmadd_pd(p03, q12, first);
                        let second02 = _mm512_fmadd_pd(p12, q03, second01);
                        let second03 = _mm512_fnmadd_pd(p13, q02, second02);
                        let second = _mm512_fmadd_pd(p23, q01, second03);
                        second_a[row_pair * 15 + col_pair] = second;
                        let mut direct_lane = [0.0f64; 8];
                        let mut exchange_lane = [0.0f64; 8];
                        for lane in 0..8 {
                            let r_eta = rows_a[lane][eta];
                            let r_xi = rows_a[lane][xi];
                            let c_z = cols_a[lane][z];
                            let c_y = cols_a[lane][y];
                            direct_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_z) * n_a + r_xi) * n_a) + c_y);
                            exchange_lane[lane] = *jsl_a
                                .get_unchecked((((r_eta * n_a + c_y) * n_a + r_xi) * n_a) + c_z);
                        }
                        let jdiff = _mm512_sub_pd(
                            _mm512_loadu_pd(direct_lane.as_ptr()),
                            _mm512_loadu_pd(exchange_lane.as_ptr()),
                        );
                        if ((eta + xi + z + y) & 1) == 0 {
                            j_a = _mm512_fmadd_pd(second, jdiff, j_a);
                        } else {
                            j_a = _mm512_fnmadd_pd(second, jdiff, j_a);
                        }
                    }
                }
            }
        }

        // Build all first cofactors from the same second-minor array, avoiding any repeated
        // determinant algebra.
        let mut cof_a: [__m512d; 36] = [_mm512_setzero_pd(); 36];
        for eta in 0..6 {
            let r = if eta == 0 { 1usize } else { 0usize };
            let r_minor = if r < eta { r } else { r - 1 };
            for z in 0..6 {
                let mut value = _mm512_setzero_pd();
                for c in 0..6 {
                    if c == z {
                        continue;
                    }
                    let c_minor = if c < z { c } else { c - 1 };
                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                    let row_pair = row0 * (12 - row0 - 1) / 2 + (row1 - row0 - 1);
                    let col_pair = col0 * (12 - col0 - 1) / 2 + (col1 - col0 - 1);
                    let term = _mm512_mul_pd(d_a[r * 6 + c], second_a[row_pair * 15 + col_pair]);
                    value = if ((r_minor + c_minor) & 1) == 0 {
                        _mm512_add_pd(value, term)
                    } else {
                        _mm512_sub_pd(value, term)
                    };
                }
                cof_a[eta * 6 + z] = if ((eta + z) & 1) == 0 {
                    value
                } else {
                    _mm512_sub_pd(_mm512_setzero_pd(), value)
                };
            }
        }
        let mut det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
        for z in 1..6 {
            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
        }
        let fh_a_t = w.aa.fh_t_slice(0, 0);
        let vv_a_t = w.aa.v_t_slice(0, 0, 0);
        let vm_a_t = w.ab.vab_t_slice(0, 0, 0);
        let fh_a = std::slice::from_raw_parts(fh_a_t.as_ptr().cast::<f64>(), fh_a_t.len());
        let vv_a = std::slice::from_raw_parts(vv_a_t.as_ptr().cast::<f64>(), vv_a_t.len());
        let vm_a = std::slice::from_raw_parts(vm_a_t.as_ptr().cast::<f64>(), vm_a_t.len());
        let mut replacement_a = _mm512_setzero_pd();
        for z in 0..6 {
            for eta in 0..6 {
                let mut lane_values = [0.0f64; 8];
                for lane in 0..8 {
                    let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                    lane_values[lane] = *fh_a.get_unchecked(index)
                        + *vv_a.get_unchecked(index)
                        + *vm_a.get_unchecked(index);
                }
                let values = _mm512_loadu_pd(lane_values.as_ptr());
                replacement_a = _mm512_fmadd_pd(cof_a[eta * 6 + z], values, replacement_a);
            }
        }
        let det_b = _mm512_set1_pd(1.0);
        let j_b = _mm512_setzero_pd();
        let replacement_b = _mm512_setzero_pd();
        let ii_term = _mm512_setzero_pd();

        // Combine the scalar Hamiltonian intermediates once before the determinant product.
        let f0ha = *std::ptr::from_ref(&w.aa.f0h[0]).cast::<f64>();
        let v0a = *std::ptr::from_ref(&w.aa.v0[0]).cast::<f64>();
        let f0hb = *std::ptr::from_ref(&w.bb.f0h[0]).cast::<f64>();
        let v0b = *std::ptr::from_ref(&w.bb.v0[0]).cast::<f64>();
        let vab0 = *std::ptr::from_ref(&w.ab.vab0[0][0]).cast::<f64>();
        let g0 = _mm512_set1_pd(enuc + f0ha + 0.5 * v0a + f0hb + 0.5 * v0b + vab0);
        let det_ab = _mm512_mul_pd(det_a, det_b);
        let mut core = _mm512_mul_pd(g0, det_ab);
        core = _mm512_fnmadd_pd(det_b, replacement_a, core);
        core = _mm512_fnmadd_pd(det_a, replacement_b, core);
        core = _mm512_fmadd_pd(j_a, det_b, core);
        core = _mm512_fmadd_pd(j_b, det_a, core);
        core = _mm512_add_pd(core, ii_term);
        let phase_a = *std::ptr::from_ref(&w.aa.phase).cast::<f64>();
        let phase_b = *std::ptr::from_ref(&w.bb.phase).cast::<f64>();
        let ref_pref = phase_a * w.aa.tilde_s_prod * phase_b * w.bb.tilde_s_prod;
        let pref = _mm512_mul_pd(
            _mm512_loadu_pd(excitation_phase.as_ptr()),
            _mm512_set1_pd(ref_pref),
        );
        let h_v = _mm512_mul_pd(core, pref);
        let s_v = _mm512_mul_pd(det_ab, pref);
        _mm512_storeu_pd(h.as_mut_ptr(), h_v);
        _mm512_storeu_pd(s.as_mut_ptr(), s_v);
    }
}

/// Evaluate an arbitrary contraction-rank Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// This is the generic fused fallback when the predecoded four-label cache is insufficient or
/// `L_\alpha + L_\beta > 6`. Each spin contraction determinant and adjugate is evaluated once
/// and its cofactors are reused by all operator contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Full bra excitation.
/// - `w_ex`: Full ket excitation.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `scratch`: Reusable spin-resolved Wick workspace.
/// - `tol`: Numerical tolerance used when evaluating determinant adjugates.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_gen_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &Excitation,
    w_ex: &Excitation,
    excitation_phase: f64,
    enuc: f64,
    scratch: &mut WickScratchSpin<T>,
    tol: f64,
) -> (T, T) {
    let la = x_ex.alpha.holes.count_ones() as usize + w_ex.alpha.holes.count_ones() as usize;
    let lb = x_ex.beta.holes.count_ones() as usize + w_ex.beta.holes.count_ones() as usize;
    let zero = <T as From<f64>>::from(0.0);
    let half = <T as From<f64>>::from(0.5);

    prepare_same(&w.aa, &x_ex.alpha, &w_ex.alpha, &mut scratch.aa);
    prepare_same(&w.bb, &x_ex.beta, &w_ex.beta, &mut scratch.bb);

    let (det_a, have_a) = if la == 0 {
        (<T as From<f64>>::from(1.0), true)
    } else if let Some(value) = adjugate_transpose(
        scratch.aa.adjt_det.as_mut_slice(),
        scratch.aa.invs.as_mut_slice(),
        scratch.aa.lu.as_mut_slice(),
        scratch.aa.det0.as_slice(),
        la,
        tol,
    ) {
        (value, true)
    } else {
        (det(scratch.aa.det0.as_slice(), la).unwrap_or(zero), false)
    };

    let (det_b, have_b) = if lb == 0 {
        (<T as From<f64>>::from(1.0), true)
    } else if let Some(value) = adjugate_transpose(
        scratch.bb.adjt_det.as_mut_slice(),
        scratch.bb.invs.as_mut_slice(),
        scratch.bb.lu.as_mut_slice(),
        scratch.bb.det0.as_slice(),
        lb,
        tol,
    ) {
        (value, true)
    } else {
        (det(scratch.bb.det0.as_slice(), lb).unwrap_or(zero), false)
    };
    let mut same_a = zero;
    let mut same_b = zero;
    let mut h2ab = zero;

    if have_a {
        same_a = (w.aa.f0h[0] + half * w.aa.v0[0]) * det_a;
        if la > 0 {
            let fh = w.aa.fh_t_slice(0, 0);
            let vv = w.aa.v_t_slice(0, 0, 0);
            let cof = scratch.aa.adjt_det.as_slice();
            let n = w.aa.n();
            for z in 0..la {
                let base = scratch.aa.cols[z] * n;
                for eta in 0..la {
                    same_a -= cof[eta * la + z]
                        * (fh[base + scratch.aa.rows[eta]] + vv[base + scratch.aa.rows[eta]]);
                }
            }
        }
        if la >= 2 {
            let d = scratch.aa.det0.as_slice();
            let rows = scratch.aa.rows.as_slice();
            let cols = scratch.aa.cols.as_slice();
            let jsl = w.aa.j_slice(0);
            let n = w.aa.n();
            let mut minor = vec![zero; (la - 2) * (la - 2)];
            for eta in 0..la {
                for xi in (eta + 1)..la {
                    for z in 0..la {
                        for y in (z + 1)..la {
                            let mut ii = 0usize;
                            for r in 0..la {
                                if r == eta || r == xi {
                                    continue;
                                }
                                let mut jj = 0usize;
                                for c in 0..la {
                                    if c == z || c == y {
                                        continue;
                                    }
                                    minor[ii * (la - 2) + jj] = d[r * la + c];
                                    jj += 1;
                                }
                                ii += 1;
                            }
                            let second = det(&minor, la - 2).unwrap_or(zero);
                            let direct =
                                jsl[(((rows[eta] * n + cols[z]) * n + rows[xi]) * n) + cols[y]];
                            let exchange =
                                jsl[(((rows[eta] * n + cols[y]) * n + rows[xi]) * n) + cols[z]];
                            let term = second * (direct - exchange);
                            if ((eta + xi + z + y) & 1) == 0 {
                                same_a += term;
                            } else {
                                same_a -= term;
                            }
                        }
                    }
                }
            }
        }
    }

    if have_b {
        same_b = (w.bb.f0h[0] + half * w.bb.v0[0]) * det_b;
        if lb > 0 {
            let fh = w.bb.fh_t_slice(0, 0);
            let vv = w.bb.v_t_slice(0, 0, 0);
            let cof = scratch.bb.adjt_det.as_slice();
            let n = w.bb.n();
            for z in 0..lb {
                let base = scratch.bb.cols[z] * n;
                for eta in 0..lb {
                    same_b -= cof[eta * lb + z]
                        * (fh[base + scratch.bb.rows[eta]] + vv[base + scratch.bb.rows[eta]]);
                }
            }
        }
        if lb >= 2 {
            let d = scratch.bb.det0.as_slice();
            let rows = scratch.bb.rows.as_slice();
            let cols = scratch.bb.cols.as_slice();
            let jsl = w.bb.j_slice(0);
            let n = w.bb.n();
            let mut minor = vec![zero; (lb - 2) * (lb - 2)];
            for eta in 0..lb {
                for xi in (eta + 1)..lb {
                    for z in 0..lb {
                        for y in (z + 1)..lb {
                            let mut ii = 0usize;
                            for r in 0..lb {
                                if r == eta || r == xi {
                                    continue;
                                }
                                let mut jj = 0usize;
                                for c in 0..lb {
                                    if c == z || c == y {
                                        continue;
                                    }
                                    minor[ii * (lb - 2) + jj] = d[r * lb + c];
                                    jj += 1;
                                }
                                ii += 1;
                            }
                            let second = det(&minor, lb - 2).unwrap_or(zero);
                            let direct =
                                jsl[(((rows[eta] * n + cols[z]) * n + rows[xi]) * n) + cols[y]];
                            let exchange =
                                jsl[(((rows[eta] * n + cols[y]) * n + rows[xi]) * n) + cols[z]];
                            let term = second * (direct - exchange);
                            if ((eta + xi + z + y) & 1) == 0 {
                                same_b += term;
                            } else {
                                same_b -= term;
                            }
                        }
                    }
                }
            }
        }
    }

    if have_a && have_b {
        h2ab = w.ab.vab0[0][0] * det_a * det_b;
        let n = w.ab.n();
        if la > 0 {
            let va = w.ab.vab_t_slice(0, 0, 0);
            let cofa = scratch.aa.adjt_det.as_slice();
            let mut replacement = zero;
            for z in 0..la {
                let base = scratch.aa.cols[z] * n;
                for eta in 0..la {
                    replacement += cofa[eta * la + z] * va[base + scratch.aa.rows[eta]];
                }
            }
            h2ab -= replacement * det_b;
        }
        if lb > 0 {
            let vb = w.ab.vba_t_slice(0, 0, 0);
            let cofb = scratch.bb.adjt_det.as_slice();
            let mut replacement = zero;
            for y in 0..lb {
                let base = scratch.bb.cols[y] * n;
                for xi in 0..lb {
                    replacement += cofb[xi * lb + y] * vb[base + scratch.bb.rows[xi]];
                }
            }
            h2ab -= replacement * det_a;
        }
        if la > 0 && lb > 0 {
            let iisl = w.ab.iiab_slice(0, 0, 0, 0);
            let cofa = scratch.aa.adjt_det.as_slice();
            let cofb = scratch.bb.adjt_det.as_slice();
            for z in 0..la {
                for eta in 0..la {
                    let base_a = (scratch.aa.rows[eta] * n + scratch.aa.cols[z]) * n * n;
                    let mut inner = zero;
                    for y in 0..lb {
                        for xi in 0..lb {
                            inner += cofb[xi * lb + y]
                                * iisl[base_a + scratch.bb.rows[xi] * n + scratch.bb.cols[y]];
                        }
                    }
                    h2ab += cofa[eta * la + z] * inner;
                }
            }
        }
    }

    let sa_pref = w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod);
    let sb_pref = w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod);
    let excitation = <T as From<f64>>::from(excitation_phase);
    let s = excitation * sa_pref * sb_pref * det_a * det_b;
    let mut h = <T as From<f64>>::from(enuc) * s;
    h += excitation * sa_pref * sb_pref * (same_a * det_b + same_b * det_a + h2ab);
    (h, s)
}

/// Evaluate the fused Hamiltonian and overlap for nonzero reference-pair nullity.
/// The alpha- and beta-spin same-spin distribution sums are traversed once each and reuse every
/// mixed contraction determinant and adjugate across overlap, one-electron and same-spin
/// two-electron terms. The different-spin term is then evaluated in its own natural distribution
/// space. This generic path supports all excitation ranks represented by `Excitation`.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Full bra excitation.
/// - `w_ex`: Full ket excitation.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy; retained for the common interface.
/// - `scratch`: Reusable spin-resolved Wick workspace.
/// - `tol`: Numerical tolerance used by generic determinant and adjugate evaluation.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_gen_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &Excitation,
    w_ex: &Excitation,
    excitation_phase: f64,
    _enuc: f64,
    scratch: &mut WickScratchSpin<T>,
    tol: f64,
) -> (T, T) {
    let la = x_ex.alpha.holes.count_ones() as usize + w_ex.alpha.holes.count_ones() as usize;
    let lb = x_ex.beta.holes.count_ones() as usize + w_ex.beta.holes.count_ones() as usize;

    let zero = <T as From<f64>>::from(0.0);
    let one = <T as From<f64>>::from(1.0);
    let two = <T as From<f64>>::from(2.0);
    let half = <T as From<f64>>::from(0.5);

    // Construct the all-zero and all-one endpoint contraction determinants once per spin.
    prepare_same(&w.aa, &x_ex.alpha, &w_ex.alpha, &mut scratch.aa);
    prepare_same(&w.bb, &x_ex.beta, &w_ex.beta, &mut scratch.bb);

    let mut sa = zero;
    let mut h1a = zero;
    let mut h2aa = zero;
    let na = w.aa.n();

    // Embed overlap and one-body distributions in the two-operator same-spin distribution.
    // `m_2 = 0` identifies one-body terms and `m_1 = m_2 = 0` identifies overlap terms.
    mix_dets_same(&w.aa, la, 2, &mut scratch.aa, |bits, scratch| {
        let m1 = bit(bits, 0);
        let m2 = bit(bits, 1);

        let det_det = if la == 0 {
            Some(one)
        } else {
            adjugate_transpose_generic(
                scratch.adjt_det.as_mut_slice(),
                scratch.det_mix.as_slice(),
                la,
                tol,
            )
        };

        if let Some(det_det) = det_det {
            // Accumulate the overlap only for the embedded distribution with both operator bits
            // zero.
            if m1 == 0 && m2 == 0 {
                sa += det_det;
            }

            // Accumulate the one-electron term when the second operator assignment is zero.
            if m2 == 0 {
                let mut contrib = det_det * w.aa.f0h[m1];
                let f0 = w.aa.fh_t_slice(m1, 0);
                let f1 = w.aa.fh_t_slice(m1, 1);

                for k in 0..la {
                    let mk = bit(bits, k + 2);
                    let ck = scratch.cols[k];
                    let fsl = if mk == 0 { f0 } else { f1 };
                    let base = ck * na;

                    let corr = column_replacement_correction(
                        la,
                        scratch.det_mix.as_slice(),
                        scratch.adjt_det.as_slice(),
                        k,
                        |r| fsl[base + scratch.rows[r]],
                    );

                    contrib -= det_det + corr;
                }

                h1a += contrib;
            }

            // Start the same-spin two-electron term from its scalar intermediate.
            let mut contrib = w.aa.v0[m1 + m2] * det_det;
            let v0 = w.aa.v_t_slice(m1, m2, 0);
            let v1 = w.aa.v_t_slice(m1, m2, 1);

            // Reuse the same cofactor matrix for every one-column two-electron replacement.
            for k in 0..la {
                let mk = bit(bits, k + 2);
                let ck = scratch.cols[k];
                let vsl = if mk == 0 { v0 } else { v1 };
                let base = ck * na;

                let corr = column_replacement_correction(
                    la,
                    scratch.det_mix.as_slice(),
                    scratch.adjt_det.as_slice(),
                    k,
                    |r| vsl[base + scratch.rows[r]],
                );

                contrib -= two * (det_det + corr);
            }

            if la >= 2 {
                let layout = ReplacementLayout {
                    n: na,
                    rows: scratch.rows.as_slice(),
                    cols: scratch.cols.as_slice(),
                };

                // Evaluate the two-column `\mathcal J` contribution from first minors of the
                // same mixed determinant. The existing replacement helpers preserve the stored
                // pair-exchange symmetry and assignment ordering of the original evaluator.
                for i in 0..la {
                    for j in 0..la {
                        let phase = if ((i + j) & 1) == 0 { one } else { -one };
                        let ri = scratch.rows[i];
                        let cj = scratch.cols[j];
                        let mj = bit(bits, j + 2);

                        minor_adjt(
                            scratch.det_mix.as_slice(),
                            Minor {
                                l: la,
                                row: i,
                                col: j,
                            },
                            &mut scratch.det_mix2,
                            &mut scratch.adjt_det2,
                            tol,
                            |lm1, _det_minor, cof_minor, _det_det2| {
                                for k2 in 0..lm1 {
                                    let k_full = if k2 < j { k2 } else { k2 + 1 };
                                    let mk = bit(bits, k_full + 2);
                                    let (slot, swap) = jslot(m1, m2, mk, mj);
                                    let jsl = w.aa.j_slice(slot);

                                    let det_repl =
                                        column_replacement_det(lm1, cof_minor, k2, |r| {
                                            j_replacement(
                                                jsl,
                                                layout,
                                                DetIndex { row: i, col: j },
                                                DetIndex { row: r, col: k2 },
                                                DetIndex { row: ri, col: cj },
                                                swap,
                                            )
                                        });

                                    contrib += phase * det_repl;
                                }
                            },
                        );
                    }
                }
            }

            h2aa += contrib;
        } else if m1 == 0 && m2 == 0 {
            // Preserve overlap evaluation when the adjugate path rejects a singular mixed
            // determinant.
            sa += det(scratch.det_mix.as_slice(), la).unwrap_or(zero);
        }
    });

    let mut sb = zero;
    let mut h1b = zero;
    let mut h2bb = zero;
    let nb = w.bb.n();

    // Repeat the same fused distribution traversal for the beta-spin contraction determinant.
    mix_dets_same(&w.bb, lb, 2, &mut scratch.bb, |bits, scratch| {
        let m1 = bit(bits, 0);
        let m2 = bit(bits, 1);

        let det_det = if lb == 0 {
            Some(one)
        } else {
            adjugate_transpose_generic(
                scratch.adjt_det.as_mut_slice(),
                scratch.det_mix.as_slice(),
                lb,
                tol,
            )
        };

        if let Some(det_det) = det_det {
            if m1 == 0 && m2 == 0 {
                sb += det_det;
            }

            if m2 == 0 {
                let mut contrib = det_det * w.bb.f0h[m1];
                let f0 = w.bb.fh_t_slice(m1, 0);
                let f1 = w.bb.fh_t_slice(m1, 1);

                for k in 0..lb {
                    let mk = bit(bits, k + 2);
                    let ck = scratch.cols[k];
                    let fsl = if mk == 0 { f0 } else { f1 };
                    let base = ck * nb;

                    let corr = column_replacement_correction(
                        lb,
                        scratch.det_mix.as_slice(),
                        scratch.adjt_det.as_slice(),
                        k,
                        |r| fsl[base + scratch.rows[r]],
                    );

                    contrib -= det_det + corr;
                }

                h1b += contrib;
            }

            let mut contrib = w.bb.v0[m1 + m2] * det_det;
            let v0 = w.bb.v_t_slice(m1, m2, 0);
            let v1 = w.bb.v_t_slice(m1, m2, 1);

            for k in 0..lb {
                let mk = bit(bits, k + 2);
                let ck = scratch.cols[k];
                let vsl = if mk == 0 { v0 } else { v1 };
                let base = ck * nb;

                let corr = column_replacement_correction(
                    lb,
                    scratch.det_mix.as_slice(),
                    scratch.adjt_det.as_slice(),
                    k,
                    |r| vsl[base + scratch.rows[r]],
                );

                contrib -= two * (det_det + corr);
            }

            if lb >= 2 {
                let layout = ReplacementLayout {
                    n: nb,
                    rows: scratch.rows.as_slice(),
                    cols: scratch.cols.as_slice(),
                };

                for i in 0..lb {
                    for j in 0..lb {
                        let phase = if ((i + j) & 1) == 0 { one } else { -one };
                        let ri = scratch.rows[i];
                        let cj = scratch.cols[j];
                        let mj = bit(bits, j + 2);

                        minor_adjt(
                            scratch.det_mix.as_slice(),
                            Minor {
                                l: lb,
                                row: i,
                                col: j,
                            },
                            &mut scratch.det_mix2,
                            &mut scratch.adjt_det2,
                            tol,
                            |lm1, _det_minor, cof_minor, _det_det2| {
                                for k2 in 0..lm1 {
                                    let k_full = if k2 < j { k2 } else { k2 + 1 };
                                    let mk = bit(bits, k_full + 2);
                                    let (slot, swap) = jslot(m1, m2, mk, mj);
                                    let jsl = w.bb.j_slice(slot);

                                    let det_repl =
                                        column_replacement_det(lm1, cof_minor, k2, |r| {
                                            j_replacement(
                                                jsl,
                                                layout,
                                                DetIndex { row: i, col: j },
                                                DetIndex { row: r, col: k2 },
                                                DetIndex { row: ri, col: cj },
                                                swap,
                                            )
                                        });

                                    contrib += phase * det_repl;
                                }
                            },
                        );
                    }
                }
            }

            h2bb += contrib;
        } else if m1 == 0 && m2 == 0 {
            sb += det(scratch.det_mix.as_slice(), lb).unwrap_or(zero);
        }
    });

    let mut h2ab = zero;

    // The mixed-spin operator has one operator assignment per spin, so its distribution space is
    // different from the same-spin `L + 2` traversal above and is evaluated once in its natural
    // form.
    scratch.diff.ensure_diff(la, lb);

    let rows_a = scratch.aa.rows.as_slice();
    let cols_a = scratch.aa.cols.as_slice();
    let rows_b = scratch.bb.rows.as_slice();
    let cols_b = scratch.bb.cols.as_slice();
    let deta0 = scratch.aa.det0.as_slice();
    let deta1 = scratch.aa.det1.as_slice();
    let detb0 = scratch.bb.det0.as_slice();
    let detb1 = scratch.bb.det1.as_slice();

    let layout_b = ReplacementLayout {
        n: w.ab.n(),
        rows: rows_b,
        cols: cols_b,
    };

    get_det_adjt_diff(
        w,
        (la, lb),
        &mut scratch.diff,
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
            let ma0 = bit(bits_a, 0);
            let mb0 = bit(bits_b, 0);
            let mut contrib = w.ab.vab0[ma0][mb0] * det_deta * det_detb;
            let n = w.ab.n();

            let vab0 = w.ab.vab_t_slice(ma0, mb0, 0);
            let vab1 = w.ab.vab_t_slice(ma0, mb0, 1);

            for (k, &ck) in cols_a.iter().enumerate().take(la) {
                let mak = bit(bits_a, k + 1);
                let vsl = if mak == 0 { vab0 } else { vab1 };
                let base = ck * n;

                let det_repl = column_replacement_det(la, scratch.adjt_deta.as_slice(), k, |r| {
                    vsl[base + rows_a[r]]
                });

                contrib -= det_repl * det_detb;
            }

            let vba0 = w.ab.vba_t_slice(mb0, ma0, 0);
            let vba1 = w.ab.vba_t_slice(mb0, ma0, 1);

            for (k, &ck) in cols_b.iter().enumerate().take(lb) {
                let mbk = bit(bits_b, k + 1);
                let vsl = if mbk == 0 { vba0 } else { vba1 };
                let base = ck * n;

                let det_repl = column_replacement_det(lb, scratch.adjt_detb.as_slice(), k, |r| {
                    vsl[base + rows_b[r]]
                });

                contrib -= det_repl * det_deta;
            }

            // Contract the alpha cofactor with a beta-column replacement of `\mathcal{II}`.
            // This reproduces the existing assignment ordering while avoiding a separate tensor
            // pass.
            for i in 0..la {
                let ra = rows_a[i];

                for j in 0..la {
                    let ca = cols_a[j];
                    let cofa = scratch.adjt_deta.as_slice()[i * la + j];
                    let ma1 = bit(bits_a, j + 1);

                    for k in 0..lb {
                        let mbk = bit(bits_b, k + 1);
                        let iisl = w.ab.iiab_slice(ma0, ma1, mb0, mbk);

                        let det_repl =
                            column_replacement_det(lb, scratch.adjt_detb.as_slice(), k, |r| {
                                ii_replacement(
                                    iisl,
                                    layout_b,
                                    DetIndex { row: r, col: k },
                                    DetIndex { row: ra, col: ca },
                                    true,
                                )
                            });

                        contrib += cofa * det_repl;
                    }
                }
            }

            h2ab += contrib;
        },
    );

    let ref_pref = (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
        * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod));
    let pref = <T as From<f64>>::from(excitation_phase) * ref_pref;

    let s = pref * sa * sb;
    let h = pref * (h1a * sb + h1b * sa + half * h2aa * sb + half * h2bb * sa + h2ab);

    (h, s)
}
