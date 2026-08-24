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
use crate::time_call;
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
    ex: (&Excitation, &Excitation),
    cache: (&ExcitationCache, &ExcitationCache),
    excitation_phase: f64,
    enuc: f64,
    scratch: &mut WickScratchSpin<T>,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_prepared,
        {
            let (x_ex, w_ex) = ex;
            let (x_cache, w_cache) = cache;

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
    )
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
    basis: (&[DetState<f64>], &[ReducedTwoSpinDetState]),
    requests: &[(usize, usize, usize)],
    enuc: f64,
    scratch: &mut WickScratchSpin<f64>,
    tol: f64,
    out: &mut [(f64, f64)],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_prepared_batched,
        {
            let (basis, reduced_basis) = basis;

            #[cfg(target_arch = "x86_64")]
            if w.aa.m == 0 && w.bb.m == 0 {
                if std::arch::is_x86_feature_detected!("avx512f") {
                    // Bin requests by fixed `(L_alpha,L_beta)` so each SIMD packet evaluates the same
                    // GNME determinant/cofactor algebra with lane-local excitation labels and phases.
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
                            // Triangular indexing stores every supported rank pair with
                            // L_alpha + L_beta <= 6 in a compact 28-bin table.
                            let bin = la * (15 - la) / 2 + lb;
                            let count = counts[bin];

                            x_bins[bin][count] = x_cache;
                            w_bins[bin][count] = w_cache;
                            phases[bin][count] = x_det.phase * w_det.phase;
                            outputs[bin][count] = output;
                            counts[bin] += 1;

                            if counts[bin] == 8 {
                                // A full AVX-512 packet evaluates eight independent determinant pairs
                                // using the same constant-rank Hamiltonian formula.
                                let mut h = [0.0f64; 8];
                                let mut s = [0.0f64; 8];

                                unsafe {
                                    xw_hamiltonian_overlap_m0_prepared_f64x8(
                                        w,
                                        (la, lb),
                                        (&x_bins[bin], &w_bins[bin]),
                                        &phases[bin],
                                        enuc,
                                        (&mut h, &mut s),
                                    );
                                }

                                for lane in 0..8 {
                                    out[outputs[bin][lane]] = (h[lane], s[lane]);
                                }
                                counts[bin] = 0;
                            }
                        } else {
                            // Unsupported rank pairs keep the scalar prepared path, which evaluates
                            // the same GNME expression without fixed-width SIMD batching.
                            let x_state = &basis[a];
                            let w_state = &basis[b];

                            out[output] = xw_hamiltonian_overlap_prepared(
                                w,
                                (&x_state.excitation, &w_state.excitation),
                                (&x_cache, &w_cache),
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

                            // Pad incomplete SIMD packets with a valid lane; only the original `count`
                            // outputs are copied back, so padding contributes no matrix elements.
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
                                    (la, lb),
                                    (&x_bins[bin], &w_bins[bin]),
                                    &phases[bin],
                                    enuc,
                                    (&mut h, &mut s),
                                );
                            }

                            for lane in 0..count {
                                out[outputs[bin][lane]] = (h[lane], s[lane]);
                            }
                        }
                    }
                    return;
                }

                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    // AVX2 follows the same fixed-rank binning as AVX-512, with four determinant
                    // pairs per packet instead of eight.
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
                            // The bin index depends only on the compile-time Hamiltonian ranks, not
                            // on the lane-local orbital labels.
                            let bin = la * (15 - la) / 2 + lb;
                            let count = counts[bin];

                            x_bins[bin][count] = x_cache;
                            w_bins[bin][count] = w_cache;
                            phases[bin][count] = x_det.phase * w_det.phase;
                            outputs[bin][count] = output;
                            counts[bin] += 1;

                            if counts[bin] == 4 {
                                // A full AVX2 packet evaluates four independent fixed-rank GNME pairs.
                                let mut h = [0.0f64; 4];
                                let mut s = [0.0f64; 4];

                                unsafe {
                                    xw_hamiltonian_overlap_m0_prepared_f64x4(
                                        w,
                                        (la, lb),
                                        (&x_bins[bin], &w_bins[bin]),
                                        &phases[bin],
                                        enuc,
                                        (&mut h, &mut s),
                                    );
                                }

                                for lane in 0..4 {
                                    out[outputs[bin][lane]] = (h[lane], s[lane]);
                                }
                                counts[bin] = 0;
                            }
                        } else {
                            // Fall back to the prepared scalar evaluator when fixed-rank SIMD does not
                            // cover the determinant pair.
                            let x_state = &basis[a];
                            let w_state = &basis[b];

                            out[output] = xw_hamiltonian_overlap_prepared(
                                w,
                                (&x_state.excitation, &w_state.excitation),
                                (&x_cache, &w_cache),
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

                            // Pad the tail packet so the SIMD kernel can run at fixed width; only the
                            // requested lanes are written back.
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
                                    (la, lb),
                                    (&x_bins[bin], &w_bins[bin]),
                                    &phases[bin],
                                    enuc,
                                    (&mut h, &mut s),
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
                // Baseline path for nonzero zero-overlap counts or machines without the required SIMD:
                // evaluate the full prepared Hamiltonian/overlap expression one determinant pair at a time.
                let x_det = &reduced_basis[a];
                let w_det = &reduced_basis[b];
                let x_state = &basis[a];
                let w_state = &basis[b];

                out[output] = xw_hamiltonian_overlap_prepared(
                    w,
                    (&x_state.excitation, &w_state.excitation),
                    (&x_det.excitation_cache, &w_det.excitation_cache),
                    x_det.phase * w_det.phase,
                    enuc,
                    scratch,
                    tol,
                );
            }
        }
    )
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
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_m0_prepared,
        {
            // Dispatch by `(L_alpha,L_beta)` so each fixed rank is monomorphised into the same
            // constant-time Hamiltonian/cofactor formula with stack storage sized to that rank.
            match (la, lb) {
                (0, 0) => xw_hamiltonian_overlap_m0_prepared_const::<T, 0, 0, 1, 1, 1, 1, 1, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (0, 1) => xw_hamiltonian_overlap_m0_prepared_const::<T, 0, 1, 1, 1, 1, 1, 1, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (0, 2) => xw_hamiltonian_overlap_m0_prepared_const::<T, 0, 2, 1, 4, 1, 1, 1, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (0, 3) => xw_hamiltonian_overlap_m0_prepared_const::<T, 0, 3, 1, 9, 1, 3, 1, 9>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (0, 4) => xw_hamiltonian_overlap_m0_prepared_const::<T, 0, 4, 1, 16, 1, 6, 1, 36>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (0, 5) => {
                    xw_hamiltonian_overlap_m0_prepared_const::<T, 0, 5, 1, 25, 1, 10, 1, 100>(
                        w,
                        x_ex,
                        w_ex,
                        excitation_phase,
                        enuc,
                    )
                }
                (0, 6) => {
                    xw_hamiltonian_overlap_m0_prepared_const::<T, 0, 6, 1, 36, 1, 15, 1, 225>(
                        w,
                        x_ex,
                        w_ex,
                        excitation_phase,
                        enuc,
                    )
                }
                (1, 0) => xw_hamiltonian_overlap_m0_prepared_const::<T, 1, 0, 1, 1, 1, 1, 1, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (1, 1) => xw_hamiltonian_overlap_m0_prepared_const::<T, 1, 1, 1, 1, 1, 1, 1, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (1, 2) => xw_hamiltonian_overlap_m0_prepared_const::<T, 1, 2, 1, 4, 1, 1, 1, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (1, 3) => xw_hamiltonian_overlap_m0_prepared_const::<T, 1, 3, 1, 9, 1, 3, 1, 9>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (1, 4) => xw_hamiltonian_overlap_m0_prepared_const::<T, 1, 4, 1, 16, 1, 6, 1, 36>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (1, 5) => {
                    xw_hamiltonian_overlap_m0_prepared_const::<T, 1, 5, 1, 25, 1, 10, 1, 100>(
                        w,
                        x_ex,
                        w_ex,
                        excitation_phase,
                        enuc,
                    )
                }
                (2, 0) => xw_hamiltonian_overlap_m0_prepared_const::<T, 2, 0, 4, 1, 1, 1, 1, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (2, 1) => xw_hamiltonian_overlap_m0_prepared_const::<T, 2, 1, 4, 1, 1, 1, 1, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (2, 2) => xw_hamiltonian_overlap_m0_prepared_const::<T, 2, 2, 4, 4, 1, 1, 1, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (2, 3) => xw_hamiltonian_overlap_m0_prepared_const::<T, 2, 3, 4, 9, 1, 3, 1, 9>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (2, 4) => xw_hamiltonian_overlap_m0_prepared_const::<T, 2, 4, 4, 16, 1, 6, 1, 36>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (3, 0) => xw_hamiltonian_overlap_m0_prepared_const::<T, 3, 0, 9, 1, 3, 1, 9, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (3, 1) => xw_hamiltonian_overlap_m0_prepared_const::<T, 3, 1, 9, 1, 3, 1, 9, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (3, 2) => xw_hamiltonian_overlap_m0_prepared_const::<T, 3, 2, 9, 4, 3, 1, 9, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (3, 3) => xw_hamiltonian_overlap_m0_prepared_const::<T, 3, 3, 9, 9, 3, 3, 9, 9>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (4, 0) => xw_hamiltonian_overlap_m0_prepared_const::<T, 4, 0, 16, 1, 6, 1, 36, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (4, 1) => xw_hamiltonian_overlap_m0_prepared_const::<T, 4, 1, 16, 1, 6, 1, 36, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (4, 2) => xw_hamiltonian_overlap_m0_prepared_const::<T, 4, 2, 16, 4, 6, 1, 36, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                ),
                (5, 0) => {
                    xw_hamiltonian_overlap_m0_prepared_const::<T, 5, 0, 25, 1, 10, 1, 100, 1>(
                        w,
                        x_ex,
                        w_ex,
                        excitation_phase,
                        enuc,
                    )
                }
                (5, 1) => {
                    xw_hamiltonian_overlap_m0_prepared_const::<T, 5, 1, 25, 1, 10, 1, 100, 1>(
                        w,
                        x_ex,
                        w_ex,
                        excitation_phase,
                        enuc,
                    )
                }
                (6, 0) => {
                    xw_hamiltonian_overlap_m0_prepared_const::<T, 6, 0, 36, 1, 15, 1, 225, 1>(
                        w,
                        x_ex,
                        w_ex,
                        excitation_phase,
                        enuc,
                    )
                }
                _ => unreachable!(),
            }
        }
    )
}

/// Evaluate the fixed-rank `(L_\alpha, L_\beta)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// For each spin sector `\sigma`, the overlap determinant is
/// `D^\sigma_{ij}=X^{(0)}_{r_i c_j}` for `i >= j` and
/// `D^\sigma_{ij}=Y^{(0)}_{r_i c_j}` for `i < j`.
/// The overlap contribution is
/// `S = p\det\mathbf D_{\alpha,\mathrm{ov}}\det\mathbf D_{\beta,\mathrm{ov}}`, with
/// `p` the excitation phase times the two reduced reference-overlap factors.
/// The one-body contribution in each spin sector is the `m = 0` form
/// `F^\sigma = F^\sigma_0\det\mathbf D_{\sigma,\mathrm{ov}}`
/// `- \sum_z\det\mathbf D_{\sigma,\mathrm{ov}}^{z\rightarrow\boldsymbol{\mathcal F}_z}`.
/// The same-spin two-body contribution uses the three Laplace classes:
/// `V^\sigma_0\det\mathbf D_{\sigma,\mathrm{ov}}`;
/// `-2\sum_z\det\mathbf D_{\sigma,\mathrm{ov}}^{z\rightarrow\boldsymbol{\mathcal V}_z}`,
/// and the second-minor contraction
/// `\sum_{z<y}\sum_{\eta<\xi}\phi_{\eta\xi}^{zy}`
/// `\mathcal J^\sigma_{\eta z,\xi y}\det\mathbf D_{\sigma,\mathrm{ov}}[\eta,\xi|z,y]`.
/// The mixed-spin contribution is
/// `V^{\alpha\beta}_0\det\mathbf D_{\alpha,\mathrm{ov}}\det\mathbf D_{\beta,\mathrm{ov}}`
/// minus the alpha and beta one-column replacements, plus the cofactor contraction
/// `\sum_{z,y}\sum_{\eta,\xi}\operatorname{cof}[\mathbf D_{\alpha,\mathrm{ov}}]_{\eta z}`
/// `\mathcal {II}_{\eta z,\xi y}`
/// `\operatorname{cof}[\mathbf D_{\beta,\mathrm{ov}}]_{\xi y}`.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_prepared_const<
    T: NOCIScalar,
    const LA: usize,
    const LB: usize,
    const DA: usize,
    const DB: usize,
    const PA: usize,
    const PB: usize,
    const SA: usize,
    const SB: usize,
>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_m0_prepared_const,
        {
            let zero = <T as From<f64>>::from(0.0);
            let one = <T as From<f64>>::from(1.0);
            let half = <T as From<f64>>::from(0.5);
            let mut rows_a = [0usize; 6];
            let mut cols_a = [0usize; 6];
            let mut d_a = [zero; DA];
            let mut cof_a = [zero; DA];
            let mut second_a = [zero; SA];
            let mut det_a = one;
            let mut j_a = zero;
            let mut replacement_a = zero;

            // Build the alpha rows r_eta and columns c_z for D_{alpha,ov}; x-excitations
            // contribute (a,i) labels and w-excitations contribute (j,b) labels.
            if LA > 0 {
                let nocc = w.aa.nocc;
                let nvirt = w.aa.nmo - nocc;
                let x_rank = usize::from(x_ex.alpha.rank);
                let x_indices = &x_ex.alpha.indices;
                let w_indices = &w_ex.alpha.indices;
                for i in 0..x_rank {
                    rows_a[i] = usize::from(x_indices[4 + i]) - nocc;
                    cols_a[i] = usize::from(x_indices[i]);
                }
                for i in x_rank..LA {
                    let k = i - x_rank;
                    rows_a[i] = nvirt + usize::from(w_indices[k]);
                    cols_a[i] = usize::from(w_indices[4 + k]);
                }

                // Form D_{alpha,ov}[eta,z] from the m_i = 0 fundamental contractions:
                // X^{(0)}_{r_eta c_z} on and below the diagonal, Y^{(0)}_{r_eta c_z} above it.
                let n = w.aa.n();
                let x0 = w.aa.x_slice(0);
                let y0 = w.aa.y_slice(0);
                for i in 0..LA {
                    let row = rows_a[i] * n;
                    for j in 0..LA {
                        d_a[i * LA + j] = if i >= j {
                            x0[row + cols_a[j]]
                        } else {
                            y0[row + cols_a[j]]
                        };
                    }
                }
                if LA == 1 {
                    cof_a[0] = one;
                    det_a = d_a[0];
                } else {
                    // Same-spin double Laplace class C_3:
                    // sum_{z<y,eta<xi} phi J_{eta z,xi y} det D_{alpha,ov}[eta,xi|z,y].
                    let pairs_a = LA * (LA - 1) / 2;
                    let jsl = w.aa.j_slice(0);
                    let n2 = n * n;
                    let n3 = n2 * n;
                    for eta in 0..LA {
                        for xi in (eta + 1)..LA {
                            let row_pair = eta * (2 * LA - eta - 1) / 2 + (xi - eta - 1);
                            for z in 0..LA {
                                for y in (z + 1)..LA {
                                    let col_pair = z * (2 * LA - z - 1) / 2 + (y - z - 1);
                                    let mut minor = [zero; 16];
                                    let mut ii = 0usize;
                                    for r in 0..LA {
                                        if r == eta || r == xi {
                                            continue;
                                        }
                                        let mut jj = 0usize;
                                        for c in 0..LA {
                                            if c == z || c == y {
                                                continue;
                                            }
                                            minor[ii * (LA - 2) + jj] = d_a[r * LA + c];
                                            jj += 1;
                                        }
                                        ii += 1;
                                    }
                                    let second_rank = LA - 2;
                                    let second =
                                        det(&minor[..second_rank * second_rank], second_rank)
                                            .unwrap_or(zero);
                                    second_a[row_pair * pairs_a + col_pair] = second;
                                    let direct_base =
                                        rows_a[eta] * n3 + cols_a[z] * n2 + rows_a[xi] * n;
                                    let exchange_base =
                                        rows_a[eta] * n3 + cols_a[y] * n2 + rows_a[xi] * n;
                                    let term = second
                                        * (jsl[direct_base + cols_a[y]]
                                            - jsl[exchange_base + cols_a[z]]);
                                    if ((eta + xi + z + y) & 1) == 0 {
                                        j_a += term;
                                    } else {
                                        j_a -= term;
                                    }
                                }
                            }
                        }
                    }

                    // Reconstruct cof[D_{alpha,ov}]_{eta z}=(-1)^{eta+z} det D[eta|z]
                    // from the second-minor table, then expand det D along the first row.
                    for eta in 0..LA {
                        let r = if eta == 0 { 1usize } else { 0usize };
                        let r_minor = if r < eta { r } else { r - 1 };
                        for z in 0..LA {
                            let mut value = zero;
                            for c in 0..LA {
                                if c == z {
                                    continue;
                                }
                                let c_minor = if c < z { c } else { c - 1 };
                                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                                let row_pair = row0 * (2 * LA - row0 - 1) / 2 + (row1 - row0 - 1);
                                let col_pair = col0 * (2 * LA - col0 - 1) / 2 + (col1 - col0 - 1);
                                let term =
                                    d_a[r * LA + c] * second_a[row_pair * pairs_a + col_pair];
                                if ((r_minor + c_minor) & 1) == 0 {
                                    value += term;
                                } else {
                                    value -= term;
                                }
                            }
                            cof_a[eta * LA + z] = if ((eta + z) & 1) == 0 { value } else { -value };
                        }
                    }
                    det_a = d_a[0] * cof_a[0];
                    for z in 1..LA {
                        det_a += d_a[z] * cof_a[z];
                    }
                }

                // One-column Hamiltonian replacements:
                // sum_z det D_{alpha,ov}^{z -> H_z} = sum_{eta,z} cof[D]_{eta z} H_{eta z}.
                let hcol0 = w.aa.hcol0_t_slice();
                for z in 0..LA {
                    let base = cols_a[z] * n;
                    for eta in 0..LA {
                        replacement_a += cof_a[eta * LA + z] * hcol0[base + rows_a[eta]];
                    }
                }
            }
            let mut rows_b = [0usize; 6];
            let mut cols_b = [0usize; 6];
            let mut d_b = [zero; DB];
            let mut cof_b = [zero; DB];
            let mut second_b = [zero; SB];
            let mut det_b = one;
            let mut j_b = zero;
            let mut replacement_b = zero;

            // Build the beta rows r_eta and columns c_z for D_{beta,ov}; x-excitations
            // contribute (a,i) labels and w-excitations contribute (j,b) labels.
            if LB > 0 {
                let nocc = w.bb.nocc;
                let nvirt = w.bb.nmo - nocc;
                let x_rank = usize::from(x_ex.beta.rank);
                let x_indices = &x_ex.beta.indices;
                let w_indices = &w_ex.beta.indices;
                for i in 0..x_rank {
                    rows_b[i] = usize::from(x_indices[4 + i]) - nocc;
                    cols_b[i] = usize::from(x_indices[i]);
                }
                for i in x_rank..LB {
                    let k = i - x_rank;
                    rows_b[i] = nvirt + usize::from(w_indices[k]);
                    cols_b[i] = usize::from(w_indices[4 + k]);
                }

                // Form D_{beta,ov}[eta,z] from the m_i = 0 fundamental contractions:
                // X^{(0)}_{r_eta c_z} on and below the diagonal, Y^{(0)}_{r_eta c_z} above it.
                let n = w.bb.n();
                let x0 = w.bb.x_slice(0);
                let y0 = w.bb.y_slice(0);
                for i in 0..LB {
                    let row = rows_b[i] * n;
                    for j in 0..LB {
                        d_b[i * LB + j] = if i >= j {
                            x0[row + cols_b[j]]
                        } else {
                            y0[row + cols_b[j]]
                        };
                    }
                }
                if LB == 1 {
                    cof_b[0] = one;
                    det_b = d_b[0];
                } else {
                    // Same-spin double Laplace class C_3:
                    // sum_{z<y,eta<xi} phi J_{eta z,xi y} det D_{beta,ov}[eta,xi|z,y].
                    let pairs_b = LB * (LB - 1) / 2;
                    let jsl = w.bb.j_slice(0);
                    let n2 = n * n;
                    let n3 = n2 * n;
                    for eta in 0..LB {
                        for xi in (eta + 1)..LB {
                            let row_pair = eta * (2 * LB - eta - 1) / 2 + (xi - eta - 1);
                            for z in 0..LB {
                                for y in (z + 1)..LB {
                                    let col_pair = z * (2 * LB - z - 1) / 2 + (y - z - 1);
                                    let mut minor = [zero; 16];
                                    let mut ii = 0usize;
                                    for r in 0..LB {
                                        if r == eta || r == xi {
                                            continue;
                                        }
                                        let mut jj = 0usize;
                                        for c in 0..LB {
                                            if c == z || c == y {
                                                continue;
                                            }
                                            minor[ii * (LB - 2) + jj] = d_b[r * LB + c];
                                            jj += 1;
                                        }
                                        ii += 1;
                                    }
                                    let second_rank = LB - 2;
                                    let second =
                                        det(&minor[..second_rank * second_rank], second_rank)
                                            .unwrap_or(zero);
                                    second_b[row_pair * pairs_b + col_pair] = second;
                                    let direct_base =
                                        rows_b[eta] * n3 + cols_b[z] * n2 + rows_b[xi] * n;
                                    let exchange_base =
                                        rows_b[eta] * n3 + cols_b[y] * n2 + rows_b[xi] * n;
                                    let term = second
                                        * (jsl[direct_base + cols_b[y]]
                                            - jsl[exchange_base + cols_b[z]]);
                                    if ((eta + xi + z + y) & 1) == 0 {
                                        j_b += term;
                                    } else {
                                        j_b -= term;
                                    }
                                }
                            }
                        }
                    }

                    // Reconstruct cof[D_{beta,ov}]_{eta z}=(-1)^{eta+z} det D[eta|z]
                    // from the second-minor table, then expand det D along the first row.
                    for eta in 0..LB {
                        let r = if eta == 0 { 1usize } else { 0usize };
                        let r_minor = if r < eta { r } else { r - 1 };
                        for z in 0..LB {
                            let mut value = zero;
                            for c in 0..LB {
                                if c == z {
                                    continue;
                                }
                                let c_minor = if c < z { c } else { c - 1 };
                                let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                                let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                                let row_pair = row0 * (2 * LB - row0 - 1) / 2 + (row1 - row0 - 1);
                                let col_pair = col0 * (2 * LB - col0 - 1) / 2 + (col1 - col0 - 1);
                                let term =
                                    d_b[r * LB + c] * second_b[row_pair * pairs_b + col_pair];
                                if ((r_minor + c_minor) & 1) == 0 {
                                    value += term;
                                } else {
                                    value -= term;
                                }
                            }
                            cof_b[eta * LB + z] = if ((eta + z) & 1) == 0 { value } else { -value };
                        }
                    }
                    det_b = d_b[0] * cof_b[0];
                    for z in 1..LB {
                        det_b += d_b[z] * cof_b[z];
                    }
                }

                // One-column Hamiltonian replacements:
                // sum_z det D_{beta,ov}^{z -> H_z} = sum_{eta,z} cof[D]_{eta z} H_{eta z}.
                let hcol0 = w.bb.hcol0_t_slice();
                for z in 0..LB {
                    let base = cols_b[z] * n;
                    for eta in 0..LB {
                        replacement_b += cof_b[eta * LB + z] * hcol0[base + rows_b[eta]];
                    }
                }
            }

            // Mixed-spin double replacement:
            // sum_{z,y,eta,xi} cof[D_alpha]_{eta z} II_{eta z,xi y} cof[D_beta]_{xi y}.
            let mut ii_term = zero;
            if LA > 0 && LB > 0 {
                let iisl = w.ab.iiab_slice(0, 0, 0, 0);
                let n = w.ab.n();
                let n2 = n * n;
                let n3 = n2 * n;
                if LA <= LB {
                    for z in 0..LA {
                        for eta in 0..LA {
                            let base_a = rows_a[eta] * n3 + cols_a[z] * n2;
                            let mut inner = zero;
                            for y in 0..LB {
                                for xi in 0..LB {
                                    inner += cof_b[xi * LB + y]
                                        * iisl[base_a + rows_b[xi] * n + cols_b[y]];
                                }
                            }
                            ii_term += cof_a[eta * LA + z] * inner;
                        }
                    }
                } else {
                    for y in 0..LB {
                        for xi in 0..LB {
                            let suffix_b = rows_b[xi] * n + cols_b[y];
                            let mut inner = zero;
                            for z in 0..LA {
                                for eta in 0..LA {
                                    let base_a = rows_a[eta] * n3 + cols_a[z] * n2;
                                    inner += cof_a[eta * LA + z] * iisl[base_a + suffix_b];
                                }
                            }
                            ii_term += cof_b[xi * LB + y] * inner;
                        }
                    }
                }
            }

            // Assemble the m_alpha=m_beta=0 Hamiltonian classes:
            // scalar V_0 terms times det_alpha det_beta, minus one-column replacements, plus J and II.
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
            let pref = <T as From<f64>>::from(excitation_phase)
                * w.aa.phase
                * <T as From<f64>>::from(w.aa.tilde_s_prod)
                * w.bb.phase
                * <T as From<f64>>::from(w.bb.tilde_s_prod);
            (pref * core, pref * det_ab)
        }
    )
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
    rank: (usize, usize),
    ex: (&[ExcitationCache; 4], &[ExcitationCache; 4]),
    excitation_phase: &[f64; 4],
    enuc: f64,
    out: (&mut [f64], &mut [f64]),
) {
    unsafe {
        // Select the AVX2 const-generic kernel for the shared spin ranks of this four-lane packet.
        let (la, lb) = rank;
        let (x_ex, w_ex) = ex;
        let (h, s) = out;
        match (la, lb) {
            (0, 0) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 0, 0, 1, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 1) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 0, 1, 1, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 2) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 0, 2, 1, 4, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 3) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 0, 3, 1, 9, 1, 3, 1, 9>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 4) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 0, 4, 1, 16, 1, 6, 1, 36>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (0, 5) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 0, 5, 1, 25, 1, 10, 1, 100>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (0, 6) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 0, 6, 1, 36, 1, 15, 1, 225>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (1, 0) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 1, 0, 1, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 1) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 1, 1, 1, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 2) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 1, 2, 1, 4, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 3) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 1, 3, 1, 9, 1, 3, 1, 9>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 4) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 1, 4, 1, 16, 1, 6, 1, 36>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (1, 5) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 1, 5, 1, 25, 1, 10, 1, 100>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (2, 0) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 2, 0, 4, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 1) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 2, 1, 4, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 2) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 2, 2, 4, 4, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 3) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 2, 3, 4, 9, 1, 3, 1, 9>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 4) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 2, 4, 4, 16, 1, 6, 1, 36>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (3, 0) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 3, 0, 9, 1, 3, 1, 9, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 1) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 3, 1, 9, 1, 3, 1, 9, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 2) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 3, 2, 9, 4, 3, 1, 9, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 3) => xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 3, 3, 9, 9, 3, 3, 9, 9>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (4, 0) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 4, 0, 16, 1, 6, 1, 36, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (4, 1) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 4, 1, 16, 1, 6, 1, 36, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (4, 2) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 4, 2, 16, 4, 6, 1, 36, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (5, 0) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 5, 0, 25, 1, 10, 1, 100, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (5, 1) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 5, 1, 25, 1, 10, 1, 100, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (6, 0) => {
                xw_hamiltonian_overlap_m0_prepared_f64x4_const::<T, 6, 0, 36, 1, 15, 1, 225, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
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
    rank: (usize, usize),
    ex: (&[ExcitationCache; 8], &[ExcitationCache; 8]),
    excitation_phase: &[f64; 8],
    enuc: f64,
    out: (&mut [f64], &mut [f64]),
) {
    unsafe {
        // Select the AVX-512 const-generic kernel for the shared spin ranks of this eight-lane packet.
        let (la, lb) = rank;
        let (x_ex, w_ex) = ex;
        let (h, s) = out;
        match (la, lb) {
            (0, 0) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 0, 0, 1, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 1) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 0, 1, 1, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 2) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 0, 2, 1, 4, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 3) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 0, 3, 1, 9, 1, 3, 1, 9>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (0, 4) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 0, 4, 1, 16, 1, 6, 1, 36>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (0, 5) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 0, 5, 1, 25, 1, 10, 1, 100>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (0, 6) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 0, 6, 1, 36, 1, 15, 1, 225>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (1, 0) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 1, 0, 1, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 1) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 1, 1, 1, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 2) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 1, 2, 1, 4, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 3) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 1, 3, 1, 9, 1, 3, 1, 9>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (1, 4) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 1, 4, 1, 16, 1, 6, 1, 36>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (1, 5) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 1, 5, 1, 25, 1, 10, 1, 100>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (2, 0) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 2, 0, 4, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 1) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 2, 1, 4, 1, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 2) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 2, 2, 4, 4, 1, 1, 1, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 3) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 2, 3, 4, 9, 1, 3, 1, 9>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (2, 4) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 2, 4, 4, 16, 1, 6, 1, 36>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (3, 0) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 3, 0, 9, 1, 3, 1, 9, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 1) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 3, 1, 9, 1, 3, 1, 9, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 2) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 3, 2, 9, 4, 3, 1, 9, 1>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (3, 3) => xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 3, 3, 9, 9, 3, 3, 9, 9>(
                w,
                x_ex,
                w_ex,
                excitation_phase,
                enuc,
                h,
                s,
            ),
            (4, 0) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 4, 0, 16, 1, 6, 1, 36, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (4, 1) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 4, 1, 16, 1, 6, 1, 36, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (4, 2) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 4, 2, 16, 4, 6, 1, 36, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (5, 0) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 5, 0, 25, 1, 10, 1, 100, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (5, 1) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 5, 1, 25, 1, 10, 1, 100, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            (6, 0) => {
                xw_hamiltonian_overlap_m0_prepared_f64x8_const::<T, 6, 0, 36, 1, 15, 1, 225, 1>(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    h,
                    s,
                )
            }
            _ => unreachable!(),
        }
    }
}

/// Evaluate 4 independent real fixed-rank `(L_\alpha, L_\beta)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// This is the packed `f64x4` evaluation of the same determinant, cofactor, same-spin
/// second-minor and mixed-spin cofactor contractions as `xw_hamiltonian_overlap_m0_prepared_const`.
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
unsafe fn xw_hamiltonian_overlap_m0_prepared_f64x4_const<
    T: NOCIScalar,
    const LA: usize,
    const LB: usize,
    const DA: usize,
    const DB: usize,
    const PA: usize,
    const PB: usize,
    const SA: usize,
    const SB: usize,
>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_m0_prepared_f64x4_const,
        {
            unsafe {
                let zero_v = _mm256_setzero_pd();
                let one_v = _mm256_set1_pd(1.0);

                // SIMD determinant helpers for the second-minor ranks.
                let det3 = |m: &[__m256d; 16]| -> __m256d {
                    let t0 = _mm256_fmsub_pd(m[4], m[8], _mm256_mul_pd(m[5], m[7]));
                    let mut out = _mm256_mul_pd(m[0], t0);
                    let t1 = _mm256_fmsub_pd(m[3], m[8], _mm256_mul_pd(m[5], m[6]));
                    out = _mm256_fnmadd_pd(m[1], t1, out);
                    let t2 = _mm256_fmsub_pd(m[3], m[7], _mm256_mul_pd(m[4], m[6]));
                    _mm256_fmadd_pd(m[2], t2, out)
                };
                let det4 = |m: &[__m256d; 16]| -> __m256d {
                    let mut out = zero_v;
                    for col in 0..4 {
                        let mut subm = [zero_v; 16];
                        let mut ii = 0usize;
                        for r in 1..4 {
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == col {
                                    continue;
                                }
                                subm[ii * 3 + jj] = m[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let term = _mm256_mul_pd(m[col], det3(&subm));
                        if (col & 1) == 0 {
                            out = _mm256_add_pd(out, term);
                        } else {
                            out = _mm256_sub_pd(out, term);
                        }
                    }
                    out
                };
                let det_small = |minor: &[__m256d; 16], n: usize| -> __m256d {
                    match n {
                        0 => one_v,
                        1 => minor[0],
                        2 => _mm256_fmsub_pd(minor[0], minor[3], _mm256_mul_pd(minor[1], minor[2])),
                        3 => det3(minor),
                        4 => det4(minor),
                        _ => unreachable!(),
                    }
                };
                let mut rows_a = [[0usize; 6]; 4];
                let mut cols_a = [[0usize; 6]; 4];
                let mut d_a = [zero_v; DA];
                let mut cof_a = [zero_v; DA];
                let mut second_a = [zero_v; SA];
                let mut det_a = one_v;
                let mut j_a = zero_v;
                let mut replacement_a = zero_v;

                // Lane-wise alpha D_{ov} labels: x-excitations contribute (a,i), w-excitations (j,b).
                if LA > 0 {
                    let nocc = w.aa.nocc;
                    let nvirt = w.aa.nmo - nocc;
                    for lane in 0..4 {
                        let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
                        let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
                        let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
                        for i in 0..x_rank {
                            rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc;
                            cols_a[lane][i] = usize::from(x_indices[i]);
                        }
                        for i in x_rank..LA {
                            let k = i - x_rank;
                            rows_a[lane][i] = nvirt + usize::from(w_indices[k]);
                            cols_a[lane][i] = usize::from(w_indices[4 + k]);
                        }
                    }

                    // Gather D_{alpha,ov}[eta,z] = X^{(0)} on/below the diagonal, Y^{(0)} above it.
                    let n = w.aa.n();
                    let x0_t = w.aa.x_slice(0);
                    let y0_t = w.aa.y_slice(0);
                    let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
                    let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
                    for i in 0..LA {
                        for j in 0..LA {
                            let mut values = [0.0f64; 4];
                            for lane in 0..4 {
                                let index = rows_a[lane][i] * n + cols_a[lane][j];
                                values[lane] = if i >= j {
                                    *x0.get_unchecked(index)
                                } else {
                                    *y0.get_unchecked(index)
                                };
                            }
                            d_a[i * LA + j] = _mm256_loadu_pd(values.as_ptr());
                        }
                    }
                    if LA == 1 {
                        cof_a[0] = one_v;
                        det_a = d_a[0];
                    } else {
                        // Packed same-spin C_3 term:
                        // sum phi J_{eta z,xi y} det D_{alpha,ov}[eta,xi|z,y] in each lane.
                        let pairs_a = LA * (LA - 1) / 2;
                        let jsl_t = w.aa.j_slice(0);
                        let jsl =
                            std::slice::from_raw_parts(jsl_t.as_ptr().cast::<f64>(), jsl_t.len());
                        let n2 = n * n;
                        let n3 = n2 * n;
                        for eta in 0..LA {
                            for xi in (eta + 1)..LA {
                                let row_pair = eta * (2 * LA - eta - 1) / 2 + (xi - eta - 1);
                                for z in 0..LA {
                                    for y in (z + 1)..LA {
                                        let col_pair = z * (2 * LA - z - 1) / 2 + (y - z - 1);
                                        let mut minor = [zero_v; 16];
                                        let mut ii = 0usize;
                                        for r in 0..LA {
                                            if r == eta || r == xi {
                                                continue;
                                            }
                                            let mut jj = 0usize;
                                            for c in 0..LA {
                                                if c == z || c == y {
                                                    continue;
                                                }
                                                minor[ii * (LA - 2) + jj] = d_a[r * LA + c];
                                                jj += 1;
                                            }
                                            ii += 1;
                                        }
                                        let second = det_small(&minor, LA - 2);
                                        second_a[row_pair * pairs_a + col_pair] = second;
                                        let mut direct_lane = [0.0f64; 4];
                                        let mut exchange_lane = [0.0f64; 4];
                                        for lane in 0..4 {
                                            let direct_base = rows_a[lane][eta] * n3
                                                + cols_a[lane][z] * n2
                                                + rows_a[lane][xi] * n;
                                            let exchange_base = rows_a[lane][eta] * n3
                                                + cols_a[lane][y] * n2
                                                + rows_a[lane][xi] * n;
                                            direct_lane[lane] =
                                                *jsl.get_unchecked(direct_base + cols_a[lane][y]);
                                            exchange_lane[lane] =
                                                *jsl.get_unchecked(exchange_base + cols_a[lane][z]);
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

                        // Packed cof[D_{alpha,ov}]_{eta z}=(-1)^{eta+z} det D[eta|z],
                        // reconstructed from second minors before expanding det D.
                        for eta in 0..LA {
                            let r = if eta == 0 { 1usize } else { 0usize };
                            let r_minor = if r < eta { r } else { r - 1 };
                            for z in 0..LA {
                                let mut value = zero_v;
                                for c in 0..LA {
                                    if c == z {
                                        continue;
                                    }
                                    let c_minor = if c < z { c } else { c - 1 };
                                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                                    let row_pair =
                                        row0 * (2 * LA - row0 - 1) / 2 + (row1 - row0 - 1);
                                    let col_pair =
                                        col0 * (2 * LA - col0 - 1) / 2 + (col1 - col0 - 1);
                                    let term = _mm256_mul_pd(
                                        d_a[r * LA + c],
                                        second_a[row_pair * pairs_a + col_pair],
                                    );
                                    if ((r_minor + c_minor) & 1) == 0 {
                                        value = _mm256_add_pd(value, term);
                                    } else {
                                        value = _mm256_sub_pd(value, term);
                                    }
                                }
                                cof_a[eta * LA + z] = if ((eta + z) & 1) == 0 {
                                    value
                                } else {
                                    _mm256_sub_pd(zero_v, value)
                                };
                            }
                        }
                        det_a = _mm256_mul_pd(d_a[0], cof_a[0]);
                        for z in 1..LA {
                            det_a = _mm256_fmadd_pd(d_a[z], cof_a[z], det_a);
                        }
                    }

                    // Packed one-column replacements sum cof[D_alpha]_{eta z} H^alpha_{eta z}.
                    let hcol0_t = w.aa.hcol0_t_slice();
                    let hcol0 =
                        std::slice::from_raw_parts(hcol0_t.as_ptr().cast::<f64>(), hcol0_t.len());
                    for z in 0..LA {
                        for eta in 0..LA {
                            let mut values = [0.0f64; 4];
                            for lane in 0..4 {
                                values[lane] =
                                    *hcol0.get_unchecked(cols_a[lane][z] * n + rows_a[lane][eta]);
                            }
                            replacement_a = _mm256_fmadd_pd(
                                cof_a[eta * LA + z],
                                _mm256_loadu_pd(values.as_ptr()),
                                replacement_a,
                            );
                        }
                    }
                }
                let mut rows_b = [[0usize; 6]; 4];
                let mut cols_b = [[0usize; 6]; 4];
                let mut d_b = [zero_v; DB];
                let mut cof_b = [zero_v; DB];
                let mut second_b = [zero_v; SB];
                let mut det_b = one_v;
                let mut j_b = zero_v;
                let mut replacement_b = zero_v;

                // Lane-wise beta D_{ov} labels: x-excitations contribute (a,i), w-excitations (j,b).
                if LB > 0 {
                    let nocc = w.bb.nocc;
                    let nvirt = w.bb.nmo - nocc;
                    for lane in 0..4 {
                        let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
                        let x_indices = &x_ex.get_unchecked(lane).beta.indices;
                        let w_indices = &w_ex.get_unchecked(lane).beta.indices;
                        for i in 0..x_rank {
                            rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc;
                            cols_b[lane][i] = usize::from(x_indices[i]);
                        }
                        for i in x_rank..LB {
                            let k = i - x_rank;
                            rows_b[lane][i] = nvirt + usize::from(w_indices[k]);
                            cols_b[lane][i] = usize::from(w_indices[4 + k]);
                        }
                    }

                    // Gather D_{beta,ov}[eta,z] = X^{(0)} on/below the diagonal, Y^{(0)} above it.
                    let n = w.bb.n();
                    let x0_t = w.bb.x_slice(0);
                    let y0_t = w.bb.y_slice(0);
                    let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
                    let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
                    for i in 0..LB {
                        for j in 0..LB {
                            let mut values = [0.0f64; 4];
                            for lane in 0..4 {
                                let index = rows_b[lane][i] * n + cols_b[lane][j];
                                values[lane] = if i >= j {
                                    *x0.get_unchecked(index)
                                } else {
                                    *y0.get_unchecked(index)
                                };
                            }
                            d_b[i * LB + j] = _mm256_loadu_pd(values.as_ptr());
                        }
                    }
                    if LB == 1 {
                        cof_b[0] = one_v;
                        det_b = d_b[0];
                    } else {
                        // Packed same-spin C_3 term:
                        // sum phi J_{eta z,xi y} det D_{beta,ov}[eta,xi|z,y] in each lane.
                        let pairs_b = LB * (LB - 1) / 2;
                        let jsl_t = w.bb.j_slice(0);
                        let jsl =
                            std::slice::from_raw_parts(jsl_t.as_ptr().cast::<f64>(), jsl_t.len());
                        let n2 = n * n;
                        let n3 = n2 * n;
                        for eta in 0..LB {
                            for xi in (eta + 1)..LB {
                                let row_pair = eta * (2 * LB - eta - 1) / 2 + (xi - eta - 1);
                                for z in 0..LB {
                                    for y in (z + 1)..LB {
                                        let col_pair = z * (2 * LB - z - 1) / 2 + (y - z - 1);
                                        let mut minor = [zero_v; 16];
                                        let mut ii = 0usize;
                                        for r in 0..LB {
                                            if r == eta || r == xi {
                                                continue;
                                            }
                                            let mut jj = 0usize;
                                            for c in 0..LB {
                                                if c == z || c == y {
                                                    continue;
                                                }
                                                minor[ii * (LB - 2) + jj] = d_b[r * LB + c];
                                                jj += 1;
                                            }
                                            ii += 1;
                                        }
                                        let second = det_small(&minor, LB - 2);
                                        second_b[row_pair * pairs_b + col_pair] = second;
                                        let mut direct_lane = [0.0f64; 4];
                                        let mut exchange_lane = [0.0f64; 4];
                                        for lane in 0..4 {
                                            let direct_base = rows_b[lane][eta] * n3
                                                + cols_b[lane][z] * n2
                                                + rows_b[lane][xi] * n;
                                            let exchange_base = rows_b[lane][eta] * n3
                                                + cols_b[lane][y] * n2
                                                + rows_b[lane][xi] * n;
                                            direct_lane[lane] =
                                                *jsl.get_unchecked(direct_base + cols_b[lane][y]);
                                            exchange_lane[lane] =
                                                *jsl.get_unchecked(exchange_base + cols_b[lane][z]);
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

                        // Packed cof[D_{beta,ov}]_{eta z}=(-1)^{eta+z} det D[eta|z],
                        // reconstructed from second minors before expanding det D.
                        for eta in 0..LB {
                            let r = if eta == 0 { 1usize } else { 0usize };
                            let r_minor = if r < eta { r } else { r - 1 };
                            for z in 0..LB {
                                let mut value = zero_v;
                                for c in 0..LB {
                                    if c == z {
                                        continue;
                                    }
                                    let c_minor = if c < z { c } else { c - 1 };
                                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                                    let row_pair =
                                        row0 * (2 * LB - row0 - 1) / 2 + (row1 - row0 - 1);
                                    let col_pair =
                                        col0 * (2 * LB - col0 - 1) / 2 + (col1 - col0 - 1);
                                    let term = _mm256_mul_pd(
                                        d_b[r * LB + c],
                                        second_b[row_pair * pairs_b + col_pair],
                                    );
                                    if ((r_minor + c_minor) & 1) == 0 {
                                        value = _mm256_add_pd(value, term);
                                    } else {
                                        value = _mm256_sub_pd(value, term);
                                    }
                                }
                                cof_b[eta * LB + z] = if ((eta + z) & 1) == 0 {
                                    value
                                } else {
                                    _mm256_sub_pd(zero_v, value)
                                };
                            }
                        }
                        det_b = _mm256_mul_pd(d_b[0], cof_b[0]);
                        for z in 1..LB {
                            det_b = _mm256_fmadd_pd(d_b[z], cof_b[z], det_b);
                        }
                    }

                    // Packed one-column replacements sum cof[D_beta]_{eta z} H^beta_{eta z}.
                    let hcol0_t = w.bb.hcol0_t_slice();
                    let hcol0 =
                        std::slice::from_raw_parts(hcol0_t.as_ptr().cast::<f64>(), hcol0_t.len());
                    for z in 0..LB {
                        for eta in 0..LB {
                            let mut values = [0.0f64; 4];
                            for lane in 0..4 {
                                values[lane] =
                                    *hcol0.get_unchecked(cols_b[lane][z] * n + rows_b[lane][eta]);
                            }
                            replacement_b = _mm256_fmadd_pd(
                                cof_b[eta * LB + z],
                                _mm256_loadu_pd(values.as_ptr()),
                                replacement_b,
                            );
                        }
                    }
                }

                // Packed mixed-spin double replacement:
                // sum cof[D_alpha]_{eta z} II_{eta z,xi y} cof[D_beta]_{xi y}.
                let mut ii_term = zero_v;
                if LA > 0 && LB > 0 {
                    let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
                    let iisl =
                        std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
                    let n = w.ab.n();
                    let n2 = n * n;
                    let n3 = n2 * n;
                    if LA <= LB {
                        for z in 0..LA {
                            for eta in 0..LA {
                                let mut inner = zero_v;
                                for y in 0..LB {
                                    for xi in 0..LB {
                                        let mut values = [0.0f64; 4];
                                        for lane in 0..4 {
                                            let base_a =
                                                rows_a[lane][eta] * n3 + cols_a[lane][z] * n2;
                                            values[lane] = *iisl.get_unchecked(
                                                base_a + rows_b[lane][xi] * n + cols_b[lane][y],
                                            );
                                        }
                                        inner = _mm256_fmadd_pd(
                                            cof_b[xi * LB + y],
                                            _mm256_loadu_pd(values.as_ptr()),
                                            inner,
                                        );
                                    }
                                }
                                ii_term = _mm256_fmadd_pd(cof_a[eta * LA + z], inner, ii_term);
                            }
                        }
                    } else {
                        for y in 0..LB {
                            for xi in 0..LB {
                                let mut inner = zero_v;
                                for z in 0..LA {
                                    for eta in 0..LA {
                                        let mut values = [0.0f64; 4];
                                        for lane in 0..4 {
                                            let base_a =
                                                rows_a[lane][eta] * n3 + cols_a[lane][z] * n2;
                                            values[lane] = *iisl.get_unchecked(
                                                base_a + rows_b[lane][xi] * n + cols_b[lane][y],
                                            );
                                        }
                                        inner = _mm256_fmadd_pd(
                                            cof_a[eta * LA + z],
                                            _mm256_loadu_pd(values.as_ptr()),
                                            inner,
                                        );
                                    }
                                }
                                ii_term = _mm256_fmadd_pd(cof_b[xi * LB + y], inner, ii_term);
                            }
                        }
                    }
                }

                // Packed final GNME assembly: scalar V_0 det_alpha det_beta, one-column
                // replacements, same-spin J second minors, mixed-spin II, then the common prefactor.
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
                _mm256_storeu_pd(h.as_mut_ptr(), _mm256_mul_pd(core, pref));
                _mm256_storeu_pd(s.as_mut_ptr(), _mm256_mul_pd(det_ab, pref));
            }
        }
    )
}

/// Evaluate 8 independent real fixed-rank `(L_\alpha, L_\beta)` Hamiltonian and overlap
/// matrix elements for `m_\alpha = m_\beta = 0`.
/// Each SIMD lane is one determinant pair and all lanes share the same reference pair
/// and contraction ranks.
/// This is the packed `f64x8` evaluation of the same determinant, cofactor, same-spin
/// second-minor and mixed-spin cofactor contractions as `xw_hamiltonian_overlap_m0_prepared_const`.
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
unsafe fn xw_hamiltonian_overlap_m0_prepared_f64x8_const<
    T: NOCIScalar,
    const LA: usize,
    const LB: usize,
    const DA: usize,
    const DB: usize,
    const PA: usize,
    const PB: usize,
    const SA: usize,
    const SB: usize,
>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64],
    s: &mut [f64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_m0_prepared_f64x8_const,
        {
            unsafe {
                let zero_v = _mm512_setzero_pd();
                let one_v = _mm512_set1_pd(1.0);

                // SIMD determinant helpers for the second-minor ranks.
                let det3 = |m: &[__m512d; 16]| -> __m512d {
                    let t0 = _mm512_fmsub_pd(m[4], m[8], _mm512_mul_pd(m[5], m[7]));
                    let mut out = _mm512_mul_pd(m[0], t0);
                    let t1 = _mm512_fmsub_pd(m[3], m[8], _mm512_mul_pd(m[5], m[6]));
                    out = _mm512_fnmadd_pd(m[1], t1, out);
                    let t2 = _mm512_fmsub_pd(m[3], m[7], _mm512_mul_pd(m[4], m[6]));
                    _mm512_fmadd_pd(m[2], t2, out)
                };
                let det4 = |m: &[__m512d; 16]| -> __m512d {
                    let mut out = zero_v;
                    for col in 0..4 {
                        let mut subm = [zero_v; 16];
                        let mut ii = 0usize;
                        for r in 1..4 {
                            let mut jj = 0usize;
                            for c in 0..4 {
                                if c == col {
                                    continue;
                                }
                                subm[ii * 3 + jj] = m[r * 4 + c];
                                jj += 1;
                            }
                            ii += 1;
                        }
                        let term = _mm512_mul_pd(m[col], det3(&subm));
                        if (col & 1) == 0 {
                            out = _mm512_add_pd(out, term);
                        } else {
                            out = _mm512_sub_pd(out, term);
                        }
                    }
                    out
                };
                let det_small = |minor: &[__m512d; 16], n: usize| -> __m512d {
                    match n {
                        0 => one_v,
                        1 => minor[0],
                        2 => _mm512_fmsub_pd(minor[0], minor[3], _mm512_mul_pd(minor[1], minor[2])),
                        3 => det3(minor),
                        4 => det4(minor),
                        _ => unreachable!(),
                    }
                };
                let mut rows_a = [[0usize; 6]; 8];
                let mut cols_a = [[0usize; 6]; 8];
                let mut d_a = [zero_v; DA];
                let mut cof_a = [zero_v; DA];
                let mut second_a = [zero_v; SA];
                let mut det_a = one_v;
                let mut j_a = zero_v;
                let mut replacement_a = zero_v;

                // Lane-wise alpha D_{ov} labels: x-excitations contribute (a,i), w-excitations (j,b).
                if LA > 0 {
                    let nocc = w.aa.nocc;
                    let nvirt = w.aa.nmo - nocc;
                    for lane in 0..8 {
                        let x_rank = usize::from(x_ex.get_unchecked(lane).alpha.rank);
                        let x_indices = &x_ex.get_unchecked(lane).alpha.indices;
                        let w_indices = &w_ex.get_unchecked(lane).alpha.indices;
                        for i in 0..x_rank {
                            rows_a[lane][i] = usize::from(x_indices[4 + i]) - nocc;
                            cols_a[lane][i] = usize::from(x_indices[i]);
                        }
                        for i in x_rank..LA {
                            let k = i - x_rank;
                            rows_a[lane][i] = nvirt + usize::from(w_indices[k]);
                            cols_a[lane][i] = usize::from(w_indices[4 + k]);
                        }
                    }

                    // Gather D_{alpha,ov}[eta,z] = X^{(0)} on/below the diagonal, Y^{(0)} above it.
                    let n = w.aa.n();
                    let x0_t = w.aa.x_slice(0);
                    let y0_t = w.aa.y_slice(0);
                    let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
                    let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
                    for i in 0..LA {
                        for j in 0..LA {
                            let mut values = [0.0f64; 8];
                            for lane in 0..8 {
                                let index = rows_a[lane][i] * n + cols_a[lane][j];
                                values[lane] = if i >= j {
                                    *x0.get_unchecked(index)
                                } else {
                                    *y0.get_unchecked(index)
                                };
                            }
                            d_a[i * LA + j] = _mm512_loadu_pd(values.as_ptr());
                        }
                    }
                    if LA == 1 {
                        cof_a[0] = one_v;
                        det_a = d_a[0];
                    } else {
                        // Packed same-spin C_3 term:
                        // sum phi J_{eta z,xi y} det D_{alpha,ov}[eta,xi|z,y] in each lane.
                        let pairs_a = LA * (LA - 1) / 2;
                        let jsl_t = w.aa.j_slice(0);
                        let jsl =
                            std::slice::from_raw_parts(jsl_t.as_ptr().cast::<f64>(), jsl_t.len());
                        let n2 = n * n;
                        let n3 = n2 * n;
                        for eta in 0..LA {
                            for xi in (eta + 1)..LA {
                                let row_pair = eta * (2 * LA - eta - 1) / 2 + (xi - eta - 1);
                                for z in 0..LA {
                                    for y in (z + 1)..LA {
                                        let col_pair = z * (2 * LA - z - 1) / 2 + (y - z - 1);
                                        let mut minor = [zero_v; 16];
                                        let mut ii = 0usize;
                                        for r in 0..LA {
                                            if r == eta || r == xi {
                                                continue;
                                            }
                                            let mut jj = 0usize;
                                            for c in 0..LA {
                                                if c == z || c == y {
                                                    continue;
                                                }
                                                minor[ii * (LA - 2) + jj] = d_a[r * LA + c];
                                                jj += 1;
                                            }
                                            ii += 1;
                                        }
                                        let second = det_small(&minor, LA - 2);
                                        second_a[row_pair * pairs_a + col_pair] = second;
                                        let mut direct_lane = [0.0f64; 8];
                                        let mut exchange_lane = [0.0f64; 8];
                                        for lane in 0..8 {
                                            let direct_base = rows_a[lane][eta] * n3
                                                + cols_a[lane][z] * n2
                                                + rows_a[lane][xi] * n;
                                            let exchange_base = rows_a[lane][eta] * n3
                                                + cols_a[lane][y] * n2
                                                + rows_a[lane][xi] * n;
                                            direct_lane[lane] =
                                                *jsl.get_unchecked(direct_base + cols_a[lane][y]);
                                            exchange_lane[lane] =
                                                *jsl.get_unchecked(exchange_base + cols_a[lane][z]);
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

                        // Packed cof[D_{alpha,ov}]_{eta z}=(-1)^{eta+z} det D[eta|z],
                        // reconstructed from second minors before expanding det D.
                        for eta in 0..LA {
                            let r = if eta == 0 { 1usize } else { 0usize };
                            let r_minor = if r < eta { r } else { r - 1 };
                            for z in 0..LA {
                                let mut value = zero_v;
                                for c in 0..LA {
                                    if c == z {
                                        continue;
                                    }
                                    let c_minor = if c < z { c } else { c - 1 };
                                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                                    let row_pair =
                                        row0 * (2 * LA - row0 - 1) / 2 + (row1 - row0 - 1);
                                    let col_pair =
                                        col0 * (2 * LA - col0 - 1) / 2 + (col1 - col0 - 1);
                                    let term = _mm512_mul_pd(
                                        d_a[r * LA + c],
                                        second_a[row_pair * pairs_a + col_pair],
                                    );
                                    if ((r_minor + c_minor) & 1) == 0 {
                                        value = _mm512_add_pd(value, term);
                                    } else {
                                        value = _mm512_sub_pd(value, term);
                                    }
                                }
                                cof_a[eta * LA + z] = if ((eta + z) & 1) == 0 {
                                    value
                                } else {
                                    _mm512_sub_pd(zero_v, value)
                                };
                            }
                        }
                        det_a = _mm512_mul_pd(d_a[0], cof_a[0]);
                        for z in 1..LA {
                            det_a = _mm512_fmadd_pd(d_a[z], cof_a[z], det_a);
                        }
                    }

                    // Packed one-column replacements sum cof[D_alpha]_{eta z} H^alpha_{eta z}.
                    let hcol0_t = w.aa.hcol0_t_slice();
                    let hcol0 =
                        std::slice::from_raw_parts(hcol0_t.as_ptr().cast::<f64>(), hcol0_t.len());
                    for z in 0..LA {
                        for eta in 0..LA {
                            let mut values = [0.0f64; 8];
                            for lane in 0..8 {
                                values[lane] =
                                    *hcol0.get_unchecked(cols_a[lane][z] * n + rows_a[lane][eta]);
                            }
                            replacement_a = _mm512_fmadd_pd(
                                cof_a[eta * LA + z],
                                _mm512_loadu_pd(values.as_ptr()),
                                replacement_a,
                            );
                        }
                    }
                }
                let mut rows_b = [[0usize; 6]; 8];
                let mut cols_b = [[0usize; 6]; 8];
                let mut d_b = [zero_v; DB];
                let mut cof_b = [zero_v; DB];
                let mut second_b = [zero_v; SB];
                let mut det_b = one_v;
                let mut j_b = zero_v;
                let mut replacement_b = zero_v;

                // Lane-wise beta D_{ov} labels: x-excitations contribute (a,i), w-excitations (j,b).
                if LB > 0 {
                    let nocc = w.bb.nocc;
                    let nvirt = w.bb.nmo - nocc;
                    for lane in 0..8 {
                        let x_rank = usize::from(x_ex.get_unchecked(lane).beta.rank);
                        let x_indices = &x_ex.get_unchecked(lane).beta.indices;
                        let w_indices = &w_ex.get_unchecked(lane).beta.indices;
                        for i in 0..x_rank {
                            rows_b[lane][i] = usize::from(x_indices[4 + i]) - nocc;
                            cols_b[lane][i] = usize::from(x_indices[i]);
                        }
                        for i in x_rank..LB {
                            let k = i - x_rank;
                            rows_b[lane][i] = nvirt + usize::from(w_indices[k]);
                            cols_b[lane][i] = usize::from(w_indices[4 + k]);
                        }
                    }

                    // Gather D_{beta,ov}[eta,z] = X^{(0)} on/below the diagonal, Y^{(0)} above it.
                    let n = w.bb.n();
                    let x0_t = w.bb.x_slice(0);
                    let y0_t = w.bb.y_slice(0);
                    let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
                    let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
                    for i in 0..LB {
                        for j in 0..LB {
                            let mut values = [0.0f64; 8];
                            for lane in 0..8 {
                                let index = rows_b[lane][i] * n + cols_b[lane][j];
                                values[lane] = if i >= j {
                                    *x0.get_unchecked(index)
                                } else {
                                    *y0.get_unchecked(index)
                                };
                            }
                            d_b[i * LB + j] = _mm512_loadu_pd(values.as_ptr());
                        }
                    }
                    if LB == 1 {
                        cof_b[0] = one_v;
                        det_b = d_b[0];
                    } else {
                        // Packed same-spin C_3 term:
                        // sum phi J_{eta z,xi y} det D_{beta,ov}[eta,xi|z,y] in each lane.
                        let pairs_b = LB * (LB - 1) / 2;
                        let jsl_t = w.bb.j_slice(0);
                        let jsl =
                            std::slice::from_raw_parts(jsl_t.as_ptr().cast::<f64>(), jsl_t.len());
                        let n2 = n * n;
                        let n3 = n2 * n;
                        for eta in 0..LB {
                            for xi in (eta + 1)..LB {
                                let row_pair = eta * (2 * LB - eta - 1) / 2 + (xi - eta - 1);
                                for z in 0..LB {
                                    for y in (z + 1)..LB {
                                        let col_pair = z * (2 * LB - z - 1) / 2 + (y - z - 1);
                                        let mut minor = [zero_v; 16];
                                        let mut ii = 0usize;
                                        for r in 0..LB {
                                            if r == eta || r == xi {
                                                continue;
                                            }
                                            let mut jj = 0usize;
                                            for c in 0..LB {
                                                if c == z || c == y {
                                                    continue;
                                                }
                                                minor[ii * (LB - 2) + jj] = d_b[r * LB + c];
                                                jj += 1;
                                            }
                                            ii += 1;
                                        }
                                        let second = det_small(&minor, LB - 2);
                                        second_b[row_pair * pairs_b + col_pair] = second;
                                        let mut direct_lane = [0.0f64; 8];
                                        let mut exchange_lane = [0.0f64; 8];
                                        for lane in 0..8 {
                                            let direct_base = rows_b[lane][eta] * n3
                                                + cols_b[lane][z] * n2
                                                + rows_b[lane][xi] * n;
                                            let exchange_base = rows_b[lane][eta] * n3
                                                + cols_b[lane][y] * n2
                                                + rows_b[lane][xi] * n;
                                            direct_lane[lane] =
                                                *jsl.get_unchecked(direct_base + cols_b[lane][y]);
                                            exchange_lane[lane] =
                                                *jsl.get_unchecked(exchange_base + cols_b[lane][z]);
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

                        // Packed cof[D_{beta,ov}]_{eta z}=(-1)^{eta+z} det D[eta|z],
                        // reconstructed from second minors before expanding det D.
                        for eta in 0..LB {
                            let r = if eta == 0 { 1usize } else { 0usize };
                            let r_minor = if r < eta { r } else { r - 1 };
                            for z in 0..LB {
                                let mut value = zero_v;
                                for c in 0..LB {
                                    if c == z {
                                        continue;
                                    }
                                    let c_minor = if c < z { c } else { c - 1 };
                                    let (row0, row1) = if eta < r { (eta, r) } else { (r, eta) };
                                    let (col0, col1) = if z < c { (z, c) } else { (c, z) };
                                    let row_pair =
                                        row0 * (2 * LB - row0 - 1) / 2 + (row1 - row0 - 1);
                                    let col_pair =
                                        col0 * (2 * LB - col0 - 1) / 2 + (col1 - col0 - 1);
                                    let term = _mm512_mul_pd(
                                        d_b[r * LB + c],
                                        second_b[row_pair * pairs_b + col_pair],
                                    );
                                    if ((r_minor + c_minor) & 1) == 0 {
                                        value = _mm512_add_pd(value, term);
                                    } else {
                                        value = _mm512_sub_pd(value, term);
                                    }
                                }
                                cof_b[eta * LB + z] = if ((eta + z) & 1) == 0 {
                                    value
                                } else {
                                    _mm512_sub_pd(zero_v, value)
                                };
                            }
                        }
                        det_b = _mm512_mul_pd(d_b[0], cof_b[0]);
                        for z in 1..LB {
                            det_b = _mm512_fmadd_pd(d_b[z], cof_b[z], det_b);
                        }
                    }

                    // Packed one-column replacements sum cof[D_beta]_{eta z} H^beta_{eta z}.
                    let hcol0_t = w.bb.hcol0_t_slice();
                    let hcol0 =
                        std::slice::from_raw_parts(hcol0_t.as_ptr().cast::<f64>(), hcol0_t.len());
                    for z in 0..LB {
                        for eta in 0..LB {
                            let mut values = [0.0f64; 8];
                            for lane in 0..8 {
                                values[lane] =
                                    *hcol0.get_unchecked(cols_b[lane][z] * n + rows_b[lane][eta]);
                            }
                            replacement_b = _mm512_fmadd_pd(
                                cof_b[eta * LB + z],
                                _mm512_loadu_pd(values.as_ptr()),
                                replacement_b,
                            );
                        }
                    }
                }

                // Packed mixed-spin double replacement:
                // sum cof[D_alpha]_{eta z} II_{eta z,xi y} cof[D_beta]_{xi y}.
                let mut ii_term = zero_v;
                if LA > 0 && LB > 0 {
                    let iisl_t = w.ab.iiab_slice(0, 0, 0, 0);
                    let iisl =
                        std::slice::from_raw_parts(iisl_t.as_ptr().cast::<f64>(), iisl_t.len());
                    let n = w.ab.n();
                    let n2 = n * n;
                    let n3 = n2 * n;
                    if LA <= LB {
                        for z in 0..LA {
                            for eta in 0..LA {
                                let mut inner = zero_v;
                                for y in 0..LB {
                                    for xi in 0..LB {
                                        let mut values = [0.0f64; 8];
                                        for lane in 0..8 {
                                            let base_a =
                                                rows_a[lane][eta] * n3 + cols_a[lane][z] * n2;
                                            values[lane] = *iisl.get_unchecked(
                                                base_a + rows_b[lane][xi] * n + cols_b[lane][y],
                                            );
                                        }
                                        inner = _mm512_fmadd_pd(
                                            cof_b[xi * LB + y],
                                            _mm512_loadu_pd(values.as_ptr()),
                                            inner,
                                        );
                                    }
                                }
                                ii_term = _mm512_fmadd_pd(cof_a[eta * LA + z], inner, ii_term);
                            }
                        }
                    } else {
                        for y in 0..LB {
                            for xi in 0..LB {
                                let mut inner = zero_v;
                                for z in 0..LA {
                                    for eta in 0..LA {
                                        let mut values = [0.0f64; 8];
                                        for lane in 0..8 {
                                            let base_a =
                                                rows_a[lane][eta] * n3 + cols_a[lane][z] * n2;
                                            values[lane] = *iisl.get_unchecked(
                                                base_a + rows_b[lane][xi] * n + cols_b[lane][y],
                                            );
                                        }
                                        inner = _mm512_fmadd_pd(
                                            cof_a[eta * LA + z],
                                            _mm512_loadu_pd(values.as_ptr()),
                                            inner,
                                        );
                                    }
                                }
                                ii_term = _mm512_fmadd_pd(cof_b[xi * LB + y], inner, ii_term);
                            }
                        }
                    }
                }

                // Packed final GNME assembly: scalar V_0 det_alpha det_beta, one-column
                // replacements, same-spin J second minors, mixed-spin II, then the common prefactor.
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
                _mm512_storeu_pd(h.as_mut_ptr(), _mm512_mul_pd(core, pref));
                _mm512_storeu_pd(s.as_mut_ptr(), _mm512_mul_pd(det_ab, pref));
            }
        }
    )
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
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_m0_gen_prepared,
        {
            // Generic m = 0 path: build spin-sector D_ov determinants and cofactors once,
            // then reuse them for scalar, one-column, same-spin J and mixed-spin II terms.
            let la =
                x_ex.alpha.holes.count_ones() as usize + w_ex.alpha.holes.count_ones() as usize;
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
                    let cof = scratch.aa.adjt_det.as_slice();
                    let n = w.aa.n();
                    if have_b {
                        let hcol0 = w.aa.hcol0_t_slice();
                        for z in 0..la {
                            let base = scratch.aa.cols[z] * n;
                            for eta in 0..la {
                                same_a -= cof[eta * la + z] * hcol0[base + scratch.aa.rows[eta]];
                            }
                        }
                    } else {
                        let fh = w.aa.fh_t_slice(0, 0);
                        let vv = w.aa.v_t_slice(0, 0, 0);
                        for z in 0..la {
                            let base = scratch.aa.cols[z] * n;
                            for eta in 0..la {
                                same_a -= cof[eta * la + z]
                                    * (fh[base + scratch.aa.rows[eta]]
                                        + vv[base + scratch.aa.rows[eta]]);
                            }
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
                                        let mut minor_col = 0usize;
                                        for c in 0..la {
                                            if c == z || c == y {
                                                continue;
                                            }
                                            minor[ii * (la - 2) + minor_col] = d[r * la + c];
                                            minor_col += 1;
                                        }
                                        ii += 1;
                                    }
                                    let second = det(&minor, la - 2).unwrap_or(zero);
                                    let n2 = n * n;
                                    let n3 = n2 * n;
                                    let row_eta_n3 = rows[eta] * n3;
                                    let row_xi_n = rows[xi] * n;
                                    let direct_base = row_eta_n3 + cols[z] * n2 + row_xi_n;
                                    let exchange_base = row_eta_n3 + cols[y] * n2 + row_xi_n;
                                    let direct = jsl[direct_base + cols[y]];
                                    let exchange = jsl[exchange_base + cols[z]];
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
                    let cof = scratch.bb.adjt_det.as_slice();
                    let n = w.bb.n();
                    if have_a {
                        let hcol0 = w.bb.hcol0_t_slice();
                        for z in 0..lb {
                            let base = scratch.bb.cols[z] * n;
                            for eta in 0..lb {
                                same_b -= cof[eta * lb + z] * hcol0[base + scratch.bb.rows[eta]];
                            }
                        }
                    } else {
                        let fh = w.bb.fh_t_slice(0, 0);
                        let vv = w.bb.v_t_slice(0, 0, 0);
                        for z in 0..lb {
                            let base = scratch.bb.cols[z] * n;
                            for eta in 0..lb {
                                same_b -= cof[eta * lb + z]
                                    * (fh[base + scratch.bb.rows[eta]]
                                        + vv[base + scratch.bb.rows[eta]]);
                            }
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
                                        let mut minor_col = 0usize;
                                        for c in 0..lb {
                                            if c == z || c == y {
                                                continue;
                                            }
                                            minor[ii * (lb - 2) + minor_col] = d[r * lb + c];
                                            minor_col += 1;
                                        }
                                        ii += 1;
                                    }
                                    let second = det(&minor, lb - 2).unwrap_or(zero);
                                    let n2 = n * n;
                                    let n3 = n2 * n;
                                    let row_eta_n3 = rows[eta] * n3;
                                    let row_xi_n = rows[xi] * n;
                                    let direct_base = row_eta_n3 + cols[z] * n2 + row_xi_n;
                                    let exchange_base = row_eta_n3 + cols[y] * n2 + row_xi_n;
                                    let direct = jsl[direct_base + cols[y]];
                                    let exchange = jsl[exchange_base + cols[z]];
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
                if la > 0 && lb > 0 {
                    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
                    let cofa = scratch.aa.adjt_det.as_slice();
                    let cofb = scratch.bb.adjt_det.as_slice();
                    for z in 0..la {
                        for eta in 0..la {
                            let n2 = n * n;
                            let n3 = n2 * n;
                            let base_a = scratch.aa.rows[eta] * n3 + scratch.aa.cols[z] * n2;
                            let mut inner = zero;
                            for y in 0..lb {
                                for xi in 0..lb {
                                    inner += cofb[xi * lb + y]
                                        * iisl
                                            [base_a + scratch.bb.rows[xi] * n + scratch.bb.cols[y]];
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
    )
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
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_gen_prepared,
        {
            let la =
                x_ex.alpha.holes.count_ones() as usize + w_ex.alpha.holes.count_ones() as usize;
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
                        let mut i = 0usize;
                        while i < la {
                            let mut j = 0usize;
                            while j < la {
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
                                j += 1;
                            }
                            i += 1;
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

                        let mut i = 0usize;
                        while i < lb {
                            let mut j = 0usize;
                            while j < lb {
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
                                j += 1;
                            }
                            i += 1;
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

                        let det_repl =
                            column_replacement_det(la, scratch.adjt_deta.as_slice(), k, |r| {
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

                        let det_repl =
                            column_replacement_det(lb, scratch.adjt_detb.as_slice(), k, |r| {
                                vsl[base + rows_b[r]]
                            });

                        contrib -= det_repl * det_deta;
                    }

                    // Contract the alpha cofactor with a beta-column replacement of `\mathcal{II}`.
                    // This reproduces the existing assignment ordering while avoiding a separate tensor
                    // pass.
                    let mut i = 0usize;
                    while i < la {
                        let ra = rows_a[i];

                        let mut j = 0usize;
                        while j < la {
                            let ca = cols_a[j];
                            let cofa = scratch.adjt_deta.as_slice()[i * la + j];
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
                            j += 1;
                        }
                        i += 1;
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
    )
}
