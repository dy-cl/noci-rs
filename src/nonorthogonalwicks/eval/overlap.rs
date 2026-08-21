// nonorthogonalwicks/eval/overlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_mul_pd,
    _mm256_set1_pd, _mm256_storeu_pd, _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_fnmadd_pd,
    _mm512_loadu_pd, _mm512_mul_pd, _mm512_set1_pd, _mm512_storeu_pd,
};

// Crate-root imports.
#[cfg(target_arch = "x86_64")]
use crate::ExcitationSpinCache;
use crate::maths::{det, det_lu_l5, det_lu_l6};
use crate::noci::NOCIScalar;
use crate::time_call;
use crate::{DetState, ExcitationSpin, ReducedOneSpinDetState};

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::mix_dets_same;
use super::prepare::{construct_determinant_indices, prepare_same};

/// Evaluate the same-spin overlap between excited determinants generated from the reference pair
/// `\langle{}^x\Psi| and |{}^w\Psi\rangle:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L=m}}`
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
        // The contraction determinant has dimension L = L_x + L_w.
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

/// Evaluate a same-spin overlap for factor-table construction, using the direct overlap-only path
/// `when m = 0 and L \leq 6:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L=m}}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L).`
/// The direct path avoids preparing reusable Hamiltonian scratch data. Other cases use `prepare_same`
/// followed by the general overlap evaluator.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage used by the prepared evaluation path.
/// # Returns
/// - `f64`: Same-spin overlap excluding excitation phases applied outside the Wick evaluation.
#[inline(always)]
pub(crate) fn xw_overlap_same_f64(
    w: &SameSpinView<'_, f64>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<f64>,
) -> f64 {
    // Determine the contraction-determinant dimension L = L_x + L_w.
    let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;

    // No distribution satisfying \sum_i m_i = m exists when m > L.
    if w.m > l {
        return 0.0;
    }

    // For m = 0 and L \leq 6, construct and evaluate \mathbf D_{\mathrm{ov}}(0,\ldots,0)
    // directly without populating the reusable scratch representation.
    if w.m == 0 && l <= 6 {
        return xw_overlap_m0_direct_f64(w, l_ex, g_ex);
    }

    // Prepare the all-m_i = 0 and, where required, all-m_i = 1 contraction determinants
    // before applying the standard overlap evaluation.
    prepare_same(w, l_ex, g_ex, scratch);
    xw_overlap(w, l_ex, g_ex, scratch)
}

/// Inputs and outputs for one row of real same-spin overlap factors.
pub(crate) struct SameSpinOverlapBatch<'a> {
    /// Determinant basis used only by generic fallback evaluation.
    pub(crate) basis: &'a [DetState<f64>],
    /// Reduced target spin representative shared by the row.
    pub(crate) target: ReducedOneSpinDetState,
    /// Reduced source spin representatives in output-column order.
    pub(crate) sources: &'a [ReducedOneSpinDetState],
    /// Whether the target belongs to the left reference in `w`.
    pub(crate) target_left: bool,
    /// Whether to evaluate alpha-spin rather than beta-spin overlap factors.
    pub(crate) alpha: bool,
    /// Output same-spin overlap factors in source-representative order.
    pub(crate) out: &'a mut [f64],
}

/// Evaluate one row of real same-spin overlaps for one ordered reference pair.
/// The target representative is paired with every source representative. Requests with `m = 0`,
/// `L = 1,\ldots,6`, and individual excitation ranks at most four are grouped by fixed contraction
/// rank and evaluated with the widest available SIMD kernel. Incomplete groups use the scalar
/// overlap-only path. Other requests use the generic overlap-only evaluator.
/// Excitation phases are applied here so each output is the complete alpha- or beta-spin factor.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `batch`: One row of same-spin overlap-factor work.
/// - `scratch`: Reusable Wick workspace for scalar fallback evaluation.
/// # Returns
/// - `()`: Writes one complete same-spin overlap-factor row into `batch.out`.
pub(crate) fn xw_overlap_same_f64_batched(
    w: &SameSpinView<'_, f64>,
    batch: SameSpinOverlapBatch<'_>,
    scratch: &mut WickScratch<f64>,
) {
    let SameSpinOverlapBatch {
        basis,
        target,
        sources,
        target_left,
        alpha,
        out,
    } = batch;
    let target_cache = target.excitation_cache;
    let target_phase = target.phase;

    #[cfg(target_arch = "x86_64")]
    if w.m == 0 {
        if std::arch::is_x86_feature_detected!("avx512f") {
            let mut bins = [[ExcitationSpinCache::default(); 8]; 7];
            let mut phases = [[1.0f64; 8]; 7];
            let mut outputs = [[0usize; 8]; 7];
            let mut counts = [0usize; 7];

            for (col, source) in sources.iter().enumerate() {
                let source_cache = source.excitation_cache;
                let l = usize::from(target_cache.rank) + usize::from(source_cache.rank);

                if target_cache.rank <= 4 && source_cache.rank <= 4 && (1..=6).contains(&l) {
                    let count = counts[l];
                    bins[l][count] = source_cache;
                    phases[l][count] = source.phase;
                    outputs[l][count] = col;
                    counts[l] += 1;

                    if counts[l] == 8 {
                        let target_batch = [target_cache; 8];
                        let source_batch = bins[l];
                        let (x_ex, w_ex) = if target_left {
                            (&target_batch, &source_batch)
                        } else {
                            (&source_batch, &target_batch)
                        };
                        let mut overlap = [0.0f64; 8];

                        unsafe {
                            xw_overlap_m0_prepared_f64x8(w, l, x_ex, w_ex, &mut overlap);
                        }

                        for lane in 0..8 {
                            out[outputs[l][lane]] = target_phase * phases[l][lane] * overlap[lane];
                        }
                        counts[l] = 0;
                    }
                } else {
                    let target_state = &basis[target.det];
                    let source_state = &basis[source.det];
                    let (target_ex, source_ex) = if alpha {
                        (
                            &target_state.excitation.alpha,
                            &source_state.excitation.alpha,
                        )
                    } else {
                        (&target_state.excitation.beta, &source_state.excitation.beta)
                    };
                    let (x_ex, w_ex) = if target_left {
                        (target_ex, source_ex)
                    } else {
                        (source_ex, target_ex)
                    };
                    out[col] =
                        target_phase * source.phase * xw_overlap_same_f64(w, x_ex, w_ex, scratch);
                }
            }

            for l in 1..=6 {
                for &col in outputs[l][..counts[l]].iter() {
                    let source = sources[col];
                    let target_state = &basis[target.det];
                    let source_state = &basis[source.det];
                    let (target_ex, source_ex) = if alpha {
                        (
                            &target_state.excitation.alpha,
                            &source_state.excitation.alpha,
                        )
                    } else {
                        (&target_state.excitation.beta, &source_state.excitation.beta)
                    };
                    let (x_ex, w_ex) = if target_left {
                        (target_ex, source_ex)
                    } else {
                        (source_ex, target_ex)
                    };
                    out[col] =
                        target_phase * source.phase * xw_overlap_same_f64(w, x_ex, w_ex, scratch);
                }
            }
            return;
        }

        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            let mut bins = [[ExcitationSpinCache::default(); 4]; 7];
            let mut phases = [[1.0f64; 4]; 7];
            let mut outputs = [[0usize; 4]; 7];
            let mut counts = [0usize; 7];

            for (col, source) in sources.iter().enumerate() {
                let source_cache = source.excitation_cache;
                let l = usize::from(target_cache.rank) + usize::from(source_cache.rank);

                if target_cache.rank <= 4 && source_cache.rank <= 4 && (1..=6).contains(&l) {
                    let count = counts[l];
                    bins[l][count] = source_cache;
                    phases[l][count] = source.phase;
                    outputs[l][count] = col;
                    counts[l] += 1;

                    if counts[l] == 4 {
                        let target_batch = [target_cache; 4];
                        let source_batch = bins[l];
                        let (x_ex, w_ex) = if target_left {
                            (&target_batch, &source_batch)
                        } else {
                            (&source_batch, &target_batch)
                        };
                        let mut overlap = [0.0f64; 4];

                        unsafe {
                            xw_overlap_m0_prepared_f64x4(w, l, x_ex, w_ex, &mut overlap);
                        }

                        for lane in 0..4 {
                            out[outputs[l][lane]] = target_phase * phases[l][lane] * overlap[lane];
                        }
                        counts[l] = 0;
                    }
                } else {
                    let target_state = &basis[target.det];
                    let source_state = &basis[source.det];
                    let (target_ex, source_ex) = if alpha {
                        (
                            &target_state.excitation.alpha,
                            &source_state.excitation.alpha,
                        )
                    } else {
                        (&target_state.excitation.beta, &source_state.excitation.beta)
                    };
                    let (x_ex, w_ex) = if target_left {
                        (target_ex, source_ex)
                    } else {
                        (source_ex, target_ex)
                    };
                    out[col] =
                        target_phase * source.phase * xw_overlap_same_f64(w, x_ex, w_ex, scratch);
                }
            }

            for l in 1..=6 {
                for &col in outputs[l][..counts[l]].iter() {
                    let source = sources[col];
                    let target_state = &basis[target.det];
                    let source_state = &basis[source.det];
                    let (target_ex, source_ex) = if alpha {
                        (
                            &target_state.excitation.alpha,
                            &source_state.excitation.alpha,
                        )
                    } else {
                        (&target_state.excitation.beta, &source_state.excitation.beta)
                    };
                    let (x_ex, w_ex) = if target_left {
                        (target_ex, source_ex)
                    } else {
                        (source_ex, target_ex)
                    };
                    out[col] =
                        target_phase * source.phase * xw_overlap_same_f64(w, x_ex, w_ex, scratch);
                }
            }
            return;
        }
    }

    let target_state = &basis[target.det];
    let target_ex = if alpha {
        &target_state.excitation.alpha
    } else {
        &target_state.excitation.beta
    };

    for (col, source) in sources.iter().enumerate() {
        let source_state = &basis[source.det];
        let source_ex = if alpha {
            &source_state.excitation.alpha
        } else {
            &source_state.excitation.beta
        };
        let (x_ex, w_ex) = if target_left {
            (target_ex, source_ex)
        } else {
            (source_ex, target_ex)
        };
        out[col] = target_phase * source.phase * xw_overlap_same_f64(w, x_ex, w_ex, scratch);
    }
}

/// Dispatch 4 real `m = 0` overlaps to the fixed-rank AVX2/FMA kernel.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `x_ex`: 4 x-reference excitation caches.
/// - `w_ex`: 4 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 4 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_prepared_f64x4(
    w: &SameSpinView<'_, f64>,
    l: usize,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    unsafe {
        match l {
            1 => xw_overlap_m0_l1_prepared_f64x4(w, x_ex, w_ex, overlap),
            2 => xw_overlap_m0_l2_prepared_f64x4(w, x_ex, w_ex, overlap),
            3 => xw_overlap_m0_l3_prepared_f64x4(w, x_ex, w_ex, overlap),
            4 => xw_overlap_m0_l4_prepared_f64x4(w, x_ex, w_ex, overlap),
            5 => xw_overlap_m0_l5_prepared_f64x4(w, x_ex, w_ex, overlap),
            6 => xw_overlap_m0_l6_prepared_f64x4(w, x_ex, w_ex, overlap),
            _ => unreachable!(),
        }
    }
}

/// Dispatch 8 real `m = 0` overlaps to the fixed-rank AVX-512 kernel.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `x_ex`: 8 x-reference excitation caches.
/// - `w_ex`: 8 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 8 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_prepared_f64x8(
    w: &SameSpinView<'_, f64>,
    l: usize,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    unsafe {
        match l {
            1 => xw_overlap_m0_l1_prepared_f64x8(w, x_ex, w_ex, overlap),
            2 => xw_overlap_m0_l2_prepared_f64x8(w, x_ex, w_ex, overlap),
            3 => xw_overlap_m0_l3_prepared_f64x8(w, x_ex, w_ex, overlap),
            4 => xw_overlap_m0_l4_prepared_f64x8(w, x_ex, w_ex, overlap),
            5 => xw_overlap_m0_l5_prepared_f64x8(w, x_ex, w_ex, overlap),
            6 => xw_overlap_m0_l6_prepared_f64x8(w, x_ex, w_ex, overlap),
            _ => unreachable!(),
        }
    }
}

/// Prepare and evaluate 4 independent real fixed-rank `L = 1` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX2/FMA` arithmetic evaluates 4
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 1` contraction entry is loaded per lane, and the explicit minor/cofactor
/// formulation uses zero determinant multiplications without recomputing a determinant product.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 4 x-reference excitation caches.
/// - `w_ex`: 4 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 4 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_l1_prepared_f64x4(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 4]; 1];
        let mut lane = 0;

        while lane < 4 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 1];
            let mut cols = [0usize; 1];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..1 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly one contraction entry, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            lane += 1;
        }

        let det_v = _mm256_loadu_pd(d[0].as_ptr());
        let overlap_v = _mm256_mul_pd(det_v, _mm256_set1_pd(pref));
        _mm256_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 1` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates 8
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 1` contraction entry is loaded per lane, and the explicit minor/cofactor
/// formulation uses zero determinant multiplications without recomputing a determinant product.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 8 x-reference excitation caches.
/// - `w_ex`: 8 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 8 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_l1_prepared_f64x8(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 8]; 1];
        let mut lane = 0;

        while lane < 8 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 1];
            let mut cols = [0usize; 1];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..1 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly one contraction entry, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            lane += 1;
        }

        let det_v = _mm512_loadu_pd(d[0].as_ptr());
        let overlap_v = _mm512_mul_pd(det_v, _mm512_set1_pd(pref));
        _mm512_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 4 independent real fixed-rank `L = 2` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX2/FMA` arithmetic evaluates 4
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 4` contraction entries are loaded per lane, and the explicit minor/cofactor
/// formulation uses two determinant multiplications without recomputing a determinant product.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 4 x-reference excitation caches.
/// - `w_ex`: 4 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 4 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_l2_prepared_f64x4(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 4]; 4];
        let mut lane = 0;

        while lane < 4 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 2];
            let mut cols = [0usize; 2];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..2 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly `4` contraction entries, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];

            d[2][lane] = x0[rows[1] * n + cols[0]];
            d[3][lane] = x0[rows[1] * n + cols[1]];
            lane += 1;
        }

        // The two products in `D_{00} D_{11} - D_{01} D_{10}` are both required.
        let det_v = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[0].as_ptr()),
            _mm256_loadu_pd(d[3].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[1].as_ptr()),
                _mm256_loadu_pd(d[2].as_ptr()),
            ),
        );
        let overlap_v = _mm256_mul_pd(det_v, _mm256_set1_pd(pref));
        _mm256_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 2` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates 8
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 4` contraction entries are loaded per lane, and the explicit minor/cofactor
/// formulation uses two determinant multiplications without recomputing a determinant product.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 8 x-reference excitation caches.
/// - `w_ex`: 8 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 8 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_l2_prepared_f64x8(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 8]; 4];
        let mut lane = 0;

        while lane < 8 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 2];
            let mut cols = [0usize; 2];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..2 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly `4` contraction entries, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];

            d[2][lane] = x0[rows[1] * n + cols[0]];
            d[3][lane] = x0[rows[1] * n + cols[1]];
            lane += 1;
        }

        // The two products in `D_{00} D_{11} - D_{01} D_{10}` are both required.
        let det_v = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[0].as_ptr()),
            _mm512_loadu_pd(d[3].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[1].as_ptr()),
                _mm512_loadu_pd(d[2].as_ptr()),
            ),
        );
        let overlap_v = _mm512_mul_pd(det_v, _mm512_set1_pd(pref));
        _mm512_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 4 independent real fixed-rank `L = 3` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX2/FMA` arithmetic evaluates 4
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 9` contraction entries are loaded per lane, and the explicit minor/cofactor
/// formulation uses nine determinant multiplications without recomputing a determinant product.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 4 x-reference excitation caches.
/// - `w_ex`: 4 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 4 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_l3_prepared_f64x4(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 4]; 9];
        let mut lane = 0;

        while lane < 4 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 3];
            let mut cols = [0usize; 3];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..3 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly `9` contraction entries, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            d[2][lane] = y0[rows[0] * n + cols[2]];

            d[3][lane] = x0[rows[1] * n + cols[0]];
            d[4][lane] = x0[rows[1] * n + cols[1]];
            d[5][lane] = y0[rows[1] * n + cols[2]];

            d[6][lane] = x0[rows[2] * n + cols[0]];
            d[7][lane] = x0[rows[2] * n + cols[1]];
            d[8][lane] = x0[rows[2] * n + cols[2]];
            lane += 1;
        }

        // The three distinct `2 \\times 2` minors require six products and the first-row
        // expansion requires three. No determinant product is recomputed.
        let m0 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[4].as_ptr()),
            _mm256_loadu_pd(d[8].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[7].as_ptr()),
            ),
        );
        let m1 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[3].as_ptr()),
            _mm256_loadu_pd(d[8].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[6].as_ptr()),
            ),
        );
        let m2 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[3].as_ptr()),
            _mm256_loadu_pd(d[7].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[6].as_ptr()),
            ),
        );
        let det_v = _mm256_mul_pd(_mm256_loadu_pd(d[0].as_ptr()), m0);
        let det_v = _mm256_fnmadd_pd(_mm256_loadu_pd(d[1].as_ptr()), m1, det_v);
        let det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[2].as_ptr()), m2, det_v);
        let overlap_v = _mm256_mul_pd(det_v, _mm256_set1_pd(pref));
        _mm256_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 3` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates 8
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 9` contraction entries are loaded per lane, and the explicit minor/cofactor
/// formulation uses nine determinant multiplications without recomputing a determinant product.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 8 x-reference excitation caches.
/// - `w_ex`: 8 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 8 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_l3_prepared_f64x8(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 8]; 9];
        let mut lane = 0;

        while lane < 8 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 3];
            let mut cols = [0usize; 3];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..3 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly `9` contraction entries, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            d[2][lane] = y0[rows[0] * n + cols[2]];

            d[3][lane] = x0[rows[1] * n + cols[0]];
            d[4][lane] = x0[rows[1] * n + cols[1]];
            d[5][lane] = y0[rows[1] * n + cols[2]];

            d[6][lane] = x0[rows[2] * n + cols[0]];
            d[7][lane] = x0[rows[2] * n + cols[1]];
            d[8][lane] = x0[rows[2] * n + cols[2]];
            lane += 1;
        }

        // The three distinct `2 \\times 2` minors require six products and the first-row
        // expansion requires three. No determinant product is recomputed.
        let m0 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[4].as_ptr()),
            _mm512_loadu_pd(d[8].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[7].as_ptr()),
            ),
        );
        let m1 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[3].as_ptr()),
            _mm512_loadu_pd(d[8].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[6].as_ptr()),
            ),
        );
        let m2 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[3].as_ptr()),
            _mm512_loadu_pd(d[7].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[6].as_ptr()),
            ),
        );
        let det_v = _mm512_mul_pd(_mm512_loadu_pd(d[0].as_ptr()), m0);
        let det_v = _mm512_fnmadd_pd(_mm512_loadu_pd(d[1].as_ptr()), m1, det_v);
        let det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[2].as_ptr()), m2, det_v);
        let overlap_v = _mm512_mul_pd(det_v, _mm512_set1_pd(pref));
        _mm512_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 4 independent real fixed-rank `L = 4` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX2/FMA` arithmetic evaluates 4
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 16` contraction entries are loaded per lane, and the explicit minor/cofactor
/// formulation uses 28 determinant multiplications without recomputing a determinant product.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 4 x-reference excitation caches.
/// - `w_ex`: 4 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 4 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_l4_prepared_f64x4(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 4]; 16];
        let mut lane = 0;

        while lane < 4 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 4];
            let mut cols = [0usize; 4];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..4 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly `16` contraction entries, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            d[2][lane] = y0[rows[0] * n + cols[2]];
            d[3][lane] = y0[rows[0] * n + cols[3]];

            d[4][lane] = x0[rows[1] * n + cols[0]];
            d[5][lane] = x0[rows[1] * n + cols[1]];
            d[6][lane] = y0[rows[1] * n + cols[2]];
            d[7][lane] = y0[rows[1] * n + cols[3]];

            d[8][lane] = x0[rows[2] * n + cols[0]];
            d[9][lane] = x0[rows[2] * n + cols[1]];
            d[10][lane] = x0[rows[2] * n + cols[2]];
            d[11][lane] = y0[rows[2] * n + cols[3]];

            d[12][lane] = x0[rows[3] * n + cols[0]];
            d[13][lane] = x0[rows[3] * n + cols[1]];
            d[14][lane] = x0[rows[3] * n + cols[2]];
            d[15][lane] = x0[rows[3] * n + cols[3]];
            lane += 1;
        }

        // The six distinct `2 \\times 2` minors of the final two rows require 12 products.
        // Reusing them in all four `3 \\times 3` cofactors requires 12 more products, and the
        // first-row expansion requires four. No determinant product is recomputed.
        let s0 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[10].as_ptr()),
            _mm256_loadu_pd(d[15].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[11].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
            ),
        );
        let s1 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[9].as_ptr()),
            _mm256_loadu_pd(d[15].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[11].as_ptr()),
                _mm256_loadu_pd(d[13].as_ptr()),
            ),
        );
        let s2 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[9].as_ptr()),
            _mm256_loadu_pd(d[14].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[10].as_ptr()),
                _mm256_loadu_pd(d[13].as_ptr()),
            ),
        );
        let s3 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[8].as_ptr()),
            _mm256_loadu_pd(d[15].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[11].as_ptr()),
                _mm256_loadu_pd(d[12].as_ptr()),
            ),
        );
        let s4 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[8].as_ptr()),
            _mm256_loadu_pd(d[14].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[10].as_ptr()),
                _mm256_loadu_pd(d[12].as_ptr()),
            ),
        );
        let s5 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[8].as_ptr()),
            _mm256_loadu_pd(d[13].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_loadu_pd(d[12].as_ptr()),
            ),
        );

        let c0 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[5].as_ptr()),
            s0,
            _mm256_mul_pd(_mm256_loadu_pd(d[6].as_ptr()), s1),
        );
        let c0 = _mm256_fmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), s2, c0);
        let c1 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[4].as_ptr()),
            s0,
            _mm256_mul_pd(_mm256_loadu_pd(d[6].as_ptr()), s3),
        );
        let c1 = _mm256_fmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), s4, c1);
        let c2 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[4].as_ptr()),
            s1,
            _mm256_mul_pd(_mm256_loadu_pd(d[5].as_ptr()), s3),
        );
        let c2 = _mm256_fmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), s5, c2);
        let c3 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[4].as_ptr()),
            s2,
            _mm256_mul_pd(_mm256_loadu_pd(d[5].as_ptr()), s4),
        );
        let c3 = _mm256_fmadd_pd(_mm256_loadu_pd(d[6].as_ptr()), s5, c3);

        let det_v = _mm256_mul_pd(_mm256_loadu_pd(d[0].as_ptr()), c0);
        let det_v = _mm256_fnmadd_pd(_mm256_loadu_pd(d[1].as_ptr()), c1, det_v);
        let det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[2].as_ptr()), c2, det_v);
        let det_v = _mm256_fnmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), c3, det_v);
        let overlap_v = _mm256_mul_pd(det_v, _mm256_set1_pd(pref));
        _mm256_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 4` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates 8
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 16` contraction entries are loaded per lane, and the explicit minor/cofactor
/// formulation uses 28 determinant multiplications without recomputing a determinant product.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 8 x-reference excitation caches.
/// - `w_ex`: 8 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 8 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_l4_prepared_f64x8(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 8]; 16];
        let mut lane = 0;

        while lane < 8 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 4];
            let mut cols = [0usize; 4];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..4 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly `16` contraction entries, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            d[2][lane] = y0[rows[0] * n + cols[2]];
            d[3][lane] = y0[rows[0] * n + cols[3]];

            d[4][lane] = x0[rows[1] * n + cols[0]];
            d[5][lane] = x0[rows[1] * n + cols[1]];
            d[6][lane] = y0[rows[1] * n + cols[2]];
            d[7][lane] = y0[rows[1] * n + cols[3]];

            d[8][lane] = x0[rows[2] * n + cols[0]];
            d[9][lane] = x0[rows[2] * n + cols[1]];
            d[10][lane] = x0[rows[2] * n + cols[2]];
            d[11][lane] = y0[rows[2] * n + cols[3]];

            d[12][lane] = x0[rows[3] * n + cols[0]];
            d[13][lane] = x0[rows[3] * n + cols[1]];
            d[14][lane] = x0[rows[3] * n + cols[2]];
            d[15][lane] = x0[rows[3] * n + cols[3]];
            lane += 1;
        }

        // The six distinct `2 \\times 2` minors of the final two rows require 12 products.
        // Reusing them in all four `3 \\times 3` cofactors requires 12 more products, and the
        // first-row expansion requires four. No determinant product is recomputed.
        let s0 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[10].as_ptr()),
            _mm512_loadu_pd(d[15].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[11].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
            ),
        );
        let s1 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[9].as_ptr()),
            _mm512_loadu_pd(d[15].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[11].as_ptr()),
                _mm512_loadu_pd(d[13].as_ptr()),
            ),
        );
        let s2 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[9].as_ptr()),
            _mm512_loadu_pd(d[14].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[10].as_ptr()),
                _mm512_loadu_pd(d[13].as_ptr()),
            ),
        );
        let s3 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[8].as_ptr()),
            _mm512_loadu_pd(d[15].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[11].as_ptr()),
                _mm512_loadu_pd(d[12].as_ptr()),
            ),
        );
        let s4 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[8].as_ptr()),
            _mm512_loadu_pd(d[14].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[10].as_ptr()),
                _mm512_loadu_pd(d[12].as_ptr()),
            ),
        );
        let s5 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[8].as_ptr()),
            _mm512_loadu_pd(d[13].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_loadu_pd(d[12].as_ptr()),
            ),
        );

        let c0 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[5].as_ptr()),
            s0,
            _mm512_mul_pd(_mm512_loadu_pd(d[6].as_ptr()), s1),
        );
        let c0 = _mm512_fmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), s2, c0);
        let c1 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[4].as_ptr()),
            s0,
            _mm512_mul_pd(_mm512_loadu_pd(d[6].as_ptr()), s3),
        );
        let c1 = _mm512_fmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), s4, c1);
        let c2 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[4].as_ptr()),
            s1,
            _mm512_mul_pd(_mm512_loadu_pd(d[5].as_ptr()), s3),
        );
        let c2 = _mm512_fmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), s5, c2);
        let c3 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[4].as_ptr()),
            s2,
            _mm512_mul_pd(_mm512_loadu_pd(d[5].as_ptr()), s4),
        );
        let c3 = _mm512_fmadd_pd(_mm512_loadu_pd(d[6].as_ptr()), s5, c3);

        let det_v = _mm512_mul_pd(_mm512_loadu_pd(d[0].as_ptr()), c0);
        let det_v = _mm512_fnmadd_pd(_mm512_loadu_pd(d[1].as_ptr()), c1, det_v);
        let det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[2].as_ptr()), c2, det_v);
        let det_v = _mm512_fnmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), c3, det_v);
        let overlap_v = _mm512_mul_pd(det_v, _mm512_set1_pd(pref));
        _mm512_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 4 independent real fixed-rank `L = 5` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX2/FMA` arithmetic evaluates 4
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 25` contraction entries are loaded per lane. Within the explicit
/// minor/cofactor hierarchy, every distinct column-subset minor is formed exactly once:
/// `10` rank-2 minors require `20` products, `10` rank-3 minors require `30` products,
/// `5` rank-4 minors require `20` products, and the first-row expansion requires `5`.
/// This gives `75` determinant multiplications with no determinant product recomputed.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 4 x-reference excitation caches.
/// - `w_ex`: 4 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 4 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_l5_prepared_f64x4(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 4]; 25];
        let mut lane = 0;

        while lane < 4 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 5];
            let mut cols = [0usize; 5];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..5 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly `25` contraction entries, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            d[2][lane] = y0[rows[0] * n + cols[2]];
            d[3][lane] = y0[rows[0] * n + cols[3]];
            d[4][lane] = y0[rows[0] * n + cols[4]];

            d[5][lane] = x0[rows[1] * n + cols[0]];
            d[6][lane] = x0[rows[1] * n + cols[1]];
            d[7][lane] = y0[rows[1] * n + cols[2]];
            d[8][lane] = y0[rows[1] * n + cols[3]];
            d[9][lane] = y0[rows[1] * n + cols[4]];

            d[10][lane] = x0[rows[2] * n + cols[0]];
            d[11][lane] = x0[rows[2] * n + cols[1]];
            d[12][lane] = x0[rows[2] * n + cols[2]];
            d[13][lane] = y0[rows[2] * n + cols[3]];
            d[14][lane] = y0[rows[2] * n + cols[4]];

            d[15][lane] = x0[rows[3] * n + cols[0]];
            d[16][lane] = x0[rows[3] * n + cols[1]];
            d[17][lane] = x0[rows[3] * n + cols[2]];
            d[18][lane] = x0[rows[3] * n + cols[3]];
            d[19][lane] = y0[rows[3] * n + cols[4]];

            d[20][lane] = x0[rows[4] * n + cols[0]];
            d[21][lane] = x0[rows[4] * n + cols[1]];
            d[22][lane] = x0[rows[4] * n + cols[2]];
            d[23][lane] = x0[rows[4] * n + cols[3]];
            d[24][lane] = x0[rows[4] * n + cols[4]];
            lane += 1;
        }

        let m2_01 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[15].as_ptr()),
            _mm256_loadu_pd(d[21].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[16].as_ptr()),
                _mm256_loadu_pd(d[20].as_ptr()),
            ),
        );
        let m2_02 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[15].as_ptr()),
            _mm256_loadu_pd(d[22].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[17].as_ptr()),
                _mm256_loadu_pd(d[20].as_ptr()),
            ),
        );
        let m2_03 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[15].as_ptr()),
            _mm256_loadu_pd(d[23].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[18].as_ptr()),
                _mm256_loadu_pd(d[20].as_ptr()),
            ),
        );
        let m2_04 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[15].as_ptr()),
            _mm256_loadu_pd(d[24].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[19].as_ptr()),
                _mm256_loadu_pd(d[20].as_ptr()),
            ),
        );
        let m2_12 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[16].as_ptr()),
            _mm256_loadu_pd(d[22].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[17].as_ptr()),
                _mm256_loadu_pd(d[21].as_ptr()),
            ),
        );
        let m2_13 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[16].as_ptr()),
            _mm256_loadu_pd(d[23].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[18].as_ptr()),
                _mm256_loadu_pd(d[21].as_ptr()),
            ),
        );
        let m2_14 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[16].as_ptr()),
            _mm256_loadu_pd(d[24].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[19].as_ptr()),
                _mm256_loadu_pd(d[21].as_ptr()),
            ),
        );
        let m2_23 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[17].as_ptr()),
            _mm256_loadu_pd(d[23].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[18].as_ptr()),
                _mm256_loadu_pd(d[22].as_ptr()),
            ),
        );
        let m2_24 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[17].as_ptr()),
            _mm256_loadu_pd(d[24].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[19].as_ptr()),
                _mm256_loadu_pd(d[22].as_ptr()),
            ),
        );
        let m2_34 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[18].as_ptr()),
            _mm256_loadu_pd(d[24].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[19].as_ptr()),
                _mm256_loadu_pd(d[23].as_ptr()),
            ),
        );

        let m3_012 = _mm256_mul_pd(_mm256_loadu_pd(d[10].as_ptr()), m2_12);
        let m3_012 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[11].as_ptr()), m2_02, m3_012);
        let m3_012 = _mm256_fmadd_pd(_mm256_loadu_pd(d[12].as_ptr()), m2_01, m3_012);
        let m3_013 = _mm256_mul_pd(_mm256_loadu_pd(d[10].as_ptr()), m2_13);
        let m3_013 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[11].as_ptr()), m2_03, m3_013);
        let m3_013 = _mm256_fmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m2_01, m3_013);
        let m3_014 = _mm256_mul_pd(_mm256_loadu_pd(d[10].as_ptr()), m2_14);
        let m3_014 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[11].as_ptr()), m2_04, m3_014);
        let m3_014 = _mm256_fmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m2_01, m3_014);
        let m3_023 = _mm256_mul_pd(_mm256_loadu_pd(d[10].as_ptr()), m2_23);
        let m3_023 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[12].as_ptr()), m2_03, m3_023);
        let m3_023 = _mm256_fmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m2_02, m3_023);
        let m3_024 = _mm256_mul_pd(_mm256_loadu_pd(d[10].as_ptr()), m2_24);
        let m3_024 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[12].as_ptr()), m2_04, m3_024);
        let m3_024 = _mm256_fmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m2_02, m3_024);
        let m3_034 = _mm256_mul_pd(_mm256_loadu_pd(d[10].as_ptr()), m2_34);
        let m3_034 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m2_04, m3_034);
        let m3_034 = _mm256_fmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m2_03, m3_034);
        let m3_123 = _mm256_mul_pd(_mm256_loadu_pd(d[11].as_ptr()), m2_23);
        let m3_123 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[12].as_ptr()), m2_13, m3_123);
        let m3_123 = _mm256_fmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m2_12, m3_123);
        let m3_124 = _mm256_mul_pd(_mm256_loadu_pd(d[11].as_ptr()), m2_24);
        let m3_124 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[12].as_ptr()), m2_14, m3_124);
        let m3_124 = _mm256_fmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m2_12, m3_124);
        let m3_134 = _mm256_mul_pd(_mm256_loadu_pd(d[11].as_ptr()), m2_34);
        let m3_134 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m2_14, m3_134);
        let m3_134 = _mm256_fmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m2_13, m3_134);
        let m3_234 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m2_34);
        let m3_234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m2_24, m3_234);
        let m3_234 = _mm256_fmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m2_23, m3_234);

        let m4_0123 = _mm256_mul_pd(_mm256_loadu_pd(d[5].as_ptr()), m3_123);
        let m4_0123 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[6].as_ptr()), m3_023, m4_0123);
        let m4_0123 = _mm256_fmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m3_013, m4_0123);
        let m4_0123 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[8].as_ptr()), m3_012, m4_0123);
        let m4_0124 = _mm256_mul_pd(_mm256_loadu_pd(d[5].as_ptr()), m3_124);
        let m4_0124 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[6].as_ptr()), m3_024, m4_0124);
        let m4_0124 = _mm256_fmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m3_014, m4_0124);
        let m4_0124 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[9].as_ptr()), m3_012, m4_0124);
        let m4_0134 = _mm256_mul_pd(_mm256_loadu_pd(d[5].as_ptr()), m3_134);
        let m4_0134 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[6].as_ptr()), m3_034, m4_0134);
        let m4_0134 = _mm256_fmadd_pd(_mm256_loadu_pd(d[8].as_ptr()), m3_014, m4_0134);
        let m4_0134 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[9].as_ptr()), m3_013, m4_0134);
        let m4_0234 = _mm256_mul_pd(_mm256_loadu_pd(d[5].as_ptr()), m3_234);
        let m4_0234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m3_034, m4_0234);
        let m4_0234 = _mm256_fmadd_pd(_mm256_loadu_pd(d[8].as_ptr()), m3_024, m4_0234);
        let m4_0234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[9].as_ptr()), m3_023, m4_0234);
        let m4_1234 = _mm256_mul_pd(_mm256_loadu_pd(d[6].as_ptr()), m3_234);
        let m4_1234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m3_134, m4_1234);
        let m4_1234 = _mm256_fmadd_pd(_mm256_loadu_pd(d[8].as_ptr()), m3_124, m4_1234);
        let m4_1234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[9].as_ptr()), m3_123, m4_1234);

        let det_v = _mm256_mul_pd(_mm256_loadu_pd(d[0].as_ptr()), m4_1234);
        let det_v = _mm256_fnmadd_pd(_mm256_loadu_pd(d[1].as_ptr()), m4_0234, det_v);
        let det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[2].as_ptr()), m4_0134, det_v);
        let det_v = _mm256_fnmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m4_0124, det_v);
        let det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[4].as_ptr()), m4_0123, det_v);
        let overlap_v = _mm256_mul_pd(det_v, _mm256_set1_pd(pref));
        _mm256_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 5` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates 8
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 25` contraction entries are loaded per lane. Within the explicit
/// minor/cofactor hierarchy, every distinct column-subset minor is formed exactly once:
/// `10` rank-2 minors require `20` products, `10` rank-3 minors require `30` products,
/// `5` rank-4 minors require `20` products, and the first-row expansion requires `5`.
/// This gives `75` determinant multiplications with no determinant product recomputed.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 8 x-reference excitation caches.
/// - `w_ex`: 8 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 8 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_l5_prepared_f64x8(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 8]; 25];
        let mut lane = 0;

        while lane < 8 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 5];
            let mut cols = [0usize; 5];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..5 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly `25` contraction entries, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            d[2][lane] = y0[rows[0] * n + cols[2]];
            d[3][lane] = y0[rows[0] * n + cols[3]];
            d[4][lane] = y0[rows[0] * n + cols[4]];

            d[5][lane] = x0[rows[1] * n + cols[0]];
            d[6][lane] = x0[rows[1] * n + cols[1]];
            d[7][lane] = y0[rows[1] * n + cols[2]];
            d[8][lane] = y0[rows[1] * n + cols[3]];
            d[9][lane] = y0[rows[1] * n + cols[4]];

            d[10][lane] = x0[rows[2] * n + cols[0]];
            d[11][lane] = x0[rows[2] * n + cols[1]];
            d[12][lane] = x0[rows[2] * n + cols[2]];
            d[13][lane] = y0[rows[2] * n + cols[3]];
            d[14][lane] = y0[rows[2] * n + cols[4]];

            d[15][lane] = x0[rows[3] * n + cols[0]];
            d[16][lane] = x0[rows[3] * n + cols[1]];
            d[17][lane] = x0[rows[3] * n + cols[2]];
            d[18][lane] = x0[rows[3] * n + cols[3]];
            d[19][lane] = y0[rows[3] * n + cols[4]];

            d[20][lane] = x0[rows[4] * n + cols[0]];
            d[21][lane] = x0[rows[4] * n + cols[1]];
            d[22][lane] = x0[rows[4] * n + cols[2]];
            d[23][lane] = x0[rows[4] * n + cols[3]];
            d[24][lane] = x0[rows[4] * n + cols[4]];
            lane += 1;
        }

        let m2_01 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[15].as_ptr()),
            _mm512_loadu_pd(d[21].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[16].as_ptr()),
                _mm512_loadu_pd(d[20].as_ptr()),
            ),
        );
        let m2_02 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[15].as_ptr()),
            _mm512_loadu_pd(d[22].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[17].as_ptr()),
                _mm512_loadu_pd(d[20].as_ptr()),
            ),
        );
        let m2_03 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[15].as_ptr()),
            _mm512_loadu_pd(d[23].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[18].as_ptr()),
                _mm512_loadu_pd(d[20].as_ptr()),
            ),
        );
        let m2_04 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[15].as_ptr()),
            _mm512_loadu_pd(d[24].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[19].as_ptr()),
                _mm512_loadu_pd(d[20].as_ptr()),
            ),
        );
        let m2_12 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[16].as_ptr()),
            _mm512_loadu_pd(d[22].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[17].as_ptr()),
                _mm512_loadu_pd(d[21].as_ptr()),
            ),
        );
        let m2_13 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[16].as_ptr()),
            _mm512_loadu_pd(d[23].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[18].as_ptr()),
                _mm512_loadu_pd(d[21].as_ptr()),
            ),
        );
        let m2_14 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[16].as_ptr()),
            _mm512_loadu_pd(d[24].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[19].as_ptr()),
                _mm512_loadu_pd(d[21].as_ptr()),
            ),
        );
        let m2_23 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[17].as_ptr()),
            _mm512_loadu_pd(d[23].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[18].as_ptr()),
                _mm512_loadu_pd(d[22].as_ptr()),
            ),
        );
        let m2_24 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[17].as_ptr()),
            _mm512_loadu_pd(d[24].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[19].as_ptr()),
                _mm512_loadu_pd(d[22].as_ptr()),
            ),
        );
        let m2_34 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[18].as_ptr()),
            _mm512_loadu_pd(d[24].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[19].as_ptr()),
                _mm512_loadu_pd(d[23].as_ptr()),
            ),
        );

        let m3_012 = _mm512_mul_pd(_mm512_loadu_pd(d[10].as_ptr()), m2_12);
        let m3_012 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[11].as_ptr()), m2_02, m3_012);
        let m3_012 = _mm512_fmadd_pd(_mm512_loadu_pd(d[12].as_ptr()), m2_01, m3_012);
        let m3_013 = _mm512_mul_pd(_mm512_loadu_pd(d[10].as_ptr()), m2_13);
        let m3_013 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[11].as_ptr()), m2_03, m3_013);
        let m3_013 = _mm512_fmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m2_01, m3_013);
        let m3_014 = _mm512_mul_pd(_mm512_loadu_pd(d[10].as_ptr()), m2_14);
        let m3_014 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[11].as_ptr()), m2_04, m3_014);
        let m3_014 = _mm512_fmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m2_01, m3_014);
        let m3_023 = _mm512_mul_pd(_mm512_loadu_pd(d[10].as_ptr()), m2_23);
        let m3_023 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[12].as_ptr()), m2_03, m3_023);
        let m3_023 = _mm512_fmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m2_02, m3_023);
        let m3_024 = _mm512_mul_pd(_mm512_loadu_pd(d[10].as_ptr()), m2_24);
        let m3_024 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[12].as_ptr()), m2_04, m3_024);
        let m3_024 = _mm512_fmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m2_02, m3_024);
        let m3_034 = _mm512_mul_pd(_mm512_loadu_pd(d[10].as_ptr()), m2_34);
        let m3_034 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m2_04, m3_034);
        let m3_034 = _mm512_fmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m2_03, m3_034);
        let m3_123 = _mm512_mul_pd(_mm512_loadu_pd(d[11].as_ptr()), m2_23);
        let m3_123 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[12].as_ptr()), m2_13, m3_123);
        let m3_123 = _mm512_fmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m2_12, m3_123);
        let m3_124 = _mm512_mul_pd(_mm512_loadu_pd(d[11].as_ptr()), m2_24);
        let m3_124 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[12].as_ptr()), m2_14, m3_124);
        let m3_124 = _mm512_fmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m2_12, m3_124);
        let m3_134 = _mm512_mul_pd(_mm512_loadu_pd(d[11].as_ptr()), m2_34);
        let m3_134 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m2_14, m3_134);
        let m3_134 = _mm512_fmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m2_13, m3_134);
        let m3_234 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m2_34);
        let m3_234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m2_24, m3_234);
        let m3_234 = _mm512_fmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m2_23, m3_234);

        let m4_0123 = _mm512_mul_pd(_mm512_loadu_pd(d[5].as_ptr()), m3_123);
        let m4_0123 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[6].as_ptr()), m3_023, m4_0123);
        let m4_0123 = _mm512_fmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m3_013, m4_0123);
        let m4_0123 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[8].as_ptr()), m3_012, m4_0123);
        let m4_0124 = _mm512_mul_pd(_mm512_loadu_pd(d[5].as_ptr()), m3_124);
        let m4_0124 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[6].as_ptr()), m3_024, m4_0124);
        let m4_0124 = _mm512_fmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m3_014, m4_0124);
        let m4_0124 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[9].as_ptr()), m3_012, m4_0124);
        let m4_0134 = _mm512_mul_pd(_mm512_loadu_pd(d[5].as_ptr()), m3_134);
        let m4_0134 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[6].as_ptr()), m3_034, m4_0134);
        let m4_0134 = _mm512_fmadd_pd(_mm512_loadu_pd(d[8].as_ptr()), m3_014, m4_0134);
        let m4_0134 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[9].as_ptr()), m3_013, m4_0134);
        let m4_0234 = _mm512_mul_pd(_mm512_loadu_pd(d[5].as_ptr()), m3_234);
        let m4_0234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m3_034, m4_0234);
        let m4_0234 = _mm512_fmadd_pd(_mm512_loadu_pd(d[8].as_ptr()), m3_024, m4_0234);
        let m4_0234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[9].as_ptr()), m3_023, m4_0234);
        let m4_1234 = _mm512_mul_pd(_mm512_loadu_pd(d[6].as_ptr()), m3_234);
        let m4_1234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m3_134, m4_1234);
        let m4_1234 = _mm512_fmadd_pd(_mm512_loadu_pd(d[8].as_ptr()), m3_124, m4_1234);
        let m4_1234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[9].as_ptr()), m3_123, m4_1234);

        let det_v = _mm512_mul_pd(_mm512_loadu_pd(d[0].as_ptr()), m4_1234);
        let det_v = _mm512_fnmadd_pd(_mm512_loadu_pd(d[1].as_ptr()), m4_0234, det_v);
        let det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[2].as_ptr()), m4_0134, det_v);
        let det_v = _mm512_fnmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m4_0124, det_v);
        let det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[4].as_ptr()), m4_0123, det_v);
        let overlap_v = _mm512_mul_pd(det_v, _mm512_set1_pd(pref));
        _mm512_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 4 independent real fixed-rank `L = 6` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX2/FMA` arithmetic evaluates 4
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 36` contraction entries are loaded per lane. Within the explicit
/// minor/cofactor hierarchy, every distinct column-subset minor is formed exactly once:
/// `15` rank-2 minors require `30` products, `20` rank-3 minors require `60` products,
/// `15` rank-4 minors require `60` products and `6` rank-5 minors require `30` products,
/// while the first-row expansion requires `6`.
/// This gives `186` determinant multiplications with no determinant product recomputed.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 4 x-reference excitation caches.
/// - `w_ex`: 4 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 4 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_l6_prepared_f64x4(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 4]; 36];
        let mut lane = 0;

        while lane < 4 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 6];
            let mut cols = [0usize; 6];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..6 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly `36` contraction entries, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            d[2][lane] = y0[rows[0] * n + cols[2]];
            d[3][lane] = y0[rows[0] * n + cols[3]];
            d[4][lane] = y0[rows[0] * n + cols[4]];
            d[5][lane] = y0[rows[0] * n + cols[5]];

            d[6][lane] = x0[rows[1] * n + cols[0]];
            d[7][lane] = x0[rows[1] * n + cols[1]];
            d[8][lane] = y0[rows[1] * n + cols[2]];
            d[9][lane] = y0[rows[1] * n + cols[3]];
            d[10][lane] = y0[rows[1] * n + cols[4]];
            d[11][lane] = y0[rows[1] * n + cols[5]];

            d[12][lane] = x0[rows[2] * n + cols[0]];
            d[13][lane] = x0[rows[2] * n + cols[1]];
            d[14][lane] = x0[rows[2] * n + cols[2]];
            d[15][lane] = y0[rows[2] * n + cols[3]];
            d[16][lane] = y0[rows[2] * n + cols[4]];
            d[17][lane] = y0[rows[2] * n + cols[5]];

            d[18][lane] = x0[rows[3] * n + cols[0]];
            d[19][lane] = x0[rows[3] * n + cols[1]];
            d[20][lane] = x0[rows[3] * n + cols[2]];
            d[21][lane] = x0[rows[3] * n + cols[3]];
            d[22][lane] = y0[rows[3] * n + cols[4]];
            d[23][lane] = y0[rows[3] * n + cols[5]];

            d[24][lane] = x0[rows[4] * n + cols[0]];
            d[25][lane] = x0[rows[4] * n + cols[1]];
            d[26][lane] = x0[rows[4] * n + cols[2]];
            d[27][lane] = x0[rows[4] * n + cols[3]];
            d[28][lane] = x0[rows[4] * n + cols[4]];
            d[29][lane] = y0[rows[4] * n + cols[5]];

            d[30][lane] = x0[rows[5] * n + cols[0]];
            d[31][lane] = x0[rows[5] * n + cols[1]];
            d[32][lane] = x0[rows[5] * n + cols[2]];
            d[33][lane] = x0[rows[5] * n + cols[3]];
            d[34][lane] = x0[rows[5] * n + cols[4]];
            d[35][lane] = x0[rows[5] * n + cols[5]];
            lane += 1;
        }

        let m2_01 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[24].as_ptr()),
            _mm256_loadu_pd(d[31].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[25].as_ptr()),
                _mm256_loadu_pd(d[30].as_ptr()),
            ),
        );
        let m2_02 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[24].as_ptr()),
            _mm256_loadu_pd(d[32].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[26].as_ptr()),
                _mm256_loadu_pd(d[30].as_ptr()),
            ),
        );
        let m2_03 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[24].as_ptr()),
            _mm256_loadu_pd(d[33].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[27].as_ptr()),
                _mm256_loadu_pd(d[30].as_ptr()),
            ),
        );
        let m2_04 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[24].as_ptr()),
            _mm256_loadu_pd(d[34].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[28].as_ptr()),
                _mm256_loadu_pd(d[30].as_ptr()),
            ),
        );
        let m2_05 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[24].as_ptr()),
            _mm256_loadu_pd(d[35].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[29].as_ptr()),
                _mm256_loadu_pd(d[30].as_ptr()),
            ),
        );
        let m2_12 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[25].as_ptr()),
            _mm256_loadu_pd(d[32].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[26].as_ptr()),
                _mm256_loadu_pd(d[31].as_ptr()),
            ),
        );
        let m2_13 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[25].as_ptr()),
            _mm256_loadu_pd(d[33].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[27].as_ptr()),
                _mm256_loadu_pd(d[31].as_ptr()),
            ),
        );
        let m2_14 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[25].as_ptr()),
            _mm256_loadu_pd(d[34].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[28].as_ptr()),
                _mm256_loadu_pd(d[31].as_ptr()),
            ),
        );
        let m2_15 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[25].as_ptr()),
            _mm256_loadu_pd(d[35].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[29].as_ptr()),
                _mm256_loadu_pd(d[31].as_ptr()),
            ),
        );
        let m2_23 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[26].as_ptr()),
            _mm256_loadu_pd(d[33].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[27].as_ptr()),
                _mm256_loadu_pd(d[32].as_ptr()),
            ),
        );
        let m2_24 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[26].as_ptr()),
            _mm256_loadu_pd(d[34].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[28].as_ptr()),
                _mm256_loadu_pd(d[32].as_ptr()),
            ),
        );
        let m2_25 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[26].as_ptr()),
            _mm256_loadu_pd(d[35].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[29].as_ptr()),
                _mm256_loadu_pd(d[32].as_ptr()),
            ),
        );
        let m2_34 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[27].as_ptr()),
            _mm256_loadu_pd(d[34].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[28].as_ptr()),
                _mm256_loadu_pd(d[33].as_ptr()),
            ),
        );
        let m2_35 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[27].as_ptr()),
            _mm256_loadu_pd(d[35].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[29].as_ptr()),
                _mm256_loadu_pd(d[33].as_ptr()),
            ),
        );
        let m2_45 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[28].as_ptr()),
            _mm256_loadu_pd(d[35].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[29].as_ptr()),
                _mm256_loadu_pd(d[34].as_ptr()),
            ),
        );

        let m3_012 = _mm256_mul_pd(_mm256_loadu_pd(d[18].as_ptr()), m2_12);
        let m3_012 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[19].as_ptr()), m2_02, m3_012);
        let m3_012 = _mm256_fmadd_pd(_mm256_loadu_pd(d[20].as_ptr()), m2_01, m3_012);
        let m3_013 = _mm256_mul_pd(_mm256_loadu_pd(d[18].as_ptr()), m2_13);
        let m3_013 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[19].as_ptr()), m2_03, m3_013);
        let m3_013 = _mm256_fmadd_pd(_mm256_loadu_pd(d[21].as_ptr()), m2_01, m3_013);
        let m3_014 = _mm256_mul_pd(_mm256_loadu_pd(d[18].as_ptr()), m2_14);
        let m3_014 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[19].as_ptr()), m2_04, m3_014);
        let m3_014 = _mm256_fmadd_pd(_mm256_loadu_pd(d[22].as_ptr()), m2_01, m3_014);
        let m3_015 = _mm256_mul_pd(_mm256_loadu_pd(d[18].as_ptr()), m2_15);
        let m3_015 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[19].as_ptr()), m2_05, m3_015);
        let m3_015 = _mm256_fmadd_pd(_mm256_loadu_pd(d[23].as_ptr()), m2_01, m3_015);
        let m3_023 = _mm256_mul_pd(_mm256_loadu_pd(d[18].as_ptr()), m2_23);
        let m3_023 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[20].as_ptr()), m2_03, m3_023);
        let m3_023 = _mm256_fmadd_pd(_mm256_loadu_pd(d[21].as_ptr()), m2_02, m3_023);
        let m3_024 = _mm256_mul_pd(_mm256_loadu_pd(d[18].as_ptr()), m2_24);
        let m3_024 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[20].as_ptr()), m2_04, m3_024);
        let m3_024 = _mm256_fmadd_pd(_mm256_loadu_pd(d[22].as_ptr()), m2_02, m3_024);
        let m3_025 = _mm256_mul_pd(_mm256_loadu_pd(d[18].as_ptr()), m2_25);
        let m3_025 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[20].as_ptr()), m2_05, m3_025);
        let m3_025 = _mm256_fmadd_pd(_mm256_loadu_pd(d[23].as_ptr()), m2_02, m3_025);
        let m3_034 = _mm256_mul_pd(_mm256_loadu_pd(d[18].as_ptr()), m2_34);
        let m3_034 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[21].as_ptr()), m2_04, m3_034);
        let m3_034 = _mm256_fmadd_pd(_mm256_loadu_pd(d[22].as_ptr()), m2_03, m3_034);
        let m3_035 = _mm256_mul_pd(_mm256_loadu_pd(d[18].as_ptr()), m2_35);
        let m3_035 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[21].as_ptr()), m2_05, m3_035);
        let m3_035 = _mm256_fmadd_pd(_mm256_loadu_pd(d[23].as_ptr()), m2_03, m3_035);
        let m3_045 = _mm256_mul_pd(_mm256_loadu_pd(d[18].as_ptr()), m2_45);
        let m3_045 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[22].as_ptr()), m2_05, m3_045);
        let m3_045 = _mm256_fmadd_pd(_mm256_loadu_pd(d[23].as_ptr()), m2_04, m3_045);
        let m3_123 = _mm256_mul_pd(_mm256_loadu_pd(d[19].as_ptr()), m2_23);
        let m3_123 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[20].as_ptr()), m2_13, m3_123);
        let m3_123 = _mm256_fmadd_pd(_mm256_loadu_pd(d[21].as_ptr()), m2_12, m3_123);
        let m3_124 = _mm256_mul_pd(_mm256_loadu_pd(d[19].as_ptr()), m2_24);
        let m3_124 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[20].as_ptr()), m2_14, m3_124);
        let m3_124 = _mm256_fmadd_pd(_mm256_loadu_pd(d[22].as_ptr()), m2_12, m3_124);
        let m3_125 = _mm256_mul_pd(_mm256_loadu_pd(d[19].as_ptr()), m2_25);
        let m3_125 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[20].as_ptr()), m2_15, m3_125);
        let m3_125 = _mm256_fmadd_pd(_mm256_loadu_pd(d[23].as_ptr()), m2_12, m3_125);
        let m3_134 = _mm256_mul_pd(_mm256_loadu_pd(d[19].as_ptr()), m2_34);
        let m3_134 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[21].as_ptr()), m2_14, m3_134);
        let m3_134 = _mm256_fmadd_pd(_mm256_loadu_pd(d[22].as_ptr()), m2_13, m3_134);
        let m3_135 = _mm256_mul_pd(_mm256_loadu_pd(d[19].as_ptr()), m2_35);
        let m3_135 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[21].as_ptr()), m2_15, m3_135);
        let m3_135 = _mm256_fmadd_pd(_mm256_loadu_pd(d[23].as_ptr()), m2_13, m3_135);
        let m3_145 = _mm256_mul_pd(_mm256_loadu_pd(d[19].as_ptr()), m2_45);
        let m3_145 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[22].as_ptr()), m2_15, m3_145);
        let m3_145 = _mm256_fmadd_pd(_mm256_loadu_pd(d[23].as_ptr()), m2_14, m3_145);
        let m3_234 = _mm256_mul_pd(_mm256_loadu_pd(d[20].as_ptr()), m2_34);
        let m3_234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[21].as_ptr()), m2_24, m3_234);
        let m3_234 = _mm256_fmadd_pd(_mm256_loadu_pd(d[22].as_ptr()), m2_23, m3_234);
        let m3_235 = _mm256_mul_pd(_mm256_loadu_pd(d[20].as_ptr()), m2_35);
        let m3_235 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[21].as_ptr()), m2_25, m3_235);
        let m3_235 = _mm256_fmadd_pd(_mm256_loadu_pd(d[23].as_ptr()), m2_23, m3_235);
        let m3_245 = _mm256_mul_pd(_mm256_loadu_pd(d[20].as_ptr()), m2_45);
        let m3_245 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[22].as_ptr()), m2_25, m3_245);
        let m3_245 = _mm256_fmadd_pd(_mm256_loadu_pd(d[23].as_ptr()), m2_24, m3_245);
        let m3_345 = _mm256_mul_pd(_mm256_loadu_pd(d[21].as_ptr()), m2_45);
        let m3_345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[22].as_ptr()), m2_35, m3_345);
        let m3_345 = _mm256_fmadd_pd(_mm256_loadu_pd(d[23].as_ptr()), m2_34, m3_345);

        let m4_0123 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m3_123);
        let m4_0123 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m3_023, m4_0123);
        let m4_0123 = _mm256_fmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m3_013, m4_0123);
        let m4_0123 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[15].as_ptr()), m3_012, m4_0123);
        let m4_0124 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m3_124);
        let m4_0124 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m3_024, m4_0124);
        let m4_0124 = _mm256_fmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m3_014, m4_0124);
        let m4_0124 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[16].as_ptr()), m3_012, m4_0124);
        let m4_0125 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m3_125);
        let m4_0125 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m3_025, m4_0125);
        let m4_0125 = _mm256_fmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m3_015, m4_0125);
        let m4_0125 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[17].as_ptr()), m3_012, m4_0125);
        let m4_0134 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m3_134);
        let m4_0134 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m3_034, m4_0134);
        let m4_0134 = _mm256_fmadd_pd(_mm256_loadu_pd(d[15].as_ptr()), m3_014, m4_0134);
        let m4_0134 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[16].as_ptr()), m3_013, m4_0134);
        let m4_0135 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m3_135);
        let m4_0135 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m3_035, m4_0135);
        let m4_0135 = _mm256_fmadd_pd(_mm256_loadu_pd(d[15].as_ptr()), m3_015, m4_0135);
        let m4_0135 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[17].as_ptr()), m3_013, m4_0135);
        let m4_0145 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m3_145);
        let m4_0145 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[13].as_ptr()), m3_045, m4_0145);
        let m4_0145 = _mm256_fmadd_pd(_mm256_loadu_pd(d[16].as_ptr()), m3_015, m4_0145);
        let m4_0145 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[17].as_ptr()), m3_014, m4_0145);
        let m4_0234 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m3_234);
        let m4_0234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m3_034, m4_0234);
        let m4_0234 = _mm256_fmadd_pd(_mm256_loadu_pd(d[15].as_ptr()), m3_024, m4_0234);
        let m4_0234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[16].as_ptr()), m3_023, m4_0234);
        let m4_0235 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m3_235);
        let m4_0235 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m3_035, m4_0235);
        let m4_0235 = _mm256_fmadd_pd(_mm256_loadu_pd(d[15].as_ptr()), m3_025, m4_0235);
        let m4_0235 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[17].as_ptr()), m3_023, m4_0235);
        let m4_0245 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m3_245);
        let m4_0245 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m3_045, m4_0245);
        let m4_0245 = _mm256_fmadd_pd(_mm256_loadu_pd(d[16].as_ptr()), m3_025, m4_0245);
        let m4_0245 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[17].as_ptr()), m3_024, m4_0245);
        let m4_0345 = _mm256_mul_pd(_mm256_loadu_pd(d[12].as_ptr()), m3_345);
        let m4_0345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[15].as_ptr()), m3_045, m4_0345);
        let m4_0345 = _mm256_fmadd_pd(_mm256_loadu_pd(d[16].as_ptr()), m3_035, m4_0345);
        let m4_0345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[17].as_ptr()), m3_034, m4_0345);
        let m4_1234 = _mm256_mul_pd(_mm256_loadu_pd(d[13].as_ptr()), m3_234);
        let m4_1234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m3_134, m4_1234);
        let m4_1234 = _mm256_fmadd_pd(_mm256_loadu_pd(d[15].as_ptr()), m3_124, m4_1234);
        let m4_1234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[16].as_ptr()), m3_123, m4_1234);
        let m4_1235 = _mm256_mul_pd(_mm256_loadu_pd(d[13].as_ptr()), m3_235);
        let m4_1235 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m3_135, m4_1235);
        let m4_1235 = _mm256_fmadd_pd(_mm256_loadu_pd(d[15].as_ptr()), m3_125, m4_1235);
        let m4_1235 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[17].as_ptr()), m3_123, m4_1235);
        let m4_1245 = _mm256_mul_pd(_mm256_loadu_pd(d[13].as_ptr()), m3_245);
        let m4_1245 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[14].as_ptr()), m3_145, m4_1245);
        let m4_1245 = _mm256_fmadd_pd(_mm256_loadu_pd(d[16].as_ptr()), m3_125, m4_1245);
        let m4_1245 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[17].as_ptr()), m3_124, m4_1245);
        let m4_1345 = _mm256_mul_pd(_mm256_loadu_pd(d[13].as_ptr()), m3_345);
        let m4_1345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[15].as_ptr()), m3_145, m4_1345);
        let m4_1345 = _mm256_fmadd_pd(_mm256_loadu_pd(d[16].as_ptr()), m3_135, m4_1345);
        let m4_1345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[17].as_ptr()), m3_134, m4_1345);
        let m4_2345 = _mm256_mul_pd(_mm256_loadu_pd(d[14].as_ptr()), m3_345);
        let m4_2345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[15].as_ptr()), m3_245, m4_2345);
        let m4_2345 = _mm256_fmadd_pd(_mm256_loadu_pd(d[16].as_ptr()), m3_235, m4_2345);
        let m4_2345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[17].as_ptr()), m3_234, m4_2345);

        let m5_01234 = _mm256_mul_pd(_mm256_loadu_pd(d[6].as_ptr()), m4_1234);
        let m5_01234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m4_0234, m5_01234);
        let m5_01234 = _mm256_fmadd_pd(_mm256_loadu_pd(d[8].as_ptr()), m4_0134, m5_01234);
        let m5_01234 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[9].as_ptr()), m4_0124, m5_01234);
        let m5_01234 = _mm256_fmadd_pd(_mm256_loadu_pd(d[10].as_ptr()), m4_0123, m5_01234);
        let m5_01235 = _mm256_mul_pd(_mm256_loadu_pd(d[6].as_ptr()), m4_1235);
        let m5_01235 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m4_0235, m5_01235);
        let m5_01235 = _mm256_fmadd_pd(_mm256_loadu_pd(d[8].as_ptr()), m4_0135, m5_01235);
        let m5_01235 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[9].as_ptr()), m4_0125, m5_01235);
        let m5_01235 = _mm256_fmadd_pd(_mm256_loadu_pd(d[11].as_ptr()), m4_0123, m5_01235);
        let m5_01245 = _mm256_mul_pd(_mm256_loadu_pd(d[6].as_ptr()), m4_1245);
        let m5_01245 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m4_0245, m5_01245);
        let m5_01245 = _mm256_fmadd_pd(_mm256_loadu_pd(d[8].as_ptr()), m4_0145, m5_01245);
        let m5_01245 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[10].as_ptr()), m4_0125, m5_01245);
        let m5_01245 = _mm256_fmadd_pd(_mm256_loadu_pd(d[11].as_ptr()), m4_0124, m5_01245);
        let m5_01345 = _mm256_mul_pd(_mm256_loadu_pd(d[6].as_ptr()), m4_1345);
        let m5_01345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m4_0345, m5_01345);
        let m5_01345 = _mm256_fmadd_pd(_mm256_loadu_pd(d[9].as_ptr()), m4_0145, m5_01345);
        let m5_01345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[10].as_ptr()), m4_0135, m5_01345);
        let m5_01345 = _mm256_fmadd_pd(_mm256_loadu_pd(d[11].as_ptr()), m4_0134, m5_01345);
        let m5_02345 = _mm256_mul_pd(_mm256_loadu_pd(d[6].as_ptr()), m4_2345);
        let m5_02345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[8].as_ptr()), m4_0345, m5_02345);
        let m5_02345 = _mm256_fmadd_pd(_mm256_loadu_pd(d[9].as_ptr()), m4_0245, m5_02345);
        let m5_02345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[10].as_ptr()), m4_0235, m5_02345);
        let m5_02345 = _mm256_fmadd_pd(_mm256_loadu_pd(d[11].as_ptr()), m4_0234, m5_02345);
        let m5_12345 = _mm256_mul_pd(_mm256_loadu_pd(d[7].as_ptr()), m4_2345);
        let m5_12345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[8].as_ptr()), m4_1345, m5_12345);
        let m5_12345 = _mm256_fmadd_pd(_mm256_loadu_pd(d[9].as_ptr()), m4_1245, m5_12345);
        let m5_12345 = _mm256_fnmadd_pd(_mm256_loadu_pd(d[10].as_ptr()), m4_1235, m5_12345);
        let m5_12345 = _mm256_fmadd_pd(_mm256_loadu_pd(d[11].as_ptr()), m4_1234, m5_12345);

        let det_v = _mm256_mul_pd(_mm256_loadu_pd(d[0].as_ptr()), m5_12345);
        let det_v = _mm256_fnmadd_pd(_mm256_loadu_pd(d[1].as_ptr()), m5_02345, det_v);
        let det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[2].as_ptr()), m5_01345, det_v);
        let det_v = _mm256_fnmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m5_01245, det_v);
        let det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[4].as_ptr()), m5_01235, det_v);
        let det_v = _mm256_fnmadd_pd(_mm256_loadu_pd(d[5].as_ptr()), m5_01234, det_v);
        let overlap_v = _mm256_mul_pd(det_v, _mm256_set1_pd(pref));
        _mm256_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 6` overlaps for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates 8
/// independent contraction determinants without horizontal reductions between pairs.
/// Exactly `L^2 = 36` contraction entries are loaded per lane. Within the explicit
/// minor/cofactor hierarchy, every distinct column-subset minor is formed exactly once:
/// `15` rank-2 minors require `30` products, `20` rank-3 minors require `60` products,
/// `15` rank-4 minors require `60` products and `6` rank-5 minors require `30` products,
/// while the first-row expansion requires `6`.
/// This gives `186` determinant multiplications with no determinant product recomputed.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `x_ex`: 8 x-reference excitation caches.
/// - `w_ex`: 8 w-reference excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 8 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_l6_prepared_f64x8(
    w: &SameSpinView<'_, f64>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    unsafe {
        let n = w.n();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let pref = w.phase * w.tilde_s_prod;
        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;

        let mut d = [[0.0f64; 8]; 36];
        let mut lane = 0;

        while lane < 8 {
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);
            let mut rows = [0usize; 6];
            let mut cols = [0usize; 6];
            let x_rank = usize::from(x_data.rank);

            for i in 0..x_rank {
                rows[i] = usize::from(x_data.indices[4 + i]) - nocc;
                cols[i] = usize::from(x_data.indices[i]);
            }
            for i in x_rank..6 {
                let j = i - x_rank;
                rows[i] = nvirt + usize::from(w_data.indices[j]);
                cols[i] = usize::from(w_data.indices[4 + j]);
            }

            // Read exactly `36` contraction entries, the `L^2` input lower bound.
            d[0][lane] = x0[rows[0] * n + cols[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            d[2][lane] = y0[rows[0] * n + cols[2]];
            d[3][lane] = y0[rows[0] * n + cols[3]];
            d[4][lane] = y0[rows[0] * n + cols[4]];
            d[5][lane] = y0[rows[0] * n + cols[5]];

            d[6][lane] = x0[rows[1] * n + cols[0]];
            d[7][lane] = x0[rows[1] * n + cols[1]];
            d[8][lane] = y0[rows[1] * n + cols[2]];
            d[9][lane] = y0[rows[1] * n + cols[3]];
            d[10][lane] = y0[rows[1] * n + cols[4]];
            d[11][lane] = y0[rows[1] * n + cols[5]];

            d[12][lane] = x0[rows[2] * n + cols[0]];
            d[13][lane] = x0[rows[2] * n + cols[1]];
            d[14][lane] = x0[rows[2] * n + cols[2]];
            d[15][lane] = y0[rows[2] * n + cols[3]];
            d[16][lane] = y0[rows[2] * n + cols[4]];
            d[17][lane] = y0[rows[2] * n + cols[5]];

            d[18][lane] = x0[rows[3] * n + cols[0]];
            d[19][lane] = x0[rows[3] * n + cols[1]];
            d[20][lane] = x0[rows[3] * n + cols[2]];
            d[21][lane] = x0[rows[3] * n + cols[3]];
            d[22][lane] = y0[rows[3] * n + cols[4]];
            d[23][lane] = y0[rows[3] * n + cols[5]];

            d[24][lane] = x0[rows[4] * n + cols[0]];
            d[25][lane] = x0[rows[4] * n + cols[1]];
            d[26][lane] = x0[rows[4] * n + cols[2]];
            d[27][lane] = x0[rows[4] * n + cols[3]];
            d[28][lane] = x0[rows[4] * n + cols[4]];
            d[29][lane] = y0[rows[4] * n + cols[5]];

            d[30][lane] = x0[rows[5] * n + cols[0]];
            d[31][lane] = x0[rows[5] * n + cols[1]];
            d[32][lane] = x0[rows[5] * n + cols[2]];
            d[33][lane] = x0[rows[5] * n + cols[3]];
            d[34][lane] = x0[rows[5] * n + cols[4]];
            d[35][lane] = x0[rows[5] * n + cols[5]];
            lane += 1;
        }

        let m2_01 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[24].as_ptr()),
            _mm512_loadu_pd(d[31].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[25].as_ptr()),
                _mm512_loadu_pd(d[30].as_ptr()),
            ),
        );
        let m2_02 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[24].as_ptr()),
            _mm512_loadu_pd(d[32].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[26].as_ptr()),
                _mm512_loadu_pd(d[30].as_ptr()),
            ),
        );
        let m2_03 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[24].as_ptr()),
            _mm512_loadu_pd(d[33].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[27].as_ptr()),
                _mm512_loadu_pd(d[30].as_ptr()),
            ),
        );
        let m2_04 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[24].as_ptr()),
            _mm512_loadu_pd(d[34].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[28].as_ptr()),
                _mm512_loadu_pd(d[30].as_ptr()),
            ),
        );
        let m2_05 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[24].as_ptr()),
            _mm512_loadu_pd(d[35].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[29].as_ptr()),
                _mm512_loadu_pd(d[30].as_ptr()),
            ),
        );
        let m2_12 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[25].as_ptr()),
            _mm512_loadu_pd(d[32].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[26].as_ptr()),
                _mm512_loadu_pd(d[31].as_ptr()),
            ),
        );
        let m2_13 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[25].as_ptr()),
            _mm512_loadu_pd(d[33].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[27].as_ptr()),
                _mm512_loadu_pd(d[31].as_ptr()),
            ),
        );
        let m2_14 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[25].as_ptr()),
            _mm512_loadu_pd(d[34].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[28].as_ptr()),
                _mm512_loadu_pd(d[31].as_ptr()),
            ),
        );
        let m2_15 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[25].as_ptr()),
            _mm512_loadu_pd(d[35].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[29].as_ptr()),
                _mm512_loadu_pd(d[31].as_ptr()),
            ),
        );
        let m2_23 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[26].as_ptr()),
            _mm512_loadu_pd(d[33].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[27].as_ptr()),
                _mm512_loadu_pd(d[32].as_ptr()),
            ),
        );
        let m2_24 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[26].as_ptr()),
            _mm512_loadu_pd(d[34].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[28].as_ptr()),
                _mm512_loadu_pd(d[32].as_ptr()),
            ),
        );
        let m2_25 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[26].as_ptr()),
            _mm512_loadu_pd(d[35].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[29].as_ptr()),
                _mm512_loadu_pd(d[32].as_ptr()),
            ),
        );
        let m2_34 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[27].as_ptr()),
            _mm512_loadu_pd(d[34].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[28].as_ptr()),
                _mm512_loadu_pd(d[33].as_ptr()),
            ),
        );
        let m2_35 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[27].as_ptr()),
            _mm512_loadu_pd(d[35].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[29].as_ptr()),
                _mm512_loadu_pd(d[33].as_ptr()),
            ),
        );
        let m2_45 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[28].as_ptr()),
            _mm512_loadu_pd(d[35].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[29].as_ptr()),
                _mm512_loadu_pd(d[34].as_ptr()),
            ),
        );

        let m3_012 = _mm512_mul_pd(_mm512_loadu_pd(d[18].as_ptr()), m2_12);
        let m3_012 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[19].as_ptr()), m2_02, m3_012);
        let m3_012 = _mm512_fmadd_pd(_mm512_loadu_pd(d[20].as_ptr()), m2_01, m3_012);
        let m3_013 = _mm512_mul_pd(_mm512_loadu_pd(d[18].as_ptr()), m2_13);
        let m3_013 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[19].as_ptr()), m2_03, m3_013);
        let m3_013 = _mm512_fmadd_pd(_mm512_loadu_pd(d[21].as_ptr()), m2_01, m3_013);
        let m3_014 = _mm512_mul_pd(_mm512_loadu_pd(d[18].as_ptr()), m2_14);
        let m3_014 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[19].as_ptr()), m2_04, m3_014);
        let m3_014 = _mm512_fmadd_pd(_mm512_loadu_pd(d[22].as_ptr()), m2_01, m3_014);
        let m3_015 = _mm512_mul_pd(_mm512_loadu_pd(d[18].as_ptr()), m2_15);
        let m3_015 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[19].as_ptr()), m2_05, m3_015);
        let m3_015 = _mm512_fmadd_pd(_mm512_loadu_pd(d[23].as_ptr()), m2_01, m3_015);
        let m3_023 = _mm512_mul_pd(_mm512_loadu_pd(d[18].as_ptr()), m2_23);
        let m3_023 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[20].as_ptr()), m2_03, m3_023);
        let m3_023 = _mm512_fmadd_pd(_mm512_loadu_pd(d[21].as_ptr()), m2_02, m3_023);
        let m3_024 = _mm512_mul_pd(_mm512_loadu_pd(d[18].as_ptr()), m2_24);
        let m3_024 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[20].as_ptr()), m2_04, m3_024);
        let m3_024 = _mm512_fmadd_pd(_mm512_loadu_pd(d[22].as_ptr()), m2_02, m3_024);
        let m3_025 = _mm512_mul_pd(_mm512_loadu_pd(d[18].as_ptr()), m2_25);
        let m3_025 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[20].as_ptr()), m2_05, m3_025);
        let m3_025 = _mm512_fmadd_pd(_mm512_loadu_pd(d[23].as_ptr()), m2_02, m3_025);
        let m3_034 = _mm512_mul_pd(_mm512_loadu_pd(d[18].as_ptr()), m2_34);
        let m3_034 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[21].as_ptr()), m2_04, m3_034);
        let m3_034 = _mm512_fmadd_pd(_mm512_loadu_pd(d[22].as_ptr()), m2_03, m3_034);
        let m3_035 = _mm512_mul_pd(_mm512_loadu_pd(d[18].as_ptr()), m2_35);
        let m3_035 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[21].as_ptr()), m2_05, m3_035);
        let m3_035 = _mm512_fmadd_pd(_mm512_loadu_pd(d[23].as_ptr()), m2_03, m3_035);
        let m3_045 = _mm512_mul_pd(_mm512_loadu_pd(d[18].as_ptr()), m2_45);
        let m3_045 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[22].as_ptr()), m2_05, m3_045);
        let m3_045 = _mm512_fmadd_pd(_mm512_loadu_pd(d[23].as_ptr()), m2_04, m3_045);
        let m3_123 = _mm512_mul_pd(_mm512_loadu_pd(d[19].as_ptr()), m2_23);
        let m3_123 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[20].as_ptr()), m2_13, m3_123);
        let m3_123 = _mm512_fmadd_pd(_mm512_loadu_pd(d[21].as_ptr()), m2_12, m3_123);
        let m3_124 = _mm512_mul_pd(_mm512_loadu_pd(d[19].as_ptr()), m2_24);
        let m3_124 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[20].as_ptr()), m2_14, m3_124);
        let m3_124 = _mm512_fmadd_pd(_mm512_loadu_pd(d[22].as_ptr()), m2_12, m3_124);
        let m3_125 = _mm512_mul_pd(_mm512_loadu_pd(d[19].as_ptr()), m2_25);
        let m3_125 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[20].as_ptr()), m2_15, m3_125);
        let m3_125 = _mm512_fmadd_pd(_mm512_loadu_pd(d[23].as_ptr()), m2_12, m3_125);
        let m3_134 = _mm512_mul_pd(_mm512_loadu_pd(d[19].as_ptr()), m2_34);
        let m3_134 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[21].as_ptr()), m2_14, m3_134);
        let m3_134 = _mm512_fmadd_pd(_mm512_loadu_pd(d[22].as_ptr()), m2_13, m3_134);
        let m3_135 = _mm512_mul_pd(_mm512_loadu_pd(d[19].as_ptr()), m2_35);
        let m3_135 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[21].as_ptr()), m2_15, m3_135);
        let m3_135 = _mm512_fmadd_pd(_mm512_loadu_pd(d[23].as_ptr()), m2_13, m3_135);
        let m3_145 = _mm512_mul_pd(_mm512_loadu_pd(d[19].as_ptr()), m2_45);
        let m3_145 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[22].as_ptr()), m2_15, m3_145);
        let m3_145 = _mm512_fmadd_pd(_mm512_loadu_pd(d[23].as_ptr()), m2_14, m3_145);
        let m3_234 = _mm512_mul_pd(_mm512_loadu_pd(d[20].as_ptr()), m2_34);
        let m3_234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[21].as_ptr()), m2_24, m3_234);
        let m3_234 = _mm512_fmadd_pd(_mm512_loadu_pd(d[22].as_ptr()), m2_23, m3_234);
        let m3_235 = _mm512_mul_pd(_mm512_loadu_pd(d[20].as_ptr()), m2_35);
        let m3_235 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[21].as_ptr()), m2_25, m3_235);
        let m3_235 = _mm512_fmadd_pd(_mm512_loadu_pd(d[23].as_ptr()), m2_23, m3_235);
        let m3_245 = _mm512_mul_pd(_mm512_loadu_pd(d[20].as_ptr()), m2_45);
        let m3_245 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[22].as_ptr()), m2_25, m3_245);
        let m3_245 = _mm512_fmadd_pd(_mm512_loadu_pd(d[23].as_ptr()), m2_24, m3_245);
        let m3_345 = _mm512_mul_pd(_mm512_loadu_pd(d[21].as_ptr()), m2_45);
        let m3_345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[22].as_ptr()), m2_35, m3_345);
        let m3_345 = _mm512_fmadd_pd(_mm512_loadu_pd(d[23].as_ptr()), m2_34, m3_345);

        let m4_0123 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m3_123);
        let m4_0123 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m3_023, m4_0123);
        let m4_0123 = _mm512_fmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m3_013, m4_0123);
        let m4_0123 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[15].as_ptr()), m3_012, m4_0123);
        let m4_0124 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m3_124);
        let m4_0124 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m3_024, m4_0124);
        let m4_0124 = _mm512_fmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m3_014, m4_0124);
        let m4_0124 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[16].as_ptr()), m3_012, m4_0124);
        let m4_0125 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m3_125);
        let m4_0125 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m3_025, m4_0125);
        let m4_0125 = _mm512_fmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m3_015, m4_0125);
        let m4_0125 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[17].as_ptr()), m3_012, m4_0125);
        let m4_0134 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m3_134);
        let m4_0134 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m3_034, m4_0134);
        let m4_0134 = _mm512_fmadd_pd(_mm512_loadu_pd(d[15].as_ptr()), m3_014, m4_0134);
        let m4_0134 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[16].as_ptr()), m3_013, m4_0134);
        let m4_0135 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m3_135);
        let m4_0135 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m3_035, m4_0135);
        let m4_0135 = _mm512_fmadd_pd(_mm512_loadu_pd(d[15].as_ptr()), m3_015, m4_0135);
        let m4_0135 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[17].as_ptr()), m3_013, m4_0135);
        let m4_0145 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m3_145);
        let m4_0145 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[13].as_ptr()), m3_045, m4_0145);
        let m4_0145 = _mm512_fmadd_pd(_mm512_loadu_pd(d[16].as_ptr()), m3_015, m4_0145);
        let m4_0145 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[17].as_ptr()), m3_014, m4_0145);
        let m4_0234 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m3_234);
        let m4_0234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m3_034, m4_0234);
        let m4_0234 = _mm512_fmadd_pd(_mm512_loadu_pd(d[15].as_ptr()), m3_024, m4_0234);
        let m4_0234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[16].as_ptr()), m3_023, m4_0234);
        let m4_0235 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m3_235);
        let m4_0235 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m3_035, m4_0235);
        let m4_0235 = _mm512_fmadd_pd(_mm512_loadu_pd(d[15].as_ptr()), m3_025, m4_0235);
        let m4_0235 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[17].as_ptr()), m3_023, m4_0235);
        let m4_0245 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m3_245);
        let m4_0245 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m3_045, m4_0245);
        let m4_0245 = _mm512_fmadd_pd(_mm512_loadu_pd(d[16].as_ptr()), m3_025, m4_0245);
        let m4_0245 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[17].as_ptr()), m3_024, m4_0245);
        let m4_0345 = _mm512_mul_pd(_mm512_loadu_pd(d[12].as_ptr()), m3_345);
        let m4_0345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[15].as_ptr()), m3_045, m4_0345);
        let m4_0345 = _mm512_fmadd_pd(_mm512_loadu_pd(d[16].as_ptr()), m3_035, m4_0345);
        let m4_0345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[17].as_ptr()), m3_034, m4_0345);
        let m4_1234 = _mm512_mul_pd(_mm512_loadu_pd(d[13].as_ptr()), m3_234);
        let m4_1234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m3_134, m4_1234);
        let m4_1234 = _mm512_fmadd_pd(_mm512_loadu_pd(d[15].as_ptr()), m3_124, m4_1234);
        let m4_1234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[16].as_ptr()), m3_123, m4_1234);
        let m4_1235 = _mm512_mul_pd(_mm512_loadu_pd(d[13].as_ptr()), m3_235);
        let m4_1235 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m3_135, m4_1235);
        let m4_1235 = _mm512_fmadd_pd(_mm512_loadu_pd(d[15].as_ptr()), m3_125, m4_1235);
        let m4_1235 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[17].as_ptr()), m3_123, m4_1235);
        let m4_1245 = _mm512_mul_pd(_mm512_loadu_pd(d[13].as_ptr()), m3_245);
        let m4_1245 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[14].as_ptr()), m3_145, m4_1245);
        let m4_1245 = _mm512_fmadd_pd(_mm512_loadu_pd(d[16].as_ptr()), m3_125, m4_1245);
        let m4_1245 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[17].as_ptr()), m3_124, m4_1245);
        let m4_1345 = _mm512_mul_pd(_mm512_loadu_pd(d[13].as_ptr()), m3_345);
        let m4_1345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[15].as_ptr()), m3_145, m4_1345);
        let m4_1345 = _mm512_fmadd_pd(_mm512_loadu_pd(d[16].as_ptr()), m3_135, m4_1345);
        let m4_1345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[17].as_ptr()), m3_134, m4_1345);
        let m4_2345 = _mm512_mul_pd(_mm512_loadu_pd(d[14].as_ptr()), m3_345);
        let m4_2345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[15].as_ptr()), m3_245, m4_2345);
        let m4_2345 = _mm512_fmadd_pd(_mm512_loadu_pd(d[16].as_ptr()), m3_235, m4_2345);
        let m4_2345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[17].as_ptr()), m3_234, m4_2345);

        let m5_01234 = _mm512_mul_pd(_mm512_loadu_pd(d[6].as_ptr()), m4_1234);
        let m5_01234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m4_0234, m5_01234);
        let m5_01234 = _mm512_fmadd_pd(_mm512_loadu_pd(d[8].as_ptr()), m4_0134, m5_01234);
        let m5_01234 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[9].as_ptr()), m4_0124, m5_01234);
        let m5_01234 = _mm512_fmadd_pd(_mm512_loadu_pd(d[10].as_ptr()), m4_0123, m5_01234);
        let m5_01235 = _mm512_mul_pd(_mm512_loadu_pd(d[6].as_ptr()), m4_1235);
        let m5_01235 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m4_0235, m5_01235);
        let m5_01235 = _mm512_fmadd_pd(_mm512_loadu_pd(d[8].as_ptr()), m4_0135, m5_01235);
        let m5_01235 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[9].as_ptr()), m4_0125, m5_01235);
        let m5_01235 = _mm512_fmadd_pd(_mm512_loadu_pd(d[11].as_ptr()), m4_0123, m5_01235);
        let m5_01245 = _mm512_mul_pd(_mm512_loadu_pd(d[6].as_ptr()), m4_1245);
        let m5_01245 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m4_0245, m5_01245);
        let m5_01245 = _mm512_fmadd_pd(_mm512_loadu_pd(d[8].as_ptr()), m4_0145, m5_01245);
        let m5_01245 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[10].as_ptr()), m4_0125, m5_01245);
        let m5_01245 = _mm512_fmadd_pd(_mm512_loadu_pd(d[11].as_ptr()), m4_0124, m5_01245);
        let m5_01345 = _mm512_mul_pd(_mm512_loadu_pd(d[6].as_ptr()), m4_1345);
        let m5_01345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m4_0345, m5_01345);
        let m5_01345 = _mm512_fmadd_pd(_mm512_loadu_pd(d[9].as_ptr()), m4_0145, m5_01345);
        let m5_01345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[10].as_ptr()), m4_0135, m5_01345);
        let m5_01345 = _mm512_fmadd_pd(_mm512_loadu_pd(d[11].as_ptr()), m4_0134, m5_01345);
        let m5_02345 = _mm512_mul_pd(_mm512_loadu_pd(d[6].as_ptr()), m4_2345);
        let m5_02345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[8].as_ptr()), m4_0345, m5_02345);
        let m5_02345 = _mm512_fmadd_pd(_mm512_loadu_pd(d[9].as_ptr()), m4_0245, m5_02345);
        let m5_02345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[10].as_ptr()), m4_0235, m5_02345);
        let m5_02345 = _mm512_fmadd_pd(_mm512_loadu_pd(d[11].as_ptr()), m4_0234, m5_02345);
        let m5_12345 = _mm512_mul_pd(_mm512_loadu_pd(d[7].as_ptr()), m4_2345);
        let m5_12345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[8].as_ptr()), m4_1345, m5_12345);
        let m5_12345 = _mm512_fmadd_pd(_mm512_loadu_pd(d[9].as_ptr()), m4_1245, m5_12345);
        let m5_12345 = _mm512_fnmadd_pd(_mm512_loadu_pd(d[10].as_ptr()), m4_1235, m5_12345);
        let m5_12345 = _mm512_fmadd_pd(_mm512_loadu_pd(d[11].as_ptr()), m4_1234, m5_12345);

        let det_v = _mm512_mul_pd(_mm512_loadu_pd(d[0].as_ptr()), m5_12345);
        let det_v = _mm512_fnmadd_pd(_mm512_loadu_pd(d[1].as_ptr()), m5_02345, det_v);
        let det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[2].as_ptr()), m5_01345, det_v);
        let det_v = _mm512_fnmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m5_01245, det_v);
        let det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[4].as_ptr()), m5_01235, det_v);
        let det_v = _mm512_fnmadd_pd(_mm512_loadu_pd(d[5].as_ptr()), m5_01234, det_v);
        let overlap_v = _mm512_mul_pd(det_v, _mm512_set1_pd(pref));
        _mm512_storeu_pd(overlap.as_mut_ptr(), overlap_v);
    }
}

/// `Evaluate the same-spin overlap directly when m = 0 and L \leq 6:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0).`
/// The row labels are the x-reference particles followed by the w-reference holes, while the column
/// labels are the x-reference holes followed by the w-reference particles. The determinant contains
/// `X^{(0)} on and below the diagonal and Y^{(0)} above the diagonal.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// # Returns
/// - `f64`: Same-spin overlap excluding excitation phases applied outside the Wick evaluation.
#[inline(always)]
pub(crate) fn xw_overlap_m0_direct_f64(
    w: &SameSpinView<'_, f64>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
) -> f64 {
    // Split L into the bra- and ket-reference excitation ranks and form {}^{xw}\tilde S
    // from the separately stored orbital-pairing phase and non-zero singular-value product.
    let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
    let pref = w.phase * w.tilde_s_prod;

    // With no excitation pairs, the determinant is the empty determinant with value one.
    if l == 0 {
        return pref;
    }

    // Read the m_i = 0 fundamental contractions and allocate the row and column labels of
    // \mathbf D_{\mathrm{ov}}(0,\ldots,0).
    let n = w.n();
    let x0 = w.x_slice(0);
    let y0 = w.y_slice(0);
    let mut rows = [0usize; 6];
    let mut cols = [0usize; 6];

    construct_determinant_indices(l_ex, g_ex, w, &mut rows[..l], &mut cols[..l]);

    // Evaluate {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0) using a
    // fixed-rank determinant expression.
    match l {
        // \det\mathbf D_{\mathrm{ov}}(0) = X_{r_0c_0}^{(0)}.
        1 => pref * x0[rows[0] * n + cols[0]],
        2 => {
            // \mathbf D_{\mathrm{ov}} = [[X_{r_0c_0},Y_{r_0c_1}],
            //                            [X_{r_1c_0},X_{r_1c_1}]].
            let d0 = x0[rows[0] * n + cols[0]];
            let d1 = y0[rows[0] * n + cols[1]];
            let d2 = x0[rows[1] * n + cols[0]];
            let d3 = x0[rows[1] * n + cols[1]];

            pref * (d0 * d3 - d1 * d2)
        }
        3 => {
            // Build the rank-three X/Y contraction determinant explicitly.
            let d0 = x0[rows[0] * n + cols[0]];
            let d1 = y0[rows[0] * n + cols[1]];
            let d2 = y0[rows[0] * n + cols[2]];
            let d3 = x0[rows[1] * n + cols[0]];
            let d4 = x0[rows[1] * n + cols[1]];
            let d5 = y0[rows[1] * n + cols[2]];
            let d6 = x0[rows[2] * n + cols[0]];
            let d7 = x0[rows[2] * n + cols[1]];
            let d8 = x0[rows[2] * n + cols[2]];

            pref * (d0 * (d4 * d8 - d5 * d7) - d1 * (d3 * d8 - d5 * d6) + d2 * (d3 * d7 - d4 * d6))
        }
        4 => {
            // Build \mathbf D_{\mathrm{ov}} with X on and below the diagonal and Y above it.
            let mut d = [0.0; 16];

            for i in 0..4 {
                let row = rows[i] * n;

                for j in 0..4 {
                    d[i * 4 + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            pref * det(&d, 4).unwrap_or(0.0)
        }
        5 => {
            // Build the rank-five contraction determinant before evaluating it by LU factorisation.
            let mut lu = [0.0; 25];

            for i in 0..5 {
                let row = rows[i] * n;

                for j in 0..5 {
                    lu[i * 5 + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            pref * det_lu_l5(&mut lu).unwrap_or(0.0)
        }
        6 => {
            // Build the rank-six contraction determinant before evaluating it by LU factorisation.
            let mut lu = [0.0; 36];

            for i in 0..6 {
                let row = rows[i] * n;

                for j in 0..6 {
                    lu[i * 6 + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            pref * det_lu_l6(&mut lu).unwrap_or(0.0)
        }
        _ => {
            // Retain the general determinant path for completeness; the direct API currently calls
            // this function only for L \leq 6.
            let mut d = [0.0; 36];

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

            pref * det(&d[..l * l], l).unwrap_or(0.0)
        }
    }
}

/// Evaluate the same-spin overlap when m = 0:
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0).`
/// `Fixed-rank determinant kernels are used for L = 1,\ldots,6; arbitrary ranks use the general`
/// `determinant routine. For L = 0, the overlap is the reduced reference overlap {}^{xw}\tilde S.`
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
        // Evaluate {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0) with a
        // fixed-rank kernel where available.
        match l {
            // The empty contraction determinant has determinant one.
            0 => w.phase * <T as From<f64>>::from(w.tilde_s_prod),
            1 => xw_overlap_m0_l1(w, scratch),
            2 => xw_overlap_m0_l2(w, scratch),
            3 => xw_overlap_m0_l3(w, scratch),
            4 => xw_overlap_m0_l4(w, scratch),
            5 => xw_overlap_m0_l5(w, scratch),
            6 => xw_overlap_m0_l6(w, scratch),
            _ => {
                // Evaluate the prepared arbitrary-rank contraction determinant directly.
                w.phase
                    * <T as From<f64>>::from(w.tilde_s_prod)
                    * det(scratch.det0.as_slice(), l).unwrap_or(<T as From<f64>>::from(0.0))
            }
        }
    })
}

/// `Evaluate the fixed-rank L = 1 overlap when m = 0:`
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0) = {}^{xw}\tilde S D_{00}^{(0)}.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-one contraction determinant.
/// # Returns
/// - `T`: `Same-spin overlap for L = 1 and m = 0.`
#[inline(always)]
fn xw_overlap_m0_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap_m0_l1, {
        // \det\mathbf D_{\mathrm{ov}}(0) = D_{00}^{(0)}.
        let d = scratch.det0.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * d[0]
    })
}

/// `Evaluate the fixed-rank L = 2 overlap when m = 0:`
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,0).`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-two contraction determinant.
/// # Returns
/// - `T`: `Same-spin overlap for L = 2 and m = 0.`
#[inline(always)]
fn xw_overlap_m0_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap_m0_l2, {
        // \det\mathbf D_{\mathrm{ov}}(0,0) = D_{00}D_{11} - D_{01}D_{10}.
        let d = scratch.det0.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * (d[0] * d[3] - d[1] * d[2])
    })
}

/// `Evaluate the fixed-rank L = 3 overlap when m = 0:`
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,0,0).`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-three contraction determinant.
/// # Returns
/// - `T`: `Same-spin overlap for L = 3 and m = 0.`
#[inline(always)]
fn xw_overlap_m0_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap_m0_l3, {
        // Expand \det\mathbf D_{\mathrm{ov}}(0,0,0) along its first row.
        let d = scratch.det0.as_slice();
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * (d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
                + d[2] * (d[3] * d[7] - d[4] * d[6]))
    })
}

/// `Evaluate the fixed-rank L = 4 overlap when m = 0:`
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,0,0,0).`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-four contraction determinant.
/// # Returns
/// - `T`: `Same-spin overlap for L = 4 and m = 0.`
#[inline(always)]
fn xw_overlap_m0_l4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap_m0_l4, {
        // Evaluate the prepared rank-four contraction determinant by first-row cofactor expansion.
        let det = det(&scratch.det0.as_slice()[..16], 4).unwrap_or(<T as From<f64>>::from(0.0));
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det
    })
}

/// `Evaluate the fixed-rank L = 5 overlap when m = 0:`
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,0,0,0,0).`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-five contraction determinant.
/// # Returns
/// - `T`: `Same-spin overlap for L = 5 and m = 0.`
#[inline(always)]
fn xw_overlap_m0_l5<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    // Copy \mathbf D_{\mathrm{ov}}(0,0,0,0,0) because the LU determinant routine overwrites its input.
    let mut lu = [T::from_real(0.0); 25];
    lu.copy_from_slice(&scratch.det0.as_slice()[..25]);
    let det = det_lu_l5(&mut lu).unwrap_or(<T as From<f64>>::from(0.0));
    w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det
}

/// `Evaluate the fixed-rank L = 6 overlap when m = 0:`
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,0,0,0,0,0).`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-six contraction determinant.
/// # Returns
/// - `T`: `Same-spin overlap for L = 6 and m = 0.`
#[inline(always)]
fn xw_overlap_m0_l6<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    // Copy \mathbf D_{\mathrm{ov}}(0,0,0,0,0,0) because the LU determinant routine overwrites its input.
    let mut lu = [T::from_real(0.0); 36];
    lu.copy_from_slice(&scratch.det0.as_slice()[..36]);
    let det = det_lu_l6(&mut lu).unwrap_or(<T as From<f64>>::from(0.0));
    w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det
}

/// Evaluate the same-spin overlap when m = L. The only allowed distribution is
/// `(m_1,\ldots,m_L) = (1,\ldots,1), so:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,\ldots,1).`
/// `Fixed-rank determinant kernels are used for L = 1,2,3.`
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
        // Evaluate {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,\ldots,1) with a
        // fixed-rank kernel where available.
        match l {
            // This branch is retained for completeness; xw_overlap dispatches L = m = 0 to the m = 0 path.
            0 => w.phase * <T as From<f64>>::from(w.tilde_s_prod),
            1 => xw_overlap_ml_l1(w, scratch),
            2 => xw_overlap_ml_l2(w, scratch),
            3 => xw_overlap_ml_l3(w, scratch),
            _ => {
                // Evaluate the prepared arbitrary-rank all-m_i = 1 determinant directly.
                w.phase
                    * <T as From<f64>>::from(w.tilde_s_prod)
                    * det(scratch.det1.as_slice(), l).unwrap_or(<T as From<f64>>::from(0.0))
            }
        }
    })
}

/// `Evaluate the fixed-rank L = m = 1 overlap:`
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1) = {}^{xw}\tilde S D_{00}^{(1)}.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `scratch`: `Prepared rank-one m_1 = 1 contraction determinant.`
/// # Returns
/// - `T`: `Same-spin overlap for L = m = 1.`
#[inline(always)]
fn xw_overlap_ml_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap_ml_l1, {
        // \det\mathbf D_{\mathrm{ov}}(1) = D_{00}^{(1)}.
        let d = scratch.det1.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * d[0]
    })
}

/// `Evaluate the fixed-rank L = m = 2 overlap:`
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,1).`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `scratch`: `Prepared rank-two all-m_i = 1 contraction determinant.`
/// # Returns
/// - `T`: `Same-spin overlap for L = m = 2.`
#[inline(always)]
fn xw_overlap_ml_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap_ml_l2, {
        // \det\mathbf D_{\mathrm{ov}}(1,1) = D_{00}D_{11} - D_{01}D_{10}.
        let d = scratch.det1.as_slice();
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * (d[0] * d[3] - d[1] * d[2])
    })
}

/// `Evaluate the fixed-rank L = m = 3 overlap:`
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,1,1).`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `scratch`: `Prepared rank-three all-m_i = 1 contraction determinant.`
/// # Returns
/// - `T`: `Same-spin overlap for L = m = 3.`
#[inline(always)]
fn xw_overlap_ml_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_overlap_ml_l3, {
        // Expand \det\mathbf D_{\mathrm{ov}}(1,1,1) along its first row.
        let d = scratch.det1.as_slice();
        w.phase
            * <T as From<f64>>::from(w.tilde_s_prod)
            * (d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
                + d[2] * (d[3] * d[7] - d[4] * d[6]))
    })
}

/// Evaluate the same-spin overlap for 0 < m < L by summing every allowed distribution:
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L=m}}`
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

        // Enumerate the \binom{L}{m} distributions satisfying \sum_i m_i = m and construct
        // each \mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L) by selecting its columns from `det0` or `det1`.
        mix_dets_same(w, l, 0, scratch, |_, scratch| {
            let d = scratch.det_mix.as_slice();

            // Evaluate the determinant using direct low-rank expressions where available.
            let contrib = match l {
                1 => d[0],
                2 => d[0] * d[3] - d[1] * d[2],
                3 => {
                    d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
                        + d[2] * (d[3] * d[7] - d[4] * d[6])
                }
                _ => det(d, l).unwrap_or(<T as From<f64>>::from(0.0)),
            };
            // Add \det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L) to the constrained sum.
            acc += contrib;
        });

        // Apply the orbital-pairing phase to the product of non-zero singular values to recover
        // {}^{xw}\tilde S\sum_{\{m_i\}}\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L).
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
    })
}
