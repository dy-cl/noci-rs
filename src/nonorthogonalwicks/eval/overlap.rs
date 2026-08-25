// nonorthogonalwicks/eval/overlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_fmadd_pd, _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd,
    _mm256_setzero_pd, _mm256_storeu_pd, _mm512_fmadd_pd, _mm512_fnmadd_pd, _mm512_loadu_pd,
    _mm512_mul_pd, _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd,
};

// Crate-root imports.
#[cfg(target_arch = "x86_64")]
use crate::ExcitationSpinCache;
use crate::maths::{det, det_const};
use crate::noci::NOCIScalar;
use crate::time_call;
use crate::{DetState, ExcitationSpin, ReducedOneSpinDetState};

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::mix_dets_same;
use super::prepare::{
    construct_determinant_indices, construct_determinant_indices_const, prepare_same,
};

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
            // Group same-rank overlap requests so each SIMD packet evaluates the same
            // S_tilde det D_ov(0,...,0) formula with lane-local excitation labels.
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
        // Dispatch to the AVX2 compile-time rank that matches this packet's shared L.
        match l {
            1 => xw_overlap_m0_prepared_f64x4_const::<1>(w, x_ex, w_ex, overlap),
            2 => xw_overlap_m0_prepared_f64x4_const::<2>(w, x_ex, w_ex, overlap),
            3 => xw_overlap_m0_prepared_f64x4_const::<3>(w, x_ex, w_ex, overlap),
            4 => xw_overlap_m0_prepared_f64x4_const::<4>(w, x_ex, w_ex, overlap),
            5 => xw_overlap_m0_prepared_f64x4_const::<5>(w, x_ex, w_ex, overlap),
            6 => xw_overlap_m0_prepared_f64x4_const::<6>(w, x_ex, w_ex, overlap),
            _ => unreachable!(),
        }
    }
}

/// Dispatch 4 real fixed-rank `m = 0` overlaps using compile-time `L`.
/// Each SIMD lane evaluates
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`, with the same
/// contraction rank `L` and lane-local excitation labels.
/// # Safety
/// - The caller must ensure CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_prepared_f64x4_const<const L: usize>(
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
        let full = (1usize << L) - 1;
        let mut d_lanes = [[0.0f64; 4]; 36];

        for (lane, (x_data, w_data)) in x_ex.iter().zip(w_ex.iter()).enumerate() {
            let mut rows = [0usize; L];
            let mut cols = [0usize; L];

            construct_determinant_indices_const::<f64, L>(
                x_data.rank,
                &x_data.indices,
                &w_data.indices,
                w,
                &mut rows,
                &mut cols,
            );

            // Pack corresponding D_ov entries from four independent Wick pairs into
            // one AVX2 vector lane group: X fills eta >= z and Y fills eta < z.
            for (eta, &row) in rows.iter().enumerate() {
                let base = eta * L;

                for (z, &col) in cols.iter().enumerate() {
                    let src = row * n + col;
                    d_lanes[base + z][lane] = if eta >= z { x0[src] } else { y0[src] };
                }
            }
        }

        let mut d = [_mm256_setzero_pd(); 36];
        for idx in 0..L * L {
            d[idx] = _mm256_loadu_pd(d_lanes[idx].as_ptr());
        }

        let mut minors = [_mm256_setzero_pd(); 64];
        for c in 0..L {
            minors[1usize << c] = d[(L - 1) * L + c];
        }

        // Compile-time L selects one monomorphised determinant expansion. The subset minors
        // are the same Laplace cofactors as the old fixed-rank kernels, but generated by
        // the single const body and evaluated as packed SIMD values.
        let mut size = 2usize;
        while size <= L {
            let row = L - size;
            let mut next = [_mm256_setzero_pd(); 64];
            let mut mask = full;

            loop {
                if mask.count_ones() as usize == size {
                    let mut acc = _mm256_setzero_pd();
                    let mut pos = 0usize;

                    for c in 0..L {
                        let bit = 1usize << c;

                        if (mask & bit) != 0 {
                            let term = minors[mask ^ bit];
                            acc = if (pos & 1) == 0 {
                                _mm256_fmadd_pd(d[row * L + c], term, acc)
                            } else {
                                _mm256_fnmadd_pd(d[row * L + c], term, acc)
                            };
                            pos += 1;
                        }
                    }

                    next[mask] = acc;
                }

                if mask == 0 {
                    break;
                }
                mask = (mask - 1) & full;
            }

            minors = next;
            size += 1;
        }

        let overlap_v = _mm256_mul_pd(minors[full], _mm256_set1_pd(pref));
        _mm256_storeu_pd(overlap.as_mut_ptr(), overlap_v);
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
        // Dispatch to the AVX-512 compile-time rank that matches this packet's shared L.
        match l {
            1 => xw_overlap_m0_prepared_f64x8_const::<1>(w, x_ex, w_ex, overlap),
            2 => xw_overlap_m0_prepared_f64x8_const::<2>(w, x_ex, w_ex, overlap),
            3 => xw_overlap_m0_prepared_f64x8_const::<3>(w, x_ex, w_ex, overlap),
            4 => xw_overlap_m0_prepared_f64x8_const::<4>(w, x_ex, w_ex, overlap),
            5 => xw_overlap_m0_prepared_f64x8_const::<5>(w, x_ex, w_ex, overlap),
            6 => xw_overlap_m0_prepared_f64x8_const::<6>(w, x_ex, w_ex, overlap),
            _ => unreachable!(),
        }
    }
}

/// Dispatch 8 real fixed-rank `m = 0` overlaps using compile-time `L`.
/// Each SIMD lane evaluates
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`, with the same
/// contraction rank `L` and lane-local excitation labels.
/// # Safety
/// - The caller must ensure CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_prepared_f64x8_const<const L: usize>(
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
        let full = (1usize << L) - 1;
        let mut d_lanes = [[0.0f64; 8]; 36];

        for (lane, (x_data, w_data)) in x_ex.iter().zip(w_ex.iter()).enumerate() {
            let mut rows = [0usize; L];
            let mut cols = [0usize; L];

            construct_determinant_indices_const::<f64, L>(
                x_data.rank,
                &x_data.indices,
                &w_data.indices,
                w,
                &mut rows,
                &mut cols,
            );

            // Pack corresponding D_ov entries from eight independent Wick pairs into
            // one AVX-512 vector lane group: X fills eta >= z and Y fills eta < z.
            for (eta, &row) in rows.iter().enumerate() {
                let base = eta * L;

                for (z, &col) in cols.iter().enumerate() {
                    let src = row * n + col;
                    d_lanes[base + z][lane] = if eta >= z { x0[src] } else { y0[src] };
                }
            }
        }

        let mut d = [_mm512_setzero_pd(); 36];
        for idx in 0..L * L {
            d[idx] = _mm512_loadu_pd(d_lanes[idx].as_ptr());
        }

        let mut minors = [_mm512_setzero_pd(); 64];
        for c in 0..L {
            minors[1usize << c] = d[(L - 1) * L + c];
        }

        // Compile-time L selects one monomorphised determinant expansion. The subset minors
        // are the same Laplace cofactors as the old fixed-rank kernels, but generated by
        // the single const body and evaluated as packed SIMD values.
        let mut size = 2usize;
        while size <= L {
            let row = L - size;
            let mut next = [_mm512_setzero_pd(); 64];
            let mut mask = full;

            loop {
                if mask.count_ones() as usize == size {
                    let mut acc = _mm512_setzero_pd();
                    let mut pos = 0usize;

                    for c in 0..L {
                        let bit = 1usize << c;

                        if (mask & bit) != 0 {
                            let term = minors[mask ^ bit];
                            acc = if (pos & 1) == 0 {
                                _mm512_fmadd_pd(d[row * L + c], term, acc)
                            } else {
                                _mm512_fnmadd_pd(d[row * L + c], term, acc)
                            };
                            pos += 1;
                        }
                    }

                    next[mask] = acc;
                }

                if mask == 0 {
                    break;
                }
                mask = (mask - 1) & full;
            }

            minors = next;
            size += 1;
        }

        let overlap_v = _mm512_mul_pd(minors[full], _mm512_set1_pd(pref));
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

    match l {
        1 => xw_overlap_m0_direct_f64_const::<1>(pref, n, x0, y0, &rows, &cols),
        2 => xw_overlap_m0_direct_f64_const::<2>(pref, n, x0, y0, &rows, &cols),
        3 => xw_overlap_m0_direct_f64_const::<3>(pref, n, x0, y0, &rows, &cols),
        4 => xw_overlap_m0_direct_f64_const::<4>(pref, n, x0, y0, &rows, &cols),
        5 => xw_overlap_m0_direct_f64_const::<5>(pref, n, x0, y0, &rows, &cols),
        6 => xw_overlap_m0_direct_f64_const::<6>(pref, n, x0, y0, &rows, &cols),
        _ => unreachable!(),
    }
}

/// Evaluate one real direct all-`m_i = 0` overlap with compile-time contraction rank:
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// The direct path constructs
/// `D_{ij}^{(0)} = X_{r_i c_j}^{(0)}` for `i >= j` and
/// `D_{ij}^{(0)} = Y_{r_i c_j}^{(0)}` for `i < j` from the supplied row and column
/// orbital labels, then evaluates the fixed-rank determinant.
/// # Arguments:
/// - `pref`: Reduced reference-overlap prefactor `{}^{xw}\tilde S`.
/// - `n`: Fundamental-contraction matrix dimension.
/// - `x0`: Lower-triangle and diagonal contraction source.
/// - `y0`: Upper-triangle contraction source.
/// - `rows`: Row orbital labels `r_i`.
/// - `cols`: Column orbital labels `c_j`.
/// # Returns:
/// - `f64`: Same-spin overlap contribution.
#[inline(always)]
fn xw_overlap_m0_direct_f64_const<const L: usize>(
    pref: f64,
    n: usize,
    x0: &[f64],
    y0: &[f64],
    rows: &[usize; 6],
    cols: &[usize; 6],
) -> f64 {
    let mut d = [0.0; 36];

    // Build D_ov(0,...,0) from the fixed contraction labels, then evaluate
    // the direct overlap factor S_tilde det D_ov.
    for i in 0..L {
        let row = rows[i] * n;

        for j in 0..L {
            d[i * L + j] = if i >= j {
                x0[row + cols[j]]
            } else {
                y0[row + cols[j]]
            };
        }
    }

    pref * det_const::<f64, L>(&d[..L * L]).unwrap_or(0.0)
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
            0 => xw_overlap_m0_const::<T, 0>(w, scratch),
            1 => xw_overlap_m0_const::<T, 1>(w, scratch),
            2 => xw_overlap_m0_const::<T, 2>(w, scratch),
            3 => xw_overlap_m0_const::<T, 3>(w, scratch),
            4 => xw_overlap_m0_const::<T, 4>(w, scratch),
            5 => xw_overlap_m0_const::<T, 5>(w, scratch),
            6 => xw_overlap_m0_const::<T, 6>(w, scratch),
            _ => {
                // Evaluate the prepared arbitrary-rank contraction determinant directly.
                w.phase
                    * <T as From<f64>>::from(w.tilde_s_prod)
                    * det(scratch.det0.as_slice(), l).unwrap_or(<T as From<f64>>::from(0.0))
            }
        }
    })
}

/// Evaluate the fixed-rank all-`m_i = 0` overlap with compile-time contraction rank:
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// For `L = 0`, this returns the empty determinant contribution
/// `{}^{xw}\tilde S`. For `L = 1,\ldots,6`, the determinant helper uses the
/// same fixed-rank formula previously implemented by the rank-specific kernels.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared `\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// # Returns
/// - `T`: Same-spin overlap for fixed `L` and `m = 0`.
#[inline(always)]
fn xw_overlap_m0_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_overlap_m0_const,
        {
            let d = scratch.det0.as_slice();
            // For `m = 0`, the GNME zero-overlap distribution sum contains only the all-zero
            // assignment, giving {}^{xw}\tilde S det D_ov(0,...,0).
            w.phase
                * <T as From<f64>>::from(w.tilde_s_prod)
                * det_const::<T, L>(&d[..L * L]).unwrap_or(<T as From<f64>>::from(0.0))
        }
    )
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
            0 => xw_overlap_ml_const::<T, 0>(w, scratch),
            1 => xw_overlap_ml_const::<T, 1>(w, scratch),
            2 => xw_overlap_ml_const::<T, 2>(w, scratch),
            3 => xw_overlap_ml_const::<T, 3>(w, scratch),
            _ => {
                // Evaluate the prepared arbitrary-rank all-m_i = 1 determinant directly.
                w.phase
                    * <T as From<f64>>::from(w.tilde_s_prod)
                    * det(scratch.det1.as_slice(), l).unwrap_or(<T as From<f64>>::from(0.0))
            }
        }
    })
}

/// Evaluate the fixed-rank all-`m_i = 1` overlap with compile-time contraction rank:
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,\ldots,1)`.
/// For this path `L = m`; every column of `\mathbf D_{\mathrm{ov}}` is selected from
/// the prepared all-`m_i = 1` determinant.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `scratch`: Prepared `\mathbf D_{\mathrm{ov}}(1,\ldots,1)`.
/// # Returns
/// - `T`: Same-spin overlap for fixed `L = m`.
#[inline(always)]
fn xw_overlap_ml_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_overlap_ml_const,
        {
            let d = scratch.det1.as_slice();
            // For `m = L`, every contraction receives one zero-overlap index, so the only
            // surviving term is {}^{xw}\tilde S det D_ov(1,...,1).
            w.phase
                * <T as From<f64>>::from(w.tilde_s_prod)
                * det_const::<T, L>(&d[..L * L]).unwrap_or(<T as From<f64>>::from(0.0))
        }
    )
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
        match l {
            1 => xw_overlap_gen_const::<T, 1>(w, scratch, &mut acc),
            2 => xw_overlap_gen_const::<T, 2>(w, scratch, &mut acc),
            3 => xw_overlap_gen_const::<T, 3>(w, scratch, &mut acc),
            _ => {
                mix_dets_same(w, l, 0, scratch, |_, scratch| {
                    let d = scratch.det_mix.as_slice();
                    acc += det(d, l).unwrap_or(<T as From<f64>>::from(0.0));
                });
            }
        }

        // Apply the orbital-pairing phase to the product of non-zero singular values to recover
        // {}^{xw}\tilde S\sum_{\{m_i\}}\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L).
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
    })
}

/// Sum fixed-rank mixed-distribution determinants for `0 < m < L`:
/// `\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L=m}}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L)`, with `m_i \in \{0,1\}`.
/// Each distribution is constructed by selecting each column from the prepared
/// all-`m_i = 0` or all-`m_i = 1` determinant before evaluating the fixed-rank
/// determinant.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `scratch`: Prepared determinant buffers `\mathbf D_{\mathrm{ov}}(0,\ldots,0)`,
///   `\mathbf D_{\mathrm{ov}}(1,\ldots,1)`, and mixed storage.
/// - `acc`: Accumulated determinant sum.
/// # Returns:
/// - `()`: Adds every constrained-distribution determinant to `acc`.
#[inline(always)]
fn xw_overlap_gen_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    acc: &mut T,
) {
    mix_dets_same(w, L, 0, scratch, |_, scratch| {
        let d = scratch.det_mix.as_slice();
        // Each mixed determinant is one term in the constrained GNME sum over
        // m_i in {0,1} with \sum_i m_i = m.
        *acc += det_const::<T, L>(&d[..L * L]).unwrap_or(<T as From<f64>>::from(0.0));
    });
}
