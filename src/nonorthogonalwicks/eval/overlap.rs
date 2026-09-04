// nonorthogonalwicks/eval/overlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

// External crate imports.
use num_complex::Complex64;

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
use super::dispatch::dispatch_overlap_ranks;
use super::helpers::mix_dets_same;
use super::prepare::{construct_determinant_indices, prepare_same};
#[cfg(target_arch = "x86_64")]
use super::simd::{C64x4, C64x8, F64x4, F64x8};

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
/// For `m = 0` and `L \leq 6`, the direct overlap-only path evaluates the single determinant
/// without preparing reusable Hamiltonian scratch data. Other cases use `prepare_same`
/// followed by the general overlap evaluator.
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
    // Determine the contraction-determinant dimension `L = L_x + L_w`.
    let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;

    // No distribution satisfying `\sum_i m_i = m` exists when `m > L`.
    if w.m > l {
        return <T as From<f64>>::from(0.0);
    }

    // For `m = 0` and `L \leq 6`, construct and evaluate
    // `\mathbf D_{\mathrm{ov}}(0,\ldots,0)`
    // directly without populating the reusable scratch representation.
    if w.m == 0 && l <= 6 {
        return xw_overlap_m0_direct(w, l_ex, g_ex);
    }

    // Prepare the all-`m_i = 0` and, where required, all-`m_i = 1` contraction determinants
    // before applying the standard overlap evaluation.
    prepare_same(w, l_ex, g_ex, scratch);
    xw_overlap(w, l_ex, g_ex, scratch)
}

/// Inputs and outputs for one row of same-spin overlap factors.
pub(crate) struct SameSpinOverlapBatch<'a, T: NOCIScalar> {
    /// Determinant basis used only by generic fallback evaluation.
    pub(crate) basis: &'a [DetState<T>],
    /// Reduced target spin representative shared by the row.
    pub(crate) target: ReducedOneSpinDetState,
    /// Reduced source spin representatives in output-column order.
    pub(crate) sources: &'a [ReducedOneSpinDetState],
    /// Whether the target belongs to the left reference in `w`.
    pub(crate) target_left: bool,
    /// Whether to evaluate alpha-spin rather than beta-spin overlap factors.
    pub(crate) alpha: bool,
    /// Output same-spin overlap factors in source-representative order.
    pub(crate) out: &'a mut [T],
}

/// Evaluate one row of same-spin overlaps for one ordered reference pair.
/// Every output is
/// `p_x p_w {}^{xw}\tilde S\sum_{m_1+\cdots+m_L = m}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L)`.
/// The target representative is paired with every source representative. Requests with `m = 0`,
/// scalar type `f64` or `Complex64`, `L = 1,\ldots,6`, and individual excitation ranks at most four
/// are grouped by `(RX,RW,L)` and evaluated with the widest available SIMD kernel. Incomplete
/// groups use the scalar overlap-only path. Other requests use the generic overlap-only evaluator.
/// Excitation phases are applied here so each output is the complete alpha- or beta-spin factor.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `batch`: One row of same-spin overlap-factor work.
/// - `scratch`: Reusable Wick workspace for scalar fallback evaluation.
/// # Returns
/// - `()`: Writes one complete same-spin overlap-factor row into `batch.out`.
pub(crate) fn xw_overlap_prepared_batched<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    batch: SameSpinOverlapBatch<'_, T>,
    scratch: &mut WickScratch<T>,
) {
    let SameSpinOverlapBatch {
        basis,
        target,
        sources,
        target_left,
        alpha,
        out,
    } = batch;
    #[cfg(target_arch = "x86_64")]
    if w.m == 0 && TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            // SAFETY: The explicit `TypeId` check proves `T = f64`, so the output row has the
            // same layout as `f64` for the duration of the SIMD helper call.
            let out_f64 = std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len());
            if try_xw_overlap_prepared_f64_simd(
                w,
                basis,
                (target, sources),
                (target_left, alpha),
                scratch,
                out_f64,
            ) {
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    if w.m == 0 && TypeId::of::<T>() == TypeId::of::<Complex64>() {
        unsafe {
            // SAFETY: The explicit `TypeId` check proves `T = Complex64`, so the output row has
            // the same layout as `Complex64` for the duration of the SIMD helper call.
            let out_c64 =
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<Complex64>(), out.len());
            if try_xw_overlap_prepared_c64_simd(
                w,
                basis,
                (target, sources),
                (target_left, alpha),
                scratch,
                out_c64,
            ) {
                return;
            }
        }
    }

    xw_overlap_prepared_scalar_row(
        w,
        basis,
        (target, sources),
        (target_left, alpha),
        scratch,
        out,
    );
}

/// Evaluate one same-spin overlap row through the scalar overlap-only path.
/// Every output is
/// `p_x p_w {}^{xw}\tilde S\sum_{m_1+\cdots+m_L = m}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `basis`: Determinant basis containing full excitation masks.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `out`: Same-spin overlap output row.
/// # Returns
/// - `()`: Writes one complete same-spin overlap-factor row.
fn xw_overlap_prepared_scalar_row<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    out: &mut [T],
) {
    let (target, sources) = reps;
    let (target_left, alpha) = flags;
    for (col, source) in sources.iter().enumerate() {
        out[col] = xw_overlap_prepared_scalar_value(
            w,
            basis,
            (target, *source),
            (target_left, alpha),
            scratch,
        );
    }
}

/// Evaluate one complete same-spin overlap factor through the scalar overlap-only path.
/// The returned factor is
/// `p_x p_w {}^{xw}\tilde S\sum_{m_1+\cdots+m_L = m}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `basis`: Determinant basis containing full excitation masks.
/// - `reps`: Target and source spin representatives.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// # Returns
/// - `T`: Same-spin overlap including both excitation phases.
#[inline(always)]
fn xw_overlap_prepared_scalar_value<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, ReducedOneSpinDetState),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
) -> T {
    let (target, source) = reps;
    let (target_left, alpha) = flags;
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
    T::from_real(target.phase * source.phase) * xw_overlap_prepared(w, x_ex, w_ex, scratch)
}

/// Try to evaluate one real same-spin overlap row with fixed-rank SIMD kernels.
/// Each supported output is
/// `p_x p_w {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `out`: Real overlap output row.
/// # Returns
/// - `bool`: Whether an available SIMD path evaluated the complete row.
/// # Safety
/// - The caller must prove `T = f64` and `w.m = 0`.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_overlap_prepared_f64_simd<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    out: &mut [f64],
) -> bool {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            xw_overlap_prepared_f64x8_row(w, basis, reps, flags, scratch, out);
        }
        return true;
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            xw_overlap_prepared_f64x4_row(w, basis, reps, flags, scratch, out);
        }
        return true;
    }
    false
}

/// Try to evaluate one complex same-spin overlap row with fixed-rank SIMD kernels.
/// Each supported output is
/// `p_x p_w {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `out`: Complex overlap output row.
/// # Returns
/// - `bool`: Whether an available SIMD path evaluated the complete row.
/// # Safety
/// - The caller must prove `T = Complex64` and `w.m = 0`.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_overlap_prepared_c64_simd<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    out: &mut [Complex64],
) -> bool {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            xw_overlap_prepared_c64x8_row(w, basis, reps, flags, scratch, out);
        }
        return true;
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            xw_overlap_prepared_c64x4_row(w, basis, reps, flags, scratch, out);
        }
        return true;
    }
    false
}

/// Evaluate one real same-spin overlap row with four-lane AVX2/FMA packets.
/// Each supported lane computes
/// `p_x p_w {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`; incomplete packets
/// and ranks outside the fixed-rank dispatch table use the scalar overlap-only path.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0` and `T = f64`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `out`: Real overlap output row.
/// # Returns
/// - `()`: Writes one complete same-spin overlap-factor row into `out`.
/// # Safety
/// - The caller must prove `T = f64`, `w.m = 0`, and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_prepared_f64x4_row<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    out: &mut [f64],
) {
    let (target, sources) = reps;
    let (target_left, _) = flags;
    let target_cache = target.excitation_cache;
    let target_rank = usize::from(target_cache.rank);
    let mut bins = [[ExcitationSpinCache::default(); 4]; 5];
    let mut phases = [[1.0f64; 4]; 5];
    let mut outputs = [[0usize; 4]; 5];
    let mut counts = [0usize; 5];

    for (col, source) in sources.iter().enumerate() {
        let source_cache = source.excitation_cache;
        let source_rank = usize::from(source_cache.rank);
        let ranks = if target_left {
            (target_rank, source_rank)
        } else {
            (source_rank, target_rank)
        };

        if target_rank <= 4 && source_rank <= 4 && (1..=6).contains(&(target_rank + source_rank)) {
            let count = counts[source_rank];
            bins[source_rank][count] = source_cache;
            phases[source_rank][count] = source.phase;
            outputs[source_rank][count] = col;
            counts[source_rank] += 1;

            if counts[source_rank] == 4 {
                let source_batch = bins[source_rank];
                let mut overlap = [0.0f64; 4];
                unsafe {
                    xw_overlap_m0_prepared_f64x4(
                        w,
                        ranks,
                        target_left,
                        &target_cache,
                        &source_batch,
                        &mut overlap,
                    );
                }
                for lane in 0..4 {
                    out[outputs[source_rank][lane]] =
                        target.phase * phases[source_rank][lane] * overlap[lane];
                }
                counts[source_rank] = 0;
            }
        } else {
            let value =
                xw_overlap_prepared_scalar_value(w, basis, (target, *source), flags, scratch);
            out[col] = unsafe { *std::ptr::from_ref(&value).cast::<f64>() };
        }
    }

    for source_rank in 0..5 {
        for &col in &outputs[source_rank][..counts[source_rank]] {
            let value =
                xw_overlap_prepared_scalar_value(w, basis, (target, sources[col]), flags, scratch);
            out[col] = unsafe { *std::ptr::from_ref(&value).cast::<f64>() };
        }
    }
}

/// Evaluate one real same-spin overlap row with eight-lane AVX-512 packets.
/// Each supported lane computes
/// `p_x p_w {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`; incomplete packets
/// and ranks outside the fixed-rank dispatch table use the scalar overlap-only path.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0` and `T = f64`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `out`: Real overlap output row.
/// # Returns
/// - `()`: Writes one complete same-spin overlap-factor row into `out`.
/// # Safety
/// - The caller must prove `T = f64`, `w.m = 0`, and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_prepared_f64x8_row<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    out: &mut [f64],
) {
    let (target, sources) = reps;
    let (target_left, _) = flags;
    let target_cache = target.excitation_cache;
    let target_rank = usize::from(target_cache.rank);
    let mut bins = [[ExcitationSpinCache::default(); 8]; 5];
    let mut phases = [[1.0f64; 8]; 5];
    let mut outputs = [[0usize; 8]; 5];
    let mut counts = [0usize; 5];

    for (col, source) in sources.iter().enumerate() {
        let source_cache = source.excitation_cache;
        let source_rank = usize::from(source_cache.rank);
        let ranks = if target_left {
            (target_rank, source_rank)
        } else {
            (source_rank, target_rank)
        };

        if target_rank <= 4 && source_rank <= 4 && (1..=6).contains(&(target_rank + source_rank)) {
            let count = counts[source_rank];
            bins[source_rank][count] = source_cache;
            phases[source_rank][count] = source.phase;
            outputs[source_rank][count] = col;
            counts[source_rank] += 1;

            if counts[source_rank] == 8 {
                let source_batch = bins[source_rank];
                let mut overlap = [0.0f64; 8];
                unsafe {
                    xw_overlap_m0_prepared_f64x8(
                        w,
                        ranks,
                        target_left,
                        &target_cache,
                        &source_batch,
                        &mut overlap,
                    );
                }
                for lane in 0..8 {
                    out[outputs[source_rank][lane]] =
                        target.phase * phases[source_rank][lane] * overlap[lane];
                }
                counts[source_rank] = 0;
            }
        } else {
            let value =
                xw_overlap_prepared_scalar_value(w, basis, (target, *source), flags, scratch);
            out[col] = unsafe { *std::ptr::from_ref(&value).cast::<f64>() };
        }
    }

    for source_rank in 0..5 {
        for &col in &outputs[source_rank][..counts[source_rank]] {
            let value =
                xw_overlap_prepared_scalar_value(w, basis, (target, sources[col]), flags, scratch);
            out[col] = unsafe { *std::ptr::from_ref(&value).cast::<f64>() };
        }
    }
}

/// Evaluate one complex same-spin overlap row with four-lane AVX2/FMA packets.
/// Each supported lane computes
/// `p_x p_w {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`; incomplete packets
/// and ranks outside the fixed-rank dispatch table use the scalar overlap-only path.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0` and `T = Complex64`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `out`: Complex overlap output row.
/// # Returns
/// - `()`: Writes one complete same-spin overlap-factor row into `out`.
/// # Safety
/// - The caller must prove `T = Complex64`, `w.m = 0`, and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_prepared_c64x4_row<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    out: &mut [Complex64],
) {
    let (target, sources) = reps;
    let (target_left, _) = flags;
    let target_cache = target.excitation_cache;
    let target_rank = usize::from(target_cache.rank);
    let mut bins = [[ExcitationSpinCache::default(); 4]; 5];
    let mut phases = [[1.0f64; 4]; 5];
    let mut outputs = [[0usize; 4]; 5];
    let mut counts = [0usize; 5];

    for (col, source) in sources.iter().enumerate() {
        let source_cache = source.excitation_cache;
        let source_rank = usize::from(source_cache.rank);
        let ranks = if target_left {
            (target_rank, source_rank)
        } else {
            (source_rank, target_rank)
        };

        if target_rank <= 4 && source_rank <= 4 && (1..=6).contains(&(target_rank + source_rank)) {
            let count = counts[source_rank];
            bins[source_rank][count] = source_cache;
            phases[source_rank][count] = source.phase;
            outputs[source_rank][count] = col;
            counts[source_rank] += 1;

            if counts[source_rank] == 4 {
                let source_batch = bins[source_rank];
                let mut overlap = [Complex64::new(0.0, 0.0); 4];
                unsafe {
                    xw_overlap_m0_prepared_c64x4(
                        w,
                        ranks,
                        target_left,
                        &target_cache,
                        &source_batch,
                        &mut overlap,
                    );
                }
                for lane in 0..4 {
                    out[outputs[source_rank][lane]] =
                        overlap[lane] * (target.phase * phases[source_rank][lane]);
                }
                counts[source_rank] = 0;
            }
        } else {
            let value =
                xw_overlap_prepared_scalar_value(w, basis, (target, *source), flags, scratch);
            out[col] = unsafe { *std::ptr::from_ref(&value).cast::<Complex64>() };
        }
    }

    for source_rank in 0..5 {
        for &col in &outputs[source_rank][..counts[source_rank]] {
            let value =
                xw_overlap_prepared_scalar_value(w, basis, (target, sources[col]), flags, scratch);
            out[col] = unsafe { *std::ptr::from_ref(&value).cast::<Complex64>() };
        }
    }
}

/// Evaluate one complex same-spin overlap row with eight-lane AVX-512 packets.
/// Each supported lane computes
/// `p_x p_w {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`; incomplete packets
/// and ranks outside the fixed-rank dispatch table use the scalar overlap-only path.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0` and `T = Complex64`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `out`: Complex overlap output row.
/// # Returns
/// - `()`: Writes one complete same-spin overlap-factor row into `out`.
/// # Safety
/// - The caller must prove `T = Complex64`, `w.m = 0`, and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_prepared_c64x8_row<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    out: &mut [Complex64],
) {
    let (target, sources) = reps;
    let (target_left, _) = flags;
    let target_cache = target.excitation_cache;
    let target_rank = usize::from(target_cache.rank);
    let mut bins = [[ExcitationSpinCache::default(); 8]; 5];
    let mut phases = [[1.0f64; 8]; 5];
    let mut outputs = [[0usize; 8]; 5];
    let mut counts = [0usize; 5];

    for (col, source) in sources.iter().enumerate() {
        let source_cache = source.excitation_cache;
        let source_rank = usize::from(source_cache.rank);
        let ranks = if target_left {
            (target_rank, source_rank)
        } else {
            (source_rank, target_rank)
        };

        if target_rank <= 4 && source_rank <= 4 && (1..=6).contains(&(target_rank + source_rank)) {
            let count = counts[source_rank];
            bins[source_rank][count] = source_cache;
            phases[source_rank][count] = source.phase;
            outputs[source_rank][count] = col;
            counts[source_rank] += 1;

            if counts[source_rank] == 8 {
                let source_batch = bins[source_rank];
                let mut overlap = [Complex64::new(0.0, 0.0); 8];
                unsafe {
                    xw_overlap_m0_prepared_c64x8(
                        w,
                        ranks,
                        target_left,
                        &target_cache,
                        &source_batch,
                        &mut overlap,
                    );
                }
                for lane in 0..8 {
                    out[outputs[source_rank][lane]] =
                        overlap[lane] * (target.phase * phases[source_rank][lane]);
                }
                counts[source_rank] = 0;
            }
        } else {
            let value =
                xw_overlap_prepared_scalar_value(w, basis, (target, *source), flags, scratch);
            out[col] = unsafe { *std::ptr::from_ref(&value).cast::<Complex64>() };
        }
    }

    for source_rank in 0..5 {
        for &col in &outputs[source_rank][..counts[source_rank]] {
            let value =
                xw_overlap_prepared_scalar_value(w, basis, (target, sources[col]), flags, scratch);
            out[col] = unsafe { *std::ptr::from_ref(&value).cast::<Complex64>() };
        }
    }
}

/// Dispatch 4 real `m = 0` overlaps to the fixed-rank AVX2/FMA kernel.
/// Each output is `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `ranks`: Bra- and ket-reference excitation ranks `(RX,RW)`.
/// - `x_fixed`: Whether the fixed target excitation belongs to the x reference.
/// - `fixed`: Fixed target excitation cache.
/// - `varying`: 4 source excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 4 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, and that `ranks` agrees with
///   the valid excitation labels stored in `fixed` and `varying`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ranks: (usize, usize),
    x_fixed: bool,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    if x_fixed {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L| unsafe {
                xw_overlap_m0_prepared_f64x4_const::<_, RX, RW, L, true>(w, fixed, varying, overlap)
            },
            unreachable!(),
        )
    } else {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L| unsafe {
                xw_overlap_m0_prepared_f64x4_const::<_, RX, RW, L, false>(
                    w, fixed, varying, overlap,
                )
            },
            unreachable!(),
        )
    }
}

/// Dispatch 4 real fixed-rank `m = 0` overlaps using compile-time `(RX,RW,L)`.
/// Each SIMD lane evaluates
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`, with the same
/// contraction rank `L` and lane-local excitation labels.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0` and `T = f64`.
/// - `fixed`: Excitation cache shared by all four lanes; `XFIX` selects its reference side.
/// - `varying`: Four lane-local excitation caches belonging to the other reference side.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes four same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid cached labels, and
///   compile-time ranks satisfying `RX + RW = L` with `1 <= L <= 6` and matching the cache ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_prepared_f64x4_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, T>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    let n = w.n();
    // `x0` and `y0` store the `X^{(0)}` and `Y^{(0)}` fundamental contractions.
    let x0 = w.x_slice(0).as_ptr().cast::<f64>();
    let y0 = w.y_slice(0).as_ptr().cast::<f64>();
    // `pref = p\,{}^{xw}\tilde S` contains the reference-pair phase and reduced overlap.
    let phase = unsafe { *std::ptr::from_ref(&w.phase).cast::<f64>() };
    let pref = phase * w.tilde_s_prod;
    // `full = \{0,\ldots,L-1\}` is the complete contraction-column subset.
    let full = (1usize << L) - 1;
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    // Select the lane-local bra excitation `{}^x\Psi_{i\cdots}^{a\cdots}`.
    let x_data = |lane: usize| -> &ExcitationSpinCache {
        if XFIX {
            fixed
        } else {
            unsafe { varying.get_unchecked(lane) }
        }
    };
    // Select the lane-local ket excitation `{}^w\Psi_{j\cdots}^{b\cdots}`.
    let w_data = |lane: usize| -> &ExcitationSpinCache {
        if XFIX {
            unsafe { varying.get_unchecked(lane) }
        } else {
            fixed
        }
    };
    // Rows are ordered as `r_\eta \in V_x\cup O_w`, namely x-particles followed by w-holes.
    let row_index = |eta: usize, lane: usize| -> usize {
        if eta < RX {
            usize::from(unsafe { *x_data(lane).indices.get_unchecked(4 + eta) }) - nocc
        } else {
            nvirt + usize::from(unsafe { *w_data(lane).indices.get_unchecked(eta - RX) })
        }
    };
    // Columns are ordered as `c_z \in O_x\cup V_w`, namely x-holes followed by w-particles.
    let col_index = |z: usize, lane: usize| -> usize {
        if z < RX {
            usize::from(unsafe { *x_data(lane).indices.get_unchecked(z) })
        } else {
            usize::from(unsafe { *w_data(lane).indices.get_unchecked(4 + z - RX) })
        }
    };
    // `D_{\eta z} = X^{(0)}_{r_\eta c_z}` for `\eta \geq z`, otherwise
    // `D_{\eta z} = Y^{(0)}_{r_\eta c_z}`.
    let load_d = |eta: usize, z: usize| -> F64x4 {
        let matrix = if eta >= z { x0 } else { y0 };
        let row_fixed = if eta < RX { XFIX } else { !XFIX };
        let col_fixed = if z < RX { XFIX } else { !XFIX };
        // A lane-invariant contraction is broadcast; otherwise gather the four `D_{\eta z}` values.
        if row_fixed && col_fixed {
            let src = row_index(eta, 0) * n + col_index(z, 0);
            F64x4::splat(unsafe { *matrix.add(src) })
        } else {
            F64x4::from_values(
                unsafe { *matrix.add(row_index(eta, 0) * n + col_index(z, 0)) },
                unsafe { *matrix.add(row_index(eta, 1) * n + col_index(z, 1)) },
                unsafe { *matrix.add(row_index(eta, 2) * n + col_index(z, 2)) },
                unsafe { *matrix.add(row_index(eta, 3) * n + col_index(z, 3)) },
            )
        }
    };
    // Construct the packed overlap contraction matrices `\mathbf D_{\mathrm{ov}}`.
    let mut d = [F64x4::zero(); 36];
    for eta in 0..L {
        for z in 0..L {
            d[eta * L + z] = load_d(eta, z);
        }
    }

    // Initialise `M_{\{c\}} = D_{L-1,c}`, the one-column minors of the final row.
    let mut minors = [F64x4::zero(); 64];
    for c in 0..L {
        minors[1usize << c] = d[(L - 1) * L + c];
    }

    // Compile-time `L` selects one monomorphised subset-minor Laplace expansion whose entries
    // are evaluated as packed SIMD values.
    // For each column subset `S`, evaluate
    // `M_S = \sum_{c\in S}(-1)^{\operatorname{pos}(c,S)}D_{L-|S|,c}M_{S\setminus\{c\}}`.
    let mut size = 2usize;
    while size <= L {
        let row = L - size;
        let mut next = [F64x4::zero(); 64];
        let mut mask = full;

        loop {
            if mask.count_ones() as usize == size {
                let mut acc = F64x4::zero();
                let mut pos = 0usize;

                for c in 0..L {
                    let bit = 1usize << c;

                    if (mask & bit) != 0 {
                        let term = minors[mask ^ bit];
                        acc = if (pos & 1) == 0 {
                            F64x4::madd(acc, d[row * L + c], term)
                        } else {
                            F64x4::msub(acc, d[row * L + c], term)
                        };
                        pos += 1;
                    }
                }

                // Store `M_S` for `S` represented by `mask`.
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

    // `S = p\,{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}` in every SIMD lane.
    let overlap_v = F64x4::mul(minors[full], F64x4::splat(pref));
    overlap_v.store(overlap);
}

/// Dispatch 8 real `m = 0` overlaps to the fixed-rank AVX-512 kernel.
/// Each output is `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `ranks`: Bra- and ket-reference excitation ranks `(RX,RW)`.
/// - `x_fixed`: Whether the fixed target excitation belongs to the x reference.
/// - `fixed`: Fixed target excitation cache.
/// - `varying`: 8 source excitation caches.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 8 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, and that `ranks` agrees with the
///   valid excitation labels stored in `fixed` and `varying`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ranks: (usize, usize),
    x_fixed: bool,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    if x_fixed {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L| unsafe {
                xw_overlap_m0_prepared_f64x8_const::<_, RX, RW, L, true>(w, fixed, varying, overlap)
            },
            unreachable!(),
        )
    } else {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L| unsafe {
                xw_overlap_m0_prepared_f64x8_const::<_, RX, RW, L, false>(
                    w, fixed, varying, overlap,
                )
            },
            unreachable!(),
        )
    }
}

/// Dispatch 8 real fixed-rank `m = 0` overlaps using compile-time `(RX,RW,L)`.
/// Each SIMD lane evaluates
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`, with the same
/// contraction rank `L` and lane-local excitation labels.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0` and `T = f64`.
/// - `fixed`: Excitation cache shared by all eight lanes; `XFIX` selects its reference side.
/// - `varying`: Eight lane-local excitation caches belonging to the other reference side.
/// - `overlap`: Real overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes eight same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid cached labels, and
///   compile-time ranks satisfying `RX + RW = L` with `1 <= L <= 6` and matching the cache ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_prepared_f64x8_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, T>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    let n = w.n();
    // `x0` and `y0` store the `X^{(0)}` and `Y^{(0)}` fundamental contractions.
    let x0 = w.x_slice(0).as_ptr().cast::<f64>();
    let y0 = w.y_slice(0).as_ptr().cast::<f64>();
    // `pref = p\,{}^{xw}\tilde S` contains the reference-pair phase and reduced overlap.
    let phase = unsafe { *std::ptr::from_ref(&w.phase).cast::<f64>() };
    let pref = phase * w.tilde_s_prod;
    // `full = \{0,\ldots,L-1\}` is the complete contraction-column subset.
    let full = (1usize << L) - 1;
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    // Select the lane-local bra excitation `{}^x\Psi_{i\cdots}^{a\cdots}`.
    let x_data = |lane: usize| -> &ExcitationSpinCache {
        if XFIX {
            fixed
        } else {
            unsafe { varying.get_unchecked(lane) }
        }
    };
    // Select the lane-local ket excitation `{}^w\Psi_{j\cdots}^{b\cdots}`.
    let w_data = |lane: usize| -> &ExcitationSpinCache {
        if XFIX {
            unsafe { varying.get_unchecked(lane) }
        } else {
            fixed
        }
    };
    // Rows are ordered as `r_\eta \in V_x\cup O_w`, namely x-particles followed by w-holes.
    let row_index = |eta: usize, lane: usize| -> usize {
        if eta < RX {
            usize::from(unsafe { *x_data(lane).indices.get_unchecked(4 + eta) }) - nocc
        } else {
            nvirt + usize::from(unsafe { *w_data(lane).indices.get_unchecked(eta - RX) })
        }
    };
    // Columns are ordered as `c_z \in O_x\cup V_w`, namely x-holes followed by w-particles.
    let col_index = |z: usize, lane: usize| -> usize {
        if z < RX {
            usize::from(unsafe { *x_data(lane).indices.get_unchecked(z) })
        } else {
            usize::from(unsafe { *w_data(lane).indices.get_unchecked(4 + z - RX) })
        }
    };
    // `D_{\eta z} = X^{(0)}_{r_\eta c_z}` for `\eta \geq z`, otherwise
    // `D_{\eta z} = Y^{(0)}_{r_\eta c_z}`.
    let load_d = |eta: usize, z: usize| -> F64x8 {
        let matrix = if eta >= z { x0 } else { y0 };
        let row_fixed = if eta < RX { XFIX } else { !XFIX };
        let col_fixed = if z < RX { XFIX } else { !XFIX };
        // A lane-invariant contraction is broadcast; otherwise gather the eight `D_{\eta z}` values.
        if row_fixed && col_fixed {
            let src = row_index(eta, 0) * n + col_index(z, 0);
            F64x8::splat(unsafe { *matrix.add(src) })
        } else {
            F64x8::from_values([
                unsafe { *matrix.add(row_index(eta, 0) * n + col_index(z, 0)) },
                unsafe { *matrix.add(row_index(eta, 1) * n + col_index(z, 1)) },
                unsafe { *matrix.add(row_index(eta, 2) * n + col_index(z, 2)) },
                unsafe { *matrix.add(row_index(eta, 3) * n + col_index(z, 3)) },
                unsafe { *matrix.add(row_index(eta, 4) * n + col_index(z, 4)) },
                unsafe { *matrix.add(row_index(eta, 5) * n + col_index(z, 5)) },
                unsafe { *matrix.add(row_index(eta, 6) * n + col_index(z, 6)) },
                unsafe { *matrix.add(row_index(eta, 7) * n + col_index(z, 7)) },
            ])
        }
    };
    // Construct the packed overlap contraction matrices `\mathbf D_{\mathrm{ov}}`.
    let mut d = [F64x8::zero(); 36];
    for eta in 0..L {
        for z in 0..L {
            d[eta * L + z] = load_d(eta, z);
        }
    }

    // Initialise `M_{\{c\}} = D_{L-1,c}`, the one-column minors of the final row.
    let mut minors = [F64x8::zero(); 64];
    for c in 0..L {
        minors[1usize << c] = d[(L - 1) * L + c];
    }

    // Compile-time `L` selects one monomorphised subset-minor Laplace expansion whose entries
    // are evaluated as packed SIMD values.
    // For each column subset `S`, evaluate
    // `M_S = \sum_{c\in S}(-1)^{\operatorname{pos}(c,S)}D_{L-|S|,c}M_{S\setminus\{c\}}`.
    let mut size = 2usize;
    while size <= L {
        let row = L - size;
        let mut next = [F64x8::zero(); 64];
        let mut mask = full;

        loop {
            if mask.count_ones() as usize == size {
                let mut acc = F64x8::zero();
                let mut pos = 0usize;

                for c in 0..L {
                    let bit = 1usize << c;

                    if (mask & bit) != 0 {
                        let term = minors[mask ^ bit];
                        acc = if (pos & 1) == 0 {
                            F64x8::madd(acc, d[row * L + c], term)
                        } else {
                            F64x8::msub(acc, d[row * L + c], term)
                        };
                        pos += 1;
                    }
                }

                // Store `M_S` for `S` represented by `mask`.
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

    // `S = p\,{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}` in every SIMD lane.
    let overlap_v = F64x8::mul(minors[full], F64x8::splat(pref));
    overlap_v.store(overlap);
}

/// Dispatch 4 complex `m = 0` overlaps to the fixed-rank AVX2/FMA kernel.
/// Each output is `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `ranks`: Bra- and ket-reference excitation ranks `(RX,RW)`.
/// - `x_fixed`: Whether the fixed target excitation belongs to the x reference.
/// - `fixed`: Fixed target excitation cache.
/// - `varying`: 4 source excitation caches.
/// - `overlap`: Complex overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 4 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure `T = Complex64`, CPU support for `AVX2/FMA`, and that `ranks` agrees
///   with the valid excitation labels stored in `fixed` and `varying`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_prepared_c64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ranks: (usize, usize),
    x_fixed: bool,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
    overlap: &mut [Complex64; 4],
) {
    if x_fixed {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L| unsafe {
                xw_overlap_m0_prepared_c64x4_const::<_, RX, RW, L, true>(w, fixed, varying, overlap)
            },
            unreachable!(),
        )
    } else {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L| unsafe {
                xw_overlap_m0_prepared_c64x4_const::<_, RX, RW, L, false>(
                    w, fixed, varying, overlap,
                )
            },
            unreachable!(),
        )
    }
}

/// Dispatch 4 complex fixed-rank `m = 0` overlaps using compile-time `(RX,RW,L)`.
/// Each SIMD lane evaluates
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`, with the same
/// contraction rank `L` and lane-local excitation labels.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0` and `T = Complex64`.
/// - `fixed`: Excitation cache shared by all four lanes; `XFIX` selects its reference side.
/// - `varying`: Four lane-local excitation caches belonging to the other reference side.
/// - `overlap`: Complex overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes four same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure `T = Complex64`, CPU support for `AVX2/FMA`, and compile-time ranks
///   satisfying `RX + RW = L` with `1 <= L <= 6` and matching valid cached labels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_prepared_c64x4_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, T>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
    overlap: &mut [Complex64; 4],
) {
    let n = w.n();
    // `x0` and `y0` store the `X^{(0)}` and `Y^{(0)}` fundamental contractions.
    let x0 = w.x_slice(0).as_ptr().cast::<Complex64>();
    let y0 = w.y_slice(0).as_ptr().cast::<Complex64>();
    // `pref = p\,{}^{xw}\tilde S` contains the reference-pair phase and reduced overlap.
    let phase = unsafe { *std::ptr::from_ref(&w.phase).cast::<Complex64>() };
    let pref = phase * w.tilde_s_prod;
    // `full = \{0,\ldots,L-1\}` is the complete contraction-column subset.
    let full = (1usize << L) - 1;
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    // Select the lane-local bra excitation `{}^x\Psi_{i\cdots}^{a\cdots}`.
    let x_data = |lane: usize| -> &ExcitationSpinCache {
        if XFIX {
            fixed
        } else {
            unsafe { varying.get_unchecked(lane) }
        }
    };
    // Select the lane-local ket excitation `{}^w\Psi_{j\cdots}^{b\cdots}`.
    let w_data = |lane: usize| -> &ExcitationSpinCache {
        if XFIX {
            unsafe { varying.get_unchecked(lane) }
        } else {
            fixed
        }
    };
    // Rows are ordered as `r_\eta \in V_x\cup O_w`, namely x-particles followed by w-holes.
    let row_index = |eta: usize, lane: usize| -> usize {
        if eta < RX {
            usize::from(unsafe { *x_data(lane).indices.get_unchecked(4 + eta) }) - nocc
        } else {
            nvirt + usize::from(unsafe { *w_data(lane).indices.get_unchecked(eta - RX) })
        }
    };
    // Columns are ordered as `c_z \in O_x\cup V_w`, namely x-holes followed by w-particles.
    let col_index = |z: usize, lane: usize| -> usize {
        if z < RX {
            usize::from(unsafe { *x_data(lane).indices.get_unchecked(z) })
        } else {
            usize::from(unsafe { *w_data(lane).indices.get_unchecked(4 + z - RX) })
        }
    };
    // `D_{\eta z} = X^{(0)}_{r_\eta c_z}` for `\eta \geq z`, otherwise
    // `D_{\eta z} = Y^{(0)}_{r_\eta c_z}`.
    let load_d = |eta: usize, z: usize| -> C64x4 {
        let matrix = if eta >= z { x0 } else { y0 };
        let row_fixed = if eta < RX { XFIX } else { !XFIX };
        let col_fixed = if z < RX { XFIX } else { !XFIX };
        // A lane-invariant contraction is broadcast; otherwise gather the four `D_{\eta z}` values.
        if row_fixed && col_fixed {
            let value = unsafe { *matrix.add(row_index(eta, 0) * n + col_index(z, 0)) };
            C64x4::splat(value.re, value.im)
        } else {
            C64x4::from_values(
                unsafe { *matrix.add(row_index(eta, 0) * n + col_index(z, 0)) },
                unsafe { *matrix.add(row_index(eta, 1) * n + col_index(z, 1)) },
                unsafe { *matrix.add(row_index(eta, 2) * n + col_index(z, 2)) },
                unsafe { *matrix.add(row_index(eta, 3) * n + col_index(z, 3)) },
            )
        }
    };
    // Construct the packed overlap contraction matrices `\mathbf D_{\mathrm{ov}}`.
    let mut d = [C64x4::zero(); 36];
    for eta in 0..L {
        for z in 0..L {
            d[eta * L + z] = load_d(eta, z);
        }
    }

    // Initialise `M_{\{c\}} = D_{L-1,c}`, the one-column minors of the final row.
    let mut minors = [C64x4::zero(); 64];
    for c in 0..L {
        minors[1usize << c] = d[(L - 1) * L + c];
    }

    // Compile-time `L` selects one monomorphised subset-minor Laplace expansion whose entries
    // are evaluated as packed SIMD values.
    // For each column subset `S`, evaluate
    // `M_S = \sum_{c\in S}(-1)^{\operatorname{pos}(c,S)}D_{L-|S|,c}M_{S\setminus\{c\}}`.
    let mut size = 2usize;
    while size <= L {
        let row = L - size;
        let mut next = [C64x4::zero(); 64];
        let mut mask = full;
        loop {
            if mask.count_ones() as usize == size {
                let mut acc = C64x4::zero();
                let mut pos = 0usize;
                for c in 0..L {
                    let bit = 1usize << c;
                    if (mask & bit) != 0 {
                        let term = minors[mask ^ bit];
                        acc = if (pos & 1) == 0 {
                            C64x4::madd(acc, d[row * L + c], term)
                        } else {
                            C64x4::msub(acc, d[row * L + c], term)
                        };
                        pos += 1;
                    }
                }
                // Store `M_S` for `S` represented by `mask`.
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

    // `S = p\,{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}` in every SIMD lane.
    let overlap_v = C64x4::mul(minors[full], C64x4::splat(pref.re, pref.im));
    let mut re = [0.0f64; 4];
    let mut im = [0.0f64; 4];
    overlap_v.store(&mut re, &mut im);
    for lane in 0..4 {
        overlap[lane] = Complex64::new(re[lane], im[lane]);
    }
}

/// Dispatch 8 complex `m = 0` overlaps to the fixed-rank AVX-512 kernel.
/// Each output is `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `ranks`: Bra- and ket-reference excitation ranks `(RX,RW)`.
/// - `x_fixed`: Whether the fixed target excitation belongs to the x reference.
/// - `fixed`: Fixed target excitation cache.
/// - `varying`: 8 source excitation caches.
/// - `overlap`: Complex overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes 8 same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure `T = Complex64`, CPU support for `AVX-512`, and that `ranks` agrees
///   with the valid excitation labels stored in `fixed` and `varying`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_prepared_c64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ranks: (usize, usize),
    x_fixed: bool,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
    overlap: &mut [Complex64; 8],
) {
    if x_fixed {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L| unsafe {
                xw_overlap_m0_prepared_c64x8_const::<_, RX, RW, L, true>(w, fixed, varying, overlap)
            },
            unreachable!(),
        )
    } else {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L| unsafe {
                xw_overlap_m0_prepared_c64x8_const::<_, RX, RW, L, false>(
                    w, fixed, varying, overlap,
                )
            },
            unreachable!(),
        )
    }
}

/// Dispatch 8 complex fixed-rank `m = 0` overlaps using compile-time `(RX,RW,L)`.
/// Each SIMD lane evaluates
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`, with the same
/// contraction rank `L` and lane-local excitation labels.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0` and `T = Complex64`.
/// - `fixed`: Excitation cache shared by all eight lanes; `XFIX` selects its reference side.
/// - `varying`: Eight lane-local excitation caches belonging to the other reference side.
/// - `overlap`: Complex overlap outputs in SIMD-lane order, excluding excitation phases.
/// # Returns
/// - `()`: Writes eight same-spin Wick overlaps into `overlap`.
/// # Safety
/// - The caller must ensure `T = Complex64`, CPU support for `AVX-512`, and compile-time ranks
///   satisfying `RX + RW = L` with `1 <= L <= 6` and matching valid cached labels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_prepared_c64x8_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, T>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
    overlap: &mut [Complex64; 8],
) {
    let n = w.n();
    // `x0` and `y0` store the `X^{(0)}` and `Y^{(0)}` fundamental contractions.
    let x0 = w.x_slice(0).as_ptr().cast::<Complex64>();
    let y0 = w.y_slice(0).as_ptr().cast::<Complex64>();
    // `pref = p\,{}^{xw}\tilde S` contains the reference-pair phase and reduced overlap.
    let phase = unsafe { *std::ptr::from_ref(&w.phase).cast::<Complex64>() };
    let pref = phase * w.tilde_s_prod;
    // `full = \{0,\ldots,L-1\}` is the complete contraction-column subset.
    let full = (1usize << L) - 1;
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    // Select the lane-local bra excitation `{}^x\Psi_{i\cdots}^{a\cdots}`.
    let x_data = |lane: usize| -> &ExcitationSpinCache {
        if XFIX {
            fixed
        } else {
            unsafe { varying.get_unchecked(lane) }
        }
    };
    // Select the lane-local ket excitation `{}^w\Psi_{j\cdots}^{b\cdots}`.
    let w_data = |lane: usize| -> &ExcitationSpinCache {
        if XFIX {
            unsafe { varying.get_unchecked(lane) }
        } else {
            fixed
        }
    };
    // Rows are ordered as `r_\eta \in V_x\cup O_w`, namely x-particles followed by w-holes.
    let row_index = |eta: usize, lane: usize| -> usize {
        if eta < RX {
            usize::from(unsafe { *x_data(lane).indices.get_unchecked(4 + eta) }) - nocc
        } else {
            nvirt + usize::from(unsafe { *w_data(lane).indices.get_unchecked(eta - RX) })
        }
    };
    // Columns are ordered as `c_z \in O_x\cup V_w`, namely x-holes followed by w-particles.
    let col_index = |z: usize, lane: usize| -> usize {
        if z < RX {
            usize::from(unsafe { *x_data(lane).indices.get_unchecked(z) })
        } else {
            usize::from(unsafe { *w_data(lane).indices.get_unchecked(4 + z - RX) })
        }
    };
    // `D_{\eta z} = X^{(0)}_{r_\eta c_z}` for `\eta \geq z`, otherwise
    // `D_{\eta z} = Y^{(0)}_{r_\eta c_z}`.
    let load_d = |eta: usize, z: usize| -> C64x8 {
        let matrix = if eta >= z { x0 } else { y0 };
        let row_fixed = if eta < RX { XFIX } else { !XFIX };
        let col_fixed = if z < RX { XFIX } else { !XFIX };
        // A lane-invariant contraction is broadcast; otherwise gather the eight `D_{\eta z}` values.
        if row_fixed && col_fixed {
            let value = unsafe { *matrix.add(row_index(eta, 0) * n + col_index(z, 0)) };
            C64x8::splat(value.re, value.im)
        } else {
            C64x8::from_values([
                unsafe { *matrix.add(row_index(eta, 0) * n + col_index(z, 0)) },
                unsafe { *matrix.add(row_index(eta, 1) * n + col_index(z, 1)) },
                unsafe { *matrix.add(row_index(eta, 2) * n + col_index(z, 2)) },
                unsafe { *matrix.add(row_index(eta, 3) * n + col_index(z, 3)) },
                unsafe { *matrix.add(row_index(eta, 4) * n + col_index(z, 4)) },
                unsafe { *matrix.add(row_index(eta, 5) * n + col_index(z, 5)) },
                unsafe { *matrix.add(row_index(eta, 6) * n + col_index(z, 6)) },
                unsafe { *matrix.add(row_index(eta, 7) * n + col_index(z, 7)) },
            ])
        }
    };
    // Construct the packed overlap contraction matrices `\mathbf D_{\mathrm{ov}}`.
    let mut d = [C64x8::zero(); 36];
    for eta in 0..L {
        for z in 0..L {
            d[eta * L + z] = load_d(eta, z);
        }
    }

    // Initialise `M_{\{c\}} = D_{L-1,c}`, the one-column minors of the final row.
    let mut minors = [C64x8::zero(); 64];
    for c in 0..L {
        minors[1usize << c] = d[(L - 1) * L + c];
    }

    // Compile-time `L` selects one monomorphised subset-minor Laplace expansion whose entries
    // are evaluated as packed SIMD values.
    // For each column subset `S`, evaluate
    // `M_S = \sum_{c\in S}(-1)^{\operatorname{pos}(c,S)}D_{L-|S|,c}M_{S\setminus\{c\}}`.
    let mut size = 2usize;
    while size <= L {
        let row = L - size;
        let mut next = [C64x8::zero(); 64];
        let mut mask = full;
        loop {
            if mask.count_ones() as usize == size {
                let mut acc = C64x8::zero();
                let mut pos = 0usize;
                for c in 0..L {
                    let bit = 1usize << c;
                    if (mask & bit) != 0 {
                        let term = minors[mask ^ bit];
                        acc = if (pos & 1) == 0 {
                            C64x8::madd(acc, d[row * L + c], term)
                        } else {
                            C64x8::msub(acc, d[row * L + c], term)
                        };
                        pos += 1;
                    }
                }
                // Store `M_S` for `S` represented by `mask`.
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

    // `S = p\,{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}` in every SIMD lane.
    let overlap_v = C64x8::mul(minors[full], C64x8::splat(pref.re, pref.im));
    let mut re = [0.0f64; 8];
    let mut im = [0.0f64; 8];
    overlap_v.store(&mut re, &mut im);
    for lane in 0..8 {
        overlap[lane] = Complex64::new(re[lane], im[lane]);
    }
}

/// Evaluate the same-spin overlap directly when `m = 0` and `L \leq 6`:
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

    dispatch_overlap_ranks!(
        (rx, rw),
        |RX, RW, L| xw_overlap_m0_direct_const::<T, RX, RW, L>(w, l_ex, g_ex),
        xw_overlap_m0_direct_gen(w, l_ex, g_ex, l),
    )
}

/// Evaluate one direct all-`m_i = 0` overlap with compile-time `(RX,RW,L)`:
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
/// The direct path constructs
/// `D_{ij}^{(0)} = X_{r_i c_j}^{(0)}` for `i >= j` and
/// `D_{ij}^{(0)} = Y_{r_i c_j}^{(0)}` for `i < j`, then evaluates the fixed-rank determinant.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// # Returns
/// - `T`: Same-spin overlap contribution.
#[inline(always)]
fn xw_overlap_m0_direct_const<T: NOCIScalar, const RX: usize, const RW: usize, const L: usize>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
) -> T {
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    let mut rows = [0usize; L];
    let mut cols = [0usize; L];
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

    let n = w.n();
    let x0 = w.x_slice(0);
    let y0 = w.y_slice(0);
    let zero = <T as From<f64>>::from(0.0);
    let mut d = [zero; 36];

    // Build `\mathbf D_{\mathrm{ov}}(0,\ldots,0)` from the fixed contraction labels, then
    // evaluate `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}`.
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

    w.phase
        * <T as From<f64>>::from(w.tilde_s_prod)
        * det_const::<T, L>(&d[..L * L]).unwrap_or(zero)
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
    let n = w.n();
    let x0 = w.x_slice(0);
    let y0 = w.y_slice(0);
    let zero = <T as From<f64>>::from(0.0);
    let mut rows = [0usize; 6];
    let mut cols = [0usize; 6];
    let mut d = [zero; 36];
    construct_determinant_indices(x_ex, w_ex, w, &mut rows[..l], &mut cols[..l]);

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

    w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det(&d[..l * l], l).unwrap_or(zero)
}

/// Evaluate the same-spin overlap when m = 0:
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
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
        // Evaluate `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)` with a
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
/// same fixed-rank determinant formula for each supported rank.
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
            // assignment, giving `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(0,\ldots,0)`.
            w.phase
                * <T as From<f64>>::from(w.tilde_s_prod)
                * det_const::<T, L>(&d[..L * L]).unwrap_or(<T as From<f64>>::from(0.0))
        }
    )
}

/// Evaluate the same-spin overlap when m = L. The only allowed distribution is
/// `(m_1,\ldots,m_L) = (1,\ldots,1), so:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,\ldots,1)`.
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
        // Evaluate `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,\ldots,1)` with a
        // fixed-rank kernel where available.
        match l {
            // This branch is retained for completeness; xw_overlap dispatches L = m = 0 to the m = 0 path.
            0 => xw_overlap_ml_const::<T, 0>(w, scratch),
            1 => xw_overlap_ml_const::<T, 1>(w, scratch),
            2 => xw_overlap_ml_const::<T, 2>(w, scratch),
            3 => xw_overlap_ml_const::<T, 3>(w, scratch),
            _ => {
                // Evaluate the prepared arbitrary-rank all-`m_i = 1` determinant directly.
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
            // surviving term is `{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}(1,\ldots,1)`.
            w.phase
                * <T as From<f64>>::from(w.tilde_s_prod)
                * det_const::<T, L>(&d[..L * L]).unwrap_or(<T as From<f64>>::from(0.0))
        }
    )
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
        // `{}^{xw}\tilde S\sum_{\{m_i\}}\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L)`.
        w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
    })
}

/// Sum fixed-rank mixed-distribution determinants for `0 < m < L`:
/// `\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L = m}}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L)`, with `m_i \in \{0,1\}`.
/// Each distribution is constructed by selecting each column from the prepared
/// all-`m_i = 0` or all-`m_i = 1` determinant before evaluating the fixed-rank
/// determinant.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `scratch`: Prepared determinant buffers `\mathbf D_{\mathrm{ov}}(0,\ldots,0)`,
///   `\mathbf D_{\mathrm{ov}}(1,\ldots,1)`, and mixed storage.
/// - `acc`: Accumulated determinant sum.
/// # Returns
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
        // `m_i \in \{0,1\}` with `\sum_i m_i = m`.
        *acc += det_const::<T, L>(&d[..L * L]).unwrap_or(<T as From<f64>>::from(0.0));
    });
}
