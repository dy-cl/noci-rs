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
use crate::config::MAXEXCIT;
#[cfg(target_arch = "x86_64")]
use crate::maths::{C64x4, C64x8, F64x4, F64x8, Simd, det_simd_const};
use crate::maths::{det_const, det_dynamic};
use crate::noci::NOCIScalar;
use crate::time_call;
use crate::{DetState, ExcitationSpin, ReducedOneSpinDetState};

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::dispatch::{dispatch_overlap_ranks, dispatch_overlap_scalar_ranks, dispatch_pair_ranks};
use super::helpers::mix_dets_same;
use super::prepare::{construct_determinant_indices, prepare_same};

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
/// For `m = 0`, the direct overlap-only path evaluates the single determinant without preparing
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
            // SAFETY: The explicit `TypeId` check proves every generic value has its `f64`
            // instantiation for the duration of the SIMD helper call.
            let w_f64 = &*std::ptr::from_ref(w).cast::<SameSpinView<'_, f64>>();
            let basis_f64 =
                std::slice::from_raw_parts(basis.as_ptr().cast::<DetState<f64>>(), basis.len());
            let scratch_f64 = &mut *std::ptr::from_mut(scratch).cast::<WickScratch<f64>>();
            let out_f64 = std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len());
            if try_xw_overlap_prepared_f64_simd(
                w_f64,
                basis_f64,
                (target, sources),
                (target_left, alpha),
                scratch_f64,
                out_f64,
            ) {
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    if w.m == 0 && TypeId::of::<T>() == TypeId::of::<Complex64>() {
        unsafe {
            // SAFETY: The explicit `TypeId` check proves every generic value has its `Complex64`
            // instantiation for the duration of the SIMD helper call.
            let w_c64 = &*std::ptr::from_ref(w).cast::<SameSpinView<'_, Complex64>>();
            let basis_c64 = std::slice::from_raw_parts(
                basis.as_ptr().cast::<DetState<Complex64>>(),
                basis.len(),
            );
            let scratch_c64 = &mut *std::ptr::from_mut(scratch).cast::<WickScratch<Complex64>>();
            let out_c64 =
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<Complex64>(), out.len());
            if try_xw_overlap_prepared_c64_simd(
                w_c64,
                basis_c64,
                (target, sources),
                (target_left, alpha),
                scratch_c64,
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

/// Fixed-rank packed overlap kernel selected by one target-feature entry point.
#[cfg(target_arch = "x86_64")]
type OverlapSimdKernel<T, const N: usize> = unsafe fn(
    &SameSpinView<'_, T>,
    (usize, usize),
    bool,
    &ExcitationSpinCache,
    &[ExcitationSpinCache; N],
    &mut [T; N],
);

/// Try to evaluate one real same-spin overlap row with fixed-rank SIMD kernels.
/// # Arguments:
/// - `w`: Real same-spin reference-pair Wick intermediates with `m = 0`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `out`: Real overlap output row.
/// # Returns
/// - `bool`: Whether an available SIMD path evaluated the complete row.
/// # Safety
/// - `w` must contain no zero-overlap orbital pairs.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_overlap_prepared_f64_simd(
    w: &SameSpinView<'_, f64>,
    basis: &[DetState<f64>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<f64>,
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
/// # Arguments:
/// - `w`: Complex same-spin reference-pair Wick intermediates with `m = 0`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `out`: Complex overlap output row.
/// # Returns
/// - `bool`: Whether an available SIMD path evaluated the complete row.
/// # Safety
/// - `w` must contain no zero-overlap orbital pairs.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_overlap_prepared_c64_simd(
    w: &SameSpinView<'_, Complex64>,
    basis: &[DetState<Complex64>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<Complex64>,
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

/// Evaluate one same-spin overlap row using packed fixed-rank kernels and scalar tails.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `out`: Overlap output row.
/// - `kernel`: Fixed-rank packed kernel behind the active target-feature boundary.
/// # Returns
/// - `()`: Writes one complete same-spin overlap-factor row.
/// # Safety
/// - `kernel` must support the current CPU and use `N` packed lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xw_overlap_prepared_simd_row<T: NOCIScalar, const N: usize>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    out: &mut [T],
    kernel: OverlapSimdKernel<T, N>,
) {
    let (target, sources) = reps;
    let (target_left, _) = flags;
    let target_cache = target.excitation_cache;
    let target_rank = usize::from(target_cache.rank);
    let mut bins = [[ExcitationSpinCache::default(); N]; MAXEXCIT + 1];
    let mut phases = [[1.0f64; N]; MAXEXCIT + 1];
    let mut outputs = [[0usize; N]; MAXEXCIT + 1];
    let mut counts = [0usize; MAXEXCIT + 1];

    for (col, source) in sources.iter().enumerate() {
        let source_cache = source.excitation_cache;
        let source_rank = usize::from(source_cache.rank);
        let ranks = if target_left {
            (target_rank, source_rank)
        } else {
            (source_rank, target_rank)
        };

        if target_rank <= MAXEXCIT && source_rank <= MAXEXCIT && target_rank + source_rank != 0 {
            let count = counts[source_rank];
            bins[source_rank][count] = source_cache;
            phases[source_rank][count] = source.phase;
            outputs[source_rank][count] = col;
            counts[source_rank] += 1;

            if counts[source_rank] == N {
                let mut overlap = [T::from_real(0.0); N];
                unsafe {
                    kernel(
                        w,
                        ranks,
                        target_left,
                        &target_cache,
                        &bins[source_rank],
                        &mut overlap,
                    );
                }
                for lane in 0..N {
                    out[outputs[source_rank][lane]] =
                        T::from_real(target.phase * phases[source_rank][lane]) * overlap[lane];
                }
                counts[source_rank] = 0;
            }
        } else {
            out[col] =
                xw_overlap_prepared_scalar_value(w, basis, (target, *source), flags, scratch);
        }
    }

    for source_rank in 0..=MAXEXCIT {
        for &col in &outputs[source_rank][..counts[source_rank]] {
            out[col] =
                xw_overlap_prepared_scalar_value(w, basis, (target, sources[col]), flags, scratch);
        }
    }
}

/// Evaluate one real same-spin overlap row with four-lane AVX2/FMA packets.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives.
/// - `flags`: Target-reference and spin flags.
/// - `scratch`: Reusable Wick workspace.
/// - `out`: Real overlap output row.
/// # Returns
/// - `()`: Writes one complete row.
/// # Safety
/// - The current CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_prepared_f64x4_row(
    w: &SameSpinView<'_, f64>,
    basis: &[DetState<f64>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<f64>,
    out: &mut [f64],
) {
    unsafe {
        xw_overlap_prepared_simd_row::<f64, 4>(
            w,
            basis,
            reps,
            flags,
            scratch,
            out,
            xw_overlap_m0_prepared_f64x4,
        );
    }
}

/// Evaluate one real same-spin overlap row with eight-lane AVX-512 packets.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives.
/// - `flags`: Target-reference and spin flags.
/// - `scratch`: Reusable Wick workspace.
/// - `out`: Real overlap output row.
/// # Returns
/// - `()`: Writes one complete row.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_prepared_f64x8_row(
    w: &SameSpinView<'_, f64>,
    basis: &[DetState<f64>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<f64>,
    out: &mut [f64],
) {
    unsafe {
        xw_overlap_prepared_simd_row::<f64, 8>(
            w,
            basis,
            reps,
            flags,
            scratch,
            out,
            xw_overlap_m0_prepared_f64x8,
        );
    }
}

/// Evaluate one complex same-spin overlap row with four-lane AVX2/FMA packets.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives.
/// - `flags`: Target-reference and spin flags.
/// - `scratch`: Reusable Wick workspace.
/// - `out`: Complex overlap output row.
/// # Returns
/// - `()`: Writes one complete row.
/// # Safety
/// - The current CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_prepared_c64x4_row(
    w: &SameSpinView<'_, Complex64>,
    basis: &[DetState<Complex64>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<Complex64>,
    out: &mut [Complex64],
) {
    unsafe {
        xw_overlap_prepared_simd_row::<Complex64, 4>(
            w,
            basis,
            reps,
            flags,
            scratch,
            out,
            xw_overlap_m0_prepared_c64x4,
        );
    }
}

/// Evaluate one complex same-spin overlap row with eight-lane AVX-512 packets.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `reps`: Target representative and source representatives.
/// - `flags`: Target-reference and spin flags.
/// - `scratch`: Reusable Wick workspace.
/// - `out`: Complex overlap output row.
/// # Returns
/// - `()`: Writes one complete row.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_prepared_c64x8_row(
    w: &SameSpinView<'_, Complex64>,
    basis: &[DetState<Complex64>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<Complex64>,
    out: &mut [Complex64],
) {
    unsafe {
        xw_overlap_prepared_simd_row::<Complex64, 8>(
            w,
            basis,
            reps,
            flags,
            scratch,
            out,
            xw_overlap_m0_prepared_c64x8,
        );
    }
}

/// Evaluate packed fixed-rank overlaps using one packed arithmetic implementation.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `fixed`: Excitation cache shared by all lanes.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Overlap outputs in lane order.
/// # Returns
/// - `()`: Writes `LANES` overlap values.
/// # Safety
/// - Cached labels must match `RX` and `RW`; caller must establish `V` CPU support.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xw_overlap_m0_prepared_simd_const<
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
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; LANES],
    overlap: &mut [T; LANES],
) {
    let n = w.n();
    let x0 = w.x_slice(0);
    let y0 = w.y_slice(0);
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    let x_data = |lane: usize| -> &ExcitationSpinCache {
        if XFIX {
            fixed
        } else {
            unsafe { varying.get_unchecked(lane) }
        }
    };
    let w_data = |lane: usize| -> &ExcitationSpinCache {
        if XFIX {
            unsafe { varying.get_unchecked(lane) }
        } else {
            fixed
        }
    };
    let row_index = |eta: usize, lane: usize| -> usize {
        if eta < RX {
            usize::from(unsafe { *x_data(lane).particles.get_unchecked(eta) }) - nocc
        } else {
            nvirt + usize::from(unsafe { *w_data(lane).holes.get_unchecked(eta - RX) })
        }
    };
    let col_index = |z: usize, lane: usize| -> usize {
        if z < RX {
            usize::from(unsafe { *x_data(lane).holes.get_unchecked(z) })
        } else {
            usize::from(unsafe { *w_data(lane).particles.get_unchecked(z - RX) })
        }
    };
    let zero = V::zero();
    let mut d = [zero; D];

    for eta in 0..L {
        for z in 0..L {
            let matrix = if eta >= z { x0 } else { y0 };
            let row_fixed = if eta < RX { XFIX } else { !XFIX };
            let col_fixed = if z < RX { XFIX } else { !XFIX };
            d[eta * L + z] = if row_fixed && col_fixed {
                let src = row_index(eta, 0) * n + col_index(z, 0);
                V::splat(unsafe { *matrix.get_unchecked(src) })
            } else {
                let mut values = [T::from_real(0.0); LANES];
                for (lane, value) in values.iter_mut().enumerate() {
                    let src = row_index(eta, lane) * n + col_index(z, lane);
                    *value = unsafe { *matrix.get_unchecked(src) };
                }
                V::load(&values)
            };
        }
    }

    let determinant = det_simd_const::<V, LANES, L, D>(&d);
    let pref = w.phase * T::from_real(w.tilde_s_prod);
    V::store(V::mul(determinant, V::splat(pref)), overlap);
}

/// Dispatch four real overlaps to a compile-time rank arm.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `ranks`: Bra and ket excitation ranks.
/// - `x_fixed`: Whether the shared cache belongs to the bra.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Real overlap outputs.
/// # Returns
/// - `()`: Writes four overlap values.
/// # Safety
/// - The current CPU must support AVX2 and FMA; cached ranks must match `ranks`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_prepared_f64x4(
    w: &SameSpinView<'_, f64>,
    ranks: (usize, usize),
    x_fixed: bool,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    if x_fixed {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L, D| unsafe {
                xw_overlap_m0_prepared_f64x4_const::<RX, RW, L, D, true>(w, fixed, varying, overlap)
            },
            (),
        )
    } else {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L, D| unsafe {
                xw_overlap_m0_prepared_f64x4_const::<RX, RW, L, D, false>(
                    w, fixed, varying, overlap,
                )
            },
            (),
        )
    }
}

/// Evaluate four real fixed-rank overlaps with AVX2/FMA.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Real overlap outputs.
/// # Returns
/// - `()`: Writes four overlap values.
/// # Safety
/// - The current CPU must support AVX2 and FMA; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_prepared_f64x4_const<
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, f64>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
    overlap: &mut [f64; 4],
) {
    unsafe {
        xw_overlap_m0_prepared_simd_const::<f64, F64x4, 4, RX, RW, L, D, XFIX>(
            w, fixed, varying, overlap,
        );
    }
}

/// Dispatch eight real overlaps to a compile-time rank arm.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `ranks`: Bra and ket excitation ranks.
/// - `x_fixed`: Whether the shared cache belongs to the bra.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Real overlap outputs.
/// # Returns
/// - `()`: Writes eight overlap values.
/// # Safety
/// - The current CPU must support AVX-512F; cached ranks must match `ranks`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_prepared_f64x8(
    w: &SameSpinView<'_, f64>,
    ranks: (usize, usize),
    x_fixed: bool,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    if x_fixed {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L, D| unsafe {
                xw_overlap_m0_prepared_f64x8_const::<RX, RW, L, D, true>(w, fixed, varying, overlap)
            },
            (),
        )
    } else {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L, D| unsafe {
                xw_overlap_m0_prepared_f64x8_const::<RX, RW, L, D, false>(
                    w, fixed, varying, overlap,
                )
            },
            (),
        )
    }
}

/// Evaluate eight real fixed-rank overlaps with AVX-512F.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Real overlap outputs.
/// # Returns
/// - `()`: Writes eight overlap values.
/// # Safety
/// - The current CPU must support AVX-512F; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_prepared_f64x8_const<
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, f64>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
    overlap: &mut [f64; 8],
) {
    unsafe {
        xw_overlap_m0_prepared_simd_const::<f64, F64x8, 8, RX, RW, L, D, XFIX>(
            w, fixed, varying, overlap,
        );
    }
}

/// Dispatch four complex overlaps to a compile-time rank arm.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `ranks`: Bra and ket excitation ranks.
/// - `x_fixed`: Whether the shared cache belongs to the bra.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Complex overlap outputs.
/// # Returns
/// - `()`: Writes four overlap values.
/// # Safety
/// - The current CPU must support AVX2 and FMA; cached ranks must match `ranks`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_prepared_c64x4(
    w: &SameSpinView<'_, Complex64>,
    ranks: (usize, usize),
    x_fixed: bool,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
    overlap: &mut [Complex64; 4],
) {
    if x_fixed {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L, D| unsafe {
                xw_overlap_m0_prepared_c64x4_const::<RX, RW, L, D, true>(w, fixed, varying, overlap)
            },
            (),
        )
    } else {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L, D| unsafe {
                xw_overlap_m0_prepared_c64x4_const::<RX, RW, L, D, false>(
                    w, fixed, varying, overlap,
                )
            },
            (),
        )
    }
}

/// Evaluate four complex fixed-rank overlaps with AVX2/FMA.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Complex overlap outputs.
/// # Returns
/// - `()`: Writes four overlap values.
/// # Safety
/// - The current CPU must support AVX2 and FMA; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_overlap_m0_prepared_c64x4_const<
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, Complex64>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
    overlap: &mut [Complex64; 4],
) {
    unsafe {
        xw_overlap_m0_prepared_simd_const::<Complex64, C64x4, 4, RX, RW, L, D, XFIX>(
            w, fixed, varying, overlap,
        );
    }
}

/// Dispatch eight complex overlaps to a compile-time rank arm.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `ranks`: Bra and ket excitation ranks.
/// - `x_fixed`: Whether the shared cache belongs to the bra.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Complex overlap outputs.
/// # Returns
/// - `()`: Writes eight overlap values.
/// # Safety
/// - The current CPU must support AVX-512F; cached ranks must match `ranks`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_prepared_c64x8(
    w: &SameSpinView<'_, Complex64>,
    ranks: (usize, usize),
    x_fixed: bool,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
    overlap: &mut [Complex64; 8],
) {
    if x_fixed {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L, D| unsafe {
                xw_overlap_m0_prepared_c64x8_const::<RX, RW, L, D, true>(w, fixed, varying, overlap)
            },
            (),
        )
    } else {
        dispatch_overlap_ranks!(
            ranks,
            |RX, RW, L, D| unsafe {
                xw_overlap_m0_prepared_c64x8_const::<RX, RW, L, D, false>(
                    w, fixed, varying, overlap,
                )
            },
            (),
        )
    }
}

/// Evaluate eight complex fixed-rank overlaps with AVX-512F.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Complex overlap outputs.
/// # Returns
/// - `()`: Writes eight overlap values.
/// # Safety
/// - The current CPU must support AVX-512F; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_m0_prepared_c64x8_const<
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, Complex64>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
    overlap: &mut [Complex64; 8],
) {
    unsafe {
        xw_overlap_m0_prepared_simd_const::<Complex64, C64x8, 8, RX, RW, L, D, XFIX>(
            w, fixed, varying, overlap,
        );
    }
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

    dispatch_overlap_scalar_ranks!(
        (rx, rw),
        |RX, RW, L, D| xw_overlap_m0_direct_const::<T, RX, RW, L, D>(w, l_ex, g_ex),
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
fn xw_overlap_m0_direct_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
>(
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
    let mut d = [zero; D];

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

    w.phase * <T as From<f64>>::from(w.tilde_s_prod) * det_const::<T, L, D>(&d)
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
    let mut rows = vec![0usize; l];
    let mut cols = vec![0usize; l];
    let mut d = vec![zero; l * l];
    construct_determinant_indices(x_ex, w_ex, w, &mut rows, &mut cols);

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
