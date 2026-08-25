// nonorthogonalwicks/eval/prepareonebodyoverlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

// External crate imports.
use num_complex::Complex64;

// Crate-root imports.
use crate::maths::{adjugate_transpose, adjugate_transpose_const, build_d, build_d_const, det};
use crate::noci::NOCIScalar;
use crate::time_call;
use crate::{DetState, ExcitationSpin, ExcitationSpinCache, ReducedOneSpinDetState};

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, mix_dets_same,
};
use super::prepare::{construct_determinant_indices, construct_determinant_indices_const};
#[cfg(target_arch = "x86_64")]
use super::simd::{C64x4, C64x8, F64x4, F64x8};

/// Inputs and outputs for one row of same-spin one-body factors.
pub(crate) struct SameSpinOneBodyBatch<'a, T: NOCIScalar> {
    /// Determinant basis used by scalar fallback evaluation.
    pub(crate) basis: &'a [DetState<T>],
    /// Reduced target spin representative shared by this factor row.
    pub(crate) target: ReducedOneSpinDetState,
    /// Reduced source spin representatives in output-column order.
    pub(crate) sources: &'a [ReducedOneSpinDetState],
    /// Whether the target determinant belongs to the left Wick reference.
    pub(crate) target_left: bool,
    /// Whether alpha-spin rather than beta-spin factors are being evaluated.
    pub(crate) alpha: bool,
    /// Same-spin overlap factor outputs.
    pub(crate) overlap: &'a mut [T],
    /// Same-spin generalised-Fock factor outputs.
    pub(crate) fock: &'a mut [T],
}

/// Evaluate one row of same-spin overlap and generalised-Fock factors.
/// Requests with `m = 0`, scalar type `f64` or `Complex64`, and `L = 1,\ldots,4` are grouped by
/// contraction rank and evaluated with the widest available one-body SIMD kernel. Unsupported
/// requests and unavailable SIMD targets use the scalar prepared Wick evaluator.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `batch`: One row of same-spin one-body-factor work.
/// - `scratch`: Reusable same-spin Wick workspace for scalar fallback evaluation.
/// - `tol`: Numerical tolerance used by scalar prepared Wick evaluation.
/// # Returns
/// - `()`: Writes one complete same-spin overlap and generalised-Fock factor row.
pub(crate) fn xw_f_overlap_prepared_batched<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    batch: SameSpinOneBodyBatch<'_, T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) {
    let SameSpinOneBodyBatch {
        basis,
        target,
        sources,
        target_left,
        alpha,
        overlap,
        fock,
    } = batch;

    #[cfg(target_arch = "x86_64")]
    if w.m == 0 && TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            // SAFETY: The explicit `TypeId` check above proves `T = f64`, so the row storage has
            // the same layout as `f64` for the duration of the SIMD helper call.
            let overlap_f64 =
                std::slice::from_raw_parts_mut(overlap.as_mut_ptr().cast::<f64>(), overlap.len());
            let fock_f64 =
                std::slice::from_raw_parts_mut(fock.as_mut_ptr().cast::<f64>(), fock.len());

            if try_xw_f_overlap_prepared_f64_simd(
                w,
                basis,
                (target, sources),
                (target_left, alpha),
                scratch,
                tol,
                (overlap_f64, fock_f64),
            ) {
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    if w.m == 0 && TypeId::of::<T>() == TypeId::of::<Complex64>() {
        unsafe {
            // SAFETY: The explicit `TypeId` check above proves `T = Complex64`, so the row storage
            // has the same layout as `Complex64` for the duration of the SIMD helper call.
            let overlap_c64 = std::slice::from_raw_parts_mut(
                overlap.as_mut_ptr().cast::<Complex64>(),
                overlap.len(),
            );
            let fock_c64 =
                std::slice::from_raw_parts_mut(fock.as_mut_ptr().cast::<Complex64>(), fock.len());

            if try_xw_f_overlap_prepared_c64_simd(
                w,
                basis,
                (target, sources),
                (target_left, alpha),
                scratch,
                tol,
                (overlap_c64, fock_c64),
            ) {
                return;
            }
        }
    }

    xw_f_overlap_prepared_scalar_row(
        w,
        basis,
        (target, sources),
        (target_left, alpha),
        scratch,
        tol,
        (overlap, fock),
    );
}

/// Evaluate one same-spin one-body factor row through the scalar prepared Wick path.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `basis`: Determinant basis containing full excitation masks.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `tol`: Numerical tolerance used by scalar prepared Wick evaluation.
/// - `out`: Overlap and generalised-Fock output rows.
/// # Returns
/// - `()`: Writes one complete same-spin factor row.
fn xw_f_overlap_prepared_scalar_row<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    tol: f64,
    out: (&mut [T], &mut [T]),
) {
    let (target_rep, sources) = reps;
    let (target_left, alpha) = flags;
    let (overlap, fock) = out;
    let target_phase = target_rep.phase;
    let target = &basis[target_rep.det];
    let target_ex = if alpha {
        &target.excitation.alpha
    } else {
        &target.excitation.beta
    };

    for (col, source_rep) in sources.iter().enumerate() {
        let source = &basis[source_rep.det];
        let source_ex = if alpha {
            &source.excitation.alpha
        } else {
            &source.excitation.beta
        };
        let (x_ex, w_ex) = if target_left {
            (target_ex, source_ex)
        } else {
            (source_ex, target_ex)
        };
        let (s, f) = xw_f_overlap_prepared(w, x_ex, w_ex, scratch, tol);
        let phase = T::from_real(target_phase * source_rep.phase);

        overlap[col] = phase * s;
        fock[col] = phase * f;
    }
}

/// Evaluate one same-spin one-body factor through the scalar prepared Wick path.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `basis`: Determinant basis containing full excitation masks.
/// - `reps`: Target and source spin representatives.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `tol`: Numerical tolerance used by scalar prepared Wick evaluation.
/// # Returns
/// - `(T, T)`: Same-spin overlap and generalised-Fock factors before excitation phase.
#[cfg(target_arch = "x86_64")]
fn xw_f_overlap_prepared_scalar_value<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, ReducedOneSpinDetState),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    let (target_rep, source_rep) = reps;
    let (target_left, alpha) = flags;
    let target = &basis[target_rep.det];
    let source = &basis[source_rep.det];
    let (target_ex, source_ex) = if alpha {
        (&target.excitation.alpha, &source.excitation.alpha)
    } else {
        (&target.excitation.beta, &source.excitation.beta)
    };
    let (x_ex, w_ex) = if target_left {
        (target_ex, source_ex)
    } else {
        (source_ex, target_ex)
    };
    xw_f_overlap_prepared(w, x_ex, w_ex, scratch, tol)
}

/// Try to evaluate one real same-spin factor row with fixed-rank SIMD kernels.
/// Rank-compatible requests are binned by `L = 1,\ldots,4`; unsupported requests use the scalar
/// prepared Wick evaluator inside the same row traversal.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `basis`: Determinant basis containing full excitation masks.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `tol`: Numerical tolerance used by scalar fallback evaluation.
/// - `out`: Real overlap and generalised-Fock output rows.
/// # Returns
/// - `bool`: Whether SIMD support existed and the complete row was written.
/// # Safety
/// - The caller must ensure `T = f64` before reinterpreting output storage as `f64`.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_f_overlap_prepared_f64_simd<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    tol: f64,
    out: (&mut [f64], &mut [f64]),
) -> bool {
    let (target_rep, sources) = reps;
    let (target_left, alpha) = flags;
    let (overlap, fock) = out;
    let target_cache = target_rep.excitation_cache;
    let target_phase = target_rep.phase;

    unsafe {
        if is_x86_feature_detected!("avx512f") {
            let mut bins = [[ExcitationSpinCache::default(); 8]; 5];
            let mut phases = [[1.0f64; 8]; 5];
            let mut outputs = [[0usize; 8]; 5];
            let mut counts = [0usize; 5];

            for (col, source_rep) in sources.iter().enumerate() {
                let source_cache = source_rep.excitation_cache;
                let source_phase = source_rep.phase;
                let l = usize::from(target_cache.rank) + usize::from(source_cache.rank);

                if (1..=4).contains(&l) {
                    let count = counts[l];
                    bins[l][count] = source_cache;
                    phases[l][count] = source_phase;
                    outputs[l][count] = col;
                    counts[l] += 1;

                    if counts[l] == 8 {
                        flush_f64x8_bin(
                            w,
                            (target_cache, target_phase, target_left),
                            (l, 8),
                            &mut bins[l],
                            &phases[l],
                            &outputs[l],
                            (&mut *overlap, &mut *fock),
                        );
                        counts[l] = 0;
                    }
                } else {
                    let (s, f) = xw_f_overlap_prepared_scalar_value(
                        w,
                        basis,
                        (target_rep, *source_rep),
                        (target_left, alpha),
                        scratch,
                        tol,
                    );
                    let phase = target_phase * source_phase;
                    overlap[col] = phase * *std::ptr::from_ref(&s).cast::<f64>();
                    fock[col] = phase * *std::ptr::from_ref(&f).cast::<f64>();
                }
            }

            for l in 1..=4 {
                flush_f64x8_bin(
                    w,
                    (target_cache, target_phase, target_left),
                    (l, counts[l]),
                    &mut bins[l],
                    &phases[l],
                    &outputs[l],
                    (&mut *overlap, &mut *fock),
                );
            }
            return true;
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut bins = [[ExcitationSpinCache::default(); 4]; 5];
            let mut phases = [[1.0f64; 4]; 5];
            let mut outputs = [[0usize; 4]; 5];
            let mut counts = [0usize; 5];

            for (col, source_rep) in sources.iter().enumerate() {
                let source_cache = source_rep.excitation_cache;
                let source_phase = source_rep.phase;
                let l = usize::from(target_cache.rank) + usize::from(source_cache.rank);

                if (1..=4).contains(&l) {
                    let count = counts[l];
                    bins[l][count] = source_cache;
                    phases[l][count] = source_phase;
                    outputs[l][count] = col;
                    counts[l] += 1;

                    if counts[l] == 4 {
                        flush_f64x4_bin(
                            w,
                            (target_cache, target_phase, target_left),
                            (l, 4),
                            &mut bins[l],
                            &phases[l],
                            &outputs[l],
                            (&mut *overlap, &mut *fock),
                        );
                        counts[l] = 0;
                    }
                } else {
                    let (s, f) = xw_f_overlap_prepared_scalar_value(
                        w,
                        basis,
                        (target_rep, *source_rep),
                        (target_left, alpha),
                        scratch,
                        tol,
                    );
                    let phase = target_phase * source_phase;
                    overlap[col] = phase * *std::ptr::from_ref(&s).cast::<f64>();
                    fock[col] = phase * *std::ptr::from_ref(&f).cast::<f64>();
                }
            }

            for l in 1..=4 {
                flush_f64x4_bin(
                    w,
                    (target_cache, target_phase, target_left),
                    (l, counts[l]),
                    &mut bins[l],
                    &phases[l],
                    &outputs[l],
                    (&mut *overlap, &mut *fock),
                );
            }
            return true;
        }
    }

    false
}

/// Flush one partially-filled or full real AVX-512 rank bin.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64`.
/// - `target`: Target excitation cache, target phase and target-left flag.
/// - `rank`: Shared contraction rank and valid lane count.
/// - `bins`: Source excitation cache packet.
/// - `phases`: Source excitation phases in packet order.
/// - `outputs`: Output columns in packet order.
/// - `out`: Real overlap and generalised-Fock output rows.
/// # Returns
/// - `()`: Writes `count` output lanes.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
unsafe fn flush_f64x8_bin<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    target: (ExcitationSpinCache, f64, bool),
    rank: (usize, usize),
    bins: &mut [ExcitationSpinCache; 8],
    phases: &[f64; 8],
    outputs: &[usize; 8],
    out: (&mut [f64], &mut [f64]),
) {
    let (target_cache, target_phase, target_left) = target;
    let (l, count) = rank;
    let (overlap, fock) = out;
    if count == 0 {
        return;
    }

    let fill = bins[0];
    for slot in bins.iter_mut().skip(count) {
        *slot = fill;
    }

    let target_batch = [target_cache; 8];
    let source_batch = *bins;
    let (x_ex, w_ex) = if target_left {
        (&target_batch, &source_batch)
    } else {
        (&source_batch, &target_batch)
    };
    let mut s = [0.0f64; 8];
    let mut f = [0.0f64; 8];
    unsafe {
        xw_f_overlap_m0_prepared_f64x8(w, l, x_ex, w_ex, &mut s, &mut f);
    }

    for lane in 0..count {
        let output = outputs[lane];
        let phase = target_phase * phases[lane];
        overlap[output] = s[lane] * phase;
        fock[output] = f[lane] * phase;
    }
}

/// Flush one partially-filled or full real AVX2 rank bin.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64`.
/// - `target`: Target excitation cache, target phase and target-left flag.
/// - `rank`: Shared contraction rank and valid lane count.
/// - `bins`: Source excitation cache packet.
/// - `phases`: Source excitation phases in packet order.
/// - `outputs`: Output columns in packet order.
/// - `out`: Real overlap and generalised-Fock output rows.
/// # Returns
/// - `()`: Writes `count` output lanes.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
unsafe fn flush_f64x4_bin<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    target: (ExcitationSpinCache, f64, bool),
    rank: (usize, usize),
    bins: &mut [ExcitationSpinCache; 4],
    phases: &[f64; 4],
    outputs: &[usize; 4],
    out: (&mut [f64], &mut [f64]),
) {
    let (target_cache, target_phase, target_left) = target;
    let (l, count) = rank;
    let (overlap, fock) = out;
    if count == 0 {
        return;
    }

    let fill = bins[0];
    for slot in bins.iter_mut().skip(count) {
        *slot = fill;
    }

    let target_batch = [target_cache; 4];
    let source_batch = *bins;
    let (x_ex, w_ex) = if target_left {
        (&target_batch, &source_batch)
    } else {
        (&source_batch, &target_batch)
    };
    let mut s = [0.0f64; 4];
    let mut f = [0.0f64; 4];
    unsafe {
        xw_f_overlap_m0_prepared_f64x4(w, l, x_ex, w_ex, &mut s, &mut f);
    }

    for lane in 0..count {
        let output = outputs[lane];
        let phase = target_phase * phases[lane];
        overlap[output] = s[lane] * phase;
        fock[output] = f[lane] * phase;
    }
}

/// Try to evaluate one complex same-spin factor row with fixed-rank SIMD kernels.
/// Rank-compatible requests are binned by `L = 1,\ldots,4`; unsupported requests use the scalar
/// prepared Wick evaluator inside the same row traversal.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `basis`: Determinant basis containing full excitation masks.
/// - `reps`: Target representative and source representatives in output-column order.
/// - `flags`: Whether the target is left, and whether alpha-spin factors are being evaluated.
/// - `scratch`: Reusable same-spin Wick evaluator workspace.
/// - `tol`: Numerical tolerance used by scalar fallback evaluation.
/// - `out`: Complex overlap and generalised-Fock output rows.
/// # Returns
/// - `bool`: Whether SIMD support existed and the complete row was written.
/// # Safety
/// - The caller must ensure `T = Complex64` before reinterpreting output storage as `Complex64`.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_f_overlap_prepared_c64_simd<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    flags: (bool, bool),
    scratch: &mut WickScratch<T>,
    tol: f64,
    out: (&mut [Complex64], &mut [Complex64]),
) -> bool {
    let (target_rep, sources) = reps;
    let (target_left, alpha) = flags;
    let (overlap, fock) = out;
    let target_cache = target_rep.excitation_cache;
    let target_phase = target_rep.phase;

    unsafe {
        if is_x86_feature_detected!("avx512f") {
            let mut bins = [[ExcitationSpinCache::default(); 8]; 5];
            let mut phases = [[1.0f64; 8]; 5];
            let mut outputs = [[0usize; 8]; 5];
            let mut counts = [0usize; 5];

            for (col, source_rep) in sources.iter().enumerate() {
                let source_cache = source_rep.excitation_cache;
                let source_phase = source_rep.phase;
                let l = usize::from(target_cache.rank) + usize::from(source_cache.rank);

                if (1..=4).contains(&l) {
                    let count = counts[l];
                    bins[l][count] = source_cache;
                    phases[l][count] = source_phase;
                    outputs[l][count] = col;
                    counts[l] += 1;

                    if counts[l] == 8 {
                        flush_c64x8_bin(
                            w,
                            (target_cache, target_phase, target_left),
                            (l, 8),
                            &mut bins[l],
                            &phases[l],
                            &outputs[l],
                            (&mut *overlap, &mut *fock),
                        );
                        counts[l] = 0;
                    }
                } else {
                    let (s, f) = xw_f_overlap_prepared_scalar_value(
                        w,
                        basis,
                        (target_rep, *source_rep),
                        (target_left, alpha),
                        scratch,
                        tol,
                    );
                    let phase = target_phase * source_phase;
                    overlap[col] = *std::ptr::from_ref(&s).cast::<Complex64>() * phase;
                    fock[col] = *std::ptr::from_ref(&f).cast::<Complex64>() * phase;
                }
            }

            for l in 1..=4 {
                flush_c64x8_bin(
                    w,
                    (target_cache, target_phase, target_left),
                    (l, counts[l]),
                    &mut bins[l],
                    &phases[l],
                    &outputs[l],
                    (&mut *overlap, &mut *fock),
                );
            }
            return true;
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut bins = [[ExcitationSpinCache::default(); 4]; 5];
            let mut phases = [[1.0f64; 4]; 5];
            let mut outputs = [[0usize; 4]; 5];
            let mut counts = [0usize; 5];

            for (col, source_rep) in sources.iter().enumerate() {
                let source_cache = source_rep.excitation_cache;
                let source_phase = source_rep.phase;
                let l = usize::from(target_cache.rank) + usize::from(source_cache.rank);

                if (1..=4).contains(&l) {
                    let count = counts[l];
                    bins[l][count] = source_cache;
                    phases[l][count] = source_phase;
                    outputs[l][count] = col;
                    counts[l] += 1;

                    if counts[l] == 4 {
                        flush_c64x4_bin(
                            w,
                            (target_cache, target_phase, target_left),
                            (l, 4),
                            &mut bins[l],
                            &phases[l],
                            &outputs[l],
                            (&mut *overlap, &mut *fock),
                        );
                        counts[l] = 0;
                    }
                } else {
                    let (s, f) = xw_f_overlap_prepared_scalar_value(
                        w,
                        basis,
                        (target_rep, *source_rep),
                        (target_left, alpha),
                        scratch,
                        tol,
                    );
                    let phase = target_phase * source_phase;
                    overlap[col] = *std::ptr::from_ref(&s).cast::<Complex64>() * phase;
                    fock[col] = *std::ptr::from_ref(&f).cast::<Complex64>() * phase;
                }
            }

            for l in 1..=4 {
                flush_c64x4_bin(
                    w,
                    (target_cache, target_phase, target_left),
                    (l, counts[l]),
                    &mut bins[l],
                    &phases[l],
                    &outputs[l],
                    (&mut *overlap, &mut *fock),
                );
            }
            return true;
        }
    }

    false
}

/// Flush one partially-filled or full complex AVX-512 rank bin.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64`.
/// - `target`: Target excitation cache, target phase and target-left flag.
/// - `rank`: Shared contraction rank and valid lane count.
/// - `bins`: Source excitation cache packet.
/// - `phases`: Source excitation phases in packet order.
/// - `outputs`: Output columns in packet order.
/// - `out`: Complex overlap and generalised-Fock output rows.
/// # Returns
/// - `()`: Writes `count` output lanes.
/// # Safety
/// - The caller must ensure `T = Complex64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
unsafe fn flush_c64x8_bin<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    target: (ExcitationSpinCache, f64, bool),
    rank: (usize, usize),
    bins: &mut [ExcitationSpinCache; 8],
    phases: &[f64; 8],
    outputs: &[usize; 8],
    out: (&mut [Complex64], &mut [Complex64]),
) {
    let (target_cache, target_phase, target_left) = target;
    let (l, count) = rank;
    let (overlap, fock) = out;
    if count == 0 {
        return;
    }

    let fill = bins[0];
    for slot in bins.iter_mut().skip(count) {
        *slot = fill;
    }

    let target_batch = [target_cache; 8];
    let source_batch = *bins;
    let (x_ex, w_ex) = if target_left {
        (&target_batch, &source_batch)
    } else {
        (&source_batch, &target_batch)
    };
    let mut s = [Complex64::new(0.0, 0.0); 8];
    let mut f = [Complex64::new(0.0, 0.0); 8];
    unsafe {
        xw_f_overlap_m0_prepared_c64x8(w, l, x_ex, w_ex, &mut s, &mut f);
    }

    for lane in 0..count {
        let output = outputs[lane];
        let phase = target_phase * phases[lane];
        overlap[output] = s[lane] * phase;
        fock[output] = f[lane] * phase;
    }
}

/// Flush one partially-filled or full complex AVX2 rank bin.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64`.
/// - `target`: Target excitation cache, target phase and target-left flag.
/// - `rank`: Shared contraction rank and valid lane count.
/// - `bins`: Source excitation cache packet.
/// - `phases`: Source excitation phases in packet order.
/// - `outputs`: Output columns in packet order.
/// - `out`: Complex overlap and generalised-Fock output rows.
/// # Returns
/// - `()`: Writes `count` output lanes.
/// # Safety
/// - The caller must ensure `T = Complex64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
unsafe fn flush_c64x4_bin<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    target: (ExcitationSpinCache, f64, bool),
    rank: (usize, usize),
    bins: &mut [ExcitationSpinCache; 4],
    phases: &[f64; 4],
    outputs: &[usize; 4],
    out: (&mut [Complex64], &mut [Complex64]),
) {
    let (target_cache, target_phase, target_left) = target;
    let (l, count) = rank;
    let (overlap, fock) = out;
    if count == 0 {
        return;
    }

    let fill = bins[0];
    for slot in bins.iter_mut().skip(count) {
        *slot = fill;
    }

    let target_batch = [target_cache; 4];
    let source_batch = *bins;
    let (x_ex, w_ex) = if target_left {
        (&target_batch, &source_batch)
    } else {
        (&source_batch, &target_batch)
    };
    let mut s = [Complex64::new(0.0, 0.0); 4];
    let mut f = [Complex64::new(0.0, 0.0); 4];
    unsafe {
        xw_f_overlap_m0_prepared_c64x4(w, l, x_ex, w_ex, &mut s, &mut f);
    }

    for lane in 0..count {
        let output = outputs[lane];
        let phase = target_phase * phases[lane];
        overlap[output] = s[lane] * phase;
        fock[output] = f[lane] * phase;
    }
}

/// Prepare and evaluate the same-spin overlap and generalised-Fock matrix element between excited
/// determinants generated from the reference pair `\langle{}^x\Psi| and |{}^w\Psi\rangle:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L=m}}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L),`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat F|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+1}\\m_1+\cdots+m_{L+1}=m}}`
/// `[{}^x F_0^{(m_1)}\det\mathbf D_{\mathrm{ov}}(m_2,\ldots,m_{L+1})`
/// `- \sum_{z=1}^{L}\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}`
/// `(m_1,\ldots,m_{L+1})].`
/// `For m = 0 and L = 1,\ldots,4, the contraction determinant is consumed directly while it is`
/// `constructed, avoiding the separate prepare-then-evaluate traversal. For arbitrary L and m > 0,`
/// `the required contraction determinants are constructed once before the constrained distribution sum.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `x_ex`: `Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.`
/// - `w_ex`: `Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.`
/// - `scratch`: Scratch storage for determinant labels, generic contraction determinants and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)`.
#[inline(always)]
fn xw_f_overlap_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap, {
        // The zero-overlap count decides whether the prepared GNME expression is the single
        // all-zero determinant or a constrained sum over mixed `m_i` distributions.
        if w.m == 0 {
            xw_f_overlap_m0_prepared(w, x_ex, w_ex, scratch, tol)
        } else {
            xw_f_overlap_gen_prepared(w, x_ex, w_ex, scratch, tol)
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
fn xw_f_overlap_m0_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0, {
        // For m = 0, the rank L determines the one prepared D_ov determinant and its
        // one-body replacement determinants.
        let l = x_ex.holes.count_ones() as usize + w_ex.holes.count_ones() as usize;

        match l {
            0 => {
                let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
                (pref, pref * w.f0f[0])
            }
            1 => xw_f_overlap_m0_prepared_const::<T, 1>(w, x_ex, w_ex, scratch, tol),
            2 => xw_f_overlap_m0_prepared_const::<T, 2>(w, x_ex, w_ex, scratch, tol),
            3 => xw_f_overlap_m0_prepared_const::<T, 3>(w, x_ex, w_ex, scratch, tol),
            4 => xw_f_overlap_m0_prepared_const::<T, 4>(w, x_ex, w_ex, scratch, tol),
            _ => xw_f_overlap_m0_gen_prepared(w, x_ex, w_ex, scratch, l, tol),
        }
    })
}

/// Dispatch 4 real `m = 0` matrix elements to the fixed-rank AVX2/FMA kernel.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `x_ex`: 4 x-reference excitation caches.
/// - `w_ex`: 4 w-reference excitation caches.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        // Select the AVX2 const-rank kernel for four same-rank one-body/overlap pairs.
        match l {
            1 => xw_f_overlap_m0_prepared_f64x4_const::<T, 1>(w, x_ex, w_ex, overlap, fock),
            2 => xw_f_overlap_m0_prepared_f64x4_const::<T, 2>(w, x_ex, w_ex, overlap, fock),
            3 => xw_f_overlap_m0_prepared_f64x4_const::<T, 3>(w, x_ex, w_ex, overlap, fock),
            4 => xw_f_overlap_m0_prepared_f64x4_const::<T, 4>(w, x_ex, w_ex, overlap, fock),
            _ => unreachable!(),
        }
    }
}

/// Dispatch 8 real `m = 0` matrix elements to the fixed-rank AVX-512 kernel.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `x_ex`: 8 x-reference excitation caches.
/// - `w_ex`: 8 w-reference excitation caches.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        // Select the AVX-512 const-rank kernel for eight same-rank one-body/overlap pairs.
        match l {
            1 => xw_f_overlap_m0_prepared_f64x8_const::<T, 1>(w, x_ex, w_ex, overlap, fock),
            2 => xw_f_overlap_m0_prepared_f64x8_const::<T, 2>(w, x_ex, w_ex, overlap, fock),
            3 => xw_f_overlap_m0_prepared_f64x8_const::<T, 3>(w, x_ex, w_ex, overlap, fock),
            4 => xw_f_overlap_m0_prepared_f64x8_const::<T, 4>(w, x_ex, w_ex, overlap, fock),
            _ => unreachable!(),
        }
    }
}

/// Dispatch 4 complex `m = 0` matrix elements to the fixed-rank AVX2/FMA kernel.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `x_ex`: 4 x-reference excitation caches.
/// - `w_ex`: 4 w-reference excitation caches.
/// - `overlap`: Complex overlap output slice in SIMD-lane order.
/// - `fock`: Complex generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = Complex64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_prepared_c64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [Complex64],
    fock: &mut [Complex64],
) {
    unsafe {
        // Select the AVX2 const-rank kernel for four same-rank complex one-body/overlap pairs.
        match l {
            1 => xw_f_overlap_m0_prepared_c64x4_const::<T, 1>(w, x_ex, w_ex, overlap, fock),
            2 => xw_f_overlap_m0_prepared_c64x4_const::<T, 2>(w, x_ex, w_ex, overlap, fock),
            3 => xw_f_overlap_m0_prepared_c64x4_const::<T, 3>(w, x_ex, w_ex, overlap, fock),
            4 => xw_f_overlap_m0_prepared_c64x4_const::<T, 4>(w, x_ex, w_ex, overlap, fock),
            _ => unreachable!(),
        }
    }
}

/// Dispatch 8 complex `m = 0` matrix elements to the fixed-rank AVX-512 kernel.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `x_ex`: 8 x-reference excitation caches.
/// - `w_ex`: 8 w-reference excitation caches.
/// - `overlap`: Complex overlap output slice in SIMD-lane order.
/// - `fock`: Complex generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = Complex64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_prepared_c64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [Complex64],
    fock: &mut [Complex64],
) {
    unsafe {
        // Select the AVX-512 const-rank kernel for eight same-rank complex one-body/overlap pairs.
        match l {
            1 => xw_f_overlap_m0_prepared_c64x8_const::<T, 1>(w, x_ex, w_ex, overlap, fock),
            2 => xw_f_overlap_m0_prepared_c64x8_const::<T, 2>(w, x_ex, w_ex, overlap, fock),
            3 => xw_f_overlap_m0_prepared_c64x8_const::<T, 3>(w, x_ex, w_ex, overlap, fock),
            4 => xw_f_overlap_m0_prepared_c64x8_const::<T, 4>(w, x_ex, w_ex, overlap, fock),
            _ => unreachable!(),
        }
    }
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
fn xw_f_overlap_m0_prepared_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            scratch.ensure_same(L);

            // Construct the rows r_eta and columns c_z of D_ov from the x- and w-excitation
            // operators in the order used by the GNME determinant.
            construct_determinant_indices(
                x_ex,
                w_ex,
                w,
                scratch.rows.as_mut_slice(),
                scratch.cols.as_mut_slice(),
            );

            // For m = 0, every column uses the m_i = 0 fundamental contractions:
            // D_ov[eta,z] = X^{(0)}_{r_eta c_z} for eta >= z, otherwise Y^{(0)}_{r_eta c_z}.
            let x0 = w.x(0);
            let y0 = w.y(0);
            build_d_const::<T, L>(
                scratch.det0.as_mut_slice(),
                &x0,
                &y0,
                scratch.rows.as_slice(),
                scratch.cols.as_slice(),
            );

            let det0 = &scratch.det0.as_slice()[..L * L];
            let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

            // Evaluate det D_ov and cof[D_ov]_{eta z}=(-1)^{eta+z} det D_ov[eta|z]
            // for the overlap part and the one-body column replacements.
            if let Some(det) = adjugate_transpose_const::<T, L>(
                scratch.adjt_det.as_mut_slice(),
                scratch.invs.as_mut_slice(),
                scratch.lu.as_mut_slice(),
                det0,
                tol,
            ) {
                let n = w.n();
                let fsl = w.ff_t_slice(0, 0);
                let rows = scratch.rows.as_slice();
                let cols = scratch.cols.as_slice();
                let cof = scratch.adjt_det.as_slice();
                let mut repl = <T as From<f64>>::from(0.0);

                // Laplace expansion of the inserted one-body row gives
                // sum_z det D_ov^{z -> F_z} = sum_{eta,z} cof[D_ov]_{eta z} F_{eta z}.
                for z in 0..L {
                    let base = cols[z] * n;

                    for eta in 0..L {
                        repl += cof[eta * L + z] * fsl[base + rows[eta]];
                    }
                }

                // Return S_tilde det D_ov and S_tilde [F_0 det D_ov - C:F].
                (pref * det, pref * (det * w.f0f[0] - repl))
            } else {
                (<T as From<f64>>::from(0.0), <T as From<f64>>::from(0.0))
            }
        }
    )
}

/// Prepare and evaluate 4 independent real fixed-rank `L` overlap and generalised-Fock matrix
/// elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair with
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}`.
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}`
/// `\mathcal F_{\eta z}^{(0,0)}]`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `x_ex`: 4 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 4 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_prepared_f64x4_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            unsafe {
                let n = w.n();
                let x0_t = w.x_slice(0);
                let y0_t = w.y_slice(0);
                let fsl_t = w.ff_t_slice(0, 0);
                let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
                let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
                let fsl = std::slice::from_raw_parts(fsl_t.as_ptr().cast::<f64>(), fsl_t.len());
                let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
                let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
                let pref = phase * w.tilde_s_prod;
                let mut d_lanes = [[0.0f64; 4]; 16];
                let mut f_lanes = [[0.0f64; 4]; 16];

                for lane in 0..4 {
                    let x_data = x_ex.get_unchecked(lane);
                    let w_data = w_ex.get_unchecked(lane);
                    let mut rows = [0usize; L];
                    let mut cols = [0usize; L];

                    construct_determinant_indices_const::<T, L>(
                        x_data.rank,
                        &x_data.indices,
                        &w_data.indices,
                        w,
                        &mut rows,
                        &mut cols,
                    );

                    // Pack D_ov and the one-body replacement column entries for the
                    // same four Wick pairs before evaluating cof[D_ov] lane-wise.
                    for eta in 0..L {
                        let d_base = eta * L;

                        for z in 0..L {
                            let src = rows[eta] * n + cols[z];
                            d_lanes[d_base + z][lane] = if eta >= z { x0[src] } else { y0[src] };
                            f_lanes[d_base + z][lane] = fsl[cols[z] * n + rows[eta]];
                        }
                    }
                }

                let zero_v = F64x4::zero();
                let one_v = F64x4::splat(1.0);

                let (det, repl) = if L == 4 {
                    let dvec = |row: usize, col: usize| F64x4::load(&d_lanes[row * L + col]);
                    let fvec = |row: usize, col: usize| F64x4::load(&f_lanes[row * L + col]);

                    // The old 16 cofactors contain `16 x 3 = 48` minor occurrences. Preserving their exact
                    // expansions leaves `6 + 6 + 6 = 18` distinct minors, so 18 is the lower bound for this DAG.
                    // AVX2 cannot keep all 16 `D_{ij}` plus these intermediates live, so `D_{ij}` is reloaded by group.

                    // `B_{ab} = D_{2a}D_{3b} - D_{2b}D_{3a}` supplies cofactor rows 0 and 1.
                    let (b01, b02, b03, b12, b13, b23) = {
                        let d20 = dvec(2, 0);
                        let d21 = dvec(2, 1);
                        let d22 = dvec(2, 2);
                        let d23 = dvec(2, 3);

                        let d30 = dvec(3, 0);
                        let d31 = dvec(3, 1);

                        let b01 = F64x4::minor(d20, d31, d21, d30);

                        let d32 = dvec(3, 2);

                        let b02 = F64x4::minor(d20, d32, d22, d30);
                        let b12 = F64x4::minor(d21, d32, d22, d31);

                        let d33 = dvec(3, 3);

                        let b03 = F64x4::minor(d20, d33, d23, d30);
                        let b13 = F64x4::minor(d21, d33, d23, d31);
                        let b23 = F64x4::minor(d22, d33, d23, d32);

                        (b01, b02, b03, b12, b13, b23)
                    };

                    // Form `det(\mathbf D) = \sum_j D_{0j}C_{0j}` and row 0 of `C:\mathcal F`.
                    let mut det_v = F64x4::zero();
                    let mut repl0 = F64x4::zero();

                    {
                        let d10 = dvec(1, 0);
                        let d11 = dvec(1, 1);
                        let d12 = dvec(1, 2);
                        let d13 = dvec(1, 3);

                        let cof00 = F64x4::cof_pos(d11, b23, d12, b13, d13, b12);
                        det_v = F64x4::madd(det_v, dvec(0, 0), cof00);
                        repl0 = F64x4::madd(repl0, fvec(0, 0), cof00);

                        let cof01 = F64x4::cof_neg(d10, b23, d12, b03, d13, b02);
                        det_v = F64x4::madd(det_v, dvec(0, 1), cof01);
                        repl0 = F64x4::madd(repl0, fvec(0, 1), cof01);

                        let cof02 = F64x4::cof_pos(d10, b13, d11, b03, d13, b01);
                        det_v = F64x4::madd(det_v, dvec(0, 2), cof02);
                        repl0 = F64x4::madd(repl0, fvec(0, 2), cof02);

                        let cof03 = F64x4::cof_neg(d10, b12, d11, b02, d12, b01);
                        det_v = F64x4::madd(det_v, dvec(0, 3), cof03);
                        repl0 = F64x4::madd(repl0, fvec(0, 3), cof03);
                    }

                    // Store `det(\mathbf D)` at its first natural endpoint so it does not remain live across all cofactors.
                    let mut det_lane = [0.0f64; 4];
                    det_v.store(&mut det_lane);

                    // Reuse the same six `B_{ab}` values for row 1 of `C:\mathcal F`.
                    let mut repl1 = F64x4::zero();

                    {
                        let d00 = dvec(0, 0);
                        let d01 = dvec(0, 1);
                        let d02 = dvec(0, 2);
                        let d03 = dvec(0, 3);

                        let cof10 = F64x4::cof_neg(d01, b23, d02, b13, d03, b12);
                        repl1 = F64x4::madd(repl1, fvec(1, 0), cof10);

                        let cof11 = F64x4::cof_pos(d00, b23, d02, b03, d03, b02);
                        repl1 = F64x4::madd(repl1, fvec(1, 1), cof11);

                        let cof12 = F64x4::cof_neg(d00, b13, d01, b03, d03, b01);
                        repl1 = F64x4::madd(repl1, fvec(1, 2), cof12);

                        let cof13 = F64x4::cof_pos(d00, b12, d01, b02, d02, b01);
                        repl1 = F64x4::madd(repl1, fvec(1, 3), cof13);
                    }

                    let repl01 = F64x4::add(repl0, repl1);

                    // `Q_{ab} = D_{1a}D_{3b} - D_{1b}D_{3a}` supplies cofactor row 2.
                    let (q01, q02, q03, q12, q13, q23) = {
                        let d10 = dvec(1, 0);
                        let d11 = dvec(1, 1);
                        let d12 = dvec(1, 2);
                        let d13 = dvec(1, 3);

                        let d30 = dvec(3, 0);
                        let d31 = dvec(3, 1);

                        let q01 = F64x4::minor(d10, d31, d11, d30);

                        let d32 = dvec(3, 2);

                        let q02 = F64x4::minor(d10, d32, d12, d30);
                        let q12 = F64x4::minor(d11, d32, d12, d31);

                        let d33 = dvec(3, 3);

                        let q03 = F64x4::minor(d10, d33, d13, d30);
                        let q13 = F64x4::minor(d11, d33, d13, d31);
                        let q23 = F64x4::minor(d12, d33, d13, d32);

                        (q01, q02, q03, q12, q13, q23)
                    };

                    let mut repl2 = F64x4::zero();

                    {
                        let d00 = dvec(0, 0);
                        let d01 = dvec(0, 1);
                        let d02 = dvec(0, 2);
                        let d03 = dvec(0, 3);

                        let cof20 = F64x4::cof_pos(d01, q23, d02, q13, d03, q12);
                        repl2 = F64x4::madd(repl2, fvec(2, 0), cof20);

                        let cof21 = F64x4::cof_neg(d00, q23, d02, q03, d03, q02);
                        repl2 = F64x4::madd(repl2, fvec(2, 1), cof21);

                        let cof22 = F64x4::cof_pos(d00, q13, d01, q03, d03, q01);
                        repl2 = F64x4::madd(repl2, fvec(2, 2), cof22);

                        let cof23 = F64x4::cof_neg(d00, q12, d01, q02, d02, q01);
                        repl2 = F64x4::madd(repl2, fvec(2, 3), cof23);
                    }

                    // `R_{ab} = D_{1a}D_{2b} - D_{1b}D_{2a}` supplies cofactor row 3.
                    let (r01, r02, r03, r12, r13, r23) = {
                        let d10 = dvec(1, 0);
                        let d11 = dvec(1, 1);
                        let d12 = dvec(1, 2);
                        let d13 = dvec(1, 3);

                        let d20 = dvec(2, 0);
                        let d21 = dvec(2, 1);

                        let r01 = F64x4::minor(d10, d21, d11, d20);

                        let d22 = dvec(2, 2);

                        let r02 = F64x4::minor(d10, d22, d12, d20);
                        let r12 = F64x4::minor(d11, d22, d12, d21);

                        let d23 = dvec(2, 3);

                        let r03 = F64x4::minor(d10, d23, d13, d20);
                        let r13 = F64x4::minor(d11, d23, d13, d21);
                        let r23 = F64x4::minor(d12, d23, d13, d22);

                        (r01, r02, r03, r12, r13, r23)
                    };

                    let mut repl3 = F64x4::zero();

                    {
                        let d00 = dvec(0, 0);
                        let d01 = dvec(0, 1);
                        let d02 = dvec(0, 2);
                        let d03 = dvec(0, 3);

                        let cof30 = F64x4::cof_neg(d01, r23, d02, r13, d03, r12);
                        repl3 = F64x4::madd(repl3, fvec(3, 0), cof30);

                        let cof31 = F64x4::cof_pos(d00, r23, d02, r03, d03, r02);
                        repl3 = F64x4::madd(repl3, fvec(3, 1), cof31);

                        let cof32 = F64x4::cof_neg(d00, r13, d01, r03, d03, r01);
                        repl3 = F64x4::madd(repl3, fvec(3, 2), cof32);

                        let cof33 = F64x4::cof_pos(d00, r12, d01, r02, d02, r01);
                        repl3 = F64x4::madd(repl3, fvec(3, 3), cof33);
                    }

                    // Preserve the previous contraction tree `((repl0 + repl1) + (repl2 + repl3))`.
                    let repl23 = F64x4::add(repl2, repl3);
                    let repl_v = F64x4::add(repl01, repl23);
                    let det_v = F64x4::load(&det_lane);

                    (det_v, repl_v)
                } else {
                    let mut d = [F64x4::zero(); 16];
                    let mut ff = [F64x4::zero(); 16];
                    for idx in 0..L * L {
                        d[idx] = F64x4::load(&d_lanes[idx]);
                        ff[idx] = F64x4::load(&f_lanes[idx]);
                    }

                    let mut cof = [zero_v; 16];
                    let det;

                    if L == 1 {
                        cof[0] = one_v;
                        det = d[0];
                    } else if L == 2 {
                        cof[0] = d[3];
                        cof[1] = F64x4::sub(zero_v, d[2]);
                        cof[2] = F64x4::sub(zero_v, d[1]);
                        cof[3] = d[0];
                        det = F64x4::minor(d[0], d[3], d[1], d[2]);
                    } else {
                        for eta in 0..L {
                            let mut rows_keep = [0usize; 2];
                            let mut ri = 0usize;
                            for r in 0..L {
                                if r != eta {
                                    rows_keep[ri] = r;
                                    ri += 1;
                                }
                            }

                            for z in 0..L {
                                let mut cols_keep = [0usize; 2];
                                let mut ci = 0usize;
                                for c in 0..L {
                                    if c != z {
                                        cols_keep[ci] = c;
                                        ci += 1;
                                    }
                                }

                                let value = F64x4::minor(
                                    d[rows_keep[0] * L + cols_keep[0]],
                                    d[rows_keep[1] * L + cols_keep[1]],
                                    d[rows_keep[0] * L + cols_keep[1]],
                                    d[rows_keep[1] * L + cols_keep[0]],
                                );
                                cof[eta * L + z] = if ((eta + z) & 1) == 0 {
                                    value
                                } else {
                                    F64x4::sub(zero_v, value)
                                };
                            }
                        }

                        let mut det_acc = F64x4::mul(d[0], cof[0]);
                        for z in 1..L {
                            det_acc = F64x4::madd(det_acc, d[z], cof[z]);
                        }
                        det = det_acc;
                    }

                    let mut repl = F64x4::zero();
                    for eta in 0..L {
                        for z in 0..L {
                            repl = F64x4::madd(repl, cof[eta * L + z], ff[eta * L + z]);
                        }
                    }

                    (det, repl)
                };

                let pref_v = F64x4::splat(pref);
                let f0_v = F64x4::splat(f0);

                let overlap_v = F64x4::mul(det, pref_v);
                let fock_v = F64x4::mul(F64x4::mul_sub(det, f0_v, repl), pref_v);

                let mut det_lane = [0.0f64; 4];
                let mut overlap_lane = [0.0f64; 4];
                let mut fock_lane = [0.0f64; 4];
                det.store(&mut det_lane);
                overlap_v.store(&mut overlap_lane);
                fock_v.store(&mut fock_lane);

                // Store four packed overlap and one-body lanes, zeroing non-finite determinant lanes.
                for lane in 0..4 {
                    if det_lane[lane].is_finite() {
                        overlap[lane] = overlap_lane[lane];
                        fock[lane] = fock_lane[lane];
                    } else {
                        overlap[lane] = 0.0;
                        fock[lane] = 0.0;
                    }
                }
            }
        }
    )
}

/// Prepare and evaluate 4 independent complex fixed-rank `L` overlap and generalised-Fock matrix
/// elements for `m = 0`.
/// Each SIMD lane evaluates `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}` and
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}} - C:\mathcal F]`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `x_ex`: 4 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 4 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Complex overlap output slice in SIMD-lane order.
/// - `fock`: Complex generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = Complex64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_prepared_c64x4_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [Complex64],
    fock: &mut [Complex64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            unsafe {
                let n = w.n();
                let x0_t = w.x_slice(0);
                let y0_t = w.y_slice(0);
                let fsl_t = w.ff_t_slice(0, 0);
                let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<Complex64>(), x0_t.len());
                let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<Complex64>(), y0_t.len());
                let fsl =
                    std::slice::from_raw_parts(fsl_t.as_ptr().cast::<Complex64>(), fsl_t.len());
                let phase = *std::ptr::from_ref(&w.phase).cast::<Complex64>();
                let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<Complex64>();
                let pref = phase * w.tilde_s_prod;
                let mut d_re = [[0.0f64; 4]; 16];
                let mut d_im = [[0.0f64; 4]; 16];
                let mut f_re = [[0.0f64; 4]; 16];
                let mut f_im = [[0.0f64; 4]; 16];

                for lane in 0..4 {
                    let x_data = x_ex.get_unchecked(lane);
                    let w_data = w_ex.get_unchecked(lane);
                    let mut rows = [0usize; L];
                    let mut cols = [0usize; L];

                    construct_determinant_indices_const::<T, L>(
                        x_data.rank,
                        &x_data.indices,
                        &w_data.indices,
                        w,
                        &mut rows,
                        &mut cols,
                    );

                    // Pack D_ov and the one-body replacement column entries for the
                    // same four Wick pairs before evaluating cof[D_ov] lane-wise.
                    for eta in 0..L {
                        let d_base = eta * L;

                        for z in 0..L {
                            let src = rows[eta] * n + cols[z];
                            let d_value = if eta >= z { x0[src] } else { y0[src] };
                            let f_value = fsl[cols[z] * n + rows[eta]];
                            d_re[d_base + z][lane] = d_value.re;
                            d_im[d_base + z][lane] = d_value.im;
                            f_re[d_base + z][lane] = f_value.re;
                            f_im[d_base + z][lane] = f_value.im;
                        }
                    }
                }

                let dvec = |row: usize, col: usize| {
                    C64x4::load(&d_re[row * L + col], &d_im[row * L + col])
                };
                let fvec = |row: usize, col: usize| {
                    C64x4::load(&f_re[row * L + col], &f_im[row * L + col])
                };
                let zero = C64x4::zero();
                let (det, repl) = if L == 1 {
                    // For L = 1, det D_ov is the sole contraction and the replacement is F_00.
                    (dvec(0, 0), fvec(0, 0))
                } else if L == 2 {
                    // For L = 2, use the explicit determinant and one-column replacement formula:
                    // `det = D00 D11 - D01 D10`.
                    // `repl = F00 D11 - F01 D10 - F10 D01 + F11 D00`.
                    let d00 = dvec(0, 0);
                    let d01 = dvec(0, 1);
                    let d10 = dvec(1, 0);
                    let d11 = dvec(1, 1);
                    let f00 = fvec(0, 0);
                    let f01 = fvec(0, 1);
                    let f10 = fvec(1, 0);
                    let f11 = fvec(1, 1);
                    let det = C64x4::minor(d00, d11, d01, d10);
                    let mut repl = C64x4::mul(f00, d11);
                    repl = C64x4::msub(repl, f01, d10);
                    repl = C64x4::msub(repl, f10, d01);
                    repl = C64x4::madd(repl, f11, d00);
                    (det, repl)
                } else if L == 3 {
                    // Calculate the nine distinct cofactors exactly once, then form
                    // det D_ov through row 0 and contract every cofactor with `\mathcal F`.
                    let d00 = dvec(0, 0);
                    let d01 = dvec(0, 1);
                    let d02 = dvec(0, 2);
                    let d10 = dvec(1, 0);
                    let d11 = dvec(1, 1);
                    let d12 = dvec(1, 2);
                    let d20 = dvec(2, 0);
                    let d21 = dvec(2, 1);
                    let d22 = dvec(2, 2);
                    let c00 = C64x4::minor(d11, d22, d12, d21);
                    let c01 = C64x4::sub(zero, C64x4::minor(d10, d22, d12, d20));
                    let c02 = C64x4::minor(d10, d21, d11, d20);
                    let c10 = C64x4::sub(zero, C64x4::minor(d01, d22, d02, d21));
                    let c11 = C64x4::minor(d00, d22, d02, d20);
                    let c12 = C64x4::sub(zero, C64x4::minor(d00, d21, d01, d20));
                    let c20 = C64x4::minor(d01, d12, d02, d11);
                    let c21 = C64x4::sub(zero, C64x4::minor(d00, d12, d02, d10));
                    let c22 = C64x4::minor(d00, d11, d01, d10);
                    let mut det = C64x4::mul(d00, c00);
                    det = C64x4::madd(det, d01, c01);
                    det = C64x4::madd(det, d02, c02);
                    let cof = [c00, c01, c02, c10, c11, c12, c20, c21, c22];
                    let mut repl = C64x4::zero();
                    for eta in 0..3 {
                        for z in 0..3 {
                            repl = C64x4::madd(repl, fvec(eta, z), cof[eta * 3 + z]);
                        }
                    }
                    (det, repl)
                } else {
                    let (det, repl) = c64x4_l4_dag::<L>(&dvec, &fvec);
                    (det, repl)
                };

                let pref_v = C64x4::splat(pref.re, pref.im);
                let f0_v = C64x4::splat(f0.re, f0.im);

                let overlap_v = C64x4::mul(pref_v, det);
                let fock_v = C64x4::mul(pref_v, C64x4::mul_sub(f0_v, det, repl));

                let mut det_re = [0.0f64; 4];
                let mut det_im = [0.0f64; 4];
                let mut overlap_re = [0.0f64; 4];
                let mut overlap_im = [0.0f64; 4];
                let mut fock_re = [0.0f64; 4];
                let mut fock_im = [0.0f64; 4];
                det.store(&mut det_re, &mut det_im);
                overlap_v.store(&mut overlap_re, &mut overlap_im);
                fock_v.store(&mut fock_re, &mut fock_im);

                // Store four packed overlap and one-body lanes, zeroing non-finite determinant lanes.
                for lane in 0..4 {
                    if det_re[lane].is_finite() && det_im[lane].is_finite() {
                        overlap[lane] = Complex64::new(overlap_re[lane], overlap_im[lane]);
                        fock[lane] = Complex64::new(fock_re[lane], fock_im[lane]);
                    } else {
                        overlap[lane] = Complex64::new(0.0, 0.0);
                        fock[lane] = Complex64::new(0.0, 0.0);
                    }
                }
            }
        }
    )
}

/// Evaluate the complex `L = 4` one-body cofactor DAG for 4 SIMD lanes.
/// This preserves the 18 distinct `2 x 2` minors used by the real AVX2 kernel:
/// row-pair groups `B = (2,3)`, `Q = (1,3)` and `R = (1,2)`.
/// # Arguments:
/// - `dvec`: Loader for packed contraction determinant entries.
/// - `fvec`: Loader for packed one-body replacement entries.
/// # Returns
/// - `(C64x4, C64x4)`: Packed determinant and `C:\mathcal F` replacement contraction.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn c64x4_l4_dag<const L: usize>(
    dvec: &impl Fn(usize, usize) -> C64x4,
    fvec: &impl Fn(usize, usize) -> C64x4,
) -> (C64x4, C64x4) {
    // The old 16 cofactors contain `16 x 3 = 48` minor occurrences. Preserving their exact
    // expansions leaves `6 + 6 + 6 = 18` distinct minors, so 18 is the lower bound for this DAG.
    // Complex SIMD values use two registers, so `D_{ij}` is loaded on demand by row-pair group.

    // `B_{ab} = D_{2a}D_{3b} - D_{2b}D_{3a}` supplies cofactor rows 0 and 1.
    let (b01, b02, b03, b12, b13, b23) = {
        let d20 = dvec(2, 0);
        let d21 = dvec(2, 1);
        let d22 = dvec(2, 2);
        let d23 = dvec(2, 3);
        let d30 = dvec(3, 0);
        let d31 = dvec(3, 1);
        let d32 = dvec(3, 2);
        let d33 = dvec(3, 3);
        (
            C64x4::minor(d20, d31, d21, d30),
            C64x4::minor(d20, d32, d22, d30),
            C64x4::minor(d20, d33, d23, d30),
            C64x4::minor(d21, d32, d22, d31),
            C64x4::minor(d21, d33, d23, d31),
            C64x4::minor(d22, d33, d23, d32),
        )
    };

    let mut det = C64x4::zero();
    let mut repl0 = C64x4::zero();
    let mut repl1 = C64x4::zero();

    // Form `det(\mathbf D) = \sum_j D_{0j}C_{0j}` and row 0 of `C:\mathcal F`.
    {
        let d10 = dvec(1, 0);
        let d11 = dvec(1, 1);
        let d12 = dvec(1, 2);
        let d13 = dvec(1, 3);

        let cof00 = C64x4::cof_pos(d11, b23, d12, b13, d13, b12);
        det = C64x4::madd(det, dvec(0, 0), cof00);
        repl0 = C64x4::madd(repl0, fvec(0, 0), cof00);

        let cof01 = C64x4::cof_neg(d10, b23, d12, b03, d13, b02);
        det = C64x4::madd(det, dvec(0, 1), cof01);
        repl0 = C64x4::madd(repl0, fvec(0, 1), cof01);

        let cof02 = C64x4::cof_pos(d10, b13, d11, b03, d13, b01);
        det = C64x4::madd(det, dvec(0, 2), cof02);
        repl0 = C64x4::madd(repl0, fvec(0, 2), cof02);

        let cof03 = C64x4::cof_neg(d10, b12, d11, b02, d12, b01);
        det = C64x4::madd(det, dvec(0, 3), cof03);
        repl0 = C64x4::madd(repl0, fvec(0, 3), cof03);
    }

    // Reuse the same six `B_{ab}` values for row 1 of `C:\mathcal F`.
    {
        let d00 = dvec(0, 0);
        let d01 = dvec(0, 1);
        let d02 = dvec(0, 2);
        let d03 = dvec(0, 3);

        let cof10 = C64x4::cof_neg(d01, b23, d02, b13, d03, b12);
        repl1 = C64x4::madd(repl1, fvec(1, 0), cof10);

        let cof11 = C64x4::cof_pos(d00, b23, d02, b03, d03, b02);
        repl1 = C64x4::madd(repl1, fvec(1, 1), cof11);

        let cof12 = C64x4::cof_neg(d00, b13, d01, b03, d03, b01);
        repl1 = C64x4::madd(repl1, fvec(1, 2), cof12);

        let cof13 = C64x4::cof_pos(d00, b12, d01, b02, d02, b01);
        repl1 = C64x4::madd(repl1, fvec(1, 3), cof13);
    }

    let repl01 = C64x4::add(repl0, repl1);

    // `Q_{ab} = D_{1a}D_{3b} - D_{1b}D_{3a}` supplies cofactor row 2.
    let (q01, q02, q03, q12, q13, q23) = {
        let d10 = dvec(1, 0);
        let d11 = dvec(1, 1);
        let d12 = dvec(1, 2);
        let d13 = dvec(1, 3);
        let d30 = dvec(3, 0);
        let d31 = dvec(3, 1);
        let d32 = dvec(3, 2);
        let d33 = dvec(3, 3);
        (
            C64x4::minor(d10, d31, d11, d30),
            C64x4::minor(d10, d32, d12, d30),
            C64x4::minor(d10, d33, d13, d30),
            C64x4::minor(d11, d32, d12, d31),
            C64x4::minor(d11, d33, d13, d31),
            C64x4::minor(d12, d33, d13, d32),
        )
    };

    let mut repl2 = C64x4::zero();

    {
        let d00 = dvec(0, 0);
        let d01 = dvec(0, 1);
        let d02 = dvec(0, 2);
        let d03 = dvec(0, 3);

        let cof20 = C64x4::cof_pos(d01, q23, d02, q13, d03, q12);
        repl2 = C64x4::madd(repl2, fvec(2, 0), cof20);

        let cof21 = C64x4::cof_neg(d00, q23, d02, q03, d03, q02);
        repl2 = C64x4::madd(repl2, fvec(2, 1), cof21);

        let cof22 = C64x4::cof_pos(d00, q13, d01, q03, d03, q01);
        repl2 = C64x4::madd(repl2, fvec(2, 2), cof22);

        let cof23 = C64x4::cof_neg(d00, q12, d01, q02, d02, q01);
        repl2 = C64x4::madd(repl2, fvec(2, 3), cof23);
    }

    // `R_{ab} = D_{1a}D_{2b} - D_{1b}D_{2a}` supplies cofactor row 3.
    let (r01, r02, r03, r12, r13, r23) = {
        let d10 = dvec(1, 0);
        let d11 = dvec(1, 1);
        let d12 = dvec(1, 2);
        let d13 = dvec(1, 3);
        let d20 = dvec(2, 0);
        let d21 = dvec(2, 1);
        let d22 = dvec(2, 2);
        let d23 = dvec(2, 3);
        (
            C64x4::minor(d10, d21, d11, d20),
            C64x4::minor(d10, d22, d12, d20),
            C64x4::minor(d10, d23, d13, d20),
            C64x4::minor(d11, d22, d12, d21),
            C64x4::minor(d11, d23, d13, d21),
            C64x4::minor(d12, d23, d13, d22),
        )
    };

    let mut repl3 = C64x4::zero();

    {
        let d00 = dvec(0, 0);
        let d01 = dvec(0, 1);
        let d02 = dvec(0, 2);
        let d03 = dvec(0, 3);

        let cof30 = C64x4::cof_neg(d01, r23, d02, r13, d03, r12);
        repl3 = C64x4::madd(repl3, fvec(3, 0), cof30);

        let cof31 = C64x4::cof_pos(d00, r23, d02, r03, d03, r02);
        repl3 = C64x4::madd(repl3, fvec(3, 1), cof31);

        let cof32 = C64x4::cof_neg(d00, r13, d01, r03, d03, r01);
        repl3 = C64x4::madd(repl3, fvec(3, 2), cof32);

        let cof33 = C64x4::cof_pos(d00, r12, d01, r02, d02, r01);
        repl3 = C64x4::madd(repl3, fvec(3, 3), cof33);
    }

    // Preserve the previous contraction tree `((repl0 + repl1) + (repl2 + repl3))`.
    let repl23 = C64x4::add(repl2, repl3);
    (det, C64x4::add(repl01, repl23))
}

/// Prepare and evaluate 8 independent real fixed-rank `L` overlap and generalised-Fock matrix
/// elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair with
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}`.
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}`
/// `\mathcal F_{\eta z}^{(0,0)}]`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `x_ex`: 8 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 8 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512F`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_prepared_f64x8_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            unsafe {
                let n = w.n();
                let x0_t = w.x_slice(0);
                let y0_t = w.y_slice(0);
                let fsl_t = w.ff_t_slice(0, 0);
                let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
                let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
                let fsl = std::slice::from_raw_parts(fsl_t.as_ptr().cast::<f64>(), fsl_t.len());
                let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
                let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
                let pref = phase * w.tilde_s_prod;
                let mut d_lanes = [[0.0f64; 8]; 16];
                let mut f_lanes = [[0.0f64; 8]; 16];

                for lane in 0..8 {
                    let x_data = x_ex.get_unchecked(lane);
                    let w_data = w_ex.get_unchecked(lane);
                    let mut rows = [0usize; L];
                    let mut cols = [0usize; L];

                    construct_determinant_indices_const::<T, L>(
                        x_data.rank,
                        &x_data.indices,
                        &w_data.indices,
                        w,
                        &mut rows,
                        &mut cols,
                    );

                    // Pack D_ov and the one-body replacement column entries for the
                    // same eight Wick pairs before evaluating cof[D_ov] lane-wise.
                    for eta in 0..L {
                        let d_base = eta * L;

                        for z in 0..L {
                            let src = rows[eta] * n + cols[z];
                            d_lanes[d_base + z][lane] = if eta >= z { x0[src] } else { y0[src] };
                            f_lanes[d_base + z][lane] = fsl[cols[z] * n + rows[eta]];
                        }
                    }
                }

                let zero_v = F64x8::zero();
                let one_v = F64x8::splat(1.0);
                let mut d = [F64x8::zero(); 16];
                let mut ff = [F64x8::zero(); 16];
                for idx in 0..L * L {
                    d[idx] = F64x8::load(&d_lanes[idx]);
                    ff[idx] = F64x8::load(&f_lanes[idx]);
                }

                let (det, repl) = if L == 4 {
                    let fvec = |row: usize, col: usize| ff[row * L + col];

                    // Preserving the explicit `L = 4` cofactor DAG reduces 48 minor occurrences to exactly
                    // `3 binom(4,2) = 18` distinct minors. Each is required, so 18 is the lower bound for this DAG.
                    // AVX-512 has 32 ZMM registers, allowing the 16 distinct `D_{ij}` inputs to remain resident.

                    let d00 = d[0];
                    let d01 = d[1];
                    let d02 = d[2];
                    let d03 = d[3];

                    let d10 = d[4];
                    let d11 = d[5];
                    let d12 = d[6];
                    let d13 = d[7];

                    let d20 = d[8];
                    let d21 = d[9];
                    let d22 = d[10];
                    let d23 = d[11];

                    let d30 = d[12];
                    let d31 = d[13];
                    let d32 = d[14];
                    let d33 = d[15];

                    // `B_{ab}` contains the six row-pair `(2,3)` minors used by cofactor rows 0 and 1.
                    let b01 = F64x8::minor(d20, d31, d21, d30);
                    let b02 = F64x8::minor(d20, d32, d22, d30);
                    let b03 = F64x8::minor(d20, d33, d23, d30);
                    let b12 = F64x8::minor(d21, d32, d22, d31);
                    let b13 = F64x8::minor(d21, d33, d23, d31);
                    let b23 = F64x8::minor(d22, d33, d23, d32);

                    let mut det_v = F64x8::zero();
                    let mut repl0 = F64x8::zero();
                    let mut repl1 = F64x8::zero();

                    // Form `det(\mathbf D)` through row 0 while contracting cofactor row 0 with `\mathcal F`.
                    let cof00 = F64x8::cof_pos(d11, b23, d12, b13, d13, b12);
                    det_v = F64x8::madd(det_v, d00, cof00);
                    repl0 = F64x8::madd(repl0, fvec(0, 0), cof00);

                    let cof01 = F64x8::cof_neg(d10, b23, d12, b03, d13, b02);
                    det_v = F64x8::madd(det_v, d01, cof01);
                    repl0 = F64x8::madd(repl0, fvec(0, 1), cof01);

                    let cof02 = F64x8::cof_pos(d10, b13, d11, b03, d13, b01);
                    det_v = F64x8::madd(det_v, d02, cof02);
                    repl0 = F64x8::madd(repl0, fvec(0, 2), cof02);

                    let cof03 = F64x8::cof_neg(d10, b12, d11, b02, d12, b01);
                    det_v = F64x8::madd(det_v, d03, cof03);
                    repl0 = F64x8::madd(repl0, fvec(0, 3), cof03);

                    // The same `B_{ab}` values give cofactor row 1 without any further minor evaluation.
                    let cof10 = F64x8::cof_neg(d01, b23, d02, b13, d03, b12);
                    repl1 = F64x8::madd(repl1, fvec(1, 0), cof10);

                    let cof11 = F64x8::cof_pos(d00, b23, d02, b03, d03, b02);
                    repl1 = F64x8::madd(repl1, fvec(1, 1), cof11);

                    let cof12 = F64x8::cof_neg(d00, b13, d01, b03, d03, b01);
                    repl1 = F64x8::madd(repl1, fvec(1, 2), cof12);

                    let cof13 = F64x8::cof_pos(d00, b12, d01, b02, d02, b01);
                    repl1 = F64x8::madd(repl1, fvec(1, 3), cof13);

                    let repl01 = F64x8::add(repl0, repl1);

                    // `Q_{ab}` contains the six row-pair `(1,3)` minors required by cofactor row 2.
                    let q01 = F64x8::minor(d10, d31, d11, d30);
                    let q02 = F64x8::minor(d10, d32, d12, d30);
                    let q03 = F64x8::minor(d10, d33, d13, d30);
                    let q12 = F64x8::minor(d11, d32, d12, d31);
                    let q13 = F64x8::minor(d11, d33, d13, d31);
                    let q23 = F64x8::minor(d12, d33, d13, d32);

                    let mut repl2 = F64x8::zero();

                    let cof20 = F64x8::cof_pos(d01, q23, d02, q13, d03, q12);
                    repl2 = F64x8::madd(repl2, fvec(2, 0), cof20);

                    let cof21 = F64x8::cof_neg(d00, q23, d02, q03, d03, q02);
                    repl2 = F64x8::madd(repl2, fvec(2, 1), cof21);

                    let cof22 = F64x8::cof_pos(d00, q13, d01, q03, d03, q01);
                    repl2 = F64x8::madd(repl2, fvec(2, 2), cof22);

                    let cof23 = F64x8::cof_neg(d00, q12, d01, q02, d02, q01);
                    repl2 = F64x8::madd(repl2, fvec(2, 3), cof23);

                    // `R_{ab}` contains the final six row-pair `(1,2)` minors required by cofactor row 3.
                    let r01 = F64x8::minor(d10, d21, d11, d20);
                    let r02 = F64x8::minor(d10, d22, d12, d20);
                    let r03 = F64x8::minor(d10, d23, d13, d20);
                    let r12 = F64x8::minor(d11, d22, d12, d21);
                    let r13 = F64x8::minor(d11, d23, d13, d21);
                    let r23 = F64x8::minor(d12, d23, d13, d22);

                    let mut repl3 = F64x8::zero();

                    let cof30 = F64x8::cof_neg(d01, r23, d02, r13, d03, r12);
                    repl3 = F64x8::madd(repl3, fvec(3, 0), cof30);

                    let cof31 = F64x8::cof_pos(d00, r23, d02, r03, d03, r02);
                    repl3 = F64x8::madd(repl3, fvec(3, 1), cof31);

                    let cof32 = F64x8::cof_neg(d00, r13, d01, r03, d03, r01);
                    repl3 = F64x8::madd(repl3, fvec(3, 2), cof32);

                    let cof33 = F64x8::cof_pos(d00, r12, d01, r02, d02, r01);
                    repl3 = F64x8::madd(repl3, fvec(3, 3), cof33);

                    // Preserve `C:\mathcal F = (repl0 + repl1) + (repl2 + repl3)` from the old kernel.
                    let repl23 = F64x8::add(repl2, repl3);
                    let repl_v = F64x8::add(repl01, repl23);

                    (det_v, repl_v)
                } else {
                    let mut cof = [zero_v; 16];
                    let det;

                    if L == 1 {
                        cof[0] = one_v;
                        det = d[0];
                    } else if L == 2 {
                        cof[0] = d[3];
                        cof[1] = F64x8::sub(zero_v, d[2]);
                        cof[2] = F64x8::sub(zero_v, d[1]);
                        cof[3] = d[0];
                        det = F64x8::minor(d[0], d[3], d[1], d[2]);
                    } else {
                        for eta in 0..L {
                            let mut rows_keep = [0usize; 2];
                            let mut ri = 0usize;
                            for r in 0..L {
                                if r != eta {
                                    rows_keep[ri] = r;
                                    ri += 1;
                                }
                            }

                            for z in 0..L {
                                let mut cols_keep = [0usize; 2];
                                let mut ci = 0usize;
                                for c in 0..L {
                                    if c != z {
                                        cols_keep[ci] = c;
                                        ci += 1;
                                    }
                                }

                                let value = F64x8::minor(
                                    d[rows_keep[0] * L + cols_keep[0]],
                                    d[rows_keep[1] * L + cols_keep[1]],
                                    d[rows_keep[0] * L + cols_keep[1]],
                                    d[rows_keep[1] * L + cols_keep[0]],
                                );
                                cof[eta * L + z] = if ((eta + z) & 1) == 0 {
                                    value
                                } else {
                                    F64x8::sub(zero_v, value)
                                };
                            }
                        }

                        let mut det_acc = F64x8::mul(d[0], cof[0]);
                        for z in 1..L {
                            det_acc = F64x8::madd(det_acc, d[z], cof[z]);
                        }
                        det = det_acc;
                    }

                    let mut repl = F64x8::zero();
                    for eta in 0..L {
                        for z in 0..L {
                            repl = F64x8::madd(repl, cof[eta * L + z], ff[eta * L + z]);
                        }
                    }

                    (det, repl)
                };

                let pref_v = F64x8::splat(pref);
                let f0_v = F64x8::splat(f0);

                let overlap_v = F64x8::mul(det, pref_v);
                let fock_v = F64x8::mul(F64x8::mul_sub(det, f0_v, repl), pref_v);

                let mut det_lane = [0.0f64; 8];
                let mut overlap_lane = [0.0f64; 8];
                let mut fock_lane = [0.0f64; 8];
                det.store(&mut det_lane);
                overlap_v.store(&mut overlap_lane);
                fock_v.store(&mut fock_lane);

                // Store eight packed overlap and one-body lanes, zeroing non-finite determinant lanes.
                for lane in 0..8 {
                    if det_lane[lane].is_finite() {
                        overlap[lane] = overlap_lane[lane];
                        fock[lane] = fock_lane[lane];
                    } else {
                        overlap[lane] = 0.0;
                        fock[lane] = 0.0;
                    }
                }
            }
        }
    )
}

/// Prepare and evaluate 8 independent complex fixed-rank `L` overlap and generalised-Fock matrix
/// elements for `m = 0`.
/// Each SIMD lane evaluates `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}` and
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}} - C:\mathcal F]`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `x_ex`: 8 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 8 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Complex overlap output slice in SIMD-lane order.
/// - `fock`: Complex generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = Complex64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_prepared_c64x8_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [Complex64],
    fock: &mut [Complex64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            unsafe {
                let n = w.n();
                let x0_t = w.x_slice(0);
                let y0_t = w.y_slice(0);
                let fsl_t = w.ff_t_slice(0, 0);
                let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<Complex64>(), x0_t.len());
                let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<Complex64>(), y0_t.len());
                let fsl =
                    std::slice::from_raw_parts(fsl_t.as_ptr().cast::<Complex64>(), fsl_t.len());
                let phase = *std::ptr::from_ref(&w.phase).cast::<Complex64>();
                let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<Complex64>();
                let pref = phase * w.tilde_s_prod;
                let mut d_re = [[0.0f64; 8]; 16];
                let mut d_im = [[0.0f64; 8]; 16];
                let mut f_re = [[0.0f64; 8]; 16];
                let mut f_im = [[0.0f64; 8]; 16];

                for lane in 0..8 {
                    let x_data = x_ex.get_unchecked(lane);
                    let w_data = w_ex.get_unchecked(lane);
                    let mut rows = [0usize; L];
                    let mut cols = [0usize; L];

                    construct_determinant_indices_const::<T, L>(
                        x_data.rank,
                        &x_data.indices,
                        &w_data.indices,
                        w,
                        &mut rows,
                        &mut cols,
                    );

                    // Pack D_ov and the one-body replacement column entries for the
                    // same eight Wick pairs before evaluating cof[D_ov] lane-wise.
                    for eta in 0..L {
                        let d_base = eta * L;

                        for z in 0..L {
                            let src = rows[eta] * n + cols[z];
                            let d_value = if eta >= z { x0[src] } else { y0[src] };
                            let f_value = fsl[cols[z] * n + rows[eta]];
                            d_re[d_base + z][lane] = d_value.re;
                            d_im[d_base + z][lane] = d_value.im;
                            f_re[d_base + z][lane] = f_value.re;
                            f_im[d_base + z][lane] = f_value.im;
                        }
                    }
                }

                let dvec = |row: usize, col: usize| {
                    C64x8::load(&d_re[row * L + col], &d_im[row * L + col])
                };
                let fvec = |row: usize, col: usize| {
                    C64x8::load(&f_re[row * L + col], &f_im[row * L + col])
                };
                let zero = C64x8::zero();
                let (det, repl) = if L == 1 {
                    // For L = 1, det D_ov is the sole contraction and the replacement is F_00.
                    (dvec(0, 0), fvec(0, 0))
                } else if L == 2 {
                    // For L = 2, use the explicit determinant and one-column replacement formula:
                    // `det = D00 D11 - D01 D10`.
                    // `repl = F00 D11 - F01 D10 - F10 D01 + F11 D00`.
                    let d00 = dvec(0, 0);
                    let d01 = dvec(0, 1);
                    let d10 = dvec(1, 0);
                    let d11 = dvec(1, 1);
                    let f00 = fvec(0, 0);
                    let f01 = fvec(0, 1);
                    let f10 = fvec(1, 0);
                    let f11 = fvec(1, 1);
                    let det = C64x8::minor(d00, d11, d01, d10);
                    let mut repl = C64x8::mul(f00, d11);
                    repl = C64x8::msub(repl, f01, d10);
                    repl = C64x8::msub(repl, f10, d01);
                    repl = C64x8::madd(repl, f11, d00);
                    (det, repl)
                } else if L == 3 {
                    // Calculate the nine distinct cofactors exactly once, then form
                    // det D_ov through row 0 and contract every cofactor with `\mathcal F`.
                    let d00 = dvec(0, 0);
                    let d01 = dvec(0, 1);
                    let d02 = dvec(0, 2);
                    let d10 = dvec(1, 0);
                    let d11 = dvec(1, 1);
                    let d12 = dvec(1, 2);
                    let d20 = dvec(2, 0);
                    let d21 = dvec(2, 1);
                    let d22 = dvec(2, 2);
                    let c00 = C64x8::minor(d11, d22, d12, d21);
                    let c01 = C64x8::sub(zero, C64x8::minor(d10, d22, d12, d20));
                    let c02 = C64x8::minor(d10, d21, d11, d20);
                    let c10 = C64x8::sub(zero, C64x8::minor(d01, d22, d02, d21));
                    let c11 = C64x8::minor(d00, d22, d02, d20);
                    let c12 = C64x8::sub(zero, C64x8::minor(d00, d21, d01, d20));
                    let c20 = C64x8::minor(d01, d12, d02, d11);
                    let c21 = C64x8::sub(zero, C64x8::minor(d00, d12, d02, d10));
                    let c22 = C64x8::minor(d00, d11, d01, d10);
                    let mut det = C64x8::mul(d00, c00);
                    det = C64x8::madd(det, d01, c01);
                    det = C64x8::madd(det, d02, c02);
                    let cof = [c00, c01, c02, c10, c11, c12, c20, c21, c22];
                    let mut repl = C64x8::zero();
                    for eta in 0..3 {
                        for z in 0..3 {
                            repl = C64x8::madd(repl, fvec(eta, z), cof[eta * 3 + z]);
                        }
                    }
                    (det, repl)
                } else {
                    c64x8_l4_dag::<L>(&dvec, &fvec)
                };

                let pref_v = C64x8::splat(pref.re, pref.im);
                let f0_v = C64x8::splat(f0.re, f0.im);

                let overlap_v = C64x8::mul(pref_v, det);
                let fock_v = C64x8::mul(pref_v, C64x8::mul_sub(f0_v, det, repl));

                let mut det_re = [0.0f64; 8];
                let mut det_im = [0.0f64; 8];
                let mut overlap_re = [0.0f64; 8];
                let mut overlap_im = [0.0f64; 8];
                let mut fock_re = [0.0f64; 8];
                let mut fock_im = [0.0f64; 8];
                det.store(&mut det_re, &mut det_im);
                overlap_v.store(&mut overlap_re, &mut overlap_im);
                fock_v.store(&mut fock_re, &mut fock_im);

                // Store eight packed overlap and one-body lanes, zeroing non-finite determinant lanes.
                for lane in 0..8 {
                    if det_re[lane].is_finite() && det_im[lane].is_finite() {
                        overlap[lane] = Complex64::new(overlap_re[lane], overlap_im[lane]);
                        fock[lane] = Complex64::new(fock_re[lane], fock_im[lane]);
                    } else {
                        overlap[lane] = Complex64::new(0.0, 0.0);
                        fock[lane] = Complex64::new(0.0, 0.0);
                    }
                }
            }
        }
    )
}

/// Evaluate the complex `L = 4` one-body cofactor DAG for 8 SIMD lanes.
/// This preserves the 18 distinct `2 x 2` minors used by the real AVX-512 kernel:
/// row-pair groups `B = (2,3)`, `Q = (1,3)` and `R = (1,2)`.
/// # Arguments:
/// - `dvec`: Loader for packed contraction determinant entries.
/// - `fvec`: Loader for packed one-body replacement entries.
/// # Returns
/// - `(C64x8, C64x8)`: Packed determinant and `C:\mathcal F` replacement contraction.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn c64x8_l4_dag<const L: usize>(
    dvec: &impl Fn(usize, usize) -> C64x8,
    fvec: &impl Fn(usize, usize) -> C64x8,
) -> (C64x8, C64x8) {
    // The old 16 cofactors contain `16 x 3 = 48` minor occurrences. Preserving their exact
    // expansions leaves `6 + 6 + 6 = 18` distinct minors, so 18 is the lower bound for this DAG.
    // Complex SIMD values use two registers, so `D_{ij}` is loaded on demand by row-pair group.

    // `B_{ab} = D_{2a}D_{3b} - D_{2b}D_{3a}` supplies cofactor rows 0 and 1.
    let (b01, b02, b03, b12, b13, b23) = {
        let d20 = dvec(2, 0);
        let d21 = dvec(2, 1);
        let d22 = dvec(2, 2);
        let d23 = dvec(2, 3);
        let d30 = dvec(3, 0);
        let d31 = dvec(3, 1);
        let d32 = dvec(3, 2);
        let d33 = dvec(3, 3);
        (
            C64x8::minor(d20, d31, d21, d30),
            C64x8::minor(d20, d32, d22, d30),
            C64x8::minor(d20, d33, d23, d30),
            C64x8::minor(d21, d32, d22, d31),
            C64x8::minor(d21, d33, d23, d31),
            C64x8::minor(d22, d33, d23, d32),
        )
    };

    let mut det = C64x8::zero();
    let mut repl0 = C64x8::zero();
    let mut repl1 = C64x8::zero();

    // Form `det(\mathbf D) = \sum_j D_{0j}C_{0j}` and row 0 of `C:\mathcal F`.
    {
        let d10 = dvec(1, 0);
        let d11 = dvec(1, 1);
        let d12 = dvec(1, 2);
        let d13 = dvec(1, 3);

        let cof00 = C64x8::cof_pos(d11, b23, d12, b13, d13, b12);
        det = C64x8::madd(det, dvec(0, 0), cof00);
        repl0 = C64x8::madd(repl0, fvec(0, 0), cof00);

        let cof01 = C64x8::cof_neg(d10, b23, d12, b03, d13, b02);
        det = C64x8::madd(det, dvec(0, 1), cof01);
        repl0 = C64x8::madd(repl0, fvec(0, 1), cof01);

        let cof02 = C64x8::cof_pos(d10, b13, d11, b03, d13, b01);
        det = C64x8::madd(det, dvec(0, 2), cof02);
        repl0 = C64x8::madd(repl0, fvec(0, 2), cof02);

        let cof03 = C64x8::cof_neg(d10, b12, d11, b02, d12, b01);
        det = C64x8::madd(det, dvec(0, 3), cof03);
        repl0 = C64x8::madd(repl0, fvec(0, 3), cof03);
    }

    // Reuse the same six `B_{ab}` values for row 1 of `C:\mathcal F`.
    {
        let d00 = dvec(0, 0);
        let d01 = dvec(0, 1);
        let d02 = dvec(0, 2);
        let d03 = dvec(0, 3);

        let cof10 = C64x8::cof_neg(d01, b23, d02, b13, d03, b12);
        repl1 = C64x8::madd(repl1, fvec(1, 0), cof10);

        let cof11 = C64x8::cof_pos(d00, b23, d02, b03, d03, b02);
        repl1 = C64x8::madd(repl1, fvec(1, 1), cof11);

        let cof12 = C64x8::cof_neg(d00, b13, d01, b03, d03, b01);
        repl1 = C64x8::madd(repl1, fvec(1, 2), cof12);

        let cof13 = C64x8::cof_pos(d00, b12, d01, b02, d02, b01);
        repl1 = C64x8::madd(repl1, fvec(1, 3), cof13);
    }

    let repl01 = C64x8::add(repl0, repl1);

    // `Q_{ab} = D_{1a}D_{3b} - D_{1b}D_{3a}` supplies cofactor row 2.
    let (q01, q02, q03, q12, q13, q23) = {
        let d10 = dvec(1, 0);
        let d11 = dvec(1, 1);
        let d12 = dvec(1, 2);
        let d13 = dvec(1, 3);
        let d30 = dvec(3, 0);
        let d31 = dvec(3, 1);
        let d32 = dvec(3, 2);
        let d33 = dvec(3, 3);
        (
            C64x8::minor(d10, d31, d11, d30),
            C64x8::minor(d10, d32, d12, d30),
            C64x8::minor(d10, d33, d13, d30),
            C64x8::minor(d11, d32, d12, d31),
            C64x8::minor(d11, d33, d13, d31),
            C64x8::minor(d12, d33, d13, d32),
        )
    };

    let mut repl2 = C64x8::zero();
    {
        let d00 = dvec(0, 0);
        let d01 = dvec(0, 1);
        let d02 = dvec(0, 2);
        let d03 = dvec(0, 3);

        let cof20 = C64x8::cof_pos(d01, q23, d02, q13, d03, q12);
        repl2 = C64x8::madd(repl2, fvec(2, 0), cof20);

        let cof21 = C64x8::cof_neg(d00, q23, d02, q03, d03, q02);
        repl2 = C64x8::madd(repl2, fvec(2, 1), cof21);

        let cof22 = C64x8::cof_pos(d00, q13, d01, q03, d03, q01);
        repl2 = C64x8::madd(repl2, fvec(2, 2), cof22);

        let cof23 = C64x8::cof_neg(d00, q12, d01, q02, d02, q01);
        repl2 = C64x8::madd(repl2, fvec(2, 3), cof23);
    }

    // `R_{ab} = D_{1a}D_{2b} - D_{1b}D_{2a}` supplies cofactor row 3.
    let (r01, r02, r03, r12, r13, r23) = {
        let d10 = dvec(1, 0);
        let d11 = dvec(1, 1);
        let d12 = dvec(1, 2);
        let d13 = dvec(1, 3);
        let d20 = dvec(2, 0);
        let d21 = dvec(2, 1);
        let d22 = dvec(2, 2);
        let d23 = dvec(2, 3);
        (
            C64x8::minor(d10, d21, d11, d20),
            C64x8::minor(d10, d22, d12, d20),
            C64x8::minor(d10, d23, d13, d20),
            C64x8::minor(d11, d22, d12, d21),
            C64x8::minor(d11, d23, d13, d21),
            C64x8::minor(d12, d23, d13, d22),
        )
    };

    let mut repl3 = C64x8::zero();
    {
        let d00 = dvec(0, 0);
        let d01 = dvec(0, 1);
        let d02 = dvec(0, 2);
        let d03 = dvec(0, 3);

        let cof30 = C64x8::cof_neg(d01, r23, d02, r13, d03, r12);
        repl3 = C64x8::madd(repl3, fvec(3, 0), cof30);

        let cof31 = C64x8::cof_pos(d00, r23, d02, r03, d03, r02);
        repl3 = C64x8::madd(repl3, fvec(3, 1), cof31);

        let cof32 = C64x8::cof_neg(d00, r13, d01, r03, d03, r01);
        repl3 = C64x8::madd(repl3, fvec(3, 2), cof32);

        let cof33 = C64x8::cof_pos(d00, r12, d01, r02, d02, r01);
        repl3 = C64x8::madd(repl3, fvec(3, 3), cof33);
    }

    // Preserve the previous contraction tree `((repl0 + repl1) + (repl2 + repl3))`.
    let repl23 = C64x8::add(repl2, repl3);
    (det, C64x8::add(repl01, repl23))
}

/// Prepare and evaluate the generic-rank overlap and generalised-Fock matrix element for `m = 0`.
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}},`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{z=1}^{L}\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}].`
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
fn xw_f_overlap_m0_gen_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    l: usize,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_gen,
        {
            // Generic m = 0 path: build D_ov once, then use cof[D_ov] for every
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
            build_d(
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

            if let Some(det_det) = adjugate_transpose(
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
                    let corr = column_replacement_correction(
                        l,
                        det0,
                        scratch.adjt_det.as_slice(),
                        b,
                        |r| fsl[base + scratch.rows[r]],
                    );
                    contrib -= det_det + corr;
                }

                (pref * det_det, pref * contrib)
            } else {
                let overlap = pref * det(det0, l).unwrap_or(zero);
                (overlap, zero)
            }
        }
    )
}

/// Prepare and evaluate the overlap and generalised-Fock matrix element when `m > 0` by summing
/// the allowed one-body distributions
/// `m_1 + \cdots + m_{L+1} = m, \qquad m_i \in \{0,1\}.`
/// `The determinant labels and endpoint contraction determinants`
/// `\mathbf D_{\mathrm{ov}}(0,\ldots,0)` and `\mathbf D_{\mathrm{ov}}(1,\ldots,1)` are built once.
/// `The first assignment selects {}^x F_0^{(m_1)} and the operator side of each`
/// `\mathcal F^{(m_1,m_j)} column; the remaining assignments select the columns of`
/// `\mathbf D_{\mathrm{ov}}. Terms with m_1 = 0 also satisfy the overlap constraint`
/// `m_2+\cdots+m_{L+1}=m and are accumulated into the overlap without a second distribution loop.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for endpoint and mixed contraction determinants, cofactors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` summed over all distributions.
#[inline(always)]
fn xw_f_overlap_gen_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_gen, {
        // General zero-overlap path: prepare the all-zero and all-one endpoint determinants,
        // then sum the GNME one-body expression over every allowed mixed distribution.
        let l = x_ex.holes.count_ones() as usize + w_ex.holes.count_ones() as usize;
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
        build_d(
            scratch.det0.as_mut_slice(),
            l,
            &x0,
            &y0,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

        let x1 = w.x(1);
        let y1 = w.y(1);
        build_d(
            scratch.det1.as_mut_slice(),
            l,
            &x1,
            &y1,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

        let zero = <T as From<f64>>::from(0.0);
        let n = w.n();
        let mut overlap_acc = zero;
        let mut fock_acc = zero;

        mix_dets_same(w, l, 1, scratch, |bits, scratch| {
            let mi = bit(bits, 0);

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
                overlap_acc += det(scratch.det_mix.as_slice(), l).unwrap_or(zero);
            }
        });

        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        (pref * overlap_acc, pref * fock_acc)
    })
}
