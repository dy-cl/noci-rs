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
use super::dispatch::dispatch_onebody_ranks;
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, mix_dets_same,
};
use super::prepare::construct_determinant_indices;
#[cfg(target_arch = "x86_64")]
use super::simd::{C64x4, C64x8, F64x4, F64x8};

#[cfg(target_arch = "x86_64")]
type PreparedSimdKernel<T, R, const N: usize> = for<'a> unsafe fn(
    &SameSpinView<'a, T>,
    &ExcitationSpinCache,
    &[ExcitationSpinCache; N],
    &mut [R],
    &mut [R],
);

/// Immutable determinant and excitation metadata shared by same-spin SIMD paths.
#[cfg(target_arch = "x86_64")]
struct SameSpinSimdInput<'a> {
    /// Reduced target spin representative shared by this factor row.
    target: ReducedOneSpinDetState,
    /// Reduced source spin representatives in output-column order.
    sources: &'a [ReducedOneSpinDetState],
    /// Logical source component IDs in fixed-rank Wick evaluation order.
    source_order: &'a [usize],
    /// Boundaries of equal-rank, common-hole source groups in `source_order`.
    source_groups: &'a [usize],
    /// Source excitation caches in fixed-rank Wick evaluation order.
    source_caches: &'a [ExcitationSpinCache],
    /// Source excitation phases in fixed-rank Wick evaluation order.
    source_phases: &'a [f64],
    /// Whether target determinant belongs to left Wick reference.
    target_left: bool,
    /// Whether alpha-spin rather than beta-spin factors are being evaluated.
    alpha: bool,
}

/// Inputs and outputs for one row of same-spin one-body factors.
pub(crate) struct SameSpinOneBodyBatch<'a, T: NOCIScalar> {
    /// Determinant basis used by scalar fallback evaluation.
    pub(crate) basis: &'a [DetState<T>],
    /// Reduced target spin representative shared by this factor row.
    pub(crate) target: ReducedOneSpinDetState,
    /// Reduced source spin representatives in output-column order.
    pub(crate) sources: &'a [ReducedOneSpinDetState],
    /// Logical source component IDs in fixed-rank Wick evaluation order.
    pub(crate) source_order: &'a [usize],
    /// Boundaries of equal-rank, common-hole source groups in `source_order`.
    pub(crate) source_groups: &'a [usize],
    /// Source excitation caches in fixed-rank Wick evaluation order.
    pub(crate) source_caches: &'a [ExcitationSpinCache],
    /// Source excitation phases in fixed-rank Wick evaluation order.
    pub(crate) source_phases: &'a [f64],
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
        source_order,
        source_groups,
        source_caches,
        source_phases,
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
                SameSpinSimdInput {
                    target,
                    sources,
                    source_order,
                    source_groups,
                    source_caches,
                    source_phases,
                    target_left,
                    alpha,
                },
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
                SameSpinSimdInput {
                    target,
                    sources,
                    source_order,
                    source_groups,
                    source_caches,
                    source_phases,
                    target_left,
                    alpha,
                },
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
/// AVX2 requests are traversed in static `(rank, holes, particles)` order and packetised within
/// common-hole groups; unsupported requests use the scalar prepared Wick evaluator.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `basis`: Determinant basis containing full excitation masks.
/// - `input`: Target/source representatives, evaluation metadata, and spin/orientation flags.
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
    input: SameSpinSimdInput<'_>,
    scratch: &mut WickScratch<T>,
    tol: f64,
    out: (&mut [f64], &mut [f64]),
) -> bool {
    let SameSpinSimdInput {
        target: target_rep,
        sources,
        source_order,
        source_groups,
        source_caches,
        source_phases,
        target_left,
        alpha,
    } = input;
    let (overlap, fock) = out;
    let target_cache = target_rep.excitation_cache;
    let target_phase = target_rep.phase;

    unsafe {
        if is_x86_feature_detected!("avx512f") {
            let target_rank = usize::from(target_cache.rank);

            for bounds in source_groups.windows(2) {
                let group_start = *bounds.get_unchecked(0);
                let group_end = *bounds.get_unchecked(1);
                let source_rank = usize::from(source_caches.get_unchecked(group_start).rank);
                let l = target_rank + source_rank;

                if (1..=4).contains(&l) {
                    let ranks = if target_left {
                        (target_rank, source_rank)
                    } else {
                        (source_rank, target_rank)
                    };
                    let kernel: PreparedSimdKernel<T, f64, 8> = if target_left {
                        dispatch_onebody_ranks!(
                            ranks,
                            |RX, RW, L| xw_f_overlap_m0_prepared_f64x8_const::<T, RX, RW, L, true>,
                            unreachable!(),
                        )
                    } else {
                        dispatch_onebody_ranks!(
                            ranks,
                            |RX, RW, L| xw_f_overlap_m0_prepared_f64x8_const::<T, RX, RW, L, false>,
                            unreachable!(),
                        )
                    };
                    let mut packet_start = group_start;

                    while group_end - packet_start >= 8 {
                        let packet = &*source_caches
                            .as_ptr()
                            .add(packet_start)
                            .cast::<[ExcitationSpinCache; 8]>();
                        let mut s = [0.0f64; 8];
                        let mut f = [0.0f64; 8];
                        kernel(w, &target_cache, packet, &mut s, &mut f);

                        for lane in 0..8 {
                            let position = packet_start + lane;
                            let output = *source_order.get_unchecked(position);
                            let phase = target_phase * *source_phases.get_unchecked(position);
                            *overlap.get_unchecked_mut(output) = s[lane] * phase;
                            *fock.get_unchecked_mut(output) = f[lane] * phase;
                        }

                        packet_start += 8;
                    }

                    if packet_start < group_end {
                        let count = group_end - packet_start;
                        let fill_cache = *source_caches.get_unchecked(packet_start);
                        let mut packet = [fill_cache; 8];

                        for (lane, value) in packet.iter_mut().enumerate().take(count) {
                            *value = *source_caches.get_unchecked(packet_start + lane);
                        }

                        let mut s = [0.0f64; 8];
                        let mut f = [0.0f64; 8];
                        kernel(w, &target_cache, &packet, &mut s, &mut f);

                        for lane in 0..count {
                            let position = packet_start + lane;
                            let output = *source_order.get_unchecked(position);
                            let phase = target_phase * *source_phases.get_unchecked(position);
                            *overlap.get_unchecked_mut(output) = s[lane] * phase;
                            *fock.get_unchecked_mut(output) = f[lane] * phase;
                        }
                    }
                } else {
                    for position in group_start..group_end {
                        let source_id = *source_order.get_unchecked(position);
                        let source_rep = *sources.get_unchecked(source_id);
                        let (s, f) = xw_f_overlap_prepared_scalar_value(
                            w,
                            basis,
                            (target_rep, source_rep),
                            (target_left, alpha),
                            scratch,
                            tol,
                        );
                        let phase = target_phase * source_rep.phase;
                        *overlap.get_unchecked_mut(source_id) =
                            phase * *std::ptr::from_ref(&s).cast::<f64>();
                        *fock.get_unchecked_mut(source_id) =
                            phase * *std::ptr::from_ref(&f).cast::<f64>();
                    }
                }
            }
            return true;
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let target_rank = usize::from(target_cache.rank);

            for bounds in source_groups.windows(2) {
                let group_start = *bounds.get_unchecked(0);
                let group_end = *bounds.get_unchecked(1);
                let source_rank = usize::from(source_caches.get_unchecked(group_start).rank);
                let l = target_rank + source_rank;

                if (1..=4).contains(&l) {
                    let ranks = if target_left {
                        (target_rank, source_rank)
                    } else {
                        (source_rank, target_rank)
                    };
                    let kernel: PreparedSimdKernel<T, f64, 4> = if target_left {
                        dispatch_onebody_ranks!(
                            ranks,
                            |RX, RW, L| xw_f_overlap_m0_prepared_f64x4_const::<T, RX, RW, L, true>,
                            unreachable!(),
                        )
                    } else {
                        dispatch_onebody_ranks!(
                            ranks,
                            |RX, RW, L| xw_f_overlap_m0_prepared_f64x4_const::<T, RX, RW, L, false>,
                            unreachable!(),
                        )
                    };
                    let mut packet_start = group_start;

                    while group_end - packet_start >= 4 {
                        let packet = &*source_caches
                            .as_ptr()
                            .add(packet_start)
                            .cast::<[ExcitationSpinCache; 4]>();
                        let mut s = [0.0f64; 4];
                        let mut f = [0.0f64; 4];
                        kernel(w, &target_cache, packet, &mut s, &mut f);

                        for lane in 0..4 {
                            let position = packet_start + lane;
                            let output = *source_order.get_unchecked(position);
                            let phase = target_phase * *source_phases.get_unchecked(position);
                            *overlap.get_unchecked_mut(output) = s[lane] * phase;
                            *fock.get_unchecked_mut(output) = f[lane] * phase;
                        }

                        packet_start += 4;
                    }

                    if packet_start < group_end {
                        let count = group_end - packet_start;
                        let fill_cache = *source_caches.get_unchecked(packet_start);
                        let mut packet = [fill_cache; 4];

                        for (lane, value) in packet.iter_mut().enumerate().take(count) {
                            *value = *source_caches.get_unchecked(packet_start + lane);
                        }

                        let mut s = [0.0f64; 4];
                        let mut f = [0.0f64; 4];
                        kernel(w, &target_cache, &packet, &mut s, &mut f);

                        for lane in 0..count {
                            let position = packet_start + lane;
                            let output = *source_order.get_unchecked(position);
                            let phase = target_phase * *source_phases.get_unchecked(position);
                            *overlap.get_unchecked_mut(output) = s[lane] * phase;
                            *fock.get_unchecked_mut(output) = f[lane] * phase;
                        }
                    }
                } else {
                    for position in group_start..group_end {
                        let source_id = *source_order.get_unchecked(position);
                        let source_rep = *sources.get_unchecked(source_id);
                        let (s, f) = xw_f_overlap_prepared_scalar_value(
                            w,
                            basis,
                            (target_rep, source_rep),
                            (target_left, alpha),
                            scratch,
                            tol,
                        );
                        let phase = target_phase * source_rep.phase;
                        *overlap.get_unchecked_mut(source_id) =
                            phase * *std::ptr::from_ref(&s).cast::<f64>();
                        *fock.get_unchecked_mut(source_id) =
                            phase * *std::ptr::from_ref(&f).cast::<f64>();
                    }
                }
            }
            return true;
        }
    }

    false
}

/// Try to evaluate one complex same-spin factor row with fixed-rank SIMD kernels.
/// Rank-compatible requests are binned by `L = 1,\ldots,4`; unsupported requests use the scalar
/// prepared Wick evaluator inside the same row traversal.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `basis`: Determinant basis containing full excitation masks.
/// - `input`: Target/source representatives, evaluation metadata, and spin/orientation flags.
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
    input: SameSpinSimdInput<'_>,
    scratch: &mut WickScratch<T>,
    tol: f64,
    out: (&mut [Complex64], &mut [Complex64]),
) -> bool {
    let SameSpinSimdInput {
        target: target_rep,
        sources,
        source_order,
        source_groups,
        source_caches,
        source_phases,
        target_left,
        alpha,
    } = input;
    let (overlap, fock) = out;
    let target_cache = target_rep.excitation_cache;
    let target_phase = target_rep.phase;

    unsafe {
        if is_x86_feature_detected!("avx512f") {
            let target_rank = usize::from(target_cache.rank);

            for bounds in source_groups.windows(2) {
                let group_start = *bounds.get_unchecked(0);
                let group_end = *bounds.get_unchecked(1);
                let source_rank = usize::from(source_caches.get_unchecked(group_start).rank);
                let l = target_rank + source_rank;

                if (1..=4).contains(&l) {
                    let ranks = if target_left {
                        (target_rank, source_rank)
                    } else {
                        (source_rank, target_rank)
                    };
                    let kernel: PreparedSimdKernel<T, Complex64, 8> = if target_left {
                        dispatch_onebody_ranks!(
                            ranks,
                            |RX, RW, L| xw_f_overlap_m0_prepared_c64x8_const::<T, RX, RW, L, true>,
                            unreachable!(),
                        )
                    } else {
                        dispatch_onebody_ranks!(
                            ranks,
                            |RX, RW, L| xw_f_overlap_m0_prepared_c64x8_const::<T, RX, RW, L, false>,
                            unreachable!(),
                        )
                    };
                    let mut packet_start = group_start;

                    while group_end - packet_start >= 8 {
                        let packet = &*source_caches
                            .as_ptr()
                            .add(packet_start)
                            .cast::<[ExcitationSpinCache; 8]>();
                        let mut s = [Complex64::new(0.0, 0.0); 8];
                        let mut f = [Complex64::new(0.0, 0.0); 8];
                        kernel(w, &target_cache, packet, &mut s, &mut f);

                        for lane in 0..8 {
                            let position = packet_start + lane;
                            let output = *source_order.get_unchecked(position);
                            let phase = target_phase * *source_phases.get_unchecked(position);
                            *overlap.get_unchecked_mut(output) = s[lane] * phase;
                            *fock.get_unchecked_mut(output) = f[lane] * phase;
                        }

                        packet_start += 8;
                    }

                    if packet_start < group_end {
                        let count = group_end - packet_start;
                        let fill_cache = *source_caches.get_unchecked(packet_start);
                        let mut packet = [fill_cache; 8];

                        for (lane, value) in packet.iter_mut().enumerate().take(count) {
                            *value = *source_caches.get_unchecked(packet_start + lane);
                        }

                        let mut s = [Complex64::new(0.0, 0.0); 8];
                        let mut f = [Complex64::new(0.0, 0.0); 8];
                        kernel(w, &target_cache, &packet, &mut s, &mut f);

                        for lane in 0..count {
                            let position = packet_start + lane;
                            let output = *source_order.get_unchecked(position);
                            let phase = target_phase * *source_phases.get_unchecked(position);
                            *overlap.get_unchecked_mut(output) = s[lane] * phase;
                            *fock.get_unchecked_mut(output) = f[lane] * phase;
                        }
                    }
                } else {
                    for position in group_start..group_end {
                        let source_id = *source_order.get_unchecked(position);
                        let source_rep = *sources.get_unchecked(source_id);
                        let (s, f) = xw_f_overlap_prepared_scalar_value(
                            w,
                            basis,
                            (target_rep, source_rep),
                            (target_left, alpha),
                            scratch,
                            tol,
                        );
                        let phase = target_phase * source_rep.phase;
                        *overlap.get_unchecked_mut(source_id) =
                            *std::ptr::from_ref(&s).cast::<Complex64>() * phase;
                        *fock.get_unchecked_mut(source_id) =
                            *std::ptr::from_ref(&f).cast::<Complex64>() * phase;
                    }
                }
            }
            return true;
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let target_rank = usize::from(target_cache.rank);

            for bounds in source_groups.windows(2) {
                let group_start = *bounds.get_unchecked(0);
                let group_end = *bounds.get_unchecked(1);
                let source_rank = usize::from(source_caches.get_unchecked(group_start).rank);
                let l = target_rank + source_rank;

                if (1..=4).contains(&l) {
                    let ranks = if target_left {
                        (target_rank, source_rank)
                    } else {
                        (source_rank, target_rank)
                    };
                    let kernel: PreparedSimdKernel<T, Complex64, 4> = if target_left {
                        dispatch_onebody_ranks!(
                            ranks,
                            |RX, RW, L| xw_f_overlap_m0_prepared_c64x4_const::<T, RX, RW, L, true>,
                            unreachable!(),
                        )
                    } else {
                        dispatch_onebody_ranks!(
                            ranks,
                            |RX, RW, L| xw_f_overlap_m0_prepared_c64x4_const::<T, RX, RW, L, false>,
                            unreachable!(),
                        )
                    };
                    let mut packet_start = group_start;

                    while group_end - packet_start >= 4 {
                        let packet = &*source_caches
                            .as_ptr()
                            .add(packet_start)
                            .cast::<[ExcitationSpinCache; 4]>();
                        let mut s = [Complex64::new(0.0, 0.0); 4];
                        let mut f = [Complex64::new(0.0, 0.0); 4];
                        kernel(w, &target_cache, packet, &mut s, &mut f);

                        for lane in 0..4 {
                            let position = packet_start + lane;
                            let output = *source_order.get_unchecked(position);
                            let phase = target_phase * *source_phases.get_unchecked(position);
                            *overlap.get_unchecked_mut(output) = s[lane] * phase;
                            *fock.get_unchecked_mut(output) = f[lane] * phase;
                        }

                        packet_start += 4;
                    }

                    if packet_start < group_end {
                        let count = group_end - packet_start;
                        let fill_cache = *source_caches.get_unchecked(packet_start);
                        let mut packet = [fill_cache; 4];

                        for (lane, value) in packet.iter_mut().enumerate().take(count) {
                            *value = *source_caches.get_unchecked(packet_start + lane);
                        }

                        let mut s = [Complex64::new(0.0, 0.0); 4];
                        let mut f = [Complex64::new(0.0, 0.0); 4];
                        kernel(w, &target_cache, &packet, &mut s, &mut f);

                        for lane in 0..count {
                            let position = packet_start + lane;
                            let output = *source_order.get_unchecked(position);
                            let phase = target_phase * *source_phases.get_unchecked(position);
                            *overlap.get_unchecked_mut(output) = s[lane] * phase;
                            *fock.get_unchecked_mut(output) = f[lane] * phase;
                        }
                    }
                } else {
                    for position in group_start..group_end {
                        let source_id = *source_order.get_unchecked(position);
                        let source_rep = *sources.get_unchecked(source_id);
                        let (s, f) = xw_f_overlap_prepared_scalar_value(
                            w,
                            basis,
                            (target_rep, source_rep),
                            (target_left, alpha),
                            scratch,
                            tol,
                        );
                        let phase = target_phase * source_rep.phase;
                        *overlap.get_unchecked_mut(source_id) =
                            *std::ptr::from_ref(&s).cast::<Complex64>() * phase;
                        *fock.get_unchecked_mut(source_id) =
                            *std::ptr::from_ref(&f).cast::<Complex64>() * phase;
                    }
                }
            }
            return true;
        }
    }

    false
}

/// Prepare and evaluate the same-spin overlap and generalised-Fock matrix element between excited
/// determinants generated from the reference pair `\langle{}^x\Psi| and |{}^w\Psi\rangle:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L = m}}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L),`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat F|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+1}\\m_1+\cdots+m_{L+1} = m}}`
/// `[{}^x F_0^{(m_1)}\det\mathbf D_{\mathrm{ov}}(m_2,\ldots,m_{L+1})`
/// `- \sum_{z = 1}^{L}\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}`
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
pub(crate) fn xw_f_overlap_prepared<T: NOCIScalar>(
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
        let rx = x_ex.holes.count_ones() as usize;
        let rw = w_ex.holes.count_ones() as usize;

        if rx == 0 && rw == 0 {
            let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
            (pref, pref * w.f0f[0])
        } else {
            dispatch_onebody_ranks!(
                (rx, rw),
                |RX, RW, L| xw_f_overlap_m0_prepared_const::<T, RX, RW, L>(
                    w, x_ex, w_ex, scratch, tol,
                ),
                xw_f_overlap_m0_gen_prepared(w, x_ex, w_ex, scratch, rx + rw, tol),
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
fn xw_f_overlap_m0_prepared_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
>(
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

            let det0 = &scratch.det0.as_slice()[..L * L];
            let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

            // Evaluate `\det\mathbf D_{\mathrm{ov}}` and
            // `\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z} = (-1)^{\eta+z}`
            // `\det\mathbf D_{\mathrm{ov}}[\eta|z]`.
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
/// Evaluate four real fixed-rank `L = 1,\ldots,4` one-body/overlap factors for `(RX,RW)`.
/// The rank pair and fixed side are compile-time constants, source packets share their hole tuple,
/// and the existing 18-minor cofactor algebra consumes directly packed contraction entries.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `fixed`: Fixed target excitation cache.
/// - `varying`: Four source excitation caches sharing their hole tuple.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid cached orbital indices,
///   and compile-time ranks satisfying `RX + RW = L` with `1 <= L <= 4`.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_prepared_f64x4_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, T>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            unsafe {
                let n = w.n();
                let nocc = w.nocc;
                let nvirt = w.nmo - nocc;
                // `x0` and `y0` store the `X^{(0)}` and `Y^{(0)}` fundamental contractions.
                let x0 = w.x_slice(0).as_ptr().cast::<f64>();
                let y0 = w.y_slice(0).as_ptr().cast::<f64>();
                // `fsl` stores the one-body column intermediate `\mathcal F^{(0,0)}`.
                let fsl = w.ff_t_slice(0, 0).as_ptr().cast::<f64>();
                // `phase` is the reference-pair orbital phase.
                let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
                // `f0 = {}^x F_0^{(0)}` is the scalar one-body intermediate.
                let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
                // `pref = p\,{}^{xw}\tilde S` is the phase-weighted reduced overlap.
                let pref = phase * w.tilde_s_prod;

                // Select the lane-local bra excitation `{}^x\Psi_{i\cdots}^{a\cdots}`.
                let x_data = |lane: usize| -> &ExcitationSpinCache {
                    if XFIX {
                        fixed
                    } else {
                        varying.get_unchecked(lane)
                    }
                };
                // Select the lane-local ket excitation `{}^w\Psi_{j\cdots}^{b\cdots}`.
                let w_data = |lane: usize| -> &ExcitationSpinCache {
                    if XFIX {
                        varying.get_unchecked(lane)
                    } else {
                        fixed
                    }
                };
                // Rows are `r_\eta \in V_x\cup O_w`, with x-particles before w-holes.
                let row_index = |eta: usize, lane: usize| -> usize {
                    if eta < RX {
                        usize::from(*x_data(lane).indices.get_unchecked(4 + eta)) - nocc
                    } else {
                        nvirt + usize::from(*w_data(lane).indices.get_unchecked(eta - RX))
                    }
                };
                // Columns are `c_z \in O_x\cup V_w`, with x-holes before w-particles.
                let col_index = |z: usize, lane: usize| -> usize {
                    if z < RX {
                        usize::from(*x_data(lane).indices.get_unchecked(z))
                    } else {
                        usize::from(*w_data(lane).indices.get_unchecked(4 + z - RX))
                    }
                };
                // `D_{\eta z} = X^{(0)}_{r_\eta c_z}` for `\eta \geq z`, otherwise
                // `D_{\eta z} = Y^{(0)}_{r_\eta c_z}`.
                let load_d = |eta: usize, z: usize| -> F64x4 {
                    let matrix = if eta >= z { x0 } else { y0 };
                    let invariant = if XFIX { z < RX } else { eta >= RX };

                    // Broadcast lane-invariant entries and gather lane-dependent entries.
                    if invariant {
                        let row = row_index(eta, 0);
                        let col = col_index(z, 0);
                        F64x4::splat(*matrix.add(row * n + col))
                    } else if XFIX {
                        let row_base = row_index(eta, 0) * n;
                        let col0 = col_index(z, 0);
                        let col1 = col_index(z, 1);
                        let col2 = col_index(z, 2);
                        let col3 = col_index(z, 3);
                        F64x4::from_values(
                            *matrix.add(row_base + col0),
                            *matrix.add(row_base + col1),
                            *matrix.add(row_base + col2),
                            *matrix.add(row_base + col3),
                        )
                    } else {
                        let col = col_index(z, 0);
                        let row0 = row_index(eta, 0);
                        let row1 = row_index(eta, 1);
                        let row2 = row_index(eta, 2);
                        let row3 = row_index(eta, 3);
                        F64x4::from_values(
                            *matrix.add(row0 * n + col),
                            *matrix.add(row1 * n + col),
                            *matrix.add(row2 * n + col),
                            *matrix.add(row3 * n + col),
                        )
                    }
                };
                // `fvec(\eta,z) = \mathcal F_{r_\eta c_z}^{(0,0)}` in each SIMD lane.
                let fvec = |eta: usize, z: usize| -> F64x4 {
                    let invariant = if XFIX { z < RX } else { eta >= RX };

                    // Broadcast lane-invariant intermediates and gather lane-dependent intermediates.
                    if invariant {
                        let row = row_index(eta, 0);
                        let col = col_index(z, 0);
                        F64x4::splat(*fsl.add(col * n + row))
                    } else if XFIX {
                        let row = row_index(eta, 0);
                        let col0 = col_index(z, 0);
                        let col1 = col_index(z, 1);
                        let col2 = col_index(z, 2);
                        let col3 = col_index(z, 3);
                        F64x4::from_values(
                            *fsl.add(col0 * n + row),
                            *fsl.add(col1 * n + row),
                            *fsl.add(col2 * n + row),
                            *fsl.add(col3 * n + row),
                        )
                    } else {
                        let col_base = col_index(z, 0) * n;
                        let row0 = row_index(eta, 0);
                        let row1 = row_index(eta, 1);
                        let row2 = row_index(eta, 2);
                        let row3 = row_index(eta, 3);
                        F64x4::from_values(
                            *fsl.add(col_base + row0),
                            *fsl.add(col_base + row1),
                            *fsl.add(col_base + row2),
                            *fsl.add(col_base + row3),
                        )
                    }
                };
                let zero = F64x4::zero();
                // Evaluate `\det\mathbf D_{\mathrm{ov}}` and
                // `R = \sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                let (det, repl) = match L {
                    // For `L = 1`, `\det\mathbf D = D_{00}` and `R = \mathcal F_{00}`.
                    1 => (load_d(0, 0), fvec(0, 0)),
                    2 => {
                        let d00 = load_d(0, 0);
                        let d01 = load_d(0, 1);
                        let d10 = load_d(1, 0);
                        let d11 = load_d(1, 1);
                        // `\det\mathbf D = D_{00}D_{11} - D_{01}D_{10}`.
                        let det = F64x4::minor(d00, d11, d01, d10);
                        // `R = \mathcal F_{00}D_{11} - \mathcal F_{01}D_{10}`
                        // `- \mathcal F_{10}D_{01} + \mathcal F_{11}D_{00}`.
                        let mut repl = F64x4::mul(fvec(0, 0), d11);
                        repl = F64x4::msub(repl, fvec(0, 1), d10);
                        repl = F64x4::msub(repl, fvec(1, 0), d01);
                        repl = F64x4::madd(repl, fvec(1, 1), d00);
                        (det, repl)
                    }
                    // For `L = 3`, write `C_{\eta z} = \operatorname{cof}[\mathbf D]_{\eta z}`.
                    3 => {
                        let d00 = load_d(0, 0);
                        let d01 = load_d(0, 1);
                        let d02 = load_d(0, 2);
                        let d10 = load_d(1, 0);
                        let d11 = load_d(1, 1);
                        let d12 = load_d(1, 2);
                        let d20 = load_d(2, 0);
                        let d21 = load_d(2, 1);
                        let d22 = load_d(2, 2);

                        // `C_{00} = D_{11}D_{22} - D_{12}D_{21}`.
                        let c00 = F64x4::minor(d11, d22, d12, d21);
                        // Begin `\det\mathbf D = \sum_z D_{0z}C_{0z}` and
                        // `R = \sum_{\eta z}\mathcal F_{\eta z}C_{\eta z}`.
                        let mut det = F64x4::mul(d00, c00);
                        let mut repl = F64x4::mul(fvec(0, 0), c00);
                        // `C_{01} = -(D_{10}D_{22} - D_{12}D_{20})`.
                        let c01 = F64x4::sub(zero, F64x4::minor(d10, d22, d12, d20));
                        det = F64x4::madd(det, d01, c01);
                        repl = F64x4::madd(repl, fvec(0, 1), c01);
                        // `C_{02} = D_{10}D_{21} - D_{11}D_{20}`.
                        let c02 = F64x4::minor(d10, d21, d11, d20);
                        det = F64x4::madd(det, d02, c02);
                        repl = F64x4::madd(repl, fvec(0, 2), c02);
                        // `C_{10} = -(D_{01}D_{22} - D_{02}D_{21})`.
                        let c10 = F64x4::sub(zero, F64x4::minor(d01, d22, d02, d21));
                        repl = F64x4::madd(repl, fvec(1, 0), c10);
                        // `C_{11} = D_{00}D_{22} - D_{02}D_{20}`.
                        let c11 = F64x4::minor(d00, d22, d02, d20);
                        repl = F64x4::madd(repl, fvec(1, 1), c11);
                        // `C_{12} = -(D_{00}D_{21} - D_{01}D_{20})`.
                        let c12 = F64x4::sub(zero, F64x4::minor(d00, d21, d01, d20));
                        repl = F64x4::madd(repl, fvec(1, 2), c12);
                        // `C_{20} = D_{01}D_{12} - D_{02}D_{11}`.
                        let c20 = F64x4::minor(d01, d12, d02, d11);
                        repl = F64x4::madd(repl, fvec(2, 0), c20);
                        // `C_{21} = -(D_{00}D_{12} - D_{02}D_{10})`.
                        let c21 = F64x4::sub(zero, F64x4::minor(d00, d12, d02, d10));
                        repl = F64x4::madd(repl, fvec(2, 1), c21);
                        // `C_{22} = D_{00}D_{11} - D_{01}D_{10}`.
                        let c22 = F64x4::minor(d00, d11, d01, d10);
                        repl = F64x4::madd(repl, fvec(2, 2), c22);
                        (det, repl)
                    }
                    4 => {
                        let mut d = [F64x4::zero(); 16];
                        for eta in 0..4 {
                            for z in 0..4 {
                                *d.get_unchecked_mut(eta * 4 + z) = load_d(eta, z);
                            }
                        }
                        let dvec = |row: usize, col: usize| *d.get_unchecked(row * 4 + col);

                        // The 16 cofactors contain `16 x 3 = 48` minor occurrences. Factoring their
                        // expansions leaves `6 + 6 + 6 = 18` distinct minors in this DAG.
                        // Reload `D_{ij}` by group to keep the 18 minor intermediates within the
                        // available register budget.

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

                        // Form `\det\mathbf D = \sum_j D_{0j}\operatorname{cof}[\mathbf D]_{0j}` and
                        // row 0 of `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                        let mut det_v = F64x4::zero();
                        let mut repl0 = F64x4::zero();

                        {
                            let d10 = dvec(1, 0);
                            let d11 = dvec(1, 1);
                            let d12 = dvec(1, 2);
                            let d13 = dvec(1, 3);

                            // `C_{00} = D_{11}B_{23} - D_{12}B_{13} + D_{13}B_{12}`.
                            let cof00 = F64x4::cof_pos(d11, b23, d12, b13, d13, b12);
                            det_v = F64x4::madd(det_v, dvec(0, 0), cof00);
                            repl0 = F64x4::madd(repl0, fvec(0, 0), cof00);

                            // `C_{01} = -D_{10}B_{23} + D_{12}B_{03} - D_{13}B_{02}`.
                            let cof01 = F64x4::cof_neg(d10, b23, d12, b03, d13, b02);
                            det_v = F64x4::madd(det_v, dvec(0, 1), cof01);
                            repl0 = F64x4::madd(repl0, fvec(0, 1), cof01);

                            // `C_{02} = D_{10}B_{13} - D_{11}B_{03} + D_{13}B_{01}`.
                            let cof02 = F64x4::cof_pos(d10, b13, d11, b03, d13, b01);
                            det_v = F64x4::madd(det_v, dvec(0, 2), cof02);
                            repl0 = F64x4::madd(repl0, fvec(0, 2), cof02);

                            // `C_{03} = -D_{10}B_{12} + D_{11}B_{02} - D_{12}B_{01}`.
                            let cof03 = F64x4::cof_neg(d10, b12, d11, b02, d12, b01);
                            det_v = F64x4::madd(det_v, dvec(0, 3), cof03);
                            repl0 = F64x4::madd(repl0, fvec(0, 3), cof03);
                        }

                        // Store `\det\mathbf D` early to reduce live registers during the remaining
                        // cofactor evaluation.
                        let mut det_lane = [0.0f64; 4];
                        det_v.store(&mut det_lane);

                        // Reuse the six `B_{ab}` values for row 1 of
                        // `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                        let mut repl1 = F64x4::zero();

                        {
                            let d00 = dvec(0, 0);
                            let d01 = dvec(0, 1);
                            let d02 = dvec(0, 2);
                            let d03 = dvec(0, 3);

                            // `C_{10} = -D_{01}B_{23} + D_{02}B_{13} - D_{03}B_{12}`.
                            let cof10 = F64x4::cof_neg(d01, b23, d02, b13, d03, b12);
                            repl1 = F64x4::madd(repl1, fvec(1, 0), cof10);

                            // `C_{11} = D_{00}B_{23} - D_{02}B_{03} + D_{03}B_{02}`.
                            let cof11 = F64x4::cof_pos(d00, b23, d02, b03, d03, b02);
                            repl1 = F64x4::madd(repl1, fvec(1, 1), cof11);

                            // `C_{12} = -D_{00}B_{13} + D_{01}B_{03} - D_{03}B_{01}`.
                            let cof12 = F64x4::cof_neg(d00, b13, d01, b03, d03, b01);
                            repl1 = F64x4::madd(repl1, fvec(1, 2), cof12);

                            // `C_{13} = D_{00}B_{12} - D_{01}B_{02} + D_{02}B_{01}`.
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

                            // `C_{20} = D_{01}Q_{23} - D_{02}Q_{13} + D_{03}Q_{12}`.
                            let cof20 = F64x4::cof_pos(d01, q23, d02, q13, d03, q12);
                            repl2 = F64x4::madd(repl2, fvec(2, 0), cof20);

                            // `C_{21} = -D_{00}Q_{23} + D_{02}Q_{03} - D_{03}Q_{02}`.
                            let cof21 = F64x4::cof_neg(d00, q23, d02, q03, d03, q02);
                            repl2 = F64x4::madd(repl2, fvec(2, 1), cof21);

                            // `C_{22} = D_{00}Q_{13} - D_{01}Q_{03} + D_{03}Q_{01}`.
                            let cof22 = F64x4::cof_pos(d00, q13, d01, q03, d03, q01);
                            repl2 = F64x4::madd(repl2, fvec(2, 2), cof22);

                            // `C_{23} = -D_{00}Q_{12} + D_{01}Q_{02} - D_{02}Q_{01}`.
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

                            // `C_{30} = -D_{01}R_{23} + D_{02}R_{13} - D_{03}R_{12}`.
                            let cof30 = F64x4::cof_neg(d01, r23, d02, r13, d03, r12);
                            repl3 = F64x4::madd(repl3, fvec(3, 0), cof30);

                            // `C_{31} = D_{00}R_{23} - D_{02}R_{03} + D_{03}R_{02}`.
                            let cof31 = F64x4::cof_pos(d00, r23, d02, r03, d03, r02);
                            repl3 = F64x4::madd(repl3, fvec(3, 1), cof31);

                            // `C_{32} = -D_{00}R_{13} + D_{01}R_{03} - D_{03}R_{01}`.
                            let cof32 = F64x4::cof_neg(d00, r13, d01, r03, d03, r01);
                            repl3 = F64x4::madd(repl3, fvec(3, 2), cof32);

                            // `C_{33} = D_{00}R_{12} - D_{01}R_{02} + D_{02}R_{01}`.
                            let cof33 = F64x4::cof_pos(d00, r12, d01, r02, d02, r01);
                            repl3 = F64x4::madd(repl3, fvec(3, 3), cof33);
                        }

                        // Evaluate `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}` with
                        // `((repl0 + repl1) + (repl2 + repl3))`.
                        let repl23 = F64x4::add(repl2, repl3);
                        let repl_v = F64x4::add(repl01, repl23);
                        let det_v = F64x4::load(&det_lane);

                        (det_v, repl_v)
                    }
                    _ => unreachable!(),
                };

                let pref_v = F64x4::splat(pref);
                let f0_v = F64x4::splat(f0);

                // `S = p\,{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}`.
                let overlap_v = F64x4::mul(det, pref_v);
                // `F = p\,{}^{xw}\tilde S(F_0\det\mathbf D_{\mathrm{ov}} - R)`.
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

/// Evaluate four complex fixed-rank `L = 1,\ldots,4` one-body/overlap factors for `(RX,RW)`.
/// The rank pair and fixed side are compile-time constants, source packets share their hole tuple,
/// and the existing 18-minor cofactor algebra consumes directly packed contraction entries.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `fixed`: Fixed target excitation cache.
/// - `varying`: Four source excitation caches sharing their hole tuple.
/// - `overlap`: Complex overlap output slice in SIMD-lane order.
/// - `fock`: Complex generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes four overlaps and generalised-Fock matrix elements.
/// # Safety
/// - The caller must ensure `T = Complex64`, CPU support for `AVX2/FMA`, valid cached orbital indices,
///   and compile-time ranks satisfying `RX + RW = L` with `1 <= L <= 4`.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_prepared_c64x4_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, T>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
    overlap: &mut [Complex64],
    fock: &mut [Complex64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            unsafe {
                let n = w.n();
                let nocc = w.nocc;
                let nvirt = w.nmo - nocc;
                // `x0` and `y0` store the `X^{(0)}` and `Y^{(0)}` fundamental contractions.
                let x0 = w.x_slice(0).as_ptr().cast::<Complex64>();
                let y0 = w.y_slice(0).as_ptr().cast::<Complex64>();
                // `fsl` stores the one-body column intermediate `\mathcal F^{(0,0)}`.
                let fsl = w.ff_t_slice(0, 0).as_ptr().cast::<Complex64>();
                // `phase` is the reference-pair orbital phase.
                let phase = *std::ptr::from_ref(&w.phase).cast::<Complex64>();
                // `f0 = {}^x F_0^{(0)}` is the scalar one-body intermediate.
                let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<Complex64>();
                // `pref = p\,{}^{xw}\tilde S` is the phase-weighted reduced overlap.
                let pref = phase * w.tilde_s_prod;

                // Select the lane-local bra excitation `{}^x\Psi_{i\cdots}^{a\cdots}`.
                let x_data = |lane: usize| -> &ExcitationSpinCache {
                    if XFIX {
                        fixed
                    } else {
                        varying.get_unchecked(lane)
                    }
                };
                // Select the lane-local ket excitation `{}^w\Psi_{j\cdots}^{b\cdots}`.
                let w_data = |lane: usize| -> &ExcitationSpinCache {
                    if XFIX {
                        varying.get_unchecked(lane)
                    } else {
                        fixed
                    }
                };
                // Rows are `r_\eta \in V_x\cup O_w`, with x-particles before w-holes.
                let row_index = |eta: usize, lane: usize| -> usize {
                    if eta < RX {
                        usize::from(*x_data(lane).indices.get_unchecked(4 + eta)) - nocc
                    } else {
                        nvirt + usize::from(*w_data(lane).indices.get_unchecked(eta - RX))
                    }
                };
                // Columns are `c_z \in O_x\cup V_w`, with x-holes before w-particles.
                let col_index = |z: usize, lane: usize| -> usize {
                    if z < RX {
                        usize::from(*x_data(lane).indices.get_unchecked(z))
                    } else {
                        usize::from(*w_data(lane).indices.get_unchecked(4 + z - RX))
                    }
                };
                // `D_{\eta z} = X^{(0)}_{r_\eta c_z}` for `\eta \geq z`, otherwise
                // `D_{\eta z} = Y^{(0)}_{r_\eta c_z}`.
                let load_d = |eta: usize, z: usize| -> C64x4 {
                    let matrix = if eta >= z { x0 } else { y0 };
                    let invariant = if XFIX { z < RX } else { eta >= RX };

                    // Broadcast lane-invariant entries and gather lane-dependent entries.
                    if invariant {
                        let row = row_index(eta, 0);
                        let col = col_index(z, 0);
                        {
                            let value = *matrix.add(row * n + col);
                            C64x4::splat(value.re, value.im)
                        }
                    } else if XFIX {
                        let row_base = row_index(eta, 0) * n;
                        let col0 = col_index(z, 0);
                        let col1 = col_index(z, 1);
                        let col2 = col_index(z, 2);
                        let col3 = col_index(z, 3);
                        C64x4::from_values(
                            *matrix.add(row_base + col0),
                            *matrix.add(row_base + col1),
                            *matrix.add(row_base + col2),
                            *matrix.add(row_base + col3),
                        )
                    } else {
                        let col = col_index(z, 0);
                        let row0 = row_index(eta, 0);
                        let row1 = row_index(eta, 1);
                        let row2 = row_index(eta, 2);
                        let row3 = row_index(eta, 3);
                        C64x4::from_values(
                            *matrix.add(row0 * n + col),
                            *matrix.add(row1 * n + col),
                            *matrix.add(row2 * n + col),
                            *matrix.add(row3 * n + col),
                        )
                    }
                };
                // `load_f(\eta,z) = \mathcal F_{r_\eta c_z}^{(0,0)}` in each SIMD lane.
                let load_f = |eta: usize, z: usize| -> C64x4 {
                    let invariant = if XFIX { z < RX } else { eta >= RX };

                    // Broadcast lane-invariant intermediates and gather lane-dependent intermediates.
                    if invariant {
                        let row = row_index(eta, 0);
                        let col = col_index(z, 0);
                        {
                            let value = *fsl.add(col * n + row);
                            C64x4::splat(value.re, value.im)
                        }
                    } else if XFIX {
                        let row = row_index(eta, 0);
                        let col0 = col_index(z, 0);
                        let col1 = col_index(z, 1);
                        let col2 = col_index(z, 2);
                        let col3 = col_index(z, 3);
                        C64x4::from_values(
                            *fsl.add(col0 * n + row),
                            *fsl.add(col1 * n + row),
                            *fsl.add(col2 * n + row),
                            *fsl.add(col3 * n + row),
                        )
                    } else {
                        let col_base = col_index(z, 0) * n;
                        let row0 = row_index(eta, 0);
                        let row1 = row_index(eta, 1);
                        let row2 = row_index(eta, 2);
                        let row3 = row_index(eta, 3);
                        C64x4::from_values(
                            *fsl.add(col_base + row0),
                            *fsl.add(col_base + row1),
                            *fsl.add(col_base + row2),
                            *fsl.add(col_base + row3),
                        )
                    }
                };
                let zero = C64x4::zero();
                // Evaluate `\det\mathbf D_{\mathrm{ov}}` and
                // `R = \sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                let (det, repl) = match L {
                    // For `L = 1`, `\det\mathbf D = D_{00}` and `R = \mathcal F_{00}`.
                    1 => (load_d(0, 0), load_f(0, 0)),
                    2 => {
                        let d00 = load_d(0, 0);
                        let d01 = load_d(0, 1);
                        let d10 = load_d(1, 0);
                        let d11 = load_d(1, 1);
                        // `\det\mathbf D = D_{00}D_{11} - D_{01}D_{10}`.
                        let det = C64x4::minor(d00, d11, d01, d10);
                        // `R = \mathcal F_{00}D_{11} - \mathcal F_{01}D_{10}`
                        // `- \mathcal F_{10}D_{01} + \mathcal F_{11}D_{00}`.
                        let mut repl = C64x4::mul(load_f(0, 0), d11);
                        repl = C64x4::msub(repl, load_f(0, 1), d10);
                        repl = C64x4::msub(repl, load_f(1, 0), d01);
                        repl = C64x4::madd(repl, load_f(1, 1), d00);
                        (det, repl)
                    }
                    // For `L = 3`, write `C_{\eta z} = \operatorname{cof}[\mathbf D]_{\eta z}`.
                    3 => {
                        let d00 = load_d(0, 0);
                        let d01 = load_d(0, 1);
                        let d02 = load_d(0, 2);
                        let d10 = load_d(1, 0);
                        let d11 = load_d(1, 1);
                        let d12 = load_d(1, 2);
                        let d20 = load_d(2, 0);
                        let d21 = load_d(2, 1);
                        let d22 = load_d(2, 2);

                        // `C_{00} = D_{11}D_{22} - D_{12}D_{21}`.
                        let c00 = C64x4::minor(d11, d22, d12, d21);
                        // Begin `\det\mathbf D = \sum_z D_{0z}C_{0z}` and
                        // `R = \sum_{\eta z}\mathcal F_{\eta z}C_{\eta z}`.
                        let mut det = C64x4::mul(d00, c00);
                        let mut repl = C64x4::mul(load_f(0, 0), c00);
                        // `C_{01} = -(D_{10}D_{22} - D_{12}D_{20})`.
                        let c01 = C64x4::sub(zero, C64x4::minor(d10, d22, d12, d20));
                        det = C64x4::madd(det, d01, c01);
                        repl = C64x4::madd(repl, load_f(0, 1), c01);
                        // `C_{02} = D_{10}D_{21} - D_{11}D_{20}`.
                        let c02 = C64x4::minor(d10, d21, d11, d20);
                        det = C64x4::madd(det, d02, c02);
                        repl = C64x4::madd(repl, load_f(0, 2), c02);
                        // `C_{10} = -(D_{01}D_{22} - D_{02}D_{21})`.
                        let c10 = C64x4::sub(zero, C64x4::minor(d01, d22, d02, d21));
                        repl = C64x4::madd(repl, load_f(1, 0), c10);
                        // `C_{11} = D_{00}D_{22} - D_{02}D_{20}`.
                        let c11 = C64x4::minor(d00, d22, d02, d20);
                        repl = C64x4::madd(repl, load_f(1, 1), c11);
                        // `C_{12} = -(D_{00}D_{21} - D_{01}D_{20})`.
                        let c12 = C64x4::sub(zero, C64x4::minor(d00, d21, d01, d20));
                        repl = C64x4::madd(repl, load_f(1, 2), c12);
                        // `C_{20} = D_{01}D_{12} - D_{02}D_{11}`.
                        let c20 = C64x4::minor(d01, d12, d02, d11);
                        repl = C64x4::madd(repl, load_f(2, 0), c20);
                        // `C_{21} = -(D_{00}D_{12} - D_{02}D_{10})`.
                        let c21 = C64x4::sub(zero, C64x4::minor(d00, d12, d02, d10));
                        repl = C64x4::madd(repl, load_f(2, 1), c21);
                        // `C_{22} = D_{00}D_{11} - D_{01}D_{10}`.
                        let c22 = C64x4::minor(d00, d11, d01, d10);
                        repl = C64x4::madd(repl, load_f(2, 2), c22);
                        (det, repl)
                    }
                    4 => {
                        let mut d = [C64x4::zero(); 16];
                        for eta in 0..4 {
                            for z in 0..4 {
                                *d.get_unchecked_mut(eta * 4 + z) = load_d(eta, z);
                            }
                        }
                        let dvec = |row: usize, col: usize| *d.get_unchecked(row * 4 + col);
                        // The 16 cofactors contain `16 x 3 = 48` minor occurrences. Factoring their
                        // expansions leaves `6 + 6 + 6 = 18` distinct minors in this DAG.
                        // Complex SIMD values use two registers, so `D_{ij}` is loaded on demand
                        // by row-pair group.

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

                        // Form `\det\mathbf D = \sum_j D_{0j}\operatorname{cof}[\mathbf D]_{0j}` and
                        // row 0 of `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                        {
                            let d10 = dvec(1, 0);
                            let d11 = dvec(1, 1);
                            let d12 = dvec(1, 2);
                            let d13 = dvec(1, 3);

                            // `C_{00} = D_{11}B_{23} - D_{12}B_{13} + D_{13}B_{12}`.
                            let cof00 = C64x4::cof_pos(d11, b23, d12, b13, d13, b12);
                            det = C64x4::madd(det, dvec(0, 0), cof00);
                            repl0 = C64x4::madd(repl0, load_f(0, 0), cof00);

                            // `C_{01} = -D_{10}B_{23} + D_{12}B_{03} - D_{13}B_{02}`.
                            let cof01 = C64x4::cof_neg(d10, b23, d12, b03, d13, b02);
                            det = C64x4::madd(det, dvec(0, 1), cof01);
                            repl0 = C64x4::madd(repl0, load_f(0, 1), cof01);

                            // `C_{02} = D_{10}B_{13} - D_{11}B_{03} + D_{13}B_{01}`.
                            let cof02 = C64x4::cof_pos(d10, b13, d11, b03, d13, b01);
                            det = C64x4::madd(det, dvec(0, 2), cof02);
                            repl0 = C64x4::madd(repl0, load_f(0, 2), cof02);

                            // `C_{03} = -D_{10}B_{12} + D_{11}B_{02} - D_{12}B_{01}`.
                            let cof03 = C64x4::cof_neg(d10, b12, d11, b02, d12, b01);
                            det = C64x4::madd(det, dvec(0, 3), cof03);
                            repl0 = C64x4::madd(repl0, load_f(0, 3), cof03);
                        }

                        // Keep `\det\mathbf D` live while evaluating the remaining cofactor
                        // rows contributing to `R`.
                        // Reuse the six `B_{ab}` values for row 1 of
                        // `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                        {
                            let d00 = dvec(0, 0);
                            let d01 = dvec(0, 1);
                            let d02 = dvec(0, 2);
                            let d03 = dvec(0, 3);

                            // `C_{10} = -D_{01}B_{23} + D_{02}B_{13} - D_{03}B_{12}`.
                            let cof10 = C64x4::cof_neg(d01, b23, d02, b13, d03, b12);
                            repl1 = C64x4::madd(repl1, load_f(1, 0), cof10);

                            // `C_{11} = D_{00}B_{23} - D_{02}B_{03} + D_{03}B_{02}`.
                            let cof11 = C64x4::cof_pos(d00, b23, d02, b03, d03, b02);
                            repl1 = C64x4::madd(repl1, load_f(1, 1), cof11);

                            // `C_{12} = -D_{00}B_{13} + D_{01}B_{03} - D_{03}B_{01}`.
                            let cof12 = C64x4::cof_neg(d00, b13, d01, b03, d03, b01);
                            repl1 = C64x4::madd(repl1, load_f(1, 2), cof12);

                            // `C_{13} = D_{00}B_{12} - D_{01}B_{02} + D_{02}B_{01}`.
                            let cof13 = C64x4::cof_pos(d00, b12, d01, b02, d02, b01);
                            repl1 = C64x4::madd(repl1, load_f(1, 3), cof13);
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

                            // `C_{20} = D_{01}Q_{23} - D_{02}Q_{13} + D_{03}Q_{12}`.
                            let cof20 = C64x4::cof_pos(d01, q23, d02, q13, d03, q12);
                            repl2 = C64x4::madd(repl2, load_f(2, 0), cof20);

                            // `C_{21} = -D_{00}Q_{23} + D_{02}Q_{03} - D_{03}Q_{02}`.
                            let cof21 = C64x4::cof_neg(d00, q23, d02, q03, d03, q02);
                            repl2 = C64x4::madd(repl2, load_f(2, 1), cof21);

                            // `C_{22} = D_{00}Q_{13} - D_{01}Q_{03} + D_{03}Q_{01}`.
                            let cof22 = C64x4::cof_pos(d00, q13, d01, q03, d03, q01);
                            repl2 = C64x4::madd(repl2, load_f(2, 2), cof22);

                            // `C_{23} = -D_{00}Q_{12} + D_{01}Q_{02} - D_{02}Q_{01}`.
                            let cof23 = C64x4::cof_neg(d00, q12, d01, q02, d02, q01);
                            repl2 = C64x4::madd(repl2, load_f(2, 3), cof23);
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

                            // `C_{30} = -D_{01}R_{23} + D_{02}R_{13} - D_{03}R_{12}`.
                            let cof30 = C64x4::cof_neg(d01, r23, d02, r13, d03, r12);
                            repl3 = C64x4::madd(repl3, load_f(3, 0), cof30);

                            // `C_{31} = D_{00}R_{23} - D_{02}R_{03} + D_{03}R_{02}`.
                            let cof31 = C64x4::cof_pos(d00, r23, d02, r03, d03, r02);
                            repl3 = C64x4::madd(repl3, load_f(3, 1), cof31);

                            // `C_{32} = -D_{00}R_{13} + D_{01}R_{03} - D_{03}R_{01}`.
                            let cof32 = C64x4::cof_neg(d00, r13, d01, r03, d03, r01);
                            repl3 = C64x4::madd(repl3, load_f(3, 2), cof32);

                            // `C_{33} = D_{00}R_{12} - D_{01}R_{02} + D_{02}R_{01}`.
                            let cof33 = C64x4::cof_pos(d00, r12, d01, r02, d02, r01);
                            repl3 = C64x4::madd(repl3, load_f(3, 3), cof33);
                        }

                        // Evaluate `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}` with
                        // `((repl0 + repl1) + (repl2 + repl3))`.
                        let repl23 = C64x4::add(repl2, repl3);
                        (det, C64x4::add(repl01, repl23))
                    }
                    _ => unreachable!(),
                };

                let pref_v = C64x4::splat(pref.re, pref.im);
                let f0_v = C64x4::splat(f0.re, f0.im);
                // `S = p\,{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}`.
                let overlap_v = C64x4::mul(pref_v, det);
                // `F = p\,{}^{xw}\tilde S(F_0\det\mathbf D_{\mathrm{ov}} - R)`.
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
                    if det_re.get_unchecked(lane).is_finite()
                        && det_im.get_unchecked(lane).is_finite()
                    {
                        *overlap.get_unchecked_mut(lane) = Complex64::new(
                            *overlap_re.get_unchecked(lane),
                            *overlap_im.get_unchecked(lane),
                        );
                        *fock.get_unchecked_mut(lane) = Complex64::new(
                            *fock_re.get_unchecked(lane),
                            *fock_im.get_unchecked(lane),
                        );
                    } else {
                        *overlap.get_unchecked_mut(lane) = Complex64::new(0.0, 0.0);
                        *fock.get_unchecked_mut(lane) = Complex64::new(0.0, 0.0);
                    }
                }
            }
        }
    )
}

/// Evaluate eight real fixed-rank `L = 1,\ldots,4` one-body/overlap factors for `(RX,RW)`.
/// The rank pair and fixed side are compile-time constants, source packets share their hole tuple,
/// and the existing 18-minor cofactor algebra consumes directly packed contraction entries.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `fixed`: Fixed target excitation cache.
/// - `varying`: Eight source excitation caches sharing their hole tuple.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64`, CPU support for `AVX-512F`, valid cached orbital indices,
///   and compile-time ranks satisfying `RX + RW = L` with `1 <= L <= 4`.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_prepared_f64x8_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, T>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            unsafe {
                let n = w.n();
                let nocc = w.nocc;
                let nvirt = w.nmo - nocc;
                // `x0` and `y0` store the `X^{(0)}` and `Y^{(0)}` fundamental contractions.
                let x0 = w.x_slice(0).as_ptr().cast::<f64>();
                let y0 = w.y_slice(0).as_ptr().cast::<f64>();
                // `fsl` stores the one-body column intermediate `\mathcal F^{(0,0)}`.
                let fsl = w.ff_t_slice(0, 0).as_ptr().cast::<f64>();
                // `phase` is the reference-pair orbital phase.
                let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
                // `f0 = {}^x F_0^{(0)}` is the scalar one-body intermediate.
                let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
                // `pref = p\,{}^{xw}\tilde S` is the phase-weighted reduced overlap.
                let pref = phase * w.tilde_s_prod;

                // Select the lane-local bra excitation `{}^x\Psi_{i\cdots}^{a\cdots}`.
                let x_data = |lane: usize| -> &ExcitationSpinCache {
                    if XFIX {
                        fixed
                    } else {
                        varying.get_unchecked(lane)
                    }
                };
                // Select the lane-local ket excitation `{}^w\Psi_{j\cdots}^{b\cdots}`.
                let w_data = |lane: usize| -> &ExcitationSpinCache {
                    if XFIX {
                        varying.get_unchecked(lane)
                    } else {
                        fixed
                    }
                };
                // Rows are `r_\eta \in V_x\cup O_w`, with x-particles before w-holes.
                let row_index = |eta: usize, lane: usize| -> usize {
                    if eta < RX {
                        usize::from(*x_data(lane).indices.get_unchecked(4 + eta)) - nocc
                    } else {
                        nvirt + usize::from(*w_data(lane).indices.get_unchecked(eta - RX))
                    }
                };
                // Columns are `c_z \in O_x\cup V_w`, with x-holes before w-particles.
                let col_index = |z: usize, lane: usize| -> usize {
                    if z < RX {
                        usize::from(*x_data(lane).indices.get_unchecked(z))
                    } else {
                        usize::from(*w_data(lane).indices.get_unchecked(4 + z - RX))
                    }
                };
                // `D_{\eta z} = X^{(0)}_{r_\eta c_z}` for `\eta \geq z`, otherwise
                // `D_{\eta z} = Y^{(0)}_{r_\eta c_z}`.
                let load_d = |eta: usize, z: usize| -> F64x8 {
                    let matrix = if eta >= z { x0 } else { y0 };
                    let invariant = if XFIX { z < RX } else { eta >= RX };

                    // Broadcast lane-invariant entries and gather lane-dependent entries.
                    if invariant {
                        let row = row_index(eta, 0);
                        let col = col_index(z, 0);
                        F64x8::splat(*matrix.add(row * n + col))
                    } else if XFIX {
                        let row_base = row_index(eta, 0) * n;
                        let col0 = col_index(z, 0);
                        let col1 = col_index(z, 1);
                        let col2 = col_index(z, 2);
                        let col3 = col_index(z, 3);
                        let col4 = col_index(z, 4);
                        let col5 = col_index(z, 5);
                        let col6 = col_index(z, 6);
                        let col7 = col_index(z, 7);
                        F64x8::from_values([
                            *matrix.add(row_base + col0),
                            *matrix.add(row_base + col1),
                            *matrix.add(row_base + col2),
                            *matrix.add(row_base + col3),
                            *matrix.add(row_base + col4),
                            *matrix.add(row_base + col5),
                            *matrix.add(row_base + col6),
                            *matrix.add(row_base + col7),
                        ])
                    } else {
                        let col = col_index(z, 0);
                        let row0 = row_index(eta, 0);
                        let row1 = row_index(eta, 1);
                        let row2 = row_index(eta, 2);
                        let row3 = row_index(eta, 3);
                        let row4 = row_index(eta, 4);
                        let row5 = row_index(eta, 5);
                        let row6 = row_index(eta, 6);
                        let row7 = row_index(eta, 7);
                        F64x8::from_values([
                            *matrix.add(row0 * n + col),
                            *matrix.add(row1 * n + col),
                            *matrix.add(row2 * n + col),
                            *matrix.add(row3 * n + col),
                            *matrix.add(row4 * n + col),
                            *matrix.add(row5 * n + col),
                            *matrix.add(row6 * n + col),
                            *matrix.add(row7 * n + col),
                        ])
                    }
                };
                // `fvec(\eta,z) = \mathcal F_{r_\eta c_z}^{(0,0)}` in each SIMD lane.
                let fvec = |eta: usize, z: usize| -> F64x8 {
                    let invariant = if XFIX { z < RX } else { eta >= RX };

                    // Broadcast lane-invariant intermediates and gather lane-dependent intermediates.
                    if invariant {
                        let row = row_index(eta, 0);
                        let col = col_index(z, 0);
                        F64x8::splat(*fsl.add(col * n + row))
                    } else if XFIX {
                        let row = row_index(eta, 0);
                        let col0 = col_index(z, 0);
                        let col1 = col_index(z, 1);
                        let col2 = col_index(z, 2);
                        let col3 = col_index(z, 3);
                        let col4 = col_index(z, 4);
                        let col5 = col_index(z, 5);
                        let col6 = col_index(z, 6);
                        let col7 = col_index(z, 7);
                        F64x8::from_values([
                            *fsl.add(col0 * n + row),
                            *fsl.add(col1 * n + row),
                            *fsl.add(col2 * n + row),
                            *fsl.add(col3 * n + row),
                            *fsl.add(col4 * n + row),
                            *fsl.add(col5 * n + row),
                            *fsl.add(col6 * n + row),
                            *fsl.add(col7 * n + row),
                        ])
                    } else {
                        let col_base = col_index(z, 0) * n;
                        let row0 = row_index(eta, 0);
                        let row1 = row_index(eta, 1);
                        let row2 = row_index(eta, 2);
                        let row3 = row_index(eta, 3);
                        let row4 = row_index(eta, 4);
                        let row5 = row_index(eta, 5);
                        let row6 = row_index(eta, 6);
                        let row7 = row_index(eta, 7);
                        F64x8::from_values([
                            *fsl.add(col_base + row0),
                            *fsl.add(col_base + row1),
                            *fsl.add(col_base + row2),
                            *fsl.add(col_base + row3),
                            *fsl.add(col_base + row4),
                            *fsl.add(col_base + row5),
                            *fsl.add(col_base + row6),
                            *fsl.add(col_base + row7),
                        ])
                    }
                };
                let zero = F64x8::zero();
                // Evaluate `\det\mathbf D_{\mathrm{ov}}` and
                // `R = \sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                let (det, repl) = match L {
                    // For `L = 1`, `\det\mathbf D = D_{00}` and `R = \mathcal F_{00}`.
                    1 => (load_d(0, 0), fvec(0, 0)),
                    2 => {
                        let d00 = load_d(0, 0);
                        let d01 = load_d(0, 1);
                        let d10 = load_d(1, 0);
                        let d11 = load_d(1, 1);
                        // `\det\mathbf D = D_{00}D_{11} - D_{01}D_{10}`.
                        let det = F64x8::minor(d00, d11, d01, d10);
                        // `R = \mathcal F_{00}D_{11} - \mathcal F_{01}D_{10}`
                        // `- \mathcal F_{10}D_{01} + \mathcal F_{11}D_{00}`.
                        let mut repl = F64x8::mul(fvec(0, 0), d11);
                        repl = F64x8::msub(repl, fvec(0, 1), d10);
                        repl = F64x8::msub(repl, fvec(1, 0), d01);
                        repl = F64x8::madd(repl, fvec(1, 1), d00);
                        (det, repl)
                    }
                    // For `L = 3`, write `C_{\eta z} = \operatorname{cof}[\mathbf D]_{\eta z}`.
                    3 => {
                        let d00 = load_d(0, 0);
                        let d01 = load_d(0, 1);
                        let d02 = load_d(0, 2);
                        let d10 = load_d(1, 0);
                        let d11 = load_d(1, 1);
                        let d12 = load_d(1, 2);
                        let d20 = load_d(2, 0);
                        let d21 = load_d(2, 1);
                        let d22 = load_d(2, 2);

                        // `C_{00} = D_{11}D_{22} - D_{12}D_{21}`.
                        let c00 = F64x8::minor(d11, d22, d12, d21);
                        // Begin `\det\mathbf D = \sum_z D_{0z}C_{0z}` and
                        // `R = \sum_{\eta z}\mathcal F_{\eta z}C_{\eta z}`.
                        let mut det = F64x8::mul(d00, c00);
                        let mut repl = F64x8::mul(fvec(0, 0), c00);
                        // `C_{01} = -(D_{10}D_{22} - D_{12}D_{20})`.
                        let c01 = F64x8::sub(zero, F64x8::minor(d10, d22, d12, d20));
                        det = F64x8::madd(det, d01, c01);
                        repl = F64x8::madd(repl, fvec(0, 1), c01);
                        // `C_{02} = D_{10}D_{21} - D_{11}D_{20}`.
                        let c02 = F64x8::minor(d10, d21, d11, d20);
                        det = F64x8::madd(det, d02, c02);
                        repl = F64x8::madd(repl, fvec(0, 2), c02);
                        // `C_{10} = -(D_{01}D_{22} - D_{02}D_{21})`.
                        let c10 = F64x8::sub(zero, F64x8::minor(d01, d22, d02, d21));
                        repl = F64x8::madd(repl, fvec(1, 0), c10);
                        // `C_{11} = D_{00}D_{22} - D_{02}D_{20}`.
                        let c11 = F64x8::minor(d00, d22, d02, d20);
                        repl = F64x8::madd(repl, fvec(1, 1), c11);
                        // `C_{12} = -(D_{00}D_{21} - D_{01}D_{20})`.
                        let c12 = F64x8::sub(zero, F64x8::minor(d00, d21, d01, d20));
                        repl = F64x8::madd(repl, fvec(1, 2), c12);
                        // `C_{20} = D_{01}D_{12} - D_{02}D_{11}`.
                        let c20 = F64x8::minor(d01, d12, d02, d11);
                        repl = F64x8::madd(repl, fvec(2, 0), c20);
                        // `C_{21} = -(D_{00}D_{12} - D_{02}D_{10})`.
                        let c21 = F64x8::sub(zero, F64x8::minor(d00, d12, d02, d10));
                        repl = F64x8::madd(repl, fvec(2, 1), c21);
                        // `C_{22} = D_{00}D_{11} - D_{01}D_{10}`.
                        let c22 = F64x8::minor(d00, d11, d01, d10);
                        repl = F64x8::madd(repl, fvec(2, 2), c22);
                        (det, repl)
                    }
                    4 => {
                        let mut d = [F64x8::zero(); 16];
                        for eta in 0..4 {
                            for z in 0..4 {
                                *d.get_unchecked_mut(eta * 4 + z) = load_d(eta, z);
                            }
                        }
                        let dvec = |row: usize, col: usize| *d.get_unchecked(row * 4 + col);

                        // The 16 cofactors contain `16 x 3 = 48` minor occurrences. Factoring their
                        // expansions leaves `6 + 6 + 6 = 18` distinct minors in this DAG.
                        // Reload `D_{ij}` by group to keep the 18 minor intermediates within the
                        // available register budget.

                        // `B_{ab} = D_{2a}D_{3b} - D_{2b}D_{3a}` supplies cofactor rows 0 and 1.
                        let (b01, b02, b03, b12, b13, b23) = {
                            let d20 = dvec(2, 0);
                            let d21 = dvec(2, 1);
                            let d22 = dvec(2, 2);
                            let d23 = dvec(2, 3);

                            let d30 = dvec(3, 0);
                            let d31 = dvec(3, 1);

                            let b01 = F64x8::minor(d20, d31, d21, d30);

                            let d32 = dvec(3, 2);

                            let b02 = F64x8::minor(d20, d32, d22, d30);
                            let b12 = F64x8::minor(d21, d32, d22, d31);

                            let d33 = dvec(3, 3);

                            let b03 = F64x8::minor(d20, d33, d23, d30);
                            let b13 = F64x8::minor(d21, d33, d23, d31);
                            let b23 = F64x8::minor(d22, d33, d23, d32);

                            (b01, b02, b03, b12, b13, b23)
                        };

                        // Form `\det\mathbf D = \sum_j D_{0j}\operatorname{cof}[\mathbf D]_{0j}` and
                        // row 0 of `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                        let mut det_v = F64x8::zero();
                        let mut repl0 = F64x8::zero();

                        {
                            let d10 = dvec(1, 0);
                            let d11 = dvec(1, 1);
                            let d12 = dvec(1, 2);
                            let d13 = dvec(1, 3);

                            // `C_{00} = D_{11}B_{23} - D_{12}B_{13} + D_{13}B_{12}`.
                            let cof00 = F64x8::cof_pos(d11, b23, d12, b13, d13, b12);
                            det_v = F64x8::madd(det_v, dvec(0, 0), cof00);
                            repl0 = F64x8::madd(repl0, fvec(0, 0), cof00);

                            // `C_{01} = -D_{10}B_{23} + D_{12}B_{03} - D_{13}B_{02}`.
                            let cof01 = F64x8::cof_neg(d10, b23, d12, b03, d13, b02);
                            det_v = F64x8::madd(det_v, dvec(0, 1), cof01);
                            repl0 = F64x8::madd(repl0, fvec(0, 1), cof01);

                            // `C_{02} = D_{10}B_{13} - D_{11}B_{03} + D_{13}B_{01}`.
                            let cof02 = F64x8::cof_pos(d10, b13, d11, b03, d13, b01);
                            det_v = F64x8::madd(det_v, dvec(0, 2), cof02);
                            repl0 = F64x8::madd(repl0, fvec(0, 2), cof02);

                            // `C_{03} = -D_{10}B_{12} + D_{11}B_{02} - D_{12}B_{01}`.
                            let cof03 = F64x8::cof_neg(d10, b12, d11, b02, d12, b01);
                            det_v = F64x8::madd(det_v, dvec(0, 3), cof03);
                            repl0 = F64x8::madd(repl0, fvec(0, 3), cof03);
                        }

                        // Store `\det\mathbf D` early to reduce live registers during the remaining
                        // cofactor evaluation.
                        let mut det_lane = [0.0f64; 8];
                        det_v.store(&mut det_lane);

                        // Reuse the six `B_{ab}` values for row 1 of
                        // `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                        let mut repl1 = F64x8::zero();

                        {
                            let d00 = dvec(0, 0);
                            let d01 = dvec(0, 1);
                            let d02 = dvec(0, 2);
                            let d03 = dvec(0, 3);

                            // `C_{10} = -D_{01}B_{23} + D_{02}B_{13} - D_{03}B_{12}`.
                            let cof10 = F64x8::cof_neg(d01, b23, d02, b13, d03, b12);
                            repl1 = F64x8::madd(repl1, fvec(1, 0), cof10);

                            // `C_{11} = D_{00}B_{23} - D_{02}B_{03} + D_{03}B_{02}`.
                            let cof11 = F64x8::cof_pos(d00, b23, d02, b03, d03, b02);
                            repl1 = F64x8::madd(repl1, fvec(1, 1), cof11);

                            // `C_{12} = -D_{00}B_{13} + D_{01}B_{03} - D_{03}B_{01}`.
                            let cof12 = F64x8::cof_neg(d00, b13, d01, b03, d03, b01);
                            repl1 = F64x8::madd(repl1, fvec(1, 2), cof12);

                            // `C_{13} = D_{00}B_{12} - D_{01}B_{02} + D_{02}B_{01}`.
                            let cof13 = F64x8::cof_pos(d00, b12, d01, b02, d02, b01);
                            repl1 = F64x8::madd(repl1, fvec(1, 3), cof13);
                        }

                        let repl01 = F64x8::add(repl0, repl1);

                        // `Q_{ab} = D_{1a}D_{3b} - D_{1b}D_{3a}` supplies cofactor row 2.
                        let (q01, q02, q03, q12, q13, q23) = {
                            let d10 = dvec(1, 0);
                            let d11 = dvec(1, 1);
                            let d12 = dvec(1, 2);
                            let d13 = dvec(1, 3);

                            let d30 = dvec(3, 0);
                            let d31 = dvec(3, 1);

                            let q01 = F64x8::minor(d10, d31, d11, d30);

                            let d32 = dvec(3, 2);

                            let q02 = F64x8::minor(d10, d32, d12, d30);
                            let q12 = F64x8::minor(d11, d32, d12, d31);

                            let d33 = dvec(3, 3);

                            let q03 = F64x8::minor(d10, d33, d13, d30);
                            let q13 = F64x8::minor(d11, d33, d13, d31);
                            let q23 = F64x8::minor(d12, d33, d13, d32);

                            (q01, q02, q03, q12, q13, q23)
                        };

                        let mut repl2 = F64x8::zero();

                        {
                            let d00 = dvec(0, 0);
                            let d01 = dvec(0, 1);
                            let d02 = dvec(0, 2);
                            let d03 = dvec(0, 3);

                            // `C_{20} = D_{01}Q_{23} - D_{02}Q_{13} + D_{03}Q_{12}`.
                            let cof20 = F64x8::cof_pos(d01, q23, d02, q13, d03, q12);
                            repl2 = F64x8::madd(repl2, fvec(2, 0), cof20);

                            // `C_{21} = -D_{00}Q_{23} + D_{02}Q_{03} - D_{03}Q_{02}`.
                            let cof21 = F64x8::cof_neg(d00, q23, d02, q03, d03, q02);
                            repl2 = F64x8::madd(repl2, fvec(2, 1), cof21);

                            // `C_{22} = D_{00}Q_{13} - D_{01}Q_{03} + D_{03}Q_{01}`.
                            let cof22 = F64x8::cof_pos(d00, q13, d01, q03, d03, q01);
                            repl2 = F64x8::madd(repl2, fvec(2, 2), cof22);

                            // `C_{23} = -D_{00}Q_{12} + D_{01}Q_{02} - D_{02}Q_{01}`.
                            let cof23 = F64x8::cof_neg(d00, q12, d01, q02, d02, q01);
                            repl2 = F64x8::madd(repl2, fvec(2, 3), cof23);
                        }

                        // `R_{ab} = D_{1a}D_{2b} - D_{1b}D_{2a}` supplies cofactor row 3.
                        let (r01, r02, r03, r12, r13, r23) = {
                            let d10 = dvec(1, 0);
                            let d11 = dvec(1, 1);
                            let d12 = dvec(1, 2);
                            let d13 = dvec(1, 3);

                            let d20 = dvec(2, 0);
                            let d21 = dvec(2, 1);

                            let r01 = F64x8::minor(d10, d21, d11, d20);

                            let d22 = dvec(2, 2);

                            let r02 = F64x8::minor(d10, d22, d12, d20);
                            let r12 = F64x8::minor(d11, d22, d12, d21);

                            let d23 = dvec(2, 3);

                            let r03 = F64x8::minor(d10, d23, d13, d20);
                            let r13 = F64x8::minor(d11, d23, d13, d21);
                            let r23 = F64x8::minor(d12, d23, d13, d22);

                            (r01, r02, r03, r12, r13, r23)
                        };

                        let mut repl3 = F64x8::zero();

                        {
                            let d00 = dvec(0, 0);
                            let d01 = dvec(0, 1);
                            let d02 = dvec(0, 2);
                            let d03 = dvec(0, 3);

                            // `C_{30} = -D_{01}R_{23} + D_{02}R_{13} - D_{03}R_{12}`.
                            let cof30 = F64x8::cof_neg(d01, r23, d02, r13, d03, r12);
                            repl3 = F64x8::madd(repl3, fvec(3, 0), cof30);

                            // `C_{31} = D_{00}R_{23} - D_{02}R_{03} + D_{03}R_{02}`.
                            let cof31 = F64x8::cof_pos(d00, r23, d02, r03, d03, r02);
                            repl3 = F64x8::madd(repl3, fvec(3, 1), cof31);

                            // `C_{32} = -D_{00}R_{13} + D_{01}R_{03} - D_{03}R_{01}`.
                            let cof32 = F64x8::cof_neg(d00, r13, d01, r03, d03, r01);
                            repl3 = F64x8::madd(repl3, fvec(3, 2), cof32);

                            // `C_{33} = D_{00}R_{12} - D_{01}R_{02} + D_{02}R_{01}`.
                            let cof33 = F64x8::cof_pos(d00, r12, d01, r02, d02, r01);
                            repl3 = F64x8::madd(repl3, fvec(3, 3), cof33);
                        }

                        // Evaluate `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}` with
                        // `((repl0 + repl1) + (repl2 + repl3))`.
                        let repl23 = F64x8::add(repl2, repl3);
                        let repl_v = F64x8::add(repl01, repl23);
                        let det_v = F64x8::load(&det_lane);

                        (det_v, repl_v)
                    }
                    _ => unreachable!(),
                };

                let pref_v = F64x8::splat(pref);
                let f0_v = F64x8::splat(f0);

                // `S = p\,{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}`.
                let overlap_v = F64x8::mul(det, pref_v);
                // `F = p\,{}^{xw}\tilde S(F_0\det\mathbf D_{\mathrm{ov}} - R)`.
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
/// Evaluate eight complex fixed-rank `L = 1,\ldots,4` one-body/overlap factors for `(RX,RW)`.
/// The rank pair and fixed side are compile-time constants, source packets share their hole tuple,
/// and the existing 18-minor cofactor algebra consumes directly packed contraction entries.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `fixed`: Fixed target excitation cache.
/// - `varying`: Eight source excitation caches sharing their hole tuple.
/// - `overlap`: Complex overlap output slice in SIMD-lane order.
/// - `fock`: Complex generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes eight overlaps and generalised-Fock matrix elements.
/// # Safety
/// - The caller must ensure `T = Complex64`, CPU support for `AVX-512F`, valid cached orbital indices,
///   and compile-time ranks satisfying `RX + RW = L` with `1 <= L <= 4`.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_prepared_c64x8_const<
    T: NOCIScalar,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, T>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
    overlap: &mut [Complex64],
    fock: &mut [Complex64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            unsafe {
                let n = w.n();
                let nocc = w.nocc;
                let nvirt = w.nmo - nocc;
                // `x0` and `y0` store the `X^{(0)}` and `Y^{(0)}` fundamental contractions.
                let x0 = w.x_slice(0).as_ptr().cast::<Complex64>();
                let y0 = w.y_slice(0).as_ptr().cast::<Complex64>();
                // `fsl` stores the one-body column intermediate `\mathcal F^{(0,0)}`.
                let fsl = w.ff_t_slice(0, 0).as_ptr().cast::<Complex64>();
                // `phase` is the reference-pair orbital phase.
                let phase = *std::ptr::from_ref(&w.phase).cast::<Complex64>();
                // `f0 = {}^x F_0^{(0)}` is the scalar one-body intermediate.
                let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<Complex64>();
                // `pref = p\,{}^{xw}\tilde S` is the phase-weighted reduced overlap.
                let pref = phase * w.tilde_s_prod;

                // Select the lane-local bra excitation `{}^x\Psi_{i\cdots}^{a\cdots}`.
                let x_data = |lane: usize| -> &ExcitationSpinCache {
                    if XFIX {
                        fixed
                    } else {
                        varying.get_unchecked(lane)
                    }
                };
                // Select the lane-local ket excitation `{}^w\Psi_{j\cdots}^{b\cdots}`.
                let w_data = |lane: usize| -> &ExcitationSpinCache {
                    if XFIX {
                        varying.get_unchecked(lane)
                    } else {
                        fixed
                    }
                };
                // Rows are `r_\eta \in V_x\cup O_w`, with x-particles before w-holes.
                let row_index = |eta: usize, lane: usize| -> usize {
                    if eta < RX {
                        usize::from(*x_data(lane).indices.get_unchecked(4 + eta)) - nocc
                    } else {
                        nvirt + usize::from(*w_data(lane).indices.get_unchecked(eta - RX))
                    }
                };
                // Columns are `c_z \in O_x\cup V_w`, with x-holes before w-particles.
                let col_index = |z: usize, lane: usize| -> usize {
                    if z < RX {
                        usize::from(*x_data(lane).indices.get_unchecked(z))
                    } else {
                        usize::from(*w_data(lane).indices.get_unchecked(4 + z - RX))
                    }
                };
                // `D_{\eta z} = X^{(0)}_{r_\eta c_z}` for `\eta \geq z`, otherwise
                // `D_{\eta z} = Y^{(0)}_{r_\eta c_z}`.
                let load_d = |eta: usize, z: usize| -> C64x8 {
                    let matrix = if eta >= z { x0 } else { y0 };
                    let invariant = if XFIX { z < RX } else { eta >= RX };

                    // Broadcast lane-invariant entries and gather lane-dependent entries.
                    if invariant {
                        let row = row_index(eta, 0);
                        let col = col_index(z, 0);
                        {
                            let value = *matrix.add(row * n + col);
                            C64x8::splat(value.re, value.im)
                        }
                    } else if XFIX {
                        let row_base = row_index(eta, 0) * n;
                        let col0 = col_index(z, 0);
                        let col1 = col_index(z, 1);
                        let col2 = col_index(z, 2);
                        let col3 = col_index(z, 3);
                        let col4 = col_index(z, 4);
                        let col5 = col_index(z, 5);
                        let col6 = col_index(z, 6);
                        let col7 = col_index(z, 7);
                        C64x8::from_values([
                            *matrix.add(row_base + col0),
                            *matrix.add(row_base + col1),
                            *matrix.add(row_base + col2),
                            *matrix.add(row_base + col3),
                            *matrix.add(row_base + col4),
                            *matrix.add(row_base + col5),
                            *matrix.add(row_base + col6),
                            *matrix.add(row_base + col7),
                        ])
                    } else {
                        let col = col_index(z, 0);
                        let row0 = row_index(eta, 0);
                        let row1 = row_index(eta, 1);
                        let row2 = row_index(eta, 2);
                        let row3 = row_index(eta, 3);
                        let row4 = row_index(eta, 4);
                        let row5 = row_index(eta, 5);
                        let row6 = row_index(eta, 6);
                        let row7 = row_index(eta, 7);
                        C64x8::from_values([
                            *matrix.add(row0 * n + col),
                            *matrix.add(row1 * n + col),
                            *matrix.add(row2 * n + col),
                            *matrix.add(row3 * n + col),
                            *matrix.add(row4 * n + col),
                            *matrix.add(row5 * n + col),
                            *matrix.add(row6 * n + col),
                            *matrix.add(row7 * n + col),
                        ])
                    }
                };
                // `load_f(\eta,z) = \mathcal F_{r_\eta c_z}^{(0,0)}` in each SIMD lane.
                let load_f = |eta: usize, z: usize| -> C64x8 {
                    let invariant = if XFIX { z < RX } else { eta >= RX };

                    // Broadcast lane-invariant intermediates and gather lane-dependent intermediates.
                    if invariant {
                        let row = row_index(eta, 0);
                        let col = col_index(z, 0);
                        {
                            let value = *fsl.add(col * n + row);
                            C64x8::splat(value.re, value.im)
                        }
                    } else if XFIX {
                        let row = row_index(eta, 0);
                        let col0 = col_index(z, 0);
                        let col1 = col_index(z, 1);
                        let col2 = col_index(z, 2);
                        let col3 = col_index(z, 3);
                        let col4 = col_index(z, 4);
                        let col5 = col_index(z, 5);
                        let col6 = col_index(z, 6);
                        let col7 = col_index(z, 7);
                        C64x8::from_values([
                            *fsl.add(col0 * n + row),
                            *fsl.add(col1 * n + row),
                            *fsl.add(col2 * n + row),
                            *fsl.add(col3 * n + row),
                            *fsl.add(col4 * n + row),
                            *fsl.add(col5 * n + row),
                            *fsl.add(col6 * n + row),
                            *fsl.add(col7 * n + row),
                        ])
                    } else {
                        let col_base = col_index(z, 0) * n;
                        let row0 = row_index(eta, 0);
                        let row1 = row_index(eta, 1);
                        let row2 = row_index(eta, 2);
                        let row3 = row_index(eta, 3);
                        let row4 = row_index(eta, 4);
                        let row5 = row_index(eta, 5);
                        let row6 = row_index(eta, 6);
                        let row7 = row_index(eta, 7);
                        C64x8::from_values([
                            *fsl.add(col_base + row0),
                            *fsl.add(col_base + row1),
                            *fsl.add(col_base + row2),
                            *fsl.add(col_base + row3),
                            *fsl.add(col_base + row4),
                            *fsl.add(col_base + row5),
                            *fsl.add(col_base + row6),
                            *fsl.add(col_base + row7),
                        ])
                    }
                };
                let zero = C64x8::zero();
                // Evaluate `\det\mathbf D_{\mathrm{ov}}` and
                // `R = \sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                let (det, repl) = match L {
                    // For `L = 1`, `\det\mathbf D = D_{00}` and `R = \mathcal F_{00}`.
                    1 => (load_d(0, 0), load_f(0, 0)),
                    2 => {
                        let d00 = load_d(0, 0);
                        let d01 = load_d(0, 1);
                        let d10 = load_d(1, 0);
                        let d11 = load_d(1, 1);
                        // `\det\mathbf D = D_{00}D_{11} - D_{01}D_{10}`.
                        let det = C64x8::minor(d00, d11, d01, d10);
                        // `R = \mathcal F_{00}D_{11} - \mathcal F_{01}D_{10}`
                        // `- \mathcal F_{10}D_{01} + \mathcal F_{11}D_{00}`.
                        let mut repl = C64x8::mul(load_f(0, 0), d11);
                        repl = C64x8::msub(repl, load_f(0, 1), d10);
                        repl = C64x8::msub(repl, load_f(1, 0), d01);
                        repl = C64x8::madd(repl, load_f(1, 1), d00);
                        (det, repl)
                    }
                    // For `L = 3`, write `C_{\eta z} = \operatorname{cof}[\mathbf D]_{\eta z}`.
                    3 => {
                        let d00 = load_d(0, 0);
                        let d01 = load_d(0, 1);
                        let d02 = load_d(0, 2);
                        let d10 = load_d(1, 0);
                        let d11 = load_d(1, 1);
                        let d12 = load_d(1, 2);
                        let d20 = load_d(2, 0);
                        let d21 = load_d(2, 1);
                        let d22 = load_d(2, 2);

                        // `C_{00} = D_{11}D_{22} - D_{12}D_{21}`.
                        let c00 = C64x8::minor(d11, d22, d12, d21);
                        // Begin `\det\mathbf D = \sum_z D_{0z}C_{0z}` and
                        // `R = \sum_{\eta z}\mathcal F_{\eta z}C_{\eta z}`.
                        let mut det = C64x8::mul(d00, c00);
                        let mut repl = C64x8::mul(load_f(0, 0), c00);
                        // `C_{01} = -(D_{10}D_{22} - D_{12}D_{20})`.
                        let c01 = C64x8::sub(zero, C64x8::minor(d10, d22, d12, d20));
                        det = C64x8::madd(det, d01, c01);
                        repl = C64x8::madd(repl, load_f(0, 1), c01);
                        // `C_{02} = D_{10}D_{21} - D_{11}D_{20}`.
                        let c02 = C64x8::minor(d10, d21, d11, d20);
                        det = C64x8::madd(det, d02, c02);
                        repl = C64x8::madd(repl, load_f(0, 2), c02);
                        // `C_{10} = -(D_{01}D_{22} - D_{02}D_{21})`.
                        let c10 = C64x8::sub(zero, C64x8::minor(d01, d22, d02, d21));
                        repl = C64x8::madd(repl, load_f(1, 0), c10);
                        // `C_{11} = D_{00}D_{22} - D_{02}D_{20}`.
                        let c11 = C64x8::minor(d00, d22, d02, d20);
                        repl = C64x8::madd(repl, load_f(1, 1), c11);
                        // `C_{12} = -(D_{00}D_{21} - D_{01}D_{20})`.
                        let c12 = C64x8::sub(zero, C64x8::minor(d00, d21, d01, d20));
                        repl = C64x8::madd(repl, load_f(1, 2), c12);
                        // `C_{20} = D_{01}D_{12} - D_{02}D_{11}`.
                        let c20 = C64x8::minor(d01, d12, d02, d11);
                        repl = C64x8::madd(repl, load_f(2, 0), c20);
                        // `C_{21} = -(D_{00}D_{12} - D_{02}D_{10})`.
                        let c21 = C64x8::sub(zero, C64x8::minor(d00, d12, d02, d10));
                        repl = C64x8::madd(repl, load_f(2, 1), c21);
                        // `C_{22} = D_{00}D_{11} - D_{01}D_{10}`.
                        let c22 = C64x8::minor(d00, d11, d01, d10);
                        repl = C64x8::madd(repl, load_f(2, 2), c22);
                        (det, repl)
                    }
                    4 => {
                        let mut d = [C64x8::zero(); 16];
                        for eta in 0..4 {
                            for z in 0..4 {
                                *d.get_unchecked_mut(eta * 4 + z) = load_d(eta, z);
                            }
                        }
                        let dvec = |row: usize, col: usize| *d.get_unchecked(row * 4 + col);
                        // The 16 cofactors contain `16 x 3 = 48` minor occurrences. Factoring their
                        // expansions leaves `6 + 6 + 6 = 18` distinct minors in this DAG.
                        // Complex SIMD values use two registers, so `D_{ij}` is loaded on demand
                        // by row-pair group.

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

                        // Form `\det\mathbf D = \sum_j D_{0j}\operatorname{cof}[\mathbf D]_{0j}` and
                        // row 0 of `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                        {
                            let d10 = dvec(1, 0);
                            let d11 = dvec(1, 1);
                            let d12 = dvec(1, 2);
                            let d13 = dvec(1, 3);

                            // `C_{00} = D_{11}B_{23} - D_{12}B_{13} + D_{13}B_{12}`.
                            let cof00 = C64x8::cof_pos(d11, b23, d12, b13, d13, b12);
                            det = C64x8::madd(det, dvec(0, 0), cof00);
                            repl0 = C64x8::madd(repl0, load_f(0, 0), cof00);

                            // `C_{01} = -D_{10}B_{23} + D_{12}B_{03} - D_{13}B_{02}`.
                            let cof01 = C64x8::cof_neg(d10, b23, d12, b03, d13, b02);
                            det = C64x8::madd(det, dvec(0, 1), cof01);
                            repl0 = C64x8::madd(repl0, load_f(0, 1), cof01);

                            // `C_{02} = D_{10}B_{13} - D_{11}B_{03} + D_{13}B_{01}`.
                            let cof02 = C64x8::cof_pos(d10, b13, d11, b03, d13, b01);
                            det = C64x8::madd(det, dvec(0, 2), cof02);
                            repl0 = C64x8::madd(repl0, load_f(0, 2), cof02);

                            // `C_{03} = -D_{10}B_{12} + D_{11}B_{02} - D_{12}B_{01}`.
                            let cof03 = C64x8::cof_neg(d10, b12, d11, b02, d12, b01);
                            det = C64x8::madd(det, dvec(0, 3), cof03);
                            repl0 = C64x8::madd(repl0, load_f(0, 3), cof03);
                        }

                        // Keep `\det\mathbf D` live while evaluating the remaining cofactor
                        // rows contributing to `R`.
                        // Reuse the six `B_{ab}` values for row 1 of
                        // `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}`.
                        {
                            let d00 = dvec(0, 0);
                            let d01 = dvec(0, 1);
                            let d02 = dvec(0, 2);
                            let d03 = dvec(0, 3);

                            // `C_{10} = -D_{01}B_{23} + D_{02}B_{13} - D_{03}B_{12}`.
                            let cof10 = C64x8::cof_neg(d01, b23, d02, b13, d03, b12);
                            repl1 = C64x8::madd(repl1, load_f(1, 0), cof10);

                            // `C_{11} = D_{00}B_{23} - D_{02}B_{03} + D_{03}B_{02}`.
                            let cof11 = C64x8::cof_pos(d00, b23, d02, b03, d03, b02);
                            repl1 = C64x8::madd(repl1, load_f(1, 1), cof11);

                            // `C_{12} = -D_{00}B_{13} + D_{01}B_{03} - D_{03}B_{01}`.
                            let cof12 = C64x8::cof_neg(d00, b13, d01, b03, d03, b01);
                            repl1 = C64x8::madd(repl1, load_f(1, 2), cof12);

                            // `C_{13} = D_{00}B_{12} - D_{01}B_{02} + D_{02}B_{01}`.
                            let cof13 = C64x8::cof_pos(d00, b12, d01, b02, d02, b01);
                            repl1 = C64x8::madd(repl1, load_f(1, 3), cof13);
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

                            // `C_{20} = D_{01}Q_{23} - D_{02}Q_{13} + D_{03}Q_{12}`.
                            let cof20 = C64x8::cof_pos(d01, q23, d02, q13, d03, q12);
                            repl2 = C64x8::madd(repl2, load_f(2, 0), cof20);

                            // `C_{21} = -D_{00}Q_{23} + D_{02}Q_{03} - D_{03}Q_{02}`.
                            let cof21 = C64x8::cof_neg(d00, q23, d02, q03, d03, q02);
                            repl2 = C64x8::madd(repl2, load_f(2, 1), cof21);

                            // `C_{22} = D_{00}Q_{13} - D_{01}Q_{03} + D_{03}Q_{01}`.
                            let cof22 = C64x8::cof_pos(d00, q13, d01, q03, d03, q01);
                            repl2 = C64x8::madd(repl2, load_f(2, 2), cof22);

                            // `C_{23} = -D_{00}Q_{12} + D_{01}Q_{02} - D_{02}Q_{01}`.
                            let cof23 = C64x8::cof_neg(d00, q12, d01, q02, d02, q01);
                            repl2 = C64x8::madd(repl2, load_f(2, 3), cof23);
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

                            // `C_{30} = -D_{01}R_{23} + D_{02}R_{13} - D_{03}R_{12}`.
                            let cof30 = C64x8::cof_neg(d01, r23, d02, r13, d03, r12);
                            repl3 = C64x8::madd(repl3, load_f(3, 0), cof30);

                            // `C_{31} = D_{00}R_{23} - D_{02}R_{03} + D_{03}R_{02}`.
                            let cof31 = C64x8::cof_pos(d00, r23, d02, r03, d03, r02);
                            repl3 = C64x8::madd(repl3, load_f(3, 1), cof31);

                            // `C_{32} = -D_{00}R_{13} + D_{01}R_{03} - D_{03}R_{01}`.
                            let cof32 = C64x8::cof_neg(d00, r13, d01, r03, d03, r01);
                            repl3 = C64x8::madd(repl3, load_f(3, 2), cof32);

                            // `C_{33} = D_{00}R_{12} - D_{01}R_{02} + D_{02}R_{01}`.
                            let cof33 = C64x8::cof_pos(d00, r12, d01, r02, d02, r01);
                            repl3 = C64x8::madd(repl3, load_f(3, 3), cof33);
                        }

                        // Evaluate `\sum_{\eta z}\operatorname{cof}[\mathbf D]_{\eta z}\mathcal F_{\eta z}` with
                        // `((repl0 + repl1) + (repl2 + repl3))`.
                        let repl23 = C64x8::add(repl2, repl3);
                        (det, C64x8::add(repl01, repl23))
                    }
                    _ => unreachable!(),
                };

                let pref_v = C64x8::splat(pref.re, pref.im);
                let f0_v = C64x8::splat(f0.re, f0.im);
                // `S = p\,{}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}`.
                let overlap_v = C64x8::mul(pref_v, det);
                // `F = p\,{}^{xw}\tilde S(F_0\det\mathbf D_{\mathrm{ov}} - R)`.
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
                    if det_re.get_unchecked(lane).is_finite()
                        && det_im.get_unchecked(lane).is_finite()
                    {
                        *overlap.get_unchecked_mut(lane) = Complex64::new(
                            *overlap_re.get_unchecked(lane),
                            *overlap_im.get_unchecked(lane),
                        );
                        *fock.get_unchecked_mut(lane) = Complex64::new(
                            *fock_re.get_unchecked(lane),
                            *fock_im.get_unchecked(lane),
                        );
                    } else {
                        *overlap.get_unchecked_mut(lane) = Complex64::new(0.0, 0.0);
                        *fock.get_unchecked_mut(lane) = Complex64::new(0.0, 0.0);
                    }
                }
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
