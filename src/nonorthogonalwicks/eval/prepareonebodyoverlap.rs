// nonorthogonalwicks/eval/prepareonebodyoverlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

// External crate imports.
use num_complex::Complex64;

// Crate-root imports.
use crate::config::MAXEXCIT;
#[cfg(target_arch = "x86_64")]
use crate::maths::{C64x4, C64x8, F64x4, F64x8, Simd, adjugate_transpose_simd_const};
use crate::maths::{
    adjugate_transpose_const, adjugate_transpose_dynamic, build_d_const, build_d_dynamic,
    det_dynamic,
};
use crate::noci::NOCIScalar;
use crate::time_call;
use crate::{DetState, ExcitationSpin, ExcitationSpinCache, ReducedOneSpinDetState};

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::dispatch::{dispatch_onebody_ranks, dispatch_onebody_scalar_ranks, dispatch_pair_ranks};
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, mix_dets_same,
};
use super::prepare::construct_determinant_indices;

#[cfg(target_arch = "x86_64")]
type PreparedSimdKernel<T, const N: usize> = for<'a> unsafe fn(
    &SameSpinView<'a, T>,
    &ExcitationSpinCache,
    &[ExcitationSpinCache; N],
    &mut [T; N],
    &mut [T; N],
);

/// Rank dispatcher for one concrete same-spin packed type.
#[cfg(target_arch = "x86_64")]
type PreparedSimdSelector<T, const N: usize> =
    fn((usize, usize), bool) -> Option<PreparedSimdKernel<T, N>>;

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
            // SAFETY: The explicit `TypeId` check proves every generic value has its `f64`
            // instantiation for the duration of the SIMD helper call.
            let w_f64 = &*std::ptr::from_ref(w).cast::<SameSpinView<'_, f64>>();
            let basis_f64 =
                std::slice::from_raw_parts(basis.as_ptr().cast::<DetState<f64>>(), basis.len());
            let scratch_f64 = &mut *std::ptr::from_mut(scratch).cast::<WickScratch<f64>>();
            let overlap_f64 =
                std::slice::from_raw_parts_mut(overlap.as_mut_ptr().cast::<f64>(), overlap.len());
            let fock_f64 =
                std::slice::from_raw_parts_mut(fock.as_mut_ptr().cast::<f64>(), fock.len());

            if try_xw_f_overlap_prepared_f64_simd(
                w_f64,
                basis_f64,
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
                scratch_f64,
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
            // SAFETY: The explicit `TypeId` check proves every generic value has its `Complex64`
            // instantiation for the duration of the SIMD helper call.
            let w_c64 = &*std::ptr::from_ref(w).cast::<SameSpinView<'_, Complex64>>();
            let basis_c64 = std::slice::from_raw_parts(
                basis.as_ptr().cast::<DetState<Complex64>>(),
                basis.len(),
            );
            let scratch_c64 = &mut *std::ptr::from_mut(scratch).cast::<WickScratch<Complex64>>();
            let overlap_c64 = std::slice::from_raw_parts_mut(
                overlap.as_mut_ptr().cast::<Complex64>(),
                overlap.len(),
            );
            let fock_c64 =
                std::slice::from_raw_parts_mut(fock.as_mut_ptr().cast::<Complex64>(), fock.len());

            if try_xw_f_overlap_prepared_c64_simd(
                w_c64,
                basis_c64,
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
                scratch_c64,
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
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates with `m = 0`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `input`: Target/source representatives and evaluation metadata.
/// - `scratch`: Reusable Wick workspace.
/// - `tol`: Numerical tolerance used by scalar fallback evaluation.
/// - `out`: Real overlap and generalised-Fock output rows.
/// # Returns
/// - `bool`: Whether SIMD support existed and the complete row was written.
/// # Safety
/// - `w` must contain no zero-overlap orbital pairs.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_f_overlap_prepared_f64_simd(
    w: &SameSpinView<'_, f64>,
    basis: &[DetState<f64>],
    input: SameSpinSimdInput<'_>,
    scratch: &mut WickScratch<f64>,
    tol: f64,
    out: (&mut [f64], &mut [f64]),
) -> bool {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            xw_f_overlap_prepared_f64x8_row(w, basis, input, scratch, tol, out);
        }
        return true;
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            xw_f_overlap_prepared_f64x4_row(w, basis, input, scratch, tol, out);
        }
        return true;
    }
    false
}

/// Try to evaluate one complex same-spin factor row with fixed-rank SIMD kernels.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates with `m = 0`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `input`: Target/source representatives and evaluation metadata.
/// - `scratch`: Reusable Wick workspace.
/// - `tol`: Numerical tolerance used by scalar fallback evaluation.
/// - `out`: Complex overlap and generalised-Fock output rows.
/// # Returns
/// - `bool`: Whether SIMD support existed and the complete row was written.
/// # Safety
/// - `w` must contain no zero-overlap orbital pairs.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_f_overlap_prepared_c64_simd(
    w: &SameSpinView<'_, Complex64>,
    basis: &[DetState<Complex64>],
    input: SameSpinSimdInput<'_>,
    scratch: &mut WickScratch<Complex64>,
    tol: f64,
    out: (&mut [Complex64], &mut [Complex64]),
) -> bool {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            xw_f_overlap_prepared_c64x8_row(w, basis, input, scratch, tol, out);
        }
        return true;
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            xw_f_overlap_prepared_c64x4_row(w, basis, input, scratch, tol, out);
        }
        return true;
    }
    false
}

/// Evaluate one same-spin factor row with packed fixed-rank kernels and scalar tails.
/// # Arguments:
/// - `w`: Same-spin Wick intermediates with `m = 0`.
/// - `basis`: Determinant basis used by scalar fallback evaluation.
/// - `input`: Target/source representatives and evaluation metadata.
/// - `scratch`: Reusable Wick workspace.
/// - `tol`: Numerical tolerance used by scalar fallback evaluation.
/// - `out`: Overlap and generalised-Fock output rows.
/// - `select`: Rank dispatcher for one concrete packed type.
/// # Returns
/// - `()`: Writes one complete factor row.
/// # Safety
/// - `select` must return kernels supported by the current CPU and using `N` lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xw_f_overlap_prepared_simd_row<T: NOCIScalar, const N: usize>(
    w: &SameSpinView<'_, T>,
    basis: &[DetState<T>],
    input: SameSpinSimdInput<'_>,
    scratch: &mut WickScratch<T>,
    tol: f64,
    out: (&mut [T], &mut [T]),
    select: PreparedSimdSelector<T, N>,
) {
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
    let target_rank = usize::from(target_cache.rank);

    for bounds in source_groups.windows(2) {
        let group_start = unsafe { *bounds.get_unchecked(0) };
        let group_end = unsafe { *bounds.get_unchecked(1) };
        let source_rank = usize::from(unsafe { source_caches.get_unchecked(group_start).rank });
        let ranks = if target_left {
            (target_rank, source_rank)
        } else {
            (source_rank, target_rank)
        };

        if target_rank <= MAXEXCIT && source_rank <= MAXEXCIT && target_rank + source_rank != 0 {
            let kernel = select(ranks, target_left);
            let mut packet_start = group_start;

            if let Some(kernel) = kernel {
                while group_end - packet_start >= N {
                    let packet = unsafe {
                        &*source_caches
                            .as_ptr()
                            .add(packet_start)
                            .cast::<[ExcitationSpinCache; N]>()
                    };
                    let mut s = [T::from_real(0.0); N];
                    let mut f = [T::from_real(0.0); N];
                    unsafe {
                        kernel(w, &target_cache, packet, &mut s, &mut f);
                    }

                    for lane in 0..N {
                        let ordered = packet_start + lane;
                        let col = unsafe { *source_order.get_unchecked(ordered) };
                        let phase = T::from_real(
                            target_rep.phase * unsafe { *source_phases.get_unchecked(ordered) },
                        );
                        overlap[col] = phase * s[lane];
                        fock[col] = phase * f[lane];
                    }
                    packet_start += N;
                }
            }

            for ordered in packet_start..group_end {
                let col = unsafe { *source_order.get_unchecked(ordered) };
                let (s, f) = xw_f_overlap_prepared_scalar_value(
                    w,
                    basis,
                    (target_rep, sources[col]),
                    (target_left, alpha),
                    scratch,
                    tol,
                );
                let phase = T::from_real(target_rep.phase * sources[col].phase);
                overlap[col] = phase * s;
                fock[col] = phase * f;
            }
        } else {
            for ordered in group_start..group_end {
                let col = unsafe { *source_order.get_unchecked(ordered) };
                let (s, f) = xw_f_overlap_prepared_scalar_value(
                    w,
                    basis,
                    (target_rep, sources[col]),
                    (target_left, alpha),
                    scratch,
                    tol,
                );
                let phase = T::from_real(target_rep.phase * sources[col].phase);
                overlap[col] = phase * s;
                fock[col] = phase * f;
            }
        }
    }
}

/// Evaluate one real factor row with four-lane AVX2/FMA packets.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `basis`: Determinant basis used by scalar tails.
/// - `input`: SIMD evaluation metadata.
/// - `scratch`: Reusable Wick workspace.
/// - `tol`: Scalar fallback tolerance.
/// - `out`: Real output rows.
/// # Returns
/// - `()`: Writes one complete row.
/// # Safety
/// - The current CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_prepared_f64x4_row(
    w: &SameSpinView<'_, f64>,
    basis: &[DetState<f64>],
    input: SameSpinSimdInput<'_>,
    scratch: &mut WickScratch<f64>,
    tol: f64,
    out: (&mut [f64], &mut [f64]),
) {
    unsafe {
        xw_f_overlap_prepared_simd_row::<f64, 4>(
            w,
            basis,
            input,
            scratch,
            tol,
            out,
            select_xw_f_overlap_f64x4,
        );
    }
}

/// Evaluate one real factor row with eight-lane AVX-512 packets.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `basis`: Determinant basis used by scalar tails.
/// - `input`: SIMD evaluation metadata.
/// - `scratch`: Reusable Wick workspace.
/// - `tol`: Scalar fallback tolerance.
/// - `out`: Real output rows.
/// # Returns
/// - `()`: Writes one complete row.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_prepared_f64x8_row(
    w: &SameSpinView<'_, f64>,
    basis: &[DetState<f64>],
    input: SameSpinSimdInput<'_>,
    scratch: &mut WickScratch<f64>,
    tol: f64,
    out: (&mut [f64], &mut [f64]),
) {
    unsafe {
        xw_f_overlap_prepared_simd_row::<f64, 8>(
            w,
            basis,
            input,
            scratch,
            tol,
            out,
            select_xw_f_overlap_f64x8,
        );
    }
}

/// Evaluate one complex factor row with four-lane AVX2/FMA packets.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `basis`: Determinant basis used by scalar tails.
/// - `input`: SIMD evaluation metadata.
/// - `scratch`: Reusable Wick workspace.
/// - `tol`: Scalar fallback tolerance.
/// - `out`: Complex output rows.
/// # Returns
/// - `()`: Writes one complete row.
/// # Safety
/// - The current CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_prepared_c64x4_row(
    w: &SameSpinView<'_, Complex64>,
    basis: &[DetState<Complex64>],
    input: SameSpinSimdInput<'_>,
    scratch: &mut WickScratch<Complex64>,
    tol: f64,
    out: (&mut [Complex64], &mut [Complex64]),
) {
    unsafe {
        xw_f_overlap_prepared_simd_row::<Complex64, 4>(
            w,
            basis,
            input,
            scratch,
            tol,
            out,
            select_xw_f_overlap_c64x4,
        );
    }
}

/// Evaluate one complex factor row with eight-lane AVX-512 packets.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `basis`: Determinant basis used by scalar tails.
/// - `input`: SIMD evaluation metadata.
/// - `scratch`: Reusable Wick workspace.
/// - `tol`: Scalar fallback tolerance.
/// - `out`: Complex output rows.
/// # Returns
/// - `()`: Writes one complete row.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_prepared_c64x8_row(
    w: &SameSpinView<'_, Complex64>,
    basis: &[DetState<Complex64>],
    input: SameSpinSimdInput<'_>,
    scratch: &mut WickScratch<Complex64>,
    tol: f64,
    out: (&mut [Complex64], &mut [Complex64]),
) {
    unsafe {
        xw_f_overlap_prepared_simd_row::<Complex64, 8>(
            w,
            basis,
            input,
            scratch,
            tol,
            out,
            select_xw_f_overlap_c64x8,
        );
    }
}

/// Evaluate the prepared overlap and generalised-Fock matrix element.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Reusable Wick workspace.
/// - `tol`: Numerical tolerance used by runtime cofactor evaluation.
/// # Returns
/// - `(T, T)`: Same-spin overlap and generalised-Fock matrix element.
#[inline(always)]
pub(crate) fn xw_f_overlap_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap, {
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
            dispatch_onebody_scalar_ranks!(
                (rx, rw),
                |RX, RW, L, D| xw_f_overlap_m0_prepared_const::<T, RX, RW, L, D>(
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
    const D: usize,
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

            let det0 = &scratch.det0.as_slice()[..D];
            let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

            // Evaluate `\det\mathbf D_{\mathrm{ov}}` and
            // `\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z} = (-1)^{\eta+z}`
            // `\det\mathbf D_{\mathrm{ov}}[\eta|z]`.
            // for the overlap part and the one-body column replacements.
            let det = adjugate_transpose_const::<T, L, D>(
                &mut scratch.adjt_det.as_mut_slice()[..D],
                det0,
            );
            if det.abs() > tol {
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
/// Select one four-lane real fixed-rank one-body kernel.
/// # Arguments:
/// - `ranks`: Bra and ket excitation ranks.
/// - `x_fixed`: Whether the shared cache belongs to the bra.
/// # Returns
/// - `Option<PreparedSimdKernel<f64, 4>>`: Selected kernel when ranks are generated.
#[cfg(target_arch = "x86_64")]
fn select_xw_f_overlap_f64x4(
    ranks: (usize, usize),
    x_fixed: bool,
) -> Option<PreparedSimdKernel<f64, 4>> {
    if x_fixed {
        dispatch_onebody_ranks!(
            ranks,
            |RX, RW, L, D| Some(xw_f_overlap_m0_prepared_f64x4_const::<RX, RW, L, D, true>),
            None,
        )
    } else {
        dispatch_onebody_ranks!(
            ranks,
            |RX, RW, L, D| Some(xw_f_overlap_m0_prepared_f64x4_const::<RX, RW, L, D, false>),
            None,
        )
    }
}

/// Select one eight-lane real fixed-rank one-body kernel.
/// # Arguments:
/// - `ranks`: Bra and ket excitation ranks.
/// - `x_fixed`: Whether the shared cache belongs to the bra.
/// # Returns
/// - `Option<PreparedSimdKernel<f64, 8>>`: Selected kernel when ranks are generated.
#[cfg(target_arch = "x86_64")]
fn select_xw_f_overlap_f64x8(
    ranks: (usize, usize),
    x_fixed: bool,
) -> Option<PreparedSimdKernel<f64, 8>> {
    if x_fixed {
        dispatch_onebody_ranks!(
            ranks,
            |RX, RW, L, D| Some(xw_f_overlap_m0_prepared_f64x8_const::<RX, RW, L, D, true>),
            None,
        )
    } else {
        dispatch_onebody_ranks!(
            ranks,
            |RX, RW, L, D| Some(xw_f_overlap_m0_prepared_f64x8_const::<RX, RW, L, D, false>),
            None,
        )
    }
}

/// Select one four-lane complex fixed-rank one-body kernel.
/// # Arguments:
/// - `ranks`: Bra and ket excitation ranks.
/// - `x_fixed`: Whether the shared cache belongs to the bra.
/// # Returns
/// - `Option<PreparedSimdKernel<Complex64, 4>>`: Selected kernel when ranks are generated.
#[cfg(target_arch = "x86_64")]
fn select_xw_f_overlap_c64x4(
    ranks: (usize, usize),
    x_fixed: bool,
) -> Option<PreparedSimdKernel<Complex64, 4>> {
    if x_fixed {
        dispatch_onebody_ranks!(
            ranks,
            |RX, RW, L, D| Some(xw_f_overlap_m0_prepared_c64x4_const::<RX, RW, L, D, true>),
            None,
        )
    } else {
        dispatch_onebody_ranks!(
            ranks,
            |RX, RW, L, D| Some(xw_f_overlap_m0_prepared_c64x4_const::<RX, RW, L, D, false>),
            None,
        )
    }
}

/// Select one eight-lane complex fixed-rank one-body kernel.
/// # Arguments:
/// - `ranks`: Bra and ket excitation ranks.
/// - `x_fixed`: Whether the shared cache belongs to the bra.
/// # Returns
/// - `Option<PreparedSimdKernel<Complex64, 8>>`: Selected kernel when ranks are generated.
#[cfg(target_arch = "x86_64")]
fn select_xw_f_overlap_c64x8(
    ranks: (usize, usize),
    x_fixed: bool,
) -> Option<PreparedSimdKernel<Complex64, 8>> {
    if x_fixed {
        dispatch_onebody_ranks!(
            ranks,
            |RX, RW, L, D| Some(xw_f_overlap_m0_prepared_c64x8_const::<RX, RW, L, D, true>),
            None,
        )
    } else {
        dispatch_onebody_ranks!(
            ranks,
            |RX, RW, L, D| Some(xw_f_overlap_m0_prepared_c64x8_const::<RX, RW, L, D, false>),
            None,
        )
    }
}

/// Evaluate packed fixed-rank overlap and generalised-Fock factors.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `fixed`: Excitation cache shared by all lanes.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Overlap outputs in lane order.
/// - `fock`: Generalised-Fock outputs in lane order.
/// # Returns
/// - `()`: Writes `LANES` overlap and generalised-Fock values.
/// # Safety
/// - Cached labels must match `RX` and `RW`; caller must establish `V` CPU support.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xw_f_overlap_m0_prepared_simd_const<
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
    fock: &mut [T; LANES],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            let n = w.n();
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;
            let x0 = w.x_slice(0);
            let y0 = w.y_slice(0);
            let fsl = w.ff_t_slice(0, 0);
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
            let load = |values: &[T], eta: usize, z: usize, transpose: bool| -> V {
                let invariant = if XFIX { z < RX } else { eta >= RX };
                let index = |lane: usize| {
                    if transpose {
                        col_index(z, lane) * n + row_index(eta, lane)
                    } else {
                        row_index(eta, lane) * n + col_index(z, lane)
                    }
                };

                if invariant {
                    V::splat(unsafe { *values.get_unchecked(index(0)) })
                } else {
                    let mut lanes = [T::from_real(0.0); LANES];
                    for (lane, value) in lanes.iter_mut().enumerate() {
                        *value = unsafe { *values.get_unchecked(index(lane)) };
                    }
                    V::load(&lanes)
                }
            };
            let zero = V::zero();
            let mut d = [zero; D];
            let mut cof = [zero; D];

            for eta in 0..L {
                for z in 0..L {
                    let matrix = if eta >= z { x0 } else { y0 };
                    d[eta * L + z] = load(matrix, eta, z, false);
                }
            }

            let determinant = adjugate_transpose_simd_const::<V, LANES, L, D>(&mut cof, &d);
            let mut replacement = zero;

            for eta in 0..L {
                for z in 0..L {
                    replacement = V::madd(replacement, cof[eta * L + z], load(fsl, eta, z, true));
                }
            }

            let fock_value = V::sub(V::mul(determinant, V::splat(w.f0f[0])), replacement);
            let pref = V::splat(w.phase * T::from_real(w.tilde_s_prod));
            let overlap_value = V::mul(pref, determinant);
            let fock_value = V::mul(pref, fock_value);
            V::store(overlap_value, overlap);
            V::store(fock_value, fock);
        }
    )
}

/// Evaluate four real fixed-rank one-body factors with AVX2/FMA.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Real overlap outputs.
/// - `fock`: Real generalised-Fock outputs.
/// # Returns
/// - `()`: Writes four factor pairs.
/// # Safety
/// - The current CPU must support AVX2 and FMA; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_prepared_f64x4_const<
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
    fock: &mut [f64; 4],
) {
    unsafe {
        xw_f_overlap_m0_prepared_simd_const::<f64, F64x4, 4, RX, RW, L, D, XFIX>(
            w, fixed, varying, overlap, fock,
        );
    }
}

/// Evaluate eight real fixed-rank one-body factors with AVX-512F.
/// # Arguments:
/// - `w`: Real same-spin Wick intermediates.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Real overlap outputs.
/// - `fock`: Real generalised-Fock outputs.
/// # Returns
/// - `()`: Writes eight factor pairs.
/// # Safety
/// - The current CPU must support AVX-512F; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_prepared_f64x8_const<
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
    fock: &mut [f64; 8],
) {
    unsafe {
        xw_f_overlap_m0_prepared_simd_const::<f64, F64x8, 8, RX, RW, L, D, XFIX>(
            w, fixed, varying, overlap, fock,
        );
    }
}

/// Evaluate four complex fixed-rank one-body factors with AVX2/FMA.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Complex overlap outputs.
/// - `fock`: Complex generalised-Fock outputs.
/// # Returns
/// - `()`: Writes four factor pairs.
/// # Safety
/// - The current CPU must support AVX2 and FMA; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_prepared_c64x4_const<
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
    fock: &mut [Complex64; 4],
) {
    unsafe {
        xw_f_overlap_m0_prepared_simd_const::<Complex64, C64x4, 4, RX, RW, L, D, XFIX>(
            w, fixed, varying, overlap, fock,
        );
    }
}

/// Evaluate eight complex fixed-rank one-body factors with AVX-512F.
/// # Arguments:
/// - `w`: Complex same-spin Wick intermediates.
/// - `fixed`: Shared excitation cache.
/// - `varying`: Lane-local excitation caches.
/// - `overlap`: Complex overlap outputs.
/// - `fock`: Complex generalised-Fock outputs.
/// # Returns
/// - `()`: Writes eight factor pairs.
/// # Safety
/// - The current CPU must support AVX-512F; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_prepared_c64x8_const<
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
    fock: &mut [Complex64; 8],
) {
    unsafe {
        xw_f_overlap_m0_prepared_simd_const::<Complex64, C64x8, 8, RX, RW, L, D, XFIX>(
            w, fixed, varying, overlap, fock,
        );
    }
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
            build_d_dynamic(
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

            if let Some(det_det) = adjugate_transpose_dynamic(
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
                let overlap = pref * det_dynamic(det0, l).unwrap_or(zero);
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
        build_d_dynamic(
            scratch.det0.as_mut_slice(),
            l,
            &x0,
            &y0,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

        let x1 = w.x(1);
        let y1 = w.y(1);
        build_d_dynamic(
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
                overlap_acc += det_dynamic(scratch.det_mix.as_slice(), l).unwrap_or(zero);
            }
        });

        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        (pref * overlap_acc, pref * fock_acc)
    })
}
