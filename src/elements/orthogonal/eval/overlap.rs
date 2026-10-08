// elements/orthogonal/eval/overlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

// External crate imports.
#[cfg(target_arch = "x86_64")]
use num_complex::Complex64;

// Crate-root imports.
use crate::NOCIScalar;
use crate::ReducedTwoSpinState;
#[cfg(target_arch = "x86_64")]
use crate::maths::{C64x4, C64x8, F64x4, F64x8, Simd};

// Parent/sibling imports.
use super::dispatch::dispatch_orthogonal_ranks;

/// Evaluate `\langle D|\Phi_x\rangle` from a reduced orthogonal excitation.
/// Orthonormality of the parent orbitals restricts the fixed rank `(R_\alpha,R_\beta)` to the
/// diagonal sector `(0,0)`; every other supported rank returns zero.
/// # Arguments:
/// - `state`: Reduced phase and fixed-rank excitation labels connecting source to child.
/// # Returns
/// - `T`: Orthogonal Slater-Condon overlap matrix element.
#[inline(always)]
pub(crate) fn xw_overlap_orthogonal<T: NOCIScalar>(state: &ReducedTwoSpinState) -> T {
    let ranks = (
        usize::from(state.excitation_cache.alpha.rank),
        usize::from(state.excitation_cache.beta.rank),
    );

    // Dispatch by fixed spin excitation rank. This keeps the Slater-Condon formula compile-time
    // specialised while returning zero for every excited sector.
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| xw_overlap_orthogonal_const::<T, RA, RB>(state),
        T::from_real(0.0),
    )
}

/// Evaluate a fixed-rank orthogonal Slater-Condon overlap matrix element.
/// Only the diagonal sector survives, giving `S_{xx} = p` for the reduced phase `p`.
/// # Arguments:
/// - `state`: Reduced phase and fixed-rank orbital labels.
/// # Returns
/// - `T`: Fixed-rank overlap matrix element.
#[inline(always)]
fn xw_overlap_orthogonal_const<T: NOCIScalar, const RA: usize, const RB: usize>(
    state: &ReducedTwoSpinState
) -> T {
    if RA == 0 && RB == 0 {
        // Diagonal element: identical occupations overlap with the reduced phase.
        // `S_{xx} = p`.
        return T::from_real(state.phase);
    }

    T::from_real(0.0)
}

/// Evaluate a batch of orthogonal overlap matrix elements.
/// Diagonal requests are packetised for SIMD phase kernels; excited requests use the fixed-rank
/// scalar implementation.
/// # Arguments:
/// - `states`: Reduced phase/cache payloads for source-relative excitations.
/// - `out`: Overlap results in request order.
/// # Returns
/// - `()`: Writes every matrix element into `out`.
pub(crate) fn xw_overlap_orthogonal_batched<T: NOCIScalar>(
    states: &[ReducedTwoSpinState],
    out: &mut [T],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // Try packed kernels only after checking scalar type and CPU features. These kernels write
        // results in the same order as `states`.
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let out_f64 = std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len());
            if is_x86_feature_detected!("avx512f") {
                xw_overlap_orthogonal_simd(states, out_f64, xw_overlap_orthogonal_f64x8);
                return;
            }
            if is_x86_feature_detected!("avx2") {
                xw_overlap_orthogonal_simd(states, out_f64, xw_overlap_orthogonal_f64x4);
                return;
            }
        }

        if TypeId::of::<T>() == TypeId::of::<Complex64>() {
            let out_c64 =
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<Complex64>(), out.len());
            if is_x86_feature_detected!("avx512f") {
                xw_overlap_orthogonal_simd(states, out_c64, xw_overlap_orthogonal_c64x8);
                return;
            }
            if is_x86_feature_detected!("avx2") {
                xw_overlap_orthogonal_simd(states, out_c64, xw_overlap_orthogonal_c64x4);
                return;
            }
        }
    }

    // Scalar fallback handles unsupported CPUs and any request not accepted by packed kernels.
    for (state, value) in states.iter().zip(out) {
        *value = xw_overlap_orthogonal(state);
    }
}

/// Evaluate one orthogonal overlap batch through fixed-width diagonal kernels.
/// Requests in the `(0,0)` bin use packed phase kernels; all other ranks use the scalar
/// fixed-rank evaluator without changing request order.
/// # Arguments:
/// - `states`: Reduced excitation phases and caches.
/// - `out`: Overlap results in request order.
/// - `kernel`: CPU-specific fixed-width diagonal dispatcher.
/// # Returns
/// - `()`: Writes every matrix element into `out`.
/// # Safety
/// - The caller must provide a kernel supported by the current CPU.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::type_complexity)]
unsafe fn xw_overlap_orthogonal_simd<T: NOCIScalar, const N: usize>(
    states: &[ReducedTwoSpinState],
    out: &mut [T],
    kernel: unsafe fn((usize, usize), &[ReducedTwoSpinState; N], &mut [T; N]),
) {
    if states.is_empty() {
        return;
    }

    // Group requests by fixed excitation rank so one packed kernel evaluates one Slater-Condon
    // formula across all lanes. Incomplete groups are padded internally and truncated on writeback.
    let mut bins = [[states[0]; N]; 1];
    let mut outputs = [[0usize; N]; 1];
    let mut counts = [0usize; 1];

    for (output, state) in states.iter().enumerate() {
        let ranks = (
            usize::from(state.excitation_cache.alpha.rank),
            usize::from(state.excitation_cache.beta.rank),
        );
        let bin = match ranks {
            (0, 0) => 0,
            _ => {
                // Excited sectors vanish, so evaluate them scalar.
                out[output] = xw_overlap_orthogonal(state);
                continue;
            }
        };
        let count = counts[bin];
        bins[bin][count] = *state;
        outputs[bin][count] = output;
        counts[bin] += 1;

        if counts[bin] == N {
            // Evaluate and scatter a complete same-rank packet immediately.
            let mut values = [T::from_real(0.0); N];
            unsafe { kernel(ranks, &bins[bin], &mut values) };
            for lane in 0..N {
                out[outputs[bin][lane]] = values[lane];
            }
            counts[bin] = 0;
        }
    }

    for bin in 0..counts.len() {
        let count = counts[bin];
        if count == 0 {
            continue;
        }

        // Duplicate one valid lane to fill the final packet; only genuine lanes are written back.
        let fill = bins[bin][0];
        for lane in count..N {
            bins[bin][lane] = fill;
        }
        let ranks = (0, 0);
        let mut values = [T::from_real(0.0); N];
        unsafe { kernel(ranks, &bins[bin], &mut values) };
        for lane in 0..count {
            out[outputs[bin][lane]] = values[lane];
        }
    }
}

/// Phase one fixed-rank packet of orthogonal diagonal overlaps.
/// The diagonal sector evaluates `S_{xx} = p` as one lane-wise real phase multiplication of
/// packed one.
/// # Arguments:
/// - `states`: Reduced excitation labels and phases in lane order.
/// - `out`: Overlap outputs in lane order.
/// # Returns
/// - `()`: Writes one packet of fixed-rank diagonal elements.
/// # Safety
/// - `V` must be supported by the current CPU.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xw_overlap_orthogonal_simd_const<
    T: NOCIScalar,
    V: Simd<N, Scalar = T>,
    const N: usize,
    const RA: usize,
    const RB: usize,
>(
    states: &[ReducedTwoSpinState; N],
    out: &mut [T; N],
) {
    let mut phases = [1.0f64; N];

    if RA == 0 && RB == 0 {
        // Diagonal: scale packed one by `p` lane-wise.
        for lane in 0..N {
            phases[lane] = states[lane].phase;
        }
        V::multiply_real_lanes(V::one(), &phases).store(out);
    }
}

/// Dispatch four real orthogonal diagonal overlaps to AVX2 fixed-rank kernels.
/// # Arguments:
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Four reduced excitation states.
/// - `out`: Four overlap outputs.
/// # Returns
/// - `()`: Writes four real matrix elements.
/// # Safety
/// - The current CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_overlap_orthogonal_f64x4(
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 4],
    out: &mut [f64; 4],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe { xw_overlap_orthogonal_f64x4_const::<RA, RB>(states, out) },
        (),
    );
}

/// Evaluate four real fixed-rank orthogonal diagonal overlaps with AVX2.
/// # Arguments:
/// - `states`: Four reduced excitation states.
/// - `out`: Four overlap outputs.
/// # Returns
/// - `()`: Writes four real matrix elements.
/// # Safety
/// - The current CPU must support AVX2 and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_overlap_orthogonal_f64x4_const<const RA: usize, const RB: usize>(
    states: &[ReducedTwoSpinState; 4],
    out: &mut [f64; 4],
) {
    unsafe {
        xw_overlap_orthogonal_simd_const::<f64, F64x4, 4, RA, RB>(states, out);
    }
}

/// Dispatch eight real orthogonal diagonal overlaps to AVX-512 fixed-rank kernels.
/// # Arguments:
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Eight reduced excitation states.
/// - `out`: Eight overlap outputs.
/// # Returns
/// - `()`: Writes eight real matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_orthogonal_f64x8(
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 8],
    out: &mut [f64; 8],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe { xw_overlap_orthogonal_f64x8_const::<RA, RB>(states, out) },
        (),
    );
}

/// Evaluate eight real fixed-rank orthogonal diagonal overlaps with AVX-512F.
/// # Arguments:
/// - `states`: Eight reduced excitation states.
/// - `out`: Eight overlap outputs.
/// # Returns
/// - `()`: Writes eight real matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_orthogonal_f64x8_const<const RA: usize, const RB: usize>(
    states: &[ReducedTwoSpinState; 8],
    out: &mut [f64; 8],
) {
    unsafe {
        xw_overlap_orthogonal_simd_const::<f64, F64x8, 8, RA, RB>(states, out);
    }
}

/// Dispatch four complex orthogonal diagonal overlaps to AVX2 fixed-rank kernels.
/// # Arguments:
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Four reduced excitation states.
/// - `out`: Four overlap outputs.
/// # Returns
/// - `()`: Writes four complex matrix elements.
/// # Safety
/// - The current CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_overlap_orthogonal_c64x4(
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 4],
    out: &mut [Complex64; 4],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe { xw_overlap_orthogonal_c64x4_const::<RA, RB>(states, out) },
        (),
    );
}

/// Evaluate four complex fixed-rank orthogonal diagonal overlaps with AVX2.
/// # Arguments:
/// - `states`: Four reduced excitation states.
/// - `out`: Four overlap outputs.
/// # Returns
/// - `()`: Writes four complex matrix elements.
/// # Safety
/// - The current CPU must support AVX2 and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_overlap_orthogonal_c64x4_const<const RA: usize, const RB: usize>(
    states: &[ReducedTwoSpinState; 4],
    out: &mut [Complex64; 4],
) {
    unsafe {
        xw_overlap_orthogonal_simd_const::<Complex64, C64x4, 4, RA, RB>(states, out);
    }
}

/// Dispatch eight complex orthogonal diagonal overlaps to AVX-512 fixed-rank kernels.
/// # Arguments:
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Eight reduced excitation states.
/// - `out`: Eight overlap outputs.
/// # Returns
/// - `()`: Writes eight complex matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_orthogonal_c64x8(
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 8],
    out: &mut [Complex64; 8],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe { xw_overlap_orthogonal_c64x8_const::<RA, RB>(states, out) },
        (),
    );
}

/// Evaluate eight complex fixed-rank orthogonal diagonal overlaps with AVX-512F.
/// # Arguments:
/// - `states`: Eight reduced excitation states.
/// - `out`: Eight overlap outputs.
/// # Returns
/// - `()`: Writes eight complex matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_overlap_orthogonal_c64x8_const<const RA: usize, const RB: usize>(
    states: &[ReducedTwoSpinState; 8],
    out: &mut [Complex64; 8],
) {
    unsafe {
        xw_overlap_orthogonal_simd_const::<Complex64, C64x8, 8, RA, RB>(states, out);
    }
}
