// elements/orthogonal/eval/fock.rs

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
use crate::elements::FockMOCache;
#[cfg(target_arch = "x86_64")]
use crate::maths::{C64x4, C64x8, F64x4, F64x8, Simd};

// Parent/sibling imports.
use super::dispatch::dispatch_orthogonal_ranks;

/// Evaluate `\langle D|\hat F|\Phi_x\rangle` from a reduced orthogonal excitation.
/// The fixed rank `(R_\alpha,R_\beta)` is restricted by the one-body Fock operator to `(0,0)`,
/// `(1,0)`, or `(0,1)`; every other supported rank returns zero.
/// # Arguments:
/// - `cache`: Parent-specific orthogonal MO Fock matrices.
/// - `source`: Source alpha and beta occupation bitstrings.
/// - `state`: Reduced phase and fixed-rank excitation labels connecting source to child.
/// # Returns
/// - `T`: Orthogonal Slater-Condon Fock matrix element.
#[inline(always)]
pub(crate) fn xw_fock_orthogonal<T: NOCIScalar>(
    cache: &FockMOCache<T>,
    source: (u128, u128),
    state: &ReducedTwoSpinState,
) -> T {
    let ranks = (
        usize::from(state.excitation_cache.alpha.rank),
        usize::from(state.excitation_cache.beta.rank),
    );

    // Dispatch by fixed spin excitation rank. This keeps Slater-Condon formulas compile-time
    // specialised while returning zero for ranks unsupported by a one-body operator.
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| xw_fock_orthogonal_const::<T, RA, RB>(cache, source, state),
        T::from_real(0.0),
    )
}

/// Evaluate a fixed-rank orthogonal Slater-Condon Fock matrix element.
/// The compile-time branch implements the diagonal and alpha/beta single sectors without
/// rediscovering excitation masks or fermionic phases.
/// For excitation phase `p`, singles use `pF_{ai}` and ranks above one vanish.
/// # Arguments:
/// - `cache`: Parent-specific alpha and beta MO Fock matrices.
/// - `source`: Source alpha and beta occupation bitstrings.
/// - `state`: Reduced phase and fixed-rank orbital labels.
/// # Returns
/// - `T`: Fixed-rank Fock matrix element.
#[inline(always)]
fn xw_fock_orthogonal_const<T: NOCIScalar, const RA: usize, const RB: usize>(
    cache: &FockMOCache<T>,
    source: (u128, u128),
    state: &ReducedTwoSpinState,
) -> T {
    let alpha = &state.excitation_cache.alpha;
    let beta = &state.excitation_cache.beta;
    let phase = T::from_real(state.phase);

    if RA == 0 && RB == 0 {
        // Diagonal element: accumulate occupied alpha and beta orbital Fock energies.
        // `F_xx = \sum_{i\in O_\alpha}F^\alpha_{ii} + \sum_{i\in O_\beta}F^\beta_{ii}`.
        let mut f = T::from_real(0.0);
        let mut bits = source.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f += cache.fa[(i, i)];
        }

        let mut bits = source.1;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f += cache.fb[(i, i)];
        }

        return f;
    }

    if RA == 1 && RB == 0 {
        // Alpha single: only the signed alpha Fock coupling remains.
        let i = usize::from(alpha.holes[0]);
        let a = usize::from(alpha.particles[0]);
        // `F_{a\leftarrow i,x} = pF^\alpha_{ai}`.
        return phase * cache.fa[(a, i)];
    }

    if RA == 0 && RB == 1 {
        // Beta single is the spin-swapped alpha-single expression.
        let i = usize::from(beta.holes[0]);
        let a = usize::from(beta.particles[0]);
        // `F_{a\leftarrow i,x} = pF^\beta_{ai}`.
        return phase * cache.fb[(a, i)];
    }

    T::from_real(0.0)
}

/// Evaluate a parent-local batch of orthogonal Fock matrix elements.
/// Single-excitation requests are packetised by `(R_\alpha,R_\beta)` for SIMD gather kernels;
/// diagonal and vanishing requests use the fixed-rank scalar implementation.
/// # Arguments:
/// - `cache`: Parent-specific orthogonal MO Fock matrices.
/// - `occupations`: `\alpha` and `\beta` occupations aligned with `states` and `out`.
/// - `states`: Reduced phase/cache payloads for source-relative excitations.
/// - `out`: Fock results in request order.
/// # Returns
/// - `()`: Writes every parent-local matrix element into `out`.
pub(crate) fn xw_fock_orthogonal_batched<T: NOCIScalar>(
    cache: &FockMOCache<T>,
    occupations: &[(u128, u128)],
    states: &[ReducedTwoSpinState],
    out: &mut [T],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // Try packed kernels only after checking scalar type and CPU features. These kernels write
        // results in the same order as `occupations` and `states`.
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let cache_f64 = &*std::ptr::from_ref(cache).cast::<FockMOCache<f64>>();
            let out_f64 = std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len());
            if is_x86_feature_detected!("avx512f") {
                xw_fock_orthogonal_simd(
                    cache_f64,
                    occupations,
                    states,
                    out_f64,
                    xw_fock_orthogonal_f64x8,
                );
                return;
            }
            if is_x86_feature_detected!("avx2") {
                xw_fock_orthogonal_simd(
                    cache_f64,
                    occupations,
                    states,
                    out_f64,
                    xw_fock_orthogonal_f64x4,
                );
                return;
            }
        }

        if TypeId::of::<T>() == TypeId::of::<Complex64>() {
            let cache_c64 = &*std::ptr::from_ref(cache).cast::<FockMOCache<Complex64>>();
            let out_c64 =
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<Complex64>(), out.len());
            if is_x86_feature_detected!("avx512f") {
                xw_fock_orthogonal_simd(
                    cache_c64,
                    occupations,
                    states,
                    out_c64,
                    xw_fock_orthogonal_c64x8,
                );
                return;
            }
            if is_x86_feature_detected!("avx2") {
                xw_fock_orthogonal_simd(
                    cache_c64,
                    occupations,
                    states,
                    out_c64,
                    xw_fock_orthogonal_c64x4,
                );
                return;
            }
        }
    }

    // Scalar fallback handles unsupported CPUs and any request not accepted by packed kernels.
    for ((&occupation, state), value) in occupations.iter().zip(states).zip(out) {
        *value = xw_fock_orthogonal(cache, occupation, state);
    }
}

/// Evaluate one orthogonal Fock batch through fixed-width single-excitation kernels.
/// Requests in `(1,0)` and `(0,1)` bins use packed Fock gathers; all other ranks use the scalar
/// fixed-rank evaluator without changing request order.
/// # Arguments:
/// - `cache`: Parent-specific orthogonal MO Fock matrices.
/// - `occupations`: `\alpha` and `\beta` occupations aligned with `states` and `out`.
/// - `states`: Reduced excitation phases and caches.
/// - `out`: Fock results in request order.
/// - `kernel`: CPU-specific fixed-width single-excitation dispatcher.
/// # Returns
/// - `()`: Writes every matrix element into `out`.
/// # Safety
/// - The caller must provide a kernel supported by the current CPU.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::type_complexity)]
unsafe fn xw_fock_orthogonal_simd<T: NOCIScalar, const N: usize>(
    cache: &FockMOCache<T>,
    occupations: &[(u128, u128)],
    states: &[ReducedTwoSpinState],
    out: &mut [T],
    kernel: unsafe fn(&FockMOCache<T>, (usize, usize), &[ReducedTwoSpinState; N], &mut [T; N]),
) {
    if states.is_empty() {
        return;
    }

    // Group requests by fixed excitation rank so one packed kernel evaluates one Slater-Condon
    // formula across all lanes. Incomplete groups are padded internally and truncated on writeback.
    let mut bins = [[states[0]; N]; 2];
    let mut outputs = [[0usize; N]; 2];
    let mut counts = [0usize; 2];

    for (output, (&occupation, state)) in occupations.iter().zip(states).enumerate() {
        let ranks = (
            usize::from(state.excitation_cache.alpha.rank),
            usize::from(state.excitation_cache.beta.rank),
        );
        let bin = match ranks {
            (1, 0) => 0,
            (0, 1) => 1,
            _ => {
                // The diagonal sector needs occupied-orbital sums, so evaluate it scalar.
                out[output] = xw_fock_orthogonal(cache, occupation, state);
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
            unsafe { kernel(cache, ranks, &bins[bin], &mut values) };
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
        let ranks = match bin {
            0 => (1, 0),
            _ => (0, 1),
        };
        let mut values = [T::from_real(0.0); N];
        unsafe { kernel(cache, ranks, &bins[bin], &mut values) };
        for lane in 0..count {
            out[outputs[bin][lane]] = values[lane];
        }
    }
}

/// Gather and phase one fixed-rank packet of orthogonal single-excitation Fock elements.
/// The two supported sectors evaluate `pF_{ai}` with one indexed gather and one lane-wise real
/// phase multiplication.
/// # Arguments:
/// - `cache`: Parent-specific orthogonal MO Fock matrices.
/// - `states`: Reduced excitation labels and phases in lane order.
/// - `out`: Fock outputs in lane order.
/// # Returns
/// - `()`: Writes one packet of fixed-rank single-excitation elements.
/// # Safety
/// - `V` must be supported by the current CPU and every cached orbital label must index `cache`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xw_fock_orthogonal_simd_const<
    T: NOCIScalar,
    V: Simd<N, Scalar = T>,
    const N: usize,
    const RA: usize,
    const RB: usize,
>(
    cache: &FockMOCache<T>,
    states: &[ReducedTwoSpinState; N],
    out: &mut [T; N],
) {
    let mut indices = [0usize; N];
    let mut phases = [1.0f64; N];

    if RA == 1 && RB == 0 {
        // Alpha single: gather `pF_{ai}` from the alpha Fock matrix.
        let shape = cache.fa.shape();
        for lane in 0..N {
            let ex = &states[lane].excitation_cache.alpha;
            let i = usize::from(ex.holes[0]);
            let a = usize::from(ex.particles[0]);
            indices[lane] = a * shape[1] + i;
            phases[lane] = states[lane].phase;
        }
        let fock = cache
            .fa
            .as_slice()
            .expect("orthogonal alpha Fock matrix must be contiguous");
        let values = unsafe { V::gather(fock, &indices) };
        V::multiply_real_lanes(values, &phases).store(out);
        return;
    }

    if RA == 0 && RB == 1 {
        // Beta single: gather the spin-swapped `pF_{ai}` contribution.
        let shape = cache.fb.shape();
        for lane in 0..N {
            let ex = &states[lane].excitation_cache.beta;
            let i = usize::from(ex.holes[0]);
            let a = usize::from(ex.particles[0]);
            indices[lane] = a * shape[1] + i;
            phases[lane] = states[lane].phase;
        }
        let fock = cache
            .fb
            .as_slice()
            .expect("orthogonal beta Fock matrix must be contiguous");
        let values = unsafe { V::gather(fock, &indices) };
        V::multiply_real_lanes(values, &phases).store(out);
    }
}

/// Dispatch four real orthogonal single-excitation values to AVX2 fixed-rank kernels.
/// # Arguments:
/// - `cache`: Parent-specific real MO Fock matrices.
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Four reduced excitation states.
/// - `out`: Four Fock outputs.
/// # Returns
/// - `()`: Writes four real matrix elements.
/// # Safety
/// - The current CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_fock_orthogonal_f64x4(
    cache: &FockMOCache<f64>,
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 4],
    out: &mut [f64; 4],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe { xw_fock_orthogonal_f64x4_const::<RA, RB>(cache, states, out) },
        (),
    );
}

/// Evaluate four real fixed-rank orthogonal single-excitation values with AVX2.
/// # Arguments:
/// - `cache`: Parent-specific real MO Fock matrices.
/// - `states`: Four reduced excitation states.
/// - `out`: Four Fock outputs.
/// # Returns
/// - `()`: Writes four real matrix elements.
/// # Safety
/// - The current CPU must support AVX2 and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_fock_orthogonal_f64x4_const<const RA: usize, const RB: usize>(
    cache: &FockMOCache<f64>,
    states: &[ReducedTwoSpinState; 4],
    out: &mut [f64; 4],
) {
    unsafe {
        xw_fock_orthogonal_simd_const::<f64, F64x4, 4, RA, RB>(cache, states, out);
    }
}

/// Dispatch eight real orthogonal single-excitation values to AVX-512 fixed-rank kernels.
/// # Arguments:
/// - `cache`: Parent-specific real MO Fock matrices.
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Eight reduced excitation states.
/// - `out`: Eight Fock outputs.
/// # Returns
/// - `()`: Writes eight real matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_fock_orthogonal_f64x8(
    cache: &FockMOCache<f64>,
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 8],
    out: &mut [f64; 8],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe { xw_fock_orthogonal_f64x8_const::<RA, RB>(cache, states, out) },
        (),
    );
}

/// Evaluate eight real fixed-rank orthogonal single-excitation values with AVX-512F.
/// # Arguments:
/// - `cache`: Parent-specific real MO Fock matrices.
/// - `states`: Eight reduced excitation states.
/// - `out`: Eight Fock outputs.
/// # Returns
/// - `()`: Writes eight real matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_fock_orthogonal_f64x8_const<const RA: usize, const RB: usize>(
    cache: &FockMOCache<f64>,
    states: &[ReducedTwoSpinState; 8],
    out: &mut [f64; 8],
) {
    unsafe {
        xw_fock_orthogonal_simd_const::<f64, F64x8, 8, RA, RB>(cache, states, out);
    }
}

/// Dispatch four complex orthogonal single-excitation values to AVX2 fixed-rank kernels.
/// # Arguments:
/// - `cache`: Parent-specific complex MO Fock matrices.
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Four reduced excitation states.
/// - `out`: Four Fock outputs.
/// # Returns
/// - `()`: Writes four complex matrix elements.
/// # Safety
/// - The current CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_fock_orthogonal_c64x4(
    cache: &FockMOCache<Complex64>,
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 4],
    out: &mut [Complex64; 4],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe { xw_fock_orthogonal_c64x4_const::<RA, RB>(cache, states, out) },
        (),
    );
}

/// Evaluate four complex fixed-rank orthogonal single-excitation values with AVX2.
/// # Arguments:
/// - `cache`: Parent-specific complex MO Fock matrices.
/// - `states`: Four reduced excitation states.
/// - `out`: Four Fock outputs.
/// # Returns
/// - `()`: Writes four complex matrix elements.
/// # Safety
/// - The current CPU must support AVX2 and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_fock_orthogonal_c64x4_const<const RA: usize, const RB: usize>(
    cache: &FockMOCache<Complex64>,
    states: &[ReducedTwoSpinState; 4],
    out: &mut [Complex64; 4],
) {
    unsafe {
        xw_fock_orthogonal_simd_const::<Complex64, C64x4, 4, RA, RB>(cache, states, out);
    }
}

/// Dispatch eight complex orthogonal single-excitation values to AVX-512 fixed-rank kernels.
/// # Arguments:
/// - `cache`: Parent-specific complex MO Fock matrices.
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Eight reduced excitation states.
/// - `out`: Eight Fock outputs.
/// # Returns
/// - `()`: Writes eight complex matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_fock_orthogonal_c64x8(
    cache: &FockMOCache<Complex64>,
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 8],
    out: &mut [Complex64; 8],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe { xw_fock_orthogonal_c64x8_const::<RA, RB>(cache, states, out) },
        (),
    );
}

/// Evaluate eight complex fixed-rank orthogonal single-excitation values with AVX-512F.
/// # Arguments:
/// - `cache`: Parent-specific complex MO Fock matrices.
/// - `states`: Eight reduced excitation states.
/// - `out`: Eight Fock outputs.
/// # Returns
/// - `()`: Writes eight complex matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_fock_orthogonal_c64x8_const<const RA: usize, const RB: usize>(
    cache: &FockMOCache<Complex64>,
    states: &[ReducedTwoSpinState; 8],
    out: &mut [Complex64; 8],
) {
    unsafe {
        xw_fock_orthogonal_simd_const::<Complex64, C64x8, 8, RA, RB>(cache, states, out);
    }
}
