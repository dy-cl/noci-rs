// noci/orthogonal/eval/preparehamiltonian.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

// External crate imports.
use num_complex::Complex64;

// Crate-root imports.
#[cfg(target_arch = "x86_64")]
use crate::maths::{C64x4, C64x8, F64x4, F64x8, Simd};
use crate::noci::{MOCache, NOCIScalar};
use crate::{AoData, ReducedTwoSpinState};

// Parent/sibling imports.
use super::dispatch::dispatch_orthogonal_ranks;

/// Evaluate `\langle D|\hat H|\Phi_x\rangle` from a prepared orthogonal excitation.
/// The fixed rank `(R_\alpha,R_\beta)` is restricted by the one- plus two-body Hamiltonian to
/// `(0,0)`, `(1,0)`, `(0,1)`, `(2,0)`, `(1,1)`, or `(0,2)`.
/// # Arguments:
/// - `ao`: AO data containing the nuclear-repulsion energy.
/// - `cache`: Parent-specific orthogonal MO Hamiltonian integrals.
/// - `source`: Source alpha and beta occupation bitstrings.
/// - `state`: Prepared phase and fixed-rank excitation labels connecting source to child.
/// # Returns
/// - `T`: Orthogonal Slater-Condon Hamiltonian matrix element.
#[inline(always)]
pub(crate) fn xw_hamiltonian_orthogonal_prepared<T: NOCIScalar>(
    ao: &AoData,
    cache: &MOCache<T>,
    source: (u128, u128),
    state: &ReducedTwoSpinState,
) -> T {
    let ranks = (
        usize::from(state.excitation_cache.alpha.rank),
        usize::from(state.excitation_cache.beta.rank),
    );

    // Dispatch by fixed spin excitation rank. This keeps Slater-Condon formulas compile-time
    // specialised while returning zero for ranks unsupported by a two-body Hamiltonian.
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| xw_hamiltonian_orthogonal_prepared_const::<T, RA, RB>(ao, cache, source, state,),
        T::from_real(0.0),
    )
}

/// Evaluate a fixed-rank orthogonal Slater-Condon Hamiltonian matrix element.
/// The compile-time branch implements the diagonal, alpha/beta singles, and three double sectors
/// without rediscovering excitation masks or fermionic phases.
/// For excitation phase `p`, singles use `p(h_ai + sum_j (ai||jj))`; same-spin doubles use
/// `p(ai||jb)`, mixed-spin doubles use `p(ai|jb)`, and ranks above two vanish.
/// # Arguments:
/// - `ao`: AO data containing `E_\mathrm{nuc}`.
/// - `cache`: Parent-specific one- and two-electron MO integrals.
/// - `source`: Source alpha and beta occupation bitstrings.
/// - `state`: Prepared phase and fixed-rank orbital labels.
/// # Returns
/// - `T`: Fixed-rank Hamiltonian matrix element.
#[inline(always)]
fn xw_hamiltonian_orthogonal_prepared_const<T: NOCIScalar, const RA: usize, const RB: usize>(
    ao: &AoData,
    cache: &MOCache<T>,
    source: (u128, u128),
    state: &ReducedTwoSpinState,
) -> T {
    let alpha = &state.excitation_cache.alpha;
    let beta = &state.excitation_cache.beta;
    let phase = T::from_real(state.phase);

    if RA == 0 && RB == 0 {
        // Diagonal element: accumulate nuclear, one-electron, same-spin, and opposite-spin terms
        // from occupied source orbitals.
        // `H_xx = E_\mathrm{nuc} + \sum_{i\sigma}h^\sigma_{ii}`
        // `+ \frac12\sum_{ij\sigma}\langle ii||jj\rangle`
        // `+ \sum_{i\in O_\alpha,j\in O_\beta}(ii|jj)`.
        let mut h = T::from_real(ao.enuc);
        let mut bits = source.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.ha[(i, i)];
        }

        let mut bits = source.1;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.hb[(i, i)];
        }

        let mut bits_i = source.0;
        while bits_i != 0 {
            let i = bits_i.trailing_zeros() as usize;
            bits_i &= bits_i - 1;
            let mut bits_j = source.0;
            while bits_j != 0 {
                let j = bits_j.trailing_zeros() as usize;
                bits_j &= bits_j - 1;
                h += T::from_real(0.5) * cache.eri_aa_asym[(i, i, j, j)];
            }
        }

        let mut bits_i = source.1;
        while bits_i != 0 {
            let i = bits_i.trailing_zeros() as usize;
            bits_i &= bits_i - 1;
            let mut bits_j = source.1;
            while bits_j != 0 {
                let j = bits_j.trailing_zeros() as usize;
                bits_j &= bits_j - 1;
                h += T::from_real(0.5) * cache.eri_bb_asym[(i, i, j, j)];
            }
        }

        let mut bits_i = source.0;
        while bits_i != 0 {
            let i = bits_i.trailing_zeros() as usize;
            bits_i &= bits_i - 1;
            let mut bits_j = source.1;
            while bits_j != 0 {
                let j = bits_j.trailing_zeros() as usize;
                bits_j &= bits_j - 1;
                h += cache.eri_ab_coul[(i, i, j, j)];
            }
        }

        return h;
    }

    if RA == 1 && RB == 0 {
        // Alpha single: remove the hole from alpha Coulomb/exchange sums while retaining every
        // beta occupied contribution.
        let i = usize::from(alpha.holes[0]);
        let a = usize::from(alpha.particles[0]);
        // `H_{a\leftarrow i,x} = p[h^\alpha_{ai}`
        // `+ \sum_{j\in O_\alpha\setminus i}(ai||jj)`
        // `+ \sum_{j\in O_\beta}(ai|jj)]`.
        let mut h = cache.ha[(a, i)];
        let mut bits = source.0 & !(1u128 << i);
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.eri_aa_asym[(a, i, j, j)];
        }

        let mut bits = source.1;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.eri_ab_coul[(a, i, j, j)];
        }

        return phase * h;
    }

    if RA == 0 && RB == 1 {
        // Beta single is the spin-swapped alpha-single expression.
        let i = usize::from(beta.holes[0]);
        let a = usize::from(beta.particles[0]);
        // `H_{a\leftarrow i,x} = p[h^\beta_{ai}`
        // `+ \sum_{j\in O_\beta\setminus i}(ai||jj)`
        // `+ \sum_{j\in O_\alpha}(jj|ia)]`.
        let mut h = cache.hb[(a, i)];
        let mut bits = source.1 & !(1u128 << i);
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.eri_bb_asym[(a, i, j, j)];
        }

        let mut bits = source.0;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.eri_ab_coul[(j, j, i, a)];
        }

        return phase * h;
    }

    if RA == 2 && RB == 0 {
        // Same-spin alpha double: only the antisymmetrised two-electron contribution remains.
        let i = usize::from(alpha.holes[0]);
        let j = usize::from(alpha.holes[1]);
        let a = usize::from(alpha.particles[0]);
        let b = usize::from(alpha.particles[1]);
        // `H_{ab\leftarrow ij,x} = p\langle ai||jb\rangle` for alpha-alpha doubles.
        return phase * cache.eri_aa_asym[(a, i, j, b)];
    }

    if RA == 0 && RB == 2 {
        // Same-spin beta double is the spin-swapped alpha-double expression.
        let i = usize::from(beta.holes[0]);
        let j = usize::from(beta.holes[1]);
        let a = usize::from(beta.particles[0]);
        let b = usize::from(beta.particles[1]);
        // `H_{ab\leftarrow ij,x} = p\langle ai||jb\rangle` for beta-beta doubles.
        return phase * cache.eri_bb_asym[(a, i, j, b)];
    }

    if RA == 1 && RB == 1 {
        // Mixed-spin double: only the Coulomb coupling between alpha and beta replacements remains.
        let i = usize::from(alpha.holes[0]);
        let j = usize::from(beta.holes[0]);
        let a = usize::from(alpha.particles[0]);
        let b = usize::from(beta.particles[0]);
        // `H_{ab\leftarrow ij,x} = p(ai|jb)` for alpha-beta doubles.
        return phase * cache.eri_ab_coul[(a, i, j, b)];
    }

    T::from_real(0.0)
}

/// Evaluate a parent-local batch of prepared orthogonal Hamiltonian matrix elements.
/// Double-excitation requests are packetised by `(R_\alpha,R_\beta)` for SIMD gather kernels;
/// diagonal and single requests use the fixed-rank scalar implementation.
/// # Arguments:
/// - `ao`: AO data containing the nuclear-repulsion energy.
/// - `cache`: Parent-specific orthogonal MO Hamiltonian integrals.
/// - `basis`: Retained determinant basis used to recover source occupations.
/// - `sources`: Source determinant indices aligned with `states` and `out`.
/// - `states`: Prepared phase/cache payloads for source-relative excitations.
/// - `out`: Hamiltonian results in request order.
/// # Returns
/// - `()`: Writes every parent-local matrix element into `out`.
pub(crate) fn xw_hamiltonian_orthogonal_prepared_batched<T: NOCIScalar>(
    ao: &AoData,
    cache: &MOCache<T>,
    occupations: &[(u128, u128)],
    states: &[ReducedTwoSpinState],
    out: &mut [T],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // Try packed kernels only after checking scalar type and CPU features. These kernels write
        // results in the same order as `occupations` and `states`.
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let cache_f64 = &*std::ptr::from_ref(cache).cast::<MOCache<f64>>();
            let out_f64 = std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len());
            if is_x86_feature_detected!("avx512f") {
                xw_hamiltonian_orthogonal_prepared_simd(
                    ao,
                    cache_f64,
                    occupations,
                    states,
                    out_f64,
                    xw_hamiltonian_orthogonal_prepared_f64x8,
                );
                return;
            }
            if is_x86_feature_detected!("avx2") {
                xw_hamiltonian_orthogonal_prepared_simd(
                    ao,
                    cache_f64,
                    occupations,
                    states,
                    out_f64,
                    xw_hamiltonian_orthogonal_prepared_f64x4,
                );
                return;
            }
        }

        if TypeId::of::<T>() == TypeId::of::<Complex64>() {
            let cache_c64 = &*std::ptr::from_ref(cache).cast::<MOCache<Complex64>>();
            let out_c64 =
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<Complex64>(), out.len());
            if is_x86_feature_detected!("avx512f") {
                xw_hamiltonian_orthogonal_prepared_simd(
                    ao,
                    cache_c64,
                    occupations,
                    states,
                    out_c64,
                    xw_hamiltonian_orthogonal_prepared_c64x8,
                );
                return;
            }
            if is_x86_feature_detected!("avx2") {
                xw_hamiltonian_orthogonal_prepared_simd(
                    ao,
                    cache_c64,
                    occupations,
                    states,
                    out_c64,
                    xw_hamiltonian_orthogonal_prepared_c64x4,
                );
                return;
            }
        }
    }

    // Scalar fallback handles unsupported CPUs and any request not accepted by packed kernels.
    for ((&occupation, state), value) in occupations.iter().zip(states).zip(out) {
        *value = xw_hamiltonian_orthogonal_prepared(ao, cache, occupation, state);
    }
}

/// Evaluate one prepared orthogonal batch through fixed-width double-excitation kernels.
/// Requests in `(2,0)`, `(1,1)`, and `(0,2)` bins use packed ERI gathers; all other ranks use
/// the scalar fixed-rank evaluator without changing request order.
/// # Arguments:
/// - `ao`: AO data containing the nuclear-repulsion energy.
/// - `cache`: Parent-specific orthogonal MO Hamiltonian integrals.
/// - `basis`: Retained determinant basis used by scalar single-excitation fallbacks.
/// - `sources`: Source determinant indices.
/// - `states`: Prepared excitation phases and caches.
/// - `out`: Hamiltonian results in request order.
/// - `kernel`: CPU-specific fixed-width double-excitation dispatcher.
/// # Returns
/// - `()`: Writes every matrix element into `out`.
/// # Safety
/// - The caller must provide a kernel supported by the current CPU.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::type_complexity)]
unsafe fn xw_hamiltonian_orthogonal_prepared_simd<T: NOCIScalar, const N: usize>(
    ao: &AoData,
    cache: &MOCache<T>,
    occupations: &[(u128, u128)],
    states: &[ReducedTwoSpinState],
    out: &mut [T],
    kernel: unsafe fn(&MOCache<T>, (usize, usize), &[ReducedTwoSpinState; N], &mut [T; N]),
) {
    if states.is_empty() {
        return;
    }

    // Group requests by fixed excitation rank so one packed kernel evaluates one Slater-Condon
    // formula across all lanes. Incomplete groups are padded internally and truncated on writeback.
    let mut bins = [[states[0]; N]; 3];
    let mut outputs = [[0usize; N]; 3];
    let mut counts = [0usize; 3];

    for (output, (&occupation, state)) in occupations.iter().zip(states).enumerate() {
        let ranks = (
            usize::from(state.excitation_cache.alpha.rank),
            usize::from(state.excitation_cache.beta.rank),
        );
        let bin = match ranks {
            (2, 0) => 0,
            (1, 1) => 1,
            (0, 2) => 2,
            _ => {
                // Diagonal and single sectors need occupied-orbital sums, so evaluate them scalar.
                out[output] = xw_hamiltonian_orthogonal_prepared(ao, cache, occupation, state);
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
            0 => (2, 0),
            1 => (1, 1),
            _ => (0, 2),
        };
        let mut values = [T::from_real(0.0); N];
        unsafe { kernel(cache, ranks, &bins[bin], &mut values) };
        for lane in 0..count {
            out[outputs[bin][lane]] = values[lane];
        }
    }
}

/// Gather and phase one fixed-rank packet of orthogonal double-excitation ERIs.
/// The three supported sectors evaluate `p(ai||jb)` or `p(ai|jb)` with one indexed gather and
/// one lane-wise real phase multiplication.
/// # Arguments:
/// - `cache`: Parent-specific orthogonal MO Hamiltonian integrals.
/// - `states`: Prepared excitation labels and phases in lane order.
/// - `out`: Hamiltonian outputs in lane order.
/// # Returns
/// - `()`: Writes one packet of fixed-rank double-excitation elements.
/// # Safety
/// - `V` must be supported by the current CPU and every cached orbital label must index `cache`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xw_hamiltonian_orthogonal_prepared_simd_const<
    T: NOCIScalar,
    V: Simd<N, Scalar = T>,
    const N: usize,
    const RA: usize,
    const RB: usize,
>(
    cache: &MOCache<T>,
    states: &[ReducedTwoSpinState; N],
    out: &mut [T; N],
) {
    let mut indices = [0usize; N];
    let mut phases = [1.0f64; N];

    if RA == 2 && RB == 0 {
        // Alpha double: gather `p(ai||jb)` from antisymmetrised alpha ERIs.
        let shape = cache.eri_aa_asym.shape();
        for lane in 0..N {
            let ex = &states[lane].excitation_cache.alpha;
            let i = usize::from(ex.holes[0]);
            let j = usize::from(ex.holes[1]);
            let a = usize::from(ex.particles[0]);
            let b = usize::from(ex.particles[1]);
            indices[lane] = ((a * shape[1] + i) * shape[2] + j) * shape[3] + b;
            phases[lane] = states[lane].phase;
        }
        let eri = cache
            .eri_aa_asym
            .as_slice()
            .expect("orthogonal alpha-alpha ERIs must be contiguous");
        let values = unsafe { V::gather(eri, &indices) };
        V::multiply_real_lanes(values, &phases).store(out);
        return;
    }

    if RA == 0 && RB == 2 {
        // Beta double: gather the spin-swapped `p(ai||jb)` contribution.
        let shape = cache.eri_bb_asym.shape();
        for lane in 0..N {
            let ex = &states[lane].excitation_cache.beta;
            let i = usize::from(ex.holes[0]);
            let j = usize::from(ex.holes[1]);
            let a = usize::from(ex.particles[0]);
            let b = usize::from(ex.particles[1]);
            indices[lane] = ((a * shape[1] + i) * shape[2] + j) * shape[3] + b;
            phases[lane] = states[lane].phase;
        }
        let eri = cache
            .eri_bb_asym
            .as_slice()
            .expect("orthogonal beta-beta ERIs must be contiguous");
        let values = unsafe { V::gather(eri, &indices) };
        V::multiply_real_lanes(values, &phases).store(out);
        return;
    }

    if RA == 1 && RB == 1 {
        // Mixed-spin double: exchange is absent, leaving the Coulomb term `p(ai|jb)`.
        let shape = cache.eri_ab_coul.shape();
        for lane in 0..N {
            let alpha = &states[lane].excitation_cache.alpha;
            let beta = &states[lane].excitation_cache.beta;
            let i = usize::from(alpha.holes[0]);
            let j = usize::from(beta.holes[0]);
            let a = usize::from(alpha.particles[0]);
            let b = usize::from(beta.particles[0]);
            indices[lane] = ((a * shape[1] + i) * shape[2] + j) * shape[3] + b;
            phases[lane] = states[lane].phase;
        }
        let eri = cache
            .eri_ab_coul
            .as_slice()
            .expect("orthogonal alpha-beta ERIs must be contiguous");
        let values = unsafe { V::gather(eri, &indices) };
        V::multiply_real_lanes(values, &phases).store(out);
    }
}

/// Dispatch four real orthogonal double-excitation values to AVX2 fixed-rank kernels.
/// # Arguments:
/// - `cache`: Parent-specific real MO integrals.
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Four prepared excitation states.
/// - `out`: Four Hamiltonian outputs.
/// # Returns
/// - `()`: Writes four real matrix elements.
/// # Safety
/// - The current CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_hamiltonian_orthogonal_prepared_f64x4(
    cache: &MOCache<f64>,
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 4],
    out: &mut [f64; 4],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe {
            xw_hamiltonian_orthogonal_prepared_f64x4_const::<RA, RB>(cache, states, out)
        },
        (),
    );
}

/// Evaluate four real fixed-rank orthogonal double-excitation values with AVX2.
/// # Arguments:
/// - `cache`: Parent-specific real MO integrals.
/// - `states`: Four prepared excitation states.
/// - `out`: Four Hamiltonian outputs.
/// # Returns
/// - `()`: Writes four real matrix elements.
/// # Safety
/// - The current CPU must support AVX2 and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_hamiltonian_orthogonal_prepared_f64x4_const<const RA: usize, const RB: usize>(
    cache: &MOCache<f64>,
    states: &[ReducedTwoSpinState; 4],
    out: &mut [f64; 4],
) {
    unsafe {
        xw_hamiltonian_orthogonal_prepared_simd_const::<f64, F64x4, 4, RA, RB>(cache, states, out);
    }
}

/// Dispatch eight real orthogonal double-excitation values to AVX-512 fixed-rank kernels.
/// # Arguments:
/// - `cache`: Parent-specific real MO integrals.
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Eight prepared excitation states.
/// - `out`: Eight Hamiltonian outputs.
/// # Returns
/// - `()`: Writes eight real matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_orthogonal_prepared_f64x8(
    cache: &MOCache<f64>,
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 8],
    out: &mut [f64; 8],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe {
            xw_hamiltonian_orthogonal_prepared_f64x8_const::<RA, RB>(cache, states, out)
        },
        (),
    );
}

/// Evaluate eight real fixed-rank orthogonal double-excitation values with AVX-512F.
/// # Arguments:
/// - `cache`: Parent-specific real MO integrals.
/// - `states`: Eight prepared excitation states.
/// - `out`: Eight Hamiltonian outputs.
/// # Returns
/// - `()`: Writes eight real matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_orthogonal_prepared_f64x8_const<const RA: usize, const RB: usize>(
    cache: &MOCache<f64>,
    states: &[ReducedTwoSpinState; 8],
    out: &mut [f64; 8],
) {
    unsafe {
        xw_hamiltonian_orthogonal_prepared_simd_const::<f64, F64x8, 8, RA, RB>(cache, states, out);
    }
}

/// Dispatch four complex orthogonal double-excitation values to AVX2 fixed-rank kernels.
/// # Arguments:
/// - `cache`: Parent-specific complex MO integrals.
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Four prepared excitation states.
/// - `out`: Four Hamiltonian outputs.
/// # Returns
/// - `()`: Writes four complex matrix elements.
/// # Safety
/// - The current CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_hamiltonian_orthogonal_prepared_c64x4(
    cache: &MOCache<Complex64>,
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 4],
    out: &mut [Complex64; 4],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe {
            xw_hamiltonian_orthogonal_prepared_c64x4_const::<RA, RB>(cache, states, out)
        },
        (),
    );
}

/// Evaluate four complex fixed-rank orthogonal double-excitation values with AVX2.
/// # Arguments:
/// - `cache`: Parent-specific complex MO integrals.
/// - `states`: Four prepared excitation states.
/// - `out`: Four Hamiltonian outputs.
/// # Returns
/// - `()`: Writes four complex matrix elements.
/// # Safety
/// - The current CPU must support AVX2 and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xw_hamiltonian_orthogonal_prepared_c64x4_const<const RA: usize, const RB: usize>(
    cache: &MOCache<Complex64>,
    states: &[ReducedTwoSpinState; 4],
    out: &mut [Complex64; 4],
) {
    unsafe {
        xw_hamiltonian_orthogonal_prepared_simd_const::<Complex64, C64x4, 4, RA, RB>(
            cache, states, out,
        );
    }
}

/// Dispatch eight complex orthogonal double-excitation values to AVX-512 fixed-rank kernels.
/// # Arguments:
/// - `cache`: Parent-specific complex MO integrals.
/// - `ranks`: Common alpha and beta excitation ranks.
/// - `states`: Eight prepared excitation states.
/// - `out`: Eight Hamiltonian outputs.
/// # Returns
/// - `()`: Writes eight complex matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_orthogonal_prepared_c64x8(
    cache: &MOCache<Complex64>,
    ranks: (usize, usize),
    states: &[ReducedTwoSpinState; 8],
    out: &mut [Complex64; 8],
) {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| unsafe {
            xw_hamiltonian_orthogonal_prepared_c64x8_const::<RA, RB>(cache, states, out)
        },
        (),
    );
}

/// Evaluate eight complex fixed-rank orthogonal double-excitation values with AVX-512F.
/// # Arguments:
/// - `cache`: Parent-specific complex MO integrals.
/// - `states`: Eight prepared excitation states.
/// - `out`: Eight Hamiltonian outputs.
/// # Returns
/// - `()`: Writes eight complex matrix elements.
/// # Safety
/// - The current CPU must support AVX-512F and cached labels must match `(RA,RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_orthogonal_prepared_c64x8_const<const RA: usize, const RB: usize>(
    cache: &MOCache<Complex64>,
    states: &[ReducedTwoSpinState; 8],
    out: &mut [Complex64; 8],
) {
    unsafe {
        xw_hamiltonian_orthogonal_prepared_simd_const::<Complex64, C64x8, 8, RA, RB>(
            cache, states, out,
        );
    }
}
