// elements/nonorthogonalwicks/eval/batch.rs
//! Rows of same-spin Wick factors for one ordered reference pair.
//!
//! A row pairs one target spin determinant with many source spin determinants. Every factor family,
//! the overlap and the one-body factors, supplies only its scalar evaluator and its packed
//! fixed-rank `m = 0` evaluator; this module sums near-zero branches, dispatches to the widest
//! packed kernel the CPU supports, groups sources into packets of one fixed-rank kernel and
//! restores the excitation phases.

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;

// External crate imports.
#[cfg(target_arch = "x86_64")]
use num_complex::Complex64;

// Crate-root imports.
use crate::ExcitationSpinCache;
use crate::NOCIScalar;
#[cfg(target_arch = "x86_64")]
use crate::config::MAXEXCIT;
use crate::determinant::ExcitedSpinState;
#[cfg(target_arch = "x86_64")]
use crate::maths::{Lanes, Packed, Simd, lanes};

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
#[cfg(target_arch = "x86_64")]
use super::dispatch::{dispatch_pair_ranks, dispatch_spin_ranks};

/// A same-spin Wick factor family evaluating `K` factors for every determinant pair, such as the
/// overlap alone or the overlap together with a one-body operator.
pub(crate) trait SpinFactor<const K: usize> {
    /// Evaluate the factors of one determinant pair through the scalar path, excluding the
    /// excitation phases.
    /// # Arguments:
    /// - `self`: Factor family and its evaluation parameters.
    /// - `w`: Same-spin reference-pair Wick intermediates.
    /// - `pair`: Bra and ket determinants, oriented as the reference pair `(x,w)`.
    /// - `scratch`: Reusable same-spin Wick workspace.
    /// # Returns
    /// - `[T; K]`: Factors of the determinant pair.
    fn values<T: NOCIScalar>(
        &self,
        w: &SameSpinView<'_, T>,
        pair: (&ExcitedSpinState, &ExcitedSpinState),
        scratch: &mut WickScratch<T>,
    ) -> [T; K];

    /// Evaluate the `m = 0` factors of one packet of fixed excitation ranks, excluding the
    /// excitation phases.
    /// # Arguments:
    /// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
    /// - `packet`: Contraction labels of every lane.
    /// # Returns
    /// - `[[T; LANES]; K]`: Factors of every lane.
    /// # Safety
    /// - Cached labels must match `RX` and `RW`, and the caller must establish CPU support for `V`.
    #[cfg(target_arch = "x86_64")]
    unsafe fn packed<
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
        packet: &Packet<'_, LANES>,
    ) -> [[T; LANES]; K];
}

/// Source spin determinants of factor rows in packed evaluation order. Sources are sorted by
/// excitation rank, holes and particles, so every group of equal rank and holes fills packets of
/// one fixed-rank kernel whose hole contractions are shared by every lane.
#[derive(Default)]
pub(crate) struct SpinSources {
    /// Source spin determinants in evaluation order.
    states: Vec<ExcitedSpinState>,
    /// Excitation cache of every source in evaluation order, contiguous so that a packet of
    /// sources is one array.
    caches: Vec<ExcitationSpinCache>,
    /// Output column of every source in evaluation order.
    columns: Vec<usize>,
    /// Boundaries of the equal-rank, common-hole groups in `states`.
    groups: Vec<usize>,
}

impl SpinSources {
    /// Sort source spin determinants into packed evaluation order.
    /// # Arguments:
    /// - `sources`: Source spin determinants in output-column order.
    /// # Returns
    /// - `Self`: Sources in evaluation order with their output columns and group boundaries.
    pub(crate) fn new(sources: &[ExcitedSpinState]) -> Self {
        // Order sources by rank, holes and particles so equal-rank, common-hole sources are
        // adjacent; ties keep source order.
        let mut columns: Vec<usize> = (0..sources.len()).collect();
        columns.sort_unstable_by(|&i, &j| {
            let ic = sources[i].state.excitation_cache;
            let jc = sources[j].state.excitation_cache;
            ic.rank
                .cmp(&jc.rank)
                .then_with(|| ic.holes.cmp(&jc.holes))
                .then_with(|| ic.particles.cmp(&jc.particles))
                .then_with(|| i.cmp(&j))
        });
        let states: Vec<ExcitedSpinState> = columns.iter().map(|&col| sources[col]).collect();
        let caches = states
            .iter()
            .map(|source| source.state.excitation_cache)
            .collect();

        // A new group starts wherever the rank or the holes change.
        let mut groups = vec![0];
        for position in 1..states.len() {
            let previous = states[position - 1].state.excitation_cache;
            let current = states[position].state.excitation_cache;
            if current.rank != previous.rank || current.holes != previous.holes {
                groups.push(position);
            }
        }
        if !states.is_empty() {
            groups.push(states.len());
        }

        Self {
            states,
            caches,
            columns,
            groups,
        }
    }

    /// Return the number of source spin determinants.
    /// # Arguments:
    /// - `self`: Sources in evaluation order.
    /// # Returns
    /// - `usize`: Number of sources, the length of every output row.
    pub(crate) fn len(&self) -> usize {
        self.states.len()
    }
}

/// One row of same-spin factors, pairing a target spin determinant with every source.
pub(crate) struct SpinFactorRow<'a, T, const K: usize> {
    /// Target spin determinant shared by the row.
    pub(crate) target: ExcitedSpinState,
    /// Source spin determinants in packed evaluation order.
    pub(crate) sources: &'a SpinSources,
    /// Whether the target belongs to the left reference of the Wick intermediates.
    pub(crate) target_left: bool,
    /// Output factors in source-column order, one row per factor.
    pub(crate) out: [&'a mut [T]; K],
}

/// Contraction labels of one packet. The fixed determinant is shared by every lane and the varying
/// determinants, one per lane, share their holes, so an entry of the contraction determinant is
/// shared by every lane whenever its labels come from the fixed determinant or from the holes.
#[cfg(target_arch = "x86_64")]
pub(crate) struct Packet<'a, const LANES: usize> {
    /// Excitation cache shared by every lane.
    fixed: &'a ExcitationSpinCache,
    /// Excitation cache of every lane.
    varying: &'a [ExcitationSpinCache; LANES],
    /// Number of occupied orbitals of the spin sector.
    nocc: usize,
    /// Number of virtual orbitals of the spin sector.
    nvirt: usize,
    /// Row length of the contraction matrices.
    n: usize,
}

#[cfg(target_arch = "x86_64")]
impl<'a, const LANES: usize> Packet<'a, LANES> {
    /// Collect the contraction labels of one packet.
    /// # Arguments:
    /// - `w`: Same-spin reference-pair Wick intermediates.
    /// - `fixed`: Excitation cache shared by every lane.
    /// - `varying`: Excitation cache of every lane.
    /// # Returns
    /// - `Self`: Contraction labels of the packet.
    fn new<T: NOCIScalar>(
        w: &SameSpinView<'_, T>,
        fixed: &'a ExcitationSpinCache,
        varying: &'a [ExcitationSpinCache; LANES],
    ) -> Self {
        Self {
            fixed,
            varying,
            nocc: w.nocc,
            nvirt: w.nmo - w.nocc,
            n: w.n(),
        }
    }

    /// Load entry `(\eta, z)` of a contraction matrix for every lane, `M_{r_\eta c_z}`, or
    /// `M_{c_z r_\eta}` when `transpose` is set. Rows `r_\eta` run over the bra particles and then
    /// the ket holes, columns `c_z` over the bra holes and then the ket particles. Entries shared
    /// by every lane are broadcast instead of gathered.
    /// # Arguments:
    /// - `values`: Row-major contraction matrix.
    /// - `eta`: Row of the contraction determinant.
    /// - `z`: Column of the contraction determinant.
    /// - `transpose`: Whether to read the transposed entry.
    /// # Returns
    /// - `V`: Entry of every lane.
    /// # Safety
    /// - Cached labels must match `RX` and index within `values`.
    #[inline(always)]
    pub(crate) unsafe fn load<
        T: Copy,
        V: Simd<LANES, Scalar = T>,
        const RX: usize,
        const XFIX: bool,
    >(
        &self,
        values: &[T],
        eta: usize,
        z: usize,
        transpose: bool,
    ) -> V {
        // The bra is fixed when `XFIX`; otherwise the ket is.
        let bra = |lane: usize| -> &ExcitationSpinCache {
            if XFIX {
                self.fixed
            } else {
                unsafe { self.varying.get_unchecked(lane) }
            }
        };
        let ket = |lane: usize| -> &ExcitationSpinCache {
            if XFIX {
                unsafe { self.varying.get_unchecked(lane) }
            } else {
                self.fixed
            }
        };
        let index = |lane: usize| -> usize {
            let row = if eta < RX {
                usize::from(unsafe { *bra(lane).particles.get_unchecked(eta) }) - self.nocc
            } else {
                self.nvirt + usize::from(unsafe { *ket(lane).holes.get_unchecked(eta - RX) })
            };
            let col = if z < RX {
                usize::from(unsafe { *bra(lane).holes.get_unchecked(z) })
            } else {
                usize::from(unsafe { *ket(lane).particles.get_unchecked(z - RX) })
            };
            if transpose {
                col * self.n + row
            } else {
                row * self.n + col
            }
        };

        // With a fixed bra only the ket particles vary, in columns `z \geq R_x`; with a fixed ket
        // only the bra particles vary, in rows `\eta < R_x`.
        let shared = if XFIX { z < RX } else { eta >= RX };
        if shared {
            V::splat(unsafe { *values.get_unchecked(index(0)) })
        } else {
            let mut lanes = [unsafe { *values.get_unchecked(index(0)) }; LANES];
            for (lane, value) in lanes.iter_mut().enumerate().skip(1) {
                *value = unsafe { *values.get_unchecked(index(lane)) };
            }
            V::load(&lanes)
        }
    }
}

/// Packed fixed-rank kernel of one factor family behind a target-feature boundary.
#[cfg(target_arch = "x86_64")]
type PacketKernel<T, const K: usize, const N: usize> = for<'a> unsafe fn(
    &SameSpinView<'a, T>,
    &ExcitationSpinCache,
    &[ExcitationSpinCache; N],
) -> [[T; N]; K];

/// Evaluate one row of same-spin factors for one ordered reference pair. Every output is
/// `p_x p_w {}^{xw}\tilde S\sum_{m_1+\cdots+m_L = m}` of the family's determinant expression.
/// Groups with `m = 0` and generated ranks are evaluated in packets of the widest available SIMD
/// kernel, padding the last packet of each group with a valid lane; every other source uses the
/// scalar evaluator.
/// # Arguments:
/// - `factor`: Factor family and its evaluation parameters.
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `row`: Target, sources and outputs of the row.
/// - `scratch`: Reusable same-spin Wick workspace for scalar evaluation.
/// # Returns
/// - `()`: Writes every factor of the row into `row.out`.
pub(crate) fn xw_spin_factor_row<F: SpinFactor<K>, T: NOCIScalar, const K: usize>(
    factor: &F,
    w: &SameSpinView<'_, T>,
    mut row: SpinFactorRow<'_, T, K>,
    scratch: &mut WickScratch<T>,
) {
    // A spin sector with near-zero orbital pairs is the sum of its branches.
    if w.nbranches() > 1 {
        let zero = T::from_real(0.0);
        let ncol = row.out[0].len();
        let mut parts: [Vec<T>; K] = std::array::from_fn(|_| vec![zero; ncol]);
        for out in row.out.iter_mut() {
            out.fill(zero);
        }
        for k in 0..w.nbranches() {
            let branch = SpinFactorRow {
                target: row.target,
                sources: row.sources,
                target_left: row.target_left,
                out: parts.each_mut().map(|part| part.as_mut_slice()),
            };
            xw_spin_factor_row(factor, &w.branch(k), branch, scratch);
            for (out, part) in row.out.iter_mut().zip(&parts) {
                for (value, branch) in out.iter_mut().zip(part) {
                    *value += *branch;
                }
            }
        }
        return;
    }

    // Packed kernels exist for real and complex scalars with `m = 0`.
    #[cfg(target_arch = "x86_64")]
    if w.m == 0 {
        // SAFETY: each call names the packed type that the `TypeId` check proves equal to `T`.
        let packed = unsafe {
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                xw_spin_factor_row_packed::<F, T, f64, K>(factor, w, &mut row, scratch)
            } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                xw_spin_factor_row_packed::<F, T, Complex64, K>(factor, w, &mut row, scratch)
            } else {
                false
            }
        };
        if packed {
            return;
        }
    }

    // Without a packed kernel every source uses the scalar evaluator.
    for position in 0..row.sources.len() {
        xw_spin_factor_scalar(factor, w, &mut row, position, scratch);
    }
}

/// Evaluate the factors of the row target and one source through the scalar path and write them
/// into the output column of the source.
/// # Arguments:
/// - `factor`: Factor family and its evaluation parameters.
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `row`: Row holding the target, sources and outputs.
/// - `position`: Position of the source in evaluation order.
/// - `scratch`: Reusable same-spin Wick workspace.
/// # Returns
/// - `()`: Writes one column of `row.out`.
fn xw_spin_factor_scalar<F: SpinFactor<K>, T: NOCIScalar, const K: usize>(
    factor: &F,
    w: &SameSpinView<'_, T>,
    row: &mut SpinFactorRow<'_, T, K>,
    position: usize,
    scratch: &mut WickScratch<T>,
) {
    // Orient the pair as the ordered reference pair `(x,w)`.
    let source = &row.sources.states[position];
    let pair = if row.target_left {
        (&row.target, source)
    } else {
        (source, &row.target)
    };
    let values = factor.values(w, pair, scratch);
    write_spin_factors(row, position, values);
}

/// Write the factors of one source into its output column, restoring the target and source
/// excitation phases `p_x p_w`.
/// # Arguments:
/// - `row`: Row receiving the factors.
/// - `position`: Position of the source in evaluation order.
/// - `values`: Factors of the pair, excluding the excitation phases.
/// # Returns
/// - `()`: Writes one column of `row.out`.
fn write_spin_factors<T: NOCIScalar, const K: usize>(
    row: &mut SpinFactorRow<'_, T, K>,
    position: usize,
    values: [T; K],
) {
    let column = row.sources.columns[position];
    let phase = T::from_real(row.target.state.phase * row.sources.states[position].state.phase);
    for (out, value) in row.out.iter_mut().zip(values) {
        out[column] = phase * value;
    }
}

/// Evaluate one row with the packed kernels of scalar type `U`, the scalar type `T` of the row.
/// # Arguments:
/// - `factor`: Factor family and its evaluation parameters.
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `row`: Target, sources and outputs of the row.
/// - `scratch`: Reusable same-spin Wick workspace for scalar evaluation.
/// # Returns
/// - `bool`: Whether the CPU supports a packed kernel and the complete row was written.
/// # Safety
/// - `T` and `U` must be the same type.
#[cfg(target_arch = "x86_64")]
unsafe fn xw_spin_factor_row_packed<
    F: SpinFactor<K>,
    T: NOCIScalar,
    U: NOCIScalar + Packed,
    const K: usize,
>(
    factor: &F,
    w: &SameSpinView<'_, T>,
    row: &mut SpinFactorRow<'_, T, K>,
    scratch: &mut WickScratch<T>,
) -> bool {
    let Some(lanes) = lanes() else {
        return false;
    };

    // SAFETY: `T` and `U` are the same type, so every reference is reinterpreted as itself.
    let w = unsafe { &*std::ptr::from_ref(w).cast::<SameSpinView<'_, U>>() };
    let scratch = unsafe { &mut *std::ptr::from_mut(scratch).cast::<WickScratch<U>>() };
    let mut row = SpinFactorRow {
        target: row.target,
        sources: row.sources,
        target_left: row.target_left,
        out: row.out.each_mut().map(|out| unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<U>(), out.len())
        }),
    };

    unsafe {
        match lanes {
            Lanes::Four => xw_spin_factor_row_x4::<F, U, K>(factor, w, &mut row, scratch),
            Lanes::Eight => xw_spin_factor_row_x8::<F, U, K>(factor, w, &mut row, scratch),
        }
    }
    true
}

/// Evaluate one row with four-lane AVX2/FMA packets.
/// # Arguments:
/// - `factor`: Factor family and its evaluation parameters.
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `row`: Target, sources and outputs of the row.
/// - `scratch`: Reusable same-spin Wick workspace for scalar evaluation.
/// # Returns
/// - `()`: Writes every factor of the row.
/// # Safety
/// - The current CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_spin_factor_row_x4<F: SpinFactor<K>, T: NOCIScalar + Packed, const K: usize>(
    factor: &F,
    w: &SameSpinView<'_, T>,
    row: &mut SpinFactorRow<'_, T, K>,
    scratch: &mut WickScratch<T>,
) {
    xw_spin_factor_row_lanes::<F, T, K, 4>(factor, w, row, scratch, select_packet_x4::<F, T, K>);
}

/// Evaluate one row with eight-lane AVX-512 packets.
/// # Arguments:
/// - `factor`: Factor family and its evaluation parameters.
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `row`: Target, sources and outputs of the row.
/// - `scratch`: Reusable same-spin Wick workspace for scalar evaluation.
/// # Returns
/// - `()`: Writes every factor of the row.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_spin_factor_row_x8<F: SpinFactor<K>, T: NOCIScalar + Packed, const K: usize>(
    factor: &F,
    w: &SameSpinView<'_, T>,
    row: &mut SpinFactorRow<'_, T, K>,
    scratch: &mut WickScratch<T>,
) {
    xw_spin_factor_row_lanes::<F, T, K, 8>(factor, w, row, scratch, select_packet_x8::<F, T, K>);
}

/// Evaluate one row in packets of `N` lanes. Every group of equal rank and holes is split into
/// packets of one fixed-rank kernel, and the last packet of a group repeats its final source in
/// the unused lanes. Groups outside the generated ranks use the scalar evaluator.
/// # Arguments:
/// - `factor`: Factor family and its evaluation parameters.
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `row`: Target, sources and outputs of the row.
/// - `scratch`: Reusable same-spin Wick workspace for scalar evaluation.
/// - `select`: Fixed-rank kernel of the packet width, by excitation ranks and fixed side.
/// # Returns
/// - `()`: Writes every factor of the row.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn xw_spin_factor_row_lanes<F: SpinFactor<K>, T: NOCIScalar, const K: usize, const N: usize>(
    factor: &F,
    w: &SameSpinView<'_, T>,
    row: &mut SpinFactorRow<'_, T, K>,
    scratch: &mut WickScratch<T>,
    select: fn((usize, usize), bool) -> Option<PacketKernel<T, K, N>>,
) {
    let target = row.target.state.excitation_cache;
    let target_rank = usize::from(target.rank);
    let sources = row.sources;

    for bounds in sources.groups.windows(2) {
        let (start, end) = (bounds[0], bounds[1]);
        let source_rank = usize::from(sources.states[start].state.excitation_cache.rank);
        let ranks = if row.target_left {
            (target_rank, source_rank)
        } else {
            (source_rank, target_rank)
        };

        // Empty determinants and ranks beyond the generated table use the scalar evaluator,
        // which covers arbitrary `L`.
        let kernel =
            if target_rank <= MAXEXCIT && source_rank <= MAXEXCIT && target_rank + source_rank != 0
            {
                select(ranks, row.target_left)
            } else {
                None
            };
        let Some(kernel) = kernel else {
            for position in start..end {
                xw_spin_factor_scalar(factor, w, row, position, scratch);
            }
            continue;
        };

        // Evaluate the group in packets, padding the last one with its final source.
        let mut padded = [ExcitationSpinCache::default(); N];
        for packet_start in (start..end).step_by(N) {
            let count = (end - packet_start).min(N);
            let varying: &[ExcitationSpinCache; N] = if count == N {
                sources.caches[packet_start..packet_start + N]
                    .try_into()
                    .unwrap()
            } else {
                padded[..count].copy_from_slice(&sources.caches[packet_start..end]);
                padded[count..].fill(sources.caches[end - 1]);
                &padded
            };

            // Scatter the genuine lanes to their output columns.
            let values = unsafe { kernel(w, &target, varying) };
            for lane in 0..count {
                let mut lane_values = [T::from_real(0.0); K];
                for (value, factor) in lane_values.iter_mut().zip(&values) {
                    *value = factor[lane];
                }
                write_spin_factors(row, packet_start + lane, lane_values);
            }
        }
    }
}

/// Select the four-lane fixed-rank kernel of one factor family.
/// # Arguments:
/// - `ranks`: Bra and ket excitation ranks.
/// - `x_fixed`: Whether the shared excitation belongs to the bra.
/// # Returns
/// - `Option<PacketKernel<T, K, 4>>`: Kernel of the ranks, or `None` outside the generated table.
#[cfg(target_arch = "x86_64")]
fn select_packet_x4<F: SpinFactor<K>, T: NOCIScalar + Packed, const K: usize>(
    ranks: (usize, usize),
    x_fixed: bool,
) -> Option<PacketKernel<T, K, 4>> {
    if x_fixed {
        dispatch_spin_ranks!(
            ranks,
            |RX, RW, L, D| Some(packet_x4::<F, T, K, RX, RW, L, D, true> as PacketKernel<T, K, 4>),
            None,
        )
    } else {
        dispatch_spin_ranks!(
            ranks,
            |RX, RW, L, D| Some(packet_x4::<F, T, K, RX, RW, L, D, false> as PacketKernel<T, K, 4>),
            None,
        )
    }
}

/// Select the eight-lane fixed-rank kernel of one factor family.
/// # Arguments:
/// - `ranks`: Bra and ket excitation ranks.
/// - `x_fixed`: Whether the shared excitation belongs to the bra.
/// # Returns
/// - `Option<PacketKernel<T, K, 8>>`: Kernel of the ranks, or `None` outside the generated table.
#[cfg(target_arch = "x86_64")]
fn select_packet_x8<F: SpinFactor<K>, T: NOCIScalar + Packed, const K: usize>(
    ranks: (usize, usize),
    x_fixed: bool,
) -> Option<PacketKernel<T, K, 8>> {
    if x_fixed {
        dispatch_spin_ranks!(
            ranks,
            |RX, RW, L, D| Some(packet_x8::<F, T, K, RX, RW, L, D, true> as PacketKernel<T, K, 8>),
            None,
        )
    } else {
        dispatch_spin_ranks!(
            ranks,
            |RX, RW, L, D| Some(packet_x8::<F, T, K, RX, RW, L, D, false> as PacketKernel<T, K, 8>),
            None,
        )
    }
}

/// Evaluate one four-lane packet of a factor family with AVX2/FMA.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `fixed`: Excitation cache shared by every lane.
/// - `varying`: Excitation cache of every lane.
/// # Returns
/// - `[[T; 4]; K]`: Factors of every lane, excluding the excitation phases.
/// # Safety
/// - The current CPU must support AVX2 and FMA, and cached labels must match the fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn packet_x4<
    F: SpinFactor<K>,
    T: NOCIScalar + Packed,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, T>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 4],
) -> [[T; 4]; K] {
    let packet = Packet::new(w, fixed, varying);
    unsafe { F::packed::<T, T::X4, 4, RX, RW, L, D, XFIX>(w, &packet) }
}

/// Evaluate one eight-lane packet of a factor family with AVX-512F.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `fixed`: Excitation cache shared by every lane.
/// - `varying`: Excitation cache of every lane.
/// # Returns
/// - `[[T; 8]; K]`: Factors of every lane, excluding the excitation phases.
/// # Safety
/// - The current CPU must support AVX-512F, and cached labels must match the fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn packet_x8<
    F: SpinFactor<K>,
    T: NOCIScalar + Packed,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const XFIX: bool,
>(
    w: &SameSpinView<'_, T>,
    fixed: &ExcitationSpinCache,
    varying: &[ExcitationSpinCache; 8],
) -> [[T; 8]; K] {
    let packet = Packet::new(w, fixed, varying);
    unsafe { F::packed::<T, T::X8, 8, RX, RW, L, D, XFIX>(w, &packet) }
}
