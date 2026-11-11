// elements/orthogonal/eval/batch.rs
//! Parent-local batches of orthogonal Slater-Condon matrix elements.
//!
//! Every operator family, the overlap, the Fock operator and the Hamiltonian, supplies only its
//! scalar fixed-rank evaluator and its packed evaluator of the rank sectors whose formula is one
//! gather; this module dispatches to the widest packed kernel the CPU supports, bins requests by
//! rank sector, pads the last packet of each sector and scatters the results back to request
//! order.

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::DeterminantData;
#[cfg(target_arch = "x86_64")]
use crate::maths::{Lanes, Simd, lanes};

// Parent/sibling imports.
#[cfg(target_arch = "x86_64")]
use super::dispatch::dispatch_orthogonal_ranks;

/// Number of fixed-rank sectors coupled by a two-body operator, `(0,0)`, `(1,0)`, `(0,1)`,
/// `(2,0)`, `(1,1)` and `(0,2)`.
#[cfg(target_arch = "x86_64")]
const SECTORS: usize = 6;

/// An operator evaluated by the Slater-Condon rules in one parent's orthonormal frame.
pub(crate) trait OrthogonalElement<T: NOCIScalar> {
    /// Return whether the fixed-rank sector `(R_\alpha, R_\beta)` is evaluated in packets.
    /// # Arguments:
    /// - `ranks`: Alpha and beta excitation ranks of the sector.
    /// # Returns
    /// - `bool`: Whether the sector has a packed evaluator.
    fn packed_sector(ranks: (usize, usize)) -> bool;

    /// Evaluate one matrix element through the scalar fixed-rank path.
    /// # Arguments:
    /// - `self`: Operator and the parent data it reads.
    /// - `occupation`: Source alpha and beta occupation bitstrings.
    /// - `state`: Reduced phase and fixed-rank excitation labels connecting source to child.
    /// # Returns
    /// - `T`: Matrix element of the pair.
    fn value(
        &self,
        occupation: (u128, u128),
        state: &DeterminantData,
    ) -> T;

    /// Evaluate one packet of matrix elements of the fixed-rank sector `(RA, RB)`.
    /// # Arguments:
    /// - `self`: Operator and the parent data it reads.
    /// - `states`: Reduced excitation labels and phases of every lane.
    /// # Returns
    /// - `[T; N]`: Matrix element of every lane.
    /// # Safety
    /// - The caller must establish CPU support for `V`, and every cached orbital label must index
    ///   the operator's data.
    #[cfg(target_arch = "x86_64")]
    unsafe fn packed<V: Simd<N, Scalar = T>, const N: usize, const RA: usize, const RB: usize>(
        &self,
        states: &[DeterminantData; N],
    ) -> [T; N];
}

/// Packed fixed-rank kernel of one operator behind a target-feature boundary.
#[cfg(target_arch = "x86_64")]
type OrthogonalPacket<E, T, const N: usize> = unsafe fn(&E, &[DeterminantData; N]) -> [T; N];

/// Selector of the packed fixed-rank kernel of one packet width, by excitation ranks.
#[cfg(target_arch = "x86_64")]
type OrthogonalSelect<E, T, const N: usize> =
    fn((usize, usize)) -> Option<OrthogonalPacket<E, T, N>>;

/// Evaluate a parent-local batch of orthogonal matrix elements. Requests in the operator's packed
/// sectors are evaluated in packets of the widest available SIMD kernel, padding the last packet of
/// each sector with a valid lane; every other request uses the scalar fixed-rank evaluator.
/// # Arguments:
/// - `element`: Operator and the parent data it reads.
/// - `occupations`: Source alpha and beta occupations aligned with `states` and `out`.
/// - `states`: Reduced phase and excitation labels of every request.
/// - `out`: Matrix elements in request order.
/// # Returns
/// - `()`: Writes every matrix element into `out`.
pub(crate) fn xw_orthogonal_batched<E: OrthogonalElement<T>, T: NOCIScalar>(
    element: &E,
    occupations: &[(u128, u128)],
    states: &[DeterminantData],
    out: &mut [T],
) {
    // Packed kernels exist on CPUs with AVX2 and FMA.
    #[cfg(target_arch = "x86_64")]
    if let Some(lanes) = lanes() {
        unsafe {
            match lanes {
                Lanes::Four => xw_orthogonal_batched_x4(element, occupations, states, out),
                Lanes::Eight => xw_orthogonal_batched_x8(element, occupations, states, out),
            }
        }
        return;
    }

    // Without a packed kernel every request uses the scalar evaluator.
    for ((&occupation, state), value) in occupations.iter().zip(states).zip(out) {
        *value = element.value(occupation, state);
    }
}

/// Evaluate a parent-local batch with four-lane AVX2/FMA packets.
/// # Arguments:
/// - `element`: Operator and the parent data it reads.
/// - `occupations`: Source alpha and beta occupations aligned with `states` and `out`.
/// - `states`: Reduced phase and excitation labels of every request.
/// - `out`: Matrix elements in request order.
/// # Returns
/// - `()`: Writes every matrix element into `out`.
/// # Safety
/// - The current CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_orthogonal_batched_x4<E: OrthogonalElement<T>, T: NOCIScalar>(
    element: &E,
    occupations: &[(u128, u128)],
    states: &[DeterminantData],
    out: &mut [T],
) {
    xw_orthogonal_batched_lanes::<E, T, 4>(
        element,
        occupations,
        states,
        out,
        select_packet_x4::<E, T>,
    );
}

/// Evaluate a parent-local batch with eight-lane AVX-512 packets.
/// # Arguments:
/// - `element`: Operator and the parent data it reads.
/// - `occupations`: Source alpha and beta occupations aligned with `states` and `out`.
/// - `states`: Reduced phase and excitation labels of every request.
/// - `out`: Matrix elements in request order.
/// # Returns
/// - `()`: Writes every matrix element into `out`.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_orthogonal_batched_x8<E: OrthogonalElement<T>, T: NOCIScalar>(
    element: &E,
    occupations: &[(u128, u128)],
    states: &[DeterminantData],
    out: &mut [T],
) {
    xw_orthogonal_batched_lanes::<E, T, 8>(
        element,
        occupations,
        states,
        out,
        select_packet_x8::<E, T>,
    );
}

/// Evaluate a parent-local batch in packets of `N` lanes. Requests of the packed sectors are binned
/// by sector, every full bin is evaluated at once, and the last packet of each bin repeats its
/// first request in the unused lanes.
/// # Arguments:
/// - `element`: Operator and the parent data it reads.
/// - `occupations`: Source alpha and beta occupations aligned with `states` and `out`.
/// - `states`: Reduced phase and excitation labels of every request.
/// - `out`: Matrix elements in request order.
/// - `select`: Fixed-rank kernel of the packet width, by excitation ranks.
/// # Returns
/// - `()`: Writes every matrix element into `out`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn xw_orthogonal_batched_lanes<E: OrthogonalElement<T>, T: NOCIScalar, const N: usize>(
    element: &E,
    occupations: &[(u128, u128)],
    states: &[DeterminantData],
    out: &mut [T],
    select: OrthogonalSelect<E, T, N>,
) {
    let Some(&first) = states.first() else {
        return;
    };
    let mut bins = [[first; N]; SECTORS];
    let mut outputs = [[0usize; N]; SECTORS];
    let mut counts = [0usize; SECTORS];
    let mut kernels: [Option<OrthogonalPacket<E, T, N>>; SECTORS] = [None; SECTORS];

    // Bin the packed sectors so every packet evaluates one Slater-Condon formula.
    for (output, (&occupation, state)) in occupations.iter().zip(states).enumerate() {
        let ranks = (
            usize::from(state.alpha.cache.rank),
            usize::from(state.beta.cache.rank),
        );
        let sector = match ranks {
            (0, 0) => 0,
            (1, 0) => 1,
            (0, 1) => 2,
            (2, 0) => 3,
            (1, 1) => 4,
            (0, 2) => 5,
            _ => SECTORS,
        };
        let kernel = if sector < SECTORS && E::packed_sector(ranks) {
            select(ranks)
        } else {
            None
        };
        let Some(kernel) = kernel else {
            out[output] = element.value(occupation, state);
            continue;
        };

        let count = counts[sector];
        bins[sector][count] = *state;
        outputs[sector][count] = output;
        kernels[sector] = Some(kernel);
        counts[sector] += 1;

        // Evaluate and scatter a complete packet immediately.
        if count + 1 == N {
            let values = unsafe { kernel(element, &bins[sector]) };
            for lane in 0..N {
                out[outputs[sector][lane]] = values[lane];
            }
            counts[sector] = 0;
        }
    }

    // Pad the last packet of every sector with its first request; only genuine lanes are written.
    for sector in 0..SECTORS {
        let count = counts[sector];
        let Some(kernel) = kernels[sector] else {
            continue;
        };
        if count == 0 {
            continue;
        }
        let fill = bins[sector][0];
        bins[sector][count..].fill(fill);
        let values = unsafe { kernel(element, &bins[sector]) };
        for lane in 0..count {
            out[outputs[sector][lane]] = values[lane];
        }
    }
}

/// Select the four-lane fixed-rank kernel of one operator.
/// # Arguments:
/// - `ranks`: Alpha and beta excitation ranks.
/// # Returns
/// - `Option<OrthogonalPacket<E, T, 4>>`: Kernel of the ranks, or `None` outside the sectors of a
///   two-body operator.
#[cfg(target_arch = "x86_64")]
fn select_packet_x4<E: OrthogonalElement<T>, T: NOCIScalar>(
    ranks: (usize, usize)
) -> Option<OrthogonalPacket<E, T, 4>> {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| Some(packet_x4::<E, T, RA, RB> as OrthogonalPacket<E, T, 4>),
        None,
    )
}

/// Select the eight-lane fixed-rank kernel of one operator.
/// # Arguments:
/// - `ranks`: Alpha and beta excitation ranks.
/// # Returns
/// - `Option<OrthogonalPacket<E, T, 8>>`: Kernel of the ranks, or `None` outside the sectors of a
///   two-body operator.
#[cfg(target_arch = "x86_64")]
fn select_packet_x8<E: OrthogonalElement<T>, T: NOCIScalar>(
    ranks: (usize, usize)
) -> Option<OrthogonalPacket<E, T, 8>> {
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| Some(packet_x8::<E, T, RA, RB> as OrthogonalPacket<E, T, 8>),
        None,
    )
}

/// Evaluate one four-lane packet of an operator with AVX2/FMA.
/// # Arguments:
/// - `element`: Operator and the parent data it reads.
/// - `states`: Reduced excitation labels and phases of every lane.
/// # Returns
/// - `[T; 4]`: Matrix element of every lane.
/// # Safety
/// - The current CPU must support AVX2 and FMA, and cached labels must match `(RA, RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn packet_x4<E: OrthogonalElement<T>, T: NOCIScalar, const RA: usize, const RB: usize>(
    element: &E,
    states: &[DeterminantData; 4],
) -> [T; 4] {
    unsafe { element.packed::<T::Packed4, 4, RA, RB>(states) }
}

/// Evaluate one eight-lane packet of an operator with AVX-512F.
/// # Arguments:
/// - `element`: Operator and the parent data it reads.
/// - `states`: Reduced excitation labels and phases of every lane.
/// # Returns
/// - `[T; 8]`: Matrix element of every lane.
/// # Safety
/// - The current CPU must support AVX-512F, and cached labels must match `(RA, RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn packet_x8<E: OrthogonalElement<T>, T: NOCIScalar, const RA: usize, const RB: usize>(
    element: &E,
    states: &[DeterminantData; 8],
) -> [T; 8] {
    unsafe { element.packed::<T::Packed8, 8, RA, RB>(states) }
}
