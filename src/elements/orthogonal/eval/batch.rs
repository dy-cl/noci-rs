// elements/orthogonal/eval/batch.rs
//! Parent-local batches of orthogonal Slater-Condon matrix elements.
//!
//! Every operator family, the overlap, the Fock operator and the Hamiltonian, supplies only its
//! scalar fixed-rank evaluator and its packed evaluator of the rank sectors whose formula is one
//! gather; this module groups requests by rank sector and evaluates the groups through the shared
//! lane batching.

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::DeterminantData;
#[cfg(target_arch = "x86_64")]
use crate::elements::LaneKernel;
use crate::elements::{LaneBatch, group_lanes, xw_lanes};
#[cfg(target_arch = "x86_64")]
use crate::maths::Simd;

// Parent/sibling imports.
#[cfg(target_arch = "x86_64")]
use super::dispatch::dispatch_orthogonal_ranks;

/// Number of fixed-rank sectors coupled by a two-body operator, `(0,0)`, `(1,0)`, `(0,1)`,
/// `(2,0)`, `(1,1)` and `(0,2)`.
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
    /// - `state`: Child data measured from the source: phases and fixed-rank excitation labels.
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
    /// - `states`: Data of every lane: fixed-rank excitation labels and phases.
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

/// Parent-local batch of one orthogonal operator, with its requests in rank-sector order.
struct OrthogonalLanes<'a, E, T> {
    /// Operator and the parent data it reads.
    element: &'a E,
    /// Source alpha and beta occupations in request order.
    occupations: &'a [(u128, u128)],
    /// Request of every lane.
    order: &'a [usize],
    /// Matrix elements in request order.
    out: &'a mut [T],
}

impl<E: OrthogonalElement<T>, T: NOCIScalar> LaneBatch for OrthogonalLanes<'_, E, T> {
    type Lane = DeterminantData;
    type Out = T;

    /// Evaluate one connection through the operator's scalar fixed-rank path.
    /// # Arguments:
    /// - `self`: Parent-local batch.
    /// - `position`: Position of the lane in rank-sector order.
    /// - `lane`: Connection of the request.
    /// # Returns
    /// - `T`: Matrix element of the request.
    fn scalar(
        &mut self,
        position: usize,
        lane: &DeterminantData,
    ) -> T {
        self.element
            .value(self.occupations[self.order[position]], lane)
    }

    /// Write the matrix element of one lane to its request.
    /// # Arguments:
    /// - `self`: Parent-local batch.
    /// - `position`: Position of the lane in rank-sector order.
    /// - `value`: Matrix element of the lane.
    /// # Returns
    /// - `()`: Writes one entry of `out`.
    fn write(
        &mut self,
        position: usize,
        value: T,
    ) {
        self.out[self.order[position]] = value;
    }

    /// Select the four-lane kernel of a rank sector the operator packs.
    /// # Arguments:
    /// - `self`: Parent-local batch.
    /// - `lane`: First connection of the sector.
    /// # Returns
    /// - `Option<LaneKernel<Self, 4>>`: Kernel of the sector, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn kernel_x4(
        &self,
        lane: &DeterminantData,
    ) -> Option<LaneKernel<Self, 4>> {
        let ranks = (
            usize::from(lane.alpha.cache.rank),
            usize::from(lane.beta.cache.rank),
        );
        if !E::packed_sector(ranks) {
            return None;
        }
        dispatch_orthogonal_ranks!(
            ranks,
            |RA, RB| Some(packet_x4::<E, T, RA, RB> as LaneKernel<Self, 4>),
            None,
        )
    }

    /// Select the eight-lane kernel of a rank sector the operator packs.
    /// # Arguments:
    /// - `self`: Parent-local batch.
    /// - `lane`: First connection of the sector.
    /// # Returns
    /// - `Option<LaneKernel<Self, 8>>`: Kernel of the sector, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn kernel_x8(
        &self,
        lane: &DeterminantData,
    ) -> Option<LaneKernel<Self, 8>> {
        let ranks = (
            usize::from(lane.alpha.cache.rank),
            usize::from(lane.beta.cache.rank),
        );
        if !E::packed_sector(ranks) {
            return None;
        }
        dispatch_orthogonal_ranks!(
            ranks,
            |RA, RB| Some(packet_x8::<E, T, RA, RB> as LaneKernel<Self, 8>),
            None,
        )
    }
}

/// Evaluate a parent-local batch of orthogonal matrix elements. Requests are grouped by rank
/// sector so every packet evaluates one Slater-Condon formula; requests in the operator's packed
/// sectors are evaluated in packets of the widest available SIMD kernel and every other request
/// uses the scalar fixed-rank evaluator.
/// # Arguments:
/// - `element`: Operator and the parent data it reads.
/// - `occupations`: Source alpha and beta occupations aligned with `states` and `out`.
/// - `states`: Data of every request: fixed-rank excitation labels and phases.
/// - `out`: Matrix elements in request order.
/// # Returns
/// - `()`: Writes every matrix element into `out`.
pub(crate) fn xw_orthogonal_batched<E: OrthogonalElement<T>, T: NOCIScalar>(
    element: &E,
    occupations: &[(u128, u128)],
    states: &[DeterminantData],
    out: &mut [T],
) {
    // Sectors beyond a two-body operator share one scalar group.
    let keys: Vec<usize> = states
        .iter()
        .map(
            |state| match (state.alpha.cache.rank, state.beta.cache.rank) {
                (0, 0) => 0,
                (1, 0) => 1,
                (0, 1) => 2,
                (2, 0) => 3,
                (1, 1) => 4,
                (0, 2) => 5,
                _ => SECTORS,
            },
        )
        .collect();
    let (order, groups) = group_lanes(&keys);
    let lanes: Vec<DeterminantData> = order.iter().map(|&request| states[request]).collect();

    // Evaluate every sector and write each value back to its request.

    let mut batch = OrthogonalLanes {
        element,
        occupations,
        order: &order,
        out,
    };
    xw_lanes(&mut batch, &lanes, &groups);
}

/// Evaluate one four-lane packet of an operator with AVX2/FMA.
/// # Arguments:
/// - `batch`: Batch holding the operator and the parent data it reads.
/// - `states`: Data of every lane: fixed-rank excitation labels and phases.
/// # Returns
/// - `[T; 4]`: Matrix element of every lane.
/// # Safety
/// - The current CPU must support AVX2 and FMA, and cached labels must match `(RA, RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn packet_x4<E: OrthogonalElement<T>, T: NOCIScalar, const RA: usize, const RB: usize>(
    batch: &OrthogonalLanes<'_, E, T>,
    states: &[DeterminantData; 4],
) -> [T; 4] {
    unsafe { batch.element.packed::<T::Packed4, 4, RA, RB>(states) }
}

/// Evaluate one eight-lane packet of an operator with AVX-512F.
/// # Arguments:
/// - `batch`: Batch holding the operator and the parent data it reads.
/// - `states`: Data of every lane: fixed-rank excitation labels and phases.
/// # Returns
/// - `[T; 8]`: Matrix element of every lane.
/// # Safety
/// - The current CPU must support AVX-512F, and cached labels must match `(RA, RB)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn packet_x8<E: OrthogonalElement<T>, T: NOCIScalar, const RA: usize, const RB: usize>(
    batch: &OrthogonalLanes<'_, E, T>,
    states: &[DeterminantData; 8],
) -> [T; 8] {
    unsafe { batch.element.packed::<T::Packed8, 8, RA, RB>(states) }
}
