// elements/batch.rs
//! Lane batching shared by every packed matrix-element evaluator.
//!
//! A batch is a slice of lanes arranged in contiguous groups whose lanes share one packed kernel:
//! the rank sector of an orthogonal connection, the rank and holes of the sources of a same-spin
//! factor row, the four excitation ranks of a two-spin Wick pair, or every index request of one
//! RDM pair. Each family supplies its scalar evaluator, the write of one result and, for every
//! packet width, the packed kernel of a group; this module dispatches to the widest width the CPU
//! supports, splits every group into packets, pads the last packet of a group with its final lane
//! and writes only the genuine lanes. `packet_x4` and `packet_x8` are the only target-feature
//! boundaries: each family supplies the packed kernel of a group as a `LanePacket`, written once
//! over the packet width.

// Crate-root imports.
#[cfg(target_arch = "x86_64")]
use crate::NOCIScalar;
#[cfg(target_arch = "x86_64")]
use crate::maths::{self, Lanes, Simd};

/// A batch of lanes of one operator family, evaluated in packets where a packed kernel exists and
/// lane by lane otherwise.
pub(crate) trait LaneBatch: Sized {
    /// Payload of one lane: an orthogonal connection, same-spin excitation cache, two-spin
    /// determinant pair or RDM index request. This associated type lets `xw_lanes` pass each
    /// family's payload directly without converting it to a shared representation.
    type Lane: Copy;
    /// Result of one lane: a scalar element, `K` same-spin factors or the pair `(H,S)`.
    /// Required alongside `Lane` so one packet loop can write each family's actual output type.
    type Out: Copy;

    /// Evaluate one lane through the scalar path.
    /// # Arguments:
    /// - `self`: Batch and the data its evaluators read.
    /// - `position`: Position of the lane in the batch.
    /// - `lane`: Payload of the lane.
    /// # Returns
    /// - `Self::Out`: Result of the lane.
    fn scalar(
        &mut self,
        position: usize,
        lane: &Self::Lane,
    ) -> Self::Out;

    /// Write the result of one lane to its output.
    /// # Arguments:
    /// - `self`: Batch holding the outputs.
    /// - `position`: Position of the lane in the batch.
    /// - `value`: Result of the lane.
    /// # Returns
    /// - `()`: Writes one output.
    fn write(
        &mut self,
        position: usize,
        value: Self::Out,
    );

    /// Select the four-lane packed kernel of the group starting at one lane.
    /// # Arguments:
    /// - `self`: Batch and the data its evaluators read.
    /// - `lane`: First lane of the group.
    /// # Returns
    /// - `Option<LaneKernel<Self, 4>>`: Kernel of the group, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn kernel_x4(
        &self,
        lane: &Self::Lane,
    ) -> Option<LaneKernel<Self, 4>>;

    /// Select the eight-lane packed kernel of the group starting at one lane.
    /// # Arguments:
    /// - `self`: Batch and the data its evaluators read.
    /// - `lane`: First lane of the group.
    /// # Returns
    /// - `Option<LaneKernel<Self, 8>>`: Kernel of the group, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn kernel_x8(
        &self,
        lane: &Self::Lane,
    ) -> Option<LaneKernel<Self, 8>>;
}

/// The packed kernel of one group of a lane batch, written once over the packet width. Each family
/// supplies one zero-size implementor whose const parameters are the fixed ranks of the group, so
/// `packet_x4` and `packet_x8` evaluate every family's kernel inside their target features.
#[cfg(target_arch = "x86_64")]
pub(crate) trait LanePacket<B: LaneBatch, T: NOCIScalar> {
    /// Evaluate one packet of the group with `N` lanes of the packed type `V`.
    /// # Arguments:
    /// - `batch`: Batch and the data its evaluators read.
    /// - `lanes`: Lanes of the packet, all in the group.
    /// # Returns
    /// - `[B::Out; N]`: Result of every lane.
    /// # Safety
    /// - The caller must establish CPU support for `V`, and every lane must match the fixed ranks.
    unsafe fn evaluate<V: Simd<N, Scalar = T>, const N: usize>(
        batch: &B,
        lanes: &[B::Lane; N],
    ) -> [B::Out; N];
}

/// Packed kernel of one group behind a target-feature boundary.
#[cfg(target_arch = "x86_64")]
pub(crate) type LaneKernel<B, const N: usize> =
    unsafe fn(&B, &[<B as LaneBatch>::Lane; N]) -> [<B as LaneBatch>::Out; N];

/// Selector of the packed kernel of the group starting at one lane, for one packet width.
#[cfg(target_arch = "x86_64")]
type LaneSelect<B, const N: usize> = fn(&B, &<B as LaneBatch>::Lane) -> Option<LaneKernel<B, N>>;

/// Evaluate a batch of grouped lanes. Groups with a packed kernel are evaluated in packets of the
/// widest width the CPU supports; every other lane uses the scalar evaluator.
/// # Arguments:
/// - `batch`: Batch and the data its evaluators read.
/// - `lanes`: Lanes in group order.
/// - `groups`: Boundaries of the groups in `lanes`, starting at zero and ending at its length.
/// # Returns
/// - `()`: Writes the result of every lane.
pub(crate) fn xw_lanes<B: LaneBatch>(
    batch: &mut B,
    lanes: &[B::Lane],
    groups: &[usize],
) {
    // Packed kernels exist on CPUs with AVX2 and FMA.
    #[cfg(target_arch = "x86_64")]
    if let Some(width) = maths::lanes() {
        match width {
            Lanes::Four => xw_lane_packets(batch, lanes, groups, B::kernel_x4),
            Lanes::Eight => xw_lane_packets(batch, lanes, groups, B::kernel_x8),
        }
        return;
    }

    // Without a packed kernel every lane uses the scalar evaluator.
    for (position, lane) in lanes.iter().enumerate() {
        let value = batch.scalar(position, lane);
        batch.write(position, value);
    }
}

/// Evaluate a batch of grouped lanes in packets of `N`. Every group with a kernel is split into
/// packets, and the last packet of a group repeats its final lane in the unused lanes.
/// # Arguments:
/// - `batch`: Batch and the data its evaluators read.
/// - `lanes`: Lanes in group order.
/// - `groups`: Boundaries of the groups in `lanes`.
/// - `select`: Kernel selector of the packet width, which the CPU must support.
/// # Returns
/// - `()`: Writes the result of every lane.
#[cfg(target_arch = "x86_64")]
fn xw_lane_packets<B: LaneBatch, const N: usize>(
    batch: &mut B,
    lanes: &[B::Lane],
    groups: &[usize],
    select: LaneSelect<B, N>,
) {
    for bounds in groups.windows(2) {
        let (start, end) = (bounds[0], bounds[1]);
        let Some(kernel) = select(batch, &lanes[start]) else {
            for (position, lane) in lanes.iter().enumerate().take(end).skip(start) {
                let value = batch.scalar(position, lane);
                batch.write(position, value);
            }
            continue;
        };

        // Evaluate full packets in place and pad the last one with the final lane of the group.
        let mut padded = [lanes[start]; N];
        for first in (start..end).step_by(N) {
            let count = (end - first).min(N);
            let packet: &[B::Lane; N] = if count == N {
                lanes[first..first + N].try_into().unwrap()
            } else {
                padded[..count].copy_from_slice(&lanes[first..end]);
                padded[count..].fill(lanes[end - 1]);
                &padded
            };

            // Only the genuine lanes are written.
            let values = unsafe { kernel(batch, packet) };
            for (lane, &value) in values.iter().enumerate().take(count) {
                batch.write(first + lane, value);
            }
        }
    }
}

/// Evaluate one four-lane packet of a group with AVX2/FMA.
/// # Arguments:
/// - `batch`: Batch and the data its evaluators read.
/// - `lanes`: Lanes of the packet, all in the group of `P`.
/// # Returns
/// - `[B::Out; 4]`: Result of every lane.
/// # Safety
/// - The current CPU must support AVX2 and FMA, and every lane must match the ranks of `P`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn packet_x4<B: LaneBatch, T: NOCIScalar, P: LanePacket<B, T>>(
    batch: &B,
    lanes: &[B::Lane; 4],
) -> [B::Out; 4] {
    unsafe { P::evaluate::<T::Packed4, 4>(batch, lanes) }
}

/// Evaluate one eight-lane packet of a group with AVX-512F.
/// # Arguments:
/// - `batch`: Batch and the data its evaluators read.
/// - `lanes`: Lanes of the packet, all in the group of `P`.
/// # Returns
/// - `[B::Out; 8]`: Result of every lane.
/// # Safety
/// - The current CPU must support AVX-512F, and every lane must match the ranks of `P`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn packet_x8<B: LaneBatch, T: NOCIScalar, P: LanePacket<B, T>>(
    batch: &B,
    lanes: &[B::Lane; 8],
) -> [B::Out; 8] {
    unsafe { P::evaluate::<T::Packed8, 8>(batch, lanes) }
}

/// Arrange lanes into contiguous groups by key, keeping request order within a group.
/// # Arguments:
/// - `keys`: Group key of every lane.
/// # Returns
/// - `(Vec<usize>, Vec<usize>)`: Lane of every position in group order, and the boundaries of the
///   groups.
pub(crate) fn group_lanes(keys: &[usize]) -> (Vec<usize>, Vec<usize>) {
    // A stable sort by key keeps request order within each group.
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&lane| keys[lane]);

    // A new group starts wherever the key changes.
    let mut bounds = vec![0];
    for position in 1..order.len() {
        if keys[order[position]] != keys[order[position - 1]] {
            bounds.push(position);
        }
    }
    if !order.is_empty() {
        bounds.push(order.len());
    }
    (order, bounds)
}
