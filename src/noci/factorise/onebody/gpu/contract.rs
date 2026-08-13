// noci/factorise/onebody/gpu/contract.rs
//! GPU dense contractions for factorised one-body NOCI operator contractions.

// External crate imports.
use cubecl::prelude::*;

// Crate-root imports.
use crate::gpu::{GpuBuffer, GpuContext, GpuRuntime};

// Parent/sibling imports.
use super::consts::{
    GEMM_TILE_DIM, GROUPED_FINAL_BLOCK_FIELDS, GROUPED_FINAL_CONTRIBUTION_FIELDS,
    GROUPED_STAGE_BLOCK_FIELDS, GROUPED_STAGE_CONTRIBUTION_FIELDS,
};
use super::data::DeviceOneBodyData;

/// Initial contraction kernel cube dimension.
const CONTRACT_CUBE_DIM: u32 = 128;

/// Launch metadata for one heterogeneous grouped rank contraction.
#[derive(Clone, Copy)]
pub(crate) struct GroupedRankLaunch {
    /// Total flattened output-tile count.
    pub(crate) tiles: usize,
    /// Number of output block descriptors.
    pub(crate) blocks: usize,
    /// Full source-spin factor row stride.
    pub(crate) factor_stride: usize,
    /// Full other-spin intermediate row stride.
    pub(crate) intermediate_stride: usize,
    /// Overlap shift used by the final contraction.
    pub(crate) lambda: f64,
    /// MPI worker id.
    pub(crate) worker: usize,
    /// MPI worker count.
    pub(crate) nworker: usize,
}

/// Launch one grouped rank stage containing every populated output sector.
/// # Arguments:
/// - Primitive factor, coefficient, component-map and intermediate buffers.
/// - `blocks`: Persistent output block descriptors.
/// - `contributions`: Persistent inner-rank contribution descriptors.
/// - `args`: Flattened tile count, strides and ownership.
/// - `a_first`: Whether to evaluate the alpha-first rather than beta-first equations.
/// # Returns
/// - `()`: Writes every grouped first-stage output with one tile writer.
pub(crate) fn launch_grouped_rank_stage(
    context: &GpuContext,
    s: &GpuBuffer<f64>,
    f: &GpuBuffer<f64>,
    d: &GpuBuffer<f64>,
    row_components: &GpuBuffer<u32>,
    col_components: &GpuBuffer<u32>,
    source_components: &GpuBuffer<u32>,
    blocks: &GpuBuffer<u32>,
    contributions: &GpuBuffer<u32>,
    out_f: &GpuBuffer<f64>,
    out_s: &GpuBuffer<f64>,
    args: GroupedRankLaunch,
    a_first: bool,
) {
    if args.tiles == 0 {
        return;
    }
    let tile = checked_u32(GEMM_TILE_DIM);
    unsafe {
        grouped_rank_stage_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(checked_u32(args.tiles), 1, 1),
            CubeDim::new_2d(tile, tile),
            s.array_arg(),
            f.array_arg(),
            d.array_arg(),
            row_components.array_arg(),
            col_components.array_arg(),
            source_components.array_arg(),
            blocks.array_arg(),
            contributions.array_arg(),
            out_f.array_arg(),
            out_s.array_arg(),
            args.blocks,
            args.factor_stride,
            args.intermediate_stride,
            args.worker,
            args.nworker,
            a_first,
        );
    }
}

/// Launch one grouped rank final containing every actual target sector.
/// # Arguments:
/// - Primitive factor, intermediate, topology, descriptor and output buffers.
/// - `args`: Flattened tile count, strides, shift and ownership.
/// - `a_first`: Whether to evaluate the alpha-first rather than beta-first equations.
/// # Returns
/// - `()`: Accumulates every grouped target sector with one tile writer.
pub(crate) fn launch_grouped_rank_final(
    context: &GpuContext,
    s: &GpuBuffer<f64>,
    f: &GpuBuffer<f64>,
    intermediate_f: &GpuBuffer<f64>,
    intermediate_s: &GpuBuffer<f64>,
    alpha_components: &GpuBuffer<u32>,
    beta_components: &GpuBuffer<u32>,
    source_components: &GpuBuffer<u32>,
    target_dets: &GpuBuffer<u32>,
    blocks: &GpuBuffer<u32>,
    contributions: &GpuBuffer<u32>,
    y: &GpuBuffer<f64>,
    args: GroupedRankLaunch,
    a_first: bool,
) {
    if args.tiles == 0 {
        return;
    }
    let tile = checked_u32(GEMM_TILE_DIM);
    unsafe {
        grouped_rank_final_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(checked_u32(args.tiles), 1, 1),
            CubeDim::new_2d(tile, tile),
            s.array_arg(),
            f.array_arg(),
            intermediate_f.array_arg(),
            intermediate_s.array_arg(),
            alpha_components.array_arg(),
            beta_components.array_arg(),
            source_components.array_arg(),
            target_dets.array_arg(),
            blocks.array_arg(),
            contributions.array_arg(),
            y.array_arg(),
            args.blocks,
            args.factor_stride,
            args.intermediate_stride,
            args.lambda,
            args.worker,
            args.nworker,
            a_first,
        );
    }
}

/// Gather one dense rank block `D^{r,s}_{ij}=x[dets[i N_s+j]]` entirely on the device.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `topology`: Persistent dense determinant maps.
/// - `x`: Source determinant vector.
/// - `packed`: Reusable packed coefficient storage.
/// - `det_offset`: Block offset in the flattened determinant map.
/// - `len`: Number of actual determinants in the dense block.
/// # Returns
/// - `()`: Writes one alpha-major row-major coefficient block.
pub(crate) fn launch_gather_rank_block(
    context: &GpuContext,
    topology: &DeviceOneBodyData,
    x: &GpuBuffer<f64>,
    packed: &GpuBuffer<f64>,
    det_offset: usize,
    len: usize,
) {
    if len == 0 {
        return;
    }
    unsafe {
        gather_rank_block_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(launch_cubes(len), 1, 1),
            CubeDim::new_1d(CONTRACT_CUBE_DIM),
            topology.dense_rank_dets.array_arg(),
            x.array_arg(),
            packed.array_arg(),
            det_offset,
            len,
        );
    }
}

/// Launch a real-buffer zeroing kernel.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `out`: Device buffer to zero.
/// - `len`: Logical number of entries to zero.
/// # Returns
/// - `()`: Writes zero to `out[0..len]`.
pub(crate) fn launch_zero_f64(
    context: &GpuContext,
    out: &GpuBuffer<f64>,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let cubes = launch_cubes(len);
    unsafe {
        zero_f64_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(CONTRACT_CUBE_DIM),
            out.array_arg(),
            len,
        );
    }
}

/// Launch the alpha-first stage contraction.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `topology`: Device determinant topology.
/// - `sa`: Alpha overlap factors for this target panel.
/// - `fa`: Alpha Fock factors for this target panel.
/// - `x`: Source vector on the device.
/// - `tf`: Output first-stage Fock table.
/// - `ts`: Output first-stage overlap table.
/// - `args`: Launch dimensions and source-parent CSR base.
/// # Returns
/// - `()`: Writes `TF` and `TS` for the panel.
pub(crate) fn launch_a_first_stage(
    context: &GpuContext,
    topology: &DeviceOneBodyData,
    sa: &GpuBuffer<f64>,
    fa: &GpuBuffer<f64>,
    x: &GpuBuffer<f64>,
    tf: &GpuBuffer<f64>,
    ts: &GpuBuffer<f64>,
    args: AFirstStageLaunch,
) {
    let nwork = args
        .nrow
        .checked_mul(args.nsb)
        .expect("GPU alpha-first stage length overflow");
    if nwork == 0 {
        return;
    }
    unsafe {
        a_first_stage_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(launch_cubes(nwork), 1, 1),
            CubeDim::new_1d(CONTRACT_CUBE_DIM),
            sa.array_arg(),
            fa.array_arg(),
            x.array_arg(),
            topology.by_beta_offsets.array_arg(),
            topology.by_beta_det.array_arg(),
            topology.by_beta_alpha.array_arg(),
            tf.array_arg(),
            ts.array_arg(),
            args.nrow,
            args.nsb,
            args.nsa,
            args.csr_base,
            args.target_component_base,
            args.worker,
            args.nworker,
        );
    }
}

/// Launch the alpha-first final contraction.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `topology`: Device determinant topology.
/// - `sb`: Full beta overlap factors.
/// - `fb`: Full beta Fock factors.
/// - `tf`: First-stage Fock table.
/// - `ts`: First-stage overlap table.
/// - `y`: Output vector to accumulate.
/// - `args`: Target-parent entry and panel dimensions.
/// # Returns
/// - `()`: Accumulates determinant-space output for the panel.
pub(crate) fn launch_a_first_final(
    context: &GpuContext,
    topology: &DeviceOneBodyData,
    sb: &GpuBuffer<f64>,
    fb: &GpuBuffer<f64>,
    tf: &GpuBuffer<f64>,
    ts: &GpuBuffer<f64>,
    y: &GpuBuffer<f64>,
    args: AFirstFinalLaunch,
) {
    if args.nentry == 0 {
        return;
    }
    unsafe {
        a_first_final_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(launch_cubes(args.nentry), 1, 1),
            CubeDim::new_1d(CONTRACT_CUBE_DIM),
            sb.array_arg(),
            fb.array_arg(),
            topology.entry_det.array_arg(),
            topology.entry_a.array_arg(),
            topology.entry_b.array_arg(),
            tf.array_arg(),
            ts.array_arg(),
            y.array_arg(),
            args.lambda,
            args.entry_base,
            args.nentry,
            args.nsb,
            args.target_alpha_component_base,
            args.target_alpha_component_end,
            args.target_beta_component_base,
            args.target_beta_component_end,
            args.worker,
            args.nworker,
        );
    }
}

/// Launch the beta-first stage contraction.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `topology`: Device determinant topology.
/// - `sb`: Beta overlap factors for this target panel.
/// - `fb`: Beta Fock factors for this target panel.
/// - `x`: Source vector on the device.
/// - `uf`: Output first-stage Fock table.
/// - `us`: Output first-stage overlap table.
/// - `args`: Launch dimensions and source-parent CSR base.
/// # Returns
/// - `()`: Writes `UF` and `US` for the panel.
pub(crate) fn launch_b_first_stage(
    context: &GpuContext,
    topology: &DeviceOneBodyData,
    sb: &GpuBuffer<f64>,
    fb: &GpuBuffer<f64>,
    x: &GpuBuffer<f64>,
    uf: &GpuBuffer<f64>,
    us: &GpuBuffer<f64>,
    args: BFirstStageLaunch,
) {
    let nwork = args
        .nsa
        .checked_mul(args.nrow)
        .expect("GPU beta-first stage length overflow");
    if nwork == 0 {
        return;
    }
    unsafe {
        b_first_stage_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(launch_cubes(nwork), 1, 1),
            CubeDim::new_1d(CONTRACT_CUBE_DIM),
            sb.array_arg(),
            fb.array_arg(),
            x.array_arg(),
            topology.by_alpha_offsets.array_arg(),
            topology.by_alpha_det.array_arg(),
            topology.by_alpha_beta.array_arg(),
            uf.array_arg(),
            us.array_arg(),
            args.nrow,
            args.nsa,
            args.nsb,
            args.csr_base,
            args.target_component_base,
            args.worker,
            args.nworker,
        );
    }
}

/// Launch the beta-first final contraction.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `topology`: Device determinant topology.
/// - `sa`: Full alpha overlap factors.
/// - `fa`: Full alpha Fock factors.
/// - `uf`: First-stage Fock table.
/// - `us`: First-stage overlap table.
/// - `y`: Output vector to accumulate.
/// - `args`: Target-parent entry and panel dimensions.
/// # Returns
/// - `()`: Accumulates determinant-space output for the panel.
pub(crate) fn launch_b_first_final(
    context: &GpuContext,
    topology: &DeviceOneBodyData,
    sa: &GpuBuffer<f64>,
    fa: &GpuBuffer<f64>,
    uf: &GpuBuffer<f64>,
    us: &GpuBuffer<f64>,
    y: &GpuBuffer<f64>,
    args: BFirstFinalLaunch,
) {
    if args.nentry == 0 {
        return;
    }
    unsafe {
        b_first_final_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(launch_cubes(args.nentry), 1, 1),
            CubeDim::new_1d(CONTRACT_CUBE_DIM),
            sa.array_arg(),
            fa.array_arg(),
            topology.entry_det.array_arg(),
            topology.entry_a.array_arg(),
            topology.entry_b.array_arg(),
            uf.array_arg(),
            us.array_arg(),
            y.array_arg(),
            args.lambda,
            args.entry_base,
            args.nentry,
            args.nsa,
            args.nrow,
            args.target_alpha_component_base,
            args.target_alpha_component_end,
            args.target_beta_component_base,
            args.target_beta_component_end,
            args.worker,
            args.nworker,
        );
    }
}

/// Alpha-first stage launch dimensions.
#[derive(Clone, Copy)]
pub(crate) struct AFirstStageLaunch {
    /// Alpha panel row count.
    pub(crate) nrow: usize,
    /// Source beta component count.
    pub(crate) nsb: usize,
    /// Source alpha component count.
    pub(crate) nsa: usize,
    /// Source-parent CSR base in `by_beta_offsets`.
    pub(crate) csr_base: usize,
    /// First target alpha component represented by panel row zero.
    pub(crate) target_component_base: usize,
    /// MPI worker id.
    pub(crate) worker: usize,
    /// MPI worker count.
    pub(crate) nworker: usize,
}

/// Alpha-first final contraction dimensions.
#[derive(Clone, Copy)]
pub(crate) struct AFirstFinalLaunch {
    /// Target-parent entry base.
    pub(crate) entry_base: usize,
    /// Number of target-parent entries.
    pub(crate) nentry: usize,
    /// Source beta component count.
    pub(crate) nsb: usize,
    /// First target alpha component represented by the alpha panel.
    pub(crate) target_alpha_component_base: usize,
    /// One-past-last target alpha component represented by the alpha panel.
    pub(crate) target_alpha_component_end: usize,
    /// First target beta component represented by the beta panel.
    pub(crate) target_beta_component_base: usize,
    /// One-past-last target beta component represented by the beta panel.
    pub(crate) target_beta_component_end: usize,
    /// Overlap shift.
    pub(crate) lambda: f64,
    /// MPI worker id.
    pub(crate) worker: usize,
    /// MPI worker count.
    pub(crate) nworker: usize,
}

/// Beta-first stage launch dimensions.
#[derive(Clone, Copy)]
pub(crate) struct BFirstStageLaunch {
    /// Beta panel row count.
    pub(crate) nrow: usize,
    /// Source alpha component count.
    pub(crate) nsa: usize,
    /// Source beta component count.
    pub(crate) nsb: usize,
    /// Source-parent CSR base in `by_alpha_offsets`.
    pub(crate) csr_base: usize,
    /// First target beta component represented by panel row zero.
    pub(crate) target_component_base: usize,
    /// MPI worker id.
    pub(crate) worker: usize,
    /// MPI worker count.
    pub(crate) nworker: usize,
}

/// Beta-first final contraction dimensions.
#[derive(Clone, Copy)]
pub(crate) struct BFirstFinalLaunch {
    /// Target-parent entry base.
    pub(crate) entry_base: usize,
    /// Number of target-parent entries.
    pub(crate) nentry: usize,
    /// Source alpha component count.
    pub(crate) nsa: usize,
    /// Beta intermediate-panel row count.
    pub(crate) nrow: usize,
    /// First target alpha component represented by the alpha panel.
    pub(crate) target_alpha_component_base: usize,
    /// One-past-last target alpha component represented by the alpha panel.
    pub(crate) target_alpha_component_end: usize,
    /// First target beta component represented by the beta panel.
    pub(crate) target_beta_component_base: usize,
    /// One-past-last target beta component represented by the beta panel.
    pub(crate) target_beta_component_end: usize,
    /// Overlap shift.
    pub(crate) lambda: f64,
    /// MPI worker id.
    pub(crate) worker: usize,
    /// MPI worker count.
    pub(crate) nworker: usize,
}

/// Convert a logical work length into CubeCL cube count.
/// # Arguments:
/// - `len`: Work-item count.
/// # Returns
/// - `u32`: Cube count.
fn launch_cubes(len: usize) -> u32 {
    checked_u32(len.div_ceil(CONTRACT_CUBE_DIM as usize))
}

/// Convert host metadata to device-width `u32`.
/// # Arguments:
/// - `value`: Host value.
/// # Returns
/// - `u32`: Checked device-width value.
fn checked_u32(value: usize) -> u32 {
    u32::try_from(value).expect("GPU contraction launch dimension exceeds u32")
}

/// Compute every grouped first-stage rank block in one heterogeneous tile grid.
/// Each tile internally sums the populated inner-rank contributions, so no output atomics are
/// required.
/// # Arguments:
/// - Primitive factors, packed coefficients, rank maps, descriptors and output arrays.
/// - `nblocks`: Number of grouped output descriptors.
/// - `factor_stride`: Full parent-local source-spin factor stride.
/// - `intermediate_stride`: Full parent-local other-spin intermediate stride.
/// - `worker`: MPI worker id.
/// - `nworker`: MPI worker count.
/// - `a_first`: Alpha-first rather than beta-first contraction orientation.
/// # Returns
/// - `()`: Writes one complete grouped intermediate tile.
#[cube(launch_unchecked)]
fn grouped_rank_stage_kernel(
    s: &Array<f64>,
    f: &Array<f64>,
    d: &Array<f64>,
    row_components: &Array<u32>,
    col_components: &Array<u32>,
    source_components: &Array<u32>,
    blocks: &Array<u32>,
    contributions: &Array<u32>,
    out_f: &mut Array<f64>,
    out_s: &mut Array<f64>,
    nblocks: usize,
    factor_stride: usize,
    intermediate_stride: usize,
    worker: usize,
    nworker: usize,
    #[comptime] a_first: bool,
) {
    let flat_tile = usize::cast_from(CUBE_POS_X);
    let mut block_index = 0usize;
    for candidate in 0usize..nblocks {
        let base = candidate * GROUPED_STAGE_BLOCK_FIELDS;
        let tile_begin = usize::cast_from(blocks[base]);
        let tile_end = usize::cast_from(blocks[base + 1usize]);
        if flat_tile >= tile_begin && flat_tile < tile_end {
            block_index = candidate;
        }
    }

    let base = block_index * GROUPED_STAGE_BLOCK_FIELDS;
    let tile_begin = usize::cast_from(blocks[base]);
    let m = usize::cast_from(blocks[base + 2usize]);
    let n = usize::cast_from(blocks[base + 3usize]);
    let row_map_offset = usize::cast_from(blocks[base + 4usize]);
    let col_map_offset = usize::cast_from(blocks[base + 5usize]);
    let contribution_begin = usize::cast_from(blocks[base + 6usize]);
    let contribution_end = usize::cast_from(blocks[base + 7usize]);
    let tiles_n = n.div_ceil(GEMM_TILE_DIM);
    let local_tile = flat_tile - tile_begin;
    let tile_row = local_tile / tiles_n;
    let tile_col = local_tile - tile_row * tiles_n;
    let ux = usize::cast_from(UNIT_POS_X);
    let uy = usize::cast_from(UNIT_POS_Y);
    let row = tile_row * GEMM_TILE_DIM + uy;
    let col = tile_col * GEMM_TILE_DIM + ux;
    let lane = uy * GEMM_TILE_DIM + ux;
    let mut left_s = SharedMemory::<f64>::new(GEMM_TILE_DIM * GEMM_TILE_DIM);
    let mut left_f = SharedMemory::<f64>::new(GEMM_TILE_DIM * GEMM_TILE_DIM);
    let mut right = SharedMemory::<f64>::new(GEMM_TILE_DIM * GEMM_TILE_DIM);
    let mut value_f = 0.0;
    let mut value_s = 0.0;

    for contribution in contribution_begin..contribution_end {
        let contribution_base = contribution * GROUPED_STAGE_CONTRIBUTION_FIELDS;
        let k = usize::cast_from(contributions[contribution_base]);
        let source_map_offset = usize::cast_from(contributions[contribution_base + 1usize]);
        let d_offset = usize::cast_from(contributions[contribution_base + 2usize]);
        let d_stride = usize::cast_from(contributions[contribution_base + 3usize]);
        let ntiles = k.div_ceil(GEMM_TILE_DIM);
        for tile in 0usize..ntiles {
            let tile_start = tile * GEMM_TILE_DIM;
            let tile_len = (k - tile_start).min(GEMM_TILE_DIM);
            let reduced_col = tile_start + ux;
            let reduced_row = tile_start + uy;
            if comptime!(a_first) {
                if row < m && reduced_col < k {
                    let target = usize::cast_from(row_components[row_map_offset + row]);
                    let source =
                        usize::cast_from(source_components[source_map_offset + reduced_col]);
                    let factor_index = target * factor_stride + source;
                    left_s[lane] = s[factor_index];
                    left_f[lane] = f[factor_index];
                } else {
                    left_s[lane] = 0.0;
                    left_f[lane] = 0.0;
                }
                if reduced_row < k && col < n {
                    right[lane] = d[d_offset + reduced_row * d_stride + col];
                } else {
                    right[lane] = 0.0;
                }
            } else {
                if row < m && reduced_col < k {
                    left_s[lane] = d[d_offset + row * d_stride + reduced_col];
                    left_f[lane] = left_s[lane];
                } else {
                    left_s[lane] = 0.0;
                    left_f[lane] = 0.0;
                }
                let load_col = tile_col * GEMM_TILE_DIM + uy;
                if reduced_col < k && load_col < n {
                    let target = usize::cast_from(col_components[col_map_offset + load_col]);
                    let source =
                        usize::cast_from(source_components[source_map_offset + reduced_col]);
                    let factor_index = target * factor_stride + source;
                    let transposed_lane = ux * GEMM_TILE_DIM + uy;
                    right[transposed_lane] = s[factor_index];
                    left_f[transposed_lane] = f[factor_index];
                } else {
                    let transposed_lane = ux * GEMM_TILE_DIM + uy;
                    right[transposed_lane] = 0.0;
                    left_f[transposed_lane] = 0.0;
                }
            }
            sync_cube();
            if row < m && col < n {
                for q in 0usize..tile_len {
                    if comptime!(a_first) {
                        value_f += left_f[uy * GEMM_TILE_DIM + q] * right[q * GEMM_TILE_DIM + ux];
                        value_s += left_s[uy * GEMM_TILE_DIM + q] * right[q * GEMM_TILE_DIM + ux];
                    } else {
                        let coefficient = left_s[uy * GEMM_TILE_DIM + q];
                        value_f += coefficient * left_f[q * GEMM_TILE_DIM + ux];
                        value_s += coefficient * right[q * GEMM_TILE_DIM + ux];
                    }
                }
            }
            sync_cube();
        }
    }

    if row < m && col < n {
        let row_component = usize::cast_from(row_components[row_map_offset + row]);
        let col_component = usize::cast_from(col_components[col_map_offset + col]);
        let owner = if comptime!(a_first) {
            row_component
        } else {
            col_component
        };
        if owner % nworker == worker {
            let output = row_component * intermediate_stride + col_component;
            out_f[output] = value_f;
            out_s[output] = value_s;
        }
    }
}

/// Compute every grouped final target rank block in one heterogeneous tile grid.
/// Each tile internally sums the complete contributing source-spin rank set and scatters once.
/// # Arguments:
/// - Primitive factors, intermediates, rank maps, determinant maps, descriptors and output array.
/// - `nblocks`: Number of actual target rank-block descriptors.
/// - `factor_stride`: Full parent-local source-spin factor stride.
/// - `intermediate_stride`: Full parent-local intermediate row stride.
/// - `lambda`: Overlap shift.
/// - `worker`: MPI worker id.
/// - `nworker`: MPI worker count.
/// - `a_first`: Alpha-first rather than beta-first contraction orientation.
/// # Returns
/// - `()`: Accumulates one complete target determinant tile.
#[cube(launch_unchecked)]
fn grouped_rank_final_kernel(
    s: &Array<f64>,
    f: &Array<f64>,
    intermediate_f: &Array<f64>,
    intermediate_s: &Array<f64>,
    alpha_components: &Array<u32>,
    beta_components: &Array<u32>,
    source_components: &Array<u32>,
    target_dets: &Array<u32>,
    blocks: &Array<u32>,
    contributions: &Array<u32>,
    y: &mut Array<f64>,
    nblocks: usize,
    factor_stride: usize,
    intermediate_stride: usize,
    lambda: f64,
    worker: usize,
    nworker: usize,
    #[comptime] a_first: bool,
) {
    let flat_tile = usize::cast_from(CUBE_POS_X);
    let mut block_index = 0usize;
    for candidate in 0usize..nblocks {
        let base = candidate * GROUPED_FINAL_BLOCK_FIELDS;
        let tile_begin = usize::cast_from(blocks[base]);
        let tile_end = usize::cast_from(blocks[base + 1usize]);
        if flat_tile >= tile_begin && flat_tile < tile_end {
            block_index = candidate;
        }
    }

    let base = block_index * GROUPED_FINAL_BLOCK_FIELDS;
    let tile_begin = usize::cast_from(blocks[base]);
    let m = usize::cast_from(blocks[base + 2usize]);
    let n = usize::cast_from(blocks[base + 3usize]);
    let alpha_map_offset = usize::cast_from(blocks[base + 4usize]);
    let beta_map_offset = usize::cast_from(blocks[base + 5usize]);
    let det_offset = usize::cast_from(blocks[base + 6usize]);
    let contribution_begin = usize::cast_from(blocks[base + 7usize]);
    let contribution_end = usize::cast_from(blocks[base + 8usize]);
    let tiles_n = n.div_ceil(GEMM_TILE_DIM);
    let local_tile = flat_tile - tile_begin;
    let tile_row = local_tile / tiles_n;
    let tile_col = local_tile - tile_row * tiles_n;
    let ux = usize::cast_from(UNIT_POS_X);
    let uy = usize::cast_from(UNIT_POS_Y);
    let row = tile_row * GEMM_TILE_DIM + uy;
    let col = tile_col * GEMM_TILE_DIM + ux;
    let lane = uy * GEMM_TILE_DIM + ux;
    let mut left_f = SharedMemory::<f64>::new(GEMM_TILE_DIM * GEMM_TILE_DIM);
    let mut left_s = SharedMemory::<f64>::new(GEMM_TILE_DIM * GEMM_TILE_DIM);
    let mut right_f = SharedMemory::<f64>::new(GEMM_TILE_DIM * GEMM_TILE_DIM);
    let mut right_s = SharedMemory::<f64>::new(GEMM_TILE_DIM * GEMM_TILE_DIM);
    let mut value = 0.0;

    for contribution in contribution_begin..contribution_end {
        let contribution_base = contribution * GROUPED_FINAL_CONTRIBUTION_FIELDS;
        let k = usize::cast_from(contributions[contribution_base]);
        let source_map_offset = usize::cast_from(contributions[contribution_base + 1usize]);
        let ntiles = k.div_ceil(GEMM_TILE_DIM);
        for tile in 0usize..ntiles {
            let tile_start = tile * GEMM_TILE_DIM;
            let tile_len = (k - tile_start).min(GEMM_TILE_DIM);
            let reduced_col = tile_start + ux;
            let reduced_row = tile_start + uy;
            if row < m && reduced_col < k {
                let alpha = usize::cast_from(alpha_components[alpha_map_offset + row]);
                let source = usize::cast_from(source_components[source_map_offset + reduced_col]);
                let index = alpha
                    * if comptime!(a_first) {
                        intermediate_stride
                    } else {
                        factor_stride
                    }
                    + source;
                if comptime!(a_first) {
                    left_f[lane] = intermediate_f[index];
                    left_s[lane] = intermediate_s[index];
                } else {
                    left_f[lane] = f[index];
                    left_s[lane] = s[index];
                }
            } else {
                left_f[lane] = 0.0;
                left_s[lane] = 0.0;
            }
            if comptime!(a_first) {
                let load_col = tile_col * GEMM_TILE_DIM + uy;
                if reduced_col < k && load_col < n {
                    let beta = usize::cast_from(beta_components[beta_map_offset + load_col]);
                    let source =
                        usize::cast_from(source_components[source_map_offset + reduced_col]);
                    let index = beta * factor_stride + source;
                    let transposed_lane = ux * GEMM_TILE_DIM + uy;
                    right_f[transposed_lane] = f[index];
                    right_s[transposed_lane] = s[index];
                } else {
                    let transposed_lane = ux * GEMM_TILE_DIM + uy;
                    right_f[transposed_lane] = 0.0;
                    right_s[transposed_lane] = 0.0;
                }
            } else if reduced_row < k && col < n {
                let beta = usize::cast_from(beta_components[beta_map_offset + col]);
                let source = usize::cast_from(source_components[source_map_offset + reduced_row]);
                let index = source * intermediate_stride + beta;
                right_f[lane] = intermediate_f[index];
                right_s[lane] = intermediate_s[index];
            } else {
                right_f[lane] = 0.0;
                right_s[lane] = 0.0;
            }
            sync_cube();
            if row < m && col < n {
                for q in 0usize..tile_len {
                    let lf = left_f[uy * GEMM_TILE_DIM + q];
                    let ls = left_s[uy * GEMM_TILE_DIM + q];
                    let rf = right_f[q * GEMM_TILE_DIM + ux];
                    let rs = right_s[q * GEMM_TILE_DIM + ux];
                    if comptime!(a_first) {
                        value += lf * rs + ls * (rf + lambda * rs);
                    } else {
                        value += ls * rf + (lf + lambda * ls) * rs;
                    }
                }
            }
            sync_cube();
        }
    }

    if row < m && col < n {
        let alpha = usize::cast_from(alpha_components[alpha_map_offset + row]);
        let beta = usize::cast_from(beta_components[beta_map_offset + col]);
        let owner = if comptime!(a_first) { alpha } else { beta };
        if owner % nworker == worker {
            let det = usize::cast_from(target_dets[det_offset + row * n + col]);
            y[det] = y[det] + value;
        }
    }
}

/// Gather one packed dense coefficient rank block.
/// # Arguments:
/// - `dets`: Persistent global determinant map.
/// - `x`: Global source determinant vector.
/// - `packed`: Packed output block.
/// - `det_offset`: Offset into `dets`.
/// - `len`: Block entry count.
/// # Returns
/// - `()`: Writes `packed[i]=x[dets[det_offset+i]]`.
#[cube(launch_unchecked)]
fn gather_rank_block_kernel(
    dets: &Array<u32>,
    x: &Array<f64>,
    packed: &mut Array<f64>,
    det_offset: usize,
    len: usize,
) {
    if ABSOLUTE_POS < len {
        let det = usize::cast_from(dets[det_offset + ABSOLUTE_POS]);
        packed[ABSOLUTE_POS] = x[det];
    }
}

/// Zero a real device buffer before accumulation.
/// # Arguments:
/// - `out`: Device buffer to zero.
/// - `len`: Number of entries.
/// # Returns
/// - `()`: Writes zero to every entry.
#[cube(launch_unchecked)]
pub(crate) fn zero_f64_kernel(
    out: &mut Array<f64>,
    len: usize,
) {
    if ABSOLUTE_POS < len {
        out[ABSOLUTE_POS] = 0.0;
    }
}

/// Alpha-first stage kernel:
/// `T^F_{\bar a,b} = \sum_a F^\alpha_{\bar a,a}D_{a,b}` and
/// `T^S_{\bar a,b} = \sum_a S^\alpha_{\bar a,a}D_{a,b}`.
/// # Arguments:
/// - `sa`: Row-major alpha overlap factors.
/// - `fa`: Row-major alpha Fock factors.
/// - `x`: Source determinant vector.
/// - `by_beta_offsets`: CSR offsets keyed by source beta component.
/// - `by_beta_det`: CSR determinant IDs.
/// - `by_beta_alpha`: CSR source alpha component IDs.
/// - `tf`: Output Fock first-stage table.
/// - `ts`: Output overlap first-stage table.
/// - `nta`: Target alpha component count.
/// - `nsb`: Source beta component count.
/// - `nsa`: Source alpha component count.
/// - `csr_base`: Source-parent CSR base in `by_beta_offsets`.
/// - `target_component_base`: First target alpha component in the panel.
/// - `worker`: MPI worker id.
/// - `nworker`: MPI worker count.
/// # Returns
/// - `()`: Writes `TF` and `TS`.
#[cube(launch_unchecked)]
pub(crate) fn a_first_stage_kernel(
    sa: &Array<f64>,
    fa: &Array<f64>,
    x: &Array<f64>,
    by_beta_offsets: &Array<u32>,
    by_beta_det: &Array<u32>,
    by_beta_alpha: &Array<u32>,
    tf: &mut Array<f64>,
    ts: &mut Array<f64>,
    nrow: usize,
    nsb: usize,
    nsa: usize,
    csr_base: usize,
    target_component_base: usize,
    worker: usize,
    nworker: usize,
) {
    if ABSOLUTE_POS >= nrow * nsb {
        terminate!();
    }
    let row = ABSOLUTE_POS / nsb;
    let abar = target_component_base + row;
    let b = ABSOLUTE_POS - row * nsb;
    if abar % nworker != worker {
        terminate!();
    }
    let mut vf = 0.0;
    let mut vs = 0.0;
    let start = usize::cast_from(by_beta_offsets[csr_base + b]);
    let end = usize::cast_from(by_beta_offsets[csr_base + b + 1usize]);
    for p in start..end {
        let det = usize::cast_from(by_beta_det[p]);
        let a = usize::cast_from(by_beta_alpha[p]);
        let xe = x[det];
        vf += fa[row * nsa + a] * xe;
        vs += sa[row * nsa + a] * xe;
    }
    tf[ABSOLUTE_POS] = vf;
    ts[ABSOLUTE_POS] = vs;
}

/// Alpha-first final contraction for one alpha-beta target panel:
/// `Y_{\bar a\bar b} += \sum_b [T^F_{\bar a b}S^\beta_{\bar b b}
/// + T^S_{\bar a b}(F^\beta_{\bar b b}+\lambda S^\beta_{\bar b b})]`.
/// # Arguments:
/// - `sb`: Row-major beta overlap factor panel.
/// - `fb`: Row-major beta Fock factor panel.
/// - `target_entry_det`: Target determinant IDs.
/// - `target_entry_a`: Target alpha components.
/// - `target_entry_b`: Target beta components.
/// - `tf`: Alpha-first Fock intermediate panel.
/// - `ts`: Alpha-first overlap intermediate panel.
/// - `y`: Output determinant vector.
/// - `lambda`: Overlap shift.
/// - `entry_base`: Target-parent entry base.
/// - `nentry`: Number of target-parent entries.
/// - `nsb`: Source beta component count.
/// - `target_alpha_component_base`: First target alpha component in the alpha panel.
/// - `target_alpha_component_end`: One-past-last target alpha component in the alpha panel.
/// - `target_beta_component_base`: First target beta component in the beta panel.
/// - `target_beta_component_end`: One-past-last target beta component in the beta panel.
/// - `worker`: MPI worker id.
/// - `nworker`: MPI worker count.
/// # Returns
/// - `()`: Accumulates this two-dimensional target panel into `y`.
#[cube(launch_unchecked)]
pub(crate) fn a_first_final_kernel(
    sb: &Array<f64>,
    fb: &Array<f64>,
    target_entry_det: &Array<u32>,
    target_entry_a: &Array<u32>,
    target_entry_b: &Array<u32>,
    tf: &Array<f64>,
    ts: &Array<f64>,
    y: &mut Array<f64>,
    lambda: f64,
    entry_base: usize,
    nentry: usize,
    nsb: usize,
    target_alpha_component_base: usize,
    target_alpha_component_end: usize,
    target_beta_component_base: usize,
    target_beta_component_end: usize,
    worker: usize,
    nworker: usize,
) {
    if ABSOLUTE_POS < nentry {
        let entry = entry_base + ABSOLUTE_POS;
        let abar = usize::cast_from(target_entry_a[entry]);
        let bbar = usize::cast_from(target_entry_b[entry]);

        if abar >= target_alpha_component_base
            && abar < target_alpha_component_end
            && bbar >= target_beta_component_base
            && bbar < target_beta_component_end
            && abar % nworker == worker
        {
            let arow = abar - target_alpha_component_base;
            let brow = bbar - target_beta_component_base;
            let mut value = 0.0;

            for b in 0usize..nsb {
                let beta = brow * nsb + b;
                value += tf[arow * nsb + b] * sb[beta]
                    + ts[arow * nsb + b] * (fb[beta] + lambda * sb[beta]);
            }

            let det = usize::cast_from(target_entry_det[entry]);
            y[det] = y[det] + value;
        }
    }
}

/// Beta-first stage kernel:
/// `U^F_{a,\bar b} = \sum_b D_{a,b}F^\beta_{\bar b,b}` and
/// `U^S_{a,\bar b} = \sum_b D_{a,b}S^\beta_{\bar b,b}`.
/// # Arguments:
/// - `sb`: Row-major beta overlap factors.
/// - `fb`: Row-major beta Fock factors.
/// - `x`: Source determinant vector.
/// - `by_alpha_offsets`: CSR offsets keyed by source alpha component.
/// - `by_alpha_det`: CSR determinant IDs.
/// - `by_alpha_beta`: CSR source beta component IDs.
/// - `uf`: Output Fock first-stage table.
/// - `us`: Output overlap first-stage table.
/// - `ntb`: Target beta component count.
/// - `nsa`: Source alpha component count.
/// - `nsb`: Source beta component count.
/// - `csr_base`: Source-parent CSR base in `by_alpha_offsets`.
/// - `target_component_base`: First target beta component in the panel.
/// - `worker`: MPI worker id.
/// - `nworker`: MPI worker count.
/// # Returns
/// - `()`: Writes `UF` and `US`.
#[cube(launch_unchecked)]
pub(crate) fn b_first_stage_kernel(
    sb: &Array<f64>,
    fb: &Array<f64>,
    x: &Array<f64>,
    by_alpha_offsets: &Array<u32>,
    by_alpha_det: &Array<u32>,
    by_alpha_beta: &Array<u32>,
    uf: &mut Array<f64>,
    us: &mut Array<f64>,
    nrow: usize,
    nsa: usize,
    nsb: usize,
    csr_base: usize,
    target_component_base: usize,
    worker: usize,
    nworker: usize,
) {
    if ABSOLUTE_POS >= nsa * nrow {
        terminate!();
    }
    let a = ABSOLUTE_POS / nrow;
    let row = ABSOLUTE_POS - a * nrow;
    let bbar = target_component_base + row;
    if bbar % nworker != worker {
        terminate!();
    }
    let mut vf = 0.0;
    let mut vs = 0.0;
    let start = usize::cast_from(by_alpha_offsets[csr_base + a]);
    let end = usize::cast_from(by_alpha_offsets[csr_base + a + 1usize]);
    for p in start..end {
        let det = usize::cast_from(by_alpha_det[p]);
        let b = usize::cast_from(by_alpha_beta[p]);
        let xe = x[det];
        vf += xe * fb[row * nsb + b];
        vs += xe * sb[row * nsb + b];
    }
    uf[ABSOLUTE_POS] = vf;
    us[ABSOLUTE_POS] = vs;
}

/// Beta-first final contraction for one alpha-beta target panel:
/// `Y_{\bar a\bar b} += \sum_a [S^\alpha_{\bar a a}U^F_{a\bar b}
/// + (F^\alpha_{\bar a a}+\lambda S^\alpha_{\bar a a})U^S_{a\bar b}]`.
/// # Arguments:
/// - `sa`: Row-major alpha overlap factor panel.
/// - `fa`: Row-major alpha Fock factor panel.
/// - `target_entry_det`: Target determinant IDs.
/// - `target_entry_a`: Target alpha components.
/// - `target_entry_b`: Target beta components.
/// - `uf`: Beta-first Fock intermediate panel.
/// - `us`: Beta-first overlap intermediate panel.
/// - `y`: Output determinant vector.
/// - `lambda`: Overlap shift.
/// - `entry_base`: Target-parent entry base.
/// - `nentry`: Number of target-parent entries.
/// - `nsa`: Source alpha component count.
/// - `nrow`: Target beta panel row count.
/// - `target_alpha_component_base`: First target alpha component in the alpha panel.
/// - `target_alpha_component_end`: One-past-last target alpha component in the alpha panel.
/// - `target_beta_component_base`: First target beta component in the beta panel.
/// - `target_beta_component_end`: One-past-last target beta component in the beta panel.
/// - `worker`: MPI worker id.
/// - `nworker`: MPI worker count.
/// # Returns
/// - `()`: Accumulates this two-dimensional target panel into `y`.
#[cube(launch_unchecked)]
pub(crate) fn b_first_final_kernel(
    sa: &Array<f64>,
    fa: &Array<f64>,
    target_entry_det: &Array<u32>,
    target_entry_a: &Array<u32>,
    target_entry_b: &Array<u32>,
    uf: &Array<f64>,
    us: &Array<f64>,
    y: &mut Array<f64>,
    lambda: f64,
    entry_base: usize,
    nentry: usize,
    nsa: usize,
    nrow: usize,
    target_alpha_component_base: usize,
    target_alpha_component_end: usize,
    target_beta_component_base: usize,
    target_beta_component_end: usize,
    worker: usize,
    nworker: usize,
) {
    if ABSOLUTE_POS < nentry {
        let entry = entry_base + ABSOLUTE_POS;
        let abar = usize::cast_from(target_entry_a[entry]);
        let bbar = usize::cast_from(target_entry_b[entry]);

        if abar >= target_alpha_component_base
            && abar < target_alpha_component_end
            && bbar >= target_beta_component_base
            && bbar < target_beta_component_end
            && bbar % nworker == worker
        {
            let arow = abar - target_alpha_component_base;
            let brow = bbar - target_beta_component_base;
            let mut value = 0.0;

            for a in 0usize..nsa {
                let alpha = arow * nsa + a;
                let intermediate = a * nrow + brow;
                value += sa[alpha] * uf[intermediate]
                    + (fa[alpha] + lambda * sa[alpha]) * us[intermediate];
            }

            let det = usize::cast_from(target_entry_det[entry]);
            y[det] = y[det] + value;
        }
    }
}
