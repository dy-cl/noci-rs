// noci/factorise/onebody/gpu/factors.rs
//! GPU factor generation for factorised one-body NOCI operator contractions.

// External crate imports.
use cubecl::prelude::*;

// Crate-root imports.
use crate::gpu::{GpuBuffer, GpuContext, GpuRuntime};
use crate::nonorthogonalwicks::gpu::eval::prepare::{build_d_gen, build_d_m0};
use crate::nonorthogonalwicks::gpu::{
    DeviceWicksShared, GpuSameSpinView, xw_overlap_f, xw_overlap_f_m0,
};

// Parent/sibling imports.
use super::contract::launch_zero_f64;
use super::data::{DeviceOneBodyData, GpuOneBodyData};

/// Build one transient same-spin factor panel containing `S` and `F`.
/// Rank-homogeneous launches retain compile-time excitation ranks while the host-known
/// zero-overlap count `m` removes exactly-zero rank blocks before they reach the device.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `wicks`: Device-resident nonorthogonal Wick intermediates.
/// - `topology`: Device-resident determinant topology.
/// - `host`: Host topology retained for rank-group offsets.
/// - `request`: Parent, spin and target-panel request.
/// - `out`: Device overlap and Fock factor-panel outputs.
/// # Returns
/// - `()`: Writes the requested transient factor panel.
pub(crate) fn build_spin_one_body_factors(
    context: &GpuContext,
    wicks: &DeviceWicksShared,
    topology: &DeviceOneBodyData,
    host: &GpuOneBodyData,
    request: FactorRequest,
    out: FactorOutput<'_>,
) {
    if request.target_component_end <= request.target_component_base
        || request.source_component_end <= request.source_component_base
    {
        return;
    }

    let max_target_rank = max_rank(host, request.alpha);
    let max_source_rank = max_target_rank;
    let pair = FactorPair {
        wslot: request.wslot,
        target_left: request.target_left,
        alpha: request.alpha,
        m: request.m,
    };

    if request.m > 1 {
        let nrow = request.target_component_end - request.target_component_base;
        let len = nrow
            .checked_mul(request.source_component_end - request.source_component_base)
            .expect("GPU factor panel length overflow");
        launch_zero_f64(context, out.s, len);
        launch_zero_f64(context, out.f, len);
    }

    for target_rank in 0..=max_target_rank {
        let target = rank_group(
            host,
            topology,
            request.target_parent,
            request.alpha,
            target_rank,
            request.target_component_base,
            request.target_component_end,
        );

        if target.len == 0 {
            continue;
        }

        for source_rank in 0..=max_source_rank {
            let source = rank_group(
                host,
                topology,
                request.source_parent,
                request.alpha,
                source_rank,
                request.source_component_base,
                request.source_component_end,
            );

            if source.len == 0 {
                continue;
            }

            let ranks = FactorRanks {
                target_rank,
                source_rank,
                l: target_rank + source_rank,
            };

            // An L-rank determinant contraction can absorb at most L zero-overlap
            // pairs in S and L + 1 in a one-body matrix element. Therefore both
            // S and F are exactly zero when m > L + 1.
            if pair.m > ranks.l + 1 {
                continue;
            }

            launch_spin_rank_block(context, wicks, topology, pair, target, source, out, ranks);
        }
    }
}

/// Build only `S_{aa}` and `F_{aa}` for one same-parent spin sector.
/// This reduces diagonal construction from a dense `n^2` factor build to `n` factors.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `wicks`: Device-resident nonorthogonal Wick intermediates.
/// - `topology`: Device-resident determinant topology.
/// - `host`: Host topology retained for rank-group offsets.
/// - `request`: Parent and spin request.
/// - `out`: Device diagonal factor outputs.
/// # Returns
/// - `()`: Writes diagonal same-spin overlap and Fock factors.
pub(crate) fn build_spin_one_body_diagonal_factors(
    context: &GpuContext,
    wicks: &DeviceWicksShared,
    topology: &DeviceOneBodyData,
    host: &GpuOneBodyData,
    request: DiagonalFactorRequest,
    out: DiagonalFactorOutput<'_>,
) {
    if request.ncomponent == 0 {
        return;
    }

    if request.m > 1 {
        launch_zero_f64(context, out.s, request.ncomponent);
        launch_zero_f64(context, out.f, request.ncomponent);
    }

    let max_rank = max_rank(host, request.alpha);
    let pair = FactorPair {
        wslot: request.wslot,
        target_left: true,
        alpha: request.alpha,
        m: request.m,
    };

    for rank in 0..=max_rank {
        let group = rank_group(
            host,
            topology,
            request.parent,
            request.alpha,
            rank,
            0,
            request.ncomponent,
        );

        if group.len == 0 {
            continue;
        }

        let l = 2 * rank;

        if pair.m > l + 1 {
            continue;
        }

        launch_spin_diagonal_rank_block(context, wicks, topology, pair, group, out, rank, l);
    }
}

/// Launch one homogeneous target/source excitation-rank factor block.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `wicks`: Device Wick buffers.
/// - `topology`: Device determinant topology.
/// - `pair`: Ordered reference-pair and spin metadata.
/// - `target`: Target rank group.
/// - `source`: Source rank group.
/// - `out`: Factor-panel output buffers.
/// - `ranks`: Compile-time target, source and total excitation ranks.
/// # Returns
/// - `()`: Launches one rank-homogeneous factor kernel.
fn launch_spin_rank_block(
    context: &GpuContext,
    wicks: &DeviceWicksShared,
    topology: &DeviceOneBodyData,
    pair: FactorPair,
    target: RankGroup<'_>,
    source: RankGroup<'_>,
    out: FactorOutput<'_>,
    ranks: FactorRanks,
) {
    let nfactors = target
        .len
        .checked_mul(source.len)
        .expect("GPU factor launch length overflow");

    if nfactors == 0 {
        return;
    }

    let cube_dim = 128u32;
    let cubes = checked_u32(nfactors.div_ceil(cube_dim as usize));
    let spin = spin_topology(topology, pair.alpha);

    unsafe {
        one_body_factor_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(cube_dim),
            wicks.slab.array_arg(),
            wicks.x_off.array_arg(),
            wicks.y_off.array_arg(),
            wicks.ff_off.array_arg(),
            wicks.phase.array_arg(),
            wicks.tilde_s_prod.array_arg(),
            wicks.f0f.array_arg(),
            wicks.nmo.array_arg(),
            wicks.nocc.array_arg(),
            spin.holes_offset.array_arg(),
            spin.parts_offset.array_arg(),
            spin.holes.array_arg(),
            spin.parts.array_arg(),
            spin.phase.array_arg(),
            target.rep_det.array_arg(),
            target.component.array_arg(),
            source.rep_det.array_arg(),
            source.component.array_arg(),
            out.s.array_arg(),
            out.f.array_arg(),
            pair.wslot,
            target.offset,
            target.len,
            source.offset,
            source.len,
            out.target_component_base,
            out.source_component_base,
            out.source_stride,
            pair.target_left,
            ranks.target_rank,
            ranks.source_rank,
            ranks.l,
            pair.m,
        );
    }
}

/// Launch one same-rank diagonal factor block.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `wicks`: Device Wick buffers.
/// - `topology`: Device determinant topology.
/// - `pair`: Same-parent reference-pair and spin metadata.
/// - `group`: Spin representatives of the requested excitation rank.
/// - `out`: Diagonal factor outputs.
/// - `rank`: Excitation rank on each side.
/// - `l`: Total contraction rank `2 rank`.
/// # Returns
/// - `()`: Launches one diagonal factor kernel.
fn launch_spin_diagonal_rank_block(
    context: &GpuContext,
    wicks: &DeviceWicksShared,
    topology: &DeviceOneBodyData,
    pair: FactorPair,
    group: RankGroup<'_>,
    out: DiagonalFactorOutput<'_>,
    rank: usize,
    l: usize,
) {
    if group.len == 0 {
        return;
    }

    let cube_dim = 128u32;
    let cubes = checked_u32(group.len.div_ceil(cube_dim as usize));
    let spin = spin_topology(topology, pair.alpha);

    unsafe {
        one_body_diagonal_factor_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(cube_dim),
            wicks.slab.array_arg(),
            wicks.x_off.array_arg(),
            wicks.y_off.array_arg(),
            wicks.ff_off.array_arg(),
            wicks.phase.array_arg(),
            wicks.tilde_s_prod.array_arg(),
            wicks.f0f.array_arg(),
            wicks.nmo.array_arg(),
            wicks.nocc.array_arg(),
            spin.holes_offset.array_arg(),
            spin.parts_offset.array_arg(),
            spin.holes.array_arg(),
            spin.parts.array_arg(),
            spin.phase.array_arg(),
            group.rep_det.array_arg(),
            group.component.array_arg(),
            out.s.array_arg(),
            out.f.array_arg(),
            pair.wslot,
            group.offset,
            group.len,
            rank,
            l,
            pair.m,
        );
    }
}

/// Parent, spin and component-panel request for one transient same-spin factor table.
#[derive(Clone, Copy)]
pub(crate) struct FactorRequest {
    /// Target parent `Q`.
    pub(crate) target_parent: usize,
    /// Source parent `P`.
    pub(crate) source_parent: usize,
    /// Flattened same-spin Wick slot.
    pub(crate) wslot: usize,
    /// Whether target representatives are the left determinants of the ordered Wick pair.
    pub(crate) target_left: bool,
    /// Whether the requested factors are alpha-spin factors.
    pub(crate) alpha: bool,
    /// Number of zero-overlap occupied-orbital pairs.
    pub(crate) m: usize,
    /// First target component represented by output row zero.
    pub(crate) target_component_base: usize,
    /// One-past-last target component represented by the panel.
    pub(crate) target_component_end: usize,
    /// First source component represented by output column zero.
    pub(crate) source_component_base: usize,
    /// One-past-last source component represented by the panel.
    pub(crate) source_component_end: usize,
}

/// Request for same-parent diagonal factor generation.
#[derive(Clone, Copy)]
pub(crate) struct DiagonalFactorRequest {
    /// Parent reference.
    pub(crate) parent: usize,
    /// Flattened same-spin Wick slot.
    pub(crate) wslot: usize,
    /// Whether the requested factors are alpha-spin factors.
    pub(crate) alpha: bool,
    /// Number of zero-overlap occupied-orbital pairs.
    pub(crate) m: usize,
    /// Number of parent-local spin components.
    pub(crate) ncomponent: usize,
}

/// Device factor-panel outputs.
#[derive(Clone, Copy)]
pub(crate) struct FactorOutput<'a> {
    /// Row-major overlap factor panel.
    pub(crate) s: &'a GpuBuffer<f64>,
    /// Row-major Fock factor panel.
    pub(crate) f: &'a GpuBuffer<f64>,
    /// First target component represented by output row zero.
    pub(crate) target_component_base: usize,
    /// First source component represented by output column zero.
    pub(crate) source_component_base: usize,
    /// Compact source-component row stride.
    pub(crate) source_stride: usize,
}

/// Device diagonal factor outputs.
#[derive(Clone, Copy)]
pub(crate) struct DiagonalFactorOutput<'a> {
    /// Diagonal overlap factors.
    pub(crate) s: &'a GpuBuffer<f64>,
    /// Diagonal Fock factors.
    pub(crate) f: &'a GpuBuffer<f64>,
}

/// Ordered reference-pair metadata shared by one factor launch.
#[derive(Clone, Copy)]
struct FactorPair {
    /// Flattened same-spin Wick slot.
    wslot: usize,
    /// Whether target representatives are left determinants.
    target_left: bool,
    /// Whether alpha-spin topology is used.
    alpha: bool,
    /// Number of zero-overlap occupied-orbital pairs.
    m: usize,
}

/// Rank-homogeneous representative group within one parent.
#[derive(Clone, Copy)]
struct RankGroup<'a> {
    /// Rank-sorted representative determinant IDs.
    rep_det: &'a GpuBuffer<u32>,
    /// Original parent-local component IDs.
    component: &'a GpuBuffer<u32>,
    /// First rank-group entry.
    offset: usize,
    /// Number of rank-group entries.
    len: usize,
}

/// Compile-time rank metadata for one factor kernel launch.
#[derive(Clone, Copy)]
struct FactorRanks {
    /// Target spin excitation rank.
    target_rank: usize,
    /// Source spin excitation rank.
    source_rank: usize,
    /// Total contraction determinant rank.
    l: usize,
}

/// Device topology for one selected spin sector.
#[derive(Clone, Copy)]
struct SpinTopology<'a> {
    /// Excitation-hole offsets keyed by determinant.
    holes_offset: &'a GpuBuffer<u32>,
    /// Excitation-particle offsets keyed by determinant.
    parts_offset: &'a GpuBuffer<u32>,
    /// Flattened excitation holes.
    holes: &'a GpuBuffer<u32>,
    /// Flattened excitation particles.
    parts: &'a GpuBuffer<u32>,
    /// External determinant excitation phase.
    phase: &'a GpuBuffer<f64>,
}

/// Select alpha or beta device excitation topology before launching the kernel.
/// # Arguments:
/// - `topology`: Complete device determinant topology.
/// - `alpha`: Whether to select alpha-spin topology.
/// # Returns
/// - `SpinTopology`: Device buffers required for the selected spin.
fn spin_topology(
    topology: &DeviceOneBodyData,
    alpha: bool,
) -> SpinTopology<'_> {
    if alpha {
        SpinTopology {
            holes_offset: &topology.alpha_holes_offset,
            parts_offset: &topology.alpha_parts_offset,
            holes: &topology.alpha_holes,
            parts: &topology.alpha_parts,
            phase: &topology.pha,
        }
    } else {
        SpinTopology {
            holes_offset: &topology.beta_holes_offset,
            parts_offset: &topology.beta_parts_offset,
            holes: &topology.beta_holes,
            parts: &topology.beta_parts,
            phase: &topology.phb,
        }
    }
}

/// Select one parent/rank representative group and restrict it to a target component panel.
/// # Arguments:
/// - `host`: Host topology containing rank-group offsets.
/// - `device`: Device topology containing rank-sorted representatives.
/// - `parent`: Parent reference index.
/// - `alpha`: Whether to select the alpha-spin grouping.
/// - `rank`: Spin excitation rank.
/// - `component_base`: First requested parent-local component.
/// - `component_end`: One-past-last requested parent-local component.
/// # Returns
/// - `RankGroup`: Device rank group restricted to the requested component interval.
fn rank_group<'a>(
    host: &GpuOneBodyData,
    device: &'a DeviceOneBodyData,
    parent: usize,
    alpha: bool,
    rank: usize,
    component_base: usize,
    component_end: usize,
) -> RankGroup<'a> {
    let max_rank = max_rank(host, alpha);

    let (offsets, components, rep_det, component) = if alpha {
        (
            &host.parent_alpha_rank_offsets,
            &host.alpha_rank_component,
            &device.alpha_rank_rep_det,
            &device.alpha_rank_component,
        )
    } else {
        (
            &host.parent_beta_rank_offsets,
            &host.beta_rank_component,
            &device.beta_rank_rep_det,
            &device.beta_rank_component,
        )
    };

    if rank > max_rank {
        return RankGroup {
            rep_det,
            component,
            offset: 0,
            len: 0,
        };
    }

    let stride = max_rank + 2;
    let base = parent
        .checked_mul(stride)
        .expect("GPU rank-group parent offset overflow");
    let group_start = offsets[base + rank];
    let group_end = offsets[base + rank + 1];
    let group_components = &components[group_start..group_end];
    let first = group_components.partition_point(|&comp| comp < component_base);
    let last = group_components.partition_point(|&comp| comp < component_end);

    RankGroup {
        rep_det,
        component,
        offset: group_start + first,
        len: last - first,
    }
}

/// Return the maximum decoded excitation rank for one spin sector.
/// # Arguments:
/// - `host`: Host determinant topology.
/// - `alpha`: Whether to return the alpha-spin maximum.
/// # Returns
/// - `usize`: Maximum decoded spin excitation rank.
fn max_rank(
    host: &GpuOneBodyData,
    alpha: bool,
) -> usize {
    if alpha {
        host.max_alpha_rank
    } else {
        host.max_beta_rank
    }
}

/// Construct contraction-determinant row and column labels directly from persistent excitation data.
/// This removes the previous thread-private `l_holes`, `l_parts`, `g_holes` and `g_parts` copies.
/// # Arguments:
/// - `w`: Same-spin Wick view.
/// - `holes_offset`: Excitation-hole offsets keyed by determinant.
/// - `parts_offset`: Excitation-particle offsets keyed by determinant.
/// - `holes`: Flattened excitation holes.
/// - `parts`: Flattened excitation particles.
/// - `ldet`: Left determinant index.
/// - `gdet`: Greater determinant index.
/// - `rows`: Output contraction-determinant row labels.
/// - `cols`: Output contraction-determinant column labels.
/// - `left_rank`: Compile-time left excitation rank.
/// - `greater_rank`: Compile-time greater excitation rank.
/// # Returns
/// - `()`: Writes the contraction-determinant labels.
#[cube]
fn fill_factor_indices(
    w: &GpuSameSpinView,
    holes_offset: &Array<u32>,
    parts_offset: &Array<u32>,
    holes: &Array<u32>,
    parts: &Array<u32>,
    ldet: usize,
    gdet: usize,
    rows: &mut Array<u32>,
    cols: &mut Array<u32>,
    #[comptime] left_rank: usize,
    #[comptime] greater_rank: usize,
) {
    let loh = usize::cast_from(holes_offset[ldet]);
    let lop = usize::cast_from(parts_offset[ldet]);
    let goh = usize::cast_from(holes_offset[gdet]);
    let gop = usize::cast_from(parts_offset[gdet]);
    let nvirt = w.nmo - w.nocc;

    for k in 0usize..left_rank {
        rows[k] = parts[lop + k] - w.nocc;
        cols[k] = holes[loh + k];
    }

    for k in 0usize..greater_rank {
        rows[left_rank + k] = nvirt + holes[goh + k];
        cols[left_rank + k] = parts[gop + k];
    }
}

/// Evaluate one same-spin overlap/Fock factor pair and write it to device output.
/// The `m = 0` compile-time path avoids `det1`, mixed-determinant and replacement-column scratch.
/// # Arguments:
/// - `slab`: Compact same-spin Wick tensor slab.
/// - `x_off`: Offsets to `X^(0)` and `X^(1)`.
/// - `y_off`: Offsets to `Y^(0)` and `Y^(1)`.
/// - `ff_off`: Offsets to transposed current-Fock intermediates.
/// - `phase`: Ordered-pair orbital-pairing phases.
/// - `tilde_s_prod`: Ordered-pair non-zero singular-value products.
/// - `f0f`: Ordered-pair scalar Fock intermediates.
/// - `nmo`: Ordered-pair molecular-orbital dimensions.
/// - `nocc`: Ordered-pair occupied-orbital dimensions.
/// - `holes_offset`: Excitation-hole offsets.
/// - `parts_offset`: Excitation-particle offsets.
/// - `holes`: Flattened excitation holes.
/// - `parts`: Flattened excitation particles.
/// - `excitation_phase`: External determinant excitation phases.
/// - `s_out`: Overlap factor output.
/// - `f_out`: Fock factor output.
/// - `wslot`: Flattened same-spin Wick slot.
/// - `ldet`: Left determinant index.
/// - `gdet`: Greater determinant index.
/// - `out`: Output factor index.
/// - `left_rank`: Compile-time left excitation rank.
/// - `greater_rank`: Compile-time greater excitation rank.
/// - `l`: Compile-time total contraction rank.
/// - `m`: Compile-time zero-overlap count.
/// # Returns
/// - `()`: Writes one overlap and one Fock factor.
#[cube]
fn evaluate_factor_entry(
    slab: &Array<f64>,
    x_off: &Array<u32>,
    y_off: &Array<u32>,
    ff_off: &Array<u32>,
    phase: &Array<f64>,
    tilde_s_prod: &Array<f64>,
    f0f: &Array<f64>,
    nmo: &Array<u32>,
    nocc: &Array<u32>,
    holes_offset: &Array<u32>,
    parts_offset: &Array<u32>,
    holes: &Array<u32>,
    parts: &Array<u32>,
    excitation_phase: &Array<f64>,
    s_out: &mut Array<f64>,
    f_out: &mut Array<f64>,
    wslot: usize,
    ldet: usize,
    gdet: usize,
    out: usize,
    #[comptime] left_rank: usize,
    #[comptime] greater_rank: usize,
    #[comptime] l: usize,
    #[comptime] m: usize,
) {
    let off2 = wslot * 2usize;
    let off4 = wslot * 4usize;

    let view = GpuSameSpinView {
        slab: slab.slice(0usize, slab.len()),
        x_off: x_off.slice(off2, off2 + 2usize),
        y_off: y_off.slice(off2, off2 + 2usize),
        ff_off: ff_off.slice(off4, off4 + 4usize),
        phase: phase[wslot],
        tilde_s_prod: tilde_s_prod[wslot],
        f0f: f0f.slice(off2, off2 + 2usize),
        m,
        nmo: nmo[wslot],
        nocc: nocc[wslot],
    };

    let phase_external = excitation_phase[ldet] * excitation_phase[gdet];

    if comptime!(l == 0usize) {
        let pref = view.phase * view.tilde_s_prod;
        let mut s = 0.0;
        let mut f = 0.0;

        if comptime!(m == 0usize) {
            s = pref;
            f = pref * view.f0f[0];
        } else if comptime!(m == 1usize) {
            f = pref * view.f0f[1];
        }

        s_out[out] = phase_external * s;
        f_out[out] = phase_external * f;
    } else {
        let mut rows = Array::<u32>::new(l);
        let mut cols = Array::<u32>::new(l);
        let mut det0 = Array::<f64>::new(l * l);

        fill_factor_indices(
            &view,
            holes_offset,
            parts_offset,
            holes,
            parts,
            ldet,
            gdet,
            &mut rows,
            &mut cols,
            left_rank,
            greater_rank,
        );

        if comptime!(m == 0usize) {
            let mut cof = Array::<f64>::new(l * l);

            build_d_m0(&view, &rows, &cols, &mut det0, l);

            let values = xw_overlap_f_m0(&view, &rows, &cols, &det0, &mut cof, l);

            s_out[out] = phase_external * values.s;
            f_out[out] = phase_external * values.f;
        } else {
            let mut det1 = Array::<f64>::new(l * l);
            let mut work = Array::<f64>::new(l * l);
            let mut cof = Array::<f64>::new(l * l);
            let mut new_col = Array::<f64>::new(l);

            build_d_gen(&view, 0usize, &rows, &cols, &mut det0, l);
            build_d_gen(&view, 1usize, &rows, &cols, &mut det1, l);

            let values = xw_overlap_f(
                &view,
                &rows,
                &cols,
                &det0,
                &det1,
                &mut work,
                &mut cof,
                &mut new_col,
                l,
            );

            s_out[out] = phase_external * values.s;
            f_out[out] = phase_external * values.f;
        }
    }
}

/// Generate one same-spin factor entry per work item for a rank-homogeneous target/source block.
/// # Arguments:
/// - `slab`: Compact Wick tensor slab.
/// - `x_off`: Wick `X` offsets.
/// - `y_off`: Wick `Y` offsets.
/// - `ff_off`: Wick Fock-intermediate offsets.
/// - `phase`: Wick orbital-pairing phases.
/// - `tilde_s_prod`: Wick singular-value products.
/// - `f0f`: Scalar Fock intermediates.
/// - `nmo`: Molecular-orbital dimensions.
/// - `nocc`: Occupied-orbital dimensions.
/// - `holes_offset`: Excitation-hole offsets.
/// - `parts_offset`: Excitation-particle offsets.
/// - `holes`: Flattened excitation holes.
/// - `parts`: Flattened excitation particles.
/// - `excitation_phase`: Determinant excitation phases.
/// - `target_rep_det`: Target rank-sorted representative determinant IDs.
/// - `target_component`: Target parent-local component IDs.
/// - `source_rep_det`: Source rank-sorted representative determinant IDs.
/// - `source_component`: Source parent-local component IDs.
/// - `s_out`: Overlap factor output.
/// - `f_out`: Fock factor output.
/// - `wslot`: Flattened same-spin Wick slot.
/// - `target_offset`: Target rank-group offset.
/// - `target_len`: Target rank-group length.
/// - `source_offset`: Source rank-group offset.
/// - `source_len`: Source rank-group length.
/// - `target_component_base`: First output target component.
/// - `source_component_base`: First output source component.
/// - `source_stride`: Compact source-component row stride.
/// - `target_left`: Whether target representatives are left determinants.
/// - `target_rank`: Target spin excitation rank.
/// - `source_rank`: Source spin excitation rank.
/// - `l`: Total contraction determinant rank.
/// - `m`: Number of zero-overlap occupied-orbital pairs.
/// # Returns
/// - `()`: Writes one factor pair per work item.
#[cube(launch_unchecked)]
fn one_body_factor_kernel(
    slab: &Array<f64>,
    x_off: &Array<u32>,
    y_off: &Array<u32>,
    ff_off: &Array<u32>,
    phase: &Array<f64>,
    tilde_s_prod: &Array<f64>,
    f0f: &Array<f64>,
    nmo: &Array<u32>,
    nocc: &Array<u32>,
    holes_offset: &Array<u32>,
    parts_offset: &Array<u32>,
    holes: &Array<u32>,
    parts: &Array<u32>,
    excitation_phase: &Array<f64>,
    target_rep_det: &Array<u32>,
    target_component: &Array<u32>,
    source_rep_det: &Array<u32>,
    source_component: &Array<u32>,
    s_out: &mut Array<f64>,
    f_out: &mut Array<f64>,
    wslot: usize,
    target_offset: usize,
    target_len: usize,
    source_offset: usize,
    source_len: usize,
    target_component_base: usize,
    source_component_base: usize,
    source_stride: usize,
    #[comptime] target_left: bool,
    #[comptime] target_rank: usize,
    #[comptime] source_rank: usize,
    #[comptime] l: usize,
    #[comptime] m: usize,
) {
    if ABSOLUTE_POS >= target_len * source_len {
        terminate!();
    }

    let tpos = ABSOLUTE_POS / source_len;
    let spos = ABSOLUTE_POS - tpos * source_len;
    let tdet = usize::cast_from(target_rep_det[target_offset + tpos]);
    let sdet = usize::cast_from(source_rep_det[source_offset + spos]);
    let tcomp = usize::cast_from(target_component[target_offset + tpos]);
    let scomp = usize::cast_from(source_component[source_offset + spos]);

    let ldet = if comptime!(target_left) { tdet } else { sdet };
    let gdet = if comptime!(target_left) { sdet } else { tdet };
    let left_rank = if comptime!(target_left) {
        target_rank
    } else {
        source_rank
    };
    let greater_rank = if comptime!(target_left) {
        source_rank
    } else {
        target_rank
    };
    let out = (tcomp - target_component_base) * source_stride + scomp - source_component_base;

    evaluate_factor_entry(
        slab,
        x_off,
        y_off,
        ff_off,
        phase,
        tilde_s_prod,
        f0f,
        nmo,
        nocc,
        holes_offset,
        parts_offset,
        holes,
        parts,
        excitation_phase,
        s_out,
        f_out,
        wslot,
        ldet,
        gdet,
        out,
        left_rank,
        greater_rank,
        l,
        m,
    );
}

/// Generate same-component factors `S_{aa}` and `F_{aa}` for one spin rank group.
/// # Arguments:
/// - `slab`: Compact Wick tensor slab.
/// - `x_off`: Wick `X` offsets.
/// - `y_off`: Wick `Y` offsets.
/// - `ff_off`: Wick Fock-intermediate offsets.
/// - `phase`: Wick orbital-pairing phases.
/// - `tilde_s_prod`: Wick singular-value products.
/// - `f0f`: Scalar Fock intermediates.
/// - `nmo`: Molecular-orbital dimensions.
/// - `nocc`: Occupied-orbital dimensions.
/// - `holes_offset`: Excitation-hole offsets.
/// - `parts_offset`: Excitation-particle offsets.
/// - `holes`: Flattened excitation holes.
/// - `parts`: Flattened excitation particles.
/// - `excitation_phase`: Determinant excitation phases.
/// - `rep_det`: Rank-sorted representative determinant IDs.
/// - `component`: Parent-local component IDs.
/// - `s_out`: Diagonal overlap factor output.
/// - `f_out`: Diagonal Fock factor output.
/// - `wslot`: Flattened same-spin Wick slot.
/// - `offset`: Rank-group offset.
/// - `len`: Rank-group length.
/// - `rank`: Excitation rank on each side.
/// - `l`: Total contraction rank `2 rank`.
/// - `m`: Number of zero-overlap occupied-orbital pairs.
/// # Returns
/// - `()`: Writes one diagonal factor pair per work item.
#[cube(launch_unchecked)]
fn one_body_diagonal_factor_kernel(
    slab: &Array<f64>,
    x_off: &Array<u32>,
    y_off: &Array<u32>,
    ff_off: &Array<u32>,
    phase: &Array<f64>,
    tilde_s_prod: &Array<f64>,
    f0f: &Array<f64>,
    nmo: &Array<u32>,
    nocc: &Array<u32>,
    holes_offset: &Array<u32>,
    parts_offset: &Array<u32>,
    holes: &Array<u32>,
    parts: &Array<u32>,
    excitation_phase: &Array<f64>,
    rep_det: &Array<u32>,
    component: &Array<u32>,
    s_out: &mut Array<f64>,
    f_out: &mut Array<f64>,
    wslot: usize,
    offset: usize,
    len: usize,
    #[comptime] rank: usize,
    #[comptime] l: usize,
    #[comptime] m: usize,
) {
    if ABSOLUTE_POS >= len {
        terminate!();
    }

    let pos = offset + ABSOLUTE_POS;
    let det = usize::cast_from(rep_det[pos]);
    let comp = usize::cast_from(component[pos]);

    evaluate_factor_entry(
        slab,
        x_off,
        y_off,
        ff_off,
        phase,
        tilde_s_prod,
        f0f,
        nmo,
        nocc,
        holes_offset,
        parts_offset,
        holes,
        parts,
        excitation_phase,
        s_out,
        f_out,
        wslot,
        det,
        det,
        comp,
        rank,
        rank,
        l,
        m,
    );
}

/// Convert a host launch count to CubeCL grid width.
/// # Arguments:
/// - `value`: Host launch count.
/// # Returns
/// - `u32`: Checked device-width launch count.
fn checked_u32(value: usize) -> u32 {
    u32::try_from(value).expect("GPU factor launch dimension exceeds u32")
}
