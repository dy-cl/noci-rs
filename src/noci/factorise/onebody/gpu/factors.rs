// noci/factorise/onebody/gpu/factors.rs
//! GPU factor generation for factorised one-body NOCI operator contractions.

// External crate imports.
use cubecl::prelude::*;

// Crate-root imports.
use crate::gpu::{GpuBuffer, GpuContext, GpuRuntime};
use crate::nonorthogonalwicks::gpu::{
    DeviceWicksShared, GpuSameSpinView, prepare_same, xw_f, xw_overlap,
};

// Parent/sibling imports.
use super::data::{DeviceOneBodyData, GpuOneBodyData};

/// Initial one-body factor kernel cube dimension.
const FACTOR_CUBE_DIM: u32 = 128;

/// Build same-spin `S` and `F` factor rows from ordered Wick data on the device.
/// This is the GPU mirror of the CPU `build_spin_one_body_factors` operation and must preserve
/// ordered-parent-pair and `target_left` semantics without introducing conjugate shortcuts.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `wicks`: Device Wick buffers.
/// - `topology`: Device determinant topology.
/// - `host`: Host topology metadata retained for rank-group offsets.
/// - `request`: Parent, spin and panel request.
/// - `out`: Device factor outputs.
/// # Returns
/// - `()`: Factor panel values are written on the device.
pub(crate) fn build_spin_one_body_factors(
    context: &GpuContext,
    wicks: &DeviceWicksShared,
    topology: &DeviceOneBodyData,
    host: &GpuOneBodyData,
    request: FactorRequest,
    out: FactorOutput<'_>,
) {
    if request.target_component_end <= request.target_component_base || request.nsource == 0 {
        return;
    }
    let max_target_rank = max_rank(host, request.alpha);
    let max_source_rank = max_target_rank;
    let pair = FactorPair {
        lp: request.lp,
        gp: request.gp,
        nref: request.nref,
        target_left: request.target_left,
        alpha: request.alpha,
    };

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
                0,
                request.nsource,
            );
            if source.len == 0 {
                continue;
            }
            launch_spin_rank_block(
                context,
                wicks,
                topology,
                pair,
                target,
                source,
                out,
                FactorRanks {
                    target_rank,
                    source_rank,
                    l: target_rank + source_rank,
                },
            );
        }
    }
}

/// Launch one homogeneous rank block of same-spin factor generation.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `wicks`: Device Wick buffers.
/// - `topology`: Device determinant topology.
/// - `pair`: Ordered parent pair and spin selector.
/// - `target`: Rank-group target representatives.
/// - `source`: Rank-group source representatives.
/// - `out`: Output S and F factor buffers in CPU-compatible row-major layout.
/// - `ranks`: Target rank, source rank and total rank `L`.
/// # Returns
/// - `()`: Launches one rank-homogeneous factor-generation kernel.
pub(crate) fn launch_spin_rank_block(
    context: &GpuContext,
    wicks: &DeviceWicksShared,
    topology: &DeviceOneBodyData,
    pair: FactorPair,
    target: RankGroup,
    source: RankGroup,
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
    let cubes = nfactors.div_ceil(FACTOR_CUBE_DIM as usize) as u32;

    unsafe {
        one_body_factor_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(FACTOR_CUBE_DIM),
            wicks.slab.array_arg(),
            wicks.x_off.array_arg(),
            wicks.y_off.array_arg(),
            wicks.ff_off.array_arg(),
            wicks.phase.array_arg(),
            wicks.tilde_s_prod.array_arg(),
            wicks.f0f.array_arg(),
            wicks.m.array_arg(),
            wicks.nmo.array_arg(),
            wicks.nocc.array_arg(),
            topology.alpha_rank.array_arg(),
            topology.alpha_holes_offset.array_arg(),
            topology.alpha_parts_offset.array_arg(),
            topology.alpha_holes.array_arg(),
            topology.alpha_parts.array_arg(),
            topology.beta_rank.array_arg(),
            topology.beta_holes_offset.array_arg(),
            topology.beta_parts_offset.array_arg(),
            topology.beta_holes.array_arg(),
            topology.beta_parts.array_arg(),
            topology.pha.array_arg(),
            topology.phb.array_arg(),
            target.rep_det.array_arg(),
            target.component.array_arg(),
            source.rep_det.array_arg(),
            source.component.array_arg(),
            out.s.array_arg(),
            out.f.array_arg(),
            pair.lp,
            pair.gp,
            pair.nref,
            target.offset,
            target.len,
            source.offset,
            source.len,
            out.target_component_base,
            out.nsource,
            pair.target_left,
            pair.alpha,
            ranks.target_rank,
            ranks.source_rank,
            ranks.l,
        );
    }
}

/// Parent, spin and component-panel request for one same-spin factor table.
#[derive(Clone, Copy)]
pub(crate) struct FactorRequest {
    /// Target parent `Q`.
    pub(crate) target_parent: usize,
    /// Source parent `P`.
    pub(crate) source_parent: usize,
    /// Left parent in the ordered Wick pair.
    pub(crate) lp: usize,
    /// Greater parent in the ordered Wick pair.
    pub(crate) gp: usize,
    /// Number of reference parents.
    pub(crate) nref: usize,
    /// Whether target representatives are left determinants.
    pub(crate) target_left: bool,
    /// Whether to build alpha or beta factors.
    pub(crate) alpha: bool,
    /// First target component represented by output row zero.
    pub(crate) target_component_base: usize,
    /// One-past-last target component represented in the output panel.
    pub(crate) target_component_end: usize,
    /// Full logical source component count.
    pub(crate) nsource: usize,
}

/// Uniform ordered-pair metadata for one rank-homogeneous launch.
#[derive(Clone, Copy)]
pub(crate) struct FactorPair {
    /// Left parent in the ordered Wick pair.
    pub(crate) lp: usize,
    /// Greater parent in the ordered Wick pair.
    pub(crate) gp: usize,
    /// Number of reference parents.
    pub(crate) nref: usize,
    /// Whether target representatives are left determinants.
    pub(crate) target_left: bool,
    /// Whether the launch builds alpha factors.
    pub(crate) alpha: bool,
}

/// Rank-homogeneous representative group within one parent.
#[derive(Clone, Copy)]
pub(crate) struct RankGroup<'a> {
    /// Rank-sorted representative determinant IDs.
    pub(crate) rep_det: &'a GpuBuffer<u32>,
    /// Original parent-local component IDs.
    pub(crate) component: &'a GpuBuffer<u32>,
    /// Group start offset.
    pub(crate) offset: usize,
    /// Group length.
    pub(crate) len: usize,
}

/// Device factor output buffers.
#[derive(Clone, Copy)]
pub(crate) struct FactorOutput<'a> {
    /// Row-major overlap factor table.
    pub(crate) s: &'a GpuBuffer<f64>,
    /// Row-major Fock factor table.
    pub(crate) f: &'a GpuBuffer<f64>,
    /// First target component represented by output row zero.
    pub(crate) target_component_base: usize,
    /// Full logical source component count.
    pub(crate) nsource: usize,
}

/// Compile-time rank specialisation arguments.
#[derive(Clone, Copy)]
pub(crate) struct FactorRanks {
    /// Target spin excitation rank.
    pub(crate) target_rank: usize,
    /// Source spin excitation rank.
    pub(crate) source_rank: usize,
    /// Total contraction determinant rank.
    pub(crate) l: usize,
}

/// Select one host-known rank group, optionally narrowed to a component panel.
/// # Arguments:
/// - `host`: Host topology metadata.
/// - `device`: Device topology buffers.
/// - `parent`: Parent reference index.
/// - `alpha`: Whether to select alpha or beta components.
/// - `rank`: Excitation rank.
/// - `component_base`: First requested parent-local component.
/// - `component_end`: One-past-last requested parent-local component.
/// # Returns
/// - `RankGroup`: Device buffers plus narrowed offset and length.
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

/// Maximum decoded excitation rank for a spin sector.
/// # Arguments:
/// - `host`: Host topology.
/// - `alpha`: Whether to select alpha or beta rank.
/// # Returns
/// - `usize`: Maximum rank.
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

/// Generate one same-spin `S` and `F` factor entry per work item for a homogeneous rank block.
/// # Arguments:
/// - Primitive Wick, topology and factor buffers.
/// # Returns
/// - `()`: Writes row-major factor entries using original component IDs.
#[cube(launch_unchecked)]
fn one_body_factor_kernel(
    slab: &Array<f64>,
    x_off: &Array<u32>,
    y_off: &Array<u32>,
    ff_off: &Array<u32>,
    phase: &Array<f64>,
    tilde_s_prod: &Array<f64>,
    f0f: &Array<f64>,
    m: &Array<u32>,
    nmo: &Array<u32>,
    nocc: &Array<u32>,
    alpha_rank: &Array<u32>,
    alpha_holes_offset: &Array<u32>,
    alpha_parts_offset: &Array<u32>,
    alpha_holes: &Array<u32>,
    alpha_parts: &Array<u32>,
    beta_rank: &Array<u32>,
    beta_holes_offset: &Array<u32>,
    beta_parts_offset: &Array<u32>,
    beta_holes: &Array<u32>,
    beta_parts: &Array<u32>,
    pha: &Array<f64>,
    phb: &Array<f64>,
    target_rep_det: &Array<u32>,
    target_component: &Array<u32>,
    source_rep_det: &Array<u32>,
    source_component: &Array<u32>,
    s_out: &mut Array<f64>,
    f_out: &mut Array<f64>,
    lp: usize,
    gp: usize,
    nref: usize,
    target_offset: usize,
    target_len: usize,
    source_offset: usize,
    source_len: usize,
    target_component_base: usize,
    nsource: usize,
    #[comptime] target_left: bool,
    #[comptime] alpha: bool,
    #[comptime] target_rank: usize,
    #[comptime] source_rank: usize,
    #[comptime] l: usize,
) {
    if ABSOLUTE_POS < target_len * source_len {
        let tpos = ABSOLUTE_POS / source_len;
        let spos = ABSOLUTE_POS - tpos * source_len;

        let tdet = usize::cast_from(target_rep_det[target_offset + tpos]);
        let sdet = usize::cast_from(source_rep_det[source_offset + spos]);
        let tcomp = usize::cast_from(target_component[target_offset + tpos]);
        let scomp = usize::cast_from(source_component[source_offset + spos]);

        let ldet = if comptime!(target_left) { tdet } else { sdet };
        let gdet = if comptime!(target_left) { sdet } else { tdet };

        let wslot = (lp * nref + gp) * 2usize + if comptime!(alpha) { 0usize } else { 1usize };
        let off2 = wslot * 2usize;
        let off4 = wslot * 4usize;

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

        let phase_external = if comptime!(alpha) {
            pha[ldet] * pha[gdet]
        } else {
            phb[ldet] * phb[gdet]
        };

        let view = GpuSameSpinView {
            slab: slab.slice(0usize, slab.len()),
            x_off: x_off.slice(off2, off2 + 2usize),
            y_off: y_off.slice(off2, off2 + 2usize),
            ff_off: ff_off.slice(off4, off4 + 4usize),
            phase: phase[wslot],
            tilde_s_prod: tilde_s_prod[wslot],
            f0f: f0f.slice(off2, off2 + 2usize),
            m: usize::cast_from(m[wslot]),
            nmo: nmo[wslot],
            nocc: nocc[wslot],
        };

        let out = (tcomp - target_component_base) * nsource + scomp;

        // L = 0 must not instantiate any zero-length private CubeCL arrays.
        if comptime!(l == 0usize) {
            let pref = view.phase * view.tilde_s_prod;
            let mut s = 0.0;
            let mut f = 0.0;

            // det(D_0x0) = 1.
            if view.m == 0usize {
                s = pref;
                f = pref * view.f0f[0];
            } else if view.m == 1usize {
                // The overlap vanishes, but the one-body operator can absorb
                // the single zero-overlap pair.
                f = pref * view.f0f[1];
            }

            s_out[out] = phase_external * s;
            f_out[out] = phase_external * f;
        } else {
            let (lr, loh, lop, gr, goh, gop) = if comptime!(alpha) {
                (
                    usize::cast_from(alpha_rank[ldet]),
                    usize::cast_from(alpha_holes_offset[ldet]),
                    usize::cast_from(alpha_parts_offset[ldet]),
                    usize::cast_from(alpha_rank[gdet]),
                    usize::cast_from(alpha_holes_offset[gdet]),
                    usize::cast_from(alpha_parts_offset[gdet]),
                )
            } else {
                (
                    usize::cast_from(beta_rank[ldet]),
                    usize::cast_from(beta_holes_offset[ldet]),
                    usize::cast_from(beta_parts_offset[ldet]),
                    usize::cast_from(beta_rank[gdet]),
                    usize::cast_from(beta_holes_offset[gdet]),
                    usize::cast_from(beta_parts_offset[gdet]),
                )
            };

            // CubeCL/CUDA cannot generate C arrays of length zero. The loops
            // below still execute zero times for a rank-zero side; only the
            // physical private allocation is padded to one element.
            let left_scratch_rank = if comptime!(left_rank == 0usize) {
                1usize
            } else {
                left_rank
            };
            let greater_scratch_rank = if comptime!(greater_rank == 0usize) {
                1usize
            } else {
                greater_rank
            };

            let mut rows = Array::<u32>::new(l);
            let mut cols = Array::<u32>::new(l);
            let mut det0 = Array::<f64>::new(l * l);
            let mut det1 = Array::<f64>::new(l * l);
            let mut work = Array::<f64>::new(l * l);
            let mut cof = Array::<f64>::new(l * l);
            let mut new_col = Array::<f64>::new(l);

            let mut l_holes = Array::<u32>::new(left_scratch_rank);
            let mut l_parts = Array::<u32>::new(left_scratch_rank);
            let mut g_holes = Array::<u32>::new(greater_scratch_rank);
            let mut g_parts = Array::<u32>::new(greater_scratch_rank);

            for k in 0usize..left_rank {
                l_holes[k] = if comptime!(alpha) {
                    alpha_holes[loh + k]
                } else {
                    beta_holes[loh + k]
                };

                l_parts[k] = if comptime!(alpha) {
                    alpha_parts[lop + k]
                } else {
                    beta_parts[lop + k]
                };
            }

            for k in 0usize..greater_rank {
                g_holes[k] = if comptime!(alpha) {
                    alpha_holes[goh + k]
                } else {
                    beta_holes[goh + k]
                };

                g_parts[k] = if comptime!(alpha) {
                    alpha_parts[gop + k]
                } else {
                    beta_parts[gop + k]
                };
            }

            prepare_same(
                &view, lr, gr, &l_holes, &l_parts, &g_holes, &g_parts, &mut rows, &mut cols,
                &mut det0, &mut det1, l,
            );

            let s = xw_overlap(&view, &det0, &det1, &mut work, l);

            let f = xw_f(
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

            s_out[out] = phase_external * s;
            f_out[out] = phase_external * f;
        }
    }
}
