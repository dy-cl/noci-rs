// noci/factorise/onebody/gpu/factors.rs
//! GPU factor generation for factorised one-body NOCI operator contractions.

// External crate imports.
use cubecl::prelude::*;
use ndarray::Array1;

// Crate-root imports.
use crate::gpu::{GpuBuffer, GpuContext, GpuRuntime};
use crate::noci::types::{NOCIData, NOCIScalar};
use crate::nonorthogonalwicks::gpu::eval::onebody::xw_f;
use crate::nonorthogonalwicks::gpu::eval::overlap::xw_overlap;
use crate::nonorthogonalwicks::gpu::eval::prepare::{GpuSameSpinView, prepare_same};
use crate::nonorthogonalwicks::gpu::types::DeviceWicksShared;

// Parent/sibling imports.
use super::data::DeviceOneBodyData;

/// Initial one-body factor kernel cube dimension.
const FACTOR_CUBE_DIM: u32 = 128;

/// Build same-spin `S` and `F` factor rows from ordered Wick data on the device.
/// This is the GPU mirror of the CPU `build_spin_one_body_factors` operation and must preserve
/// ordered-parent-pair and `target_left` semantics without introducing conjugate shortcuts.
/// # Arguments:
/// - `data`: Shared NOCI determinant data used to derive GPU-resident Wick and excitation data.
/// - `x`: Source vector, used only to bind the scalar type for kernel specialisation.
/// # Returns
/// - `()`: Factor panels are intended to remain resident on the device.
pub(crate) fn build_spin_one_body_factors<T: NOCIScalar>(
    _data: &NOCIData<'_, T>,
    _x: &Array1<T>,
) {
    eprintln!("GPU one-body same-spin factor generation is not implemented yet");
    std::process::exit(1);
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
            ArrayArg::from_raw_parts(&wicks.slab.handle, wicks.slab.len()),
            ArrayArg::from_raw_parts(&wicks.x_off.handle, wicks.x_off.len()),
            ArrayArg::from_raw_parts(&wicks.y_off.handle, wicks.y_off.len()),
            ArrayArg::from_raw_parts(&wicks.ff_off.handle, wicks.ff_off.len()),
            ArrayArg::from_raw_parts(&wicks.phase.handle, wicks.phase.len()),
            ArrayArg::from_raw_parts(&wicks.tilde_s_prod.handle, wicks.tilde_s_prod.len()),
            ArrayArg::from_raw_parts(&wicks.f0f.handle, wicks.f0f.len()),
            ArrayArg::from_raw_parts(&wicks.m.handle, wicks.m.len()),
            ArrayArg::from_raw_parts(&wicks.nmo.handle, wicks.nmo.len()),
            ArrayArg::from_raw_parts(&wicks.nocc.handle, wicks.nocc.len()),
            ArrayArg::from_raw_parts(&topology.alpha_rank.handle, topology.alpha_rank.len()),
            ArrayArg::from_raw_parts(
                &topology.alpha_holes_offset.handle,
                topology.alpha_holes_offset.len(),
            ),
            ArrayArg::from_raw_parts(
                &topology.alpha_parts_offset.handle,
                topology.alpha_parts_offset.len(),
            ),
            ArrayArg::from_raw_parts(&topology.alpha_holes.handle, topology.alpha_holes.len()),
            ArrayArg::from_raw_parts(&topology.alpha_parts.handle, topology.alpha_parts.len()),
            ArrayArg::from_raw_parts(&topology.beta_rank.handle, topology.beta_rank.len()),
            ArrayArg::from_raw_parts(
                &topology.beta_holes_offset.handle,
                topology.beta_holes_offset.len(),
            ),
            ArrayArg::from_raw_parts(
                &topology.beta_parts_offset.handle,
                topology.beta_parts_offset.len(),
            ),
            ArrayArg::from_raw_parts(&topology.beta_holes.handle, topology.beta_holes.len()),
            ArrayArg::from_raw_parts(&topology.beta_parts.handle, topology.beta_parts.len()),
            ArrayArg::from_raw_parts(&topology.pha.handle, topology.pha.len()),
            ArrayArg::from_raw_parts(&topology.phb.handle, topology.phb.len()),
            ArrayArg::from_raw_parts(&target.rep_det.handle, target.rep_det.len()),
            ArrayArg::from_raw_parts(&target.component.handle, target.component.len()),
            ArrayArg::from_raw_parts(&source.rep_det.handle, source.rep_det.len()),
            ArrayArg::from_raw_parts(&source.component.handle, source.component.len()),
            ArrayArg::from_raw_parts(&out.s.handle, out.s.len()),
            ArrayArg::from_raw_parts(&out.f.handle, out.f.len()),
            pair.lp as u32,
            pair.gp as u32,
            pair.nref as u32,
            target.offset as u32,
            target.len as u32,
            source.offset as u32,
            source.len as u32,
            out.nsource as u32,
            pair.target_left,
            pair.alpha,
            ranks.target_rank as u32,
            ranks.source_rank as u32,
            ranks.l as u32,
        );
    }
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
    lp: u32,
    gp: u32,
    nref: u32,
    target_offset: u32,
    target_len: u32,
    source_offset: u32,
    source_len: u32,
    nsource: u32,
    #[comptime] target_left: bool,
    #[comptime] alpha: bool,
    #[comptime] target_rank: u32,
    #[comptime] source_rank: u32,
    #[comptime] l: u32,
) {
    if ABSOLUTE_POS >= target_len * source_len {
        return;
    }

    let tpos = ABSOLUTE_POS / source_len;
    let spos = ABSOLUTE_POS - tpos * source_len;
    let tdet = target_rep_det[target_offset + tpos];
    let sdet = source_rep_det[source_offset + spos];
    let tcomp = target_component[target_offset + tpos];
    let scomp = source_component[source_offset + spos];
    let ldet = if comptime!(target_left) { tdet } else { sdet };
    let gdet = if comptime!(target_left) { sdet } else { tdet };
    let wslot = (lp * nref + gp) * 2u32 + if comptime!(alpha) { 0u32 } else { 1u32 };
    let off2 = wslot * 2u32;
    let off4 = wslot * 4u32;

    let mut rows = Array::<u32>::new(l);
    let mut cols = Array::<u32>::new(l);
    let mut det0 = Array::<f64>::new(l * l);
    let mut det1 = Array::<f64>::new(l * l);
    let mut work = Array::<f64>::new(l * l);
    let mut cof = Array::<f64>::new(l * l);
    let mut new_col = Array::<f64>::new(l);
    let mut l_holes = Array::<u32>::new(target_rank);
    let mut l_parts = Array::<u32>::new(target_rank);
    let mut g_holes = Array::<u32>::new(source_rank);
    let mut g_parts = Array::<u32>::new(source_rank);

    let (lr, loh, lop, gr, goh, gop, phase_external) = if comptime!(alpha) {
        (
            alpha_rank[ldet],
            alpha_holes_offset[ldet],
            alpha_parts_offset[ldet],
            alpha_rank[gdet],
            alpha_holes_offset[gdet],
            alpha_parts_offset[gdet],
            pha[ldet] * pha[gdet],
        )
    } else {
        (
            beta_rank[ldet],
            beta_holes_offset[ldet],
            beta_parts_offset[ldet],
            beta_rank[gdet],
            beta_holes_offset[gdet],
            beta_parts_offset[gdet],
            phb[ldet] * phb[gdet],
        )
    };

    for k in 0u32..target_rank {
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
    for k in 0u32..source_rank {
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

    let view = GpuSameSpinView {
        slab,
        x_off: x_off.slice(off2, 2u32),
        y_off: y_off.slice(off2, 2u32),
        ff_off: ff_off.slice(off4, 4u32),
        phase: phase[wslot],
        tilde_s_prod: tilde_s_prod[wslot],
        f0f: f0f.slice(off2, 2u32),
        m: m[wslot],
        nmo: nmo[wslot],
        nocc: nocc[wslot],
    };

    prepare_same(
        &view, lr, gr, &l_holes, &l_parts, &g_holes, &g_parts, &mut rows, &mut cols, &mut det0,
        &mut det1, l,
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
    let out = tcomp * nsource + scomp;
    s_out[out] = phase_external * s;
    f_out[out] = phase_external * f;
}
