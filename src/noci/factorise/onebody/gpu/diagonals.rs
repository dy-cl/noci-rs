// noci/factorise/onebody/gpu/diagonals.rs
//! GPU diagonal construction for factorised one-body NOCI operator contractions.

// External crate imports.
use cubecl::prelude::*;

// Crate-root imports.
use crate::gpu::{GpuBuffer, GpuContext, GpuRuntime};

// Parent/sibling imports.
use super::data::DeviceOneBodyData;

/// Initial diagonal kernel cube dimension.
const DIAGONAL_CUBE_DIM: u32 = 128;

/// Launch determinant diagonal fill from same-parent alpha and beta factors.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `topology`: Device determinant topology.
/// - `sa`: Same-parent alpha overlap factors.
/// - `fa`: Same-parent alpha Fock factors.
/// - `sb`: Same-parent beta overlap factors.
/// - `fb`: Same-parent beta Fock factors.
/// - `m_diag`: Device diagonal of `F + \lambda S`.
/// - `s_diag`: Device diagonal of `S`.
/// - `args`: Parent entry base, dimensions and shift.
/// # Returns
/// - `()`: Writes diagonal values on the device.
pub(crate) fn launch_fill_one_body_diagonal_block(
    context: &GpuContext,
    topology: &DeviceOneBodyData,
    sa: &GpuBuffer<f64>,
    fa: &GpuBuffer<f64>,
    sb: &GpuBuffer<f64>,
    fb: &GpuBuffer<f64>,
    m_diag: &GpuBuffer<f64>,
    s_diag: &GpuBuffer<f64>,
    args: DiagonalBlockLaunch,
) {
    if args.nentry == 0 {
        return;
    }
    let cubes = checked_u32(args.nentry.div_ceil(DIAGONAL_CUBE_DIM as usize));
    unsafe {
        fill_one_body_diagonal_block_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(DIAGONAL_CUBE_DIM),
            sa.array_arg(),
            fa.array_arg(),
            sb.array_arg(),
            fb.array_arg(),
            topology.entry_det.array_arg(),
            topology.entry_a.array_arg(),
            topology.entry_b.array_arg(),
            m_diag.array_arg(),
            s_diag.array_arg(),
            args.lambda,
            args.entry_base,
            args.nentry,
            args.nsa,
            args.nsb,
        );
    }
}

/// Same-parent diagonal fill launch dimensions.
#[derive(Clone, Copy)]
pub(crate) struct DiagonalBlockLaunch {
    /// Parent entry base.
    pub(crate) entry_base: usize,
    /// Number of parent entries.
    pub(crate) nentry: usize,
    /// Source alpha component count.
    pub(crate) nsa: usize,
    /// Source beta component count.
    pub(crate) nsb: usize,
    /// Overlap shift.
    pub(crate) lambda: f64,
}

/// Convert host launch metadata to CubeCL grid-width `u32`.
/// # Arguments:
/// - `value`: Host launch value.
/// # Returns
/// - `u32`: Checked CubeCL grid-width value.
fn checked_u32(value: usize) -> u32 {
    u32::try_from(value).expect("GPU diagonal launch dimension exceeds u32")
}

/// Fill determinant diagonals from one same-parent factor block on the device.
/// # Arguments:
/// - `sa`: Row-major alpha overlap factors.
/// - `fa`: Row-major alpha Fock factors.
/// - `sb`: Row-major beta overlap factors.
/// - `fb`: Row-major beta Fock factors.
/// - `entry_det`: Parent entry determinant IDs.
/// - `entry_a`: Parent entry alpha components.
/// - `entry_b`: Parent entry beta components.
/// - `m_diag`: Output diagonal of `F + \lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// - `lambda`: Scalar overlap shift.
/// - `entry_base`: Parent entry base.
/// - `nentry`: Number of parent entries.
/// - `nsa`: Source alpha component count.
/// - `nsb`: Source beta component count.
/// # Returns
/// - `()`: Writes diagonal values for actual determinants.
#[cube(launch_unchecked)]
pub(crate) fn fill_one_body_diagonal_block_kernel(
    sa: &Array<f64>,
    fa: &Array<f64>,
    sb: &Array<f64>,
    fb: &Array<f64>,
    entry_det: &Array<u32>,
    entry_a: &Array<u32>,
    entry_b: &Array<u32>,
    m_diag: &mut Array<f64>,
    s_diag: &mut Array<f64>,
    lambda: f64,
    entry_base: usize,
    nentry: usize,
    nsa: usize,
    nsb: usize,
) {
    if ABSOLUTE_POS >= nentry {
        terminate!();
    }
    let entry = entry_base + ABSOLUTE_POS;
    let a = usize::cast_from(entry_a[entry]);
    let b = usize::cast_from(entry_b[entry]);
    let saa = sa[a * nsa + a];
    let faa = fa[a * nsa + a];
    let sbb = sb[b * nsb + b];
    let fbb = fb[b * nsb + b];
    let s = saa * sbb;
    let det = entry_det[entry];
    let det = usize::cast_from(det);
    s_diag[det] = s;
    m_diag[det] = faa * sbb + saa * fbb + lambda * s;
}
