// noci/factorise/onebody/gpu/diagonals.rs
//! GPU diagonal construction for factorised one-body NOCI operator contractions.

// External crate imports.
use cubecl::prelude::*;

// Crate-root imports.
use crate::gpu::{GpuBuffer, GpuContext, GpuRuntime};

// Parent/sibling imports.
use super::data::DeviceOneBodyData;

/// Launch determinant diagonal construction from same-component spin factors.
/// For determinant `(a,b)`, evaluates
/// `(F + lambda S)_{II} = F^alpha_{aa} S^beta_{bb} + S^alpha_{aa} F^beta_{bb}
/// + lambda S^alpha_{aa} S^beta_{bb}` without dense same-spin factor tables.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `topology`: Device determinant topology.
/// - `sa`: Alpha diagonal overlap factors `S^alpha_{aa}`.
/// - `fa`: Alpha diagonal Fock factors `F^alpha_{aa}`.
/// - `sb`: Beta diagonal overlap factors `S^beta_{bb}`.
/// - `fb`: Beta diagonal Fock factors `F^beta_{bb}`.
/// - `m_diag`: Device diagonal of `F + lambda S`.
/// - `s_diag`: Device diagonal of `S`.
/// - `args`: Parent-entry dimensions and overlap shift.
/// # Returns
/// - `()`: Writes determinant-space diagonal values.
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

    let cube_dim = 128u32;
    let cubes = checked_u32(args.nentry.div_ceil(cube_dim as usize));

    unsafe {
        fill_one_body_diagonal_block_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(cube_dim),
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
        );
    }
}

/// Same-parent diagonal kernel launch dimensions.
#[derive(Clone, Copy)]
pub(crate) struct DiagonalBlockLaunch {
    /// Parent entry base.
    pub(crate) entry_base: usize,
    /// Number of parent entries.
    pub(crate) nentry: usize,
    /// Scalar overlap shift.
    pub(crate) lambda: f64,
}

/// Fill determinant diagonals from same-component alpha and beta factors.
/// # Arguments:
/// - `sa`: Alpha diagonal overlap factors.
/// - `fa`: Alpha diagonal Fock factors.
/// - `sb`: Beta diagonal overlap factors.
/// - `fb`: Beta diagonal Fock factors.
/// - `entry_det`: Parent-entry determinant IDs.
/// - `entry_a`: Parent-entry alpha component IDs.
/// - `entry_b`: Parent-entry beta component IDs.
/// - `m_diag`: Output diagonal of `F + lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// - `lambda`: Scalar overlap shift.
/// - `entry_base`: Parent-entry base.
/// - `nentry`: Number of parent entries.
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
) {
    if ABSOLUTE_POS >= nentry {
        terminate!();
    }

    let entry = entry_base + ABSOLUTE_POS;
    let a = usize::cast_from(entry_a[entry]);
    let b = usize::cast_from(entry_b[entry]);
    let saa = sa[a];
    let faa = fa[a];
    let sbb = sb[b];
    let fbb = fb[b];
    let s = saa * sbb;
    let det = usize::cast_from(entry_det[entry]);

    s_diag[det] = s;
    m_diag[det] = faa * sbb + saa * fbb + lambda * s;
}

/// Convert a host diagonal launch count to CubeCL grid width.
/// # Arguments:
/// - `value`: Host launch count.
/// # Returns
/// - `u32`: Checked device-width launch count.
fn checked_u32(value: usize) -> u32 {
    u32::try_from(value).expect("GPU diagonal launch dimension exceeds u32")
}
