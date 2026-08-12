// nonorthogonalwicks/gpu/eval/prepare.rs
//! GPU preparation of same-spin nonorthogonal Wick contraction determinants.

// External crate imports.
use cubecl::prelude::*;

/// Device-side same-spin Wick view over primitive flattened buffers.
#[derive(CubeType)]
pub(crate) struct GpuSameSpinView {
    /// Compact same-spin tensor slab.
    pub(crate) slab: Slice<f64>,
    /// Offsets to `X^(0)` and `X^(1)`.
    pub(crate) x_off: Slice<u32>,
    /// Offsets to `Y^(0)` and `Y^(1)`.
    pub(crate) y_off: Slice<u32>,
    /// Offsets to transposed `ff^(mi,mj)`.
    pub(crate) ff_off: Slice<u32>,
    /// Orbital-pairing phase.
    pub(crate) phase: f64,
    /// Product of non-zero occupied-orbital singular values.
    pub(crate) tilde_s_prod: f64,
    /// Scalar current-Fock intermediates.
    pub(crate) f0f: Slice<f64>,
    /// Number of zero-overlap occupied-orbital pairs.
    pub(crate) m: u32,
    /// Number of molecular orbitals.
    pub(crate) nmo: u32,
    /// Number of occupied orbitals.
    pub(crate) nocc: u32,
}

/// Read `X^{(m_i)}_{rc}` from compact same-spin storage.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `mi`: Fundamental-contraction assignment.
/// - `r`: Row orbital label.
/// - `c`: Column orbital label.
/// # Returns
/// - `f64`: Fundamental contraction entry.
#[cube]
pub(crate) fn x(
    w: &GpuSameSpinView,
    mi: usize,
    r: u32,
    c: u32,
) -> f64 {
    let offset = usize::cast_from(w.x_off[mi]);
    w.slab[offset + usize::cast_from(r * w.nmo + c)]
}

/// Read `Y^{(m_i)}_{rc}` from compact same-spin storage.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `mi`: Fundamental-contraction assignment.
/// - `r`: Row orbital label.
/// - `c`: Column orbital label.
/// # Returns
/// - `f64`: Fundamental contraction entry.
#[cube]
pub(crate) fn y(
    w: &GpuSameSpinView,
    mi: usize,
    r: u32,
    c: u32,
) -> f64 {
    let offset = usize::cast_from(w.y_off[mi]);
    w.slab[offset + usize::cast_from(r * w.nmo + c)]
}

/// Read transposed current-Fock one-column intermediate using CPU `ff_t_slice` `[z,r]` storage.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `mi`: Operator assignment.
/// - `mj`: Column assignment.
/// - `z`: Column orbital label.
/// - `r`: Row orbital label.
/// # Returns
/// - `f64`: `\mathcal F_{rz}^{(m_i,m_j)}`.
#[cube]
pub(crate) fn ff_t(
    w: &GpuSameSpinView,
    mi: usize,
    mj: usize,
    z: u32,
    r: u32,
) -> f64 {
    let offset = usize::cast_from(w.ff_off[mi * 2usize + mj]);
    w.slab[offset + usize::cast_from(z * w.nmo + r)]
}

/// Return the reduced reference prefactor `phase * tilde_s_prod`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// # Returns
/// - `f64`: Same-spin Wick prefactor.
#[cube]
pub(crate) fn prefactor(w: &GpuSameSpinView) -> f64 {
    w.phase * w.tilde_s_prod
}

/// Prepare the contraction-determinant quantities shared by same-spin overlap and Fock evaluators.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `l_rank`: Bra-reference excitation rank.
/// - `g_rank`: Ket-reference excitation rank.
/// - `l_holes`: Decoded bra holes.
/// - `l_parts`: Decoded bra particles.
/// - `g_holes`: Decoded ket holes.
/// - `g_parts`: Decoded ket particles.
/// - `rows`: Output determinant row labels.
/// - `cols`: Output determinant column labels.
/// - `det0`: Output `D(0,\ldots,0)`.
/// - `det1`: Output `D(1,\ldots,1)` for `m > 0`.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `()`: Writes determinant labels and required endpoint determinants.
#[cube]
pub(crate) fn prepare_same(
    w: &GpuSameSpinView,
    l_rank: usize,
    g_rank: usize,
    l_holes: &Array<u32>,
    l_parts: &Array<u32>,
    g_holes: &Array<u32>,
    g_parts: &Array<u32>,
    rows: &mut Array<u32>,
    cols: &mut Array<u32>,
    det0: &mut Array<f64>,
    det1: &mut Array<f64>,
    #[comptime] l: usize,
) {
    if w.m == 0u32 {
        prepare_same_m0(
            w, l_rank, g_rank, l_holes, l_parts, g_holes, g_parts, rows, cols, det0, l,
        );
    } else {
        prepare_same_gen(
            w, l_rank, g_rank, l_holes, l_parts, g_holes, g_parts, rows, cols, det0, det1, l,
        );
    }
}

/// `Prepare D(0,\ldots,0)` when `m = 0`.
/// # Arguments:
/// - See `prepare_same`.
/// # Returns
/// - `()`: Writes only the all-zero endpoint determinant.
#[cube]
pub(crate) fn prepare_same_m0(
    w: &GpuSameSpinView,
    l_rank: usize,
    g_rank: usize,
    l_holes: &Array<u32>,
    l_parts: &Array<u32>,
    g_holes: &Array<u32>,
    g_parts: &Array<u32>,
    rows: &mut Array<u32>,
    cols: &mut Array<u32>,
    det0: &mut Array<f64>,
    #[comptime] l: usize,
) {
    construct_determinant_indices(
        w, l_rank, g_rank, l_holes, l_parts, g_holes, g_parts, rows, cols,
    );
    build_d_m0(w, rows, cols, det0, l);
}

/// `Prepare fixed-rank L = 1 D(0)`.
/// # Arguments:
/// - See `prepare_same_m0`.
/// # Returns
/// - `()`: Writes rank-one determinant data.
#[cube]
pub(crate) fn prepare_same_m0_l1(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &mut Array<f64>,
) {
    det0[0] = x(w, 0usize, rows[0], cols[0]);
}

/// `Prepare fixed-rank L = 2 D(0,0)`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `det0`: Output rank-two determinant.
/// # Returns
/// - `()`: Writes rank-two determinant data.
#[cube]
pub(crate) fn prepare_same_m0_l2(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &mut Array<f64>,
) {
    build_d2(w, 0usize, rows, cols, det0);
}

/// `Prepare fixed-rank L = 3 D(0,0,0)`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `det0`: Output rank-three determinant.
/// # Returns
/// - `()`: Writes rank-three determinant data.
#[cube]
pub(crate) fn prepare_same_m0_l3(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &mut Array<f64>,
) {
    build_d_gen(w, 0usize, rows, cols, det0, 3usize);
}

/// `Prepare fixed-rank L = 4 D(0,0,0,0)`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `det0`: Output rank-four determinant.
/// # Returns
/// - `()`: Writes rank-four determinant data.
#[cube]
pub(crate) fn prepare_same_m0_l4(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &mut Array<f64>,
) {
    build_d_gen(w, 0usize, rows, cols, det0, 4usize);
}

/// `Prepare fixed-rank L = 5 D(0,\ldots,0)`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `det0`: Output rank-five determinant.
/// # Returns
/// - `()`: Writes rank-five determinant data.
#[cube]
pub(crate) fn prepare_same_m0_l5(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &mut Array<f64>,
) {
    build_d_gen(w, 0usize, rows, cols, det0, 5usize);
}

/// `Prepare fixed-rank L = 6 D(0,\ldots,0)`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `det0`: Output rank-six determinant.
/// # Returns
/// - `()`: Writes rank-six determinant data.
#[cube]
pub(crate) fn prepare_same_m0_l6(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &mut Array<f64>,
) {
    build_d_gen(w, 0usize, rows, cols, det0, 6usize);
}

/// Prepare `D(0,\ldots,0)` and `D(1,\ldots,1)` when `m > 0`.
/// # Arguments:
/// - See `prepare_same`.
/// # Returns
/// - `()`: Writes both endpoint determinants.
#[cube]
pub(crate) fn prepare_same_gen(
    w: &GpuSameSpinView,
    l_rank: usize,
    g_rank: usize,
    l_holes: &Array<u32>,
    l_parts: &Array<u32>,
    g_holes: &Array<u32>,
    g_parts: &Array<u32>,
    rows: &mut Array<u32>,
    cols: &mut Array<u32>,
    det0: &mut Array<f64>,
    det1: &mut Array<f64>,
    #[comptime] l: usize,
) {
    construct_determinant_indices(
        w, l_rank, g_rank, l_holes, l_parts, g_holes, g_parts, rows, cols,
    );
    build_d_gen(w, 0usize, rows, cols, det0, l);
    build_d_gen(w, 1usize, rows, cols, det1, l);
}

/// Construct row and column labels for a same-spin contraction determinant.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `l_rank`: Bra-reference excitation rank.
/// - `g_rank`: Ket-reference excitation rank.
/// - `l_holes`: Decoded bra holes.
/// - `l_parts`: Decoded bra particles.
/// - `g_holes`: Decoded ket holes.
/// - `g_parts`: Decoded ket particles.
/// - `rows`: Output rows.
/// - `cols`: Output columns.
/// # Returns
/// - `()`: Writes CPU-ordered determinant labels.
#[cube]
pub(crate) fn construct_determinant_indices(
    w: &GpuSameSpinView,
    l_rank: usize,
    g_rank: usize,
    l_holes: &Array<u32>,
    l_parts: &Array<u32>,
    g_holes: &Array<u32>,
    g_parts: &Array<u32>,
    rows: &mut Array<u32>,
    cols: &mut Array<u32>,
) {
    let nvirt = w.nmo - w.nocc;
    for k in 0usize..l_rank {
        rows[k] = l_parts[k] - w.nocc;
        cols[k] = l_holes[k];
    }
    for k in 0usize..g_rank {
        rows[l_rank + k] = nvirt + g_holes[k];
        cols[l_rank + k] = g_parts[k];
    }
}

/// Build `D(0,\ldots,0)` for rank-specialised m-zero paths.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `det0`: Output determinant.
/// - `l`: Compile-time determinant dimension.
/// # Returns
/// - `()`: Writes the fixed-rank determinant.
#[cube]
pub(crate) fn build_d_m0(
    w: &GpuSameSpinView,
    rows: &Array<u32>,
    cols: &Array<u32>,
    det0: &mut Array<f64>,
    #[comptime] l: usize,
) {
    if comptime!(l == 1usize) {
        prepare_same_m0_l1(w, rows, cols, det0);
    } else if comptime!(l == 2usize) {
        prepare_same_m0_l2(w, rows, cols, det0);
    } else if comptime!(l == 3usize) {
        prepare_same_m0_l3(w, rows, cols, det0);
    } else if comptime!(l == 4usize) {
        prepare_same_m0_l4(w, rows, cols, det0);
    } else if comptime!(l == 5usize) {
        prepare_same_m0_l5(w, rows, cols, det0);
    } else if comptime!(l == 6usize) {
        prepare_same_m0_l6(w, rows, cols, det0);
    } else {
        build_d_gen(w, 0usize, rows, cols, det0, l);
    }
}

/// Build the fixed rank-two X/Y contraction determinant.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `mi`: Fundamental-contraction assignment.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `d`: Output determinant.
/// # Returns
/// - `()`: Writes rank-two entries in CPU order.
#[cube]
pub(crate) fn build_d2(
    w: &GpuSameSpinView,
    mi: usize,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &mut Array<f64>,
) {
    d[0] = x(w, mi, rows[0], cols[0]);
    d[1] = y(w, mi, rows[0], cols[1]);
    d[2] = x(w, mi, rows[1], cols[0]);
    d[3] = x(w, mi, rows[1], cols[1]);
}

/// Build an arbitrary-rank X/Y contraction determinant.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `mi`: Fundamental-contraction assignment.
/// - `rows`: Determinant row labels.
/// - `cols`: Determinant column labels.
/// - `d`: Output determinant.
/// - `l`: Determinant dimension.
/// # Returns
/// - `()`: Writes row-major X/Y determinant entries.
#[cube]
pub(crate) fn build_d_gen(
    w: &GpuSameSpinView,
    mi: usize,
    rows: &Array<u32>,
    cols: &Array<u32>,
    d: &mut Array<f64>,
    l: usize,
) {
    for i in 0usize..l {
        for j in 0usize..l {
            d[i * l + j] = if i >= j {
                x(w, mi, rows[i], cols[j])
            } else {
                y(w, mi, rows[i], cols[j])
            };
        }
    }
}
