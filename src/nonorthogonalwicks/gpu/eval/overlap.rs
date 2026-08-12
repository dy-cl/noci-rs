// nonorthogonalwicks/gpu/eval/overlap.rs
//! GPU overlap nonorthogonal Wick evaluation.

// External crate imports.
use cubecl::prelude::*;

// Parent/sibling imports.
use super::helpers::{bit, det_or_zero, det2, det3, det4};
use super::prepare::{GpuSameSpinView, prefactor};

/// Evaluate the same-spin overlap between excited determinants generated from the ordered reference pair.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det0`: Prepared all-zero endpoint determinant.
/// - `det1`: Prepared all-one endpoint determinant when `m > 0`.
/// - `work`: Local mixed-determinant storage for fallback paths.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `f64`: Same-spin overlap excluding external determinant excitation phases.
#[cube]
pub(crate) fn xw_overlap(
    w: &GpuSameSpinView,
    det0: &Array<f64>,
    det1: &Array<f64>,
    work: &mut Array<f64>,
    #[comptime] l: usize,
) -> f64 {
    let m = usize::cast_from(w.m);
    let mut value: f64 = 0.0;
    if m > l {
        value = 0.0;
    } else if w.m == 0u32 {
        value = xw_overlap_m0(w, det0, l);
    } else if m == l {
        value = xw_overlap_ml(w, det1, l);
    } else {
        value = xw_overlap_gen(w, det0, det1, work, l);
    }
    value
}

/// Evaluate a same-spin overlap when `m = 0`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det0`: Prepared all-zero endpoint determinant.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `f64`: Same-spin overlap for `m = 0`.
#[cube]
pub(crate) fn xw_overlap_m0(
    w: &GpuSameSpinView,
    det0: &Array<f64>,
    #[comptime] l: usize,
) -> f64 {
    if comptime!(l == 0usize) {
        prefactor(w)
    } else if comptime!(l == 1usize) {
        xw_overlap_m0_l1(w, det0)
    } else if comptime!(l == 2usize) {
        xw_overlap_m0_l2(w, det0)
    } else if comptime!(l == 3usize) {
        xw_overlap_m0_l3(w, det0)
    } else if comptime!(l == 4usize) {
        xw_overlap_m0_l4(w, det0)
    } else if comptime!(l == 5usize) {
        xw_overlap_m0_l5(w, det0)
    } else if comptime!(l == 6usize) {
        xw_overlap_m0_l6(w, det0)
    } else {
        prefactor(w) * det_or_zero(det0, l)
    }
}

/// `Evaluate {}^{xw}\tilde S D_{00}^{(0)}`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det0`: Rank-one determinant.
/// # Returns
/// - `f64`: Fixed-rank overlap.
#[cube]
pub(crate) fn xw_overlap_m0_l1(
    w: &GpuSameSpinView,
    det0: &Array<f64>,
) -> f64 {
    prefactor(w) * det0[0]
}

/// `Evaluate {}^{xw}\tilde S det D(0,0)`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det0`: Rank-two determinant.
/// # Returns
/// - `f64`: Fixed-rank overlap.
#[cube]
pub(crate) fn xw_overlap_m0_l2(
    w: &GpuSameSpinView,
    det0: &Array<f64>,
) -> f64 {
    prefactor(w) * det2(det0[0], det0[1], det0[2], det0[3])
}

/// `Evaluate {}^{xw}\tilde S det D(0,0,0)`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det0`: Rank-three determinant.
/// # Returns
/// - `f64`: Fixed-rank overlap.
#[cube]
pub(crate) fn xw_overlap_m0_l3(
    w: &GpuSameSpinView,
    det0: &Array<f64>,
) -> f64 {
    prefactor(w) * det3(det0)
}

/// `Evaluate {}^{xw}\tilde S det D(0,0,0,0)`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det0`: Rank-four determinant.
/// # Returns
/// - `f64`: Fixed-rank overlap.
#[cube]
pub(crate) fn xw_overlap_m0_l4(
    w: &GpuSameSpinView,
    det0: &Array<f64>,
) -> f64 {
    prefactor(w) * det4(det0)
}

/// `Evaluate {}^{xw}\tilde S det D(0,\ldots,0)` for rank five.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det0`: Rank-five determinant.
/// # Returns
/// - `f64`: Fixed-rank overlap.
#[cube]
pub(crate) fn xw_overlap_m0_l5(
    w: &GpuSameSpinView,
    det0: &Array<f64>,
) -> f64 {
    prefactor(w) * det_or_zero(det0, 5usize)
}

/// `Evaluate {}^{xw}\tilde S det D(0,\ldots,0)` for rank six.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det0`: Rank-six determinant.
/// # Returns
/// - `f64`: Fixed-rank overlap.
#[cube]
pub(crate) fn xw_overlap_m0_l6(
    w: &GpuSameSpinView,
    det0: &Array<f64>,
) -> f64 {
    prefactor(w) * det_or_zero(det0, 6usize)
}

/// Evaluate a same-spin overlap when `m = L`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det1`: Prepared all-one endpoint determinant.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `f64`: Same-spin overlap for `m = L`.
#[cube]
pub(crate) fn xw_overlap_ml(
    w: &GpuSameSpinView,
    det1: &Array<f64>,
    #[comptime] l: usize,
) -> f64 {
    if comptime!(l == 0usize) {
        prefactor(w)
    } else if comptime!(l == 1usize) {
        xw_overlap_ml_l1(w, det1)
    } else if comptime!(l == 2usize) {
        xw_overlap_ml_l2(w, det1)
    } else if comptime!(l == 3usize) {
        xw_overlap_ml_l3(w, det1)
    } else {
        prefactor(w) * det_or_zero(det1, l)
    }
}

/// `Evaluate {}^{xw}\tilde S D_{00}^{(1)}`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det1`: Rank-one all-one determinant.
/// # Returns
/// - `f64`: Fixed-rank overlap.
#[cube]
pub(crate) fn xw_overlap_ml_l1(
    w: &GpuSameSpinView,
    det1: &Array<f64>,
) -> f64 {
    prefactor(w) * det1[0]
}

/// `Evaluate {}^{xw}\tilde S det D(1,1)`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det1`: Rank-two all-one determinant.
/// # Returns
/// - `f64`: Fixed-rank overlap.
#[cube]
pub(crate) fn xw_overlap_ml_l2(
    w: &GpuSameSpinView,
    det1: &Array<f64>,
) -> f64 {
    prefactor(w) * det2(det1[0], det1[1], det1[2], det1[3])
}

/// `Evaluate {}^{xw}\tilde S det D(1,1,1)`.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det1`: Rank-three all-one determinant.
/// # Returns
/// - `f64`: Fixed-rank overlap.
#[cube]
pub(crate) fn xw_overlap_ml_l3(
    w: &GpuSameSpinView,
    det1: &Array<f64>,
) -> f64 {
    prefactor(w) * det3(det1)
}

/// Evaluate the same-spin overlap for `0 < m < L` by summing constrained column assignments.
/// # Arguments:
/// - `w`: Device same-spin Wick view.
/// - `det0`: All-zero endpoint determinant.
/// - `det1`: All-one endpoint determinant.
/// - `work`: Mixed determinant storage.
/// - `l`: Compile-time total excitation rank.
/// # Returns
/// - `f64`: Constrained-distribution overlap.
#[cube]
pub(crate) fn xw_overlap_gen(
    w: &GpuSameSpinView,
    det0: &Array<f64>,
    det1: &Array<f64>,
    work: &mut Array<f64>,
    #[comptime] l: usize,
) -> f64 {
    let mut acc = 0.0;
    let limit = 1u32 << l;
    for bits in 0u32..limit {
        if bits.count_ones() == w.m {
            mix_dets_same(det0, det1, work, l, bits);
            acc += det_or_zero(work, l);
        }
    }
    prefactor(w) * acc
}

/// Form one mixed same-spin contraction determinant by selecting columns from endpoint determinants.
/// # Arguments:
/// - `det0`: All-zero endpoint determinant.
/// - `det1`: All-one endpoint determinant.
/// - `out`: Mixed determinant.
/// - `l`: Matrix dimension.
/// - `bits`: Column assignment bit mask.
/// # Returns
/// - `()`: Writes the mixed determinant.
#[cube]
pub(crate) fn mix_dets_same(
    det0: &Array<f64>,
    det1: &Array<f64>,
    out: &mut Array<f64>,
    l: usize,
    bits: u32,
) {
    for c in 0usize..l {
        let source_one = bit(bits, c) == 1usize;
        for r in 0usize..l {
            let i = r * l + c;
            out[i] = if source_one { det1[i] } else { det0[i] };
        }
    }
}
