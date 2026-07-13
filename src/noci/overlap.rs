// noci/overlap.rs

use crate::DetState;
use crate::nonorthogonalwicks::{WickScratchSpin, WicksPairView, WicksView};
use crate::nonorthogonalwicks::{lg_overlap, lg_overlap_same_f64, prepare_same};
use crate::time_call;

use super::naive::{build_s_pair, occ_coeffs};
use super::types::{DetPair, NOCIData, NOCIScalar};

/// Wrapper function which dispatches to overlap matrix-element evaluation routines depending on
/// user input and properties of the determinant pair involved.
/// If the determinant pair have the same Hermitian-orthonormal parent we may use the standard
/// Slater-Condon rules, if not we can either use generalised Slater-Condon rules or extended
/// non-orthogonal Wick's theorem to evaluate the matrix element.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose overlap matrix element is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Overlap matrix element between the determinant pair.
pub(crate) fn calculate_s_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair, {
        let ldet = pair.ldet;
        let gdet = pair.gdet;

        if ldet.parent == gdet.parent {
            let mocache = data
                .mocache
                .expect("Orthogonal overlap matrix elements require mocache.");
            if mocache[ldet.parent].orthogonal_slater_condon {
                return calculate_s_pair_orthogonal(ldet, gdet);
            }
        }

        if data.input.wicks.enabled {
            calculate_s_pair_wicks(ldet, gdet, data.wicks.unwrap(), scratch.unwrap())
        } else {
            calculate_s_pair_naive(data, ldet, gdet)
        }
    })
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// # Returns:
/// - `T`: Overlap matrix element between `ldet` and `gdet`.
pub(in crate::noci) fn calculate_s_pair_orthogonal<T: NOCIScalar>(
    ldet: &DetState<T>,
    gdet: &DetState<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_orthogonal, {
        if ldet.oa == gdet.oa && ldet.ob == gdet.ob {
            <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb))
        } else {
            <T as From<f64>>::from(0.0)
        }
    })
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma using
/// generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// # Returns:
/// - `T`: Overlap matrix element between `ldet` and `gdet`.
pub(in crate::noci) fn calculate_s_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_naive, {
        let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
        let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
        let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
        let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

        let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
        det_phase * pa.s * pb.s
    })
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma
/// using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// - `wicks`: View to the intermediates required for non-orthogonal Wick's theorem.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Overlap matrix element.
fn calculate_s_pair_wicks<T: NOCIScalar>(
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    wicks: &WicksView<T>,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_wicks, {
        let w = wicks.pair(ldet.parent, gdet.parent);

        let ex_la = &ldet.excitation.alpha;
        let ex_ga = &gdet.excitation.alpha;
        let ex_lb = &ldet.excitation.beta;
        let ex_gb = &gdet.excitation.beta;

        let la = ex_la.holes.len() + ex_ga.holes.len();
        let lb = ex_lb.holes.len() + ex_gb.holes.len();

        if w.aa.m > la || w.bb.m > lb {
            return <T as From<f64>>::from(0.0);
        }

        let pha = <T as From<f64>>::from(ldet.pha * gdet.pha);
        let phb = <T as From<f64>>::from(ldet.phb * gdet.phb);
        let zero = <T as From<f64>>::from(0.0);

        prepare_same(&w.aa, ex_la, ex_ga, &mut scratch.aa);
        let sa = pha * lg_overlap(&w.aa, ex_la, ex_ga, &mut scratch.aa);

        if sa == zero {
            return zero;
        }

        prepare_same(&w.bb, ex_lb, ex_gb, &mut scratch.bb);
        let sb = phb * lg_overlap(&w.bb, ex_lb, ex_gb, &mut scratch.bb);

        if sb == zero {
            return zero;
        }

        sa * sb
    })
}

/// Calculate the alpha same-spin overlap for an ordered Wick pair.
/// # Arguments:
/// - `ldet`: Left determinant.
/// - `gdet`: Right determinant.
/// - `w`: Wick intermediates for the ordered parent pair.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `f64`: Alpha same-spin overlap including determinant phases.
#[inline(always)]
pub(in crate::noci) fn calculate_s_alpha_pair_wicks(
    ldet: &DetState<f64>,
    gdet: &DetState<f64>,
    w: &WicksPairView<'_, f64>,
    scratch: &mut WickScratchSpin<f64>,
) -> f64 {
    let l_ex = &ldet.excitation.alpha;
    let g_ex = &gdet.excitation.alpha;
    ldet.pha * gdet.pha * lg_overlap_same_f64(&w.aa, l_ex, g_ex, &mut scratch.aa)
}

/// Calculate the beta same-spin overlap for an ordered Wick pair.
/// # Arguments:
/// - `ldet`: Left determinant.
/// - `gdet`: Right determinant.
/// - `w`: Wick intermediates for the ordered parent pair.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `f64`: Beta same-spin overlap including determinant phases.
#[inline(always)]
pub(in crate::noci) fn calculate_s_beta_pair_wicks(
    ldet: &DetState<f64>,
    gdet: &DetState<f64>,
    w: &WicksPairView<'_, f64>,
    scratch: &mut WickScratchSpin<f64>,
) -> f64 {
    let l_ex = &ldet.excitation.beta;
    let g_ex = &gdet.excitation.beta;
    ldet.phb * gdet.phb * lg_overlap_same_f64(&w.bb, l_ex, g_ex, &mut scratch.bb)
}
