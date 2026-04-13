// noci/overlap.rs
use crate::{AoData, SCFState};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use super::types::{DetPair, NOCIData};

use crate::nonorthogonalwicks::{prepare_same, lg_overlap};
use super::naive::{occ_coeffs, build_s_pair};

/// Wrapper function which dispatches to overlap matrix-element evaluation routines depending on
/// user input and properties of the determinant pair involved. If the determinant pair have the
/// same parents we may use the standard Slater-Condon rules, if not we can either use generalised
/// Slater-Condon rules or extended non-orthogonal Wick's theorem to evaluate the matrix element.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose overlap matrix element is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `f64`: Overlap matrix element between the determinant pair.
pub(crate) fn calculate_s_pair(data: &NOCIData<'_>, pair: DetPair<'_>, scratch: Option<&mut WickScratchSpin>) -> f64 {
    if pair.ldet.parent == pair.gdet.parent {
        calculate_s_pair_orthogonal(pair.ldet, pair.gdet)
    } else if data.input.wicks.enabled {
        calculate_s_pair_wicks(pair.ldet, pair.gdet, data.wicks.unwrap(), scratch.unwrap())
    } else {
        calculate_s_pair_naive(data.ao, pair.ldet, pair.gdet, data.tol)
    }
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma using 
/// standard Slater-Condon rules.
/// # Arguments:
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// # Returns:
/// - `f64`: Overlap matrix element between `ldet` and `gdet`.
pub(in crate::noci) fn calculate_s_pair_orthogonal(ldet: &SCFState, gdet: &SCFState) -> f64 {
    if ldet.oa == gdet.oa && ldet.ob == gdet.ob {
        (ldet.pha * gdet.pha) * (ldet.phb * gdet.phb)
    } else {
        0.0
    }
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma using 
/// generalised Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data. 
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// - `tol`: Tolerance for a number being zero. 
/// # Returns:
/// - `f64`: Overlap matrix element between `ldet` and `gdet`.
fn calculate_s_pair_naive(ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64) -> f64 {

    // Per spin occupid coefficients.
    let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

    // Overlap matrix element for this pair. 
    pa.s * pb.s
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma
/// using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// - `wicks`: View to the intermediates required for non-orthogonal Wick's theorem.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `f64`: Overlap matrix element.
fn calculate_s_pair_wicks(ldet: &SCFState, gdet: &SCFState, wicks: &WicksView, scratch: &mut WickScratchSpin) -> f64 {
    let lp = ldet.parent;
    let gp = gdet.parent;

    let w = &wicks.pair(lp, gp);

    let ex_la = &ldet.excitation.alpha;
    let ex_ga = &gdet.excitation.alpha;
    let ex_lb = &ldet.excitation.beta;
    let ex_gb = &gdet.excitation.beta;

    let pha = ldet.pha * gdet.pha;
    let phb = ldet.phb * gdet.phb;

    prepare_same(&w.aa, ex_la, ex_ga, &mut scratch.aa);
    let sa = pha * lg_overlap(&w.aa, ex_la, ex_ga, &mut scratch.aa);
    prepare_same(&w.bb, ex_lb, ex_gb, &mut scratch.bb);
    let sb = phb * lg_overlap(&w.bb, ex_lb, ex_gb, &mut scratch.bb);
    sa * sb
}

