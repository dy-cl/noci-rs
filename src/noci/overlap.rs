// noci/overlap.rs

// Crate-root imports.
use crate::DetState;
use crate::nonorthogonalwicks::{WickScratchSpin, WicksPairView, WicksView, xw_overlap_prepared};
use crate::time_call;

// Parent/sibling imports.
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

/// Calculate the overlap matrix element between determinants x and w using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
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

/// Calculate the overlap matrix element between determinants x and w using
/// generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
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

/// Calculate the overlap matrix element between determinants x and w
/// using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
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

        let la = ex_la.holes.count_ones() as usize + ex_ga.holes.count_ones() as usize;
        let lb = ex_lb.holes.count_ones() as usize + ex_gb.holes.count_ones() as usize;

        if w.aa.m > la || w.bb.m > lb {
            return <T as From<f64>>::from(0.0);
        }

        let zero = <T as From<f64>>::from(0.0);

        let sa = calculate_s_alpha_pair_wicks(ldet, gdet, &w, scratch);
        if sa == zero {
            return zero;
        }

        let sb = calculate_s_beta_pair_wicks(ldet, gdet, &w, scratch);
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
/// - `T`: Alpha same-spin overlap including determinant phases.
#[inline(always)]
pub(in crate::noci) fn calculate_s_alpha_pair_wicks<T: NOCIScalar>(
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    w: &WicksPairView<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    let l_ex = &ldet.excitation.alpha;
    let g_ex = &gdet.excitation.alpha;
    let phase = <T as From<f64>>::from(ldet.pha * gdet.pha);

    phase * xw_overlap_prepared(&w.aa, l_ex, g_ex, &mut scratch.aa)
}

/// Calculate the beta same-spin overlap for an ordered Wick pair.
/// # Arguments:
/// - `ldet`: Left determinant.
/// - `gdet`: Right determinant.
/// - `w`: Wick intermediates for the ordered parent pair.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Beta same-spin overlap including determinant phases.
#[inline(always)]
pub(in crate::noci) fn calculate_s_beta_pair_wicks<T: NOCIScalar>(
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    w: &WicksPairView<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    let l_ex = &ldet.excitation.beta;
    let g_ex = &gdet.excitation.beta;
    let phase = <T as From<f64>>::from(ldet.phb * gdet.phb);

    phase * xw_overlap_prepared(&w.bb, l_ex, g_ex, &mut scratch.bb)
}
