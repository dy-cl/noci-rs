// noci/overlap.rs

// Crate-root imports.
use crate::nonorthogonalwicks::{WickScratchSpin, WicksPairView, WicksView, xw_overlap_prepared};
use crate::time_call;

// Parent/sibling imports.
use super::naive::{build_s_pair, occ_coeffs};
use super::space::{NOCIIndex, NOCISpace};
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
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair, {
        let ldet = data.space.state(pair.ldet);
        let gdet = data.space.state(pair.gdet);

        if ldet.parent == gdet.parent {
            let mocache = data
                .mocache
                .expect("Orthogonal overlap matrix elements require mocache.");
            if mocache[ldet.parent].orthogonal_slater_condon {
                return calculate_s_pair_orthogonal(data.space, pair.ldet, pair.gdet);
            }
        }

        if data.input.wicks.enabled {
            calculate_s_pair_wicks(
                data.space,
                pair.ldet,
                pair.gdet,
                data.wicks.unwrap(),
                scratch.unwrap(),
            )
        } else {
            calculate_s_pair_naive(data, pair.ldet, pair.gdet)
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
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_orthogonal, {
        if space.occupations(ldet) == space.occupations(gdet) {
            <T as From<f64>>::from(space.phase(ldet) * space.phase(gdet))
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
    ldet: NOCIIndex,
    gdet: NOCIIndex,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_naive, {
        let lp = data.space.parent(ldet);
        let gp = data.space.parent(gdet);
        let (loa, lob) = data.space.occupations(ldet);
        let (goa, gob) = data.space.occupations(gdet);

        let l_ca_occ = occ_coeffs(&lp.ca, loa);
        let g_ca_occ = occ_coeffs(&gp.ca, goa);
        let l_cb_occ = occ_coeffs(&lp.cb, lob);
        let g_cb_occ = occ_coeffs(&gp.cb, gob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

        let det_phase = <T as From<f64>>::from(data.space.phase(ldet) * data.space.phase(gdet));
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
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    wicks: &WicksView<T>,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_wicks, {
        let lparent = space.state(ldet).parent;
        let gparent = space.state(gdet).parent;
        let w = wicks.pair(lparent, gparent);

        let (ex_la, ex_lb) = space.excitations(ldet);
        let (ex_ga, ex_gb) = space.excitations(gdet);

        let la = ex_la.holes.count_ones() as usize + ex_ga.holes.count_ones() as usize;
        let lb = ex_lb.holes.count_ones() as usize + ex_gb.holes.count_ones() as usize;

        if w.aa.m > la || w.bb.m > lb {
            return <T as From<f64>>::from(0.0);
        }

        let zero = <T as From<f64>>::from(0.0);

        let sa = calculate_s_alpha_pair_wicks(space, ldet, gdet, &w, scratch);
        if sa == zero {
            return zero;
        }

        let sb = calculate_s_beta_pair_wicks(space, ldet, gdet, &w, scratch);
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
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    w: &WicksPairView<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    let l_alpha = space.alpha(ldet);
    let g_alpha = space.alpha(gdet);
    let phase = <T as From<f64>>::from(l_alpha.reduced.phase * g_alpha.reduced.phase);

    phase
        * xw_overlap_prepared(
            &w.aa,
            &l_alpha.excitation,
            &g_alpha.excitation,
            &mut scratch.aa,
        )
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
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    w: &WicksPairView<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    let l_beta = space.beta(ldet);
    let g_beta = space.beta(gdet);
    let phase = <T as From<f64>>::from(l_beta.reduced.phase * g_beta.reduced.phase);

    phase
        * xw_overlap_prepared(
            &w.bb,
            &l_beta.excitation,
            &g_beta.excitation,
            &mut scratch.bb,
        )
}
