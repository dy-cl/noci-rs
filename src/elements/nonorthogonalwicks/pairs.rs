// elements/nonorthogonalwicks/pairs.rs
//! Determinant-pair matrix elements from the extended nonorthogonal Wick theorem.

// Crate-root imports.
use crate::determinant::{NOCIIndex, NOCISpace};
use crate::time_call;
use crate::{AoData, Excitation, NOCIScalar};

// Parent/sibling imports.
use super::{
    WickScratchSpin, WicksPairView, WicksView, xw_f_overlap_prepared,
    xw_hamiltonian_overlap_prepared, xw_overlap_prepared,
};

/// Calculate the overlap matrix element between determinants x and w
/// using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `wicks`: View to the intermediates required for non-orthogonal Wick's theorem.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `T`: Overlap matrix element.
pub(in crate::elements) fn calculate_s_pair_wicks<T: NOCIScalar>(
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

        // A zero-overlap count larger than the available excitation rank
        // makes the corresponding Wick determinant vanish.
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

        // The determinant product factorises into `\alpha` and `\beta`
        // same-spin overlap contributions.
        sa * sb
    })
}

/// Calculate the alpha same-spin overlap for an ordered Wick pair.
/// # Arguments:
/// - `ldet`: Left determinant.
/// - `gdet`: Right determinant.
/// - `w`: Wick intermediates for the ordered parent pair.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `T`: Alpha same-spin overlap including determinant phases.
#[inline(always)]
pub(crate) fn calculate_s_alpha_pair_wicks<T: NOCIScalar>(
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
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `T`: Beta same-spin overlap including determinant phases.
#[inline(always)]
pub(crate) fn calculate_s_beta_pair_wicks<T: NOCIScalar>(
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

/// Calculate both the Hamiltonian and overlap matrix elements between
/// determinants x and w using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Precomputed Wick's intermediates.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements for the pair.
pub(crate) fn calculate_hs_pair_wicks<T: NOCIScalar>(
    ao: &AoData,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    tol: f64,
    wicks: &WicksView<T>,
    scratch: &mut WickScratchSpin<T>,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair_wicks, {
        let left = space.reduced(ldet);
        let right = space.reduced(gdet);
        let w = wicks.pair(space.state(left.det).parent, space.state(right.det).parent);
        let excitation_phase = left.state.phase * right.state.phase;
        let (la, lb) = space.excitations(ldet);
        let (ga, gb) = space.excitations(gdet);
        let lex = Excitation {
            alpha: *la,
            beta: *lb,
        };
        let gex = Excitation {
            alpha: *ga,
            beta: *gb,
        };
        let lc = left.state.excitation_cache;
        let gc = right.state.excitation_cache;

        xw_hamiltonian_overlap_prepared(
            &w,
            (&lex, &gex),
            (&lc, &gc),
            excitation_phase,
            ao.enuc,
            scratch,
            tol,
        )
    })
}

/// Calculate the Fock matrix element between determinants x and w
/// using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Precomputed Wick's intermediates.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `T`: Fock matrix element between the determinant pair.
pub(in crate::elements) fn calculate_f_pair_wicks<T: NOCIScalar>(
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    tol: f64,
    wicks: &WicksView<T>,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_f_pair_wicks, {
        let lp = space.state(ldet).parent;
        let gp = space.state(gdet).parent;

        let w = &wicks.pair(lp, gp);

        let (ex_la, ex_lb) = space.excitations(ldet);
        let (ex_ga, ex_gb) = space.excitations(gdet);

        let pha = <T as From<f64>>::from(
            space.alpha(ldet).reduced.phase * space.alpha(gdet).reduced.phase,
        );
        let phb =
            <T as From<f64>>::from(space.beta(ldet).reduced.phase * space.beta(gdet).reduced.phase);

        // The one-body Fock element separates into spin sectors as
        // `F_{xw} = F^\alpha_{xw} S^\beta_{xw} + S^\alpha_{xw} F^\beta_{xw}`.
        let (sa, f1a) = xw_f_overlap_prepared(&w.aa, ex_la, ex_ga, &mut scratch.aa, tol);
        let (sb, f1b) = xw_f_overlap_prepared(&w.bb, ex_lb, ex_gb, &mut scratch.bb, tol);
        let sa = pha * sa;
        let sb = phb * sb;

        if sa.abs() == 0.0 && sb.abs() == 0.0 {
            return <T as From<f64>>::from(0.0);
        }

        pha * f1a * sb + phb * f1b * sa
    })
}

/// Calculate the shifted candidate-candidate matrix element using extended
/// non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ldet`: State `a`.
/// - `gdet`: State `b`.
/// - `tol`: Tolerance for a number being zero.
/// - `wicks`: Precomputed Wick's intermediates.
/// - `e0`: Zeroth-order energy shift.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `T`: Shifted matrix element `M_{ab}`.
pub(in crate::elements) fn calculate_m_pair_wicks<T: NOCIScalar>(
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    tol: f64,
    wicks: &WicksView<T>,
    e0: f64,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    let lp = space.state(ldet).parent;
    let gp = space.state(gdet).parent;
    let w = wicks.pair(lp, gp);

    let (ex_la, ex_lb) = space.excitations(ldet);
    let (ex_ga, ex_gb) = space.excitations(gdet);

    let pha =
        <T as From<f64>>::from(space.alpha(ldet).reduced.phase * space.alpha(gdet).reduced.phase);
    let phb =
        <T as From<f64>>::from(space.beta(ldet).reduced.phase * space.beta(gdet).reduced.phase);

    // Prepared Wick contractions supply spin overlaps and Fock elements;
    // combine them as `M_{ab} = F_{ab} - E_0 S_{ab}`.
    let (sa, f1a) = xw_f_overlap_prepared(&w.aa, ex_la, ex_ga, &mut scratch.aa, tol);
    let (sb, f1b) = xw_f_overlap_prepared(&w.bb, ex_lb, ex_gb, &mut scratch.bb, tol);
    let sa = pha * sa;
    let sb = phb * sb;
    let f = pha * f1a * sb + phb * f1b * sa;

    f - <T as From<f64>>::from(e0) * sa * sb
}
