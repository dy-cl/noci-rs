// noci/m.rs
// Crate-root imports.
use crate::basis::excitation_phase;
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView, xw_f_overlap_prepared};

// Parent/sibling imports.
use super::naive::{build_s_pair, occ_coeffs, one_electron_scalar};
use super::space::{NOCIIndex, NOCISpace};
use super::types::{DetPair, FockData, FockMOCache, NOCIData, NOCIScalar};

/// Calculate the shifted candidate-candidate matrix element
/// `M_{ab} = F_{ab} - E0 S_{ab}` without evaluating `F_{ab}` and `S_{ab}` separately.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pair`: Pair of determinants whose shifted matrix element is to be evaluated.
/// - `e0`: Zeroth-order energy shift.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Shifted matrix element `M_{ab}`.
pub(crate) fn calculate_m_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pair: DetPair,
    e0: f64,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    let lp = data.space.state(ldet).parent;
    let gp = data.space.state(gdet).parent;
    if lp == gp {
        let cache = &fock.fock_mocache[lp];
        if cache.orthogonal_slater_condon {
            return calculate_m_pair_orthogonal(cache, data.space, ldet, gdet, e0);
        }
    }

    if data.input.wicks.enabled {
        calculate_m_pair_wicks(
            data.space,
            ldet,
            gdet,
            data.tol,
            data.wicks.unwrap(),
            e0,
            scratch.unwrap(),
        )
    } else {
        calculate_m_pair_naive(fock, data, ldet, gdet, e0)
    }
}

/// Calculate the shifted candidate-candidate matrix element between determinants
/// with the same parent using standard Slater-Condon rules.
/// # Arguments:
/// - `cache`: MO-basis Fock cache for the shared parent determinant.
/// - `ldet`: State `a`.
/// - `gdet`: State `b`.
/// - `e0`: Zeroth-order energy shift.
/// # Returns:
/// - `T`: Shifted matrix element `M_{ab}`.
fn calculate_m_pair_orthogonal<T: NOCIScalar>(
    cache: &FockMOCache<T>,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    e0: f64,
) -> T {
    let (loa, lob) = space.occupations(ldet);
    let (goa, gob) = space.occupations(gdet);
    let xa = loa ^ goa;
    let xb = lob ^ gob;
    let na = xa.count_ones() as usize;
    let nb = xb.count_ones() as usize;

    if na == 0 && nb == 0 {
        let mut f = <T as From<f64>>::from(0.0);

        let mut bits = goa;
        while bits != 0 {
            let p = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f += cache.fa[(p, p)];
        }

        let mut bits = gob;
        while bits != 0 {
            let p = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f += cache.fb[(p, p)];
        }

        let s = <T as From<f64>>::from(space.phase(ldet) * space.phase(gdet));
        return f - <T as From<f64>>::from(e0) * s;
    }

    if na == 2 && nb == 0 {
        let hole = (goa & xa).trailing_zeros() as usize;
        let part = (loa & xa).trailing_zeros() as usize;
        let phase = <T as From<f64>>::from(excitation_phase(goa, &[hole], &[part]));
        return phase * cache.fa[(part, hole)];
    }

    if na == 0 && nb == 2 {
        let hole = (gob & xb).trailing_zeros() as usize;
        let part = (lob & xb).trailing_zeros() as usize;
        let phase = <T as From<f64>>::from(excitation_phase(gob, &[hole], &[part]));
        return phase * cache.fb[(part, hole)];
    }

    <T as From<f64>>::from(0.0)
}

/// Calculate the shifted candidate-candidate matrix element using generalised
/// Slater-Condon rules.
/// # Arguments:
/// - `fock`: Spin-resolved Fock matrices in the AO basis.
/// - `data`: Authoritative NOCI space, AO overlap and numerical tolerance.
/// - `ldet`: State `a`.
/// - `gdet`: State `b`.
/// - `e0`: Zeroth-order energy shift.
/// # Returns:
/// - `T`: Shifted matrix element `M_{ab}`.
fn calculate_m_pair_naive<T: NOCIScalar>(
    fock: &FockData<'_, T>,
    data: &NOCIData<'_, T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    e0: f64,
) -> T {
    let space = data.space;
    let lp = space.parent(ldet);
    let gp = space.parent(gdet);
    let (loa, lob) = space.occupations(ldet);
    let (goa, gob) = space.occupations(gdet);

    let l_ca_occ = occ_coeffs(&lp.ca, loa);
    let g_ca_occ = occ_coeffs(&gp.ca, goa);
    let l_cb_occ = occ_coeffs(&lp.cb, lob);
    let g_cb_occ = occ_coeffs(&gp.cb, gob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    let s = pa.s * pb.s;
    let f = pb.s * one_electron_scalar(fock.fa, &pa) + pa.s * one_electron_scalar(fock.fb, &pb);

    f - <T as From<f64>>::from(e0) * s
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
/// # Returns:
/// - `T`: Shifted matrix element `M_{ab}`.
fn calculate_m_pair_wicks<T: NOCIScalar>(
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

    let (sa, f1a) = xw_f_overlap_prepared(&w.aa, ex_la, ex_ga, &mut scratch.aa, tol);
    let (sb, f1b) = xw_f_overlap_prepared(&w.bb, ex_lb, ex_gb, &mut scratch.bb, tol);
    let sa = pha * sa;
    let sb = phb * sb;
    let f = pha * f1a * sb + phb * f1b * sa;

    f - <T as From<f64>>::from(e0) * sa * sb
}
