// noci/m.rs
use ndarray::Array2;

use crate::{AoData, SCFState};
use crate::basis::excitation_phase;
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use crate::nonorthogonalwicks::{prepare_same, lg_overlap, lg_f};

use super::naive::{occ_coeffs, build_s_pair, one_electron};
use super::types::{DetPair, FockData, FockMOCache, NOCIData};

/// Calculate the shifted candidate-candidate matrix element
/// `M_{ab} = F_{ab} - E0 S_{ab}` without evaluating `F_{ab}` and `S_{ab}` separately.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pair`: Pair of determinants whose shifted matrix element is to be evaluated.
/// - `e0`: Zeroth-order energy shift.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `f64`: Shifted matrix element `M_{ab}`.
pub(crate) fn calculate_m_pair(data: &NOCIData<'_>, fock: &FockData<'_>, pair: DetPair<'_>, e0: f64, scratch: Option<&mut WickScratchSpin>) -> f64 {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    if ldet.parent == gdet.parent {
        calculate_m_pair_orthogonal(&fock.fock_mocache[ldet.parent], ldet, gdet, e0)
    } else if data.input.wicks.enabled {
        calculate_m_pair_wicks(ldet, gdet, data.tol, data.wicks.unwrap(), e0, scratch.unwrap())
    } else {
        calculate_m_pair_naive(fock.fa, fock.fb, data.ao, ldet, gdet, data.tol, e0)
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
/// - `f64`: Shifted matrix element `M_{ab}`.
fn calculate_m_pair_orthogonal(cache: &FockMOCache, ldet: &SCFState, gdet: &SCFState, e0: f64) -> f64 {
    let xa = ldet.oa ^ gdet.oa;
    let xb = ldet.ob ^ gdet.ob;
    let na = xa.count_ones() as usize;
    let nb = xb.count_ones() as usize;

    if na == 0 && nb == 0 {
        let mut f = 0.0;

        let mut bits = gdet.oa;
        while bits != 0 {
            let p = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f += cache.fa[(p, p)];
        }

        let mut bits = gdet.ob;
        while bits != 0 {
            let p = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f += cache.fb[(p, p)];
        }

        let s = (ldet.pha * gdet.pha) * (ldet.phb * gdet.phb);
        return f - e0 * s;
    }

    if na == 2 && nb == 0 {
        let hole = (gdet.oa & xa).trailing_zeros() as usize;
        let part = (ldet.oa & xa).trailing_zeros() as usize;
        let phase = excitation_phase(gdet.oa, &[hole], &[part]);
        return phase * cache.fa[(part, hole)];
    }

    if na == 0 && nb == 2 {
        let hole = (gdet.ob & xb).trailing_zeros() as usize;
        let part = (ldet.ob & xb).trailing_zeros() as usize;
        let phase = excitation_phase(gdet.ob, &[hole], &[part]);
        return phase * cache.fb[(part, hole)];
    }

    0.0
}

/// Calculate the shifted candidate-candidate matrix element using generalised
/// Slater-Condon rules.
/// # Arguments:
/// - `fa`: Spin-alpha Fock matrix in AO basis.
/// - `fb`: Spin-beta Fock matrix in AO basis.
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: State `a`.
/// - `gdet`: State `b`.
/// - `tol`: Tolerance for a number being zero.
/// - `e0`: Zeroth-order energy shift.
/// # Returns:
/// - `f64`: Shifted matrix element `M_{ab}`.
fn calculate_m_pair_naive(fa: &Array2<f64>, fb: &Array2<f64>, ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64, e0: f64) -> f64 {
    let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

    let s = pa.s * pb.s;
    let f = pb.s * one_electron(fa, &pa) + pa.s * one_electron(fb, &pb);

    f - e0 * s
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
/// - `f64`: Shifted matrix element `M_{ab}`.
fn calculate_m_pair_wicks(ldet: &SCFState, gdet: &SCFState, tol: f64, wicks: &WicksView, e0: f64, scratch: &mut WickScratchSpin) -> f64 {
    let lp = ldet.parent;
    let gp = gdet.parent;
    let w = wicks.pair(lp, gp);

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

    let mut f = 0.0;

    if sb != 0.0 {
        f += pha * lg_f(&w.aa, ex_la, ex_ga, &mut scratch.aa, tol) * sb;
    }

    if sa != 0.0 {
        f += phb * lg_f(&w.bb, ex_lb, ex_gb, &mut scratch.bb, tol) * sa;
    }

    f - e0 * sa * sb
}
