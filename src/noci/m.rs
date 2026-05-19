// noci/m.rs
use ndarray::Array2;

use crate::basis::excitation_phase;
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use crate::nonorthogonalwicks::{lg_f, lg_overlap, prepare_same};
use crate::{AoData, DetState};

use super::naive::{build_s_pair, occ_coeffs, one_electron_scalar};
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
    pair: DetPair<'_, T>,
    e0: f64,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    if ldet.parent == gdet.parent {
        let cache = &fock.fock_mocache[ldet.parent];
        if cache.orthogonal_slater_condon {
            return calculate_m_pair_orthogonal(cache, ldet, gdet, e0);
        }
    }

    if data.input.wicks.enabled {
        calculate_m_pair_wicks(
            ldet,
            gdet,
            data.tol,
            data.wicks.unwrap(),
            e0,
            scratch.unwrap(),
        )
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
/// - `T`: Shifted matrix element `M_{ab}`.
fn calculate_m_pair_orthogonal<T: NOCIScalar>(
    cache: &FockMOCache<T>,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    e0: f64,
) -> T {
    let xa = ldet.oa ^ gdet.oa;
    let xb = ldet.ob ^ gdet.ob;
    let na = xa.count_ones() as usize;
    let nb = xb.count_ones() as usize;

    if na == 0 && nb == 0 {
        let mut f = <T as From<f64>>::from(0.0);

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

        let s = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
        return f - <T as From<f64>>::from(e0) * s;
    }

    if na == 2 && nb == 0 {
        let hole = (gdet.oa & xa).trailing_zeros() as usize;
        let part = (ldet.oa & xa).trailing_zeros() as usize;
        let phase = <T as From<f64>>::from(excitation_phase(gdet.oa, &[hole], &[part]));
        return phase * cache.fa[(part, hole)];
    }

    if na == 0 && nb == 2 {
        let hole = (gdet.ob & xb).trailing_zeros() as usize;
        let part = (ldet.ob & xb).trailing_zeros() as usize;
        let phase = <T as From<f64>>::from(excitation_phase(gdet.ob, &[hole], &[part]));
        return phase * cache.fb[(part, hole)];
    }

    <T as From<f64>>::from(0.0)
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
/// - `T`: Shifted matrix element `M_{ab}`.
fn calculate_m_pair_naive<T: NOCIScalar>(
    fa: &Array2<T>,
    fb: &Array2<T>,
    ao: &AoData,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    tol: f64,
    e0: f64,
) -> T {
    let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

    let s = pa.s * pb.s;
    let f = pb.s * one_electron_scalar(fa, &pa) + pa.s * one_electron_scalar(fb, &pb);
    let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));

    det_phase * (f - <T as From<f64>>::from(e0) * s)
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
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    tol: f64,
    wicks: &WicksView<T>,
    e0: f64,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    let lp = ldet.parent;
    let gp = gdet.parent;
    let w = wicks.pair(lp, gp);

    let ex_la = &ldet.excitation.alpha;
    let ex_ga = &gdet.excitation.alpha;
    let ex_lb = &ldet.excitation.beta;
    let ex_gb = &gdet.excitation.beta;

    let pha = <T as From<f64>>::from(ldet.pha * gdet.pha);
    let phb = <T as From<f64>>::from(ldet.phb * gdet.phb);

    prepare_same(&w.aa, ex_la, ex_ga, &mut scratch.aa);
    let sa = pha * lg_overlap(&w.aa, ex_la, ex_ga, &mut scratch.aa);

    prepare_same(&w.bb, ex_lb, ex_gb, &mut scratch.bb);
    let sb = phb * lg_overlap(&w.bb, ex_lb, ex_gb, &mut scratch.bb);

    let mut f = <T as From<f64>>::from(0.0);

    if sb.abs() != 0.0 {
        f += pha * lg_f(&w.aa, ex_la, ex_ga, &mut scratch.aa, tol) * sb;
    }

    if sa.abs() != 0.0 {
        f += phb * lg_f(&w.bb, ex_lb, ex_gb, &mut scratch.bb, tol) * sa;
    }

    f - <T as From<f64>>::from(e0) * sa * sb
}
