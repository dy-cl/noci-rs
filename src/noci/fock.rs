// noci/fock.rs
use ndarray::{Array2};

use crate::{AoData, SCFState};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use crate::input::Input;
use super::cache::FockMOCache;

use crate::basis::excitation_phase;
use crate::nonorthogonalwicks::{prepare_same, lg_overlap, lg_f};
use super::naive::{occ_coeffs, build_s_pair, one_electron};

/// Wrapper function which dispatches to Fock matrix-element evaluation routines depending on user
/// input and properties of the determinant pair involved. If the determinant pair have the same
/// parents we may use the standard Slater-Condon rules, if not we can either use generalised
/// Slater-Condon rules or extended non-orthogonal Wick's theorem to evaluate the matrix element.
/// # Arguments:
/// - `fa`: Spin-alpha Fock matrix.
/// - `fb`: Spin-beta Fock matrix.
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// - `tol`: Tolerance for a number being zero.
/// - `input`: User defined input options.
/// - `fock_mocache`: MO-basis Fock integral caches.
/// - `wicks`: View to the intermediates required for non-orthogonal Wick's theorem.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `f64`: Fock matrix element between `ldet` and `gdet`.
pub fn calculate_f_pair(fa: &Array2<f64>, fb: &Array2<f64>, ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64, input: &Input, 
                        fock_mocache: &[FockMOCache], wicks: Option<&WicksView>, scratch: Option<&mut WickScratchSpin>) -> f64 {
    if ldet.parent == gdet.parent {
        calculate_f_pair_orthogonal(&fock_mocache[ldet.parent], ldet, gdet)
    } else if input.wicks.enabled {
        calculate_f_pair_wicks(ldet, gdet, tol, wicks.unwrap(), scratch.unwrap())
    } else {
        calculate_f_pair_naive(fa, fb, ao, ldet, gdet, tol)
    }
}

/// Calculate the Fock matrix element between determinants \Lambda and \Gamma using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `cache`: MO-basis Fock cache for the shared parent determinant.
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// # Returns:
/// - `f64`: Fock matrix element between `ldet` and `gdet`.
pub(crate) fn calculate_f_pair_orthogonal(cache: &FockMOCache, ldet: &SCFState, gdet: &SCFState) -> f64 {
    let xa = ldet.oa ^ gdet.oa;
    let xb = ldet.ob ^ gdet.ob;

    let na = xa.count_ones() as usize;
    let nb = xb.count_ones() as usize;

    if na == 0 && nb == 0 {
        let mut f = 0.0;

        for p in 0..128 {
            if ((gdet.oa >> p) & 1) == 1 {
                f += cache.fa[(p, p)];
            }
            if ((gdet.ob >> p) & 1) == 1 {
                f += cache.fb[(p, p)];
            }
        }
        return f;
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

/// Calculate the Fock matrix element between determinants \Lambda and \Gamma using 
/// generalised Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data. 
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// - `fa`: NOCI Fock matrix spin alpha.
/// - `fb`: NOCI Fock matrix spin beta.
/// # Returns:
/// - `f64`: Fock matrix element between `ldet` and `gdet`.
pub(crate) fn calculate_f_pair_naive(fa: &Array2<f64>, fb: &Array2<f64>, ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64) -> f64 {

    // Per spin occupid coefficients.
    let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

    pb.s * one_electron(fa, &pa) + pa.s * one_electron(fb, &pb)
}

/// Calculate the Fock matrix element between determinants \Lambda and \Gamma
/// using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Precomputed Wick's intermediates.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `f64`: Fock matrix element between the determinant pair.
pub(crate) fn calculate_f_pair_wicks(ldet: &SCFState, gdet: &SCFState, tol: f64, wicks: &WicksView, scratch: &mut WickScratchSpin) -> f64 {
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

    if sa == 0.0 && sb == 0.0 {
        return 0.0;
    }

    let mut f = 0.0;
    if sb != 0.0 {
        let f1a = lg_f(&w.aa, ex_la, ex_ga, &mut scratch.aa, tol);
        f += pha * f1a * sb;
    }
    if sa != 0.0 {
        let f1b = lg_f(&w.bb, ex_lb, ex_gb, &mut scratch.bb, tol);
        f += phb * f1b * sa;
    }
    f
}

