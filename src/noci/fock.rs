// noci/fock.rs
use ndarray::{Array2};

use crate::{AoData, SCFState};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use super::types::{DetPair, FockMOCache, NOCIData, FockData};
use crate::time_call;

use crate::basis::excitation_phase;
use crate::nonorthogonalwicks::{prepare_same, lg_overlap, lg_f};
use super::naive::{occ_coeffs, build_s_pair, one_electron};

/// Wrapper function which dispatches to Fock matrix-element evaluation routines depending on
/// user input and properties of the determinant pair involved. If the determinant pair have the
/// same parents we may use the standard Slater-Condon rules, if not we can either use generalised
/// Slater-Condon rules or extended non-orthogonal Wick's theorem to evaluate the matrix element.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pair`: Pair of determinants whose Fock matrix element is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `f64`: Fock matrix element between the determinant pair.
pub(crate) fn calculate_f_pair(data: &NOCIData<'_>, fock: &FockData<'_>, pair: DetPair<'_>, scratch: Option<&mut WickScratchSpin>) -> f64 {
    time_call!(crate::timers::noci::add_calculate_f_pair, {
        if pair.ldet.parent == pair.gdet.parent {
            calculate_f_pair_orthogonal(&fock.fock_mocache[pair.ldet.parent], pair.ldet, pair.gdet)
        } else if data.input.wicks.enabled {
            calculate_f_pair_wicks(pair.ldet, pair.gdet, data.tol, data.wicks.unwrap(), scratch.unwrap())
        } else {
            calculate_f_pair_naive(fock.fa, fock.fb, data.ao, pair.ldet, pair.gdet, data.tol)
        }
    })
}

/// Calculate the Fock matrix element between determinants \Lambda and \Gamma using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `cache`: MO-basis Fock cache for the shared parent determinant.
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// # Returns:
/// - `f64`: Fock matrix element between `ldet` and `gdet`.
fn calculate_f_pair_orthogonal(cache: &FockMOCache, ldet: &SCFState, gdet: &SCFState) -> f64 {
    time_call!(crate::timers::noci::add_calculate_f_pair_orthogonal, {
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
    })
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
fn calculate_f_pair_naive(fa: &Array2<f64>, fb: &Array2<f64>, ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64) -> f64 {
    time_call!(crate::timers::noci::add_calculate_f_pair_naive, {
        // Per spin occupid coefficients.
        let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
        let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
        let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
        let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

        pb.s * one_electron(fa, &pa) + pa.s * one_electron(fb, &pb)
    })
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
fn calculate_f_pair_wicks(ldet: &SCFState, gdet: &SCFState, tol: f64, wicks: &WicksView, scratch: &mut WickScratchSpin) -> f64 {
    time_call!(crate::timers::noci::add_calculate_f_pair_wicks, {
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
    })
}

