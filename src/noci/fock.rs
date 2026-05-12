// noci/fock.rs
use ndarray::Array2;

use super::types::{DetPair, FockData, FockMOCache, NOCIData, NOCIScalar};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use crate::time_call;
use crate::{AoData, DetState};

use super::naive::{build_s_pair, occ_coeffs, one_electron_scalar};
use crate::basis::excitation_phase;
use crate::nonorthogonalwicks::{lg_f, lg_overlap, prepare_same};

/// Wrapper function which dispatches to Fock matrix-element evaluation routines depending on
/// user input and properties of the determinant pair involved. If the determinant pair have the
/// same Hermitian-orthonormal parents we may use the standard Slater-Condon rules, if not we can
/// either use generalised Slater-Condon rules or extended non-orthogonal Wick's theorem to evaluate
/// the matrix element.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pair`: Pair of determinants whose Fock matrix element is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Fock matrix element between the determinant pair.
pub(crate) fn calculate_f_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_f_pair, {
        let ldet = pair.ldet;
        let gdet = pair.gdet;

        if ldet.parent == gdet.parent {
            let cache = &fock.fock_mocache[ldet.parent];
            if cache.hermitian_orthonormal {
                return calculate_f_pair_orthogonal(cache, ldet, gdet);
            }
        }

        if data.input.wicks.enabled {
            calculate_f_pair_wicks(ldet, gdet, data.tol, data.wicks.unwrap(), scratch.unwrap())
        } else {
            calculate_f_pair_naive(fock.fa, fock.fb, data.ao, ldet, gdet, data.tol)
        }
    })
}

/// Compare naive and Wick's calculation of Fock matrix elements to ensure consistency.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pair`: Pair of determinants whose Fock matrix element is to be compared.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, f64)`: Wick's Fock matrix element, and the absolute discrepancy from the naive path.
pub(in crate::noci) fn compare_f_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> (T, f64) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    let fnv = calculate_f_pair_naive(fock.fa, fock.fb, data.ao, ldet, gdet, data.tol);
    let fw = calculate_f_pair_wicks(ldet, gdet, data.tol, data.wicks.unwrap(), scratch);

    (fw, (fnv - fw).abs())
}

/// Calculate the Fock matrix element between determinants \Lambda and \Gamma using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `cache`: MO-basis Fock cache for the shared parent determinant.
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// # Returns:
/// - `T`: Fock matrix element between `ldet` and `gdet`.
fn calculate_f_pair_orthogonal<T: NOCIScalar>(
    cache: &FockMOCache<T>,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_f_pair_orthogonal, {
        let xa = ldet.oa ^ gdet.oa;
        let xb = ldet.ob ^ gdet.ob;

        let na = xa.count_ones() as usize;
        let nb = xb.count_ones() as usize;

        if na == 0 && nb == 0 {
            let mut f = <T as From<f64>>::from(0.0);

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
/// - `T`: Fock matrix element between `ldet` and `gdet`.
fn calculate_f_pair_naive<T: NOCIScalar>(
    fa: &Array2<T>,
    fb: &Array2<T>,
    ao: &AoData,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    tol: f64,
) -> T {
    time_call!(crate::timers::noci::add_calculate_f_pair_naive, {
        // Per spin occupid coefficients.
        let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
        let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
        let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
        let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

        let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));

        det_phase * (pb.s * one_electron_scalar(fa, &pa) + pa.s * one_electron_scalar(fb, &pb))
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
/// - `T`: Fock matrix element between the determinant pair.
fn calculate_f_pair_wicks<T: NOCIScalar>(
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    tol: f64,
    wicks: &WicksView<T>,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_f_pair_wicks, {
        let lp = ldet.parent;
        let gp = gdet.parent;

        let w = &wicks.pair(lp, gp);

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

        if sa.abs() == 0.0 && sb.abs() == 0.0 {
            return <T as From<f64>>::from(0.0);
        }

        let mut f = <T as From<f64>>::from(0.0);
        if sb.abs() != 0.0 {
            let f1a = lg_f(&w.aa, ex_la, ex_ga, &mut scratch.aa, tol);
            f += pha * f1a * sb;
        }
        if sa.abs() != 0.0 {
            let f1b = lg_f(&w.bb, ex_lb, ex_gb, &mut scratch.bb, tol);
            f += phb * f1b * sa;
        }
        f
    })
}
