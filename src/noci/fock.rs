// noci/fock.rs
// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::AoData;
use crate::basis::excitation_phase;
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView, xw_f_overlap_prepared};
use crate::time_call;

// Parent/sibling imports.
use super::naive::{build_s_pair, occ_coeffs, one_electron_scalar};
use super::space::{NOCIIndex, NOCISpace};
use super::types::{DetPair, FockData, FockMOCache, NOCIData, NOCIScalar};

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
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_f_pair, {
        let lp = data.space.state(pair.ldet).parent;
        let gp = data.space.state(pair.gdet).parent;
        if lp == gp {
            let cache = &fock.fock_mocache[lp];

            if cache.orthogonal_slater_condon {
                return calculate_f_pair_orthogonal(cache, data.space, pair.ldet, pair.gdet);
            }
        }

        if data.input.wicks.enabled {
            calculate_f_pair_wicks(
                data.space,
                pair.ldet,
                pair.gdet,
                data.tol,
                data.wicks.unwrap(),
                scratch.unwrap(),
            )
        } else {
            calculate_f_pair_naive(
                fock.fa, fock.fb, data.ao, data.space, pair.ldet, pair.gdet, data.tol,
            )
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
/// - `(T, (f64, f64))`: Wick's Fock matrix element, total discrepancy from
///   the naive path, and max elementwise discrepancy.
pub(in crate::noci) fn compare_f_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pair: DetPair,
    scratch: &mut WickScratchSpin<T>,
) -> (T, (f64, f64)) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    let fnv = calculate_f_pair_naive(fock.fa, fock.fb, data.ao, data.space, ldet, gdet, data.tol);
    let fw = calculate_f_pair_wicks(
        data.space,
        ldet,
        gdet,
        data.tol,
        data.wicks.unwrap(),
        scratch,
    );

    let diff = (fnv - fw).abs();
    (fw, (diff, diff))
}

/// Calculate the Fock matrix element between determinants x and w using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `cache`: MO-basis Fock cache for the shared parent determinant.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// # Returns:
/// - `T`: Fock matrix element between `ldet` and `gdet`.
pub(in crate::noci) fn calculate_f_pair_orthogonal<T: NOCIScalar>(
    cache: &FockMOCache<T>,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
) -> T {
    time_call!(crate::timers::noci::add_calculate_f_pair_orthogonal, {
        let (loa, lob) = space.occupations(ldet);
        let (goa, gob) = space.occupations(gdet);
        let xa = loa ^ goa;
        let xb = lob ^ gob;

        let na = xa.count_ones() as usize;
        let nb = xb.count_ones() as usize;

        if na == 0 && nb == 0 {
            let mut f = <T as From<f64>>::from(0.0);

            for p in 0..128 {
                if ((goa >> p) & 1) == 1 {
                    f += cache.fa[(p, p)];
                }
                if ((gob >> p) & 1) == 1 {
                    f += cache.fb[(p, p)];
                }
            }
            return f;
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
    })
}

/// Calculate the Fock matrix element between determinants x and w using
/// generalised Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `fa`: NOCI Fock matrix spin alpha.
/// - `fb`: NOCI Fock matrix spin beta.
/// # Returns:
/// - `T`: Fock matrix element between `ldet` and `gdet`.
fn calculate_f_pair_naive<T: NOCIScalar>(
    fa: &Array2<T>,
    fb: &Array2<T>,
    ao: &AoData,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    tol: f64,
) -> T {
    time_call!(crate::timers::noci::add_calculate_f_pair_naive, {
        // Per spin occupid coefficients.
        let lp = space.parent(ldet);
        let gp = space.parent(gdet);
        let (loa, lob) = space.occupations(ldet);
        let (goa, gob) = space.occupations(gdet);

        let l_ca_occ = occ_coeffs(&lp.ca, loa);
        let g_ca_occ = occ_coeffs(&gp.ca, goa);
        let l_cb_occ = occ_coeffs(&lp.cb, lob);
        let g_cb_occ = occ_coeffs(&gp.cb, gob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

        pb.s * one_electron_scalar(fa, &pa) + pa.s * one_electron_scalar(fb, &pb)
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
/// # Returns:
/// - `T`: Fock matrix element between the determinant pair.
fn calculate_f_pair_wicks<T: NOCIScalar>(
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
