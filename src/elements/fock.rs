// elements/fock.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::{DetPair, FockData, NOCIData};
use crate::time_call;

// Parent/sibling imports.
use super::naive::calculate_f_pair_naive;
use super::nonorthogonalwicks::{WickScratchSpin, calculate_f_pair_wicks};
use super::orthogonal::calculate_f_pair_orthogonal;

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
pub(crate) fn compare_f_pair_wicks_naive<T: NOCIScalar>(
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
