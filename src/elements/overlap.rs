// elements/overlap.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::{DetPair, NOCIData};
use crate::time_call;

// Parent/sibling imports.
use super::naive::calculate_s_pair_naive;
use super::nonorthogonalwicks::{WickScratchSpin, calculate_s_pair_wicks};
use super::orthogonal::calculate_s_pair_orthogonal;

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
