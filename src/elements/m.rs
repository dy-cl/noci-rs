// elements/m.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::{DetPair, FockData, NOCIData};

// Parent/sibling imports.
use super::naive::calculate_m_pair_naive;
use super::nonorthogonalwicks::{WickScratchSpin, calculate_m_pair_wicks};
use super::orthogonal::calculate_m_pair_orthogonal;

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
