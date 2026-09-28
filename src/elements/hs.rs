// elements/hs.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::NOCIIndex;
use crate::elements::{DetPair, NOCIData};
use crate::time_call;

// Parent/sibling imports.
use super::naive::calculate_hs_pair_naive;
use super::nonorthogonalwicks::{
    WickScratchSpin, calculate_hs_pair_wicks, xw_hamiltonian_overlap_prepared_batched,
};
use super::orthogonal::calculate_hs_pair_orthogonal;

/// Wrapper function which dispatches to Hamiltonian and overlap matrix-element evaluation routines
/// depending on user input and properties of the determinant pair involved. If the determinant
/// pair have the same Hermitian-orthonormal parents we may use the standard Slater-Condon rules,
/// if not we can either use generalised Slater-Condon rules or extended non-orthogonal Wick's
/// theorem to evaluate the matrix element.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose Hamiltonian and overlap matrix elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between the determinant pair.
pub(crate) fn calculate_hs_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair, {
        let ldet = data.space.state(pair.ldet);
        let gdet = data.space.state(pair.gdet);

        if ldet.parent == gdet.parent
            && let Some(mocache) = data.mocache
        {
            let cache = &mocache[ldet.parent];
            if cache.orthogonal_slater_condon {
                return calculate_hs_pair_orthogonal(
                    data.ao, cache, data.space, pair.ldet, pair.gdet,
                );
            }
        }

        if data.input.wicks.enabled {
            calculate_hs_pair_wicks(
                data.ao,
                data.space,
                pair.ldet,
                pair.gdet,
                data.tol,
                data.wicks.unwrap(),
                scratch.unwrap(),
            )
        } else {
            calculate_hs_pair_naive(data.ao, data.space, pair.ldet, pair.gdet, data.tol)
        }
    })
}

/// Calculate batched Hamiltonian and overlap matrix elements using extended nonorthogonal Wick's
/// theorem. The determinant pairs are canonically ordered before this routine is called.
/// Same-parent Slater-Condon cases are handled here; all remaining requests are grouped once by
/// ordered reference pair before CPU-specific rank batching is delegated to the Wick evaluator.
/// # Arguments:
/// - `data`: Shared real NOCI data with precomputed Wick intermediates.
/// - `pairs`: Canonically ordered determinant-index pairs `(a, b)` with `a <= b`.
/// - `scratch`: Reusable Wick workspace for generic-rank evaluation.
/// - `out`: Hamiltonian and overlap results in the same order as `pairs`.
/// # Returns:
/// - `()`: Writes every requested `(H, S)` pair into `out`.
pub(crate) fn calculate_hs_pairs_wicks_batched(
    data: &NOCIData<'_, f64>,
    pairs: &[(usize, usize)],
    scratch: &mut WickScratchSpin<f64>,
    out: &mut [(f64, f64)],
) {
    let wicks = data.wicks.unwrap();
    let ngroups = wicks.nref * wicks.nref;
    let group_capacity = pairs.len().div_ceil(ngroups);
    let mut groups: Vec<Vec<(usize, usize, usize)>> = (0..ngroups)
        .map(|_| Vec::with_capacity(group_capacity))
        .collect();

    // Resolve same-parent Slater-Condon cases and place every remaining request into exactly one
    // ordered reference-pair group. The Wick evaluator therefore never filters unrelated pairs.
    for (output, &(a, b)) in pairs.iter().enumerate() {
        let ldet = data.space.state(NOCIIndex(a));
        let gdet = data.space.state(NOCIIndex(b));
        let l_occ = data.space.occupations(NOCIIndex(a));
        let g_occ = data.space.occupations(NOCIIndex(b));

        if ldet.parent == gdet.parent {
            if (l_occ.0 ^ g_occ.0).count_ones() + (l_occ.1 ^ g_occ.1).count_ones() > 4 {
                out[output] = (0.0, 0.0);
                continue;
            }

            if let Some(mocache) = data.mocache {
                let cache = &mocache[ldet.parent];
                if cache.orthogonal_slater_condon {
                    out[output] = calculate_hs_pair_orthogonal(
                        data.ao,
                        cache,
                        data.space,
                        NOCIIndex(a),
                        NOCIIndex(b),
                    );
                    continue;
                }
            }
        }

        let pair = ldet.parent * wicks.nref + gdet.parent;
        groups[pair].push((output, a, b));
    }

    // Each nonempty group now contains only requests belonging to one WicksPairView.
    for (pair, requests) in groups.iter().enumerate() {
        if requests.is_empty() {
            continue;
        }

        let lp = pair / wicks.nref;
        let gp = pair % wicks.nref;
        let w = wicks.pair(lp, gp);
        xw_hamiltonian_overlap_prepared_batched(
            &w,
            (data.space, &data.space.reduced),
            requests,
            data.ao.enuc,
            scratch,
            data.tol,
            out,
        );
    }
}

/// Compare naive and Wick's calculation of matrix elements to ensure consistency.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose matrix elements are to be compared.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `((T, T), (f64, f64))`: Hamiltonian and overlap matrix elements between
///   the determinant pair, total discrepancy between the naive and Wick's path,
///   and max elementwise discrepancy.
pub(crate) fn compare_hs_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    scratch: &mut WickScratchSpin<T>,
) -> ((T, T), (f64, f64)) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    let (hn, sn) = calculate_hs_pair_naive(data.ao, data.space, ldet, gdet, data.tol);
    let (hw, sw) = calculate_hs_pair_wicks(
        data.ao,
        data.space,
        ldet,
        gdet,
        data.tol,
        data.wicks.unwrap(),
        scratch,
    );

    let hdiff = (hn - hw).abs();
    let sdiff = (sn - sw).abs();
    ((hw, sw), (hdiff + sdiff, f64::max(hdiff, sdiff)))
}
