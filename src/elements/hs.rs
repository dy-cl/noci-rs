// elements/hs.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::{DetPair, NOCIData};
use crate::time_call;

// Parent/sibling imports.
use super::naive::calculate_hs_pair_naive;
use super::nonorthogonalwicks::{
    WickScratchSpin, calculate_hs_pair_wicks, xw_hamiltonian_overlap_prepared_batched,
};
use super::orthogonal::{calculate_hs_pair_orthogonal, calculate_hs_pairs_orthogonal_batched};

/// Wrapper function which dispatches Hamiltonian and overlap matrix-element evaluation for a batch
/// of determinant pairs depending on user input and properties of each pair. If a determinant
/// pair has the same Hermitian-orthonormal parent we may use the standard Slater-Condon rules, if
/// not we can either use generalised Slater-Condon rules or extended non-orthogonal Wick's theorem
/// to evaluate the matrix element. A single pair is evaluated directly; larger batches group
/// same-parent pairs by parent and Wick pairs by ordered reference pair for the SIMD kernels.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pairs`: Pairs of determinants whose Hamiltonian and overlap elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `out`: Hamiltonian and overlap matrix elements in the same order as `pairs`.
/// # Returns:
/// - `()`: Writes every requested `(H, S)` pair into `out`.
pub(crate) fn calculate_hs_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pairs: &[DetPair],
    mut scratch: Option<&mut WickScratchSpin<T>>,
    out: &mut [(T, T)],
) {
    time_call!(crate::timers::noci::add_calculate_hs_pair, {
        // A single pair has no other requests to fill SIMD lanes, so evaluate it directly.
        if let [pair] = pairs {
            out[0] = calculate_hs_pair_single(data, *pair, scratch);
            return;
        }

        let zero = T::from_real(0.0);
        let wicks = data.wicks.filter(|_| data.input.wicks.enabled);
        let nref = wicks.map_or(0, |wicks| wicks.nref);
        let mut groups = vec![Vec::new(); data.space.parents.len()];
        let mut wick_groups: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); nref * nref];

        // Place every request into exactly one parent group or ordered reference-pair group, or
        // evaluate it immediately when neither batched path applies.
        for (output, &pair) in pairs.iter().enumerate() {
            let lp = data.space.state(pair.ldet).parent;
            let gp = data.space.state(pair.gdet).parent;

            if lp == gp {
                // A one- or two-body Hamiltonian cannot connect same-parent states differing by
                // more than two excitations (four occupation bits).
                let l_occ = data.space.occupations(pair.ldet);
                let g_occ = data.space.occupations(pair.gdet);
                if (l_occ.0 ^ g_occ.0).count_ones() + (l_occ.1 ^ g_occ.1).count_ones() > 4 {
                    out[output] = (zero, zero);
                    continue;
                }

                if let Some(mocache) = data.mocache
                    && mocache[lp].orthogonal_slater_condon
                {
                    groups[lp].push((output, pair));
                    continue;
                }
            }

            if wicks.is_some() {
                wick_groups[lp * nref + gp].push((output, pair.ldet.0, pair.gdet.0));
                continue;
            }

            out[output] = calculate_hs_pair_nonorthogonal(data, pair, scratch.as_deref_mut());
        }

        // Evaluate same-parent groups with the orthogonal Slater-Condon kernels.
        if let Some(mocache) = data.mocache {
            calculate_hs_pairs_orthogonal_batched(data.ao, mocache, data.space, &groups, out);
        }

        // Each nonempty Wick group contains only requests belonging to one WicksPairView.
        if let Some(wicks) = wicks {
            let scratch = scratch.unwrap();
            for (group, requests) in wick_groups.iter().enumerate() {
                if requests.is_empty() {
                    continue;
                }

                let w = wicks.pair(group / nref, group % nref);
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
    })
}

/// Evaluate one Hamiltonian and overlap pair through the orthogonal, Wick, or naive path.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose Hamiltonian and overlap elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between the determinant pair.
fn calculate_hs_pair_single<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, T) {
    let lp = data.space.state(pair.ldet).parent;
    let gp = data.space.state(pair.gdet).parent;

    if lp == gp
        && let Some(mocache) = data.mocache
    {
        let cache = &mocache[lp];
        if cache.orthogonal_slater_condon {
            return calculate_hs_pair_orthogonal(data.ao, cache, data.space, pair.ldet, pair.gdet);
        }
    }

    calculate_hs_pair_nonorthogonal(data, pair, scratch)
}

/// Evaluate one Hamiltonian and overlap pair with Wick's theorem or the generalised Slater-Condon
/// rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose Hamiltonian and overlap elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between the determinant pair.
fn calculate_hs_pair_nonorthogonal<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, T) {
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
