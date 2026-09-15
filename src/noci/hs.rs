// noci/hs.rs
// Crate-root imports.
use crate::basis::excitation_between;
use crate::nonorthogonalwicks::{
    WickScratchSpin, WicksView, xw_hamiltonian_overlap_prepared,
    xw_hamiltonian_overlap_prepared_batched,
};
use crate::time_call;
use crate::{AoData, DetState, Excitation, ExcitationSpin, ReducedTwoSpinState};

// Parent/sibling imports.
use super::naive::{build_s_pair, occ_coeffs, one_electron, two_electron_diff, two_electron_same};
use super::orthogonal::{
    xw_hamiltonian_orthogonal_prepared, xw_hamiltonian_orthogonal_prepared_batched,
};
use super::overlap::calculate_s_pair_orthogonal;
use super::types::{DetPair, MOCache, NOCIData, NOCIScalar};

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
    pair: DetPair<'_, T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair, {
        let ldet = pair.ldet;
        let gdet = pair.gdet;

        if ldet.parent == gdet.parent
            && let Some(mocache) = data.mocache
        {
            let cache = &mocache[ldet.parent];
            if cache.orthogonal_slater_condon {
                return calculate_hs_pair_orthogonal(data.ao, cache, ldet, gdet);
            }
        }

        if data.input.wicks.enabled {
            calculate_hs_pair_wicks(
                data.ao,
                ldet,
                gdet,
                data.tol,
                data.wicks.unwrap(),
                scratch.unwrap(),
            )
        } else {
            calculate_hs_pair_naive(data.ao, ldet, gdet, data.tol)
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
/// - `reduced_basis`: Compact two-spin metadata keyed by global determinant index.
/// - `scratch`: Reusable Wick workspace for generic-rank evaluation.
/// - `out`: Hamiltonian and overlap results in the same order as `pairs`.
/// # Returns:
/// - `()`: Writes every requested `(H, S)` pair into `out`.
pub(crate) fn calculate_hs_pairs_wicks_batched(
    data: &NOCIData<'_, f64>,
    pairs: &[(usize, usize)],
    reduced_basis: &[ReducedTwoSpinState],
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
        let ldet = &data.basis[a];
        let gdet = &data.basis[b];

        if ldet.parent == gdet.parent {
            if (ldet.oa ^ gdet.oa).count_ones() + (ldet.ob ^ gdet.ob).count_ones() > 4 {
                out[output] = (0.0, 0.0);
                continue;
            }

            if let Some(mocache) = data.mocache {
                let cache = &mocache[ldet.parent];
                if cache.orthogonal_slater_condon {
                    out[output] = calculate_hs_pair_orthogonal(data.ao, cache, ldet, gdet);
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
            (data.basis, reduced_basis),
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
pub(in crate::noci) fn compare_hs_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> ((T, T), (f64, f64)) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    let (hn, sn) = calculate_hs_pair_naive(data.ao, ldet, gdet, data.tol);
    let (hw, sw) =
        calculate_hs_pair_wicks(data.ao, ldet, gdet, data.tol, data.wicks.unwrap(), scratch);

    let hdiff = (hn - hw).abs();
    let sdiff = (sn - sw).abs();
    ((hw, sw), (hdiff + sdiff, f64::max(hdiff, sdiff)))
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants x and w using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `cache`: MO-basis one and two-electron integral cache for the shared parent determinant.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`.
fn calculate_hs_pair_orthogonal<T: NOCIScalar>(
    ao: &AoData,
    cache: &MOCache<T>,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
) -> (T, T) {
    let s = calculate_s_pair_orthogonal(ldet, gdet);
    let h = calculate_h_pair_orthogonal(ao, cache, (ldet.oa, ldet.ob), (gdet.oa, gdet.ob));
    (h, s)
}

/// Calculate an orthogonal-parent Hamiltonian matrix element using shared Slater-Condon rules.
/// # Arguments:
/// - `ao`: AO integrals and nuclear-repulsion energy.
/// - `cache`: MO-basis one- and two-electron integrals for the common parent.
/// - `l_occ`: Bra alpha and beta occupation bitstrings.
/// - `g_occ`: Ket alpha and beta occupation bitstrings.
/// # Returns:
/// - `T`: Hamiltonian matrix element between the occupation-defined determinants.
pub(crate) fn calculate_h_pair_orthogonal<T: NOCIScalar>(
    ao: &AoData,
    cache: &MOCache<T>,
    l_occ: (u128, u128),
    g_occ: (u128, u128),
) -> T {
    time_call!(crate::timers::noci::add_calculate_hs_pair_orthogonal, {
        let (alpha_holes, alpha_parts) = excitation_between(g_occ.0, l_occ.0);
        let (beta_holes, beta_parts) = excitation_between(g_occ.1, l_occ.1);
        let ra = alpha_holes.count_ones() as usize;
        let rb = beta_holes.count_ones() as usize;
        if alpha_parts.count_ones() as usize != ra
            || beta_parts.count_ones() as usize != rb
            || ra + rb > 2
        {
            return T::from_real(0.0);
        }

        let excitation = Excitation {
            alpha: ExcitationSpin {
                holes: alpha_holes,
                parts: alpha_parts,
            },
            beta: ExcitationSpin {
                holes: beta_holes,
                parts: beta_parts,
            },
        };
        let state = ReducedTwoSpinState::from_excitation(g_occ, &excitation);
        xw_hamiltonian_orthogonal_prepared(ao, cache, g_occ, &state)
    })
}

/// Evaluate a batch `H_{D_kx_k}=\langle D_k^{P_k}|\hat H|\Phi_{x_k}^{P_k}\rangle`.
/// Consecutive requests are grouped into parent-local runs before prepared scalar/SIMD evaluation,
/// preserving request order without allocating parent-indexed request tables.
/// # Arguments:
/// - `data`: Shared NOCI basis, AO data, and parent MO caches.
/// - `sources`: Retained source determinant indices in request order.
/// - `states`: Prepared source-relative excitation states in request order.
/// - `out`: Hamiltonian results in request order.
/// # Returns
/// - `()`: Writes all parent-orthogonal Hamiltonian matrix elements into `out`.
pub(crate) fn calculate_h_pairs_orthogonal_batched<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    sources: &[usize],
    states: &[ReducedTwoSpinState],
    out: &mut [T],
) {
    let mocache = data
        .mocache
        .expect("orthogonal Hamiltonian batching requires parent MO caches");
    let mut start = 0usize;

    while start < sources.len() {
        let parent = data.basis[sources[start]].parent;
        let mut end = start + 1;
        while end < sources.len() && data.basis[sources[end]].parent == parent {
            end += 1;
        }

        xw_hamiltonian_orthogonal_prepared_batched(
            data.ao,
            &mocache[parent],
            data.basis,
            &sources[start..end],
            &states[start..end],
            &mut out[start..end],
        );
        start = end;
    }
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants x and w
/// using generalised Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`.
pub(in crate::noci) fn calculate_hs_pair_naive<T: NOCIScalar>(
    ao: &AoData,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair_naive, {
        // Per spin occupid coefficients.
        let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
        let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
        let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
        let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

        // Overlap matrix element for this pair.
        let s = pa.s * pb.s;

        let hnuc = match (pa.zeros.len(), pb.zeros.len()) {
            (0, 0) => <T as From<f64>>::from(ao.enuc) * s,
            _ => <T as From<f64>>::from(0.0),
        };

        let h1a = one_electron(&ao.h, &pa);
        let h1b = one_electron(&ao.h, &pb);
        let h1 = pb.s * h1a + pa.s * h1b;

        let h2aa = pb.s * two_electron_same(&ao.eri_asym, &pa);
        let h2bb = pa.s * two_electron_same(&ao.eri_asym, &pb);
        let h2ab = two_electron_diff(&ao.eri_coul, &pa, &pb);
        let h2 = h2aa + h2bb + h2ab;

        ((hnuc + h1 + h2), s)
    })
}

/// Calculate both the Hamiltonian and overlap matrix elements between
/// determinants x and w using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Precomputed Wick's intermediates.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements for the pair.
pub(in crate::noci) fn calculate_hs_pair_wicks<T: NOCIScalar>(
    ao: &AoData,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    tol: f64,
    wicks: &WicksView<T>,
    scratch: &mut WickScratchSpin<T>,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair_wicks, {
        let w = wicks.pair(ldet.parent, gdet.parent);
        let excitation_phase = (ldet.pha * gdet.pha) * (ldet.phb * gdet.phb);

        xw_hamiltonian_overlap_prepared(
            &w,
            (&ldet.excitation, &gdet.excitation),
            (&ldet.excitation_cache, &gdet.excitation_cache),
            excitation_phase,
            ao.enuc,
            scratch,
            tol,
        )
    })
}
