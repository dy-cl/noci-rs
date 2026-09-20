// noci/hs.rs
// Crate-root imports.
use crate::basis::excitation_between;
use crate::nonorthogonalwicks::{
    WickScratchSpin, WicksView, xw_hamiltonian_overlap_prepared,
    xw_hamiltonian_overlap_prepared_batched,
};
use crate::time_call;
use crate::{AoData, Excitation, ExcitationCache, ExcitationSpin, ReducedTwoSpinState};

// Parent/sibling imports.
use super::naive::{build_s_pair, occ_coeffs, one_electron, two_electron_diff, two_electron_same};
use super::orthogonal::{
    OrthogonalConnection, xw_hamiltonian_orthogonal_prepared,
    xw_hamiltonian_orthogonal_prepared_batched,
};
use super::overlap::calculate_s_pair_orthogonal;
use super::space::{NOCIIndex, NOCISpace};
use super::types::{DetPair, MOCache, NOCIData, NOCIScalar};

/// Reusable parent-and-sector grouping storage for compact orthogonal Hamiltonian requests.
pub(crate) struct OrthogonalHamiltonianScratch {
    /// Original request positions grouped by source parent and fixed-rank numerical sector.
    groups: Vec<Vec<usize>>,
}

impl OrthogonalHamiltonianScratch {
    /// Construct reusable source-parent and rank-sector request groups.
    /// # Arguments:
    /// - `nparents`: Number of source parent references.
    /// # Returns:
    /// - `Self`: Empty grouping storage with five numerical sectors per parent.
    pub(crate) fn new(nparents: usize) -> Self {
        Self {
            groups: (0..5 * nparents).map(|_| Vec::new()).collect(),
        }
    }

    /// Clear request groups while retaining their cycle-to-cycle allocation.
    /// # Arguments:
    /// - `self`: Reusable orthogonal numerical grouping storage.
    /// # Returns:
    /// - `()`: Removes all original request positions.
    fn clear(&mut self) {
        for group in &mut self.groups {
            group.clear();
        }
    }
}

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
pub(in crate::noci) fn compare_hs_pair_wicks_naive<T: NOCIScalar>(
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

/// Calculate both the overlap and Hamiltonian matrix elements between determinants x and w using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `cache`: MO-basis one and two-electron integral cache for the shared parent determinant.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`.
fn calculate_hs_pair_orthogonal<T: NOCIScalar>(
    ao: &AoData,
    cache: &MOCache<T>,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
) -> (T, T) {
    let s = calculate_s_pair_orthogonal(space, ldet, gdet);
    let h =
        calculate_h_pair_orthogonal(ao, cache, space.occupations(ldet), space.occupations(gdet));
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
        // Determine the spin-resolved excitation taking the ket occupation into the bra.
        let (alpha_holes, alpha_parts) = excitation_between(g_occ.0, l_occ.0);
        let (beta_holes, beta_parts) = excitation_between(g_occ.1, l_occ.1);
        let ra = alpha_holes.count_ones() as usize;
        let rb = beta_holes.count_ones() as usize;
        // Particle-number changes and excitation ranks above two have zero Hamiltonian coupling.
        if alpha_parts.count_ones() as usize != ra
            || beta_parts.count_ones() as usize != rb
            || ra + rb > 2
        {
            return T::from_real(0.0);
        }

        // Convert the valid connection to the reduced Slater-Condon representation.
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

/// Evaluate a compact batch `H_{D_kx_k}=\langle D_k^{P_k}|\hat H|\Phi_{x_k}^{P_k}\rangle`.
/// Requests are grouped by source parent and fixed-rank Slater-Condon sector, so each full SIMD
/// packet contains one double-excitation sector while `out` remains ordered by compact request.
/// # Arguments:
/// - `data`: Shared NOCI basis, AO data, and parent MO caches.
/// - `connections`: Relative orthogonal connection topology.
/// - `pairs`: Compact `(source, connection)` requests in stochastic request order.
/// - `scratch`: Reusable source-parent and rank-sector grouping storage.
/// - `out`: Hamiltonian results in request order.
/// # Returns:
/// - `()`: Writes all parent-orthogonal Hamiltonian matrix elements into `out`.
pub(crate) fn calculate_h_pairs_orthogonal_batched(
    data: &NOCIData<'_, f64>,
    connections: &[OrthogonalConnection],
    pairs: &[(NOCIIndex, usize)],
    scratch: &mut OrthogonalHamiltonianScratch,
    out: &mut [f64],
) {
    // Group output positions by parent and Slater-Condon sector without reordering `out`.
    scratch.clear();
    for (output, &(source, connection)) in pairs.iter().enumerate() {
        let parent = data.space.state(source).parent;
        scratch.groups[parent * 5 + connections[connection].sector()].push(output);
    }

    // Select the widest runtime-supported packet size for double-excitation kernels.
    let mocache = data
        .mocache
        .expect("orthogonal Hamiltonian batching requires parent MO caches");
    #[cfg(target_arch = "x86_64")]
    let width = if std::is_x86_feature_detected!("avx512f") {
        8
    } else if std::is_x86_feature_detected!("avx2") {
        4
    } else {
        1
    };
    #[cfg(not(target_arch = "x86_64"))]
    let width = 1;

    // Evaluate each parent/sector group using homogeneous SIMD packets and a scalar tail.
    for (parent, cache) in mocache.iter().enumerate().take(data.space.parents.len()) {
        for sector in 0..5 {
            let outputs = &scratch.groups[parent * 5 + sector];
            if outputs.is_empty() {
                continue;
            }
            let mut start = 0usize;
            // Only double sectors use the prepared vector kernels; singles remain scalar.
            while sector >= 2 && width > 1 && start + width <= outputs.len() {
                let mut occupations = [(0u128, 0u128); 8];
                let mut states = [ReducedTwoSpinState::new(1.0, ExcitationCache::default()); 8];
                let mut values = [0.0; 8];
                for lane in 0..width {
                    let output = outputs[start + lane];
                    let (source, connection) = pairs[output];
                    occupations[lane] = data.space.occupations(source);
                    states[lane] = connections[connection]
                        .reduced(data.space.alpha(source), data.space.beta(source));
                }
                // Evaluate one full packet, then scatter values back to request order.
                xw_hamiltonian_orthogonal_prepared_batched(
                    data.ao,
                    cache,
                    &occupations[..width],
                    &states[..width],
                    &mut values[..width],
                );
                for lane in 0..width {
                    out[outputs[start + lane]] = values[lane];
                }
                start += width;
            }
            // Handle singles and any incomplete SIMD packet with the scalar kernel.
            for &output in &outputs[start..] {
                let (source, connection) = pairs[output];
                let state = connections[connection]
                    .reduced(data.space.alpha(source), data.space.beta(source));
                out[output] = xw_hamiltonian_orthogonal_prepared(
                    data.ao,
                    cache,
                    data.space.occupations(source),
                    &state,
                );
            }
        }
    }
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants x and w
/// using generalised Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `space`: Determinant space containing the bra and ket states.
/// - `tol`: Numerical tolerance for the determinant overlap.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`.
pub(in crate::noci) fn calculate_hs_pair_naive<T: NOCIScalar>(
    ao: &AoData,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair_naive, {
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
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements for the pair.
pub(in crate::noci) fn calculate_hs_pair_wicks<T: NOCIScalar>(
    ao: &AoData,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    tol: f64,
    wicks: &WicksView<T>,
    scratch: &mut WickScratchSpin<T>,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair_wicks, {
        let left = space.reduced(ldet);
        let right = space.reduced(gdet);
        let w = wicks.pair(space.state(left.det).parent, space.state(right.det).parent);
        let excitation_phase = left.state.phase * right.state.phase;
        let (la, lb) = space.excitations(ldet);
        let (ga, gb) = space.excitations(gdet);
        let lex = Excitation {
            alpha: *la,
            beta: *lb,
        };
        let gex = Excitation {
            alpha: *ga,
            beta: *gb,
        };
        let lc = left.state.excitation_cache;
        let gc = right.state.excitation_cache;

        xw_hamiltonian_overlap_prepared(
            &w,
            (&lex, &gex),
            (&lc, &gc),
            excitation_phase,
            ao.enuc,
            scratch,
            tol,
        )
    })
}
