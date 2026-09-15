// noci/hs.rs
// Crate-root imports.
use crate::basis::{excitation_between, excitation_phase_bits};
use crate::nonorthogonalwicks::{
    WickScratchSpin, WicksView, xw_hamiltonian_overlap_prepared,
    xw_hamiltonian_overlap_prepared_batched,
};
use crate::time_call;
use crate::{
    AoData, DetState, Excitation, ExcitationCache, ExcitationSpin, ExcitationSpinCache,
    ReducedTwoSpinState,
};

// Parent/sibling imports.
use super::factorise::{OrthogonalComponents, OrthogonalSpinComponent, SpinFactorisation};
use super::naive::{build_s_pair, occ_coeffs, one_electron, two_electron_diff, two_electron_same};
use super::orthogonal::{
    OrthogonalConnection, xw_hamiltonian_orthogonal_prepared,
    xw_hamiltonian_orthogonal_prepared_batched,
};
use super::overlap::calculate_s_pair_orthogonal;
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

/// Evaluate a compact batch `H_{D_kx_k}=\langle D_k^{P_k}|\hat H|\Phi_{x_k}^{P_k}\rangle`.
/// Requests are grouped by source parent and fixed-rank Slater-Condon sector, so each full SIMD
/// packet contains one double-excitation sector while `out` remains ordered by compact request.
/// # Arguments:
/// - `data`: Shared NOCI basis, AO data, and parent MO caches.
/// - `factorisation`: Canonical parent-local source component IDs.
/// - `components`: Prepared occupied and virtual labels for those canonical components.
/// - `connections`: Relative orthogonal connection topology.
/// - `pairs`: Compact `(source, connection)` requests in stochastic request order.
/// - `scratch`: Reusable source-parent and rank-sector grouping storage.
/// - `out`: Hamiltonian results in request order.
/// # Returns:
/// - `()`: Writes all parent-orthogonal Hamiltonian matrix elements into `out`.
pub(crate) fn calculate_h_pairs_orthogonal_batched(
    data: &NOCIData<'_, f64>,
    factorisation: &SpinFactorisation,
    components: &OrthogonalComponents,
    connections: &[OrthogonalConnection],
    pairs: &[(usize, usize)],
    scratch: &mut OrthogonalHamiltonianScratch,
    out: &mut [f64],
) {
    scratch.clear();
    for (output, &(source, connection)) in pairs.iter().enumerate() {
        let parent = data.basis[source].parent;
        scratch.groups[parent * 5 + connections[connection].sector()].push(output);
    }

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

    for (parent, cache) in mocache.iter().enumerate().take(factorisation.nparents()) {
        for sector in 0..5 {
            let outputs = &scratch.groups[parent * 5 + sector];
            if outputs.is_empty() {
                continue;
            }
            let mut start = 0usize;
            while sector >= 2 && width > 1 && start + width <= outputs.len() {
                let mut sources = [0usize; 8];
                let mut states = [ReducedTwoSpinState::new(1.0, ExcitationCache::default()); 8];
                let mut values = [0.0; 8];
                for lane in 0..width {
                    let output = outputs[start + lane];
                    let (source, connection) = pairs[output];
                    sources[lane] = source;
                    states[lane] = prepared_orthogonal_connection(
                        data,
                        factorisation,
                        components,
                        source,
                        connections[connection],
                    );
                }
                xw_hamiltonian_orthogonal_prepared_batched(
                    data.ao,
                    cache,
                    data.basis,
                    &sources[..width],
                    &states[..width],
                    &mut values[..width],
                );
                for lane in 0..width {
                    out[outputs[start + lane]] = values[lane];
                }
                start += width;
            }
            for &output in &outputs[start..] {
                let (source, connection) = pairs[output];
                let state = prepared_orthogonal_connection(
                    data,
                    factorisation,
                    components,
                    source,
                    connections[connection],
                );
                let source = &data.basis[source];
                out[output] = xw_hamiltonian_orthogonal_prepared(
                    data.ao,
                    cache,
                    (source.oa, source.ob),
                    &state,
                );
            }
        }
    }
}

/// Resolve an orthogonal connection directly into the fixed-rank kernel payload.
/// The connection supplies ranks and canonical source components supply orbital labels, so generic
/// excitation masks are formed only transiently for the existing fermionic phase expression.
/// # Arguments:
/// - `data`: Shared retained determinant basis.
/// - `factorisation`: Canonical source alpha and beta component IDs.
/// - `components`: Prepared parent-local occupied and virtual orbital labels.
/// - `source`: Retained source determinant index.
/// - `connection`: Relative orthogonal connection topology.
/// # Returns:
/// - `ReducedTwoSpinState`: Minimal phase and fixed-rank labels consumed by the H kernel.
#[inline(always)]
fn prepared_orthogonal_connection(
    data: &NOCIData<'_, f64>,
    factorisation: &SpinFactorisation,
    components: &OrthogonalComponents,
    source: usize,
    connection: OrthogonalConnection,
) -> ReducedTwoSpinState {
    let source_state = &data.basis[source];
    let (alpha, beta) = components.components(
        source_state.parent,
        factorisation.aid(source),
        factorisation.bid(source),
    );
    let cache = fixed_rank_connection(alpha, beta, connection);
    let phase = orthogonal_phase(alpha, cache.alpha) * orthogonal_phase(beta, cache.beta);
    ReducedTwoSpinState::new(phase, cache)
}

/// Construct fixed-rank cached labels from prepared source components and one connection.
/// # Arguments:
/// - `alpha`: Prepared alpha source-component orbital labels.
/// - `beta`: Prepared beta source-component orbital labels.
/// - `connection`: Relative orthogonal connection topology.
/// # Returns:
/// - `ExcitationCache`: Direct fixed-rank orbital labels for the orthogonal H kernel.
#[inline(always)]
fn fixed_rank_connection(
    alpha: &OrthogonalSpinComponent,
    beta: &OrthogonalSpinComponent,
    connection: OrthogonalConnection,
) -> ExcitationCache {
    let mut alpha_cache = ExcitationSpinCache::default();
    let mut beta_cache = ExcitationSpinCache::default();

    match connection {
        OrthogonalConnection::AlphaSingle { occupied, virtual_ } => {
            alpha_cache.rank = 1;
            alpha_cache.holes[0] = alpha.occupied[occupied as usize];
            alpha_cache.particles[0] = alpha.virtuals[virtual_ as usize];
        }
        OrthogonalConnection::BetaSingle { occupied, virtual_ } => {
            beta_cache.rank = 1;
            beta_cache.holes[0] = beta.occupied[occupied as usize];
            beta_cache.particles[0] = beta.virtuals[virtual_ as usize];
        }
        OrthogonalConnection::AlphaDouble { occupied, virtual_ } => {
            alpha_cache.rank = 2;
            alpha_cache.holes[0] = alpha.occupied[occupied[0] as usize];
            alpha_cache.holes[1] = alpha.occupied[occupied[1] as usize];
            alpha_cache.particles[0] = alpha.virtuals[virtual_[0] as usize];
            alpha_cache.particles[1] = alpha.virtuals[virtual_[1] as usize];
        }
        OrthogonalConnection::BetaDouble { occupied, virtual_ } => {
            beta_cache.rank = 2;
            beta_cache.holes[0] = beta.occupied[occupied[0] as usize];
            beta_cache.holes[1] = beta.occupied[occupied[1] as usize];
            beta_cache.particles[0] = beta.virtuals[virtual_[0] as usize];
            beta_cache.particles[1] = beta.virtuals[virtual_[1] as usize];
        }
        OrthogonalConnection::AlphaBetaDouble {
            occupied_a,
            virtual_a,
            occupied_b,
            virtual_b,
        } => {
            alpha_cache.rank = 1;
            alpha_cache.holes[0] = alpha.occupied[occupied_a as usize];
            alpha_cache.particles[0] = alpha.virtuals[virtual_a as usize];
            beta_cache.rank = 1;
            beta_cache.holes[0] = beta.occupied[occupied_b as usize];
            beta_cache.particles[0] = beta.virtuals[virtual_b as usize];
        }
    }

    ExcitationCache {
        alpha: alpha_cache,
        beta: beta_cache,
    }
}

/// Evaluate the existing fermionic phase expression from direct fixed-rank labels.
/// # Arguments:
/// - `component`: Source-spin occupation and rank-to-orbital lookup metadata.
/// - `cache`: Fixed-rank hole and particle labels for one connection spin sector.
/// # Returns:
/// - `f64`: Fermionic phase of this spin-sector excitation.
#[inline(always)]
fn orthogonal_phase(
    component: &OrthogonalSpinComponent,
    cache: ExcitationSpinCache,
) -> f64 {
    excitation_phase_bits(
        component.occupation,
        fixed_rank_mask(cache.holes, cache.rank),
        fixed_rank_mask(cache.particles, cache.rank),
    )
}

/// Construct the physical child occupations of an orthogonal connection.
/// # Arguments:
/// - `data`: Shared retained determinant basis.
/// - `factorisation`: Canonical source alpha and beta component IDs.
/// - `components`: Prepared parent-local occupied and virtual orbital labels.
/// - `source`: Retained source determinant index.
/// - `connection`: Relative orthogonal connection topology.
/// # Returns:
/// - `(u128, u128)`: Alpha and beta occupations of the physical child determinant.
pub(crate) fn orthogonal_connection_child(
    data: &NOCIData<'_, f64>,
    factorisation: &SpinFactorisation,
    components: &OrthogonalComponents,
    source: usize,
    connection: OrthogonalConnection,
) -> (u128, u128) {
    let source_state = &data.basis[source];
    let (alpha, beta) = components.components(
        source_state.parent,
        factorisation.aid(source),
        factorisation.bid(source),
    );
    let cache = fixed_rank_connection(alpha, beta, connection);
    (
        (alpha.occupation & !fixed_rank_mask(cache.alpha.holes, cache.alpha.rank))
            | fixed_rank_mask(cache.alpha.particles, cache.alpha.rank),
        (beta.occupation & !fixed_rank_mask(cache.beta.holes, cache.beta.rank))
            | fixed_rank_mask(cache.beta.particles, cache.beta.rank),
    )
}

/// Form a spin-orbital bit mask from fixed-rank cached orbital labels.
/// # Arguments:
/// - `orbitals`: Cached hole or particle orbital labels.
/// - `rank`: Number of active labels.
/// # Returns:
/// - `u128`: Bit mask containing exactly the active orbital labels.
#[inline(always)]
fn fixed_rank_mask(
    orbitals: [u8; crate::MAXEXCIT],
    rank: u8,
) -> u128 {
    let mut bits = 0u128;
    for &orbital in orbitals.iter().take(usize::from(rank)) {
        bits |= 1u128 << orbital;
    }
    bits
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
