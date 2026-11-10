// elements/orthogonal/pairs.rs
//! Determinant-pair matrix elements from the Slater–Condon rules in one parent's orthonormal
//! basis.

// Crate-root imports.
use crate::basis::excitation_between;
use crate::determinant::{NOCIIndex, NOCISpace, OrthogonalConnection};
use crate::elements::{DetPair, FockMOCache, NOCIData, PairDeterminant};
use crate::time_call;
use crate::{Excitation, ExcitationCache, ExcitationSpin, NOCIScalar, ReducedTwoSpinState};

// Parent/sibling imports.
use super::{
    xw_fock_orthogonal, xw_hamiltonian_orthogonal, xw_hamiltonian_orthogonal_batched,
    xw_overlap_orthogonal,
};

/// Calculate the overlap matrix element between determinants x and w using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `T`: Overlap matrix element between `ldet` and `gdet`.
pub(crate) fn calculate_s_pair_orthogonal<T: NOCIScalar>(
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_orthogonal, {
        let g_occ = space.occupations(gdet);
        match prepare_orthogonal_state(space.occupations(ldet), g_occ, 0) {
            Some(state) => xw_overlap_orthogonal(&state),
            None => T::from_real(0.0),
        }
    })
}

/// Prepare the fixed-rank orthogonal excitation taking a ket occupation into a bra occupation.
/// # Arguments:
/// - `l_occ`: Bra alpha and beta occupation bitstrings.
/// - `g_occ`: Ket alpha and beta occupation bitstrings.
/// - `rank`: Largest total excitation rank coupled by the operator.
/// # Returns:
/// - `Option<ReducedTwoSpinState>`: Prepared phase and labels, or `None` when the operator
///   cannot couple the occupations.
#[inline(always)]
pub(in crate::elements) fn prepare_orthogonal_state(
    l_occ: (u128, u128),
    g_occ: (u128, u128),
    rank: usize,
) -> Option<ReducedTwoSpinState> {
    // Determine the spin-resolved excitation taking the ket occupation into the bra.
    let (alpha_holes, alpha_parts) = excitation_between(g_occ.0, l_occ.0);
    let (beta_holes, beta_parts) = excitation_between(g_occ.1, l_occ.1);
    let ra = alpha_holes.count_ones() as usize;
    let rb = beta_holes.count_ones() as usize;
    // Particle-number changes and excitation ranks above the operator rank have zero coupling.
    if alpha_parts.count_ones() as usize != ra
        || beta_parts.count_ones() as usize != rb
        || ra + rb > rank
    {
        return None;
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
    Some(ReducedTwoSpinState::from_excitation(g_occ, &excitation))
}

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

/// Evaluate a compact batch `H_{D_kx_k}=\langle D_k^{P_k}|\hat H|\Phi_{x_k}^{P_k}\rangle`.
/// Requests are grouped by source parent and fixed-rank Slater-Condon sector, so each full SIMD
/// packet contains one double-excitation sector while `out` remains ordered by compact request.
/// # Arguments:
/// - `data`: Shared NOCI basis, AO data, and parent MO caches.
/// - `pairs`: Compact `(source, connection)` requests in stochastic request order.
/// - `scratch`: Reusable source-parent and rank-sector grouping storage.
/// - `out`: Hamiltonian results in request order.
/// # Returns:
/// - `()`: Writes all parent-orthogonal Hamiltonian matrix elements into `out`.
pub(crate) fn calculate_h_pairs_orthogonal_batched(
    data: &NOCIData<'_, f64>,
    pairs: &[(NOCIIndex, OrthogonalConnection)],
    scratch: &mut OrthogonalHamiltonianScratch,
    out: &mut [f64],
) {
    // Group output positions by parent and Slater-Condon sector without reordering `out`.
    scratch.clear();
    for (output, &(source, connection)) in pairs.iter().enumerate() {
        let parent = data.space.state(source).parent;
        scratch.groups[parent * 5 + connection.sector()].push(output);
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
            // Only double sectors use the fixed-rank vector kernels; singles remain scalar.
            while sector >= 2 && width > 1 && start + width <= outputs.len() {
                let mut occupations = [(0u128, 0u128); 8];
                let mut states = [ReducedTwoSpinState::new(1.0, ExcitationCache::default()); 8];
                let mut values = [0.0; 8];
                for lane in 0..width {
                    let output = outputs[start + lane];
                    let (source, connection) = pairs[output];
                    occupations[lane] = data.space.occupations(source);
                    states[lane] =
                        connection.reduced(data.space.alpha(source), data.space.beta(source));
                }
                // Evaluate one full packet, then scatter values back to request order.
                xw_hamiltonian_orthogonal_batched(
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
                let state = connection.reduced(data.space.alpha(source), data.space.beta(source));
                out[output] = xw_hamiltonian_orthogonal(
                    data.ao,
                    cache,
                    data.space.occupations(source),
                    &state,
                );
            }
        }
    }
}

/// Calculate the Fock matrix element between determinants x and w using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `cache`: MO-basis Fock cache for the shared parent determinant.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `T`: Fock matrix element between `ldet` and `gdet`.
pub(crate) fn calculate_f_pair_orthogonal<T: NOCIScalar>(
    cache: &FockMOCache<T>,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
) -> T {
    time_call!(crate::timers::noci::add_calculate_f_pair_orthogonal, {
        // The one-body Fock operator connects only identical determinants and single excitations
        // in one spin sector.
        let g_occ = space.occupations(gdet);
        match prepare_orthogonal_state(space.occupations(ldet), g_occ, 1) {
            Some(state) => xw_fock_orthogonal(cache, g_occ, &state),
            None => T::from_real(0.0),
        }
    })
}

/// Evaluate the same-parent requests of one orthonormal parent through one batched kernel.
/// Each request is converted once to its reduced fixed-rank excitation; connections beyond
/// `rank` are written as `zero` without entering the kernel.
/// # Arguments:
/// - `space`: Determinant space holding the parent frames.
/// - `requests`: Same-parent `(output, pair)` requests of one orthonormal parent.
/// - `rank`: Largest total excitation rank coupled by the operator.
/// - `zero`: Matrix element written for uncoupled requests.
/// - `out`: Results in original request order.
/// - `kernel`: Parent-local batched evaluator taking ket occupations, connections and outputs.
/// # Returns:
/// - `()`: Writes every request into `out`.
pub(in crate::elements) fn calculate_pairs_orthogonal_batched<T, D, U>(
    space: &NOCISpace<T>,
    requests: &[(usize, DetPair<D>)],
    rank: usize,
    zero: U,
    out: &mut [U],
    mut kernel: impl FnMut(&[(u128, u128)], &[ReducedTwoSpinState], &mut [U]),
) where
    T: NOCIScalar,
    D: PairDeterminant<T>,
    U: Copy,
{
    // Prepare coupled requests and write uncoupled ones directly.
    let mut outputs = Vec::with_capacity(requests.len());
    let mut occupations = Vec::with_capacity(requests.len());
    let mut connections = Vec::with_capacity(requests.len());
    for &(output, pair) in requests {
        let g_occ = pair.gdet.occupations(space);
        match prepare_orthogonal_state(pair.ldet.occupations(space), g_occ, rank) {
            Some(connection) => {
                outputs.push(output);
                occupations.push(g_occ);
                connections.push(connection);
            }
            None => out[output] = zero,
        }
    }
    if connections.is_empty() {
        return;
    }

    // Evaluate the parent-local batch, then scatter values back to request order.
    let mut values = vec![zero; connections.len()];
    kernel(&occupations, &connections, &mut values);
    for (&output, &value) in outputs.iter().zip(&values) {
        out[output] = value;
    }
}
