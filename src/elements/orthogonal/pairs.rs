// elements/orthogonal/pairs.rs
//! Determinant-pair matrix elements from the Slater–Condon rules in one parent's orthonormal
//! basis.

// Crate-root imports.
use crate::basis::{excitation_between, excitation_phase};
use crate::determinant::{NOCIIndex, NOCISpace, OrthogonalConnection};
use crate::elements::{FockMOCache, MOCache, NOCIData};
use crate::time_call;
use crate::{AoData, Excitation, ExcitationCache, ExcitationSpin, NOCIScalar, ReducedTwoSpinState};

// Parent/sibling imports.
use super::{xw_hamiltonian_orthogonal_prepared, xw_hamiltonian_orthogonal_prepared_batched};

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
        if space.occupations(ldet) == space.occupations(gdet) {
            <T as From<f64>>::from(space.phase(ldet) * space.phase(gdet))
        } else {
            <T as From<f64>>::from(0.0)
        }
    })
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
pub(in crate::elements) fn calculate_hs_pair_orthogonal<T: NOCIScalar>(
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
            // Only double sectors use the prepared vector kernels; singles remain scalar.
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
                let state = connection.reduced(data.space.alpha(source), data.space.beta(source));
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
        let (loa, lob) = space.occupations(ldet);
        let (goa, gob) = space.occupations(gdet);
        let xa = loa ^ goa;
        let xb = lob ^ gob;

        let na = xa.count_ones() as usize;
        let nb = xb.count_ones() as usize;

        // The one-body Fock operator connects only identical determinants
        // and single excitations in one spin sector.
        if na == 0 && nb == 0 {
            // Diagonal element: sum the occupied `\alpha` and `\beta` MO Fock energies.
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

        // One hole and one particle give the signed `\alpha` Fock coupling.
        if na == 2 && nb == 0 {
            let hole = (goa & xa).trailing_zeros() as usize;
            let part = (loa & xa).trailing_zeros() as usize;
            let phase = <T as From<f64>>::from(excitation_phase(goa, &[hole], &[part]));
            return phase * cache.fa[(part, hole)];
        }

        // The `\beta` single has the analogous Slater-Condon matrix element.
        if na == 0 && nb == 2 {
            let hole = (gob & xb).trailing_zeros() as usize;
            let part = (lob & xb).trailing_zeros() as usize;
            let phase = <T as From<f64>>::from(excitation_phase(gob, &[hole], &[part]));
            return phase * cache.fb[(part, hole)];
        }
        <T as From<f64>>::from(0.0)
    })
}

/// Calculate the shifted candidate-candidate matrix element between determinants
/// with the same parent using standard Slater-Condon rules.
/// # Arguments:
/// - `cache`: MO-basis Fock cache for the shared parent determinant.
/// - `ldet`: State `a`.
/// - `gdet`: State `b`.
/// - `e0`: Zeroth-order energy shift.
/// - `space`: Determinant space containing the bra and ket states.
/// # Returns:
/// - `T`: Shifted matrix element `M_{ab}`.
pub(in crate::elements) fn calculate_m_pair_orthogonal<T: NOCIScalar>(
    cache: &FockMOCache<T>,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    e0: f64,
) -> T {
    let (loa, lob) = space.occupations(ldet);
    let (goa, gob) = space.occupations(gdet);
    let xa = loa ^ goa;
    let xb = lob ^ gob;
    let na = xa.count_ones() as usize;
    let nb = xb.count_ones() as usize;

    // A one-body Fock operator connects identical determinants or a single
    // excitation in one spin sector; higher excitation ranks vanish.
    if na == 0 && nb == 0 {
        // `M_{aa} = \sum_{i\in\text{occ}_\alpha} F^\alpha_{ii} + \sum_{i\in\text{occ}_\beta} F^\beta_{ii} - E_0 S_{aa}`.
        let mut f = <T as From<f64>>::from(0.0);

        let mut bits = goa;
        while bits != 0 {
            let p = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f += cache.fa[(p, p)];
        }

        let mut bits = gob;
        while bits != 0 {
            let p = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f += cache.fb[(p, p)];
        }

        let s = <T as From<f64>>::from(space.phase(ldet) * space.phase(gdet));
        return f - <T as From<f64>>::from(e0) * s;
    }

    // Two differing occupation bits identify one `\alpha` hole and one particle.
    if na == 2 && nb == 0 {
        let hole = (goa & xa).trailing_zeros() as usize;
        let part = (loa & xa).trailing_zeros() as usize;
        let phase = <T as From<f64>>::from(excitation_phase(goa, &[hole], &[part]));
        return phase * cache.fa[(part, hole)];
    }

    // The `\beta` single excitation has the analogous signed matrix element.
    if na == 0 && nb == 2 {
        let hole = (gob & xb).trailing_zeros() as usize;
        let part = (lob & xb).trailing_zeros() as usize;
        let phase = <T as From<f64>>::from(excitation_phase(gob, &[hole], &[part]));
        return phase * cache.fb[(part, hole)];
    }

    <T as From<f64>>::from(0.0)
}
