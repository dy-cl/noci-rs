// driver/scf.rs

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::basis::{generate_reference_noci_basis, hermitian_hnoci_basis};
use crate::determinant::Bits;
use crate::input::Input;
use crate::noci::optimised::reshf_optimise;
use crate::time_call;
use crate::{AoData, HSCFState, SCFState, StateScalar};

/// Prepared real references for downstream reference-space NOCI.
pub struct RealReferencePrep {
    /// SCF states generated for this geometry.
    pub states: Vec<SCFState>,
    /// Real reference basis before reference-space filtering.
    pub basis: Vec<SCFState<f64>>,
}

/// Prepared holomorphic references for downstream reference-space NOCI.
pub struct HolomorphicReferencePrep {
    /// Real SCF states generated for this geometry.
    pub states: Vec<SCFState>,
    /// Complex h-SCF states generated for this geometry.
    pub hstates: Vec<HSCFState>,
    /// Off axis complex h-SCF states for tracking across geometries.
    pub htracks: Vec<HSCFState>,
    /// Complex Hermitian reference basis before reference-space filtering.
    pub basis: Vec<HSCFState>,
}

/// Run SCF calculations for real reference states.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `prev_states`: Converged SCF states at previous r, used for seeding.
/// - `tol`: Tolerance up to which a number is considered zero.
/// # Returns:
/// - `RealReferencePrep`: Generated real states and reference basis.
pub fn generate_real_references(
    ao: &AoData,
    input: &mut Input,
    prev_states: &[SCFState],
    tol: f64,
) -> RealReferencePrep {
    let states = time_call!(crate::timers::general::add_run_scf, {
        if prev_states.is_empty() {
            generate_reference_noci_basis(ao, input, None, None).states
        } else {
            generate_reference_noci_basis(ao, input, Some(prev_states), None).states
        }
    });
    let mut states = match &input.reshf {
        Some(options) => time_call!(crate::timers::general::add_reshf_optimise, {
            reshf_optimise(ao, input, options, &states, tol)
        }),
        None => states,
    };
    align_orbitals(&mut states, prev_states, &ao.s);

    RealReferencePrep {
        basis: states.clone(),
        states,
    }
}

/// Run SCF calculations for holomorphic reference states.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `previous`: Converged SCF states at previous r, used for seeding and orbital alignment, the
///   h-SCF states, used for orbital alignment, and the h-SCF tracking states, used for complex
///   branch tracking.
/// # Returns:
/// - `HolomorphicReferencePrep`: Generated real, h-SCF, and Hermitian h-NOCI references.
pub fn generate_holomorphic_references(
    ao: &AoData,
    input: &mut Input,
    previous: (&[SCFState], &[HSCFState], &[HSCFState]),
) -> HolomorphicReferencePrep {
    let (prev_states, prev_hstates, prev_htracks) = previous;
    let mut refs = time_call!(crate::timers::general::add_run_scf, {
        if prev_states.is_empty() && prev_htracks.is_empty() {
            generate_reference_noci_basis(ao, input, None, None)
        } else {
            let prev_htracks = if prev_htracks.is_empty() {
                None
            } else {
                Some(prev_htracks)
            };
            generate_reference_noci_basis(ao, input, Some(prev_states), prev_htracks)
        }
    });
    align_orbitals(&mut refs.states, prev_states, &ao.s);
    align_orbitals(&mut refs.hstates, prev_hstates, &ao.s);
    let basis = hermitian_hnoci_basis(&refs.hstates, &ao.s);

    HolomorphicReferencePrep {
        states: refs.states,
        hstates: refs.hstates,
        htracks: refs.htracks,
        basis,
    }
}

/// Align the molecular orbitals of every state with the same-labelled state of the previous
/// geometry. The SCF fixes each orbital only up to its sign, and orbitals of nearly equal energy
/// may change order between geometries, so neither is comparable from one geometry to the next.
/// With `O = C^{\text{prev}T} S C`, each new orbital `j` takes the position `i` of the previous
/// orbital of the same occupancy with the largest `|O_{ij}|`, and its sign is chosen so that
/// `\mathrm{Re}\,O_{ij} > 0`. When the largest overlaps do not define a permutation the order is
/// kept and only the signs are chosen. The alignment is `C \rightarrow C P`, with `P` a signed
/// permutation that never exchanges an occupied orbital with a virtual one, so no physical
/// quantity changes. The occupied space keeps the same orbitals, so the density
/// `D = C_{\text{occ}} C_{\text{occ}}^T`, the Fock matrix and the SCF energy are unchanged. The
/// determinant `\det(C_{\text{occ}})` changes by at most an overall sign, which the NOCI
/// eigenvector absorbs, so NOCI energies are unchanged. As `P^T P = I`, the orthonormality
/// `C^T S C = I` still holds, in the bilinear holomorphic metric too. Only the labels of the
/// orbitals change, so that an excitation `i \to a` names the same orbitals at neighbouring
/// geometries and amplitudes can be carried along a scan.
/// # Arguments:
/// - `states`: States of this geometry, aligned in place.
/// - `previous`: States of the previous geometry.
/// - `s`: AO overlap matrix of this geometry.
/// # Returns:
/// - `()`: Reorders and flips the orbitals of `states`, updating their occupations.
fn align_orbitals<T: StateScalar>(
    states: &mut [SCFState<T>],
    previous: &[SCFState<T>],
    s: &Array2<f64>,
) {
    let s = s.mapv(T::from_real);

    // States new at this geometry have nothing to align with and are left as the SCF gave them.
    for state in states.iter_mut() {
        let Some(old) = previous.iter().find(|p| p.label == state.label) else {
            continue;
        };
        for (c, occ, cold, occold) in [
            (&mut state.ca, &mut state.oa, &old.ca, old.oa),
            (&mut state.cb, &mut state.ob, &old.cb, old.ob),
        ] {
            // A different basis size or electron count has no orbital correspondence.
            if c.dim() != cold.dim() || occ.count_ones() != occold.count_ones() {
                continue;
            }

            // `O_{ij} = C^{\text{prev}T}_i S C_j`, which is `\pm 1` for an orbital that only
            // changed sign or position, and close to it for small geometry steps.
            let n = c.ncols();
            let o = cold.t().dot(&s.dot(&**c));

            // Previous orbital of the same occupancy with the largest overlap with each new one.
            // Matching only within the occupied or the virtual orbitals keeps the occupied space,
            // and therefore the determinant and its density, unchanged.
            let mut place = (0..n)
                .map(|j| {
                    (0..n)
                        .filter(|&i| occold.contains(i) == occ.contains(j))
                        .max_by(|&a, &b| o[(a, j)].abs().total_cmp(&o[(b, j)].abs()))
                        .unwrap_or(j)
                })
                .collect::<Vec<_>>();

            // Two new orbitals claiming one previous orbital means the overlaps do not identify a
            // reordering, so keep the SCF order and only choose the signs.
            let mut claimed = vec![false; n];
            if place
                .iter()
                .any(|&i| std::mem::replace(&mut claimed[i], true))
            {
                place = (0..n).collect();
            }

            // Move each orbital to its place with the sign giving `\mathrm{Re}\,O_{ij} > 0`. A
            // sign flip changes the determinant by at most an overall sign, which the NOCI
            // eigenvector absorbs.
            let mut aligned = Array2::<T>::zeros(c.raw_dim());
            for (j, &i) in place.iter().enumerate() {
                let sign = if o[(i, j)].re() < 0.0 {
                    -T::one()
                } else {
                    T::one()
                };
                aligned
                    .column_mut(i)
                    .assign(&c.column(j).mapv(|x| x * sign));
            }

            // The occupation bits follow the occupied orbitals to their new positions.
            *occ = (0..n)
                .filter(|&j| occ.contains(j))
                .fold(Bits::EMPTY, |bits, j| bits | Bits::bit(place[j]));
            *c = Arc::new(aligned);
        }
    }
}
