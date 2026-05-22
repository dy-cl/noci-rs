// basis/excitation.rs

use std::sync::Arc;

use itertools::Itertools;

use crate::input::Input;
use crate::noci::NOCIScalar;
use crate::scf::spin_occupation;
use crate::{DetState, Excitation, ExcitationSpin};

/// Copy a reference determinant and replace its alpha and beta occupation bitstrings.
/// # Arguments
/// - `reference`: State from which to build an excited determinant.
/// - `occ`: Excited alpha and beta occupation bitstrings.
/// - `label_suffix`: What to append to the reference state label to indicate excitation.
/// - `parent`: Index of the parent reference determinant.
/// - `excitation`: Excitation carried by the excited state.
/// - `parent_occ`: Parent alpha and beta occupation bitstrings used to calculate the excitation phase.
/// # Returns
/// - `DetState<T>`: Excited state built from the reference determinant with modified occupancies.
fn make_excited_state<T: NOCIScalar>(
    reference: &DetState<T>,
    occ: (u128, u128),
    label_suffix: &str,
    parent: usize,
    excitation: Excitation,
    parent_occ: (u128, u128),
) -> DetState<T> {
    let pha = excitation_phase(
        parent_occ.0,
        &excitation.alpha.holes,
        &excitation.alpha.parts,
    );
    let phb = excitation_phase(parent_occ.1, &excitation.beta.holes, &excitation.beta.parts);

    DetState {
        e: T::from_real(0.0),
        oa: occ.0,
        ob: occ.1,
        pha,
        phb,
        ca: Arc::clone(&reference.ca),
        cb: Arc::clone(&reference.cb),
        da: Arc::clone(&reference.da),
        db: Arc::clone(&reference.db),
        label: format!("{} {}", reference.label, label_suffix),
        noci_basis: false,
        parent,
        excitation,
    }
}

/// Calculate fermionic sign associated with applying a set of creation and annihilation operators to a determinant described by a bitstring.
/// # Arguments:
/// - `occ`: Occupancy bitstring.
/// - `holes`: Annihilation operators indices.
/// - `parts`: Creation operator indices.
/// # Returns
/// - `f64`: Fermionic phase factor.
#[inline(always)]
pub fn excitation_phase(
    mut occ: u128,
    holes: &[usize],
    parts: &[usize],
) -> f64 {
    /// Determine whether the number of occupied orbitals below orbital index `p` is odd.
    /// # Arguments:
    /// - `bits`: Occupancy bitstring.
    /// - `p`: Orbital index.
    /// # Returns:
    /// - `bool`: `true` if the number of occupied orbitals with index less than `p` is odd, otherwise `false`.
    #[inline(always)]
    fn below(
        bits: u128,
        p: usize,
    ) -> bool {
        if p == 0 {
            false
        } else {
            ((bits & ((1u128 << p) - 1)).count_ones() & 1) != 0
        }
    }

    let mut odd = false;

    for &i in holes.iter().rev() {
        odd ^= below(occ, i);
        occ &= !(1u128 << i);
    }

    for (k, &a) in parts.iter().enumerate() {
        odd ^= below(occ, a);
        if k + 1 != parts.len() {
            occ |= 1u128 << a;
        }
    }
    if odd { -1.0 } else { 1.0 }
}

/// Construct a label describing an excitation in alpha and/or beta spin.
/// # Arguments
/// - `alpha_holes`: Occupied alpha orbital indices from which electrons are removed.
/// - `alpha_parts`: Virtual alpha orbital indices into which electrons are placed.
/// - `beta_holes`: Occupied beta orbital indices from which electrons are removed.
/// - `beta_parts`: Virtual beta orbital indices into which electrons are placed.
/// # Returns
/// - `String`: Label describing the excitation pattern.
fn excitation_label(
    alpha_holes: &[usize],
    alpha_parts: &[usize],
    beta_holes: &[usize],
    beta_parts: &[usize],
) -> String {
    let mut label = Vec::new();
    if !alpha_holes.is_empty() {
        label.push(format!(
            "alpha {} -> {}",
            alpha_holes
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" "),
            alpha_parts
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        ))
    }
    if !beta_holes.is_empty() {
        label.push(format!(
            "beta {} -> {}",
            beta_holes
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" "),
            beta_parts
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        ))
    }
    format!("({})", label.join("; "))
}

/// Construct an excitation object from alpha and beta hole/particle lists.
/// # Arguments
/// - `alpha_holes`: Occupied alpha orbital indices from which electrons are removed.
/// - `alpha_parts`: Virtual alpha orbital indices into which electrons are placed.
/// - `beta_holes`: Occupied beta orbital indices from which electrons are removed.
/// - `beta_parts`: Virtual beta orbital indices into which electrons are placed.
/// # Returns
/// - `Excitation`: Excitation object containing the specified alpha and beta spin excitations.
fn build_excitation(
    alpha_holes: &[usize],
    alpha_parts: &[usize],
    beta_holes: &[usize],
    beta_parts: &[usize],
) -> Excitation {
    Excitation {
        alpha: ExcitationSpin {
            holes: alpha_holes.to_vec(),
            parts: alpha_parts.to_vec(),
        },
        beta: ExcitationSpin {
            holes: beta_holes.to_vec(),
            parts: beta_parts.to_vec(),
        },
    }
}

/// Apply a spin-specific excitation to an occupation bitstring.
/// # Arguments
/// - `occ`: Occupation bitstring to be modified.
/// - `holes`: Occupied orbital indices from which electrons are removed.
/// - `parts`: Virtual orbital indices into which electrons are placed.
/// # Returns
/// - `u128`: New occupation bitstring with the requested excitation applied.
fn apply_excitation(
    occ: u128,
    holes: &[usize],
    parts: &[usize],
) -> u128 {
    let mut out = occ;
    for &i in holes {
        out &= !(1u128 << i);
    }
    for &a in parts {
        out |= 1u128 << a;
    }
    out
}

/// Undo a spin-specific excitation from a child occupation bitstring.
/// # Arguments
/// - `occ`: Child occupation bitstring.
/// - `holes`: Orbitals removed from the parent determinant.
/// - `parts`: Orbitals added to the parent determinant.
/// # Returns
/// - `u128`: Reconstructed parent occupation bitstring.
fn undo_excitation(
    occ: u128,
    holes: &[usize],
    parts: &[usize],
) -> u128 {
    let mut out = occ;
    for &a in parts {
        out &= !(1u128 << a);
    }
    for &i in holes {
        out |= 1u128 << i;
    }
    out
}

/// Construct the excitation mapping one occupation bitstring to another.
/// # Arguments
/// - `parent`: Parent occupation bitstring.
/// - `child`: Child occupation bitstring.
/// # Returns
/// - `(Vec<usize>, Vec<usize>)`: Hole and particle orbital indices.
fn excitation_between(
    parent: u128,
    child: u128,
) -> (Vec<usize>, Vec<usize>) {
    let holes_bits = parent & !child;
    let parts_bits = child & !parent;
    let holes = (0..128)
        .filter(|&i| ((holes_bits >> i) & 1u128) == 1)
        .collect();
    let parts = (0..128)
        .filter(|&i| ((parts_bits >> i) & 1u128) == 1)
        .collect();
    (holes, parts)
}

/// Generate a requested amount of all possible excitations on top of the given reference NOCI basis.
/// # Arguments
/// - `refs`: Array of reference states for which excitations are generated.
/// - `input`: Contains user inputted options.
/// - `include_refs`: Whether or not to include the references in the returned basis.
/// # Returns
/// - `Vec<DetState<T>>`: Generated excited basis, optionally including the reference states.
pub fn generate_excited_basis<T: NOCIScalar>(
    refs: &[DetState<T>],
    input: &Input,
    include_refs: bool,
) -> Vec<DetState<T>> {
    let mut out: Vec<DetState<T>> = Vec::new();

    for r in refs {
        let parent = r.parent;

        if include_refs {
            let mut rcopy = r.clone();
            rcopy.parent = parent;
            out.push(rcopy);
        }

        let spin_occ = spin_occupation(r);

        let mut orders = if input.excit.all {
            let max_order = (spin_occ.occ_alpha.len() + spin_occ.occ_beta.len())
                .min(spin_occ.virt_alpha.len() + spin_occ.virt_beta.len());
            (1..=max_order).collect::<Vec<_>>()
        } else {
            input.excit.orders.clone()
        };

        orders.sort_unstable();
        orders.dedup();

        for &k in &orders {
            for k_alpha in 0..=k {
                let k_beta = k - k_alpha;

                for alpha_holes in spin_occ.occ_alpha.iter().copied().combinations(k_alpha) {
                    for alpha_parts in spin_occ.virt_alpha.iter().copied().combinations(k_alpha) {
                        for beta_holes in spin_occ.occ_beta.iter().copied().combinations(k_beta) {
                            for beta_parts in
                                spin_occ.virt_beta.iter().copied().combinations(k_beta)
                            {
                                // Apply excitation to supplied state `r`. In stochastic routines
                                // this will always be a reference determinant, but in SNOCI this
                                // may be an already excited determinant relative to its parent.
                                let oa_ex = apply_excitation(r.oa, &alpha_holes, &alpha_parts);
                                let ob_ex = apply_excitation(r.ob, &beta_holes, &beta_parts);

                                // Matrix element routines interpret the excitation phase relative
                                // to the parent of `r`, so if `r` is already excited, we must undo
                                // the excitation and calculate the total excitation from the parent
                                // of `r` to the new state.
                                let parent_oa = undo_excitation(
                                    r.oa,
                                    &r.excitation.alpha.holes,
                                    &r.excitation.alpha.parts,
                                );
                                let parent_ob = undo_excitation(
                                    r.ob,
                                    &r.excitation.beta.holes,
                                    &r.excitation.beta.parts,
                                );

                                let (alpha_holes_total, alpha_parts_total) =
                                    excitation_between(parent_oa, oa_ex);
                                let (beta_holes_total, beta_parts_total) =
                                    excitation_between(parent_ob, ob_ex);

                                let label = excitation_label(
                                    &alpha_holes_total,
                                    &alpha_parts_total,
                                    &beta_holes_total,
                                    &beta_parts_total,
                                );
                                let excitation = build_excitation(
                                    &alpha_holes_total,
                                    &alpha_parts_total,
                                    &beta_holes_total,
                                    &beta_parts_total,
                                );

                                let exstate = make_excited_state(
                                    r,
                                    (oa_ex, ob_ex),
                                    &label,
                                    parent,
                                    excitation,
                                    (parent_oa, parent_ob),
                                );
                                out.push(exstate);
                            }
                        }
                    }
                }
            }
        }
    }

    out
}
