// basis/excitation.rs

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use itertools::Itertools;

// Crate-root imports.
use crate::input::Input;
use crate::noci::NOCIScalar;
use crate::scf::spin_occupation;
use crate::{DetState, Excitation, ExcitationSpin};

/// Decode the first four hole and particle orbital indices from an excitation bit mask.
/// `indices[0..4]` stores ascending hole indices and `indices[4..8]` stores ascending particle
/// indices. Fixed-rank SIMD Wick kernels only consume this cache when the spin excitation rank is
/// at most four; higher-rank paths continue to use the complete `u128` excitation masks.
/// # Arguments:
/// - `holes`: Hole orbital bit mask.
/// - `parts`: Particle orbital bit mask.
/// # Returns
/// - `(u8, [u8; 8])`: Excitation rank and cached hole/particle orbital indices.
#[inline(always)]
fn decode_excitation_indices(
    holes: u128,
    parts: u128,
) -> (u8, [u8; 8]) {
    let rank = holes.count_ones() as u8;
    let mut indices = [0u8; 8];
    let mut holes_left = holes;
    let mut parts_left = parts;

    for i in 0..4 {
        if holes_left != 0 {
            indices[i] = holes_left.trailing_zeros() as u8;
            holes_left &= holes_left - 1;
        }
        if parts_left != 0 {
            indices[4 + i] = parts_left.trailing_zeros() as u8;
            parts_left &= parts_left - 1;
        }
    }

    (rank, indices)
}

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
    let pha = excitation_phase_bits(parent_occ.0, excitation.alpha.holes, excitation.alpha.parts);
    let phb = excitation_phase_bits(parent_occ.1, excitation.beta.holes, excitation.beta.parts);
    let (rank_a, indices_a) =
        decode_excitation_indices(excitation.alpha.holes, excitation.alpha.parts);
    let (rank_b, indices_b) =
        decode_excitation_indices(excitation.beta.holes, excitation.beta.parts);

    DetState {
        e: T::from_real(0.0),
        oa: occ.0,
        ob: occ.1,
        pha,
        phb,
        rank_a,
        rank_b,
        indices_a,
        indices_b,
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

/// Calculate fermionic sign associated with applying stored excitation bit masks to a determinant.
/// # Arguments:
/// - `occ`: Occupancy bitstring.
/// - `holes`: Annihilation operator bit mask.
/// - `parts`: Creation operator bit mask.
/// # Returns
/// - `f64`: Fermionic phase factor.
#[inline(always)]
fn excitation_phase_bits(
    mut occ: u128,
    mut holes: u128,
    mut parts: u128,
) -> f64 {
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

    while holes != 0 {
        let i = 127 - holes.leading_zeros() as usize;
        odd ^= below(occ, i);
        occ &= !(1u128 << i);
        holes &= !(1u128 << i);
    }

    while parts != 0 {
        let a = parts.trailing_zeros() as usize;
        odd ^= below(occ, a);
        parts &= parts - 1;
        if parts != 0 {
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
    alpha_holes: u128,
    alpha_parts: u128,
    beta_holes: u128,
    beta_parts: u128,
) -> String {
    let format_mask = |mut bits: u128| {
        let mut orbitals = Vec::with_capacity(bits.count_ones() as usize);

        while bits != 0 {
            let p = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            orbitals.push(p.to_string());
        }

        orbitals.join(" ")
    };

    let mut label = Vec::new();
    if alpha_holes != 0 {
        label.push(format!(
            "alpha {} -> {}",
            format_mask(alpha_holes),
            format_mask(alpha_parts),
        ))
    }
    if beta_holes != 0 {
        label.push(format!(
            "beta {} -> {}",
            format_mask(beta_holes),
            format_mask(beta_parts),
        ))
    }
    format!("({})", label.join("; "))
}

/// Construct an excitation object from alpha and beta hole/particle masks.
/// # Arguments
/// - `alpha_holes`: Occupied alpha orbital indices from which electrons are removed.
/// - `alpha_parts`: Virtual alpha orbital indices into which electrons are placed.
/// - `beta_holes`: Occupied beta orbital indices from which electrons are removed.
/// - `beta_parts`: Virtual beta orbital indices into which electrons are placed.
/// # Returns
/// - `Excitation`: Excitation object containing the specified alpha and beta spin excitations.
fn build_excitation(
    alpha_holes: u128,
    alpha_parts: u128,
    beta_holes: u128,
    beta_parts: u128,
) -> Excitation {
    Excitation {
        alpha: ExcitationSpin {
            holes: alpha_holes,
            parts: alpha_parts,
        },
        beta: ExcitationSpin {
            holes: beta_holes,
            parts: beta_parts,
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
    holes: u128,
    parts: u128,
) -> u128 {
    (occ & !parts) | holes
}

/// Construct the excitation mapping one occupation bitstring to another.
/// # Arguments
/// - `parent`: Parent occupation bitstring.
/// - `child`: Child occupation bitstring.
/// # Returns
/// - `(u128, u128)`: Hole and particle orbital masks.
#[inline(always)]
fn excitation_between(
    parent: u128,
    child: u128,
) -> (u128, u128) {
    (parent & !child, child & !parent)
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
                                    r.excitation.alpha.holes,
                                    r.excitation.alpha.parts,
                                );
                                let parent_ob = undo_excitation(
                                    r.ob,
                                    r.excitation.beta.holes,
                                    r.excitation.beta.parts,
                                );

                                let (alpha_holes_total, alpha_parts_total) =
                                    excitation_between(parent_oa, oa_ex);
                                let (beta_holes_total, beta_parts_total) =
                                    excitation_between(parent_ob, ob_ex);

                                let label = excitation_label(
                                    alpha_holes_total,
                                    alpha_parts_total,
                                    beta_holes_total,
                                    beta_parts_total,
                                );
                                let excitation = build_excitation(
                                    alpha_holes_total,
                                    alpha_parts_total,
                                    beta_holes_total,
                                    beta_parts_total,
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
