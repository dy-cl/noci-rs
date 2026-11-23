// basis/replace.rs
//! Replacement of duplicate or unconverged SCF states by excited SCF states.
//!
//! When a recipe's SCF solution duplicates an earlier NOCI-basis state, or its SCF does not
//! converge, an excited state of a parent determinant takes its place in the NOCI basis, so the
//! basis keeps the requested number of determinants. The listed excitations are tried first,
//! then generated ones, each run with MOM from the parent's converged densities at the current
//! geometry so that it promotes between the parent's canonical orbitals. An excited state that
//! is itself a duplicate, or does not converge, is discarded and the next is tried.

// Standard library imports.
use std::collections::HashMap;

// Crate-root imports.
use crate::input::{Input, ReplaceOptions, SCFExcitation, Spin, StateRecipe};
use crate::scf::scf_cycle;
use crate::{AoData, SCFState};

// Parent/sibling imports.
use super::duplicate::find_duplicate;
use super::mom::run_mom_scf_state;

/// Build the excitations of the parent tried for replacement: the listed ones, then
/// `occ = -1 - o` to `vir = v` by increasing `o + v`, first as paired doubles (`spin = both`)
/// and then as alpha and beta singles. Excitations outside the parent's orbitals, that is
/// unless `-n_{\text{occ}} \le occ \le -1` and `0 \le vir < n_{\text{mo}} - n_{\text{occ}}` in
/// each excited spin, are dropped.
/// # Arguments
/// - `listed`: Excitations from the input, tried first.
/// - `nocc`: Number of alpha- and beta-spin occupied MOs.
/// - `nmo`: Number of MOs.
/// # Returns
/// - `Vec<SCFExcitation>`: Distinct excitations in the order they are tried.
fn replacement_candidates(
    listed: &[SCFExcitation],
    nocc: (usize, usize),
    nmo: usize,
) -> Vec<SCFExcitation> {
    let valid = |ex: &SCFExcitation| {
        let spins: &[usize] = match ex.spin {
            Spin::Alpha => &[nocc.0],
            Spin::Beta => &[nocc.1],
            Spin::Both => &[nocc.0, nocc.1],
        };
        spins.iter().all(|&n| {
            ex.occ < 0 && ex.occ >= -(n as i32) && ex.vir >= 0 && n + (ex.vir as usize) < nmo
        })
    };

    let mut out: Vec<SCFExcitation> = Vec::new();
    for ex in listed {
        if valid(ex) {
            out.push(ex.clone());
        } else {
            println!(
                "Skipped replacement excitation ({}, {}, {}) outside the parent's orbitals",
                ex.spin.as_str(),
                ex.occ,
                ex.vir
            );
        }
    }

    // Paired doubles keep a closed-shell parent closed-shell, so they come before singles.
    for spins in [&[Spin::Both][..], &[Spin::Alpha, Spin::Beta][..]] {
        for level in 0..nmo {
            for o in 0..=level {
                for &spin in spins {
                    let ex = SCFExcitation {
                        spin,
                        occ: -(o as i32) - 1,
                        vir: (level - o) as i32,
                    };
                    if valid(&ex) && !out.contains(&ex) {
                        out.push(ex);
                    }
                }
            }
        }
    }
    out
}

/// Run the next excitations of `parent` until one converges to a state that is not a duplicate
/// of the NOCI basis.
/// # Arguments
/// - `ao`: AO data containing integrals.
/// - `input`: User input controlling SCF and printing.
/// - `states`: States already generated at this geometry.
/// - `parent`: Determinant to excite.
/// - `excits`: Remaining excitations; those tried are consumed.
/// - `replaced`: Label of the state being replaced.
/// # Returns
/// - `Option<SCFState>`: Excited NOCI-basis state, or `None` if no excitation is left.
fn excited_replacement(
    ao: &AoData,
    input: &Input,
    states: &[SCFState],
    parent: &SCFState,
    excits: &mut impl Iterator<Item = SCFExcitation>,
    replaced: &str,
) -> Option<SCFState> {
    for ex in excits {
        let label = format!(
            "{} excit({}, {}, {})",
            parent.label,
            ex.spin.as_str(),
            ex.occ,
            ex.vir
        );
        if input.write.verbose >= 1 {
            let left = "=".repeat(45);
            let right = "=".repeat(46);
            println!("{}Begin SCF{}", left, right);
            println!("Replacement for '{}': {}", replaced, label);
        };

        // Seeding from the parent's densities makes the first Fock eigenvectors the parent's
        // canonical orbitals, so MOM promotes from the parent rather than from an atomic guess.
        let Some(state) = scf_cycle(
            (&*parent.da, &*parent.db),
            ao,
            input,
            &label,
            true,
            (false, Some(&ex), None),
        ) else {
            println!("Excited state '{}' did not converge", label);
            continue;
        };

        if let Some(j) = find_duplicate(states, &state, &ao.s, input.scf.d_tol) {
            println!(
                "Discarded excited state '{}' as duplicate of '{}'",
                label, states[j].label
            );
            continue;
        }

        println!(
            "Replaced state '{}' in NOCI basis by excited state '{}'",
            replaced, label
        );
        return Some(state);
    }

    println!(
        "No replacement excitation left for state '{}'; it stays out of the NOCI basis",
        replaced
    );
    None
}

/// Generate the SCF states of MOM recipes, replacing each duplicate or unconverged NOCI-basis
/// state by the next usable excitation of the parent, so the NOCI basis keeps one state per
/// requested recipe while excitations remain. Without duplicates or failures this gives the
/// same states as `generate_states_mom`.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: Contains user inputted options.
/// - `prev`: May or may not contain states from a previous geometry.
/// - `prev_map`: Map between the SCFState object and its label.
/// - `recipes`: Instructions for how to construct each state.
/// - `options`: Parent and listed excitations used for replacement.
/// # Returns:
/// - `Vec<SCFState>`: Generated SCF states, each replacement directly after the state it replaces.
pub(crate) fn generate_states_mom_replaced(
    ao: &AoData,
    input: &Input,
    prev: Option<&[SCFState]>,
    prev_map: &HashMap<&str, &SCFState>,
    recipes: &[StateRecipe],
    options: &ReplaceOptions,
) -> Vec<SCFState> {
    let mut out: Vec<SCFState> = Vec::with_capacity(recipes.len());
    let mut excits: Option<std::vec::IntoIter<SCFExcitation>> = None;

    for (i, recipe) in recipes.iter().enumerate() {
        if recipe.holomorphic {
            continue;
        }

        // A duplicate leaves the NOCI basis but is kept for printing and branch tracking, as in
        // `mark_duplicate_noci_states`; an unconverged state has no solution to keep.
        let missing = match run_mom_scf_state(ao, input, recipe, prev, prev_map, i) {
            Some(mut state) => {
                let dup = if state.noci_basis {
                    find_duplicate(&out, &state, &ao.s, input.scf.d_tol)
                } else {
                    None
                };
                if let Some(j) = dup {
                    println!(
                        "Removed state '{}' from NOCI basis as duplicate of '{}'",
                        state.label, out[j].label
                    );
                    state.noci_basis = false;
                }
                out.push(state);
                dup.is_some()
            }
            None => {
                println!("SCF for state '{}' did not converge", recipe.label);
                recipe.noci
            }
        };
        if !missing {
            continue;
        }

        // Fill the vacated NOCI-basis slot with an excitation of the parent: the named state,
        // or else the first NOCI-basis state.
        let parent = out
            .iter()
            .find(|st| st.noci_basis && options.parent.as_ref().is_none_or(|p| &st.label == p))
            .cloned();
        let Some(parent) = parent else {
            println!(
                "No parent state available to replace state '{}'; it stays out of the NOCI basis",
                recipe.label
            );
            continue;
        };
        // Excitations are built once per geometry, when the first state needs replacing.
        let excits = excits.get_or_insert_with(|| {
            let nocc = (
                usize::try_from(ao.nelec[0]).unwrap(),
                usize::try_from(ao.nelec[1]).unwrap(),
            );
            replacement_candidates(&options.excits, nocc, parent.ca.ncols()).into_iter()
        });
        if let Some(state) = excited_replacement(ao, input, &out, &parent, excits, &recipe.label) {
            out.push(state);
        }
    }
    out
}
