// scf/h/build.rs

use std::collections::HashMap;

use crate::input::{Input, StateRecipe};
use crate::maths::complex_metric_orthonormalize;
use crate::{AoData, HSCFState, SCFState};

use super::distance::h_density_distance_from_real;
use super::optimise::hscf_cycle;
use super::seed::{complex_orbitals_from_real, h_seed_orbitals_with_scale};
use super::select::select_hscf_candidate;
use super::types::{HSCFGenerationLookups, PartnerSeed};

/// Build one h-SCF state, including partner gating and spin-flip reuse.
/// # Arguments
/// - `ao`: AO data containing integrals and electron counts.
/// - `input`: User input controlling SCF and h-SCF.
/// - `recipes`: All state recipes.
/// - `recipe`: Holomorphic recipe being generated.
/// - `lookups`: Recipe, real-state, and h-state lookup tables.
/// - `i`: Current recipe index.
/// - `fallback`: Callback that generates a real MOM seed when no generated real seed exists.
/// # Returns
/// - `Option<HSCFState>`: Generated h-SCF state, or `None` when a partner gate says to skip it.
pub fn build_hscf_state<F>(
    ao: &AoData,
    input: &Input,
    recipes: &[StateRecipe],
    recipe: &StateRecipe,
    lookups: HSCFGenerationLookups<'_>,
    i: usize,
    fallback: F,
) -> Option<HSCFState>
where
    F: FnOnce() -> SCFState,
{
    let partner_seed = hscf_partner_seed(recipe, input, lookups.real.current, lookups.recipes);
    if matches!(partner_seed, PartnerSeed::Skip) {
        return None;
    }

    if let Some(source) = hscf_spin_flip_source(ao, recipes, recipe, i, lookups.h.current) {
        if input.write.verbose {
            println!(
                "Constructing h-SCF state '{}' by spin flip of '{}'.",
                recipe.label, source.label
            );
        }

        return Some(spin_flipped_hscf_state(source, recipe, i));
    }

    let fallback_seed;
    let seed_label = recipe.partner.as_deref().unwrap_or(recipe.label.as_str());

    let seed = match partner_seed {
        PartnerSeed::Use(current_partner) => {
            if input.write.verbose {
                println!(
                    "Seeding h-SCF state '{}' from current collapsed partner '{}'.",
                    recipe.label, seed_label
                );
            }
            current_partner
        }
        PartnerSeed::NoPartner | PartnerSeed::Skip => {
            if let Some(st) = lookups.real.current.get(seed_label).copied() {
                st
            } else {
                fallback_seed = fallback();
                &fallback_seed
            }
        }
    };

    run_hscf_state(ao, input, recipe, seed, lookups.h.previous, i)
}

fn spin_flipped_hscf_state(
    source: &HSCFState,
    recipe: &StateRecipe,
    i: usize,
) -> HSCFState {
    // The opposite spin-bias partner is obtained by swapping alpha and beta orbitals,
    // densities, occupations, and phases.
    HSCFState {
        e: source.e,
        oa: source.ob,
        ob: source.oa,
        pha: source.phb,
        phb: source.pha,
        ca: source.cb.clone(),
        cb: source.ca.clone(),
        da: source.db.clone(),
        db: source.da.clone(),
        label: recipe.label.to_string(),
        noci_basis: recipe.noci,
        parent: i,
        excitation: source.excitation.clone(),
    }
}

/// Run h-SCF continuation and fresh-seed attempts for one recipe.
/// # Arguments
/// - `ao`: AO data containing integrals and overlap.
/// - `input`: User input controlling h-SCF options and printing.
/// - `recipe`: Holomorphic recipe being optimized.
/// - `seed`: Real SCF state used for fresh h-SCF seeding.
/// - `prev_h`: Previous h-SCF states keyed by label.
/// - `i`: Recipe index used as the parent/state index.
/// # Returns
/// - `Option<HSCFState>`: Selected converged h-SCF state, or `None` if no valid branch was found.
fn run_hscf_state(
    ao: &AoData,
    input: &Input,
    recipe: &StateRecipe,
    seed: &SCFState,
    prev_h: &HashMap<&str, &HSCFState>,
    i: usize,
) -> Option<HSCFState> {
    // Prefer analytic continuation from the previous geometry.
    if let Some(st) = prev_h.get(recipe.label.as_str()).copied() {
        if input.write.verbose {
            println!(
                "Seeding h-SCF state '{}' from previous complex geometry.",
                recipe.label
            );
        }

        let ca = complex_metric_orthonormalize(&st.ca, &ao.s);
        let cb = complex_metric_orthonormalize(&st.cb, &ao.s);

        if let Some(hst) = hscf_cycle(&ca, &cb, ao, input, &recipe.label, recipe.noci, i) {
            let d = h_density_distance_from_real(&hst, seed);

            if d > input.scf.d_tol {
                println!(
                    "Accepted continued h-SCF branch '{}' with d: {:.3e}.",
                    recipe.label, d
                );
                return Some(hst);
            }

            println!(
                "Previous h-SCF branch '{}' collapsed onto real with d: {:.3e}. Trying recovery seeds.",
                recipe.label, d
            );
        } else {
            println!(
                "Previous h-SCF branch '{}' did not converge. Trying recovery seeds.",
                recipe.label
            );
        }
    }

    let mut candidates: Vec<HSCFState> = Vec::new();

    // Only reach this point if no previous h-SCF state exists, or continuation failed.
    if recipe.spin_bias.is_some() || recipe.spatial_bias.is_some() {
        let scales: &[f64] = if recipe.partner.is_some() {
            &[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
        } else {
            &[0.05]
        };

        for &scale in scales {
            let (ca, cb) = h_seed_orbitals_with_scale(seed, recipe, ao, scale);

            if let Some(hst) = hscf_cycle(&ca, &cb, ao, input, &recipe.label, recipe.noci, i) {
                candidates.push(hst);
            } else if input.write.verbose {
                println!(
                    "Fresh h-SCF seed for '{}' at scale {:.3} did not converge.",
                    recipe.label, scale
                );
            }
        }
    } else {
        let (ca, cb) = complex_orbitals_from_real(seed, ao);

        if let Some(hst) = hscf_cycle(&ca, &cb, ao, input, &recipe.label, recipe.noci, i) {
            candidates.push(hst);
        } else if input.write.verbose {
            println!(
                "Real-state h-SCF seed for '{}' did not converge.",
                recipe.label
            );
        }
    }

    select_hscf_candidate(recipe, seed, input.scf.d_tol, candidates)
}

/// Apply the optional h-SCF partner gate for one holomorphic recipe.
/// # Arguments
/// - `recipe`: Holomorphic state recipe being considered.
/// - `input`: User input controlling verbose logging.
/// - `real_map`: Generated real states keyed by label.
/// - `recipe_map`: All state recipes keyed by label.
/// # Returns
/// - `PartnerSeed`: Whether to skip the h-SCF attempt, proceed without a partner, or use a collapsed partner seed.
fn hscf_partner_seed<'a>(
    recipe: &StateRecipe,
    input: &Input,
    real_map: &HashMap<&str, &'a SCFState>,
    recipe_map: &HashMap<&str, &StateRecipe>,
) -> PartnerSeed<'a> {
    let Some(label) = recipe.partner.as_deref() else {
        return PartnerSeed::NoPartner;
    };

    let partner_state = real_map.get(label).copied();
    let partner_recipe = recipe_map.get(label).copied();

    match (partner_state, partner_recipe) {
        (Some(st), Some(partner_recipe)) if partner_recipe.noci && !st.noci_basis => {
            PartnerSeed::Use(st)
        }
        (Some(_), Some(partner_recipe)) => {
            if input.write.verbose {
                if partner_recipe.noci {
                    println!(
                        "Skipping h-SCF state '{}' because partner '{}' remains in the NOCI basis.",
                        recipe.label, label
                    );
                } else {
                    println!(
                        "Skipping h-SCF state '{}' because partner '{}' was not requested for the NOCI basis.",
                        recipe.label, label
                    );
                }
            }
            PartnerSeed::Skip
        }
        (Some(_), None) => {
            if input.write.verbose {
                println!(
                    "Skipping h-SCF state '{}' because partner recipe '{}' was not found.",
                    recipe.label, label
                );
            }
            PartnerSeed::Skip
        }
        (None, _) => {
            if input.write.verbose {
                println!(
                    "Skipping h-SCF state '{}' because partner state '{}' was not generated.",
                    recipe.label, label
                );
            }
            PartnerSeed::Skip
        }
    }
}

/// Find a previously generated spin-flipped h-SCF partner for an opposite spin-bias recipe.
/// # Arguments
/// - `ao`: AO data containing alpha/beta electron counts.
/// - `recipes`: All state recipes.
/// - `recipe`: Current holomorphic recipe.
/// - `i`: Current recipe index.
/// - `current_h`: Current-geometry h-SCF states keyed by recipe label.
/// # Returns
/// - `Option<&HSCFState>`: Previously generated h-SCF state that can be spin-flipped.
fn hscf_spin_flip_source<'a>(
    ao: &AoData,
    recipes: &'a [StateRecipe],
    recipe: &StateRecipe,
    i: usize,
    current_h: &HashMap<&str, &'a HSCFState>,
) -> Option<&'a HSCFState> {
    if ao.nelec[0] != ao.nelec[1] || recipe.spin_bias.is_none() {
        return None;
    }

    recipes
        .iter()
        .take(i)
        .find(|other| other.holomorphic && opposite_spin_bias(other, recipe))
        .and_then(|other| current_h.get(other.label.as_str()).copied())
}

/// Check whether two state recipes request opposite spin-bias patterns.
/// # Arguments
/// - `a`: First state recipe.
/// - `b`: Second state recipe.
/// # Returns
/// - `bool`: Whether both recipes have spin biases and the patterns are sign opposites.
fn opposite_spin_bias(
    a: &StateRecipe,
    b: &StateRecipe,
) -> bool {
    let (Some(sa), Some(sb)) = (&a.spin_bias, &b.spin_bias) else {
        return false;
    };

    sa.pattern.len() == sb.pattern.len()
        && sa
            .pattern
            .iter()
            .zip(sb.pattern.iter())
            .all(|(&x, &y)| x == -y)
}
