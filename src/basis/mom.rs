// basis/mom.rs

use std::collections::HashMap;

use crate::input::{Input, StateRecipe};
use crate::scf::{continue_hscf_track, initialise_hscf_track, physical_hscf_state, scf_cycle};
use crate::{AoData, HSCFState, SCFState};

use super::bias::biased_density_guess;
use super::duplicate::mark_duplicate_noci_states;
use super::types::ReferenceBasis;

/// Choose the previous-geometry real seed used for a MOM recipe.
/// # Arguments
/// - `recipe`: State recipe currently being generated.
/// - `prev_map`: Previous real SCF states keyed by label.
/// # Returns
/// - `Option<&SCFState>`: Previous state with the same label, or previous RHF for excited MOM states.
fn previous_mom_seed<'a>(
    recipe: &StateRecipe,
    prev_map: &'a HashMap<&str, &SCFState>,
) -> Option<&'a SCFState> {
    if recipe.scfexcitation.is_none() {
        prev_map.get(recipe.label.as_str()).copied()
    } else {
        prev_map.get("RHF").copied()
    }
}

/// Run the real MOM SCF path for one state recipe.
/// # Arguments
/// - `ao`: AO data containing integrals and default RHF density.
/// - `input`: User input controlling SCF and printing.
/// - `recipe`: State recipe to run.
/// - `prev`: Previous real states, used only to decide whether continuation is available.
/// - `prev_map`: Previous real SCF states keyed by label.
/// - `i`: Recipe index used as the parent/state index.
/// # Returns
/// - `SCFState`: Converged real SCF state for this recipe.
pub(crate) fn run_mom_scf_state(
    ao: &AoData,
    input: &Input,
    recipe: &StateRecipe,
    prev: Option<&[SCFState]>,
    prev_map: &HashMap<&str, &SCFState>,
    i: usize,
) -> SCFState {
    if input.write.verbose >= 1 {
        let left = "=".repeat(45);
        let right = "=".repeat(46);
        println!("{}Begin SCF{}", left, right);
        println!("State({}): {}", i + 1, recipe.label);
    };

    // If there are previous states, use them. Ground states are seeded from their previous
    // geometry equivalent, whilst excited states are seeded from RHF at the previous geometry.
    // Any excited states that may or may not be generated here are distinct from those formed
    // in the NOCI-QMC basis. Those formed here form the reference NOCI basis and are relaxed.
    let scfexcitation = recipe.scfexcitation.as_ref();
    let seed = if prev.is_some() {
        previous_mom_seed(recipe, prev_map)
    } else {
        None
    };
    let (da, db) = biased_density_guess(ao, recipe, seed);

    scf_cycle(
        (&da, &db),
        ao,
        input,
        &recipe.label,
        recipe.noci,
        i,
        (scfexcitation, None),
    )
    .expect("SCF did not converge")
}

/// Generate the SCF states using the maximum orbital overlap procedure.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: Contains user inputted options.
/// - `prev`: May or may not contain states from a previous geometry.
/// - `prev_map`: Map between the SCFState object and its label.
/// - `recipes`: Instructions for how to construct each state.
/// # Returns:
/// - `Vec<SCFState>`: Generated SCF states.
pub(crate) fn generate_states_mom(
    ao: &AoData,
    input: &Input,
    prev: Option<&[SCFState]>,
    prev_map: &HashMap<&str, &SCFState>,
    recipes: &[StateRecipe],
) -> Vec<SCFState> {
    let mut out: Vec<SCFState> = Vec::with_capacity(recipes.len());
    for (i, recipe) in recipes.iter().enumerate() {
        if recipe.holomorphic {
            continue;
        }

        let state = run_mom_scf_state(ao, input, recipe, prev, prev_map, i);
        out.push(state);
    }
    mark_duplicate_noci_states(&mut out, &ao.s, input.scf.d_tol, input.write.verbose);
    out
}

/// Generate physical h-SCF states and off-axis tracking states.
/// # Arguments:
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: User input specifications.
/// - `recipes`: State construction recipes.
/// - `real_states`: Real SCF states generated at this geometry.
/// - `prev_htracks`: Tracking states from the previous geometry.
/// # Returns:
/// - `(Vec<HSCFState>, Vec<HSCFState>)`: Physical states and tracking states.
fn generate_hscf_states_mom(
    ao: &AoData,
    input: &Input,
    recipes: &[StateRecipe],
    real_states: &[SCFState],
    prev_htracks: Option<&[HSCFState]>,
) -> (Vec<HSCFState>, Vec<HSCFState>) {
    let real_map: HashMap<&str, &SCFState> = real_states
        .iter()
        .map(|st| (st.label.as_str(), st))
        .collect();

    let prev_track_map: HashMap<&str, &HSCFState> = prev_htracks
        .unwrap_or(&[])
        .iter()
        .map(|st| (st.label.as_str(), st))
        .collect();

    // Start the physical complex basis with all promoted real solutions.
    let mut physical: Vec<HSCFState> = real_states.iter().map(HSCFState::from_real).collect();

    let mut tracks: Vec<HSCFState> = Vec::new();

    for (i, recipe) in recipes.iter().enumerate() {
        if !recipe.holomorphic {
            continue;
        }

        let previous = prev_track_map.get(recipe.label.as_str()).copied();

        let track = if let Some(previous) = previous {
            continue_hscf_track(previous, ao, input, &recipe.label, i)
        } else {
            let seed_label = recipe.partner.as_deref().unwrap_or(recipe.label.as_str());

            let seed = real_map.get(seed_label).copied().unwrap_or_else(|| {
                panic!(
                    "Initial real seed '{}' for holomorphic state '{}' was not generated.",
                    seed_label, recipe.label
                )
            });

            initialise_hscf_track(seed, ao, input, &recipe.label, i)
        }
        .unwrap_or_else(|| panic!("Failed to track holomorphic SCF state '{}'.", recipe.label));

        let state = physical_hscf_state(&track, ao, input, &recipe.label, i, recipe.noci)
            .unwrap_or_else(|| {
                panic!(
                    "Failed to relax holomorphic SCF state '{}' to lambda = 1.",
                    recipe.label
                )
            });

        tracks.push(track);
        physical.push(state);
    }

    mark_duplicate_noci_states(&mut physical, &ao.s, input.scf.d_tol, input.write.verbose);

    (physical, tracks)
}

/// Generate the MOM-backed real and holomorphic reference NOCI basis states.
/// # Arguments
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: Contains user inputted options.
/// - `prev`: Previous real states, if available for continuation.
/// - `prev_h`: Previous h-SCF states, if available for holomorphic continuation.
/// - `prev_map`: Previous real SCF states keyed by label.
/// - `recipes`: Instructions for how to construct each state.
/// # Returns
/// - `ReferenceBasis`: Real SCF states and any complex h-SCF states generated from MOM recipes.
pub(crate) fn generate_reference_basis_mom(
    ao: &AoData,
    input: &Input,
    prev: Option<&[SCFState]>,
    prev_htracks: Option<&[HSCFState]>,
    prev_map: &HashMap<&str, &SCFState>,
    recipes: &[StateRecipe],
) -> ReferenceBasis {
    let states = generate_states_mom(ao, input, prev, prev_map, recipes);
    let (hstates, htracks) = if recipes.iter().any(|recipe| recipe.holomorphic) {
        generate_hscf_states_mom(ao, input, recipes, &states, prev_htracks)
    } else {
        (Vec::new(), Vec::new())
    };

    ReferenceBasis {
        states,
        hstates,
        htracks,
    }
}
