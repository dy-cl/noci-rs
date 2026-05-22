// basis/generate.rs

use std::collections::HashMap;

use crate::input::{Input, StateType};
use crate::{AoData, HSCFState, SCFState};

use super::metadynamics::generate_reference_basis_metadynamics;
use super::mom::generate_reference_basis_mom;
use super::types::ReferenceBasis;

/// Construct a map between SCF-state labels and SCF-state objects.
/// # Arguments
/// - `states`: SCF states to index by label.
/// # Returns
/// - `HashMap<&str, &SCFState>`: SCF states keyed by label.
fn state_map(states: &[SCFState]) -> HashMap<&str, &SCFState> {
    states.iter().map(|st| (st.label.as_str(), st)).collect()
}

/// Generate the requested real and holomorphic reference NOCI basis states.
/// # Arguments
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: Contains user inputted options.
/// - `prev`: Previous real states, if available for continuation.
/// - `prev_h`: Previous h-SCF states, if available for holomorphic continuation.
/// # Returns
/// - `ReferenceBasis`: Real SCF states and any complex h-SCF states generated from the same recipes.
pub fn generate_reference_noci_basis(
    ao: &AoData,
    input: &mut Input,
    prev: Option<&[SCFState]>,
    prev_h: Option<&[HSCFState]>,
) -> ReferenceBasis {
    // Construct lookup table from state label to previous SCF states. Allows for seeding of SCF
    // states at a subsequent geometry to be done by label rather than via index which breaks
    // easily.
    let prev_map = state_map(prev.unwrap_or(&[]));

    // Move states out to allow borrows.
    let mut states = std::mem::replace(&mut input.states, StateType::Mom(Vec::new()));
    let basis = match &mut states {
        StateType::Mom(recipes) => {
            generate_reference_basis_mom(ao, &*input, prev, prev_h, &prev_map, recipes)
        }
        StateType::Metadynamics(meta) => {
            generate_reference_basis_metadynamics(ao, &*input, prev_h, &prev_map, meta)
        }
    };

    // Put back. Note that this is not very idiomatic. Should definitely refactor this somehow.
    input.states = states;
    basis
}
