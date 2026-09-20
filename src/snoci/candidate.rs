// snoci/candidate.rs

// Standard library imports.
use std::collections::HashSet;

// Crate-root imports.
use crate::input::Input;
use crate::noci::{NOCIIndex, NOCIScalar, NOCISpace};
use crate::time_call;

pub(in crate::snoci) struct CandidatePool {
    /// Current pool candidates.
    pub(in crate::snoci) candidates: Vec<NOCIIndex>,
}

impl CandidatePool {
    /// Construct the initial candidate pool of determinants from the current selected space.
    /// # Arguments
    /// - `space`: Mutable determinant space used to generate candidates.
    /// - `selected_space`: Current selected nonorthogonal determinant space.
    /// - `input`: User-defined input options.
    /// # Returns
    /// - `CandidatePool`: Initial candidate pool containing all generated candidates.
    pub(in crate::snoci) fn new<T: NOCIScalar>(
        space: &mut NOCISpace<T>,
        selected_space: &[NOCIIndex],
        input: &Input,
    ) -> Self {
        time_call!(crate::timers::snoci::add_candidate_pool_new, {
            let generated = space.excited_from(selected_space, input, false);
            let candidates = (0..generated.len())
                .map(|index| {
                    let state = generated.state(NOCIIndex(index));
                    let (oa, ob) = generated.occupations(NOCIIndex(index));
                    space.push(state.parent, oa, ob, generated.labels[index].clone())
                })
                .collect();

            Self { candidates }
        })
    }

    /// Remove any candidates from the pool that have just been selected.
    /// # Arguments
    /// - `selected`: Newly selected determinants that should no longer remain in the pool.
    /// - `space`: Determinant space used to compare candidate labels.
    /// # Returns
    /// - `()`: Updates the candidate pool in place.
    pub(in crate::snoci) fn remove_selected<T: NOCIScalar>(
        &mut self,
        selected: &[NOCIIndex],
        space: &NOCISpace<T>,
    ) {
        let selected_keys: HashSet<&str> = selected
            .iter()
            .map(|&index| space.labels[index.0].as_str())
            .collect();
        self.candidates
            .retain(|&index| !selected_keys.contains(space.labels[index.0].as_str()));
    }

    /// Update the candidate pool once the selected space has grown.
    /// # Arguments
    /// - `space`: Mutable determinant space used to generate new candidates.
    /// - `selected_space`: Updated selected nonorthogonal determinant space.
    /// - `newly_selected`: Determinants added on the most recent SNOCI iteration.
    /// - `input`: User-defined input options.
    /// # Returns
    /// - `()`: Updates the pool in place by removing newly selected states and appending
    ///   genuinely new candidate determinants.
    pub(in crate::snoci) fn update<T: NOCIScalar>(
        &mut self,
        space: &mut NOCISpace<T>,
        selected_space: &[NOCIIndex],
        newly_selected: &[NOCIIndex],
        input: &Input,
    ) {
        time_call!(crate::timers::snoci::add_candidate_pool_update, {
            if newly_selected.is_empty() {
                return;
            }

            self.remove_selected(newly_selected, space);

            // Generate excitations from the newly selected states, excluding
            // labels already present in either the selected or candidate set.
            let generated = space.excited_from(newly_selected, input, false);
            let existing: HashSet<String> = selected_space
                .iter()
                .chain(self.candidates.iter())
                .map(|&index| space.labels[index.0].clone())
                .collect();

            let new_candidates = (0..generated.len())
                .filter(|&index| !existing.contains(generated.labels[index].as_str()))
                .map(|index| {
                    let state = generated.state(NOCIIndex(index));
                    let (oa, ob) = generated.occupations(NOCIIndex(index));
                    space.push(state.parent, oa, ob, generated.labels[index].clone())
                })
                .collect::<Vec<_>>();

            self.candidates.extend(new_candidates);
        })
    }
}
