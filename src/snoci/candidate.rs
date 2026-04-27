// snoci/candidate.rs

use std::collections::HashSet;

use crate::{input::Input, SCFState};
use crate::basis::generate_excited_basis;
use crate::time_call;

pub(in crate::snoci) struct CandidatePool {
    /// Current pool candidates.
    pub(in crate::snoci) candidates: Vec<SCFState>,
}

impl CandidatePool {
    /// Construct the initial candidate pool of determinants from the current selected space.
    /// # Arguments
    /// - `selected_space`: Current selected nonorthogonal determinant space.
    /// - `input`: User-defined input options.
    /// # Returns
    /// - `CandidatePool`: Initial candidate pool containing all generated candidates.
    pub(in crate::snoci) fn new(selected_space: &[SCFState], input: &Input) -> Self {
        time_call!(crate::timers::snoci::add_candidate_pool_new, {
            let candidates = generate_excited_basis(selected_space, input, false);
            Self {candidates}
        })
    }

    /// Remove any candidates from the pool that have just been selected.
    /// # Arguments
    /// - `selected`: Newly selected determinants that should no longer remain in the pool.
    /// # Returns
    /// - `()`: Updates the candidate pool in place.
    pub(in crate::snoci) fn remove_selected(&mut self, selected: &[SCFState]) {
        let selected_keys: HashSet<&str> = selected.iter().map(|st| st.label.as_str()).collect();
        self.candidates.retain(|st| !selected_keys.contains(st.label.as_str()));
    }

    /// Update the candidate pool once the selected space has grown.
    /// # Arguments
    /// - `selected_space`: Updated selected nonorthogonal determinant space.
    /// - `newly_selected`: Determinants added on the most recent SNOCI iteration.
    /// - `input`: User-defined input options.
    /// # Returns
    /// - `()`: Updates the pool in place by removing newly selected states and appending
    ///   genuinely new candidate determinants.
    pub(in crate::snoci) fn update(&mut self, selected_space: &[SCFState], newly_selected: &[SCFState], input: &Input) {
        time_call!(crate::timers::snoci::add_candidate_pool_update, {
            if newly_selected.is_empty() {return;}

            self.remove_selected(newly_selected);

            let mut new_candidates = generate_excited_basis(newly_selected, input, false);
            let existing: HashSet<&str> = selected_space.iter()
                .chain(self.candidates.iter())
                .map(|st| st.label.as_str())
                .collect();

            new_candidates.retain(|st| !existing.contains(st.label.as_str()));
            self.candidates.extend(new_candidates);
        })
    }
}

