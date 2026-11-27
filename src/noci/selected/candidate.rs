// noci/selected/candidate.rs

// Standard library imports.
use std::collections::HashSet;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::{Bits, NOCIIndex, NOCISpace};
use crate::input::Input;
use crate::time_call;

pub(in crate::noci::selected) struct CandidatePool {
    /// Current pool candidates.
    pub(in crate::noci::selected) candidates: Vec<NOCIIndex>,
    /// Identities `(parent, occ_\alpha, occ_\beta)` of every selected or candidate determinant.
    /// A determinant is fixed by its parent orbital frame and occupations, whereas its label
    /// records the excitation path that generated it, so the same determinant reached through
    /// two paths shares an identity but not a label.
    known: HashSet<(usize, Bits, Bits)>,
}

impl CandidatePool {
    /// Construct the initial candidate pool of determinants from the current selected space.
    /// Each distinct determinant `(parent, occ_\alpha, occ_\beta)` enters the pool once, and none
    /// that is already selected.
    /// # Arguments
    /// - `space`: Mutable determinant space used to generate candidates.
    /// - `selected_space`: Current selected nonorthogonal determinant space.
    /// - `input`: User-defined input options.
    /// # Returns
    /// - `CandidatePool`: Initial candidate pool containing all distinct generated candidates.
    pub(in crate::noci::selected) fn new<T: NOCIScalar>(
        space: &mut NOCISpace<T>,
        selected_space: &[NOCIIndex],
        input: &Input,
    ) -> Self {
        time_call!(crate::timers::selected::add_candidate_pool_new, {
            let mut pool = Self {
                candidates: Vec::new(),
                known: HashSet::new(),
            };

            // The selected determinants are known before any candidate is generated.
            for &index in selected_space {
                let (oa, ob) = space.occupations(index);
                pool.known.insert((space.parent(index), oa, ob));
            }

            pool.extend(space, selected_space, input);
            pool
        })
    }

    /// Update the candidate pool once the selected space has grown.
    /// # Arguments
    /// - `space`: Mutable determinant space used to generate new candidates.
    /// - `newly_selected`: Determinants added on the most recent SNOCI iteration.
    /// - `input`: User-defined input options.
    /// # Returns
    /// - `()`: Updates the pool in place by removing newly selected states and appending
    ///   determinants not yet selected or in the pool.
    pub(in crate::noci::selected) fn update<T: NOCIScalar>(
        &mut self,
        space: &mut NOCISpace<T>,
        newly_selected: &[NOCIIndex],
        input: &Input,
    ) {
        time_call!(crate::timers::selected::add_candidate_pool_update, {
            if newly_selected.is_empty() {
                return;
            }

            // Selected candidates leave the pool but stay known, so they are never regenerated.
            let selected: HashSet<NOCIIndex> = newly_selected.iter().copied().collect();
            self.candidates.retain(|index| !selected.contains(index));

            self.extend(space, newly_selected, input);
        })
    }

    /// Append the distinct single and double excitations of `generators` that are neither
    /// selected nor already in the pool.
    /// # Arguments
    /// - `space`: Mutable determinant space receiving the new candidates.
    /// - `generators`: Determinants whose excitations are generated.
    /// - `input`: User-defined input options.
    /// # Returns
    /// - `()`: Updates the pool and its known identities in place.
    fn extend<T: NOCIScalar>(
        &mut self,
        space: &mut NOCISpace<T>,
        generators: &[NOCIIndex],
        input: &Input,
    ) {
        let generated = space.excited_from(generators, input, false);

        // `insert` returns false for an identity already selected, already in the pool, or
        // generated earlier in this batch through another excitation path.
        for index in 0..generated.len() {
            let parent = generated.spin_string_pair(NOCIIndex(index)).parent;
            let (oa, ob) = generated.occupations(NOCIIndex(index));
            if !self.known.insert((parent, oa, ob)) {
                continue;
            }

            let label = generated.labels[index].clone();
            self.candidates.push(space.push(parent, oa, ob, label));
        }
    }
}
