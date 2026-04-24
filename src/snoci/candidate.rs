// snoci/candidate.rs

use std::collections::HashSet;

use ndarray::{Array2, Axis, concatenate};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::{WicksShared};
use crate::noci::{NOCIData};
use crate::time_call;

use crate::basis::generate_excited_basis;
use crate::noci::{build_noci_s};

pub (in crate::snoci) struct CandidatePool {
    // Current pool candidates.
    pub (in crate::snoci) candidates: Vec<SCFState>,
    // Candidate-candidate overlap.
    pub (in crate::snoci) s_ab: Array2<f64>,
    // Candidate-current overlap.
    pub (in crate::snoci) s_ai: Array2<f64>, 
}

impl CandidatePool {
    /// Construct the initial candidate pool of determinants from the current selected space.
    /// # Arguments
    /// - `ao`: AO integrals and other system data.
    /// - `selected_space`: Current selected nonorthogonal determinant space.
    /// - `input`: User-defined input options.
    /// - `wicks`: Optional shared Wick's intermediates.
    /// - `tol`: Tolerance for whether a number is considered zero.
    /// # Returns
    /// - `CandidatePool`: Initial candidate pool containing all generated candidates together
    ///   with candidate-candidate and candidate-current overlap matrices.
    pub (in crate::snoci) fn new(ao: &AoData, selected_space: &[SCFState], input: &Input, wicks: Option<&WicksShared>, tol: f64) -> Self {
        time_call!(crate::timers::snoci::add_candidate_pool_new, {
            let candidates = generate_excited_basis(selected_space, input, false);
            if candidates.is_empty() {
                return Self {
                    candidates,
                    s_ab: Array2::zeros((0, 0)),
                    s_ai: Array2::zeros((0, selected_space.len())),
                };
            }
            
            let wview = wicks.as_ref().map(|ws| ws.view());
            let data = NOCIData::new(ao, &candidates, input, tol, wview);

            let (s_ab, _) = build_noci_s(&data, &candidates, &candidates, true);
            let (s_ai, _) = build_noci_s(&data, &candidates, selected_space, false);

            Self {candidates, s_ab, s_ai}
        })
    }

    /// Remove any candidates from the pool that have just been selected.
    /// # Arguments
    /// - `selected`: Newly selected determinants that should no longer remain in the pool.
    /// # Returns
    /// - `()`: Updates the candidate pool in place.
    pub (in crate::snoci) fn remove_selected(&mut self, selected: &[SCFState]) {
        let selected_keys: HashSet<&str> = selected.iter().map(|st| st.label.as_str()).collect();
        let keep: Vec<usize> = self.candidates.iter().enumerate().filter_map(|(i, st)| (!selected_keys.contains(st.label.as_str())).then_some(i)).collect();
        self.candidates = keep.iter().map(|&i| self.candidates[i].clone()).collect();
        self.s_ab = self.s_ab.select(Axis(0), &keep).select(Axis(1), &keep);
        self.s_ai = self.s_ai.select(Axis(0), &keep);
    }

    /// Remove candidates whose projected norm in the complement of the current selected space is
    /// numerically zero. Such candidates have directions almost entirely accounted for by the
    /// current selected space and would simply add null directions if included.
    /// # Arguments
    /// - `s_ij_inv`: Inverse metric in the current selected space.
    /// - `metric_tol`: Threshold below which projected candidate norms are discarded.
    /// # Returns
    /// - `()`: Updates the candidate pool in place.
    pub (in crate::snoci) fn filter_candidates(&mut self, s_ij_inv: &Array2<f64>, metric_tol: f64) {
        time_call!(crate::timers::snoci::add_candidate_pool_filter_candidates, {
            if self.candidates.is_empty() {return;}
            
            // T_{aj} = S_{ai} S^{ij}.
            let t = self.s_ai.dot(s_ij_inv);
            // S_{ab, \Omega} = S_{ab} - S_{ai} S^{ij} S_{jb}.
            let s_omega_diag = self.s_ab.diag().to_owned() - (&t * &self.s_ai).sum_axis(Axis(1));

            let keep: Vec<usize> = s_omega_diag.iter().enumerate().filter_map(|(a, &d)| (d > metric_tol).then_some(a)).collect();
            if keep.len() == self.candidates.len() {return;}

            self.candidates = keep.iter().map(|&i| self.candidates[i].clone()).collect();
            self.s_ab = self.s_ab.select(Axis(0), &keep).select(Axis(1), &keep);
            self.s_ai = self.s_ai.select(Axis(0), &keep);
        })
    }

    /// Update the candidate pool and overlap matrices once the selected space has grown.
    /// # Arguments
    /// - `ao`: AO integrals and other system data.
    /// - `selected_space`: Updated selected nonorthogonal determinant space.
    /// - `newly_selected`: Determinants added on the most recent SNOCI iteration.
    /// - `input`: User-defined input options.
    /// - `wicks`: Optional shared Wick's intermediates.
    /// - `tol`: Tolerance for whether a number is considered zero.
    /// # Returns
    /// - `()`: Updates the pool in place by removing newly selected states, extending the
    ///   candidate-current overlap block, and appending overlaps for genuinely new candidates.
    pub (in crate::snoci) fn update(&mut self, ao: &AoData, selected_space: &[SCFState], newly_selected: &[SCFState], input: &Input, wicks: Option<&WicksShared>, tol: f64) {
        time_call!(crate::timers::snoci::add_candidate_pool_update, {
            // If nothing was selected there is nothing to be done.
            if newly_selected.is_empty() {return;}
            // Remove new selections from the candidate pool.
            self.remove_selected(newly_selected);

            let wview = wicks.as_ref().map(|ws| ws.view());
            
            // Find candidate-newlyselected overlap and append to existing overlap. 
            if !self.candidates.is_empty() {
                let data_a = NOCIData::new(ao, &self.candidates, input, tol, wview);
                let (s_aj, _) = build_noci_s(&data_a, &self.candidates, newly_selected, false);
                self.s_ai = concatenate(Axis(1), &[self.s_ai.view(), s_aj.view()]).unwrap();
            }
            
            // Generate the candidates corresponding to excitations of the newly selected determinants.
            let mut new_candidates = generate_excited_basis(newly_selected, input, false);
            let existing: HashSet<&str> = selected_space.iter().chain(self.candidates.iter()).map(|st| st.label.as_str()).collect();
            new_candidates.retain(|st| !existing.contains(st.label.as_str()));
            if new_candidates.is_empty() {return;}

            let data_b = NOCIData::new(ao, &new_candidates, input, tol, wview);
            
            // Generate candidate-candidate overlap for new candidates only.
            let (s_bb, _) = build_noci_s(&data_b, &new_candidates, &new_candidates, true);  

            // Get candidate-candidate overlap between new candidates and old candidate pool.
            let s_ba = if self.candidates.is_empty() {
                Array2::<f64>::zeros((new_candidates.len(), 0))
            } else {
                build_noci_s(&data_b, &new_candidates, &self.candidates, false).0
            };

            // Candidate-current overlap between new candidates and current selected space.
            let (s_bi, _) = build_noci_s(&data_b, &new_candidates, selected_space, false);
        
            // Assemble the full new candidate-candidate overlap matrix.
            if self.candidates.is_empty() {
                self.s_ab = s_bb;
                self.s_ai = s_bi;
                self.candidates.extend(new_candidates);
                return;
            }

            self.s_ab = {
                let top = concatenate(Axis(1), &[self.s_ab.view(), s_ba.t().view()]).unwrap();
                let bot = concatenate(Axis(1), &[s_ba.view(), s_bb.view()]).unwrap();
                concatenate(Axis(0), &[top.view(), bot.view()]).unwrap()
            };

            // Assemble full new candidate-current overlap matrix.
            self.s_ai = concatenate(Axis(0), &[self.s_ai.view(), s_bi.view()]).unwrap();
            self.candidates.extend(new_candidates);
        })
    }
}


