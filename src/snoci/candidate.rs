// snoci/candidate.rs

use std::collections::HashSet;

use ndarray::{Array2, Axis};

use crate::{input::Input, SCFState};
use crate::basis::generate_excited_basis;
use crate::time_call;

use super::types::SNOCIOverlaps;

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

    /// Remove candidates whose projected norm in the complement of the current selected space is
    /// numerically zero. Such candidates have directions almost entirely accounted for by the
    /// current selected space and would only add null directions if included.
    /// # Arguments
    /// - `overlaps`: Candidate overlap blocks to filter consistently with the candidate list.
    /// - `s_ij_inv`: Inverse metric in the current selected space.
    /// - `metric_tol`: Threshold below which projected candidate norms are discarded.
    /// # Returns
    /// - `()`: Updates the candidate pool and overlap blocks in place.
    pub(in crate::snoci) fn filter_candidates(&mut self, overlaps: &mut SNOCIOverlaps, s_ij_inv: &Array2<f64>, metric_tol: f64) {
        time_call!(crate::timers::snoci::add_candidate_pool_filter_candidates, {
            if self.candidates.is_empty() {return;}

            let t = overlaps.s_ai.dot(s_ij_inv);
            let s_omega_aa = overlaps.s_ab.diag().to_owned() - (&t * &overlaps.s_ai).sum_axis(Axis(1));

            let keep: Vec<usize> = s_omega_aa.iter()
                .enumerate()
                .filter_map(|(a, &d)| (d > metric_tol).then_some(a))
                .collect();

            if keep.len() == self.candidates.len() {return;}

            self.candidates = keep.iter().map(|&i| self.candidates[i].clone()).collect();
            overlaps.s_ab = compact_square(&overlaps.s_ab, &keep);
            overlaps.s_ai = compact_rows(&overlaps.s_ai, &keep);
            overlaps.s_ia = compact_cols(&overlaps.s_ia, &keep);
        })
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

/// Return a square matrix containing only the selected rows and columns.
/// # Arguments
/// - `m`: Input square matrix.
/// - `keep`: Row and column indices to retain, in the desired output order.
/// # Returns
/// - `Array2<f64>`: Compacted square matrix with shape `(keep.len(), keep.len())`.
fn compact_square(m: &Array2<f64>, keep: &[usize]) -> Array2<f64> {
    let n = keep.len();
    let mut out = Array2::zeros((n, n));
    for (ii, &i) in keep.iter().enumerate() {
        for (jj, &j) in keep.iter().enumerate() {
            out[(ii, jj)] = m[(i, j)];
        }
    }
    out
}

/// Return a matrix containing only the selected rows.
/// # Arguments
/// - `m`: Input matrix.
/// - `keep`: Row indices to retain, in the desired output order.
/// # Returns
/// - `Array2<f64>`: Compacted row-selected matrix with shape `(keep.len(), m.ncols())`.
fn compact_rows(m: &Array2<f64>, keep: &[usize]) -> Array2<f64> {
    let mut out = Array2::zeros((keep.len(), m.ncols()));
    for (ii, &i) in keep.iter().enumerate() {
        out.row_mut(ii).assign(&m.row(i));
    }
    out
}

/// Return a matrix containing only the selected columns.
/// # Arguments
/// - `m`: Input matrix.
/// - `keep`: Column indices to retain, in the desired output order.
/// # Returns
/// - `Array2<f64>`: Compacted column-selected matrix with shape `(m.nrows(), keep.len())`.
fn compact_cols(m: &Array2<f64>, keep: &[usize]) -> Array2<f64> {
    let mut out = Array2::zeros((m.nrows(), keep.len()));
    for (jj, &j) in keep.iter().enumerate() {
        out.column_mut(jj).assign(&m.column(j));
    }
    out
}
