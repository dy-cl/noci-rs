// noci/selected/blocks.rs
//! Hamiltonian and overlap blocks of the selected space and its candidate pool.
//!
//! `H` and `S` do not depend on the corrected NOCI state, so the selected-selected and
//! candidate-selected blocks are built once and extended as determinants are promoted, rather than
//! rebuilt every SNOCI iteration. A promoted candidate's row of `H_{ai}` and `S_{ai}` is already its
//! row against the earlier selected determinants, so only the blocks involving the newly promoted
//! columns or the newly generated candidates are evaluated.

// Standard library imports.
use std::collections::{HashMap, HashSet};

// External crate imports.
use ndarray::{Array2, Axis, concatenate};

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::NOCIIndex;
use crate::elements::NOCIData;
use crate::maths::adjoint;
use crate::noci::build_noci_hs;

/// Hamiltonian and overlap blocks between the selected determinants, and between the candidate pool
/// and the selected determinants. Rows of `h_ai` and `s_ai` follow the candidate pool order and
/// columns follow the selected-space order.
pub(in crate::noci::selected) struct SNOCIBlocks<T: NOCIScalar> {
    /// Selected-selected Hamiltonian `H_{ij}`.
    pub(in crate::noci::selected) h_ii: Array2<T>,
    /// Selected-selected overlap `S_{ij}`.
    pub(in crate::noci::selected) s_ii: Array2<T>,
    /// Candidate-selected Hamiltonian `H_{ai}`.
    pub(in crate::noci::selected) h_ai: Array2<T>,
    /// Candidate-selected overlap `S_{ai}`.
    pub(in crate::noci::selected) s_ai: Array2<T>,
}

impl<T: NOCIScalar> SNOCIBlocks<T> {
    /// Build `H_{ij}`, `S_{ij}`, `H_{ai}` and `S_{ai}` for a selected space and its candidate pool.
    /// # Arguments:
    /// - `data`: Shared NOCI matrix-element data over the determinant space.
    /// - `selected`: Selected determinants, the column order of every block.
    /// - `candidates`: Candidate pool, the row order of the candidate-selected blocks.
    /// # Returns:
    /// - `SNOCIBlocks<T>`: Selected-selected and candidate-selected blocks.
    pub(in crate::noci::selected) fn new(
        data: &NOCIData<'_, T>,
        selected: &[NOCIIndex],
        candidates: &[NOCIIndex],
    ) -> Self {
        let (h_ii, s_ii, _) = build_noci_hs(data, selected, selected, true);
        let (h_ai, s_ai, _) = build_noci_hs(data, candidates, selected, false);

        Self {
            h_ii,
            s_ii,
            h_ai,
            s_ai,
        }
    }

    /// Extend the blocks after `promoted` candidates have joined the selected space and the pool has
    /// been updated. The pool keeps its surviving candidates in their earlier order and appends the
    /// newly generated ones, so with `P` the promoted rows of the earlier `H_{ai}`,
    /// `H_{ij} \leftarrow [[H_{ij}, P^\dagger], [P, H_{pp}]]`, and
    /// `H_{ai} \leftarrow [[H_{ki}, H_{kp}], [H_{ni}]]` for kept candidates `k`, promoted
    /// determinants `p` and new candidates `n`, and likewise for `S`.
    /// # Arguments:
    /// - `data`: Shared NOCI matrix-element data over the extended determinant space.
    /// - `earlier`: Candidate pool before the update, the current row order of `h_ai` and `s_ai`.
    /// - `candidates`: Candidate pool after the update.
    /// - `selected`: Selected space after the promotion, ending with `promoted`.
    /// - `promoted`: Candidates promoted to the selected space in this iteration.
    /// # Returns:
    /// - `()`: Updates the blocks in place.
    pub(in crate::noci::selected) fn extend(
        &mut self,
        data: &NOCIData<'_, T>,
        earlier: &[NOCIIndex],
        candidates: &[NOCIIndex],
        selected: &[NOCIIndex],
        promoted: &[NOCIIndex],
    ) {
        // Rows of the earlier candidate-selected blocks that were promoted, in promotion order, and
        // those that stay in the pool, which lead the updated pool in the same order.
        let row: HashMap<NOCIIndex, usize> =
            earlier.iter().enumerate().map(|(k, &a)| (a, k)).collect();
        let promoted_rows: Vec<usize> = promoted.iter().map(|a| row[a]).collect();

        let promoted_set: HashSet<NOCIIndex> = promoted.iter().copied().collect();
        let kept_rows: Vec<usize> = (0..earlier.len())
            .filter(|&k| !promoted_set.contains(&earlier[k]))
            .collect();
        let (kept, new) = candidates.split_at(kept_rows.len());

        // Border the selected-selected blocks with the promoted rows `P` and their new diagonal
        // block `D`, `[[A, P^\dagger], [P, D]]`.
        let border = |a: &Array2<T>, p: &Array2<T>, d: &Array2<T>| {
            let top = concatenate![Axis(1), a.view(), adjoint(p).view()];
            let bottom = concatenate![Axis(1), p.view(), d.view()];
            concatenate![Axis(0), top, bottom]
        };

        let (h_pp, s_pp, _) = build_noci_hs(data, promoted, promoted, true);
        self.h_ii = border(
            &self.h_ii,
            &self.h_ai.select(Axis(0), &promoted_rows),
            &h_pp,
        );
        self.s_ii = border(
            &self.s_ii,
            &self.s_ai.select(Axis(0), &promoted_rows),
            &s_pp,
        );

        // Kept candidates gain the promoted columns, and new candidates need every column.
        let (h_kp, s_kp, _) = build_noci_hs(data, kept, promoted, false);
        let (h_ni, s_ni, _) = build_noci_hs(data, new, selected, false);

        let h_kept = concatenate![Axis(1), self.h_ai.select(Axis(0), &kept_rows), h_kp];
        let s_kept = concatenate![Axis(1), self.s_ai.select(Axis(0), &kept_rows), s_kp];
        self.h_ai = concatenate![Axis(0), h_kept, h_ni];
        self.s_ai = concatenate![Axis(0), s_kept, s_ni];
    }
}
