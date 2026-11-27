// noci/selected/blocks.rs
//! Hamiltonian, overlap and spin-squared blocks of the selected space.
//!
//! `H`, `S` and `S^2` do not depend on the corrected NOCI state, so the selected-selected blocks are
//! built once and bordered as determinants are promoted, rather than rebuilt every SNOCI iteration.
//! The candidate-current blocks are never stored: the NOCI-PT2 problem needs them only contracted
//! with the current state, which `build_snoci_couplings` evaluates row by row.

// External crate imports.
use ndarray::{Array2, Axis, concatenate, s};

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::NOCIIndex;
use crate::elements::{NOCIData, build_noci_spin_square};
use crate::maths::adjoint;
use crate::noci::build_noci_hs;

/// Hamiltonian, overlap and spin-squared blocks between the selected determinants, in
/// selected-space order.
pub(in crate::noci::selected) struct SNOCIBlocks<T: NOCIScalar> {
    /// Selected-selected Hamiltonian `H_{ij}`.
    pub(in crate::noci::selected) h_ii: Array2<T>,
    /// Selected-selected overlap `S_{ij}`.
    pub(in crate::noci::selected) s_ii: Array2<T>,
    /// Selected-selected spin-squared matrix `\langle i|S^2|j\rangle`.
    pub(in crate::noci::selected) ss_ii: Array2<T>,
}

impl<T: NOCIScalar> SNOCIBlocks<T> {
    /// Build `H_{ij}`, `S_{ij}` and `\langle i|S^2|j\rangle` for a selected space.
    /// # Arguments:
    /// - `data`: Shared NOCI matrix-element data over the determinant space.
    /// - `selected`: Selected determinants.
    /// # Returns:
    /// - `SNOCIBlocks<T>`: Selected-selected blocks.
    pub(in crate::noci::selected) fn new(
        data: &NOCIData<'_, T>,
        selected: &[NOCIIndex],
    ) -> Self {
        let (h_ii, s_ii, _) = build_noci_hs(data, selected, selected, true);
        let ss_ii = build_noci_spin_square(data, selected, selected);

        Self { h_ii, s_ii, ss_ii }
    }

    /// Border the blocks with the `promoted` determinants that have joined the selected space,
    /// `A \leftarrow [[A, P^\dagger], [P, D]]`, where `P` holds the promoted rows against the earlier
    /// selected determinants and `D` the promoted-promoted block.
    /// # Arguments:
    /// - `data`: Shared NOCI matrix-element data over the extended determinant space.
    /// - `selected`: Selected space after the promotion, ending with `promoted`.
    /// - `promoted`: Determinants promoted to the selected space in this iteration.
    /// # Returns:
    /// - `()`: Updates the blocks in place.
    pub(in crate::noci::selected) fn extend(
        &mut self,
        data: &NOCIData<'_, T>,
        selected: &[NOCIIndex],
        promoted: &[NOCIIndex],
    ) {
        let earlier = &selected[..selected.len() - promoted.len()];
        let border = |a: &Array2<T>, p: &Array2<T>, d: &Array2<T>| {
            let top = concatenate![Axis(1), a.view(), adjoint(p).view()];
            let bottom = concatenate![Axis(1), p.view(), d.view()];
            concatenate![Axis(0), top, bottom]
        };

        let (h_pi, s_pi, _) = build_noci_hs(data, promoted, earlier, false);
        let (h_pp, s_pp, _) = build_noci_hs(data, promoted, promoted, true);
        self.h_ii = border(&self.h_ii, &h_pi, &h_pp);
        self.s_ii = border(&self.s_ii, &s_pi, &s_pp);

        let ss_p = build_noci_spin_square(data, promoted, selected);
        let ss_pi = ss_p.slice(s![.., ..earlier.len()]).to_owned();
        let ss_pp = ss_p.slice(s![.., earlier.len()..]).to_owned();
        self.ss_ii = border(&self.ss_ii, &ss_pi, &ss_pp);
    }
}
