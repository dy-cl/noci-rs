// noci/selected/guess.rs
//! Continuation of converged NOCI-PT2 amplitudes to the next geometry of a scan.

// Standard library imports.
use std::collections::HashMap;

// External crate imports.
use ndarray::Array1;
use num_complex::Complex64;

// Crate-root imports.
use crate::determinant::{NOCIIndex, NOCISpace};
use crate::{NOCIScalar, StateScalar};

// Parent/sibling imports.
use super::SNOCIPT2Result;

/// Converged NOCI-PT2 amplitudes of one geometry, kept as the GMRES initial guesses of the next.
/// A candidate is identified by the label of its parent reference and its alpha and beta
/// occupations, which the orbital alignment after SCF keeps comparable between geometries, so
/// candidates of references added or removed between geometries simply start from zero.
#[derive(Clone, Default)]
pub struct PT2Guess {
    /// Position of every candidate, keyed by its parent label and alpha and beta occupations.
    candidates: HashMap<(String, u128, u128), usize>,
    /// Corrected NOCI state, imaginary shift, amplitudes `a` and right-hand side `b = -V` of every
    /// solve, ordered as the candidates.
    solves: Vec<(usize, f64, Array1<Complex64>, Array1<Complex64>)>,
}

impl PT2Guess {
    /// Keep the converged amplitudes of every corrected state and imaginary shift.
    /// # Arguments:
    /// - `candidates`: Candidate determinants, ordered as the amplitudes.
    /// - `results`: NOCI-PT2 results of every corrected state and imaginary shift.
    /// # Returns:
    /// - `Self`: Candidate positions with the amplitudes and right-hand sides of every solve.
    pub(in crate::noci::selected) fn new<T: StateScalar>(
        candidates: &NOCISpace<T>,
        results: &[SNOCIPT2Result],
    ) -> Self {
        Self {
            candidates: (0..candidates.len())
                .map(|i| {
                    let (oa, ob) = candidates.occupations(NOCIIndex(i));
                    let label = candidates.parent_determinant(NOCIIndex(i)).label.clone();
                    ((label, oa, ob), i)
                })
                .collect(),
            solves: results
                .iter()
                .map(|r| (r.root, r.imag_shift, r.amplitudes.clone(), r.rhs.clone()))
                .collect(),
        }
    }

    /// Continue the amplitudes of one state and imaginary shift to the candidates of a new
    /// geometry. Candidates absent from the previous geometry start from zero. The NOCI state
    /// is fixed only up to a phase, which carries into both the right-hand side and the
    /// amplitudes, so the amplitudes are rescaled by
    /// `\phi = \langle b_{\text{prev}}, b\rangle / |\langle b_{\text{prev}}, b\rangle|`.
    /// # Arguments:
    /// - `candidates`: Candidate determinants of the new geometry.
    /// - `root`: Corrected NOCI state.
    /// - `imag_shift`: Imaginary shift.
    /// - `rhs`: Right-hand side `b = -V` of the new geometry.
    /// # Returns:
    /// - `Option<Array1<R>>`: Initial guess, or `None` when no candidate is shared.
    pub(in crate::noci::selected) fn continue_to<T, R>(
        &self,
        candidates: &NOCISpace<T>,
        root: usize,
        imag_shift: f64,
        rhs: &Array1<R>,
    ) -> Option<Array1<R>>
    where
        T: StateScalar,
        R: NOCIScalar + Into<Complex64>,
    {
        let (_, _, a, b) = self
            .solves
            .iter()
            .find(|(r, shift, _, _)| *r == root && *shift == imag_shift)?;

        // Position of every new candidate among the previous candidates.
        let matched = (0..candidates.len())
            .map(|i| {
                let (oa, ob) = candidates.occupations(NOCIIndex(i));
                let label = candidates.parent_determinant(NOCIIndex(i)).label.clone();
                self.candidates.get(&(label, oa, ob)).copied()
            })
            .collect::<Vec<_>>();

        // `\langle b_{\text{prev}}, b\rangle` over the shared candidates.
        let overlap = matched
            .iter()
            .zip(rhs.iter())
            .filter_map(|(&k, &bnew)| k.map(|k| b[k].conj() * bnew.into()))
            .sum::<Complex64>();
        if overlap.norm() == 0.0 {
            return None;
        }
        let phase = overlap / overlap.norm();

        Some(
            matched
                .iter()
                .map(|&k| k.map_or(R::from_real(0.0), |k| R::from_complex(phase * a[k])))
                .collect(),
        )
    }
}
