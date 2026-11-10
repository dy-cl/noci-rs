// determinant/excited.rs

// Crate-root imports.
use crate::basis::{excitation_between, excitation_phase_bits};
use crate::{Excitation, ExcitationSpin, ReducedOneSpinState, ReducedTwoSpinState, StateScalar};

// Parent/sibling imports.
use super::parent::ParentDeterminant;

/// Determinant in one parent orbital frame identified only by its occupations, so it need not
/// belong to a precomputed determinant space.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ExcitedDeterminant {
    /// Parent orbital frame of the determinant.
    pub(crate) parent: usize,
    /// Alpha occupation bitstring in the parent frame.
    pub(crate) alpha: u128,
    /// Beta occupation bitstring in the parent frame.
    pub(crate) beta: u128,
}

/// An `ExcitedDeterminant` with its parent-relative excitation and numerical payload, prepared
/// once for repeated matrix-element evaluation.
#[derive(Clone, Copy)]
pub(crate) struct ExcitedState {
    /// Determinant in its parent frame.
    pub(crate) det: ExcitedDeterminant,
    /// Holes and particles relative to the parent determinant in each spin.
    pub(crate) excitation: Excitation,
    /// Alpha fermionic phase relative to the parent determinant.
    pub(crate) phase_a: f64,
    /// Beta fermionic phase relative to the parent determinant.
    pub(crate) phase_b: f64,
    /// Combined phase and fixed-rank excitation caches used by numerical kernels.
    pub(crate) reduced: ReducedTwoSpinState,
}

/// One spin component of a prepared determinant: its parent-relative excitation with the phase
/// and fixed-rank cache used by the same-spin nonorthogonal kernels.
#[derive(Clone, Copy)]
pub(crate) struct ExcitedSpinState {
    /// Holes and particles relative to the parent determinant in this spin.
    pub(crate) excitation: ExcitationSpin,
    /// Phase and fixed-rank excitation cache of this spin.
    pub(crate) state: ReducedOneSpinState,
}

impl ExcitedState {
    /// Return the alpha component of the prepared determinant.
    /// # Arguments:
    /// - `self`: Prepared determinant.
    /// # Returns:
    /// - `ExcitedSpinState`: Alpha excitation, phase and fixed-rank cache.
    pub(crate) fn alpha(&self) -> ExcitedSpinState {
        ExcitedSpinState {
            excitation: self.excitation.alpha,
            state: ReducedOneSpinState::new(self.phase_a, self.reduced.excitation_cache.alpha),
        }
    }

    /// Return the beta component of the prepared determinant.
    /// # Arguments:
    /// - `self`: Prepared determinant.
    /// # Returns:
    /// - `ExcitedSpinState`: Beta excitation, phase and fixed-rank cache.
    pub(crate) fn beta(&self) -> ExcitedSpinState {
        ExcitedSpinState {
            excitation: self.excitation.beta,
            state: ReducedOneSpinState::new(self.phase_b, self.reduced.excitation_cache.beta),
        }
    }
}

impl ExcitedDeterminant {
    /// Derive the excitation relative to the parent, `|D\rangle = \phi\,\hat\tau|P\rangle`, with
    /// the phase convention of the retained determinant spaces.
    /// # Arguments:
    /// - `self`: Determinant in its parent frame.
    /// - `parents`: Parent determinants indexed by parent frame.
    /// # Returns:
    /// - `ExcitedState`: Excitation, spin phases and numerical payload.
    pub(crate) fn state<T: StateScalar>(
        &self,
        parents: &[ParentDeterminant<T>],
    ) -> ExcitedState {
        let parent = &parents[self.parent];

        // Each spin's excitation and phase follow from its occupation relative to the parent.
        let spin = |reference: u128, occupation: u128| {
            let (holes, parts) = excitation_between(reference, occupation);
            (
                ExcitationSpin { holes, parts },
                excitation_phase_bits(reference, holes, parts),
            )
        };
        let (alpha, phase_a) = spin(parent.oa, self.alpha);
        let (beta, phase_b) = spin(parent.ob, self.beta);

        let excitation = Excitation { alpha, beta };
        let reduced = ReducedTwoSpinState::new(phase_a * phase_b, excitation.cache());
        ExcitedState {
            det: *self,
            excitation,
            phase_a,
            phase_b,
            reduced,
        }
    }
}
