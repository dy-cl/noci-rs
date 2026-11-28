// elements/rdm/transition.rs
//! Spin-resolved transition overlaps and one-body transition densities of determinant pairs, from
//! which NOCI-state densities and spin-squared matrices are assembled.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::naive::calculate_spin_transition_pair_naive;
use crate::elements::nonorthogonalwicks::{WickScratchSpin, calculate_spin_transition_pair_wicks};
use crate::elements::orthogonal::calculate_spin_transition_pair_orthogonal;
use crate::elements::{DetPair, NOCIData};

/// Spin-resolved transition quantities of a determinant pair `\langle l|\cdot|r\rangle`, with the
/// excitation phases included: the spin overlaps `S^\sigma = \langle l_\sigma|r_\sigma\rangle` and
/// the covariant AO transition densities `\tilde\Gamma^\sigma = \mathbf S W^\sigma \mathbf S`, where
/// `W^\sigma_{\mu\nu}` has the ket orbital on the row and the conjugated bra orbital on the column and
/// carries its own spin overlap. The covariant form is the one the Wick intermediates provide, and
/// the contravariant AO density of the pair is
/// `\langle l|a^\dagger a|r\rangle^\alpha = \mathbf S^{-1}\tilde\Gamma^\alpha\mathbf S^{-1} S^\beta`.
/// A density is `None` where it vanishes or was not needed.
pub(crate) struct SpinTransition<T: NOCIScalar> {
    /// Spin overlaps `(S^\alpha, S^\beta)`.
    pub(crate) s: (T, T),
    /// Covariant transition densities `(\tilde\Gamma^\alpha, \tilde\Gamma^\beta)`.
    pub(crate) gamma: (Option<Array2<T>>, Option<Array2<T>>),
}

impl<T: NOCIScalar> SpinTransition<T> {
    /// Evaluate the spin-resolved transition quantities of one determinant pair. Determinants of
    /// one parent share orthonormal orbitals, so their quantities follow the Slater-Condon rules
    /// directly; different parents use the extended Wick theorem, or the generalised Slater-Condon
    /// rules when no Wick intermediates are available. A density is formed only when the other
    /// spin's overlap is non-zero, unless `cross` asks for both.
    /// # Arguments:
    /// - `data`: Shared NOCI matrix-element data.
    /// - `pair`: Bra and ket determinants.
    /// - `requests`: Every AO index pair `([\mu], [\nu])` in row-major order.
    /// - `scratch`: Scratch space for Wick's calculations.
    /// - `cross`: Whether to form each density whatever the other spin's overlap.
    /// # Returns:
    /// - `SpinTransition<T>`: Spin overlaps and covariant transition densities.
    pub(crate) fn new(
        data: &NOCIData<'_, T>,
        pair: DetPair,
        requests: &[([usize; 1], [usize; 1])],
        scratch: &mut WickScratchSpin<T>,
        cross: bool,
    ) -> Self {
        if pair.bra.parent == pair.ket.parent {
            return calculate_spin_transition_pair_orthogonal(data, pair, cross);
        }

        if data.wicks.is_some() {
            return calculate_spin_transition_pair_wicks(data, pair, requests, scratch, cross);
        }

        calculate_spin_transition_pair_naive(data, pair)
    }
}
