// noci/orthogonal/connection.rs

// Crate-root imports.
use crate::basis::excitation_phase_bits;
use crate::determinant::SpinDeterminantState;
use crate::{ExcitationCache, ExcitationSpinCache, ReducedTwoSpinState};

/// Relative occupied/virtual-rank topology of one orthogonal Hamiltonian connection.
#[derive(Clone, Copy, Debug)]
pub(crate) enum OrthogonalConnection {
    /// Replace one alpha electron.
    AlphaSingle {
        /// Rank of the removed orbital among occupied alpha orbitals.
        occupied: u8,
        /// Rank of the inserted orbital among virtual alpha orbitals.
        virtual_: u8,
    },
    /// Replace one beta electron.
    BetaSingle {
        /// Rank of the removed orbital among occupied beta orbitals.
        occupied: u8,
        /// Rank of the inserted orbital among virtual beta orbitals.
        virtual_: u8,
    },
    /// Replace two alpha electrons.
    AlphaDouble {
        /// Ranks of the removed orbitals among occupied alpha orbitals.
        occupied: [u8; 2],
        /// Ranks of the inserted orbitals among virtual alpha orbitals.
        virtual_: [u8; 2],
    },
    /// Replace two beta electrons.
    BetaDouble {
        /// Ranks of the removed orbitals among occupied beta orbitals.
        occupied: [u8; 2],
        /// Ranks of the inserted orbitals among virtual beta orbitals.
        virtual_: [u8; 2],
    },
    /// Replace one alpha and one beta electron.
    AlphaBetaDouble {
        /// Rank of the removed alpha orbital among occupied alpha orbitals.
        occupied_a: u8,
        /// Rank of the inserted alpha orbital among virtual alpha orbitals.
        virtual_a: u8,
        /// Rank of the removed beta orbital among occupied beta orbitals.
        occupied_b: u8,
        /// Rank of the inserted beta orbital among virtual beta orbitals.
        virtual_b: u8,
    },
}

impl OrthogonalConnection {
    /// Return the fixed-rank numerical sector of this connection.
    /// # Arguments:
    /// - `self`: Relative orthogonal connection topology.
    /// # Returns:
    /// - `usize`: Alpha single, beta single, alpha-alpha, beta-beta, or alpha-beta sector.
    #[inline(always)]
    pub(crate) fn sector(self) -> usize {
        match self {
            Self::AlphaSingle { .. } => 0,
            Self::BetaSingle { .. } => 1,
            Self::AlphaDouble { .. } => 2,
            Self::BetaDouble { .. } => 3,
            Self::AlphaBetaDouble { .. } => 4,
        }
    }

    /// Resolve occupied/virtual ranks into the fixed-rank physical-orbital cache.
    /// # Arguments:
    /// - `self`: Compact source-relative connection.
    /// - `alpha`: Canonical alpha source component.
    /// - `beta`: Canonical beta source component.
    /// # Returns:
    /// - `ExcitationCache`: Physical hole and particle orbital labels.
    #[inline(always)]
    pub(crate) fn excitation_cache(
        self,
        alpha: &SpinDeterminantState,
        beta: &SpinDeterminantState,
    ) -> ExcitationCache {
        let mut alpha_cache = ExcitationSpinCache::default();
        let mut beta_cache = ExcitationSpinCache::default();
        match self {
            Self::AlphaSingle { occupied, virtual_ } => {
                alpha_cache.rank = 1;
                alpha_cache.holes[0] = alpha.occupied[occupied as usize];
                alpha_cache.particles[0] = alpha.virtuals[virtual_ as usize];
            }
            Self::BetaSingle { occupied, virtual_ } => {
                beta_cache.rank = 1;
                beta_cache.holes[0] = beta.occupied[occupied as usize];
                beta_cache.particles[0] = beta.virtuals[virtual_ as usize];
            }
            Self::AlphaDouble { occupied, virtual_ } => {
                alpha_cache.rank = 2;
                alpha_cache.holes[0] = alpha.occupied[occupied[0] as usize];
                alpha_cache.holes[1] = alpha.occupied[occupied[1] as usize];
                alpha_cache.particles[0] = alpha.virtuals[virtual_[0] as usize];
                alpha_cache.particles[1] = alpha.virtuals[virtual_[1] as usize];
            }
            Self::BetaDouble { occupied, virtual_ } => {
                beta_cache.rank = 2;
                beta_cache.holes[0] = beta.occupied[occupied[0] as usize];
                beta_cache.holes[1] = beta.occupied[occupied[1] as usize];
                beta_cache.particles[0] = beta.virtuals[virtual_[0] as usize];
                beta_cache.particles[1] = beta.virtuals[virtual_[1] as usize];
            }
            Self::AlphaBetaDouble {
                occupied_a,
                virtual_a,
                occupied_b,
                virtual_b,
            } => {
                alpha_cache.rank = 1;
                alpha_cache.holes[0] = alpha.occupied[occupied_a as usize];
                alpha_cache.particles[0] = alpha.virtuals[virtual_a as usize];
                beta_cache.rank = 1;
                beta_cache.holes[0] = beta.occupied[occupied_b as usize];
                beta_cache.particles[0] = beta.virtuals[virtual_b as usize];
            }
        }
        ExcitationCache {
            alpha: alpha_cache,
            beta: beta_cache,
        }
    }

    /// Evaluate the source-relative fermionic phase and fixed-rank H payload.
    /// # Arguments:
    /// - `self`: Compact source-relative connection.
    /// - `alpha`: Canonical alpha source component.
    /// - `beta`: Canonical beta source component.
    /// # Returns:
    /// - `ReducedTwoSpinState`: Numerical payload for the orthogonal H kernel.
    #[inline(always)]
    pub(crate) fn reduced(
        self,
        alpha: &SpinDeterminantState,
        beta: &SpinDeterminantState,
    ) -> ReducedTwoSpinState {
        let cache = self.excitation_cache(alpha, beta);
        let phase_a = excitation_phase_bits(
            alpha.occupation,
            fixed_rank_mask(cache.alpha.holes, cache.alpha.rank),
            fixed_rank_mask(cache.alpha.particles, cache.alpha.rank),
        );
        let phase_b = excitation_phase_bits(
            beta.occupation,
            fixed_rank_mask(cache.beta.holes, cache.beta.rank),
            fixed_rank_mask(cache.beta.particles, cache.beta.rank),
        );
        ReducedTwoSpinState::new(phase_a * phase_b, cache)
    }

    /// Construct physical child occupations from the resolved connection cache.
    /// # Arguments:
    /// - `self`: Compact source-relative connection.
    /// - `alpha`: Canonical alpha source component.
    /// - `beta`: Canonical beta source component.
    /// # Returns:
    /// - `(u128, u128)`: Physical alpha and beta child occupations.
    #[inline(always)]
    pub(crate) fn child_occupations(
        self,
        alpha: &SpinDeterminantState,
        beta: &SpinDeterminantState,
    ) -> (u128, u128) {
        let cache = self.excitation_cache(alpha, beta);
        (
            (alpha.occupation & !fixed_rank_mask(cache.alpha.holes, cache.alpha.rank))
                | fixed_rank_mask(cache.alpha.particles, cache.alpha.rank),
            (beta.occupation & !fixed_rank_mask(cache.beta.holes, cache.beta.rank))
                | fixed_rank_mask(cache.beta.particles, cache.beta.rank),
        )
    }
}

/// Convert fixed-rank orbital labels to the physical occupation mask.
/// # Arguments:
/// - `orbitals`: Cached physical orbital labels.
/// - `rank`: Number of active labels.
/// # Returns:
/// - `u128`: Mask with exactly the active orbital labels.
#[inline(always)]
fn fixed_rank_mask(
    orbitals: [u8; crate::MAXEXCIT],
    rank: u8,
) -> u128 {
    let mut bits = 0u128;
    for &orbital in orbitals.iter().take(usize::from(rank)) {
        bits |= 1u128 << orbital;
    }
    bits
}
