// noci/orthogonal/connection.rs

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
}
