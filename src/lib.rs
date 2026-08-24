// lib.rs
//! Nonorthogonal electronic-structure methods in Rust.
//!
//! `noci-rs` implements electronic-structure calculations in determinant bases whose states
//! may be constructed from independently optimised and therefore mutually nonorthogonal
//! orbital sets. The crate provides real and holomorphic self-consistent field calculations,
//! reference nonorthogonal configuration interaction, deterministic and stochastic propagation
//! in extended NOCI spaces, NOCI-PT2, selected NOCI and feature-gated experimental NOCCMC
//! support.
//!
//! A calculation proceeds from user-defined Lua input and PySCF atomic-orbital data through
//! SCF-state generation, reference-basis selection, molecular-orbital and nonorthogonal Wick
//! preparation, and the requested post-SCF method. The common [`PostSCFData`] structure
//! provides the atomic-orbital data, converged states, selected reference basis, molecular-
//! orbital caches and numerical tolerance shared by these post-SCF calculations.
//!
//! Hamiltonian, overlap, generalised-Fock and transition-density quantities are evaluated
//! using orthogonal Slater-Condon shortcuts, the generalised Slater-Condon rules, or the
//! extended nonorthogonal Wick theorem. MPI provides distributed-memory parallelism and
//! Rayon provides shared-memory parallelism where supported by the selected method.

pub mod basis;
pub mod deterministic;
pub mod driver;
pub mod error;
pub mod input;
pub mod maths;
pub mod mpiutils;
#[cfg(feature = "nocc")]
pub mod nocc;
pub mod noci;
pub mod nonorthogonalwicks;
#[cfg(feature = "nocc")]
pub mod orbitals;
pub mod paths;
pub mod read;
pub mod scalar;
pub mod scf;
pub mod snoci;
pub mod stochastic;
pub mod timers;
pub mod utils;
pub mod write;

// External crate imports.
use ndarray::{Array1, Array2, Array4};
use serde::{Deserialize, Serialize};

// Crate-root imports.
use crate::noci::{MOCache, NOCIScalar};

pub use error::{Error, Result};
pub use scalar::{DetState, HSCFState, SCFState, StateScalar};

pub struct AoData {
    /// AO overlap matrix, (nao, nao).
    pub s: Array2<f64>,
    /// `Löwdin symmetric orthogonaliser X = S^{-1/2}.`
    pub x: Array2<f64>,
    /// Core Hamiltonian matrix, (nao, nao).
    pub h: Array2<f64>,
    /// Initial RHF ground state density matrix, (nao, nao). We can build spin biased and
    /// excited states from this ansatz.
    pub dm: Array2<f64>,
    /// Coulommb electron repulsion integrals (ERIs) stored as [a, c, b, d].
    pub eri_coul: Array4<f64>,
    /// Antisymmetrised electron repulsion integrals (ERIs) in chemists notation stored as [a, c, b, d].
    pub eri_asym: Array4<f64>,
    /// Nuclear repulsion energy, scalar.
    pub enuc: f64,
    /// Number of AOs, scalar.
    pub n: usize,
    /// Number of spin alpha and spin beta electrons, (2,).
    pub nelec: Array1<i64>,
    /// AO label strings from PySCF e.g. "0 H 1s"
    pub labels: Vec<String>,
    /// Optional FCI calculation energy from PySCF.
    pub e_fci: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Excitation {
    /// Excitation information for spin alpha.
    pub alpha: ExcitationSpin,
    /// Excitation information for spin beta.
    pub beta: ExcitationSpin,
}

impl Excitation {
    /// Construct an empty excitation descriptor.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Excitation`: Excitation with no holes or particles in either spin sector.
    pub fn empty() -> Self {
        Self {
            alpha: ExcitationSpin { holes: 0, parts: 0 },
            beta: ExcitationSpin { holes: 0, parts: 0 },
        }
    }

    /// Build the cached fixed-rank representation of this excitation.
    /// # Arguments:
    /// - `self`: Excitation to cache.
    /// # Returns:
    /// - `ExcitationCache`: Cached alpha- and beta-spin excitation data.
    #[inline(always)]
    pub fn cache(&self) -> ExcitationCache {
        ExcitationCache {
            alpha: self.alpha.cache(),
            beta: self.beta.cache(),
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct ExcitationSpin {
    /// Bit mask of previously occupied now unoccupied orbitals.
    pub holes: u128,
    /// Bit mask of previously unoccupied now occupied orbitals.
    pub parts: u128,
}

impl ExcitationSpin {
    /// Build the cached rank and orbital indices used by fixed-rank Wick kernels.
    /// # Arguments:
    /// - `self`: Spin excitation to cache.
    /// # Returns:
    /// - `ExcitationSpinCache`: Cached rank and first four hole/particle indices.
    #[inline(always)]
    pub fn cache(&self) -> ExcitationSpinCache {
        let rank = self.holes.count_ones() as u8;
        debug_assert_eq!(u32::from(rank), self.parts.count_ones());
        let mut indices = [0u8; 8];
        let mut holes = self.holes;
        let mut parts = self.parts;

        // Cache the first four hole and particle indices used by fixed-rank Wick kernels.
        for i in 0..4 {
            if holes != 0 {
                indices[i] = holes.trailing_zeros() as u8;
                holes &= holes - 1;
            }
            if parts != 0 {
                indices[4 + i] = parts.trailing_zeros() as u8;
                parts &= parts - 1;
            }
        }

        ExcitationSpinCache { rank, indices }
    }
}

/// Cached fixed-rank representation of one spin excitation.
#[derive(Clone, Copy, Serialize, Deserialize, Debug, Default)]
pub struct ExcitationSpinCache {
    /// Excitation rank relative to the parent determinant.
    pub rank: u8,
    /// First four hole then particle orbital indices.
    pub indices: [u8; 8],
}

/// Cached fixed-rank representations of both spin excitations.
#[derive(Clone, Copy, Serialize, Deserialize, Debug, Default)]
pub struct ExcitationCache {
    /// Alpha-spin excitation cache.
    pub alpha: ExcitationSpinCache,
    /// Beta-spin excitation cache.
    pub beta: ExcitationSpinCache,
}

/// Reduced one-spin determinant metadata used by fixed-rank determinant-space contractions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReducedOneSpinDetState {
    /// Global determinant index used to recover the full `DetState` when required.
    pub(crate) det: usize,
    /// Fermionic phase `\phi` relative to the parent determinant for this spin sector.
    pub(crate) phase: f64,
    /// Cached excitation rank and orbital labels for this spin sector.
    pub(crate) excitation_cache: ExcitationSpinCache,
}

impl ReducedOneSpinDetState {
    /// Construct reduced determinant metadata for one spin sector.
    /// # Arguments:
    /// - `det`: Global determinant index `I`.
    /// - `phase`: Fermionic phase `\phi_I` relative to the parent determinant.
    /// - `excitation_cache`: Cached excitation rank and orbital labels.
    /// # Returns
    /// - `ReducedOneSpinDetState`: Reduced metadata for determinant `I`.
    #[inline(always)]
    pub(crate) fn new(
        det: usize,
        phase: f64,
        excitation_cache: ExcitationSpinCache,
    ) -> Self {
        Self {
            det,
            phase,
            excitation_cache,
        }
    }

    /// Construct reduced alpha-spin metadata from a full determinant state.
    /// # Arguments:
    /// - `det`: Global determinant index `I`.
    /// - `state`: Full determinant state containing alpha-spin phase and excitation metadata.
    /// # Returns
    /// - `ReducedOneSpinDetState`: Reduced alpha-spin metadata for determinant `I`.
    #[inline(always)]
    pub(crate) fn from_alpha<T: StateScalar>(
        det: usize,
        state: &DetState<T>,
    ) -> Self {
        Self::new(det, state.pha, state.excitation_cache.alpha)
    }

    /// Construct reduced beta-spin metadata from a full determinant state.
    /// # Arguments:
    /// - `det`: Global determinant index `I`.
    /// - `state`: Full determinant state containing beta-spin phase and excitation metadata.
    /// # Returns
    /// - `ReducedOneSpinDetState`: Reduced beta-spin metadata for determinant `I`.
    #[inline(always)]
    pub(crate) fn from_beta<T: StateScalar>(
        det: usize,
        state: &DetState<T>,
    ) -> Self {
        Self::new(det, state.phb, state.excitation_cache.beta)
    }
}

/// Reduced determinant metadata for fixed-rank two-spin contractions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReducedTwoSpinDetState {
    /// Product of alpha- and beta-spin fermionic phases relative to the parent determinant.
    pub(crate) phase: f64,
    /// Cached excitation ranks and orbital labels for both spin sectors.
    pub(crate) excitation_cache: ExcitationCache,
}

impl ReducedTwoSpinDetState {
    /// Construct reduced determinant metadata for both spin sectors.
    /// # Arguments:
    /// - `phase`: Product of alpha- and beta-spin fermionic phases for determinant `I`.
    /// - `excitation_cache`: Cached excitation ranks and orbital labels for both spin sectors.
    /// # Returns
    /// - `ReducedTwoSpinDetState`: Reduced two-spin metadata for determinant `I`.
    #[inline(always)]
    pub(crate) fn new(
        phase: f64,
        excitation_cache: ExcitationCache,
    ) -> Self {
        Self {
            phase,
            excitation_cache,
        }
    }

    /// Construct reduced two-spin metadata from a full determinant state.
    /// # Arguments:
    /// - `state`: Full determinant state containing both spin phases and excitation metadata.
    /// # Returns
    /// - `ReducedTwoSpinDetState`: Reduced two-spin metadata for determinant `I`.
    #[inline(always)]
    pub(crate) fn from_state<T: StateScalar>(state: &DetState<T>) -> Self {
        Self::new(state.pha * state.phb, state.excitation_cache)
    }
}

/// Data shared by post-SCF NOCI, NOCI-QMC, and SNOCI methods.
pub struct PostSCFData<'a, T: NOCIScalar> {
    /// AO integrals and other system data.
    pub ao: &'a AoData,
    /// All converged SCF states generated by the reference-basis routine.
    pub states: &'a [DetState<T>],
    /// SCF states filtered for those requested to be in the NOCI basis.
    pub noci_reference_basis: &'a [DetState<T>],
    /// MO-basis one and two-electron integral caches.
    pub mocache: &'a [MOCache<T>],
    /// Tolerance up to which a number is considered zero.
    pub tol: f64,
}
