// lib.rs
pub mod basis;
pub mod deterministic;
pub mod input;
pub mod maths;
pub mod mpiutils;
pub mod noci;
pub mod nonorthogonalwicks;
pub mod read;
pub mod scalar;
pub mod scf;
pub mod snoci;
pub mod stochastic;
pub mod timers;
pub mod utils;
pub mod write;

use serde::{Deserialize, Serialize};

use ndarray::{Array1, Array2, Array4};

pub use scalar::{DetState, HSCFState, SCFState, StateScalar};

pub struct AoData {
    /// AO overlap matrix, (nao, nao).
    pub s: Array2<f64>,
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
    alpha: ExcitationSpin,
    /// Excitation information for spin beta.
    beta: ExcitationSpin,
}

impl Excitation {
    /// Construct an empty excitation descriptor.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Excitation`: Excitation with no holes or particles in either spin sector.
    pub fn empty() -> Self {
        Self {
            alpha: ExcitationSpin {
                holes: vec![],
                parts: vec![],
            },
            beta: ExcitationSpin {
                holes: vec![],
                parts: vec![],
            },
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ExcitationSpin {
    /// List of previously occupied now unoccupied orbitals.
    pub holes: Vec<usize>,
    /// List of previously unoccupied now occupied orbitals.
    pub parts: Vec<usize>,
}
