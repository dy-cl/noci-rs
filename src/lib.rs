// lib.rs
pub mod basis;
pub mod diis;
pub mod noci;
pub mod read;
pub mod scf;
pub mod utils;
pub mod input;
pub mod maths;
pub mod deterministic;
pub mod stochastic;
pub mod mpiutils;
pub mod nonorthogonalwicks;

use std::{sync::Arc};

use serde::{Serialize, Deserialize};

use ndarray::{Array1, Array2, Array4};

/// AO integrals and other related data storage.
pub struct AoData {
    // AO overlap matrix, (nao, nao).
    pub s: Array2<f64>, 
    // Core Hamiltonian matrix, (nao, nao).
    pub h: Array2<f64>,
    // Initial RHF ground state density matrix, (nao, nao). We can build spin biased and
    // excited states from this ansatz.
    pub dm: Array2<f64>, 
    // Coulommb electron repulsion integrals (ERIs) stored as [a, c, b, d].
    pub eri_coul: Array4<f64>,
    // Antisymmetrised electron repulsion integrals (ERIs) in chemists notation stored as [a, c, b, d].
    pub eri_asym: Array4<f64>,
    // Nuclear repulsion energy, scalar.
    pub enuc: f64, 
    // Number of AOs, scalar.
    pub n: usize, 
    // Number of spin alpha and spin beta electrons, (2,).
    pub nelec: Array1<i64>, 
    // AO label strings from PySCF e.g. "0 H 1s"
    pub labels: Vec<String>,
    // Optional FCI calculation energy from PySCF.
    pub e_fci: Option<f64>,
}

// Description of excited determinant relative to reference.
#[derive(Clone, Serialize, Deserialize)]
pub struct Excitation {
    alpha: ExcitationSpin,
    beta: ExcitationSpin,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ExcitationSpin {
    pub holes: Vec<usize>,
    pub parts: Vec<usize>
}

/// Storage for SCF state attributes, contains energy and spin MO coefficients.
#[derive(Clone, Serialize, Deserialize)]
pub struct SCFState {
    // Energy of SCF state in Ha, scalar.
    pub e: f64,  
    // MO occupancy vector for spin a orbitals, (nao,).
    pub oa: Array1<f64>, 
    // MO occupancies for spin b orbitals (nao,).
    pub ob: Array1<f64>, 
    // MO coefficients for spin a electrons, (nao, nao).
    pub ca: Arc<Array2<f64>>,  
    // MO coefficients for spin b electrons, (nao, nao).
    pub cb: Arc<Array2<f64>>,  
    // SCF converged density matrix spin a. 
    pub da: Arc<Array2<f64>>, 
    // SCF converged density matrix spin b. 
    pub db: Arc<Array2<f64>>,
    // Label defined in user input.
    pub label: String,
    // Is this state used in the NOCI basis?
    pub noci_basis: bool,
    // Index of reference parent determinant if excited for QMC basis.
    pub parent: usize,
    // Excitation relative to parent if excited for QMC basis.
    pub excitation: Excitation,
  }
