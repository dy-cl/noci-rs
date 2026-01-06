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

use std::sync::Arc;

use ndarray::{Array1, Array2, Array4};

/// AO integrals and other related data storage.
pub struct AoData {
    // AO overlap matrix, (nao, nao).
    pub s_ao: Array2<f64>, 
    // Spin block diagonal AO overlap matrix [[s_ao, 0], [0, s_ao]], (2 * nao, 2 * nao). 
    pub s_spin: Array2<f64>,  
    // Core Hamiltonian matrix, (nao, nao).
    pub h: Array2<f64>,
    // Core Hamiltonian matrix in spin block diagonal variant, (2 * nao, 2 * nao).
    pub h_spin: Array2<f64>,
    // Initial RHF ground state density matrix, (nao, nao). We can build spin biased and
    // excited states from this ansatz.
    pub dm: Array2<f64>, 
    // Electron repulsion integrals (ERIs) in chemists notation, (nao, nao, nao, nao).
    pub eri: Array4<f64>,
    // Antisymmetrised ERIs in spin orbital space, (2 * nao, 2 * nao, 2 * nao, 2 * nao).
    pub eri_spin: Array4<f64>,
    // Nuclear repulsion energy, scalar.
    pub enuc: f64, 
    // Number of AOs, scalar.
    pub nao: usize, 
    // Number of spin alpha and spin beta electrons, (2,).
    pub nelec: Array1<i64>, 
    // AO label strings from PySCF e.g. "0 H 1s"
    pub aolabels: Vec<String>,
    // Optional FCI calculation energy from PySCF.
    pub e_fci: Option<f64>,
}

/// Storage for SCF state attributes, contains energy and spin MO coefficients.
#[derive(Clone)]
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
    // MO coefficients in spin diagonal block [[ca, 0], [0, cb]], (2 * nao, 2 * nao).
    pub cs: Arc<Array2<f64>>, 
    // Occupied only MO coefficients of the spin digonal matrix, (2 * nao, nocca + noccb).
    pub cs_occ: Array2<f64>,
    // SCF converged density matrix spin a. 
    pub da: Arc<Array2<f64>>, 
    // SCF converged density matrix spin b. 
    pub db: Arc<Array2<f64>>,
    // Label defined in user input.
    pub label: String,
    // Is this state used in the NOCI basis?
    pub noci_basis: bool,
  }
