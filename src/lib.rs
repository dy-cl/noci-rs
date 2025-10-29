pub mod read;
pub mod scf; 
pub mod basis;

use ndarray::{Array1, Array2, Array4};

// Struct for storing AO integrals
pub struct AoData {
    pub s: Array2<f64>, // AO overlap matrix
    pub h: Array2<f64>, // One electron Hamiltonian matrix
    pub dm: Array2<f64>, // Density matrix
    pub eri: Array4<f64>, // Electron Repulsion Integrals (ERIs)
    pub enuc: f64, // Nuclear repulsion energy 
    pub nao: i64, // Number of AOs 
    pub nelec: Array1<i64>, // Number of spin alpha and spin beta electrons
    pub aolabels: Array2<i64> // Global AO index labels for each atom
}

pub struct SCFState {
    pub e: f64, // Energy of SCF state
    pub ca: Array2<f64>, // MO coefficients for spin alpha electrons 
    pub cb: Array2<f64>, // MO coefficients for spin beta electrons
}
