pub mod read;

use ndarray::{Array1, Array2};
pub type Array4 = ndarray::Array<f64, ndarray::Ix4>; // 4D Array for ERIs

// Struct for storing AO integrals
pub struct AoData {
    pub s: Array2<f64>, // AO overlap matrix
    pub h: Array2<f64>, // One electron Hamiltonian matrix 
    pub eri: Array4, // Electron Repulsion Integrals (ERIs)
    pub enuc: f64, // Nuclear repulsion energy 
    pub nao: i64, // Number of AOs 
    pub nelec: Array1<i64>, // Number of spin alpha and spin beta electrons
}
