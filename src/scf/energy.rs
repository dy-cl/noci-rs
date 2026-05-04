// scf/energy.rs

use ndarray::Array2;

/// Calculate the total energy of an SCF state.
/// # Arguments
/// - `da`: Alpha-spin density matrix.
/// - `db`: Beta-spin density matrix.
/// - `fa`: Alpha-spin Fock matrix.
/// - `fb`: Beta-spin Fock matrix.
/// - `h`: One-electron Hamiltonian.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns
/// - `f64`: SCF total energy.
pub(crate) fn scf_energy(da: &Array2<f64>, db: &Array2<f64>, fa: &Array2<f64>, fb: &Array2<f64>, h: &Array2<f64>, enuc: f64) -> f64 {
    let p = da + db;
    let e1 = (h * &p).sum();
    let ea = 0.5 * ((fa - h) * da).sum();
    let eb = 0.5 * ((fb - h) * db).sum();
    e1 + ea + eb + enuc
}
