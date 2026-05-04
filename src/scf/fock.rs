// scf/fock.rs

use ndarray::{s, Array2, Array4};

/// Build the spin-resolved Fock matrices for UHF.
/// Uses the Coulomb term from the total density and the exchange term from each spin density.
/// If `da = db = 0.5 * dm`, this collapses to the RHF expression.
/// # Arguments
/// - `h`: One-electron Hamiltonian.
/// - `eri`: Two-electron integrals `(pq|rs)` in chemists' notation.
/// - `da`: Alpha-spin density matrix.
/// - `db`: Beta-spin density matrix.
/// # Returns
/// - `(Array2<f64>, Array2<f64>)`: Alpha- and beta-spin Fock matrices.
pub fn form_fock_matrices(h: &Array2<f64>, eri: &Array4<f64>, da: &Array2<f64>, db: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = h.nrows();
    let mut j = Array2::<f64>::zeros((n, n));
    let mut ka = Array2::<f64>::zeros((n, n));
    let mut kb = Array2::<f64>::zeros((n, n));
    let d = da + db;

    for p in 0..n {
        for q in 0..n {
            let block = eri.slice(s![p, q, .., ..]);
            j[(p, q)] = (&block * &d).sum();
        }
    }

    for p in 0..n {
        for q in 0..n {
            let block = eri.slice(s![p, .., q, ..]);
            ka[(p, q)] = (&block * da).sum();
            kb[(p, q)] = (&block * db).sum();
        }
    }

    let fa = h + &j - &ka;
    let fb = h + &j - &kb;
    (fa, fb)
}
