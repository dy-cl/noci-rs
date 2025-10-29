use ndarray::{Array1, Array2, Array4, s};
use crate::{AoData};
use ndarray_linalg::{Eigh, UPLO};

/// Build the spin-resolved Fock matrices for UHF.
/// Uses the Coulomb term from the total density and the exchange term from each spin density.
/// If dm = 0.5 * (da + db) it collapses to the RHF expression.
/// # Arguments
///     `h`: Array2, one electron Hamiltonian.
///     `eri`: Array4, two electron integrals (pq|rs) in chemist's notation.
///     `da`: Array2, a spin density matrix.
///     `db`: Array2, b spin density matrix
fn form_fock_matrices(h: &Array2<f64>, eri: &Array4<f64>, da: &Array2<f64>, db: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {

    let n = h.nrows();
    let mut j = Array2::<f64>::zeros((n, n));
    let mut ka = Array2::<f64>::zeros((n, n));
    let mut kb = Array2::<f64>::zeros((n, n));
    let d = da + db;

    // J_{pq} = \sum_{rs} (pq|rs) {D_rs}
    for p in 0..n {
        for q in 0..n {
            let block = eri.slice(s![p, q, .., ..]);         
            j[(p, q)] = (&block * &d).sum();
        }
    }

    // K_{pq}^{\alpha} = sum_{rs} (pr|qs) D_{rs}^{\alpha}
    for p in 0..n {
        for q in 0..n {
            let block = eri.slice(s![p, .., q, ..]);        
            ka[(p, q)] = (&block * da).sum();
        }
    }

    // K_{pq}^{\beta} = sum_{rs} (pr|qs) D_{rs}^{\beta}
    for p in 0..n {
        for q in 0..n {
            let block = eri.slice(s![p, .., q, ..]);         
            kb[(p, q)] = (&block * db).sum();
        }
    }  

    let fa = h + &j - &ka;
    let fb = h + &j - &kb;

    (fa, fb)
}

/// Calculate energy of an SCF state. 
/// # Arguments 
///     `da`: Array2, a spin density matrix.
///     `db`: Array2, b spin density matrix.
///     `fa`: Array2, a spin Fock matrix.
///     `fb`: Array2, b spin Fock matrix.
///     `h`: Array2, one electron Hamiltonian. 
///     `Enuc`: f64, nuclear-nuclear repulsion energy.
fn scf_energy(da: &Array2<f64>, db: &Array2<f64>, fa: &Array2<f64>, fb: &Array2<f64>, h: &Array2<f64>, enuc: f64) -> f64 {
    let p = da + db;
    // E_1 = sum_{pq} h_{pq} P_{pq}
    let e1 = (h * &p).sum();
    // E_{\alpha} = 0.5 * sum_{pq} (F_{pq}^{\alpha} - h_{pq}) * D^{alpha}_{pq}
    let ea = 0.5 * ((fa - h) * da).sum();
    // E_{\beta} = 0.5 * sum_{pq} (F_{pq}^{\beta} - h_{pq}) * D^{beta}_{pq}
    let eb = 0.5 * ((fb - h) * db).sum();

    e1 + ea + eb + enuc
}

// Loewdin symmetric orthogonaliser which calculates X = S^{-1/2}
// # Arguments 
//  `s`: Array2, AO basis overlap matrix. Use only lower triangle.
fn loewdin_x(s: &Array2<f64>) -> Array2<f64> {
    // S = U \Lambda U^{\dagger}
    let (lambdas, evecs) = s.eigh(UPLO::Lower).unwrap();
    // Get \Lambda^{-1/2}
    let invsqrt: Array1<f64> = lambdas.mapv(|i| 1.0 / i.sqrt());
    let d = Array2::from_diag(&invsqrt);
    // X = U \Lambda^{-1/2} U^{\dagger}
    evecs.dot(&d).dot(&evecs.t())
}

/// Solve the generalized eigenproblem F C = S C e using Loewdin orthogonalization.
/// # Arguments
///     `f`: Array2, Fock or any Hermitian matrix F.
///     `s`: Array2, AO basis overlap matrix. Use only lower triangle.
fn general_evp(f: &Array2<f64>, s: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    // X = S^{-1/2}
    let x = loewdin_x(&s); 
    // \tilde{F} = X^{\dagger} F X
    let ft = x.t().dot(f).dot(&x);
    // \tilde{F} U = \epsilon U 
    let (epsilon, u) = ft.eigh(UPLO::Lower).unwrap();
    // C = X U 
    let c = x.dot(&u);

    (epsilon, c)
}

/// Unrestricted SCF cycle with Loewdin orthogonalization.
/// Uses AO integrals from AoData struct.
/// Current convergence criteria is just on energy, should be updated to use DIIS.
/// # Arguments
///     `da0`: Array2, initial spin a density matrix.
///     `db0`: Array2, initial spin b density matrix.
///     `ao`: AoData struct, contains AO integrals and metadata.
///     `max_cycle`: Integer, maximum number of SCF cycles.
///     `tol`: Float, convergence threshold.
pub fn scf_cycle(da0: &Array2<f64>, db0: &Array2<f64>, ao: &AoData, max_cycle: i32, tol: f64) -> (f64, Array2<f64>, Array2<f64>){
    
    let h = &ao.h; 
    let eri = &ao.eri;
    let s = &ao.s;
    let enuc = ao.enuc;
    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();

    let mut e = f64::INFINITY;
    let mut da = da0.clone();
    let mut db = db0.clone();
    
    let mut iter = 0;
    while iter < max_cycle {
        
        println!("=======================================================");
        println!("SCF Cycle: {}", iter);
        // Solve FC = SCe
        let (fa, fb) = form_fock_matrices(h, eri, &da, &db);
        let (_ea, ca) = general_evp(&fa, s);
        let (_eb, cb) = general_evp(&fb, s);

        // Select occupied columns based on Aufbau ordering
        // MOM of some sort should be implemented here in future
        let ca_occ = ca.slice(s![.., 0..na]);
        let cb_occ = cb.slice(s![.., 0..nb]);

        let da_new = ca_occ.dot(&ca_occ.t());
        let db_new = cb_occ.dot(&cb_occ.t());

        let (fa_new, fb_new) = form_fock_matrices(h, eri, &da_new, &db_new);
        let e_new = scf_energy(&da_new, &db_new, &fa_new, &fb_new, h, enuc);
        
        println!("Energy: {}", e_new);
        println!("=======================================================");

        if (e_new - e).abs() < tol {
            return (e_new, ca, cb);
        }
        
        da = da_new; 
        db = db_new;
        e = e_new;
        iter += 1;
        
    }
     
    println!("SCF not converged.");
    (f64::NAN, Array2::<f64>::zeros((0, 0)), Array2::<f64>::zeros((0, 0)))
}
