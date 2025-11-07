use std::usize;

// scf.rs
use ndarray::{Axis, Array1, Array2, Array4, s};

use crate::AoData;
use crate::input::{Input, Spin, Excitation};
use crate::diis::Diis;

use crate::maths::general_evp_real;

/// Build the spin-resolved Fock matrices for UHF.
/// Uses the Coulomb term from the total density and the exchange term from each spin density.
/// If dm = 0.5 * (da + db) it collapses to the RHF expression.
/// # Arguments
///     `h`: Array2, one electron Hamiltonian.
///     `eri`: Array4, two electron integrals (pq|rs) in chemist's notation.
///     `da`: Array2, a spin density matrix.
///     `db`: Array2, b spin density matrix
fn form_fock_matrices(h: &Array2<f64>, eri: &Array4<f64>, da: &Array2<f64>, db: &Array2<f64>) 
                      -> (Array2<f64>, Array2<f64>) {

    let n = h.nrows();
    let mut j = Array2::<f64>::zeros((n, n));
    let mut ka = Array2::<f64>::zeros((n, n));
    let mut kb = Array2::<f64>::zeros((n, n));
    let d = da + db;

    // J_{pq} = \sum_{rs} (pq|rs) {D_rs}.
    for p in 0..n {
        for q in 0..n {
            let block = eri.slice(s![p, q, .., ..]);         
            j[(p, q)] = (&block * &d).sum();
        }
    }

    // K_{pq}^{\alpha} = sum_{rs} (pr|qs) D_{rs}^{\alpha}.
    for p in 0..n {
        for q in 0..n {
            let block = eri.slice(s![p, .., q, ..]);        
            ka[(p, q)] = (&block * da).sum();
        }
    }

    // K_{pq}^{\beta} = sum_{rs} (pr|qs) D_{rs}^{\beta}.
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
fn scf_energy(da: &Array2<f64>, db: &Array2<f64>, fa: &Array2<f64>,     
              fb: &Array2<f64>, h: &Array2<f64>, enuc: f64) -> f64 {
    let p = da + db;
    // E_1 = sum_{pq} h_{pq} P_{pq}.
    let e1 = (h * &p).sum();
    // E_{\alpha} = 0.5 * sum_{pq} (F_{pq}^{\alpha} - h_{pq}) * D^{alpha}_{pq}.
    let ea = 0.5 * ((fa - h) * da).sum();
    // E_{\beta} = 0.5 * sum_{pq} (F_{pq}^{\beta} - h_{pq}) * D^{beta}_{pq}.
    let eb = 0.5 * ((fb - h) * db).sum();

    e1 + ea + eb + enuc
}

/// Calculate MO occupancies as diagonal of a density matrix in the MO basis.
/// T = C^{\dagger} S D S C.
/// # Arguments
///     `c`: Array2, spin MO coefficient matrix.
///     `d`: Array2, spin density matrix.
///     `s`: Array2, AO overlap matrix.
fn mo_occupancies(c: &Array2<f64>, d: &Array2<f64>, s: &Array2<f64>) -> Array1<f64> {
    let t = c.t().dot(s).dot(d).dot(s).dot(c);
    let diag = t.diag().to_owned();
    diag.mapv(|x| if x > 0.5 { 1.0 } else { 0.0 })
}

/// Select occupied MO column indices by the Maximum Overlap Method (MOM). Forms the overlap 
/// score O = C_old^T S C between iteration k-1 occupied orbitals and iteration k orbitals. Each 
/// MO is scored according to its overlap with the previous iteration. Provided a decent initial 
/// guess of the excited state this should converge to a desired state.
/// # Arguments 
///     `c_occ_old`: Array2, previous iterations occupied MO coefficients.
///     `c`: Array2, current iterations MO coefficients.
///     `s`: Array2, AO overlap matrix.
///     `nocc`: Number of occupied spin orbitals to select.
fn mom_select(c_occ_old: &Array2<f64>, c: &Array2<f64>, s: &Array2<f64>, nocc: usize)
              -> Vec<usize> {
    // O = C_old^T S C.
    let o = c_occ_old.t().dot(s).dot(c);
    // p_j = \sum_i |O_{ij}|.
    let p = o.mapv(|x| x.abs()).sum_axis(Axis(0));
    // [0, 1, 2, .., nmo - 1].
    let mut idx: Vec<usize> = (0..p.len()).collect();
    // Sort indices by descending value of p.
    idx.sort_by(|&i, &j| p[j].partial_cmp(&p[i]).unwrap());
    // Return the highest nocc overlaps as the occupied indices.
    idx.truncate(nocc);
    idx
}

/// Unrestricted SCF cycle with Loewdin orthogonalization.
/// Uses AO integrals from AoData struct.
/// # Arguments
///     `da0`: Array2, initial spin a density matrix.
///     `db0`: Array2, initial spin b density matrix.
///     `ao`: AoData struct, contains AO integrals and metadata.
///     `input`: Input struct, contains user specified input data. 
pub fn scf_cycle(da0: &Array2<f64>, db0: &Array2<f64>, ao: &AoData, input: &Input, 
                 excitation: Option<&Excitation>) 
                 -> (f64, Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>){
    
    let h = &ao.h; 
    let eri = &ao.eri;
    let s = &ao.s_ao;
    let enuc = ao.enuc;
    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();

    let mut e = f64::INFINITY;
    let mut da = da0.clone();
    let mut db = db0.clone();
    let use_diis = true;

    // DIIS setup: subspace up to 8.
    let mut diis_a = Diis::new(8);
    let mut diis_b = Diis::new(8);

    let mut ca_occ_old: Option<Array2<f64>> = None;
    let mut cb_occ_old: Option<Array2<f64>> = None;
    
    if input.verbose {
        match excitation {
            Some(ex) => {
                let sp = match ex.spin {
                    Spin::Alpha => "alpha",
                    Spin::Beta  => "beta",
                };
                println!("Requested excitation: [spin: {}, from occupied: {}, to virtual: {}]", 
                          sp, ex.occ, ex.vir);
            }
            None => {
                println!("No excitation requested.");
            }
        }
        println!("{:>4} {:>12} {:>12} {:>12}", "i", "E", "dE", "‖FDS - SDF‖");
    }
   
    let mut iter = 0;
    while iter < input.max_cycle {
        
        // Build current Fock matrices from current densities.
        let (fa_curr, fb_curr) = form_fock_matrices(h, eri, &da, &db);
        
        // Update DIIS with current F and D and attempt extrapolation if using.
        let fa = if use_diis {
            diis_a.push(&fa_curr, &da, s);
            match diis_a.extrapolate_fock() {
                Some(fa_diis) => fa_diis, // Succesful extrapolation. 
                None => fa_curr,
            }
        } else {
            fa_curr.clone()
        };
        let fb = if use_diis {
            diis_b.push(&fb_curr, &db, s);
            match diis_b.extrapolate_fock() {
                Some(fb_diis) => fb_diis, // Succesful extrapolation. 
                None => fb_curr,
            }
        } else {
            fb_curr.clone()
        };

        // Solve GEVP FC = SCe.
        let (_ea, ca) = general_evp_real(&fa, s, false, 1e-8);
        let (_eb, cb) = general_evp_real(&fb, s, false, 1e-8);
        
        // For alpha spin occupied MOs decide whether or not to excite.
        let idx_a: Vec<usize> = match &ca_occ_old {
            // If we have MOs from previous iteration use those in MOM.
            Some(c_old) => mom_select(c_old, &ca, s, na),
            // If not, seed by occupying columns specified in input. 
            None => {
                // Note this only occurs when an excitation is specified.
                if let Some(ex) = excitation {
                    if let Spin::Alpha = ex.spin {
                        let mut idx: Vec<usize> = (0..na).collect(); 
                        // Transform user input, allows user to specify occ = -1
                        // to choose HOMO.
                        let occ_abs = (nb as i32 + ex.occ) as usize;
                        let vir_abs = nb + ex.vir as usize;
                        // Remove occupied index.
                        idx.retain(|&k| k != occ_abs);
                        // Add excited index.
                        idx.push(vir_abs);
                        idx
                    // If an excitation is not specified for alpha spin electron 
                    // we just use Aufbau. 
                    } else {
                        (0..na).collect()
                    }
                // If an excitation is not specified at all we use Aufbau.
                } else {
                    (0..na).collect()
                }
            }
        };
        let ca_occ = ca.select(Axis(1), &idx_a);
        ca_occ_old = Some(ca_occ.clone());
        
        // For beta spin occupied MOs decide whether or not to excite.
        let idx_b: Vec<usize> = match &cb_occ_old {
            // If we have MOs from previous iteration use those in MOM.
            Some(c_old) => mom_select(c_old, &cb, s, nb),
            // If not, seed by occupying columns specified in input.
            None => {
                // Note this only occurs when an excitation is specified.
                if let Some(ex) = excitation {
                    if let Spin::Beta = ex.spin {
                        let mut idx: Vec<usize> = (0..nb).collect();
                        // Transform user input, allows user to specify occ = -1
                        // to choose HOMO.
                        let occ_abs = (nb as i32 + ex.occ) as usize;
                        let vir_abs = nb + ex.vir as usize;
                        // Remove occupied index.
                        idx.retain(|&k| k != occ_abs);
                        // Add excited index.
                        idx.push(vir_abs);
                        idx
                    // If an excitation is not specified for beta spin electron 
                    // we just use Aufbau.
                    } else {
                        (0..nb).collect()
                    }
                // If an excitation is not specified at all we use Aufbau.
                } else {
                    (0..nb).collect()
                }
            }
        };
        let cb_occ = cb.select(Axis(1), &idx_b);
        cb_occ_old = Some(cb_occ.clone());

        // Form new spin specific densities, Fock matrices, and energy.
        let da_new = ca_occ.dot(&ca_occ.t());
        let db_new = cb_occ.dot(&cb_occ.t());
        let (fa_new, fb_new) = form_fock_matrices(h, eri, &da_new, &db_new);
        let e_new = scf_energy(&da_new, &db_new, &fa_new, &fb_new, h, enuc);

        // Calculate current occupancies from density matrices and MO coefficients.
        // Obviously since we are currently just doing Aufbau we could hard code the occupations,
        // but this will be useful to have in future when using MOM to get excited states.
        let oa = mo_occupancies(&ca, &da_new, s);
        let ob = mo_occupancies(&cb, &db_new, s);

        // Calculate DIIS error term and dE for convergence testing. 
        // If no error can be calculated  (i.e., DIIS subspace is 1) or DIIS is off, set to large number.
        // To handle two seperate UHF DIIS spaces we take whichever has the largest error.
        let err = if use_diis {
            let err_a = diis_a.last_error_norm2().unwrap_or(f64::INFINITY).sqrt();
            let err_b = diis_b.last_error_norm2().unwrap_or(f64::INFINITY).sqrt();
            err_a.max(err_b)
        } else {
            f64::INFINITY
        };
        let d_e = (e_new - e).abs();
        
        if input.verbose{
            println!("{:4} {:12.6} {:12.4e} {:12.4e}", iter, e_new, d_e, err);
        }
        
        if d_e < input.e_tol && (!use_diis || err < input.err_tol) {
            if input.verbose{
                    println!("occa: [{:.3}]", oa);
                    println!("occb: [{:.3}]", ob);
            }
            return (e_new, ca, cb, oa, ob);
        }
        da = da_new; 
        db = db_new;
        e = e_new;
        iter += 1;
    }
     
    println!("SCF not converged.");
    (f64::NAN, Array2::<f64>::zeros((0, 0)), Array2::<f64>::zeros((0, 0)), 
     Array1::<f64>::zeros(0), Array1::<f64>::zeros(0))
}
