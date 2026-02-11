// scf.rs
use std::sync::Arc;

use ndarray::{Axis, Array1, Array2, Array4, s};

use crate::{AoData, SCFState, Excitation, ExcitationSpin};
use crate::input::{Input, Spin, SCFExcitation, StateType};
use crate::basis::{electron_distance};
use crate::diis::Diis;

use crate::maths::general_evp_real;
use crate::utils::print_array2;

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

/// Select Aufbau indices given eigenvalues and number of occupancies. 
/// # Arguments 
///     `e`: MO energies. 
///     `nocc`: Number of occupied MOs.
fn aufbau_indices(e: &Array1<f64>, nocc: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..e.len()).collect();
    idx.sort_by(|&i, &j| e[i].partial_cmp(&e[j]).unwrap());
    idx.truncate(nocc);
    idx
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

/// Construct the metadynamics bias term \sum_\Lambda {}^\Lambda D_{\mu\nu} N_{\Lambda}
/// \lambda_\Lambda exp(-\lambda_\Lambda d_{0\Lambda}^2). 
/// # Arguments:
///     `da`: Array2, spin a density matrix.
///     `db`: Array2, spin b density matrix.
///     `ao`: AoData struct, contains AO integrals and metadata.
///     `biases`: [SCFState], previously found states when using metadynamics.
///     `lambda`: f64, bias strength.
fn metadynamics_bias(da: &Array2<f64>, db: &Array2<f64>, ao: &AoData, biases: &[SCFState], lambda: f64) -> (Array2<f64>, Array2<f64>) {
    let nbf = da.nrows();
    let mut ba = Array2::<f64>::zeros((nbf, nbf));
    let mut bb = Array2::<f64>::zeros((nbf, nbf));

    // Create temporary SCFState object with only the densities present such that we can reuse the
    // electron distance function.
    let tmpscf = SCFState {e: 0.0, oa: Array1::zeros(nbf), ob: Array1::zeros(nbf), ca: Arc::new(Array2::zeros((nbf, nbf))), cb: Arc::new(Array2::zeros((nbf, nbf))), 
                           da: Arc::new(da.clone()), db: Arc::new(db.clone()), label: String::new(), noci_basis: false, parent: 0, excitation: Excitation {alpha: ExcitationSpin 
                           {holes: vec![], parts: vec![] }, beta: ExcitationSpin { holes: vec![], parts: vec![]}}};

    for bias in biases {
        // d_{0\Lambda}^2 = N - Tr({}^0 D S {}^\Lambda D S).
        let d2 = electron_distance(&tmpscf, bias, &ao.s);
        // N_{\Lambda} = Tr({}^\Lambda D S).
        let nlambda = (bias.da.dot(&ao.s)).diag().sum() + (bias.db.dot(&ao.s)).diag().sum();
        // C = N_{\Lambda} \lambda exp(-\lambda_\Lambda d_{0\Lambda}^2).
        let c = nlambda * lambda * (-lambda * d2).exp();
        // Per-spin bias term accumulation.
        ba = ba + &*bias.da * c;
        bb = bb + &*bias.db * c;
    }
    (ba, bb)
}

/// Unrestricted SCF cycle with Loewdin orthogonalization.
/// Uses AO integrals from AoData struct.
/// # Arguments
///     `da0`: Array2, initial spin a density matrix.
///     `db0`: Array2, initial spin b density matrix.
///     `ao`: AoData struct, contains AO integrals and metadata.
///     `input`: Input struct, contains user specified input data.
///     `label`: String, label for current scf state.
///     `noci_basis`: bool, whether or not to use this state in the NOCI basis.
///     `scfexcitation`: SCFExcitation, use requested excited states.
///     `i`: Index of the SCF state.
///     `biases`: [SCFState], previously found states when using metadynamics
pub fn scf_cycle(da0: &Array2<f64>, db0: &Array2<f64>, ao: &AoData, input: &Input, label: &str, noci_basis: bool, 
                 scfexcitation: Option<&SCFExcitation>, i: usize, biases: Option<&[SCFState]>) -> Option<SCFState> { 
    let h = &ao.h; 
    let eri = &ao.eri_coul;
    let s = &ao.s;
    let enuc = ao.enuc;
    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();

    let mut e = f64::INFINITY;
    let mut da = da0.clone();
    let mut db = db0.clone();

    let use_diis = true;
    let mut diis = Diis::new(input.scf.diis.space);

    let mut ca_occ_old: Option<Array2<f64>> = None;
    let mut cb_occ_old: Option<Array2<f64>> = None;
    
    if input.write.verbose {
        match scfexcitation {
            Some(ex) => {
                let sp = match ex.spin {
                    Spin::Alpha => "alpha",
                    Spin::Beta => "beta",
                    Spin::Both => "both"
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

    let lambda: Option<f64> = match &input.states {
        StateType::Metadynamics(meta) => Some(meta.lambda),
        _ => None,
    };

    let mut iter = 0;
    while iter < input.scf.max_cycle {
        
        // Form Fock matrices from current densities and add to DIIS history.
        let (mut fa_curr, mut fb_curr) = form_fock_matrices(h, eri, &da, &db);

        // If using metadynamics, add the appropriate bias.
        if let (Some(bias_states), Some(lambda)) = (biases, lambda) && !bias_states.is_empty() {
            let (ba, bb) = metadynamics_bias(&da, &db, ao, bias_states, lambda);
            fa_curr = fa_curr + ba;
            fb_curr = fb_curr + bb;
        }

        if use_diis {diis.push(&fa_curr, &fb_curr, &da, &db, s);}
        
        // Extrapolate Fock matrix to be diagonalised this iteration.
         let (fa_use, fb_use) = if use_diis {
            diis.extrapolate_fock().unwrap_or((fa_curr.clone(), fb_curr.clone()))
        } else {
            (fa_curr.clone(), fb_curr.clone())
        };
        
        // Solve GEVP FC = SCe.
        let (ea, ca) = general_evp_real(&fa_use, s, false, 1e-8);
        let (eb, cb) = general_evp_real(&fb_use, s, false, 1e-8);

        // Flags true only if excitation is requested on alpha, beta or both.
        let (mom_a, mom_b) = if let Some(ex) = scfexcitation {
            match ex.spin {
                Spin::Alpha => (true, false),
                Spin::Beta  => (false, true),
                Spin::Both  => (true, true),
            }
        } else {
            (false, false)
        };
        
        // Select MO indices to occupy for alpha spin electrons.
        let idx_a = if mom_a {
            // If using MOM attempt to utilise previous iteration's coefficients.
            match &ca_occ_old {
                Some(ca_occ_old) => mom_select(ca_occ_old, &ca, s, na),
                // If coefficients not avaliable, seed by occupying columns specified in input.
                None => {
                    let ex = scfexcitation.unwrap();
                    let mut idx: Vec<usize> = (0..na).collect();
                    // Transform user input, allows user to specify occ = -1
                    // to choose HOMO.
                    let occ_abs = (na as i32 + ex.occ) as usize;
                    let vir_abs = na + ex.vir as usize;
                    // Remove occupied index.
                    idx.retain(|&k| k != occ_abs);
                    // Add excited index.
                    idx.push(vir_abs);
                    idx
                }
            }
        // If not using MOM occupy columns according to Aufbau ordering.
        } else {
            aufbau_indices(&ea, na)
        };
        let ca_occ = ca.select(Axis(1), &idx_a);
        ca_occ_old = Some(ca_occ.clone());

        // Select MO indices to occupy for beta spin electrons.
        let idx_b = if mom_b {
            // If using MOM attempt to utilise previous iteration's coefficients.
            match &cb_occ_old {
                Some(cb_occ_old) => mom_select(cb_occ_old, &cb, s, nb),
                // If coefficients not avaliable, seed by occupying columns specified in input.
                None => {
                    let ex = scfexcitation.unwrap();
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
                }
            }
        // If not using MOM occupy columns according to Aufbau ordering.
        } else {
            aufbau_indices(&eb, nb)
        };
        let cb_occ = cb.select(Axis(1), &idx_b);
        cb_occ_old = Some(cb_occ.clone());
        
        // Form new spin specific densities, Fock matrices, and energy.
        let da_new = ca_occ.dot(&ca_occ.t());
        let db_new = cb_occ.dot(&cb_occ.t());
        let (fa_new, fb_new) = form_fock_matrices(h, eri, &da_new, &db_new);
        let e_new = scf_energy(&da_new, &db_new, &fa_new, &fb_new, h, enuc);

        // Calculate current occupancies from density matrices and MO coefficients.
        let oa = mo_occupancies(&ca, &da_new, s);
        let ob = mo_occupancies(&cb, &db_new, s);
        
        // Calculate DIIS error term and dE for convergence testing. 
        // If no error can be calculated  (i.e., DIIS subspace is 1) or DIIS is off, set to large number.
        // To handle two seperate UHF DIIS spaces we take whichever has the largest error.
        let err = if use_diis {
            diis.last_error_norm2().unwrap_or(f64::INFINITY).sqrt()
        } else {
            f64::INFINITY
        };
        let d_e = (e_new - e).abs();

        if input.write.verbose{
            println!("{:4} {:12.6} {:12.4e} {:12.4e}", iter, e_new, d_e, err);
        }
        
        if d_e < input.scf.e_tol {
            // Allow for excited determinants to point at this data without copying it in memory.
            let ca = Arc::new(ca);
            let cb = Arc::new(cb);
            let da = Arc::new(da_new);
            let db = Arc::new(db_new);

            println!("Coefficients ca:");
            print_array2(&ca);
            println!("Coefficients cb:");
            print_array2(&cb);
            
            // SCF is only performed on reference states so the excitation here is empty. This is
            // distinct from scfexcitation in which we may use an excited SCF solution as part of
            // the reference states for QMC.
            let excitation = Excitation {
                alpha: ExcitationSpin {holes: vec![],  parts: vec![]},  
                beta:  ExcitationSpin {holes: vec![],  parts: vec![]}
            };
            
            return Some(SCFState {e: e_new, oa, ob, ca, cb, da, db, label: label.to_string(), noci_basis, parent: i, excitation});
        }
        da = da_new; 
        db = db_new;
        e = e_new;
        iter += 1;
    }
    println!("SCF not converged.");
    None
}
