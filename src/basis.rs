// basis.rs
use ndarray::{Array1, Array2, Axis, s};

use crate::{AoData, SCFState};
use crate::input::Input;

/// Multiply a square sub-block of matrix by scalar. 
/// # Arguments
///     `d`: Array2, Any matrix (typically a spin density matrix) which is modified in place.
///     `idx`: [usize], AO indices which form the square sub-block. 
///     `scale`: Float, Scalar multiplicative factor.
fn scale_block(d: &mut Array2<f64>, idx: &[usize], scale: f64) {
    for &i in idx {
        for &j in idx{
            d[(i, j)] *= scale
        }
    }
}

/// Bias spin a and b density matrices towards a spin-broken UHF guess.
/// # Arguments 
///     `da`: Array2, Spin density matrix a.
///     `db`: Array2, Spin density matrix b.
///     `ia`: [usize], Global AO indices of AOs belonging to atom 0 (A). 
///     `ib`: [usize], Global AO indices of AOs belonging to atom 1 (B). 
///     `pol`: Float, Scalar factor by which given sub-blocks of the spin density matrices are biased.
///     `a`: Bool, Which of the two degenerate spin-broken UHF states are we targeting.
fn bias_density(da: &mut Array2<f64>, db: &mut Array2<f64>, ia: &[usize], ib: &[usize], pol: f64, a: bool) {
    let up = 1.0 + pol; 
    let dn = 1.0 - pol; 
    
    // (D_a')_{ij} = (1 + p)(D_a)_{ij} if i, j on atom A. 
    // (D_b')_{ij} = (1 - p)(D_b)_{ij} if i, j on atom A.
    if a {
        scale_block(da, ia, up);
        scale_block(db, ia, dn);
        scale_block(da, ib, dn);
        scale_block(db, ib, up);
    // (D_a')_{ij} = (1 - p)(D_a)_{ij} if i, j on atom B.
    // (D_b')_{ij} = (1 + p)(D_b)_{ij} if i, j on atom B.
    } else {
        scale_block(da, ia, dn); 
        scale_block(db, ia, up);
        scale_block(da, ib, up);
        scale_block(db, ib, dn);
    }
}

/// Assembles the spin diagonal MO coefficient matrix (i.e., [[ca, 0], [0, cb]]) and the 
/// occupied only variant.
/// # Arguments
///     `ca`: Array2, spin a MO coefficients.
///     `cb`: Array2, spin b MO coefficients.
///     `oa`: Occupancy vector for spin a MOs.
///     `ob`: Occupancy vector for spin b MOs.
///     `nao`: Number of AOs.
pub fn spin_block_mo_coeffs(ca: &Array2<f64>, cb: &Array2<f64>, oa: &Array1<f64>, 
                            ob: &Array1<f64>, nao: usize) -> (Array2<f64>, Array2<f64>) {

        let mut cs = Array2::<f64>::zeros((2 * nao, 2 * nao));
        cs.slice_mut(s![0..nao, 0..nao]).assign(ca);
        cs.slice_mut(s![nao..2 * nao, nao..2 * nao]).assign(cb);
        
        // Number of AOs is equal to number of MOs here.
        let mut cols: Vec<usize> = oa.iter().enumerate()
                                   .filter_map(|(i, &occ)| if occ > 0.5 { Some(i) } else { None })
                                   .collect();
        cols.extend(ob.iter().enumerate()
                    .filter_map(|(i, &occ)| if occ > 0.5 { Some(nao + i) } else { None }));

        let cs_occ = cs.select(Axis(1), &cols);

        (cs, cs_occ)
}

/// Use given AO data to construct 3 SCF states which form the NOCI(3) basis for H2. 
/// These states are RHF ground state, 2 degenerate spin-broken UHF ground states. 
/// This should be generalised to all systems in future.
/// Will require using input files to specify the NOCI basis we want generated.
/// # Arguments
///     `ao`: AoData struct, contains AO integrals and other system data. 
///     `max_cycle`: Integer, sets the maximum number of SCF cycles.
///     `e_tol`: Float, sets convergence tolerance for SCF energy.
///     `err_tol`: Float, sets convergence tolerance for DIIS error.
pub fn generate_scf_state(ao: &AoData, input: &Input) -> Vec<SCFState> {
    
    let max_cycle = input.max_cycle;
    let e_tol = input.e_tol;
    let err_tol = input.err_tol;
    let pol = input.pol;
    let verbose = input.verbose;
    
    let da0: Array2<f64> = ao.dm.clone() * 0.5;
    let db0: Array2<f64> = ao.dm.clone() * 0.5;
    
    let ia: Vec<usize> = ao.aolabels.row(0).iter().map(|&i| i as usize).collect();
    let ib: Vec<usize> = ao.aolabels.row(1).iter().map(|&i| i as usize).collect();
    
    let states_to_get = [("RHF", None), ("UHF_AB", Some(true)), ("UHF_BA", Some(false))];
    let mut out = Vec::with_capacity(states_to_get.len());

    for (i, (_label, bias_flag)) in states_to_get.iter().enumerate() {
        if verbose {
            println!("=======================Begin SCF==========================");
            println!("State({})", i + 1);
        };

        let mut da = da0.clone();
        let mut db = db0.clone();

        // Apply the respective spin biases to the UHF solutions 
        // otherwise we have equal spin density matrices in RHF.
        if let Some(is_ab) = bias_flag {bias_density(&mut da, &mut db, &ia, &ib, pol, *is_ab)}
        let (e, ca, cb, oa, ob) = crate::scf::scf_cycle(&da, &db, ao, max_cycle, 
                                                        e_tol, err_tol, verbose);

        // Form spin block diagonal MO coefficient matrix (i.e., [[ca, 0], [0, cb]]), 
        // this is later required for NOCI calculations.
        let (cs, cs_occ) = spin_block_mo_coeffs(&ca, &cb, &oa, &ob, ao.nao);
        out.push(SCFState {e, oa, ob, ca, cb, cs, cs_occ});
 
    }

    out
}
