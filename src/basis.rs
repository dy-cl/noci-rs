// basis.rs
use ndarray::{Array2};
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
    
    // (D_a')_{ij} = (1 + p)(D_a)_{ij} if i, j on atom A 
    // (D_b')_{ij} = (1 - p)(D_b)_{ij} if i, j on atom A  
    if a {
        scale_block(da, ia, up);
        scale_block(db, ia, dn);
        scale_block(da, ib, dn);
        scale_block(db, ib, up);
    // (D_a')_{ij} = (1 - p)(D_a)_{ij} if i, j on atom B
    // (D_b')_{ij} = (1 + p)(D_b)_{ij} if i, j on atom B
    } else {
        scale_block(da, ia, dn); 
        scale_block(db, ia, up);
        scale_block(da, ib, up);
        scale_block(db, ib, dn);
    }
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

    let mut out = Vec::with_capacity(3);
    
    if verbose{
        println!("=======================Begin SCF==========================");
        println!("State(1)");
    }

    // RHF ground state is seeded with unbiased spin density matrix. When passing in identical spin
    // matrices to the scf_cycle, construction of Fock Matrix, Calculation of UHF energy etc
    // becomes the RHF case.
    let (e_rhf, ca_rhf, cb_rhf) = crate::scf::scf_cycle(&da0, &db0, ao, max_cycle, 
                                                         e_tol, err_tol, verbose);
    out.push(SCFState{e: e_rhf, ca: ca_rhf, cb: cb_rhf});
    
    if verbose{
        println!("=======================Begin SCF==========================");
        println!("State(2)");
    }

    // First UHF spin-broken ground state we seed with a biased spin density matrix.
    let mut da_ab = da0.clone();
    let mut db_ab = db0.clone();
    bias_density(&mut da_ab, &mut db_ab, &ia, &ib, pol, true);
    let (e_uhf_ab, ca_uhf_ab, cb_uhf_ab) = crate::scf::scf_cycle(&da_ab, &db_ab, ao, max_cycle,                                                                               e_tol, err_tol, verbose);
    out.push(SCFState{e: e_uhf_ab, ca: ca_uhf_ab, cb: cb_uhf_ab});
    
    if verbose {
        println!("=======================Begin SCF==========================");
        println!("State(3)");
    }
    
    // Second UHF spin-broken ground state we seed with a biased spin density matrix.
    let mut da_ba = da0.clone();
    let mut db_ba = db0.clone();
    bias_density(&mut da_ba, &mut db_ba, &ia, &ib, pol, false);
    let (e_uhf_ba, ca_uhf_ba, cb_uhf_ba) = crate::scf::scf_cycle(&da_ba, &db_ba, ao, max_cycle, 
                                                                  e_tol, err_tol, verbose);
    out.push(SCFState{e: e_uhf_ba, ca: ca_uhf_ba, cb: cb_uhf_ba});
    
    if verbose{
        println!("=======================================================");
    }

    out
}
