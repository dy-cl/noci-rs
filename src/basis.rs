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

/// Calculate the distance between SCF states from Phys. Rev. Lett. 101, 193001 as 
/// d_{wx}^2 = N - {}^w D^{\mu\nu} {}^x D_{\nu\mu} = N - Tr(D_w S D_x S).
/// # Arguments 
///     `w`: SCFState, reference state from which distance is computed. 
///     `x`: SCFState, state to which distance is computed.
///     `s`: Array2, AO overlap matrix
fn electron_distance(w: &SCFState, x: &SCFState, s: &Array2<f64>) -> f64 {
    // Calculate electron number N as Tr(D_r S).
    let na = (w.da.dot(s)).diag().sum();
    let nb = (w.db.dot(s)).diag().sum();
    let n= na + nb;
    // Calculate Tr(D_w S D_x S).
    let tr_a = w.da.dot(s).dot(&x.da).dot(s).diag().sum();
    let tr_b = w.db.dot(s).dot(&x.db).dot(s).diag().sum();
    // Electron distance is the difference.
    n - (tr_a + tr_b)
}

/// Pass given AO data and previous SCF solutions to SCF cycle to form the requested NOCI basis.
/// # Arguments 
/// ao: AoData struct, contains AO integrals and other system data. 
/// input: Input struct, contains user inputted options. 
/// prev: Option<[SCFState]>, may or may not contain states from a previous geometry.
pub fn generate_scf_state(ao: &AoData, input: &Input, prev: Option<&[SCFState]>,) -> Vec<SCFState> {
    
    let da0: Array2<f64> = ao.dm.clone() * 0.5;
    let db0: Array2<f64> = ao.dm.clone() * 0.5;

    let ia: Vec<usize> = ao.aolabels.iter().enumerate()
        .filter(|(_, s)| s.split_whitespace().next().unwrap() == "0")
        .map(|(i, _)| i)
        .collect();

    let ib: Vec<usize> = ao.aolabels.iter().enumerate()
        .filter(|(_, s)| s.split_whitespace().next().unwrap() == "1")
        .map(|(i, _)| i)
        .collect();

    let mut out = Vec::with_capacity(input.states.len());

    for (i, recipe) in input.states.iter().enumerate() {
        if input.verbose {
            println!("=======================Begin SCF==========================");
            println!("State({}): {}", i + 1, recipe.label);
        };
        
        // Use RHF density matrices from PySCF as an initial guess.
        let mut da = da0.clone();
        let mut db = db0.clone();

        // If there are previous states, use them (only for ground states). Excited 
        // states are seeded from the RHF solution at the previous geometry, and  
        // MOM is used.
        let excitation = recipe.excitation.as_ref();
        let seed = match (prev, excitation.is_none()) {
            (Some(ps), true)  => ps.get(i),
            (Some(ps), false) => input.states.iter()
                                 .position(|r| r.label == "RHF")
                                 .and_then(|idx| ps.get(idx)),
            _ => None,
        };
        
        if let Some(st) = seed {
            da = st.da.clone();
            db = st.db.clone();
        }

        // Reapply the spin bias (if requested) to seperate UHF solutions requiring 
        // spin breakage from RHF solutions.
        if let Some(sb) = &recipe.spin_bias { 
                let is_ab = sb.pattern == "AB"; 
                bias_density(&mut da, &mut db, &ia, &ib, sb.pol, is_ab); 
        }

        let (e, ca, cb, oa, ob, da, db) = crate::scf::scf_cycle(&da, &db, ao, input, excitation);

        // Form spin block diagonal MO coefficient matrix (i.e., [[ca, 0], [0, cb]]), 
        // this is later required for NOCI calculations.
        let (cs, cs_occ) = spin_block_mo_coeffs(&ca, &cb, &oa, &ob, ao.nao);
        out.push(SCFState {e, oa, ob, ca, cb, cs, cs_occ, da, db, 
                           label: recipe.label.clone()});
 
    }
    
    // Calculate the distance between all SCF states and the state labelled "RHF", 
    // hopefully this is the RHF solution.
    if input.verbose {
        println!("=========================================================");
        if let Some(rhf_idx) = input.states.iter().position(|r| r.label == "RHF") {
            let rhf_state = &out[rhf_idx];
            println!("Electron distances to RHF state:");
            for (i, st) in out.iter().enumerate() {
                let d2 = electron_distance(rhf_state, st, &ao.s_ao);
                println!("State({}): {}, d^2(RHF, {}): {}", i + 1, st.label, st.label, d2);
            }
        }
    }

    out
}
