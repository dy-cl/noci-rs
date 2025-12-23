// basis.rs
use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array1, Array2};

use crate::{AoData, SCFState};
use crate::input::Input;

use crate::scf::{spin_block_mo_coeffs, scf_cycle};

pub struct SpinOccupation {
    occ_alpha: Vec<usize>,
    virt_alpha: Vec<usize>,
    occ_beta: Vec<usize>,
    virt_beta: Vec<usize>,
}

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

/// Bias density matrices towards a spatial symmetry broken RHF guess. We will have da = db.
/// # Arguments
/// # Arguments 
///     `da`: Array2, Spin density matrix a.
///     `db`: Array2, Spin density matrix b.
///     `atomao`: [Vec<usize>], Global AO indices of AOs belonging to atom i. 
///     `pol`: Float, Bias strength.
///     `pattern`: Integer, spin biasing pattern.
fn bias_spatial(da: &mut Array2<f64>, db: &mut Array2<f64>, atomao: &[Vec<usize>], pol: f64, pattern: &[i8]) { 
    let up = 1.0 + pol;
    let dn = 1.0 - pol;

    for (a, &sgn) in pattern.iter().enumerate() {
        if sgn == 0 { 
            continue; 
        }
        let idx = &atomao[a];
        if sgn > 0 {
            scale_block(da, idx, up);
            scale_block(db, idx, up);
        } else {
            scale_block(da, idx, dn);
            scale_block(db, idx, dn);
        }
    }
}

/// Bias density matrices towards a spin symmetry-broken UHF guess. We will have da != db.
/// # Arguments 
///     `da`: Array2, Spin density matrix a.
///     `db`: Array2, Spin density matrix b.
///     `atomao`: [Vec<usize>], Global AO indices of AOs belonging to atom i. 
///     `pol`: Float, Bias strength.
///     `pattern`: Integer, spin biasing pattern.
fn bias_spin(da: &mut Array2<f64>, db: &mut Array2<f64>, atomao: &[Vec<usize>], pol: f64, pattern: &[i8]) {
    let up = 1.0 + pol;
    let dn = 1.0 - pol;

    for (a, &sgn) in pattern.iter().enumerate() {
        if sgn == 0 {
            continue;
        }
        let i = &atomao[a];
        if sgn > 0 {
            scale_block(da, i, up);
            scale_block(db, i, dn);
        } else {
            scale_block(da, i, dn);
            scale_block(db, i, up);
        }
    }
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
    let tr_a = (*w.da).dot(s).dot(&*x.da).dot(s).diag().sum();
    let tr_b = (*w.db).dot(s).dot(&*x.db).dot(s).diag().sum();
    // Electron distance is the difference.
    n - (tr_a + tr_b)
}

/// Using the occupation vectors oa, ob, get positions p of oa, ob which are equal to 0 and 1. That is, 
/// the virtual and occupied column indices for each spin.
/// # Arguments
///     `st`: SCFState, state which we are finding virtual and occupied indices.
fn get_spin_occupation(st: &SCFState,) -> SpinOccupation {
    //  For each entry in oa/ob find all indices p where o[p] > 0.5.
    let occ_alpha: Vec<usize> = st.oa.iter().enumerate().filter_map(|(p, &occ)| if occ > 0.5 {Some(p)} else {None}).collect();
    let occ_beta: Vec<usize> = st.ob.iter().enumerate().filter_map(|(p, &occ)| if occ > 0.5 {Some(p)} else {None}).collect();

    //  For each entry in oa/ob find all indices p where o[p] < 0.5.
    let virt_alpha: Vec<usize> = st.oa.iter().enumerate().filter_map(|(p, &occ)| if occ <= 0.5 {Some(p)} else {None}).collect();
    let virt_beta: Vec<usize> = st.ob.iter().enumerate().filter_map(|(p, &occ)| if occ <= 0.5 {Some(p)} else {None}).collect();

    SpinOccupation{occ_alpha, virt_alpha, occ_beta, virt_beta}
}

/// Copy a reference SCF state (i.e., those that form the deterministic NOCI basis) and replace the
/// oa, ob with modified occupancies, and rebuild cs_occ accordingly.
/// # Arguments 
///    `ao`: AoData struct, contains AO integrals and other system data.
///    `reference`: SCFState, SCF state from which to build an excited state.
///    `oa_ex`: Array1, Excited occupied indices spin alpha.
///    `ob_ex`: Array1, Excited occupied indices spin beta. 
///    `label_suffix`: String, what to append to reference state label to indicate excitation.
fn make_excited_state(ao: &AoData, reference: &SCFState, oa_ex: Array1<f64>, ob_ex: Array1<f64>, label_suffix: &str,) -> SCFState{
    // Get occupied coefficient matrices.
    let (_cs, cs_occ) = spin_block_mo_coeffs(&reference.ca, &reference.cb, &oa_ex, &ob_ex, ao.nao);

    SCFState {e: 0.0, oa: oa_ex, ob: ob_ex, ca: Arc::clone(&reference.ca), cb: Arc::clone(&reference.cb), cs: Arc::clone(&reference.cs),
              da: Arc::clone(&reference.da), db: Arc::clone(&reference.db), cs_occ, label: format!("{} {}", reference.label, label_suffix),
              noci_basis: false,}
}

/// Given aolabels (which contains) information about which atom an AO belongs to, find the AO
/// indices of a set of given atoms. For example if we had aolabels = ["0 1s", "0 1s", "1 1s", "1
/// 1s"] (i.e., H2 in minimal basis) and atoms = [0] the function returns [0, 1].
/// # Arguments
///     `aolabels`: Labels which map AOs to atoms.
///     `atoms`: Atom indices for which we wish to know the corresponding AO indices.
fn ao_indices_for_atomset(aolabels: &[String], atoms: &[usize]) -> Vec<usize> {
    // Iterate over all AO labels which contain for example "2 1s".
    aolabels.iter().enumerate()
        // Take the first part of the label (e.g. "2") and keep it if the AOs atom index is in
        // atoms list.
        .filter(|(_, s)| {
            let a = s.split_whitespace().next().unwrap().parse::<usize>().unwrap();
            atoms.contains(&a)
        // Return the AO indices i.
        }).map(|(i, _)| i).collect()
}


/// Pass given AO data and previous SCF solutions to SCF cycle to form the requested reference NOCI basis.
/// # Arguments 
///     `ao`: AoData struct, contains AO integrals and other system data. 
///     `input`: Input struct, contains user inputted options. 
///     `prev`: Option<[SCFState]>, may or may not contain states from a previous geometry.
pub fn generate_reference_noci_basis(ao: &AoData, input: &Input, prev: Option<&[SCFState]>,) -> Vec<SCFState> {
    
    let da0: Array2<f64> = ao.dm.clone() * 0.5;
    let db0: Array2<f64> = ao.dm.clone() * 0.5;
    let d_tol = 1e-2;
    
    // Construct lookuptable from state label to previous SCF states. Allows for seeding of SCF
    // states at a subsequent geometry to be done by label rather than via index which breaks
    // easily.
    let mut prev_map: HashMap<&str, &SCFState> = HashMap::new();
    if let Some(ps) = prev {
        for st in ps {
            prev_map.insert(&st.label, st);
        }
    }

    let mut out: Vec<SCFState> = Vec::with_capacity(input.states.len());

    for (i, recipe) in input.states.iter().enumerate() {
        if input.write.verbose {
            let left = "=".repeat(45);
            let right = "=".repeat(46);
            println!("{}Begin SCF{}", left, right);
            println!("State({}): {}", i + 1, recipe.label);
        };
        
        // Use RHF density matrices from PySCF as an initial guess.
        let mut da = da0.clone();
        let mut db = db0.clone();

        // If there are previous states, use them. Ground states are seeded from their previous
        // geometry equivalent, whilst excited states are seeded from RHF at the previous geometry.
        // Any excited states that may or may not be generated here are distinct from those formed
        // in the NOCI-QMC basis. Those formed here form the reference NOCI basis and are relaxed.
        let excitation = recipe.excitation.as_ref();
        let seed = if prev.is_some() {
            if excitation.is_none() {
                prev_map.get(recipe.label.as_str()).copied()
            } else {
                prev_map.get("RHF").copied()
            }
        } else {
            None
        };
        
        if let Some(st) = seed {
            da = (*st.da).clone();
            db = (*st.db).clone();
        }

        // Reapply the spin bias (if requested) to seperate UHF solutions requiring 
        // spin breakage from RHF solutions.
        if let Some(sb) = &recipe.spin_bias {
            // Count atoms.
            let natoms: usize = ao.aolabels.iter().map(|s| s.split_whitespace().next().unwrap().parse::<usize>().unwrap()).max().unwrap_or(0) + 1;
            // Find which AOs belong to which atom.
            let atomao: Vec<Vec<usize>> = (0..natoms).map(|a| ao_indices_for_atomset(&ao.aolabels, &[a])).collect();
            bias_spin(&mut da, &mut db, &atomao, sb.pol, &sb.pattern);
        }
        // Reapply spatial bias (if requested).
        if let Some(spb) = &recipe.spatial_bias {
            // Count atoms.
            let natoms: usize = ao.aolabels.iter().map(|s| s.split_whitespace().next().unwrap().parse::<usize>().unwrap()).max().unwrap_or(0) + 1;
            // Find which AOs belong to which atom.
            let atomao: Vec<Vec<usize>> = (0..natoms).map(|a| ao_indices_for_atomset(&ao.aolabels, &[a])).collect();
            bias_spatial(&mut da, &mut db, &atomao, spb.pol, &spb.pattern);
        }

        let state: SCFState = scf_cycle(&da, &db, ao, input, excitation, i).expect("SCF did not converge");

        // Remove duplicate states from the basis to avoid singularity issues.
        let mut is_duplicate = false;
        for existing in &out {
            // If this state is not requested to be used in the NOCI basis we can ignore it.
            if !existing.noci_basis {
                continue;
            }
            let d2 = electron_distance(existing, &state, &ao.s_ao);
            if d2 < d_tol {
                println!("Removed state '{}' from basis as d^2({}, {}) = {:.6}", state.label, existing.label, state.label, d2);
                is_duplicate = true;
                break;
            }
        }
        // By this point we are sure there are no duplicate states
        if !is_duplicate {
            out.push(state);
        }
    }
    out
}

/// Generate a requested amount of all possible excitations on top of the given reference NOCI
/// basis. Currently not a very generalised implementation to higher levels of excitation. 
/// # Arguments
///     ao: AoData struct, contains AO integrals and other system data.
///     refs: [SCFState], array of reference states for which excitations are generated.
///     input: Input struct, contains user inputted options. 
pub fn generate_qmc_deterministic_noci_basis(ao: &AoData, refs: &[SCFState], input: &Input) -> Vec<SCFState> {
    let mut out: Vec<SCFState> = Vec::new();
    for r in refs {
        // Include reference states in NOCI-QMC basis.
        out.push(r.clone());
         // If no excitations requested for this ref, continue
        if !(input.qmc.singles || input.qmc.doubles) {
            continue;
        }

        let spin_occ = get_spin_occupation(r);

        // Single excitations 
        if input.qmc.singles {
            // Single excitations spin alpha 
            for &i in &spin_occ.occ_alpha {
                for &a in &spin_occ.virt_alpha {
                    let mut oa_ex = r.oa.clone();
                    let ob_ex = r.ob.clone();
                    oa_ex[i] = 0.0;
                    oa_ex[a] = 1.0;
                    let label = format!("(alpha {} -> {})", i, a);
                    let ex = make_excited_state(ao, r, oa_ex, ob_ex, &label);
                    out.push(ex);
                }
            }

            // Single excitations spin beta
            for &i in &spin_occ.occ_beta {
                for &a in &spin_occ.virt_beta {
                    let oa_ex = r.oa.clone();
                    let mut ob_ex = r.ob.clone();
                    ob_ex[i] = 0.0;
                    ob_ex[a] = 1.0;
                    let label = format!("(beta {} -> {})", i, a);
                    let ex = make_excited_state(ao, r, oa_ex, ob_ex, &label);
                    out.push(ex);
                }
            }
        }
        
        // Double excitations
        if input.qmc.doubles {
            // Double excitations spin alpha spin alpha 
            let occ_a = &spin_occ.occ_alpha;
            let virt_a = &spin_occ.virt_alpha;

            for oi in 0..occ_a.len() {
                for oj in (oi + 1)..occ_a.len() {
                    let i = occ_a[oi];
                    let j = occ_a[oj];

                    for va in 0..virt_a.len() {
                        for vb in (va + 1)..virt_a.len() {
                            let a = virt_a[va];
                            let b = virt_a[vb];

                            let mut oa_ex = r.oa.clone();
                            let ob_ex = r.ob.clone();

                            oa_ex[i] = 0.0;
                            oa_ex[j] = 0.0;
                            oa_ex[a] = 1.0;
                            oa_ex[b] = 1.0;

                            let label = format!("(alpha, alpha {} {} -> {} {})", i, j, a, b);
                            let ex = make_excited_state(ao, r, oa_ex, ob_ex, &label);
                            out.push(ex);
                        }
                    }
                }
            }

            // Double excitations spin beta spin beta
            let occ_b = &spin_occ.occ_beta;
            let virt_b = &spin_occ.virt_beta;

            for oi in 0..occ_b.len() {
                for oj in (oi + 1)..occ_b.len() {
                    let i = occ_b[oi];
                    let j = occ_b[oj];

                    for va in 0..virt_b.len() {
                        for vb in (va + 1)..virt_b.len() {
                            let a = virt_b[va];
                            let b = virt_b[vb];

                            let oa_ex = r.oa.clone();
                            let mut ob_ex = r.ob.clone();

                            ob_ex[i] = 0.0;
                            ob_ex[j] = 0.0;
                            ob_ex[a] = 1.0;
                            ob_ex[b] = 1.0;

                            let label = format!("(beta, beta {} {} -> {} {})", i, j, a, b);
                            let ex = make_excited_state(ao, r, oa_ex, ob_ex, &label);
                            out.push(ex);
                        }
                    }
                }
            }

            // Double excitations spin alpha spin beta 
            for &i_a in &spin_occ.occ_alpha {
                for &a_a in &spin_occ.virt_alpha {
                    for &i_b in &spin_occ.occ_beta {
                        for &a_b in &spin_occ.virt_beta {
                            let mut oa_ex = r.oa.clone();
                            let mut ob_ex = r.ob.clone();

                            oa_ex[i_a] = 0.0;
                            oa_ex[a_a] = 1.0;
                            ob_ex[i_b] = 0.0;
                            ob_ex[a_b] = 1.0;

                            let label = format!("(alpha, beta {} -> {}, {} -> {})", i_a, a_a, i_b, a_b);
                            let ex = make_excited_state(ao, r, oa_ex, ob_ex, &label);
                            out.push(ex);
                        }
                    }
                }
            }

        }

    }
    out
}
