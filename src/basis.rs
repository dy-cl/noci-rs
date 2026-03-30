// basis.rs
use std::collections::HashMap;
use std::sync::Arc;

use itertools::Itertools;
use ndarray::{Array1, Array2};
use rand::{SeedableRng};
use rand::rngs::StdRng;

use crate::{AoData, Excitation, ExcitationSpin, SCFState};
use crate::input::{Input, StateType, StateRecipe, Metadynamics};
use crate::utils::random_pattern;

use crate::scf::scf_cycle;

pub struct SpinOccupation {
    occ_alpha: Vec<usize>,
    virt_alpha: Vec<usize>,
    occ_beta: Vec<usize>,
    virt_beta: Vec<usize>,
}

/// Multiply a square sub-block of a matrix by a scalar in place.
/// # Arguments:
/// - `d`: Matrix to be modified in place.
/// - `idx`: Row and column indices defining the square sub-block.
/// - `scale`: Multiplicative factor applied to the selected sub-block.
/// # Returns:
/// - `()`: Modifies `d` in place.
fn scale_block(d: &mut Array2<f64>, idx: &[usize], scale: f64) {
    for &i in idx {
        for &j in idx{
            d[(i, j)] *= scale
        }
    }
}

/// Bias density matrices towards a spatial symmetry broken RHF guess. We will have da = db.
/// # Arguments 
/// - `da`: Spin density matrix a.
/// - `db`: Spin density matrix b.
/// - `atomao`: Global AO indices of AOs belonging to atom i. 
/// - `pol`: Bias strength.
/// - `pattern`: Spin biasing pattern.
/// # Returns
/// - `()`: Modifies `da` and `db` in place.
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
/// - `da`: Spin density matrix a.
/// - `db`: Spin density matrix b.
/// - `atomao`: Global AO indices of AOs belonging to atom i. 
/// - `pol`: Bias strength.
/// - `pattern`: Spin biasing pattern.
/// # Returns
/// - `()`: Modifies `da` and `db` in place.
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
/// - `w`: Reference state from which distance is computed. 
/// - `x`: State to which distance is computed.
/// - `s`: AO overlap matrix.
/// # Returns
/// - `f64`: Electron distance between the two SCF states.
pub fn electron_distance(w: &SCFState, x: &SCFState, s: &Array2<f64>) -> f64 {
    // Calculate electron number N as Tr(D_r S).
    let na = (w.da.dot(s)).diag().sum();
    let nb = (w.db.dot(s)).diag().sum();
    let n = na + nb;
    // Calculate Tr(D_w S D_x S).
    let tr_a = (*w.da).dot(s).dot(&*x.da).dot(s).diag().sum();
    let tr_b = (*w.db).dot(s).dot(&*x.db).dot(s).diag().sum();
    // Electron distance is the difference.
    n - (tr_a + tr_b)
}

/// Using the occupation vectors oa, ob, get positions p of oa, ob which are equal to 0 and 1. That is, 
/// the virtual and occupied column indices for each spin.
/// # Arguments
/// - `st`: State which we are finding virtual and occupied indices.
/// # Returns
/// - `SpinOccupation`: Occupied and virtual orbital indices for alpha and beta spin.
fn get_spin_occupation(st: &SCFState,) -> SpinOccupation {
    //  For each entry in oa/ob find all indices p where o[p] > 0.5.
    let occ_alpha: Vec<usize> = st.oa.iter().enumerate().filter_map(|(p, &occ)| if occ > 0.5 {Some(p)} else {None}).collect();
    let occ_beta: Vec<usize> = st.ob.iter().enumerate().filter_map(|(p, &occ)| if occ > 0.5 {Some(p)} else {None}).collect();

    //  For each entry in oa/ob find all indices p where o[p] < 0.5.
    let virt_alpha: Vec<usize> = st.oa.iter().enumerate().filter_map(|(p, &occ)| if occ <= 0.5 {Some(p)} else {None}).collect();
    let virt_beta: Vec<usize> = st.ob.iter().enumerate().filter_map(|(p, &occ)| if occ <= 0.5 {Some(p)} else {None}).collect();

    SpinOccupation{occ_alpha, virt_alpha, occ_beta, virt_beta}
}

// Copy a reference SCF state (i.e., those that form the deterministic NOCI basis) and replace the
/// oa, ob with modified occupancies, and rebuild cs_occ accordingly.
/// # Arguments 
/// - `reference`: SCF state from which to build an excited state.
/// - `oa_ex`: Excited occupied indices spin alpha.
/// - `ob_ex`: Excited occupied indices spin beta. 
/// - `label_suffix`: What to append to reference state label to indicate excitation.
/// - `parent`: Index of the parent reference determinant.
/// - `excitation`: Excitation carried by the excited state.
/// # Returns
/// - `SCFState`: Excited state built from the reference state with modified occupancies.
fn make_excited_state(reference: &SCFState, oa_ex: Array1<f64>, ob_ex: Array1<f64>, label_suffix: &str,
                      parent: usize, excitation: Excitation) -> SCFState{

    SCFState {e: 0.0, oa: oa_ex, ob: ob_ex, ca: Arc::clone(&reference.ca), cb: Arc::clone(&reference.cb), da: Arc::clone(&reference.da), 
              db: Arc::clone(&reference.db), label: format!("{} {}", reference.label, label_suffix), noci_basis: false, parent, excitation}
}

/// Given aolabels (which contains) information about which atom an AO belongs to, find the AO
/// indices of a set of given atoms. For example if we had aolabels = ["0 1s", "0 1s", "1 1s", "1
/// 1s"] (i.e., H2 in minimal basis) and atoms = `[0]` the function returns [0, 1].
/// # Arguments
/// `aolabels`: Labels which map AOs to atoms.
/// `atoms`: Atom indices for which we wish to know the corresponding AO indices.
/// # Returns
/// - `Vec<usize>`: AO indices belonging to the requested atom set.
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

/// Generate the SCF states using the maximum orbital overlap procedure.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data. 
/// - `input`: Contains user inputted options. 
/// - `prev`: May or may not contain states from a previous geometry.
/// - `prev_map`: Map between the SCFState object and its label.
/// - `recipes`: Instructions for how to construct each state.
/// # Returns:
/// - `Vec<SCFState>`: Generated SCF states.
fn generate_states_mom(ao: &AoData, input: &Input, prev: Option<&[SCFState]>, prev_map: &HashMap<&str, &SCFState>, recipes: &[StateRecipe]) -> Vec<SCFState> {

    let da0: Array2<f64> = ao.dm.clone() * 0.5;
    let db0: Array2<f64> = ao.dm.clone() * 0.5;
    let d_tol = 1e-2;
    
    let mut out: Vec<SCFState> = Vec::with_capacity(recipes.len());
    for (i, recipe) in recipes.iter().enumerate() {
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
        let scfexcitation = recipe.scfexcitation.as_ref();
        let seed = if prev.is_some() {
            if scfexcitation.is_none() {
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
            let natoms: usize = ao.labels.iter().map(|s| s.split_whitespace().next().unwrap().parse::<usize>().unwrap()).max().unwrap_or(0) + 1;
            // Find which AOs belong to which atom.
            let atomao: Vec<Vec<usize>> = (0..natoms).map(|a| ao_indices_for_atomset(&ao.labels, &[a])).collect();
            bias_spin(&mut da, &mut db, &atomao, sb.pol, &sb.pattern);
        }
        // Reapply spatial bias (if requested).
        if let Some(spb) = &recipe.spatial_bias {
            // Count atoms.
            let natoms: usize = ao.labels.iter().map(|s| s.split_whitespace().next().unwrap().parse::<usize>().unwrap()).max().unwrap_or(0) + 1;
            // Find which AOs belong to which atom.
            let atomao: Vec<Vec<usize>> = (0..natoms).map(|a| ao_indices_for_atomset(&ao.labels, &[a])).collect();
            bias_spatial(&mut da, &mut db, &atomao, spb.pol, &spb.pattern);
        }

        let label = &recipes[i].label;
        let noci_basis = recipes[i].noci;

        let state: SCFState = scf_cycle(&da, &db, ao, input, label, noci_basis, scfexcitation, i, None).expect("SCF did not converge");

        // Remove duplicate states from the basis to avoid singularity issues.
        let mut is_duplicate = false;
        for existing in &out {
            // If this state is not requested to be used in the NOCI basis we can ignore it.
            if !existing.noci_basis {
                continue;
            }
            let d2 = electron_distance(existing, &state, &ao.s);
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

/// Generate the SCF states using the SCF metadynamics procedure.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data. 
/// - `input`: Contains user inputted options. 
/// - `prev_map`: Map between the SCFState object and its label.
/// - `meta`: SCF metadynamics parameters.
/// # Returns:
/// - `Vec<SCFState>`: Generated SCF states.
fn generate_states_metadynamics(ao: &AoData, input: &Input, prev_map: &HashMap<&str, &SCFState>, meta: &mut Metadynamics) -> Vec<SCFState> {

    let da0: Array2<f64> = ao.dm.clone() * 0.5;
    let db0: Array2<f64> = ao.dm.clone() * 0.5;
    let d_tol = 1e-2;
    let rhf_tol = 1e-2;

    let mut biases_rhf: Vec<SCFState> = Vec::with_capacity(meta.nstates_rhf);
    let mut biases_uhf: Vec<SCFState> = Vec::with_capacity(meta.nstates_uhf);

    let natoms: usize = ao.labels.iter().map(|s| s.split_whitespace().next().unwrap().parse::<usize>().unwrap()).max().unwrap_or(0) + 1;
    let atomao: Vec<Vec<usize>> = (0..natoms).map(|a| ao_indices_for_atomset(&ao.labels, &[a])).collect();
    let mut states: Vec<SCFState> = Vec::with_capacity(meta.nstates_rhf + meta.nstates_uhf);
    
    let mut attempt = 0_u64;
    let mut irhf = 0;
    while biases_uhf.len() < meta.nstates_uhf || biases_rhf.len() < meta.nstates_rhf {
        
        attempt += 1;
        if attempt > (meta.max_attempts as u64) {
            println!("Maximum number of attempts to find UHF solution has been exceeded. UHF is likely not distinct from RHF at this geometry.");
            break;
        }

        // Attempt spin biased state. Usually produces UHF but can collapse to RHF.
        if  meta.nstates_uhf > 0 && biases_uhf.len() < meta.nstates_uhf {
            
            // Assume that all metadynamics found states are to be used in NOCI basis.
            let noci_basis = true;

            let pair = biases_uhf.len() / 2;
            let base = biases_uhf.len();
            if base + 1 >= meta.labels_uhf.len() {break;}

            let labela = &meta.labels_uhf[base];
            let labelb = &meta.labels_uhf[base + 1];
      
            let mut cand: [Option<SCFState>; 2] = [None, None];

            // If previous densities for the current label exist use them. Otherwise start from RHF density.
            let (mut da, mut db) = (da0.clone(), db0.clone());
            if let Some(st) = prev_map.get(labela.as_str()).copied() {
                da = (*st.da).clone();
                db = (*st.db).clone();
            } else if let Some(st) = prev_map.get(labelb.as_str()).copied() {
                da = (*st.da).clone();
                db = (*st.db).clone();
            }

            // If a bias pattern which was previously successful in obtaining a UHF state exists
            // then we should reuse it. Otherwise generate a new pattern.
            let pattern: Vec<i8> = if let Some(p) = meta.spin_patterns_uhf[base].as_ref() {
                p.clone()
            } else {
                let mut rng = StdRng::seed_from_u64(attempt.wrapping_add((pair as u64).wrapping_mul(0x9E3779B97F4A7C15_u64)));
                random_pattern(&mut rng, natoms)
            };

            let (da0, db0) = (da.clone(), db.clone());

            // Any spin-broken UHF state should have a spin-flipped degernerate counterpart, find both.
            for j in 0..2 {

                let labelidx = base + j;
                if labelidx >= meta.nstates_uhf {break;}
                let label = &meta.labels_uhf[labelidx];
                
                let (mut daj, mut dbj) = (da0.clone(), db0.clone());

                // Bias the densities as prescribed by the pattern.
                bias_spin(&mut daj, &mut dbj, &atomao, meta.spinpol, &pattern);
                
                // When j == 1 we swap the densities so as to get the spin-flipped density.
                if j == 1 {std::mem::swap(&mut daj, &mut dbj);}

                if input.write.verbose {
                    let left = "=".repeat(45);
                    let right = "=".repeat(46);
                    println!("{}Begin UHF Metadynamics Biased SCF{}", left, right);
                    println!("State({}): {}, Spin-flip: {}, Attempt: {}", labelidx, label, j, attempt);
                }

                let biased = {
                    let biasi = if biases_uhf.is_empty() {None} else {Some(biases_uhf.as_slice())};
                    scf_cycle(&daj, &dbj, ao, input, &label.to_string(), noci_basis, None, labelidx, biasi).expect("SCF did not converge")
                };
                
                if input.write.verbose {
                    let left = "=".repeat(45);
                    let right = "=".repeat(46);
                    println!("{}Begin UHF Metadynamics Relaxed SCF{}", left, right);
                    println!("State({}): {}, Spin-flip: {}, Attempt: {}", labelidx, label, j, attempt);
                }

                let relaxed = scf_cycle(&biased.da, &biased.db, ao, input, &label.to_string(), noci_basis, None, labelidx, None).expect("SCF did not converge");

                let mut candidate = relaxed;
                candidate.noci_basis = true;

                // If we have collapsed to RHF we should change the label of the state.
                let mut drhf = 0.0;
                for (&a, &b) in candidate.da.iter().zip(candidate.db.iter()) {
                    drhf += (a - b).abs();
                }
                if drhf < rhf_tol {
                    // RHF branch.
                    // Ignore state if duplicate.
                    println!("UHF candidate collapsed to RHF: drhf = {:.3e} < {:.3e}", drhf, rhf_tol);
                    let dup = biases_rhf.iter().map(|st| electron_distance(st, &candidate, &ao.s)).any(|d2| d2 < d_tol);
                    if dup {
                        println!("Removed state '{}' from basis as duplicate.", candidate.label);
                        continue;
                    }

                    // Even if zero RHF were requested, UHF and RHF may not be distinct so we add
                    // it to the states anyway to avoid having an empty basis.
                    if meta.nstates_rhf == 0 && irhf >= meta.nstates_rhf {
                        meta.nstates_rhf = 1;
                        meta.labels_rhf.push("RHF".to_string());
                        meta.spatial_patterns_rhf.push(Some(pattern.clone()));
                    }
                    if irhf >= meta.nstates_rhf {continue;}
                    candidate.label = meta.labels_rhf[irhf].clone();
                    meta.spatial_patterns_rhf[irhf] = Some(pattern.clone());

                    // All duplicates removed by this point. Since we have successfully found a new
                    // state, reset the attempts counter.
                    attempt = 0;
                    biases_rhf.push(candidate.clone());
                    states.push(candidate);
                    irhf += 1;
                } else {
                    // UHF branch.
                    // Ignore state if duplicate.
                    println!("UHF candidate stayed UHF: drhf = {:.3e} > {:.3e}", drhf, rhf_tol);
                    let dup = biases_uhf.iter().map(|st| electron_distance(st, &candidate, &ao.s)).any(|d2| d2 < d_tol);
                    if dup {
                        println!("Removed state '{}' from basis as duplicate.", candidate.label);
                        continue;
                    }

                    candidate.label = label.clone();
                    meta.spin_patterns_uhf[base] = Some(pattern.clone());
                    if biases_uhf.len() + 1 < meta.nstates_uhf && j == 0 {
                        meta.spin_patterns_uhf[base + 1] = Some(pattern.clone());
                    }

                    // All duplicates removed by this point. Since we have successfully found a new
                    // state, reset the attempts counter.
                    attempt = 0;
                    cand[j] = Some(candidate);
                }
            }

            if cand[0].is_some() && cand[1].is_some() {
                let st0 = cand[0].take().unwrap();
                let st1 = cand[1].take().unwrap();
                biases_uhf.push(st0.clone());
                biases_uhf.push(st1.clone());
                states.push(st0);
                states.push(st1);
                meta.spin_patterns_uhf[base] = Some(pattern.clone());
            } else {
                continue;
            }
        }

        // Attempt spatial biased state. Should only produce RHF.
        if meta.nstates_rhf > 0 && biases_rhf.len() < meta.nstates_rhf {

            // Assume that all metadynamics found states are to be used in NOCI basis.
            let noci_basis = true;

            let label = &meta.labels_rhf[irhf];
            
            // If previous densities for the current label exist use them. Otherwise start from RHF density.
            let (mut da, mut db) = (da0.clone(), db0.clone());
            if let Some(st) = prev_map.get(label.as_str()).copied() {
                da = (*st.da).clone();
                db = (*st.db).clone();
            }
            
            // If a bias pattern which was previously successful in obtaining an RHF state exists
            // then we should reuse it. Otherwise generate a new pattern.
            let pattern: Vec<i8> = if let Some(p) = meta.spatial_patterns_rhf[irhf].as_ref() {
                p.clone()
            } else {
                let mut rng = StdRng::seed_from_u64(attempt.wrapping_add((irhf as u64).wrapping_mul(0x9E3779B97F4A7C15_u64)));
                random_pattern(&mut rng, natoms)
            };

            // Bias the densities as prescribed by the pattern.
            bias_spin(&mut da, &mut db, &atomao, meta.spatialpol, &pattern);

            if input.write.verbose {
                let left = "=".repeat(45);
                let right = "=".repeat(46);
                println!("{}Begin RHF Metadynamics Biased SCF{}", left, right);
                println!("State({}): {}, Attempt: {}", irhf, label, attempt);
            }

            let biased = {
                let biasi = if biases_uhf.is_empty() {None} else {Some(biases_uhf.as_slice())};
                scf_cycle(&da, &db, ao, input, &label.to_string(), noci_basis, None, irhf, biasi).expect("SCF did not converge")
            };
            
            if input.write.verbose {
                let left = "=".repeat(45);
                let right = "=".repeat(46);
                println!("{}Begin RHF Metadynamics Relaxed SCF{}", left, right);
                println!("State({}): {}, Attempt: {}", irhf, label, attempt);
            }

            let relaxed = scf_cycle(&biased.da, &biased.db, ao, input, &label.to_string(), noci_basis, None, irhf, None).expect("SCF did not converge");

            let mut candidate = relaxed;
            candidate.noci_basis = true;

            // Ensure found state is not a duplicate. 
            let is_duplicate = states.iter().map(|st| (st, electron_distance(st, &candidate, &ao.s))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).is_some_and(|(closest, d2)| {
                if d2 < d_tol {
                    println!("Removed state '{}' from basis as duplicate of '{}' (d^2 = {:.6})", candidate.label, closest.label, d2);
                    true
                } else {
                    false
                }
            });
            if is_duplicate {continue;}

            // No need to check for UHF vs RHF here, if we started with equal densities we should
            // not escape this. All duplicates removed by this point. Since we have successfully found a new
            // state, reset the attempts counter.
            attempt = 0;
            biases_rhf.push(candidate.clone());
            states.push(candidate);
            irhf += 1;
        }
    }
    states
}

/// Pass given AO data and previous SCF solutions to SCF cycle to form the requested reference NOCI basis.
/// # Arguments 
/// - `ao`: Contains AO integrals and other system data. 
/// - `input`: Contains user inputted options. 
/// - `prev`: May or may not contain states from a previous geometry.
/// # Returns
/// - `Vec<SCFState>`: Requested reference NOCI basis.
pub fn generate_reference_noci_basis(ao: &AoData, input: &mut Input, prev: Option<&[SCFState]>,) -> Vec<SCFState> {
    
    // Construct lookuptable from state label to previous SCF states. Allows for seeding of SCF
    // states at a subsequent geometry to be done by label rather than via index which breaks
    // easily.
    let mut prev_map: HashMap<&str, &SCFState> = HashMap::new();
    if let Some(ps) = prev {
        for st in ps {
            prev_map.insert(&st.label, st);
        }
    }

    // Move states out to allow borrows.
    let mut states = std::mem::replace(&mut input.states, StateType::Mom(Vec::new()));

    let out = match &mut states {
        StateType::Mom(recipes) => {
            generate_states_mom(ao, &*input, prev, &prev_map, recipes)
        }
        StateType::Metadynamics(meta) => {
            generate_states_metadynamics(ao, &*input, &prev_map, meta)
        }
    };

    // Put back. Note that this is not very idiomatic. Should definetly refactor this somehow.
    input.states = states;
    out
}

/// Construct a label describing an excitation in alpha and/or beta spin.
/// # Arguments 
/// - `alpha_holes`: Occupied alpha orbital indices from which electrons are removed.
/// - `alpha_parts`: Virtual alpha orbital indices into which electrons are placed.
/// - `beta_holes`: Occupied beta orbital indices from which electrons are removed.
/// - `beta_parts`: Virtual beta orbital indices into which electrons are placed.
/// # Returns
/// - `String`: Label describing the excitation pattern.
fn excitation_label(alpha_holes: &[usize], alpha_parts: &[usize], beta_holes: &[usize], beta_parts: &[usize]) -> String {
    let mut label = Vec::new();
    if !alpha_holes.is_empty() {
        label.push(format!("alpha {} -> {}", 
                alpha_holes.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "),  
                alpha_parts.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "),
        ))
    }
    if !beta_holes.is_empty() {
        label.push(format!("beta {} -> {}", 
                beta_holes.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "),  
                beta_parts.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "),
        ))
    }
    format!("({})", label.join("; "))
}

/// Construct an excitation object from alpha and beta hole/particle lists.
/// # Arguments 
/// - `alpha_holes`: Occupied alpha orbital indices from which electrons are removed.
/// - `alpha_parts`: Virtual alpha orbital indices into which electrons are placed.
/// - `beta_holes`: Occupied beta orbital indices from which electrons are removed.
/// - `beta_parts`: Virtual beta orbital indices into which electrons are placed.
/// # Returns
/// - `Excitation`: Excitation object containing the specified alpha and beta spin excitations.
fn build_excitation(alpha_holes: &[usize], alpha_parts: &[usize], beta_holes: &[usize], beta_parts: &[usize]) -> Excitation {
    Excitation {
        alpha: ExcitationSpin {
            holes: alpha_holes.to_vec(),
            parts: alpha_parts.to_vec(),
        },
        beta: ExcitationSpin {
            holes: beta_holes.to_vec(),
            parts: beta_parts.to_vec(),
        },
    }
}

/// Apply a spin-specific excitation to an occupation vector.
/// # Arguments 
/// - `occ`: Occupation vector to be modified.
/// - `holes`: Occupied orbital indices from which electrons are removed.
/// - `parts`: Virtual orbital indices into which electrons are placed.
/// # Returns
/// - `Array1<f64>`: New occupation vector with the requested excitation applied.
fn apply_excitation(occ: &Array1<f64>, holes: &[usize], parts: &[usize]) -> Array1<f64> {
    let mut out = occ.clone();
    for &i in holes {out[i] = 0.0;}
    for &a in parts {out[a] = 1.0;}
    out
}

/// Generate a requested amount of all possible excitations on top of the given reference NOCI
/// basis.  
/// # Arguments
/// - `refs`: Array of reference states for which excitations are generated.
/// - `input`: Contains user inputted options. 
/// - `include_refs`: Whether or not to include the references in the returned basis.
/// # Returns
/// - `Vec<SCFState>`: Generated excited basis, optionally including the reference states.
pub fn generate_excited_basis(refs: &[SCFState], input: &Input, include_refs: bool) -> Vec<SCFState> {
    let mut out = Vec::new();

    let mut orders = input.excit.orders.clone();
    orders.sort_unstable();
    orders.dedup();

    for r in refs {
        let parent = r.parent;

        if include_refs {
            let mut rcopy = r.clone();
            rcopy.parent = parent;
            out.push(rcopy);
        }

        let spin_occ = get_spin_occupation(r);

        for &k in &orders {
            for k_alpha in 0..=k {
                let k_beta = k - k_alpha;

                for alpha_holes in spin_occ.occ_alpha.iter().copied().combinations(k_alpha) {
                    for alpha_parts in spin_occ.virt_alpha.iter().copied().combinations(k_alpha) {
                        for beta_holes in spin_occ.occ_beta.iter().copied().combinations(k_beta) {
                            for beta_parts in spin_occ.virt_beta.iter().copied().combinations(k_beta) {
                                let oa_ex = apply_excitation(&r.oa, &alpha_holes, &alpha_parts);
                                let ob_ex = apply_excitation(&r.ob, &beta_holes, &beta_parts);

                                let label = excitation_label(&alpha_holes, &alpha_parts, &beta_holes, &beta_parts);
                                let excitation = build_excitation(&alpha_holes, &alpha_parts, &beta_holes, &beta_parts);

                                let exstate = make_excited_state(r, oa_ex, ob_ex, &label, parent, excitation);
                                out.push(exstate);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
