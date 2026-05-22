// basis/metadynamics.rs

use std::collections::HashMap;

use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::input::{Input, Metadynamics};
use crate::scf::scf_cycle;
use crate::utils::random_pattern;
use crate::{AoData, HSCFState, SCFState};

use super::atoms::{atom_count, atomao_for_labels};
use super::bias::{bias_spatial, bias_spin};
use super::duplicate::electron_distance;
use super::types::ReferenceBasis;

/// Generate the SCF states using the SCF metadynamics procedure.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: Contains user inputted options.
/// - `prev_map`: Map between the SCFState object and its label.
/// - `meta`: SCF metadynamics parameters.
/// # Returns:
/// - `Vec<SCFState>`: Generated SCF states.
pub(crate) fn generate_states_metadynamics(
    ao: &AoData,
    input: &Input,
    prev_map: &HashMap<&str, &SCFState>,
    meta: &mut Metadynamics,
) -> Vec<SCFState> {
    let da0: Array2<f64> = ao.dm.clone() * 0.5;
    let db0: Array2<f64> = ao.dm.clone() * 0.5;
    let rhf_tol = 1e-2;

    let mut biases_rhf: Vec<SCFState> = Vec::with_capacity(meta.nstates_rhf);
    let mut biases_uhf: Vec<SCFState> = Vec::with_capacity(meta.nstates_uhf);

    let natoms = atom_count(&ao.labels);
    let atomao = atomao_for_labels(&ao.labels);
    let mut states: Vec<SCFState> = Vec::with_capacity(meta.nstates_rhf + meta.nstates_uhf);

    let mut attempt = 0_u64;
    let mut irhf = 0;
    while biases_uhf.len() < meta.nstates_uhf || biases_rhf.len() < meta.nstates_rhf {
        attempt += 1;
        if attempt > (meta.max_attempts as u64) {
            println!(
                "Maximum number of attempts to find UHF solution has been exceeded. UHF is likely not distinct from RHF at this geometry."
            );
            break;
        }

        // Attempt spin biased state. Usually produces UHF but can collapse to RHF.
        if meta.nstates_uhf > 0 && biases_uhf.len() < meta.nstates_uhf {
            // Assume that all metadynamics found states are to be used in NOCI basis.
            let noci_basis = true;

            let pair = biases_uhf.len() / 2;
            let base = biases_uhf.len();
            if base + 1 >= meta.labels_uhf.len() {
                break;
            }

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
                let mut rng = StdRng::seed_from_u64(
                    attempt.wrapping_add((pair as u64).wrapping_mul(0x9E3779B97F4A7C15_u64)),
                );
                random_pattern(&mut rng, natoms)
            };

            let (da0, db0) = (da.clone(), db.clone());

            // Any spin-broken UHF state should have a spin-flipped degenerate counterpart, find both.
            for (j, cand_slot) in cand.iter_mut().enumerate() {
                let labelidx = base + j;
                if labelidx >= meta.nstates_uhf {
                    break;
                }
                let label = &meta.labels_uhf[labelidx];

                let (mut daj, mut dbj) = (da0.clone(), db0.clone());

                // Bias the densities as prescribed by the pattern.
                bias_spin(&mut daj, &mut dbj, &atomao, meta.spinpol, &pattern);

                // When j == 1 we swap the densities so as to get the spin-flipped density.
                if j == 1 {
                    std::mem::swap(&mut daj, &mut dbj);
                }

                if input.write.verbose {
                    let left = "=".repeat(45);
                    let right = "=".repeat(46);
                    println!("{}Begin UHF Metadynamics Biased SCF{}", left, right);
                    println!(
                        "State({}): {}, Spin-flip: {}, Attempt: {}",
                        labelidx, label, j, attempt
                    );
                }

                let biased = {
                    let biasi = if biases_uhf.is_empty() {
                        None
                    } else {
                        Some(biases_uhf.as_slice())
                    };

                    scf_cycle(
                        (&daj, &dbj),
                        ao,
                        input,
                        label,
                        noci_basis,
                        labelidx,
                        (None, biasi),
                    )
                    .expect("SCF did not converge")
                };

                if input.write.verbose {
                    let left = "=".repeat(45);
                    let right = "=".repeat(46);
                    println!("{}Begin UHF Metadynamics Relaxed SCF{}", left, right);
                    println!(
                        "State({}): {}, Spin-flip: {}, Attempt: {}",
                        labelidx, label, j, attempt
                    );
                }

                let relaxed = scf_cycle(
                    (biased.da.as_ref(), biased.db.as_ref()),
                    ao,
                    input,
                    label,
                    noci_basis,
                    labelidx,
                    (None, None),
                )
                .expect("SCF did not converge");

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
                    println!(
                        "UHF candidate collapsed to RHF: drhf = {:.3e} < {:.3e}",
                        drhf, rhf_tol
                    );
                    let dup = biases_rhf
                        .iter()
                        .map(|st| electron_distance(st, &candidate, &ao.s))
                        .any(|d2| d2 < input.scf.d_tol);
                    if dup {
                        println!(
                            "Removed state '{}' from basis as duplicate.",
                            candidate.label
                        );
                        continue;
                    }

                    // Even if zero RHF were requested, UHF and RHF may not be distinct so we add
                    // it to the states anyway to avoid having an empty basis.
                    if meta.nstates_rhf == 0 && irhf >= meta.nstates_rhf {
                        meta.nstates_rhf = 1;
                        meta.labels_rhf.push("RHF".to_string());
                        meta.spatial_patterns_rhf.push(Some(pattern.clone()));
                    }
                    if irhf >= meta.nstates_rhf {
                        continue;
                    }
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
                    println!(
                        "UHF candidate stayed UHF: drhf = {:.3e} > {:.3e}",
                        drhf, rhf_tol
                    );
                    let dup = biases_uhf
                        .iter()
                        .map(|st| electron_distance(st, &candidate, &ao.s))
                        .any(|d2| d2 < input.scf.d_tol);
                    if dup {
                        println!(
                            "Removed state '{}' from basis as duplicate.",
                            candidate.label
                        );
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
                    *cand_slot = Some(candidate);
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
                let mut rng = StdRng::seed_from_u64(
                    attempt.wrapping_add((irhf as u64).wrapping_mul(0x9E3779B97F4A7C15_u64)),
                );
                random_pattern(&mut rng, natoms)
            };

            // Bias the densities as prescribed by the pattern.
            bias_spatial(&mut da, &mut db, &atomao, meta.spatialpol, &pattern);

            if input.write.verbose {
                let left = "=".repeat(45);
                let right = "=".repeat(46);
                println!("{}Begin RHF Metadynamics Biased SCF{}", left, right);
                println!("State({}): {}, Attempt: {}", irhf, label, attempt);
            }

            let biased = {
                let biasi = if biases_uhf.is_empty() {
                    None
                } else {
                    Some(biases_uhf.as_slice())
                };

                scf_cycle(
                    (&da, &db),
                    ao,
                    input,
                    label,
                    noci_basis,
                    irhf,
                    (None, biasi),
                )
                .expect("SCF did not converge")
            };

            if input.write.verbose {
                let left = "=".repeat(45);
                let right = "=".repeat(46);
                println!("{}Begin RHF Metadynamics Relaxed SCF{}", left, right);
                println!("State({}): {}, Attempt: {}", irhf, label, attempt);
            }

            let relaxed = scf_cycle(
                (biased.da.as_ref(), biased.db.as_ref()),
                ao,
                input,
                label,
                noci_basis,
                irhf,
                (None, None),
            )
            .expect("SCF did not converge");

            let mut candidate = relaxed;
            candidate.noci_basis = true;

            // Ensure found state is not a duplicate.
            let is_duplicate = states
                .iter()
                .map(|st| (st, electron_distance(st, &candidate, &ao.s)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .is_some_and(|(closest, d2)| {
                    println!(
                        "State '{}' electron distance from state '{}': {}",
                        candidate.label, closest.label, d2
                    );
                    if d2 < input.scf.d_tol {
                        println!(
                            "Removed state '{}' from basis as duplicate of '{}' (d^2 = {:.6})",
                            candidate.label, closest.label, d2
                        );
                        true
                    } else {
                        false
                    }
                });
            if is_duplicate {
                continue;
            }

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

/// Generate the metadynamics-backed reference NOCI basis states.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: Contains user inputted options.
/// - `prev_h`: Previous h-SCF states. Any value means complex metadynamics was requested.
/// - `prev_map`: Map between the SCFState object and its label.
/// - `meta`: SCF metadynamics parameters.
/// # Returns:
/// - `ReferenceBasis`: Real SCF states found by metadynamics and no h-SCF states.
pub(crate) fn generate_reference_basis_metadynamics(
    ao: &AoData,
    input: &Input,
    prev_h: Option<&[HSCFState]>,
    prev_map: &HashMap<&str, &SCFState>,
    meta: &mut Metadynamics,
) -> ReferenceBasis {
    if prev_h.is_some() {
        panic!("Holomorphic SCF metadynamics is not implemented.");
    }

    ReferenceBasis {
        states: generate_states_metadynamics(ao, input, prev_map, meta),
        hstates: Vec::new(),
    }
}
