// main.rs
use std::process::Command;
use std::time::{Instant};
use ndarray::Array1;
use num_complex::Complex64;
use std::fs::File;
use std::io::{BufWriter, Write};

use noci_rs::input::load_input;
use noci_rs::read::read_integrals;
use noci_rs::basis::{generate_reference_noci_basis, generate_qmc_noci_basis};
use noci_rs::noci::{calculate_noci_energy, build_noci_matrices};
use noci_rs::deterministic::{propagate, projected_energy};
use noci_rs::SCFState;

fn main() {
    let t_total = Instant::now();
    let input_path = match std::env::args().nth(1) {
        Some(p) => p, 
        None => {
            eprintln!("Usage: cargo run <input.lua>"); 
            std::process::exit(1);
        }
    };

    let input = load_input(&input_path);
    let mut prev_states: Vec<SCFState> = Vec::new();
   
    println!("Running SCF for {} geometries...", input.mol.r_list.len());
    for (i, r) in input.mol.r_list.iter().copied().enumerate(){
        println!("\n");
        let atoms = &input.mol.geoms[i];
        let atomsj = serde_json::to_string(atoms).unwrap();
        let t_gen = Instant::now();
        // Run interface to PySCF which generates AO integrals, overlap matrix, density matrices etc.
        let status = Command::new("python3").arg("generate.py").arg("--atoms").arg(&atomsj).arg("--basis").arg(&input.mol.basis)
                                            .arg("--unit").arg(&input.mol.unit).arg("--out").arg("data.h5").arg("--fci")
                                            .arg(if input.scf.do_fci { "true" } else { "false" }).status().unwrap();
        if !status.success() {
            eprintln!("Failed to generate mol with status {status}");
            std::process::exit(1);
        }
        let d_gen = t_gen.elapsed();
        
        // Read integrals from the generated data.
        let ao = read_integrals("data.h5");

        let mut use_prev_seed = !prev_states.is_empty();
        let mut tried_unseed = false;
        
        loop {
            let t_scf = Instant::now();
            // Calculate SCF states.
            // If we have solutions at a previous geometry, use them.
            let states = if use_prev_seed {
                generate_reference_noci_basis(&ao, &input, Some(&prev_states))
            } else {
                generate_reference_noci_basis(&ao, &input, None)
            };
            let d_scf = t_scf.elapsed();
            println!("{}", "=".repeat(100));
       
            // Determine which states are to be used in the reference NOCI basis.
            let t_noci_reference = Instant::now();
            let mut noci_reference_basis = Vec::new();
            for state in states.iter() {
                if state.noci_basis {
                    noci_reference_basis.push(state.clone());
                }
            }

            // Pass SCF states to NOCI subroutines to and NOCI energy from the given basis.
            let (e_noci, c0, d_h1_h2_reference) = calculate_noci_energy(&ao, &noci_reference_basis);
            let d_noci_reference = t_noci_reference.elapsed();

            // Construct the requested excitation space NOCI-QMC basis ontop of the reference basis and
            // get the corresponding Hamiltonian and overlap.
            let t_noci_qmc_deterministic = Instant::now();
            let t_noci_qmc_deterministic_basis_construction = Instant::now();
            println!("Building NOCI-QMC basis....");
            let noci_qmc_basis = generate_qmc_noci_basis(&ao, &noci_reference_basis, &input);
            println!("Built NOCI-QMC basis of {} determinants.", noci_qmc_basis.len());
            let n = noci_qmc_basis.len();
            println!("Calculating NOCI-QMC matrix elements for {} determinants ({} elements)...", n, n * n);
            let d_noci_qmc_deterministic_basis_construction = t_noci_qmc_deterministic_basis_construction.elapsed();
            let (h_qmc, s_qmc, d_h1_h2_qmc) = build_noci_matrices(&ao, &noci_qmc_basis);
            println!("Finished calculating NOCI-QMC matrix elements.");
            
            // Choose intial shift energy.
            let es = states[0].e; // RHF energy.
            println!("Running deterministic NOCI-QMC propagation....");
            let t_noci_qmc_deterministic_propagation = Instant::now();

            // Embed reference NOCI coefficient vector in full NOCI-QMC space.
            let n_qmc = noci_qmc_basis.len();
            let mut c0_qmc = Array1::<Complex64>::zeros(n_qmc);  
            
            // If we are not interested in plotting evolution of individual coefficients we use the
            // reference NOCI coefficients as our initial guess as this is the best guess, however,
            // the coefficients often don't change much which makes for a boring plot.
            if !input.write.write_coeffs {
                // Iterate over states in the full NOCI-QMC basis and place reference coefficients on the
                // states with reference state labels. This could break if the reference states are not in
                // the expected order so should be made more robust.
                for (i, ref_st) in noci_reference_basis.iter().enumerate() {
                    let idx = noci_qmc_basis.iter().position(|qmc_st| qmc_st.label == ref_st.label).unwrap();
                    c0_qmc[idx] = c0[i];
                }
            // If we are interested in plotting the evolution of individual coefficients we use an
            // equal weighting of all SCF states as our initial guess.
            } else {
                c0_qmc = Array1::from_elem(n_qmc, Complex64::new(1.0 / (n_qmc as f64).sqrt(), 0.0));
            };   
            println!("Initial wavefunction ansatz (C0-QMC): {}", c0_qmc);

            // Propagate coefficient vector deterministically in full NOCI-QMC basis.
            let mut coefficients = Vec::new();
            let propagation_result = propagate(&h_qmc, &s_qmc, &c0_qmc, es, &mut coefficients, &input);
            let d_noci_qmc_deterministic_propagation = t_noci_qmc_deterministic_propagation.elapsed();
            let d_noci_qmc_deterministic = t_noci_qmc_deterministic.elapsed();

            match propagation_result {
                Some(c) => {
                    let e_tot = projected_energy(&h_qmc, &s_qmc, &c);
                    println!("{}", "=".repeat(100));
                    println!("NOCI reference basis size: {}", noci_reference_basis.len());
                    println!("NOCI-QMC basis size: {}", noci_qmc_basis.len());
                    println!("{}", "=".repeat(100));

                    println!("Total PySCF time: {:?}", d_gen);
                    println!("Total SCF time: {:?}", d_scf);
                    print!("");

                    println!("Total Reference NOCI time: {:?}", d_noci_reference);
                    println!(r"  H_1 & H_2: {:?}", d_h1_h2_reference);
                    print!("");

                    println!("Total NOCI-QMC deterministic time: {:?}", d_noci_qmc_deterministic);
                    println!(r"  Basis generation:  {:?}", d_noci_qmc_deterministic_basis_construction);
                    println!(r"  H_1 & H_2: {:?}", d_h1_h2_qmc);
                    println!(r"  Deterministic propagation:  {:?}", d_noci_qmc_deterministic_propagation);
                    
                    println!("{}", "=".repeat(100));
                    println!("R: {}", r);
                    for (i, state) in states.iter().enumerate() {
                        println!("State({}): {},  E: {}", i + 1, state.label, state.e);
                    }
                    println!("State(NOCI-reference): E: {}, [E - E(RHF)]: {}", e_noci, e_noci - states[0].e);
                    println!("State(NOCI-qmc-deterministic): E: {}, [E - E(RHF)]: {}", e_tot, e_tot - states[0].e);
                    if let Some(e_fci) = ao.e_fci {
                        println!("State(FCI): E: {},  [E - E(RHF)]: {}", e_fci, e_fci - states[0].e);
                    }
                    
                    if input.write.write_coeffs {
                        // Write all coefficients to a file. Currently if we are doing multiple
                        // geometries this file will just overwrite itself and so we end up with only
                        // the coefficients of the final geometry. Also this writing is slow. 
                        println!("Writing coefficients to file...");
                        let filepath = format!("{}/{}", input.write.coeffs_dir, input.write.coeffs_filename);
                        let file = File::create(filepath).unwrap();
                        let mut writer = BufWriter::new(file);
                        for iter in &coefficients {
                            writeln!(writer, "iter {}", iter.iter).unwrap();
                            writeln!(writer, "Full coefficients:").unwrap();
                            for (i, z) in iter.c_full.iter().enumerate() {
                                writeln!(writer, "{:4} {:.8e}  {:.8e}", i, z.re, z.im).unwrap();
                            }
                            writeln!(writer, "Relevant space coefficients:").unwrap();
                            for (i, z) in iter.c_relevant.iter().enumerate() {
                                writeln!(writer, "{:4} {:.8e} {:.8e}", i, z.re, z.im).unwrap();
                            }
                            writeln!(writer, "Null space coefficients:").unwrap();
                            for (i, z) in iter.c_null.iter().enumerate() {
                                writeln!(writer, "{:4} {:.8e} {:.8e}", i, z.re, z.im).unwrap();
                            }
                            writeln!(writer).unwrap(); 
                        }
                    }
                    
                    prev_states = states.clone();
                    break;
                } None => {
                    println!("Propagation failed.");
                    if use_prev_seed && !tried_unseed {
                        println!("Retrying SCF/NOCI/NOCI-QMC without seeding from previous geometry.");
                        use_prev_seed = false;
                        tried_unseed = true;
                        continue;
                    } else {
                        println!("{}", "=".repeat(100));
                        println!("R: {}, propagation did not converge. Skipping geometry.", r);
                        break;
                    }
                }
            }
        }
    }
    println!("\n Total wall time: {:?}", t_total.elapsed());
}
