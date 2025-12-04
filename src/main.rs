// main.rs
use std::process::Command;
use std::time::{Instant};
use ndarray::Array1;
use num_complex::Complex64;

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
   
    println!("Running SCF for {} geometries...", input.r_list.len());
    for (i, r) in input.r_list.iter().copied().enumerate(){
        println!("\n");
        let atoms = &input.geoms[i];
        let atomsj = serde_json::to_string(atoms).unwrap();
        let t_gen = Instant::now();
        // Run interface to PySCF which generates AO integrals, overlap matrix, density matrices etc.
        let status = Command::new("python3").arg("generate.py").arg("--atoms").arg(&atomsj).arg("--basis").arg(&input.basis)
                                            .arg("--unit").arg(&input.unit).arg("--out").arg("data.h5").arg("--fci")
                                            .arg(if input.do_fci { "true" } else { "false" }).status().unwrap();
        if !status.success() {
            eprintln!("Failed to generate mol with status {status}");
            std::process::exit(1);
        }
        let d_gen = t_gen.elapsed();
        
        // Read integrals from the generated data and calculate SCF states.
        let ao = read_integrals("data.h5");
        let t_scf = Instant::now();
        // If we have solutions at a previous geometry, use them.
        let states = if prev_states.is_empty() {
            generate_reference_noci_basis(&ao, &input, None)
        } else {
            generate_reference_noci_basis(&ao, &input, Some(&prev_states))
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
        let (e_noci, c0, timings_reference) = calculate_noci_energy(&ao, &noci_reference_basis);
        let d_noci_reference = t_noci_reference.elapsed();

        // Construct the requested excitation space NOCI-QMC basis ontop of the reference basis and
        // get the corresponding Hamiltonian and overlap.
        let t_noci_qmc_deterministic = Instant::now();
        let t_noci_qmc_deterministic_basis_construction = Instant::now();
        println!("Building NOCI-QMC basis....");
        let noci_qmc_basis = generate_qmc_noci_basis(&ao, &noci_reference_basis, &input);
        println!("Calculating NOCI-QMC matrix elements....");
        let d_noci_qmc_deterministic_basis_construction = t_noci_qmc_deterministic_basis_construction.elapsed();
        let (h_qmc, s_qmc, timings_qmc_deterministic) = build_noci_matrices(&ao, &noci_qmc_basis);
       
        // Choose shift, timestep, maximum number of steps, and energy tolerance. 
        let es = states[0].e; // RHF energy.
        let dt = 1e-6;
        let max_steps = 10000;
        let e_tol = 1e-8;
        
        let h_qmc_shift = &h_qmc - Complex64::new(es, 0.0) * &s_qmc;
        println!("Reference energy shift (es): {}", es);
        println!("Time-step (dt): {}", dt);
        println!("{}", "=".repeat(100));

        println!("Running deterministic NOCI-QMC propagation....");
        let t_noci_qmc_deterministic_propagation = Instant::now();

        // Embed reference NOCI coefficient vector in full NOCI-QMC space.
        let n_qmc = noci_qmc_basis.len();
        let mut c0_qmc = Array1::<Complex64>::zeros(n_qmc);  
        // Iterate over states in the full NOCI-QMC basis and place reference coefficients on the
        // states with reference state labels. This could break if the reference states are not in
        // the expected order so should be made more robust.
        for (i, ref_st) in noci_reference_basis.iter().enumerate() {
            let idx = noci_qmc_basis.iter().position(|qmc_st| qmc_st.label == ref_st.label).unwrap();
            c0_qmc[idx] = c0[i];
        }
        println!("Initial wavefunction ansatz (C0-QMC): {}", c0_qmc);

        // Propagate coefficient vector deterministically in full NOCI-QMC basis. 
        let e_tot: f64 = match propagate(&h_qmc_shift, &s_qmc, &c0_qmc, es, dt, max_steps, e_tol) {
            Some(c) => projected_energy(&h_qmc_shift, &s_qmc, &c, es),
            // Propagation may converge to a ridiculous eigenvalue if H or S are singular. Fix.
            None => {eprintln!("Propagation failed, setting energy to NaN."); f64::NAN}
        };
        let d_noci_qmc_deterministic_propagation = t_noci_qmc_deterministic_propagation.elapsed();
        let d_noci_qmc_deterministic = t_noci_qmc_deterministic.elapsed();

        println!("{}", "=".repeat(100));
        println!("NOCI reference basis size: {}", noci_reference_basis.len());
        println!("NOCI-QMC basis size: {}", noci_qmc_basis.len());
        println!("{}", "=".repeat(100));

        println!("Total PySCF time: {:?}", d_gen);
        println!("Total SCF time: {:?}", d_scf);
        print!("");

        println!("Total Reference NOCI time: {:?}", d_noci_reference);
        println!(r"  {{}}^{{\mu\nu}}S:  {:?}", timings_reference.munu_s);
        println!(r"  S_{{\text{{NOCI}}}}:  {:?}", timings_reference.s_noci);
        println!(r"  S_{{\text{{red}}}}:   {:?}", timings_reference.s_red);
        println!(r"  H_1 & H_2: {:?}", timings_reference.h);
        print!("");

        println!("Total NOCI-QMC deterministic time: {:?}", d_noci_qmc_deterministic);
        println!(r"  Basis generation:  {:?}", d_noci_qmc_deterministic_basis_construction);
        println!(r"  {{}}^{{\mu\nu}}S:  {:?}", timings_qmc_deterministic.munu_s);
        println!(r"  S_{{\text{{NOCI}}}}:  {:?}", timings_qmc_deterministic.s_noci);
        println!(r"  S_{{\text{{red}}}}:   {:?}", timings_qmc_deterministic.s_red);
        println!(r"  H_1 & H_2: {:?}", timings_qmc_deterministic.h);
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
        
        prev_states = states.clone();
    }

    println!("\n Total wall time: {:?}", t_total.elapsed());
}
