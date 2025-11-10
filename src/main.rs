// main.rs
use std::process::Command;
use std::time::{Instant};

use noci_rs::input::load_input;
use noci_rs::read::read_integrals;
use noci_rs::basis::generate_scf_state;
use noci_rs::noci::calculate_noci_energy;

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
   
    println!("Running SCF for {} geometries...", input.r_list.len());
    for (i, r) in input.r_list.iter().copied().enumerate(){
        println!("\n");
        let atoms = &input.geoms[i];
        let atomsj = serde_json::to_string(atoms).unwrap();
        let t_gen = Instant::now();
        // Run interface to PySCF which generates AO integrals, overlap matrix, density matrices etc.
        let status = Command::new("python3").arg("generate.py").arg("--atoms").arg(&atomsj)
                             .arg("--basis").arg(&input.basis).arg("--unit").arg(&input.unit)
                             .arg("--out").arg("data.h5").status().unwrap();
        if !status.success() {
            eprintln!("Failed to generate mol with status {status}");
            std::process::exit(1);
        }
        let d_gen = t_gen.elapsed();
        
        // Read integrals from the generated data and calculate SCF states.
        let ao = read_integrals("data.h5");
        let t_scf = Instant::now();
        let states = generate_scf_state(&ao, &input);
        let d_scf = t_scf.elapsed();
        println!("==========================================================");
        
        for (i, state) in states.iter().enumerate() {
            println!("State({}): {},  E = {}", i, input.states[i].label, state.e);
        }           
    
        // Determine which states are to be used in the NOCI basis.
        let t_noci = Instant::now();
        let mut noci_basis = Vec::new();
        for (state, recipe) in states.iter().zip(&input.states) {
            if recipe.noci {
                noci_basis.push(state.clone());
            }
        }
                
        // Pass SCF states to NOCI subroutines to and NOCI energy from the given basis.
        let (e_noci, timings) = calculate_noci_energy(&ao, &noci_basis);
        let d_noci = t_noci.elapsed();
        println!("State(NOCI): E = {}", e_noci);

        println!("\n R: {}", r);
        println!("Total PySCF time: {:?}", d_gen);
        println!("Total SCF time: {:?}", d_scf);
        println!("Total NOCI time: {:?}", d_noci);
        println!(r"  {{}}^{{\mu\nu}}S:  {:?}", timings.munu_s);
        println!(r"  S_{{\text{{NOCI}}}}:  {:?}", timings.s_noci);
        println!(r"  S_{{\text{{red}}}}:   {:?}", timings.s_red);
        println!(r"  H_1 & H_2: {:?}s", timings.h);

    }

    println!("\n Total wall time: {:?}", t_total.elapsed());
}
