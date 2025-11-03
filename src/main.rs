// main.rs
use std::process::Command;

use noci_rs::input::load_input;
use noci_rs::read::read_integrals;
use noci_rs::basis::generate_scf_state;
use noci_rs::noci::calculate_noci_energy;

fn main() {
    let input_path = match std::env::args().nth(1) {
        Some(p) => p, 
        None => {
            eprintln!("Usage: cargo run <input.lua>"); 
            std::process::exit(1);
        }
    };
    
    let input = load_input(&input_path);
   
    println!("Running SCF for {} geometries...", input.r_list.len());
    for r in &input.r_list{
        println!("\n");
        println!("R: {}", r);
        // Run interface to PySCF which generates AO integrals, overlap matrix, density matrices etc.
        let status = Command::new("python3").arg("generate.py").arg("--r").arg(r.to_string())
                             .arg("--atom1").arg(&input.atom1).arg("--atom2").arg(&input.atom2)
                             .arg("--basis").arg(&input.basis).arg("--unit").arg(&input.unit)
                             .arg("--out").arg("data.h5").status().unwrap();
        if !status.success() {
            eprintln!("Failed to generate mol with status {status}");
            std::process::exit(1);
        }
        
        // Read integrals from the generated data and calculate SCF states.
        let ao = read_integrals("data.h5");
        let states = generate_scf_state(&ao, &input);
        println!("==========================================================");
        for (i, state) in states.iter().enumerate() {
            println!("State({i}): E = {}", state.e);
        }

        // Pass SCF states to NOCI subroutines to and NOCI energy from the given basis.
        let e_noci = calculate_noci_energy(&ao, &states);
        println!("State(NOCI): E = {}", e_noci);

    }
}
