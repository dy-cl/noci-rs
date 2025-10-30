// main.rs
use noci_rs::read::read_integrals;
use noci_rs::basis::generate_scf_state;

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p, 
        None => {
            eprintln!("Usage: cargo run <data.h5>"); std::process::exit(1);
        }
    };

    let ao = read_integrals(&path);
    println!("Nuclear Repulsion Energy: {}", ao.enuc);
    println!("Number of electrons: {}", ao.nelec);
    println!("Number of AOs: {}", ao.nao);

    let max_cycle = 1000; 
    let e_tol = 1e-8;
    let err_tol = 1e-6;
    let states = generate_scf_state(&ao, max_cycle, e_tol, err_tol);

    for (i, state) in states.iter().enumerate() {
        println!("State({i}): E = {}", state.e);
    }   
}
