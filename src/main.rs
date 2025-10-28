use ndarray::{Array2, Array4};
use noci_rs::read::read_integrals;
use noci_rs::basis::generate_scf_state;

fn print_array2(array2: &Array2<f64>) {
    for ((i, j), &v) in array2.indexed_iter() {
        println!("({i}, {j}) : {v}");
    }
}

fn print_array4(array4: &Array4<f64>) {
    for ((i, j, k, l), &v) in array4.indexed_iter() {
        println!("({i}, {j}, {k}, {l}) : {v}");
    }
}

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p, 
        None => {
            eprintln!("Usage: cargo run <data.h5>"); std::process::exit(1);
        }
    };

    let ao = read_integrals(&path);
    println!("ERIs:");
    print_array4(&ao.eri);
    println!("AO overlap matrix:");
    print_array2(&ao.s);
    println!("One electron Hamiltonian:");
    print_array2(&ao.h);
    println!("Density matrix ansatz:");
    print_array2(&ao.dm);
    println!("Nuclear Repulsion Energy: {}", ao.enuc);
    println!("Number of electrons: {}", ao.nelec);
    println!("Number of AOs: {}", ao.nao);

    let max_cycle = 50; 
    let tol = 1e-8;
    let state = generate_scf_state(&ao, max_cycle, tol);
    println!("SCF energy: {}", state.e);
    println!("Ca:");
    print_array2(&state.ca);
    println!("Cb:");
    print_array2(&state.cb);
}
