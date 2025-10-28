use noci_rs::read::read_integrals;
use ndarray::{Array2, Array4};

fn print_array2(array2: &Array2<f64>) {
    for ((i, j), &v) in array2.indexed_iter() {
        println!("{i}, {j} : {v}");
    }
}

fn print_array4(array4: &Array4<f64>) {
    for ((i, j, k, l), &v) in array4.indexed_iter() {
        println!("({i}, {j}, {k}, {l}) : {v}");
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let ao = read_integrals(&path);

    println!("ERIs:");
    print_array4(&ao.eri);
    println!("AO overlap matrix:");
    print_array2(&ao.s);
    println!("One electron Hamiltonian:");
    print_array2(&ao.h);

    println!("Nuclear Repulsion Energy: {}", ao.enuc);
    println!("Number of electrons: {}", ao.nelec);
    println!("Number of AOs: {}", ao.nao);
}
