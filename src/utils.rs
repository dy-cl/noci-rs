// utils.rs
use ndarray::{Array2, Array4};

/// Print a 2D array as a matrix.
/// # Arguments
///     `a`:  Array2, matrix to print.
pub fn print_array2(a: &Array2<f64>) {
    let (nr, nc) = a.dim();
    for i in 0..nr {
        for j in 0..nc {
            print!("{:>12.6}", a[(i, j)]);
        }
        println!();
    }
}

/// Print a 4D array as grid of 2D blocks.
/// # Arguments 
///    `t`: Array4, tensor to print. 
pub fn print_array4(t: &Array4<f64>) {
    let (np, nq, nr, ns) = t.dim();
    for p in 0..np {
        for q in 0..nq {
            for r in 0..nr {
                for s in 0..ns {
                    print!("{:>12.6}", t[(p, q, r, s)]);
                }
                println!();
            }
            println!();
        }
    }
}

