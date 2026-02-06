// utils.rs
use ndarray::{Array1, Array2, Array4};

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

/// Calculate number of coefficients (or basis states) in a coefficient vector (not including the
/// reference states) required to reach 99%, 99.9%, .... of the total coefficient weight. Yields approximate 
/// indication of the sparsity of a wavefunction.  
/// # Arguments 
///     `c`: [f64], basis coefficient vector.
///     `ref_indices`, indices of the reference states in the coefficient vector.
pub fn wavefunction_sparsity(c: &[f64], ref_indices: &[usize]) {

    // Construct mask for the references.
    let mut ref_mask = vec![false; c.len()];
    for &i in ref_indices {
        if i < c.len() {
            ref_mask[i] = true;
        }
    }

    // Exclude references from c.
    let c_tail: Vec<f64> = c.iter().enumerate().filter_map(|(i, &ci)| if ref_mask[i] {None} else {Some(ci)})
        .collect();

    // w_i = |c_i|^2.
    let mut w: Vec<f64> = c_tail.iter().map(|&x| x * x).collect();
    // \sum_i w_i = \sum_i |c_i|^2.
    let sum_w: f64 = w.iter().sum();
    // Sort weights by descending, i.e., from most to least important.
    w.sort_by(|a, b| b.partial_cmp(a).unwrap());
    
    let targets: [f64; 4] = [0.99, 0.999, 0.9999, 0.99999];
    let mut k = [0_usize; 4];

    let mut cumulative = 0.0_f64;
    let mut t = 0_usize;

    for (k0, &wi) in w.iter().enumerate() {
        // Add next largest weight and calculate fraction of total weight.
        cumulative += wi;
        let frac = cumulative / sum_w;
        
        // Record the number of coefficients k required to reach target[t].
        while t < targets.len() && frac >= targets[t] {
            k[t] = k0 + 1; 
            t += 1;
        }
        // Stop once all targets reached.
        if t == targets.len() {
            break;
        }
    }
    
    // Print diagnostics.
    println!("{}", "=".repeat(100));
    println!("Wavefunction sparsity:");
    println!("sum |c|^2 = {}", sum_w);
    println!("Number of states for 99% of ||c||^2 = {}", k[0]);
    println!("Number of states for 99.9% of ||c||^2 = {}", k[1]);
    println!("Number of states for 99.99% of||c||^2 = {}", k[2]);
    println!("Number of states for 99.999% of ||c||^2 = {}", k[3]);
}

/// Convert an occupation vector into a bitstring.
/// # Arguments:
///     `occ`: Array1, occupancy vector.
///     `tol`: f64, tolerance for a occupancy element being almost 1.
pub fn occvec_to_bits(occ: &Array1<f64>, tol: f64) -> u64 {
    let mut bits = 0u64;
    for (i, &x) in occ.iter().enumerate() {
        if x > 1.0 - tol {bits |= 1u64 << i;}
    }
    bits
}

/// Calculate fermionic sign associated with applying a set of creation and annhilation operators
/// to a determinant described by a bitstring.
/// # Arguments:
///     `occ`: u64, occupancy bitstring.
///     `holes`: [usize], annhilation operators indices.
///     `parts`: [usize], creation operator indices.
pub fn excitation_phase(mut occ: u64, holes: &[usize], parts: &[usize]) -> f64 {

    /// Calculate how many occupied orbitals are below index p.
    /// # Arguments:
    ///     `bits`: u64, occupancy bitstring.
    ///     `p`: usize, a given orbital index.
    fn below(bits: u64, p: usize) -> u32 {
        if p == 0 { return 0; }
        (bits & ((1u64 << p) - 1)).count_ones()
    }

    let mut ph = 1.0;
    
    // Remove annhilations in descending order and accumulate phase.
    let mut hs = holes.to_vec();
    hs.sort_unstable_by(|a,b| b.cmp(a));    
    for &i in &hs {
        if (below(occ, i) & 1) == 1 {ph = -ph;}
        occ &= !(1u64 << i);
    }
    
    // Add creations in ascending order and accumulate phase.
    let mut ps = parts.to_vec();
    ps.sort_unstable();                     
    for &a in &ps {
        if (below(occ, a) & 1) == 1 { ph = -ph; }
        occ |= 1u64 << a;
    }
    ph
}


