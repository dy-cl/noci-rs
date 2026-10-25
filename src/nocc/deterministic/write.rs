// nocc/deterministic/write.rs

// External crate imports.
use ndarray::Array2;
use num_complex::Complex64;

/// Print the size of one reference's projection and expansion spaces.
/// # Arguments:
/// - `reference`: Reference index.
/// - `determinants`: Number of expansion determinants.
/// - `projected`: Number of projection determinants.
/// - `excitations`: Number of cluster excitations.
/// - `connections`: Number of tabulated cluster-operator connections.
/// # Returns:
/// - `()`: Prints one summary line.
pub(super) fn print_jm_block(
    reference: usize,
    determinants: usize,
    projected: usize,
    excitations: usize,
    connections: usize,
) {
    println!(
        "Reference {}: {} expansion and {} projection determinants, {} excitations, {} connections ({:.1} per determinant).",
        reference,
        determinants,
        projected,
        excitations,
        connections,
        connections as f64 / determinants as f64
    );
}

/// Print the JM-NOCCMC propagation table header.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Prints the table headings.
pub(super) fn print_jm_table_header() {
    println!("{}", "=".repeat(100));
    println!(
        "{:<6} {:>20} {:>16} {:>16} {:>16} {:>16}",
        "iter", "E", "|dE|", "max|R|", "||R||", "||t||"
    );
}

/// Print one row of the JM-NOCCMC propagation table.
/// # Arguments:
/// - `iter`: Step number.
/// - `e`: Ground-state energy, the lowest real eigenvalue of `G`.
/// - `de`: Absolute energy change from the previous step.
/// - `rmax`: Largest residual magnitude.
/// - `rnorm`: Residual norm.
/// - `tnorm`: Amplitude norm.
/// # Returns:
/// - `()`: Prints one propagation row.
pub(super) fn print_jm_table_row(
    iter: usize,
    e: f64,
    de: f64,
    rmax: f64,
    rnorm: f64,
    tnorm: f64,
) {
    println!(
        "{:<6} {:>20.12} {:>16.6e} {:>16.6e} {:>16.6e} {:>16.6e}",
        iter, e, de, rmax, rnorm, tnorm
    );
}

/// Print the state energies, the effective overlap and the amplitude norms at convergence.
/// # Arguments:
/// - `energies`: Eigenvalues of `G`, sorted by real part.
/// - `se`: Effective overlap `S^e`.
/// - `norms`: Amplitude norm of every reference.
/// # Returns:
/// - `()`: Prints the converged JM-NOCCMC summary.
pub(super) fn print_jm_energies(
    energies: &[Complex64],
    se: &Array2<f64>,
    norms: &[f64],
) {
    println!("{}", "=".repeat(100));
    for (k, e) in energies.iter().enumerate() {
        println!("JM-NOCCMC state {}: E = {:.12} {:+.3e}i", k, e.re, e.im);
    }
    println!("Effective overlap S^e:\n{:.8}", se);
    for (x, norm) in norms.iter().enumerate() {
        println!("Reference {} amplitude norm: {:.8e}", x, norm);
    }
}

/// Print the stability of the linearised amplitude update at convergence.
/// # Arguments:
/// - `eigenvalues`: Eigenvalues of the update's linear operator, `J` or `D^{-1}J`, sorted by
///   real part.
/// - `sigma`: Smallest singular value of `J`.
/// - `condition`: Condition number of `S^e`.
/// # Returns:
/// - `()`: Prints the lowest eigenvalues, the largest stable step and the conditioning.
pub(super) fn print_jm_stability(
    eigenvalues: &[Complex64],
    sigma: f64,
    condition: f64,
) {
    let lowest = eigenvalues
        .iter()
        .take(6)
        .map(|w| format!("{:+.4e}{:+.2e}i", w.re, w.im))
        .collect::<Vec<_>>()
        .join(" ");
    // `|1 - \Delta\tau\lambda| < 1` for every eigenvalue gives `\Delta\tau < 2 Re\lambda / |\lambda|^2`.
    let limit = eigenvalues
        .iter()
        .map(|w| 2.0 * w.re / w.norm_sqr())
        .fold(f64::INFINITY, f64::min);
    let largest = eigenvalues.iter().map(|w| w.norm()).fold(0.0, f64::max);

    println!("Linearised update lowest eigenvalues: {}", lowest);
    println!("Linearised update largest |eigenvalue|: {:.4e}", largest);
    if eigenvalues.first().is_some_and(|w| w.re > 0.0) {
        println!("Largest stable step: {:.4e}", limit);
    } else {
        println!("Linearised update is unstable: an eigenvalue has a non-positive real part.");
    }
    println!("Smallest singular value of J: {:.4e}", sigma);
    println!("Condition number of S^e: {:.4e}", condition);
}
