// nocc/deterministic/write.rs

// Standard library imports.
use std::fs::File;
use std::io::{BufWriter, Write};

// External crate imports.
use ndarray::{Array1, Array2};
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
        "{:<6} {:>20} {:>14} {:>20} {:>20} {:>14} {:>14} {:>14}",
        "iter", "EProj", "|dE|", "E(G)", "Shift", "NWalk", "NRef", "max|R|"
    );
}

/// Print one row of the JM-NOCCMC propagation table.
/// # Arguments:
/// - `iter`: Step number.
/// - `energies`: Projected energy, its absolute change from the previous step and the lowest
///   state energy of `G`.
/// - `shift`: Population-control shift `E_s`.
/// - `populations`: Total population `\sum_x|N_x| + \sum_{x\mu}|n_{x\mu}|` and reference
///   population `\sum_x|N_x|`.
/// - `rmax`: Largest residual magnitude.
/// # Returns:
/// - `()`: Prints one propagation row.
pub(super) fn print_jm_table_row(
    iter: usize,
    energies: (f64, f64, f64),
    shift: f64,
    populations: (f64, f64),
    rmax: f64,
) {
    let (e, de, eg) = energies;
    let (total, nref) = populations;
    println!(
        "{:<6} {:>20.12} {:>14.6e} {:>20.12} {:>20.12} {:>14.6e} {:>14.6e} {:>14.6e}",
        iter, e, de, eg, shift, total, nref, rmax
    );
}

/// Print the state energies, the effective overlap, the reference populations and the amplitude
/// norms at the end of the propagation.
/// # Arguments:
/// - `energies`: Eigenvalues of `G`, sorted by real part.
/// - `se`: Effective overlap `S^e`.
/// - `nref`: Reference populations `N_x`, normalised to `\sum_x|N_x| = 1`.
/// - `norms`: Amplitude norm of every reference.
/// # Returns:
/// - `()`: Prints the converged JM-NOCCMC summary.
pub(super) fn print_jm_energies(
    energies: &[Complex64],
    se: &Array2<f64>,
    nref: &Array1<f64>,
    norms: &[f64],
) {
    println!("{}", "=".repeat(100));
    for (k, e) in energies.iter().enumerate() {
        println!("JM-NOCCMC state {}: E = {:.12} {:+.3e}i", k, e.re, e.im);
    }
    println!("Effective overlap S^e:\n{:.8}", se);
    for (x, (n, norm)) in nref.iter().zip(norms).enumerate() {
        println!(
            "Reference {}: population {:+.8e}, amplitude norm {:.8e}",
            x, n, norm
        );
    }
}

/// Print the stability of the linearised amplitude update. Near a solution the update
/// linearises to `I - \Delta\tau J`, which contracts only if every eigenvalue `\lambda` of `J` has
/// `\mathrm{Re}\,\lambda > 0` and `\Delta\tau < 2\,\mathrm{Re}\,\lambda / |\lambda|^2`; an eigenvalue
/// with negative real part is a growing mode, such as an intruder below a target root.
/// # Arguments:
/// - `label`: Where in the propagation the Jacobian is taken.
/// - `eigenvalues`: Eigenvalues of `J`, sorted by real part.
/// - `condition`: Condition number of `S^e`.
/// # Returns:
/// - `()`: Prints the lowest eigenvalues, the largest stable step and the conditioning.
pub(super) fn print_jm_stability(
    label: &str,
    eigenvalues: &[Complex64],
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
    let growing = eigenvalues.iter().filter(|w| w.re <= 0.0).count();

    println!("{}", "=".repeat(100));
    println!("Jacobian diagnostics at the {} amplitudes", label);
    println!("Lowest eigenvalues of J: {}", lowest);
    println!("Largest |eigenvalue| of J: {:.4e}", largest);
    println!("Eigenvalues with non-positive real part: {}", growing);
    if growing == 0 {
        println!("Largest stable step: {:.4e}", limit);
    } else {
        println!("Linearised update is unstable.");
    }
    println!("Condition number of S^e: {:.4e}", condition);
}

/// Print the partition of the Jacobian singular values into a near-null part
/// `\sigma_i < \epsilon\sigma_{\max}` and its complement, and the lowest singular values with the
/// external fractions of their right singular vectors.
/// # Arguments:
/// - `eps`: Relative singular-value cutoff `\epsilon`.
/// - `sigma`: Singular values of `J` in descending order.
/// - `lowest`: Lowest singular values in ascending order with their external fractions, if the
///   expansion-column overlap was built.
/// # Returns:
/// - `()`: Prints the singular-value partition and the lowest singular values.
pub(super) fn print_jm_null_space(
    eps: f64,
    sigma: &Array1<f64>,
    lowest: &[(f64, Option<f64>)],
) {
    let smax = sigma.first().copied().unwrap_or(0.0);
    let null = sigma.iter().filter(|&&x| x < eps * smax).count();
    println!(
        "Singular values of J: eps: {:.3e}, dim: {}, range: {}, null: {}, max sigma: {:.6e}",
        eps,
        sigma.len(),
        sigma.len() - null,
        null,
        smax
    );
    println!("Lowest singular values of J with the external fraction of their right vectors:");
    for (i, (s, fraction)) in lowest.iter().enumerate() {
        let fraction = fraction.map_or("-".to_string(), |f| format!("{:.6}", f));
        println!(
            "    {:>6}: sigma: {:.6e}, external fraction: {}",
            i, s, fraction
        );
    }
}

/// Write the amplitude error `\mathbf t_k - \mathbf t_\star` of every step resolved along the
/// right singular vectors `\mathbf v_i` of the final Jacobian, as rows
/// `iter,space,vector,sigma,component` with `component = \mathbf v_i\cdot(\mathbf t_k - \mathbf t_\star)`,
/// vectors numbered from the lowest singular value and `space` naming the near-null part
/// `\sigma_i < \epsilon\sigma_{\max}` or its complement.
/// # Arguments:
/// - `path`: Output file.
/// - `history`: Flattened amplitudes of every step.
/// - `tfinal`: Flattened final amplitudes `\mathbf t_\star`.
/// - `svd`: Singular values in descending order and right singular vectors as rows.
/// - `eps`: Relative singular-value cutoff `\epsilon`.
/// # Returns:
/// - `()`: Writes the history file.
pub(super) fn write_jm_null_history(
    path: &str,
    history: &[Array1<f64>],
    tfinal: &Array1<f64>,
    svd: (&Array1<f64>, &Array2<f64>),
    eps: f64,
) {
    let (sigma, vt) = svd;
    let smax = sigma.first().copied().unwrap_or(0.0);
    let n = sigma.len();
    let mut writer = BufWriter::new(File::create(path).unwrap());
    writeln!(writer, "iter,space,vector,sigma,component").unwrap();
    for (iter, t) in history.iter().enumerate() {
        let components = vt.dot(&(t - tfinal));
        for k in 0..n {
            let i = n - 1 - k;
            let space = if sigma[i] < eps * smax {
                "null"
            } else {
                "range"
            };
            writeln!(
                writer,
                "{},{},{},{:e},{:e}",
                iter, space, k, sigma[i], components[i]
            )
            .unwrap();
        }
    }
}
