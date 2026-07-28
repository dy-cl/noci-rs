// deterministic/write.rs

use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Eigh, Norm, UPLO};

use crate::DetState;
use crate::deterministic::{ProjPropagator, Projectors};
use crate::noci::NOCIScalar;

/// Print the largest gaps between adjacent positive overlap eigenvalues.
/// # Arguments
/// - `lambda`: Overlap eigenvalues in ascending order.
pub(super) fn print_overlap_spectrum_gaps(
    lambda: &Array1<f64>,
) {
    // Enumerate all positive eigenvalues.
    let positive = lambda
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| if x > 0.0 { Some((i, x)) } else { None })
        .collect::<Vec<_>>();

    // Iterate over every adjacent pair of eigenvalues.
    let mut gaps = positive
        .windows(2)
        .map(|pair| {
            let (i, lambda_i) = pair[0];
            let (j, lambda_j) = pair[1];
            let ratio = lambda_j / lambda_i;
            let decades = ratio.log10();

            (decades, ratio, i, j, lambda_i, lambda_j)
        })
        .collect::<Vec<_>>();

    // Sort gaps from largest to smallest.
    gaps.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("Largest positive overlap-spectrum gaps:");
    for &(decades, ratio, i, j, lambda_i, lambda_j) in gaps.iter().take(20) {
        println!(
            "    {:>6} -> {:>6}: 1 lambda: {:.16e}, 2 lambda: {:.16e}, Ratio: {:.6e}, Gap: {:.6}",
            i, j, lambda_i, lambda_j, ratio, decades,
        );
    }
}

/// Print projector cutoff and overlap-spectrum partition diagnostics.
/// # Arguments
/// - `eps`: Relative overlap eigenvalue cutoff.
/// - `lambda`: Overlap eigenvalues in ascending order.
/// - `scale`: Largest absolute overlap eigenvalue, bounded below by one.
/// - `nulltol`: Absolute overlap eigenvalue cutoff.
/// - `negativetol`: Negative overlap eigenvalue tolerance.
/// - `relevant`: Indices of retained overlap eigenvalues.
/// - `null`: Indices of discarded overlap eigenvalues.
pub(super) fn print_projector_spectrum_diagnostics(
    eps: f64,
    lambda: &Array1<f64>,
    scale: f64,
    nulltol: f64,
    negativetol: f64,
    relevant: &[usize],
    null: &[usize],
) {
    println!(
        "Projectors: eps: {:.3e}, dim(S): {}, relevant: {}, null: {}",
        eps,
        lambda.len(),
        relevant.len(),
        null.len()
    );

    println!(
        "Overlap spectrum: Max lambda: {:.16e}, relative eps: {:.16e}, actual cutoff: {:.16e}, negative tolerance: {:.16e}.",
        scale, eps, nulltol, negativetol,
    );
    println!("Largest discarded overlap eigenvalues:");
    for &i in null.iter().rev().take(10) {
        println!("    {:>6}: lambda: {:.16e}", i, lambda[i],);
    }
    println!("Smallest retained overlap eigenvalues:");
    for &i in relevant.iter().take(10) {
        println!("    {:>6}: lambda: {:.16e}", i, lambda[i],);
    }
}

/// Print retained-subspace eigensystem diagnostics.
/// # Arguments
/// - `retained_e`: Retained-subspace eigenvalues.
/// - `retained_c`: Retained-subspace eigenvectors.
pub(super) fn print_retained_subspace_diagnostics<T: NOCIScalar>(
    retained_e: &Array1<f64>,
    retained_c: &Array2<T>,
) {
    let nretained_print = retained_e.len().min(3);
    println!(
        "Retained-subspace eigenvalues: {}",
        retained_e.slice(s![..nretained_print]).to_owned()
    );
    println!(
        "Orthogonalised retained ground-state wavefunction: {}",
        retained_c.slice(s![.., 0]).to_owned()
    );
}

/// Print projected matrix-element norms used in null-space diagnostics.
/// # Arguments
/// - `sun`: S acting on null-space eigenvectors.
/// - `hun`: H acting on null-space eigenvectors.
/// - `snn`: S projected into the null subspace.
/// - `hnn`: H projected into the null subspace.
/// - `hrn`: H coupling from null to relevant subspace.
pub(super) fn print_projected_matrix_norms<T: NOCIScalar>(
    sun: &Array2<T>,
    hun: &Array2<T>,
    snn: &Array2<T>,
    hnn: &Array2<T>,
    hrn: &Array2<T>,
) {
    println!(
        "||SUn||: {:.16e}, ||HUn||: {:.16e}, ||Un^† S Un||: {:.16e}, ||Un^† H Un||: {:.16e}, ||Ur^† H Un||: {:.16e}.",
        sun.norm(),
        hun.norm(),
        snn.norm(),
        hnn.norm(),
        hrn.norm(),
    );
}

/// Print diagnostics for the initial null-space component.
/// # Arguments
/// - `sc0n`: S acting on the initial null-space component.
/// - `hc0n`: H acting on the initial null-space component.
/// - `cn_norm`: Norm of the initial null-space component.
pub(super) fn print_initial_null_diagnostics<T: NOCIScalar>(
    sc0n: &Array1<T>,
    hc0n: &Array1<T>,
    cn_norm: f64,
) {
    println!(
        "Action of S and H on initial null vector: ||Scn|| = {}, ||Hcn|| = {}.",
        sc0n.norm(),
        hc0n.norm()
    );
    println!(
        "Initial null component: ||cn||: {:.16e}, ||Scn||/||cn||: {:.16e}, ||Hcn||/||cn||: {:.16e}.",
        cn_norm,
        sc0n.norm() / cn_norm,
        hc0n.norm() / cn_norm,
    );
}

/// Print diagnostics for the projected propagator blocks.
/// # Arguments
/// - `propagator`: Propagator blocks in relevant and null subspace bases.
/// - `es`: Initial value of the non-overlap shift.
/// - `es_s`: Initial value of the overlap-transformed shift.
/// - `doverlap`: Whether direct-overlap propagation is active.
pub(super) fn print_projected_propagator_diagnostics<T: NOCIScalar>(
    propagator: &ProjPropagator<T>,
    es: f64,
    es_s: f64,
    doverlap: bool,
) {
    println!(
        "With initial shifts E_s: {}, E_s^S: {}, ||Unn||: {}, ||Urr||: {}, ||Urn||: {}, ||Unr||: {}.",
        es,
        es_s,
        propagator.unn.norm(),
        propagator.urr.norm(),
        propagator.urn.norm(),
        propagator.unr.norm()
    );

    let nnull = propagator.unn.nrows();
    if nnull == 0 {
        println!("Null-space dimension is 0.");
    } else if doverlap {
        let identity_n = Array2::<T>::eye(nnull);
        println!(
            "Direct-overlap null-space diagnostics: ||Unn - I|| = {}, ||Unr|| = {}, ||Urn|| = {}.",
            (&propagator.unn - &identity_n).norm(),
            propagator.unr.norm(),
            propagator.urn.norm()
        );
    } else {
        let (evals_unn, _) = propagator.unn.eigh(UPLO::Lower).unwrap();
        println!("Null-space propagator Unn eigenvalues: {}", evals_unn);
    }
}

/// Print the deterministic propagation table header.
/// # Arguments
/// - `doverlap`: Whether direct-overlap propagation is active.
pub(super) fn print_propagation_table_header(
    doverlap: bool,
) {
    let (
        identity_shift_label,
        overlap_shift_label,
        population_label,
        overlap_population_label,
        metric_label,
    ) = if doverlap {
        ("Identity shift", "Shift (EsS)", "||N||", "||SN||", "N^†SN")
    } else {
        ("Shift (Es)", "Shift (EsS)", "||C||", "||SC||", "C^†SC")
    };

    println!("{}", "=".repeat(100));
    println!(
        "{:<6} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
        "iter",
        "E",
        "|dE|",
        identity_shift_label,
        overlap_shift_label,
        population_label,
        overlap_population_label,
        metric_label,
    );
}

/// Print one row of the deterministic propagation table.
/// # Arguments
/// - `iter`: Iteration number.
/// - `energy`: Projected energy and absolute energy change.
/// - `shifts`: Current non-overlap and overlap-transformed shifts.
/// - `populations`: Current coefficient and overlap-weighted populations.
/// - `den`: Current overlap metric.
pub(super) fn print_propagation_table_row(
    iter: usize,
    energy: (f64, f64),
    shifts: (f64, f64),
    populations: (f64, f64),
    den: f64,
) {
    let (e, de) = energy;
    let (es, es_s) = shifts;
    let (pop_c, pop_sc) = populations;

    println!(
        "{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
        iter, e, de, es, es_s, pop_c, pop_sc, den
    );
}

/// Format a scalar coefficient with an explicit sign.
/// # Arguments
/// - `z`: Scalar coefficient.
/// # Returns
/// - `String`: Signed scalar coefficient string.
fn format_signed_scalar<T: NOCIScalar>(z: T) -> String {
    if z.im().abs() <= 1e-14 {
        format!("{:+.10e}", z.re())
    } else {
        format!("{:+.10e}{:+.10e}i", z.re(), z.im())
    }
}

/// Format a determinant label for aligned diagnostic output.
/// # Arguments:
/// - `label`: Full determinant label.
/// - `width`: Maximum displayed label width.
/// # Returns
/// - `String`: Label padded or shortened with an ellipsis.
fn format_determinant_label(
    label: &str,
    width: usize,
) -> String {
    let nchars = label.chars().count();
    if nchars <= width {
        format!("{:<width$}", label, width = width)
    } else {
        let prefix = label.chars().take(width - 3).collect::<String>();
        format!("{}...", prefix)
    }
}

/// Print the dominant canonical components of the retained ground-state wavefunction and
/// their expansions in the original non-orthogonal determinant basis.
/// # Arguments:
/// - `ground_state`: Ground-state coefficients in the canonical retained basis.
/// - `p`: Retained overlap eigenvectors and eigenvalues.
/// - `basis`: Original NOCI-QMC determinant basis in the ordering used for H and S.
/// - `nstates`: Maximum number of canonical states to print.
/// - `nterms`: Maximum number of original-basis terms to print for each canonical state.
pub(super) fn print_canonical_wavefunction<T: NOCIScalar>(
    ground_state: &Array1<T>,
    p: &Projectors<T>,
    basis: &[DetState<T>],
    nstates: usize,
    nterms: usize,
) {
    if basis.len() != p.ur.nrows() {
        println!(
            "Cannot print canonical-state labels: basis dimension {} does not match overlap dimension {}.",
            basis.len(),
            p.ur.nrows(),
        );
        return;
    }

    let mut canonical: Vec<usize> = (0..ground_state.len()).collect();
    canonical.sort_by(|&i, &j| {
        ground_state[j]
            .abs()
            .powi(2)
            .partial_cmp(&ground_state[i].abs().powi(2))
            .unwrap()
    });

    let nstates_print = nstates.min(ground_state.len());
    let nterms_print = nterms.min(p.ur.nrows());
    let norm = ground_state.iter().map(|z| z.abs().powi(2)).sum::<f64>();
    let displayed_weight = canonical
        .iter()
        .take(nstates_print)
        .map(|&i| ground_state[i].abs().powi(2))
        .sum::<f64>();

    println!("{}", "=".repeat(100));
    println!("Dominant canonical components of retained ground state");
    println!(
        "Showing {} canonical states and up to {} original-basis terms per state.",
        nstates_print, nterms_print
    );
    println!(
        "Original-basis coefficients are coefficients in a non-orthogonal basis, not probabilities."
    );
    println!("Canonical ground-state norm: {:.16e}", norm);
    println!("Displayed canonical weight: {:.16e}", displayed_weight);
    println!();

    for &i in canonical.iter().take(nstates_print) {
        let lambda_i = p.lambda_r[i];
        let vi = ground_state[i];
        let weight = vi.abs().powi(2);
        println!("Canonical state {}", i);
        println!("  overlap eigenvalue:        {:>22.16e}", lambda_i);
        println!(
            "  ground-state coefficient:  {:>22}",
            format_signed_scalar(vi)
        );
        println!("  canonical weight:          {:>22.16e}", weight);
        println!(
            "  canonical scale:           {:>22.16e}",
            1.0 / lambda_i.sqrt()
        );
        println!();
        println!(
            "  {:>4} {:>6} {:>7}  {:<68} {:>24} {:>27}",
            "Rank",
            "Basis",
            "Parent",
            "Original NOCI-QMC determinant",
            "Basis coefficient",
            "Ground-state contribution"
        );
        println!("  {}", "-".repeat(146));

        let mut terms: Vec<usize> = (0..p.ur.nrows()).collect();
        terms.sort_by(|&mu, &nu| {
            let ai_mu = p.ur[(mu, i)] / T::from_real(lambda_i.sqrt());
            let ai_nu = p.ur[(nu, i)] / T::from_real(lambda_i.sqrt());
            ai_nu.abs().partial_cmp(&ai_mu.abs()).unwrap()
        });

        for (rank, &mu) in terms.iter().take(nterms_print).enumerate() {
            let a_mu_i = p.ur[(mu, i)] / T::from_real(lambda_i.sqrt());
            let det = &basis[mu];
            println!(
                "  {:>4} {:>6} {:>7}  {:<68} {:>24} {:>27}",
                rank + 1,
                mu,
                det.parent,
                format_determinant_label(det.label.as_str(), 68),
                format_signed_scalar(a_mu_i),
                format_signed_scalar(vi * a_mu_i)
            );
        }
        println!();
    }
    println!("{}", "=".repeat(100));
}
