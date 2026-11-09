// noci/optimised/ci.rs
//! CI step of ResHF: the NOCI expansion of a fixed set of determinants and its pair weights.

// External crate imports.
use ndarray::{Array1, Array2, s};

// Crate-root imports.
use crate::maths::general_evp;

/// NOCI energies of a fixed set of ResHF determinants, with the state-averaged pair weights
/// the resonating Fock matrices are built from.
pub(super) struct ResHFExpansion {
    /// NOCI energies `E_I`, lowest first.
    pub(super) energies: Array1<f64>,
    /// Pair weights `D_{AB} = \sum_I w_I c_{AI} c_{BI}`.
    pub(super) pair_weights: Array2<f64>,
    /// Energy-weighted pair weights `D^E_{AB} = \sum_I w_I E_I c_{AI} c_{BI}`.
    pub(super) pair_energy_weights: Array2<f64>,
}

/// Solve the NOCI problem `H c = S c E` for given matrices and form the pair weights
/// `D = C_w \mathrm{diag}(w) C_w^T` and `D^E = C_w \mathrm{diag}(w E) C_w^T`, where `C_w` holds
/// the coefficients of the weighted roots. These are all the resonating Fock matrices need from
/// the coefficients.
/// # Arguments:
/// - `h`: Hamiltonian matrix `H_{AB}`, including nuclear repulsion.
/// - `s`: Overlap matrix `S_{AB}`.
/// - `weights`: State weights `w_I`; their length sets how many roots are averaged.
/// - `tol`: Tolerance up to which a number is considered zero.
/// # Returns:
/// - `ResHFExpansion`: The energies and pair weights.
pub(super) fn reshf_solve(
    h: &Array2<f64>,
    s: &Array2<f64>,
    weights: &[f64],
    tol: f64,
) -> ResHFExpansion {
    // Solve `H c = S c E`, projecting out near-singular directions of `S`.
    let (energies, coeffs) = general_evp(h, s, true, tol);
    let nw = weights.len();
    if nw > energies.len() {
        eprintln!(
            "reshf.weights has {nw} weights but the NOCI space has only {} roots",
            energies.len()
        );
        std::process::exit(1);
    }

    // Scale the weighted roots' coefficients by `w_I` and by `w_I E_I`.
    let cw = coeffs.slice(s![.., 0..nw]);
    let w = Array1::from(weights.to_vec());
    let we = &w * &energies.slice(s![0..nw]);
    let pair_weights = (&cw * &w).dot(&cw.t());
    let pair_energy_weights = (&cw * &we).dot(&cw.t());

    ResHFExpansion {
        energies,
        pair_weights,
        pair_energy_weights,
    }
}
