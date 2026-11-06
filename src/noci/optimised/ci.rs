// noci/optimised/ci.rs
//! CI step of ResHF: the NOCI expansion of a fixed set of determinants and its pair weights.

// External crate imports.
use ndarray::{Array1, Array2};

// Crate-root imports.
use crate::determinant::{NOCIIndex, NOCISpace};
use crate::elements::calculate_hs_pair_naive_reshf;
use crate::maths::general_evp;
use crate::scf::occ_first;
use crate::{AoData, SCFState};

/// NOCI expansion of a fixed set of ResHF determinants, with the state-averaged pair weights
/// the resonating Fock matrices are built from.
pub struct ResHFExpansion {
    /// Hamiltonian matrix `H_{AB} = \langle \Phi_A | \hat H | \Phi_B \rangle`, including nuclear repulsion.
    pub h: Array2<f64>,
    /// Overlap matrix `S_{AB} = \langle \Phi_A | \Phi_B \rangle`.
    pub s: Array2<f64>,
    /// NOCI energies `E_I`, lowest first.
    pub energies: Array1<f64>,
    /// Expansion coefficients; column `I` holds `c_{AI}`.
    pub coeffs: Array2<f64>,
    /// Pair weights `D_{AB} = \sum_I w_I c_{AI} c_{BI}`.
    pub pair_weights: Array2<f64>,
    /// Energy-weighted pair weights `D^E_{AB} = \sum_I w_I E_I c_{AI} c_{BI}`.
    pub pair_energy_weights: Array2<f64>,
}

/// Build `H` and `S` for determinants with fixed orbitals using the generalised Slater-Condon
/// rules, then solve `H c = S c E` and form the pair weights with [`reshf_solve`]. No Wick's
/// intermediates are needed, whatever the input's `wicks` options.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `dets`: Determinants whose orbitals are held fixed.
/// - `weights`: State weights `w_I`; their length sets how many roots are averaged.
/// - `tol`: Tolerance up to which a number is considered zero.
/// # Returns:
/// - `ResHFExpansion`: Hamiltonian and overlap matrices, energies, coefficients and pair weights.
pub fn reshf_expansion(
    ao: &AoData,
    dets: &[SCFState],
    weights: &[f64],
    tol: f64,
) -> ResHFExpansion {
    // Put occupied orbitals first, as the reference NOCI space does.
    let dets: Vec<SCFState> = dets.iter().map(occ_first).collect();
    let space = NOCISpace::from_scf(&dets);
    let n = space.len();

    // Fill the lower triangle pair by pair and mirror it: for real determinants
    // `H_{BA} = H_{AB}` and `S_{BA} = S_{AB}`.
    let mut h = Array2::<f64>::zeros((n, n));
    let mut s = Array2::<f64>::zeros((n, n));
    for a in 0..n {
        for b in 0..=a {
            let (hab, sab) =
                calculate_hs_pair_naive_reshf(ao, &space, NOCIIndex(a), NOCIIndex(b), tol);
            h[(a, b)] = hab;
            h[(b, a)] = hab;
            s[(a, b)] = sab;
            s[(b, a)] = sab;
        }
    }

    reshf_solve(h, s, weights, tol)
}

/// Solve the NOCI problem `H c = S c E` for given matrices and form the pair weights
/// `D_{AB} = \sum_I w_I c_{AI} c_{BI}` and `D^E_{AB} = \sum_I w_I E_I c_{AI} c_{BI}`, which are
/// all the resonating Fock matrices need from the coefficients.
/// # Arguments:
/// - `h`: Hamiltonian matrix `H_{AB}`, including nuclear repulsion.
/// - `s`: Overlap matrix `S_{AB}`.
/// - `weights`: State weights `w_I`; their length sets how many roots are averaged.
/// - `tol`: Tolerance up to which a number is considered zero.
/// # Returns:
/// - `ResHFExpansion`: The matrices, energies, coefficients and pair weights.
/// # Panics
/// - Panics if more state weights are given than the projected NOCI space has roots.
pub fn reshf_solve(
    h: Array2<f64>,
    s: Array2<f64>,
    weights: &[f64],
    tol: f64,
) -> ResHFExpansion {
    let n = h.nrows();

    // Solve `H c = S c E`, projecting out near-singular directions of `S`.
    let (energies, coeffs) = general_evp(&h, &s, true, tol);
    assert!(
        weights.len() <= energies.len(),
        "{} state weights given but the NOCI space has only {} roots",
        weights.len(),
        energies.len()
    );

    // Accumulate `D_{AB} = \sum_I w_I c_{AI} c_{BI}` and `D^E_{AB} = \sum_I w_I E_I c_{AI} c_{BI}`.
    let mut pair_weights = Array2::<f64>::zeros((n, n));
    let mut pair_energy_weights = Array2::<f64>::zeros((n, n));
    for (i, &w) in weights.iter().enumerate() {
        let c = coeffs.column(i);
        for a in 0..n {
            for b in 0..n {
                let wcc = w * c[a] * c[b];
                pair_weights[(a, b)] += wcc;
                pair_energy_weights[(a, b)] += wcc * energies[i];
            }
        }
    }

    ResHFExpansion {
        h,
        s,
        energies,
        coeffs,
        pair_weights,
        pair_energy_weights,
    }
}
