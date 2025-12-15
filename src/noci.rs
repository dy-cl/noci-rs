// noci.rs
use ndarray::{Array1, Array2, Array4};
use ndarray_linalg::{SVD, Determinant};
use num_complex::Complex64;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;

use crate::{AoData, SCFState};
use crate::utils::print_array2;

use crate::maths::{einsum_ba_ab, einsum_ba_acbd_dc, general_evp_complex};

// Storage for per SCF basis set pair.
pub struct Pair {
    pub munu_s_noci: Complex64,
    pub s_tilde: Array1<f64>,
    pub s_red: Complex64,
    pub zeros: Vec<usize>,
    pub c_mu_tc: Array2<Complex64>,
    pub c_nu_tc: Array2<Complex64>,
    pub phase: f64,
}

/// Calculate the occupied MO overlap {}^{\mu\nu}S for SCF states \mu and \nu as:
/// {}^{\mu\nu}S = C_{\mu}^{occ, \dagger} S C_{\nu}^{occ}.
/// # Arguments
///     `states`: Vec<SCFState>, vector of all the calculated SCF states.
///     `s_spin`: Array2, spin block diagonal AO overlap matrix.
///     `mu`: usize, index of state mu.
///     `nu`: usize, index of state nu.
fn calculate_munu_s(states: &[SCFState], s_spin: &Array2<f64>, mu: usize, nu: usize) -> Array2<f64> {
    let c_mu_occ = &states[mu].cs_occ; 
    let c_nu_occ = &states[nu].cs_occ;
    // {}^{\mu\nu}S = C_{\mu}^{occ, \dagger} S C_{\nu}^{occ}
    c_mu_occ.t().dot(&s_spin.dot(c_nu_occ))
}

/// Calculate overlap between SCF NOCI reference basis states for a pair \mu, \nu. We perform SVD
/// on {}^{\mu\nu}S states as: {}^{\mu\nu}S = U {}^{\mu\nu}S_{SVD} V^{\dagger}, form the
/// corresponding rotated MOs \tilde{C}_{\mu}^{occ} and \tilde{C}_{\mu}^{occ} and calculate
/// {}^{\mu\nu}S_{NOCI} as the product of diagonal (singular) values in {}^{\mu\nu}S_{SVD}.
/// # Arguments 
///      `states`: Vec<SCFState>, vector of all the calculated SCF states.
///      `munu_s`: Array2,  occupied orbital overlap matrix between SCF states \mu and \nu.
fn calculate_munu_s_noci(states: &[SCFState], munu_s: &Array2<f64>, mu: usize, nu: usize) -> 
    (Complex64, Array1<f64>, Array2<Complex64>, Array2<Complex64>, f64) {

    let c_mu_occ = &states[mu].cs_occ;
    let c_nu_occ = &states[nu].cs_occ;

    let (u, s_tilde, v_dag) = munu_s.svd(true, true).unwrap();
    let u = u.unwrap();
    let v = v_dag.unwrap().t().to_owned();

    // Rotate occupied MOs and store as complex.
    let c_mu_t = c_mu_occ.dot(&u);
    let c_nu_t = c_nu_occ.dot(&v);
    let c_mu_tc = c_mu_t.map(|&x| Complex64::new(x, 0.0));
    let c_nu_tc = c_nu_t.map(|&x| Complex64::new(x, 0.0));

    // Calculate phase associated with this basis pair.
    let det_u = u.det().unwrap();
    let det_v = v.det().unwrap();
    let phase = det_u * det_v;

    // Compute {}^{\mu\nu}S_{NOCI} matrix elements.
    let prod: f64 = s_tilde.iter().copied().product();
    let munu_s_noci = Complex64::new(phase * prod, 0.0);

    (munu_s_noci, s_tilde, c_mu_tc, c_nu_tc, phase)
}
    
/// Calculate the reduced NOCI overlap matrix element {}^{\mu\nu}s_red for a pair of states \mu \nu as the product of all 
/// non-zero (up to a tolerance) singular values of the SVD decomposed {}^{\mu\nu}s_tilde.
/// # Arguments
///     `s_vals`: Array1, singular values of s_tilde for all pair \mu \nu of SCF states. 
///     `tol`: f64, tolerance up to which a number is considered zero. 
fn calculate_s_red(s_vals: &Array1<f64>, tol: f64) -> (Complex64, Vec<usize>) {
    let mut prod = 1.0f64;
    let mut zeros = Vec::new();
    let mut count = 0usize;

    for (i, &v) in s_vals.iter().enumerate() {
        if v > tol {
            prod *= v;
            count += 1;
        } else {
            zeros.push(i);
        }
    }

    let s_red = Complex64::new(if count > 0 {prod} else {0.0}, 0.0);
    
    (s_red, zeros)
}

/// Calculate {}^{\mu\nu}P_i = {}^{\mu}c_i^a {}^{\nu}c_i^b* co-density matrix where {}^{\mu}c_i^a, 
/// {}^{\nu}c_i^b are the rotated MO coefficients of state mu and nu respectively.
/// # Arguments 
///     `c_mu_tilde`: Array2, U rotated MO coefficients for a given pair of states. 
///     `c_nu_tilde`: Array2, V rotated MO coefficients for a given pair of states. 
///     `i`: usize, MO index.
fn calculate_codensity_p_pair(c_mu_tilde: &Array2<Complex64>,c_nu_tilde: &Array2<Complex64>,
                              i: usize,) -> Array2<Complex64> {
    let nso = c_mu_tilde.nrows();
    let mut munu_p_i = Array2::<Complex64>::zeros((nso, nso));
    for x in 0..nso {
        for y in 0..nso {
            munu_p_i[(x, y)] = c_mu_tilde[(x, i)] * c_nu_tilde[(y, i)].conj();
        }
    }
    munu_p_i
}

/// Calculate {}^{\mu\nu}W = \sum_{i} 1 / s_i * {}^{\mu}c_i^a {}^{\nu}c_i^b* weighted co-density 
/// matrix where s_i are the singular values of the SVD decomposed s_tilde, and {}^{\mu}c_i^a, 
/// {}^{\nu}c_i^b are the rotated MO coefficients of state mu and nu respectively.
/// # Arguments 
///     `c_mu_tilde`: Array2, U rotated MO coefficients for a given pair of states. 
///     `c_nu_tilde`: Array2, V rotated MO coefficients for a given pair of states. 
///     `s_vals`: Array1, singular values of s_tilde for a given pair of states.
///     `tol`: f64, tolerance up to which a number is considered zero. 
fn calculate_codensity_w_pair(c_mu_tilde: &Array2<Complex64>,c_nu_tilde: &Array2<Complex64>,
                              s_vals: &Array1<f64>, tol: f64,) -> Array2<Complex64> {
    let (nso, nocc) = c_mu_tilde.dim();
    let mut munu_w = Array2::<Complex64>::zeros((nso, nso));

    for i in 0..nocc {
        if s_vals[i] > tol {
            let weight = Complex64::new(1.0 / s_vals[i], 0.0);
            for x in 0..nso {
                for y in 0..nso {
                    munu_w[(x, y)] += weight * c_mu_tilde[(x, i)] * c_nu_tilde[(y, i)].conj();
                }
            }
        }
    }
    munu_w
}

/// Calculate one electron and nuclear Hamiltonian matrix elements using the generalised 
/// Slater-Condon rules for a pair of Slater determinants \mu and \nu. 
/// # Arguments
///     `pair`: Pair struct, contains the following data concerning a pair of SCF states:
///         `enuc`: Scalar, nuclear repulsion energy. 
///         `s_vals`: Array3, Singular values of the SVD decomposed s_tilde for each SCF state pair.
///         `s_red`: Array2, Product of the non-zero values of s_vals for each SCF state pair.
///         `c_mu_tc`: Array2, Rotated MO coefficient matrix of determinant \mu.
///         `c_nu_tc`: Array2, Rotated MO coefficient matrix of determinant \nu.
///         `phase`: f64, Phase associated with determinant pair \mu \nu.
///         `zeros`: [usize], Array containing orbital indices whose singular values are zero for a
///          given pair \mu \nu.
///     `tol`: Float, value below which a number is considered as zero.
///     `ao`: AoData struct, contains AO integrals and other system data. 
fn one_electron_h(ao: &AoData, pair: &Pair, tol: f64) -> (Complex64, Complex64) {
                      
    let (munu_h1, munu_h_nuc) = match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w.
        0 => {
            let munu_w = calculate_codensity_w_pair(&pair.c_mu_tc, &pair.c_nu_tc, &pair.s_tilde, tol);
            let val = einsum_ba_ab(&munu_w, &ao.h_spin);
            let h1_val = pair.phase * pair.s_red * val;
            let h_nuc_val = pair.phase * pair.s_red * Complex64::new(ao.enuc, 0.0);
            (h1_val, h_nuc_val)
        }
        // With 1 zero (s_i = 0 for 1 i) we use P_i.
        1 => {
            let i = pair.zeros[0];
            let munu_p_i = calculate_codensity_p_pair(&pair.c_mu_tc, &pair.c_nu_tc, i);
            let val = einsum_ba_ab(&munu_p_i, &ao.h_spin);
            let h1_val = pair.phase * val;
            let h_nuc_val = Complex64::new(0.0, 0.0);
            (h1_val, h_nuc_val)
        // Otherwise the matrix element is zero.
        }
        _ => {
            let h1_val = Complex64::new(0.0, 0.0);
            let h_nuc_val = Complex64::new(0.0, 0.0);
            (h1_val, h_nuc_val)
        }
    };
    (munu_h1, munu_h_nuc)
}

/// Calculate two electron Hamiltonian matrix elements using the generalised 
/// Slater-Condon rules for a pair of Slater determinants \mu and \nu.
/// # Arguments
///     `pair`: Pair struct, contains the following data concerning a pair of SCF states:
///         `enuc`: Scalar, nuclear repulsion energy. 
///         `s_vals`: Array3, Singular values of the SVD decomposed s_tilde for each SCF state pair.
///         `s_red`: Array2, Product of the non-zero values of s_vals for each SCF state pair.
///         `c_mu_tc`: Array2, Rotated MO coefficient matrix of determinant \mu.
///         `c_nu_tc`: Array2, Rotated MO coefficient matrix of determinant \nu.
///         `phase`: f64, Phase associated with determinant pair \mu \nu.
///         `zeros`: [usize], Array containing orbital indices whose singular values are zero for a
///          given pair \mu \nu.
///     `tol`: Float, value below which a number is considered as zero.
///     `ao`: AoData struct, contains AO integrals and other system data. 
fn two_electron_h(ao: &AoData, pair: &Pair, tol: f64) -> Complex64 {

    match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w on both sides.
        0 => {
            let munu_w = calculate_codensity_w_pair(&pair.c_mu_tc, &pair.c_nu_tc, &pair.s_tilde, tol);
            let val = Complex64::new(0.5, 0.0) * einsum_ba_acbd_dc(&munu_w, &ao.eri_spin, &munu_w);
            pair.phase * val * pair.s_red
        }
        // With 1 zero (s_i = 0 for one index i) we use P_i on one side.
        1 => {
            let i = pair.zeros[0];
            let munu_p_i = calculate_codensity_p_pair(&pair.c_mu_tc, &pair.c_nu_tc, i);
            let munu_w = calculate_codensity_w_pair(&pair.c_mu_tc, &pair.c_nu_tc, &pair.s_tilde, tol);
            let val = einsum_ba_acbd_dc(&munu_p_i, &ao.eri_spin, &munu_w);
            pair.phase * val
        // with 2 zeros (s_i, s_j = 0 for two indices i, j) we use P_i on 
        // one side and P_j on the other.
        }
        2 => {
            let i = pair.zeros[0];
            let j = pair.zeros[1];
            let munu_p_i = calculate_codensity_p_pair(&pair.c_mu_tc, &pair.c_nu_tc, i);
            let munu_p_j = calculate_codensity_p_pair(&pair.c_mu_tc, &pair.c_nu_tc, j);
            let val = einsum_ba_acbd_dc(&munu_p_i, &ao.eri_spin, &munu_p_j);
            pair.phase * val
        }
        // Otherwise the matrix element is zero.
        _ => {Complex64::new(0.0, 0.0)}
    }
}

/// Using occupied MO coefficients of each Non-orthogonal Configuration Interaction (NOCI) basis 
/// state form the Hamiltonian and overlap matrices using the generalised Slater-Condon rules.
/// # Arguments:
///     `scfstates`: Vec<SCFState>, vector of all the calculated SCF states. 
///     `ao`: AoData struct, contains AO integrals and other system data. 
pub fn build_noci_matrices(ao: &AoData, scfstates: &[SCFState]) 
                            -> (Array2<Complex64>, Array2<Complex64>, Duration) {
    // Tolerance for a number being non-zero.
    let tol = 1e-6;
    let nstates = scfstates.len();
    
    let mut h = Array2::<Complex64>::zeros((nstates, nstates));
    let mut s = Array2::<Complex64>::zeros((nstates, nstates));

    // Build list of all upper-triangle pairs (\mu, \nu) which have \mu <= \nu (i.e., 
    // diagonal included). This can be done as {}^{\mu\nu}X[\mu, \nu] = ({}^{\mu\nu}X[\nu, \mu])^T
    // where X is either S or H.
    let pairs: Vec<(usize, usize)> = (0..nstates).flat_map(|mu| (mu..nstates).map(move |nu| (mu, nu))).collect();
    
    // Progress monitoring setup.
    let total_pairs = pairs.len();
    let counter = AtomicUsize::new(0);
    let report = total_pairs / 100;
    
    // Calculate Hamiltonian matrix elements in parallel.
    let t_h = Instant::now();
    let tmp: Vec<(usize, usize, Complex64, Complex64)> = pairs.par_iter().map(|&(mu, nu)| {
        // Progress counter.
        let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
        if done.is_multiple_of(report) || done == total_pairs {
            let frac = 100.0 * (done as f64) / (total_pairs as f64);
            println!("H and S matrix elements: {} / {}, ({:.1}%)", done, total_pairs, frac);
        }
        // Calculate matrix elements for this pair.
        let munu_s = calculate_munu_s(scfstates, &ao.s_spin, mu, nu);
        let (munu_s_noci, s_tilde, c_mu_tc, c_nu_tc, phase) = calculate_munu_s_noci(scfstates, &munu_s, mu, nu);
        let (s_red, zeros) = calculate_s_red(&s_tilde, tol);
        let pair = Pair {munu_s_noci, s_tilde, s_red, zeros, c_mu_tc, c_nu_tc, phase};
        let (munu_h1, munu_h_nuc) = one_electron_h(ao, &pair, tol);
        let munu_h2 = two_electron_h(ao, &pair, tol);
        let munu_h = munu_h1 + munu_h2 + munu_h_nuc;
        (mu, nu, munu_h, munu_s_noci)
    }).collect();
    let d_h = t_h.elapsed();

    // Scatter Hamiltonian and overlap matrix elements into full matrices. 
    for (mu, nu, h_munu, s_munu) in tmp {
        h[(mu, nu)] = h_munu;
        s[(mu, nu)] = s_munu;
        // Hermitian.
        if mu != nu {
            h[(nu, mu)] = h_munu.conj();
            s[(nu, mu)] = s_munu.conj();
        }
    }
    (h, s, d_h)
}

/// Calculate NOCI energy by solving GEVP with NOCI Hamiltonian and overlap.
/// # Arguments:
///     `scfstates`: Vec<SCFState>, vector of all the calculated SCF states. 
///     `ao`: AoData struct, contains AO integrals and other system data. 
pub fn calculate_noci_energy(ao: &AoData, scfstates: &[SCFState]) -> (f64, Array1<Complex64>, Duration) {
    let tol = 1e-8;
    let (h, s, d_h) = build_noci_matrices(ao, scfstates);
        
    println!("NOCI-reference Hamiltonian:");
    print_array2(&h.map(|z: &Complex64| z.re));
    println!("NOCI-reference Overlap:");
    print_array2(&s.map(|z: &Complex64| z.re));
    println!("Shifted NOCI-reference Hamiltonian");
    let h_shift = &h.map(|z: &Complex64| z.re) - scfstates[0].e * &s.map(|z: &Complex64| z.re);
    print_array2(&h_shift);
    let (evals, c) = general_evp_complex(&h, &s, true, tol);
    println!("GEVP eigenvalues in NOCI-reference basis: {}", evals);
    println!("{}", "=".repeat(100));

    // Assumes columns of c are energy ordered eigenvectors
    let c0 = c.column(0).to_owned(); 
    (evals[0], c0, d_h)
}


