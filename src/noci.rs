// noci.rs
use ndarray::{s, Array2, Array3, Array4, Array5};
use ndarray_linalg::{SVD};
use num_complex::Complex64;

use crate::{AoData, SCFState};

use crate::maths::{einsum_ba_ab, einsum_ba_acbd_dc, general_evp_complex};

/// Storage for NOCI quantities. This struct is currently unused but may be useful 
/// in future.
pub struct NOCIData {
    // 4D tensor of size (nstates, nstates, nocc, nocc).
    // For each possible pair of SCF states \mu and \nu we calculate the overlap 
    // {}^{\mu\nu}S = (C_\mu^{occ})^{\dagger} S_{spin} C_\nu^{occ} where both the 
    // occupied MO coefficients and the AO overlap matrix S_{spin} are their spin 
    // block diagonal variants.
    pub munu_s: Array4<f64>,
    
    // For each pair (mu,nu), perform Loewdin pairing by singular value decomposition 
    // (SVD) as S^{\mu\nu} = U S_{SVD} V^{\dagger} where S_{SVD} is diagonal. We store 
    // the diagonal elements of S_{SVD} (the “pair overlaps”) in shape (nstates, nstates, nocc).
    pub s_vals: Array3<f64>,

    // Overlap matrix of size (nstates, nstates) for the SCF states that form the NOCI
    // basis. Elements can be calculated as the product singular values of S_{SVD} 
    // for a given pair of states as {}^{\mu\nu}S_{NOCI} = det({}^{\mu\nuS_{SVD}}).
    pub s_noci: Array2<Complex64>,

    // Rotated occupied MOs \tilde{C}_{\mu}^{occ} and \tilde{C}_{\nu}^{occ}, 
    // (nstates, nstates, nocc, nocc). \tilde{C}_{\mu}^{occ} and \tilde{C}_{\mu}^{occ} 
    // are rotated via multiplication with U and V of the SVD decomposition respectively 
    // such that\tilde{C}_{\mu}^{occ, \dagger} S_{spin} \tilde{C}_{\nu}^{occ} is diagonal. 
    // This allows for easier evaluation of the one and two electron Hamiltonian matrix 
    // elements using the generalised Slater-Condon rules.
    pub c_mu_tilde: Array4<Complex64>,
    pub c_nu_tilde: Array4<Complex64>,

}

/// Calculates the occupied MO overlap {}^{\mu\nu}S for all possible pairs of basis states.
/// # Arguments
///     `states`: Vec<SCFState>, vector of all the calculated SCF states.
///     `s_spin`: Array2, spin block diagonal AO overlap matrix.
///     `nocc`: usize, number of occupied orbitals.
fn calculate_munu_s(states: &[SCFState], s_spin: &Array2<f64>, nocc: usize) -> Array4<f64> {
    let mut munu_s = Array4::<f64>::zeros((states.len(), states.len(), nocc, nocc));
    for (mu, state_mu) in states.iter().enumerate() {
        let c_mu_occ = &state_mu.cs_occ;

        for (nu, state_nu) in states.iter().enumerate() {
            let c_nu_occ = &state_nu.cs_occ;
            
            // {}^{\mu\nu}S = (C_\mu^{occ})^{\dagger} S_{spin} C_\nu^{occ}
            let munu = c_mu_occ.t().dot(s_spin).dot(c_nu_occ);
            munu_s.slice_mut(s![mu, nu, .., ..]).assign(&munu)
        }
    }
    munu_s
}

/// Calculates the NOCI basis state overlap matrix S_{NOCI} via SVD 
/// S^{\mu\nu} = U S_{SVD} V^{\dagger} of the per state MO overlaps, and forms a 
/// distinct pair of rotated MOs \tilde{C}_{\mu}^{occ} and \tilde{C}_{\mu}^{occ} for 
/// each pair of basis states.
/// # Arguments 
///      `states`: Vec<SCFState>, vector of all the calculated SCF states.
///      `munu_s`: Array4,  occupied orbital overlap tensor between all possible SCF pairs.
fn calculate_s_noci(states: &[SCFState], munu_s: &Array4<f64>) -> 
    (Array2<Complex64>, Array3<f64>, Array4<Complex64>, Array4<Complex64>) {

    let nstates = states.len();
    let nso = states[0].cs_occ.nrows();

    // Assuming all SCF states that form NOCI basis have same number of occupancies.
    let nocc = states[0].cs_occ.ncols();

    let mut s_noci = Array2::<Complex64>::zeros((nstates, nstates));
    let mut s_vals = Array3::<f64>::zeros((nstates, nstates, nocc));
    let mut c_mu_tilde = Array4::<Complex64>::zeros((nstates, nstates, nso, nocc));
    let mut c_nu_tilde = Array4::<Complex64>::zeros((nstates, nstates, nso, nocc));

    for mu in 0..nstates {
        let c_mu_occ = &states[mu].cs_occ;
        for nu in 0..nstates {
            let c_nu_occ = &states[nu].cs_occ;

            // Get relevant matrix from the 4D {}^{\munu}S tensor.
            let m = munu_s.slice(s![mu, nu, .., ..]).to_owned();

            // Perform SVD: m = U \tilde{S} V^{\dagger}.
            // true, true flags compute both U and V^{\dagger}.
            let (u, s_tilde, v_dag) = m.svd(true, true).unwrap();
            let u = u.unwrap();
            let v = v_dag.unwrap().t().to_owned();

            // Build diagonal of singular values.
            for p in 0..nocc {
                s_vals[(mu, nu, p)] = s_tilde[p];
            }

            // Rotate occupied MOs.
            let c_mu_t = c_mu_occ.dot(&u);
            let c_nu_t = c_nu_occ.dot(&v);
            
            // Compute S_{NOCI} matrix elements.
            let prod: f64 = s_tilde.iter().copied().product();
            s_noci[(mu, nu)] = Complex64::new(prod, 0.0);
            
            // Write the rotated MOs.
            c_mu_tilde.slice_mut(s![mu, nu, .., ..])
                .assign(&c_mu_t.map(|&x| Complex64::new(x, 0.0)));
            c_nu_tilde.slice_mut(s![mu, nu, .., ..])
                .assign(&c_nu_t.map(|&x| Complex64::new(x, 0.0)));

        }
    }
    (s_noci, s_vals, c_mu_tilde, c_nu_tilde)
}

/// Calculate the reduced NOCI overlap matrix s_red as the product of all 
/// non-zero (up to a tolerance) singular values of the SVD decomposed s_tilde.
/// # Arguments
///     `s_vals`: Array3, singular values of s_tilde for all pairs of SCF states. 
///     `tol`: f64, tolerance up to which a number is considered zero. 
fn calculate_s_red(s_vals: &Array3<f64>, tol: f64) -> Array2<f64> {
    let (nstates, _, _) = s_vals.dim();
    let mut s_red = Array2::<f64>::zeros((nstates, nstates));

    for mu in 0..nstates {
        for nu in 0..nstates {
            let sv = s_vals.slice(s![mu, nu, ..]);
            let mut prod = 1.0f64;
            let mut count = 0usize;

            for &v in sv.iter() {
                if v > tol {
                    prod *= v;
                    count += 1;
                }
            }
            
            let val = if count > 0 {prod} else {0.0};
            s_red[(mu, nu)] = val;
        }
    }
    s_red
}

/// Form {}^{\mu\nu}P_i the co-density matrix as an outer product between occupied 
/// coulms i of \tilde{C}_{\mu}^{occ} and \tilde{C}_{\mu}^{occ}, and the reduced 
/// co-density matrix W as {}^{\mu\nu}W = \sum_{i} 1 / s_i * {}^{\mu\nu}P_i, where 
/// s_i are the singular values of the SVD decomposed s_tilde. The full P tensor 
/// returned from this function is unreasonably large, this should be re-thought.
/// # Arguments 
///     `c_mu_tilde`: Array4, U rotated MO coefficients for each SCF state pair. 
///     `c_nu_tilde`: Array4, V rotated MO coefficients for each SCF state pair. 
///     `s_vals`: Array3, singular values of s_tilde for all pairs of SCF states.
///     `tol`: f64, tolerance up to which a number is considered zero. 
fn calculate_codensity(c_mu_tilde: &Array4<Complex64>, c_nu_tilde: &Array4<Complex64>,
                       s_vals: &Array3<f64>, tol: f64) 
                       -> (Array5<Complex64>, Array4<Complex64>) {

    let (nstates, _, nso, nocc) = c_mu_tilde.dim();
    let mut p = Array5::<Complex64>::zeros((nstates, nstates, nocc, nso, nso));
    let mut w = Array4::<Complex64>::zeros((nstates, nstates, nso, nso));

    for mu in 0..nstates {
        for nu in 0..nstates {
            // Accumulate W over occupied orbitals i.
            let mut munu_w = Array2::<Complex64>::zeros((nso, nso));

            for i in 0..nocc {
                let a = c_mu_tilde.slice(s![mu, nu, .., i]);
                let b = c_nu_tilde.slice(s![mu, nu, .., i]).map(|z| z.conj());

                // Calculate outer product. 
                let mut p_i = Array2::<Complex64>::zeros((nso, nso));
                for x in 0..nso {
                    for y in 0..nso {
                        p_i[(x, y)] = a[x] * b[y];
                    }
                }

                p.slice_mut(s![mu, nu, i, .., ..]).assign(&p_i);

                // Accumulate into W if s_val > tol.
                let s_val = s_vals[(mu, nu, i)];
                if s_val > tol {
                    let weight = Complex64::new(1.0 / s_val, 0.0);
                    for x in 0..nso {
                        for y in 0..nso {
                            munu_w[(x, y)] += weight * p_i[(x, y)];
                        }
                    }
                }
            }
            w.slice_mut(s![mu, nu, .., ..]).assign(&munu_w);
        }
    }
    

    (p, w)
}

/// Calculate one electron and nuclear Hamiltonian matrix elements using the generalised 
/// Slater-Condon rules. 
/// # Arguments 
///     `h_spin`: Array2, spin block diagonal core Hamiltonian matrix.
///     `enuc`: Scalar, nuclear repulsion energy. 
///     `s_vals`: Array3, Singular values of the SVD decomposed s_tilde for each SCF state pair.
///     `s_red`: Array2, Product of the non-zero values of s_vals for each SCF state pair. 
///     `w`: Array4, weighted co-density tensor. 
///     `p`: Array5, un-weighted co-density tensor.
///     `tol`: Float, value below which a number is considered as zero.
fn one_electron_h(h_spin: &Array2<f64>, enuc: f64, s_vals: &Array3<f64>, s_red: &Array2<f64>,
                  w: &Array4<Complex64>, p: &Array5<Complex64>, tol: f64) 
                  -> (Array2<Complex64>, Array2<Complex64>) {
                      
    let (nstates, _, nocc) = s_vals.dim();

    let mut h1 = Array2::<Complex64>::zeros((nstates, nstates));
    let mut h_nuc = Array2::<Complex64>::zeros((nstates, nstates));

    for mu in 0..nstates {
        for nu in 0..nstates {
            // Find indices i for which singular values are zero.
            let mut zeros: Vec<usize> = Vec::new();
            for i in 0..nocc {
                if s_vals[(mu, nu, i)] <= tol {
                    zeros.push(i);
                }
            }

            let sred = Complex64::new(s_red[(mu, nu)], 0.0);
            match zeros.len() {
                // With no zeros (s_i != 0 for all i) we use munu_w.
                0 => {
                    let munu_w = w.slice(s![mu, nu, .., ..]).to_owned();
                    let val = einsum_ba_ab(&munu_w, h_spin);
                    h1[(mu, nu)] = sred * val;
                    h_nuc[(mu, nu)] = sred * Complex64::new(enuc, 0.0);
                }
                // With 1 zero (s_i = 0 for 1 i) we use P_i.
                1 => {
                    let i = zeros[0];
                    let p_i = p.slice(s![mu, nu, i, .., ..]).to_owned();
                    let val = einsum_ba_ab(&p_i, h_spin);
                    h1[(mu, nu)] = sred * val;
                    h_nuc[(mu, nu)] = Complex64::new(0.0, 0.0);
                // Otherwise the matrix element is zero.
                }
                _ => {
                    h1[(mu, nu)] = Complex64::new(0.0, 0.0);
                    h_nuc[(mu, nu)] = Complex64::new(0.0, 0.0);
                }
            }

        }
    }

    (h1, h_nuc)
}

/// Calculate two electron Hamiltonian matrix elements using the generalised 
/// Slater-Condon rules. 
/// # Arguments 
///     `s_vals`: Array3, Singular values of the SVD decomposed s_tilde for each SCF state pair.
///     `s_red`: Array2, Product of the non-zero values of s_vals for each SCF state pair. 
///     `w`: Array4, weighted co-density tensor. 
///     `p`: Array5, un-weighted co-density tensor.
///     `eri_spin`: Array4, antisymmetrised ERIs in spin diagonal block.
///     `tol`: Float, value below which a number is considered as zero.
fn two_electron_h(s_vals: &Array3<f64>, s_red: &Array2<f64>, w: &Array4<Complex64>,
                  p: &Array5<Complex64>, eri_spin: &Array4<f64>, tol: f64) 
                  -> Array2<Complex64> {

    let (nstates, _, nocc) = s_vals.dim();

    let mut h2 = Array2::<Complex64>::zeros((nstates, nstates));

    for mu in 0..nstates {
        for nu in 0..nstates {
            // Find indices i for which singular values are zero.
            let mut zeros: Vec<usize> = Vec::new();
            for i in 0..nocc {
                if s_vals[(mu, nu, i)] <= tol {
                    zeros.push(i);
                }
            }

            let sred = Complex64::new(s_red[(mu, nu)], 0.0);
            match zeros.len() {
                // With no zeros (s_i != 0 for all i) we use munu_w on both sides.
                0 => {
                    let munu_w = w.slice(s![mu, nu, .., ..]).to_owned();
                    let val = Complex64::new(0.5, 0.0) 
                            * einsum_ba_acbd_dc(&munu_w, eri_spin, &munu_w);
                    h2[(mu, nu)] = val * sred; 
                }
                // With 1 zero (s_i = 0 for one index i) we use P_i on one side.
                1 => {
                    let i = zeros[0];
                    let p_i = p.slice(s![mu, nu, i, .., ..]).to_owned();
                    let munu_w = w.slice(s![mu, nu, .., ..]).to_owned();
                    let val = einsum_ba_acbd_dc(&p_i, eri_spin, &munu_w);
                    h2[(mu, nu)] = sred * val;
                // with 2 zeros (s_i, s_j = 0 for two indices i, j) we use P_i on 
                // one side and P_j on the other.
                }
                2 => {
                    let i = zeros[0];
                    let j = zeros[1];
                    let p_i = p.slice(s![mu, nu, i, .., ..]).to_owned();
                    let p_j = p.slice(s![mu, nu, j, .., ..]).to_owned();
                    let val = einsum_ba_acbd_dc(&p_i, eri_spin, &p_j);
                    h2[(mu, nu)] = sred * val;
                }
                // Otherwise the matrix element is zero.
                _ => {h2[(mu, nu)] = Complex64::new(0.0, 0.0)}
            }
        }
    }

    h2
}

/// Using MO coefficients obtained from the SCF cycle of each basis state perform 
/// Non-orthogonal Configuration Interaction (NOCI) to obtain an improved energy 
/// estimate.
/// # Arguments:
///     `scfstates`: Vec<SCFState>, vector of all the calculated SCF states. 
///     `ao`: AoData struct, contains AO integrals and other system data. 
pub fn calculate_noci_energy(ao: &AoData, scfstates: &[SCFState]) -> f64 {
    // Tolerance for a number being non-zero.
    let tol = 1e-8; 
    let nocc = (ao.nelec[0] + ao.nelec[1]) as usize;
    // Calculate {}^{\mu\nu}S and assign to NOCI data struct.
    let munu_s = calculate_munu_s(scfstates, &ao.s_spin, nocc);

    // Calculate the NOCI overlap matrix S_{NOCI} and form rotated MOs.
    // There is a distinct set of rotated MOs for every pair of SCF basis states.
    let (s_noci, s_vals, c_mu_tilde, c_nu_tilde) = calculate_s_noci(scfstates, &munu_s);

    // Calculate the reduced NOCI overlap matrix S_{red}.
    let s_red = calculate_s_red(&s_vals, tol);
    
    // Form the standard and reduced co-density matrices.
    let (p, w) = calculate_codensity(&c_mu_tilde, &c_nu_tilde, &s_vals, tol);

    // Calculate one and two electron NOCI Hamiltonian matrix elements.
    let (h1, h_nuc) = one_electron_h(&ao.h_spin, ao.enuc, &s_vals, &s_red, &w, &p, tol);
    let h2 = two_electron_h(&s_vals, &s_red, &w, &p, &ao.eri_spin, tol);
    
    // Enforce hermiticity of the Hamiltonian and S_{NOCI} matrices.
    let mut s = s_noci.clone();
    let s_dag = s.t().map(|z| z.conj());
    s = (&s + &s_dag) * Complex64::new(0.5, 0.0);
    let mut h = &h_nuc + &h1 + &h2;
    let h_dag = h.t().map(|z| z.conj());
    h = (&h + &h_dag) * Complex64::new(0.5, 0.0);

    // Use Loewdin orthgonalisation with projection to non-zero subspace of  
    // S to solve GEVP, thus obtaining ground-state energy.
    let (evals, _c) = general_evp_complex(&h, &s, true, tol);
    
    evals[0]
}
