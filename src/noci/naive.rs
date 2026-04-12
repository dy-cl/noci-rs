// noci/naive.rs
use ndarray::{Array1, Array2, Array4, Axis};
use ndarray_linalg::{SVD, Determinant};
use rayon::prelude::*;

use crate::{AoData, SCFState};

use crate::maths::{einsum_ba_ab_real, einsum_ba_abcd_cd_real};

// Storage of quantities required to compute matrix elements between determinant pairs in the naive fashion.
pub(crate) struct Pair {
    pub(crate) s: f64,
    pub(crate) s_red: f64,
    pub(crate) zeros: Vec<usize>,
    pub(crate) w: Option<Array2<f64>>,
    pub(crate) p_i: Option<Array2<f64>>,
    pub(crate) p_j: Option<Array2<f64>>,
    pub(crate) phase: f64,
}

/// Given an MO coefficient matrix and a corresponding occupancy vector, return the occupied only
/// coefficient matrix.
/// # Arguments:
/// - `c`: MO coefficient matrix.
/// - `occ`: Occupancy vector.
/// # Returns:
/// - `Array2<f64>`: Occupied-only MO coefficient matrix.
pub fn occ_coeffs(c: &Array2<f64>, occ: u128) -> Array2<f64> {
    let occ_idx: Vec<usize> = (0..c.ncols()).filter(|&i| ((occ >> i) & 1u128) == 1).collect();

    let nbas = c.nrows();
    let nocc = occ_idx.len();
    let mut c_occ = Array2::<f64>::zeros((nbas, nocc));

    for (k, &i) in occ_idx.iter().enumerate() {c_occ.column_mut(k).assign(&c.index_axis(Axis(1), i));}

    c_occ
}

/// Calculate the reduced occupied MO overlap scalar of {}^{\Lambda\Gamma} \tilde{S} as the product of
/// all non-zero singular values of the SVD'd {}^{\Lambda\Gamma} \tilde{S}.
/// # Arguments
/// - `tilde_s`: Vector of singular values, the diagonal of {}^{\Lambda\Gamma} \tilde{S}. 
/// - `tol`: Tolerance up to which a number is considered zero. 
/// # Returns:
/// - `(f64, Vec<usize>)`: Product of non-zero singular values and indices
///   of singular values treated as zero.
fn calculate_s_red(tilde_s: &Array1<f64>, tol: f64) -> (f64, Vec<usize>) {
    let mut prod = 1.0f64;
    let mut zeros = Vec::new();

    for (i, &v) in tilde_s.iter().enumerate() {
        if v.abs() > tol {
            prod *= v;      
        } else {
            zeros.push(i);
        }
    }
    (prod, zeros)
}

/// Build overlap-related intermediates for a pair of determinants.
/// An SVD is performed on the occupied MO overlap matrix
/// {}^{\Lambda\Gamma} S_{ij} = U {}^{\Lambda\Gamma}\tilde{S}_{ij} V^{\dagger},
/// each set of occupied MOs is rotated to form {}^\Lambda \tilde{C} and
/// {}^\Gamma \tilde{C}, and the quantities required for the generalised
/// Slater-Condon rules are constructed.
/// # Arguments:
/// - `l_c_occ`: Occupied MO coefficient matrix {}^\Lambda C.
/// - `g_c_occ`: Occupied MO coefficient matrix {}^\Gamma C.
/// - `s_munu`: AO overlap matrix.
/// - `tol`: Tolerance up to which a number is considered zero.
/// # Returns:
/// - `Pair`: Overlap-related intermediates for this determinant pair,
///   including the overlap, reduced overlap, zero singular-value indices,
///   rotated occupied coefficients, optional co-densities, and phase.
pub(crate) fn build_s_pair(l_c_occ: &Array2<f64>, g_c_occ: &Array2<f64>, s_munu: &Array2<f64>, tol: f64) -> Pair {

    // Occupied MO overlap.
    let s_ij = l_c_occ.t().dot(&s_munu.dot(g_c_occ));

    // SVD the MO overlap matrix.
    let (u, tilde_s, vdag) = s_ij.svd(true, true).unwrap();
    let u = u.unwrap();
    let v = vdag.unwrap().t().to_owned();
    let phase = u.det().unwrap() * v.det().unwrap();

    // Calculate the reduced MO overlap matrix.
    let (s_red, zeros) = calculate_s_red(&tilde_s, tol);

    // Rotate occupied MO coefficients
    let l_tilde_c_occ = l_c_occ.dot(&u);
    let g_tilde_c_occ = g_c_occ.dot(&v);
    
    // Calculate only the required quantities given the number of zeros.
    let w = match zeros.len() {
        0 | 1 => Some(calculate_codensity_w_pair(&l_tilde_c_occ, &g_tilde_c_occ, &tilde_s, tol)),
        _ => None,
    };

    let (p_i, p_j) = match zeros.len() {
        1 => (
            Some(calculate_codensity_p_pair(&l_tilde_c_occ, &g_tilde_c_occ, zeros[0])), 
            None
        ),
        2 => (
            Some(calculate_codensity_p_pair(&l_tilde_c_occ, &g_tilde_c_occ, zeros[0])), 
            Some(calculate_codensity_p_pair(&l_tilde_c_occ, &g_tilde_c_occ, zeros[1]))
        ),
        _ => (None, None),
    };

    // Overlap matrix element for this pair.
    let prod: f64 = tilde_s.iter().copied().product();
    let s = phase * prod;

    Pair {s, s_red, zeros, w, p_i, p_j, phase}
}
    
/// Calculate {}^{\Lambda\Gamma}P_i^{\mu\nu} = {}^{\Lambda} \tilde{C}_i^\mu {}^{\Gamma}\tilde{C}_i^\nu* 
/// co-density matrix where {}^{\Lambda} \tilde{C}_i^\mu and {}^{\Gamma}\tilde{C}_i^\nu* are the 
/// rotated MO coefficients of determinant \Lambda and \Gamma respectively.
/// # Arguments 
/// - `l_tilde_c_occ`: U rotated occupied MO coefficients for pair of states \Lambda, \Gamma. 
/// - `g_tilde_c_occ`: V rotated occupied MO coefficients for pair of states \Lambda, \Gamma. 
/// - `i`: MO index.
/// # Returns:
/// - `Array2<f64>`: Co-density matrix for occupied orbital index `i`.
fn calculate_codensity_p_pair(l_tilde_c_occ: &Array2<f64>, g_tilde_c_occ: &Array2<f64>, i: usize) -> Array2<f64> {
    let nso = l_tilde_c_occ.nrows();
    let mut munu_p_i = Array2::<f64>::zeros((nso, nso));
    for x in 0..nso {
        for y in 0..nso {
            munu_p_i[(x, y)] = l_tilde_c_occ[(x, i)] * g_tilde_c_occ[(y, i)];
        }
    }
    munu_p_i
}

/// Calculate {}^{\Lambda\Gamma}W^{\mu\nu} = \sum_{i} 1 / s_i * {}^{\Lambda}\tilde{C}_i^\mu {}^{\Gamma}\tilde{C}_i^\nu* 
/// weighted co-density  matrix where s_i are the singular values of the SVD decomposed MO overlap matrix, 
/// and {}^{\Lambda}\tilde{C}_i^\mu and {}^{\Gamma}\tilde{C}_i^\nu are the occupied rotated MO coefficients of state 
/// \Gamma and \Lambda respectively.
/// # Arguments:
/// - `l_tilde_c_occ`: U rotated occupied MO coefficients for pair of states \Lambda, \Gamma. 
/// - `g_tilde_c_occ`: V rotated occupied MO coefficients for pair of states \Lambda, \Gamma.
/// - `s_vals`: Singular values of SVD'd {}^{\Lambda\Gamma} \tilde{S}_{ij}.
/// - `tol`: Tolerance up to which a number is considered zero. 
/// # Returns:
/// - `Array2<f64>`: Weighted co-density matrix.
fn calculate_codensity_w_pair(l_tilde_c_occ: &Array2<f64>, g_tilde_c_occ: &Array2<f64>, tilde_s: &Array1<f64>, tol: f64) -> Array2<f64> {
    let mut l_tilde_c_occ_scaled = l_tilde_c_occ.to_owned();
    
    for (i, mut col) in l_tilde_c_occ_scaled.axis_iter_mut(Axis(1)).enumerate() {
        let w = if tilde_s[i].abs() > tol {1.0 / tilde_s[i]} else {0.0};
        col.mapv_inplace(|z| z * w);
    }

    l_tilde_c_occ_scaled.dot(&g_tilde_c_occ.t())
}

/// Calculate pair density {}^{\Lambda\Gamma} \rho_{ij} using the generalised Slater-Condon rules.
/// # Arguments:
/// - `pair`: Contains data concerning a pair of determinants.
/// - `nao`: Number of AOs.
/// # Returns:
/// - `Array2<f64>`: AO-basis pair density matrix for the determinant pair.
fn pair_density(pair: &Pair, nao: usize) -> Array2<f64> {
    match pair.zeros.len() {
        0 => pair.w.as_ref().unwrap().mapv(|x| x * pair.s_red * pair.phase),
        1 => pair.p_i.as_ref().unwrap().mapv(|x| x * pair.s_red * pair.phase),
        _ => Array2::zeros((nao, nao)),
    }
}

/// Calculate the alpha and beta density matrices of a multireference NOCI state.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `states`: Determinant basis of the NOCI wavefunction.
/// - `c`: Coefficients of the NOCI wavefunction.
/// - `tol`: Tolerance up to which a number is considered zero.
/// # Returns:
/// - `(Array2<f64>, Array2<f64>)`: The alpha and beta AO density matrices.
pub fn noci_density(ao: &AoData, states: &[SCFState], c: &Array1<f64>, tol: f64,) -> (Array2<f64>, Array2<f64>) {
    let nao = ao.h.nrows();
    let nst = states.len();

    (0..nst).into_par_iter().map(|i| {
        let mut da_loc = Array2::<f64>::zeros((nao, nao));
        let mut db_loc = Array2::<f64>::zeros((nao, nao));

        let ldet = &states[i];
        let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
        let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);

        for j in 0..nst {
            let gdet = &states[j];

            let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
            let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

            let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
            let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

            let rhoa = pair_density(&pa, nao);
            let rhob = pair_density(&pb, nao);

            let cij = c[i] * c[j];
            da_loc.scaled_add(cij * pb.s, &rhoa);
            db_loc.scaled_add(cij * pa.s, &rhob);
        }
    (da_loc, db_loc)
    }).reduce(|| {(Array2::<f64>::zeros((nao, nao)), Array2::<f64>::zeros((nao, nao)))}, |(mut da_a, mut db_a), (da_b, db_b)| {
        da_a += &da_b;
        db_a += &db_b;
        (da_a, db_a)
    })
}

/// Calculate one body matrix elements using the generalised 
/// Slater-Condon rules for a pair of determinants \Lambda and \Gamma. 
/// # Arguments:
/// - `o`: Operator to obtain matrix elements of.
/// - `pair`: Contains data concerning a pair of determinants.
/// # Returns:
/// - `f64`: One-electron matrix element for the determinant pair.
pub(crate) fn one_electron(o: &Array2<f64>, pair: &Pair) -> f64 {
    match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w.
        0 => pair.s_red * pair.phase * einsum_ba_ab_real(pair.w.as_ref().unwrap(), o),
        // With 1 zero (s_i = 0 for 1 i) we use P_i.
        1 => pair.s_red * pair.phase * einsum_ba_ab_real(pair.p_i.as_ref().unwrap(), o),
        // Otherwise the matrix element is zero.
        _ => 0.0,
    }
}

/// Calculate two-body matrix elements for electrons of the same spin
/// using the generalised Slater-Condon rules for a pair of determinants
/// \Lambda and \Gamma.
/// # Arguments:
/// - `o`: Antisymmetrised two-electron operator tensor.
/// - `pair`: Contains data concerning a pair of determinants.
/// # Returns:
/// - `f64`: Same-spin two-electron matrix element.
pub(crate) fn two_electron_same(o: &Array4<f64>, pair: &Pair) -> f64 {
    match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w on both sides.
        0 => 0.5 * pair.s_red * pair.phase * einsum_ba_abcd_cd_real(pair.w.as_ref().unwrap(), o, pair.w.as_ref().unwrap()),
        // With 1 zero (s_i = 0 for one index i) we use P_i on one side.
        1 => pair.s_red * pair.phase * einsum_ba_abcd_cd_real(pair.w.as_ref().unwrap(), o, pair.p_i.as_ref().unwrap()),
        // with 2 zeros (s_i, s_j = 0 for two indices i, j) we use P_i on one side and P_j on the other.
        2 => pair.s_red * pair.phase * einsum_ba_abcd_cd_real(pair.p_i.as_ref().unwrap(), o, pair.p_j.as_ref().unwrap()),
        // Otherwise the matrix element is 0.
        _ => 0.0,
    }
}

/// Calculate two-body matrix elements for electrons of opposite spin
/// using the generalised Slater-Condon rules for a pair of determinants
/// \Lambda and \Gamma.
/// # Arguments:
/// - `o`: Coulomb two-electron operator tensor.
/// - `pa`: Alpha-spin pair data.
/// - `pb`: Beta-spin pair data.
/// # Returns:
/// - `f64`: Opposite-spin two-electron matrix element.
pub(crate) fn two_electron_diff(o: &Array4<f64>, pa: &Pair, pb: &Pair) -> f64 {
    match (pa.zeros.len(), pb.zeros.len()) {
        // With no zeros (s_i != 0 for all i) for both spins we use munu_w on both sides.
        (0, 0) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.w.as_ref().unwrap(), o, pb.w.as_ref().unwrap()),
        // With one zero in beta spin only we use W^a and P_i^b.
        (0, 1) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.w.as_ref().unwrap(), o, pb.p_i.as_ref().unwrap()),
        // With one zero in alpha spin only we use P_i^a and W_b.
        (1, 0) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.p_i.as_ref().unwrap(), o, pb.w.as_ref().unwrap()),
        // With one zero in both spins we use P_i^a and P_i^b.
        (1, 1) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.p_i.as_ref().unwrap(), o, pb.p_i.as_ref().unwrap()),
        // Otherwise the matrix element is zero.
        _ => 0.0,
    }
}

