// noci/naive.rs
use ndarray::{Array1, Array2, Array4, Axis};
use ndarray_linalg::{Determinant, SVD};
use rayon::prelude::*;

use super::types::{NOCIScalar, Pair};
use crate::{AoData, DetState};

use crate::maths::{adjoint, real2_as};

/// Given an MO coefficient matrix and occupancy vector, return the occupied-only coefficient matrix.
/// # Arguments:
/// - `c`: MO coefficient matrix.
/// - `occ`: Occupancy vector.
/// # Returns:
/// - `Array2<T>`: Occupied-only MO coefficient matrix.
pub fn occ_coeffs<T: NOCIScalar>(
    c: &Array2<T>,
    occ: u128,
) -> Array2<T> {
    let occ_idx: Vec<usize> = (0..c.ncols())
        .filter(|&i| ((occ >> i) & 1u128) == 1)
        .collect();
    let mut c_occ = Array2::<T>::zeros((c.nrows(), occ_idx.len()));
    for (k, &i) in occ_idx.iter().enumerate() {
        c_occ.column_mut(k).assign(&c.index_axis(Axis(1), i));
    }
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
fn calculate_s_red(
    tilde_s: &Array1<f64>,
    tol: f64,
) -> (f64, Vec<usize>) {
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
/// # Returns:
/// - `Pair<T>`: Overlap-related intermediates for this determinant pair.
pub(in crate::noci) fn build_s_pair<T: NOCIScalar>(
    l_c_occ: &Array2<T>,
    g_c_occ: &Array2<T>,
    s_munu: &Array2<f64>,
    tol: f64,
) -> Pair<T> {
    // Occupied MO overlap.
    let s = real2_as::<T>(s_munu);
    let s_ij = adjoint(l_c_occ).dot(&s).dot(g_c_occ);

    // SVD the MO overlap matrix.
    let (u, tilde_s, vdag) = s_ij.svd(true, true).unwrap();
    let u = u.unwrap();
    let v = adjoint(&vdag.unwrap());
    let phase = u.det().unwrap() * v.det().unwrap().conj();

    // Calculate the reduced MO overlap matrix.
    let (s_red, zeros) = calculate_s_red(&tilde_s, tol);

    // Rotate occupied MO coefficients
    let l_tilde_c_occ = l_c_occ.dot(&u);
    let g_tilde_c_occ = g_c_occ.dot(&v);

    // Calculate only the required quantities given the number of zeros.
    let w = match zeros.len() {
        0 | 1 => Some(calculate_codensity_w_pair(
            &l_tilde_c_occ,
            &g_tilde_c_occ,
            &tilde_s,
            tol,
        )),
        _ => None,
    };

    let (p_i, p_j) = match zeros.len() {
        1 => (
            Some(calculate_codensity_p_pair(
                &l_tilde_c_occ,
                &g_tilde_c_occ,
                zeros[0],
            )),
            None,
        ),
        2 => (
            Some(calculate_codensity_p_pair(
                &l_tilde_c_occ,
                &g_tilde_c_occ,
                zeros[0],
            )),
            Some(calculate_codensity_p_pair(
                &l_tilde_c_occ,
                &g_tilde_c_occ,
                zeros[1],
            )),
        ),
        _ => (None, None),
    };

    // Overlap matrix element for this pair.
    let prod: f64 = tilde_s.iter().copied().product();
    let s = if zeros.is_empty() {
        phase * <T as From<f64>>::from(prod)
    } else {
        <T as From<f64>>::from(0.0)
    };

    Pair {
        s,
        s_red,
        zeros,
        w,
        p_i,
        p_j,
        phase,
    }
}

/// Calculate {}^{\Lambda\Gamma}P_i^{\mu\nu} = {}^{\Gamma} \tilde{C}_i^\mu {}^{\Lambda}\tilde{C}_i^{\nu *}
/// co-density matrix where {}^{\Lambda} \tilde{C}_i^\nu and {}^{\Gamma}\tilde{C}_i^\mu are the
/// rotated MO coefficients of determinants \Lambda and \Gamma respectively.
/// # Arguments
/// - `l_tilde_c_occ`: U rotated occupied MO coefficients for determinant \Lambda.
/// - `g_tilde_c_occ`: V rotated occupied MO coefficients for determinant \Gamma.
/// - `i`: MO index.
/// # Returns:
/// - `Array2<T>`: Co-density matrix for occupied orbital index `i`.
fn calculate_codensity_p_pair<T: NOCIScalar>(
    l_tilde_c_occ: &Array2<T>,
    g_tilde_c_occ: &Array2<T>,
    i: usize,
) -> Array2<T> {
    let nso = l_tilde_c_occ.nrows();
    let mut munu_p_i = Array2::<T>::zeros((nso, nso));
    for x in 0..nso {
        for y in 0..nso {
            munu_p_i[(x, y)] = g_tilde_c_occ[(x, i)] * l_tilde_c_occ[(y, i)].conj();
        }
    }
    munu_p_i
}

/// Calculate {}^{\Lambda\Gamma}W^{\mu\nu} = \sum_{i} 1 / s_i * {}^{\Gamma}\tilde{C}_i^\mu {}^{\Lambda}\tilde{C}_i^{\nu *}
/// weighted co-density matrix where s_i are the singular values of the SVD decomposed MO overlap matrix,
/// and {}^{\Lambda}\tilde{C}_i^\nu and {}^{\Gamma}\tilde{C}_i^\mu are the occupied rotated MO coefficients of
/// determinants \Lambda and \Gamma respectively.
/// # Arguments:
/// - `l_tilde_c_occ`: U rotated occupied MO coefficients for determinant \Lambda.
/// - `g_tilde_c_occ`: V rotated occupied MO coefficients for determinant \Gamma.
/// - `tilde_s`: Singular values of SVD'd {}^{\Lambda\Gamma} \tilde{S}_{ij}.
/// - `tol`: Tolerance up to which a number is considered zero.
/// # Returns:
/// - `Array2<T>`: Weighted co-density matrix.
fn calculate_codensity_w_pair<T: NOCIScalar>(
    l_tilde_c_occ: &Array2<T>,
    g_tilde_c_occ: &Array2<T>,
    tilde_s: &Array1<f64>,
    tol: f64,
) -> Array2<T> {
    let mut g_tilde_c_occ_scaled = g_tilde_c_occ.to_owned();

    for (i, mut col) in g_tilde_c_occ_scaled.axis_iter_mut(Axis(1)).enumerate() {
        let w = if tilde_s[i].abs() > tol {
            1.0 / tilde_s[i]
        } else {
            0.0
        };
        col.mapv_inplace(|z| z * <T as From<f64>>::from(w));
    }

    g_tilde_c_occ_scaled.dot(&adjoint(l_tilde_c_occ))
}

/// Calculate pair density {}^{\Lambda\Gamma} \rho_{ij} using the generalised Slater-Condon rules.
/// # Arguments:
/// - `pair`: Contains data concerning a pair of determinants.
/// - `nao`: Number of AOs.
/// # Returns:
/// - `Array2<T>`: AO-basis pair density matrix for the determinant pair.
pub(in crate::noci) fn pair_density<T: NOCIScalar>(
    pair: &Pair<T>,
    nao: usize,
) -> Array2<T> {
    let fac = pair.phase * <T as From<f64>>::from(pair.s_red);
    match pair.zeros.len() {
        0 => pair.w.as_ref().unwrap().mapv(|x| x * fac),
        1 => pair.p_i.as_ref().unwrap().mapv(|x| x * fac),
        _ => Array2::<T>::zeros((nao, nao)),
    }
}

/// Calculate the alpha and beta density matrices of a multireference NOCI state.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `states`: Determinant basis of the NOCI wavefunction.
/// - `c`: Coefficients of the NOCI wavefunction.
/// - `tol`: Tolerance up to which a number is considered zero.
/// # Returns:
/// - `(Array2<T>, Array2<T>)`: Alpha and beta AO density matrices.
pub fn noci_density<T: NOCIScalar>(
    ao: &AoData,
    states: &[DetState<T>],
    c: &Array1<T>,
    tol: f64,
) -> (Array2<T>, Array2<T>) {
    let nao = ao.h.nrows();
    let nst = states.len();
    (0..nst)
        .into_par_iter()
        .map(|i| {
            let mut da_loc = Array2::<T>::zeros((nao, nao));
            let mut db_loc = Array2::<T>::zeros((nao, nao));

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

                let det_phase =
                    <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));

                let cij = c[i].conj() * c[j] * det_phase;
                da_loc.scaled_add(cij * pb.s, &rhoa);
                db_loc.scaled_add(cij * pa.s, &rhob);
            }
            (da_loc, db_loc)
        })
        .reduce(
            || {
                (
                    Array2::<T>::zeros((nao, nao)),
                    Array2::<T>::zeros((nao, nao)),
                )
            },
            |(mut da_a, mut db_a), (da_b, db_b)| {
                da_a += &da_b;
                db_a += &db_b;
                (da_a, db_a)
            },
        )
}
/// Calculate one body matrix elements using the generalised
/// Slater-Condon rules for a pair of determinants \Lambda and \Gamma.
/// # Arguments:
/// - `o`: Operator to obtain matrix elements of.
/// - `pair`: Contains data concerning a pair of determinants.
/// # Returns:
/// - `T`: One-electron matrix element for the determinant pair, before external determinant excitation phase.
pub(in crate::noci) fn one_electron<T: NOCIScalar>(
    o: &Array2<f64>,
    pair: &Pair<T>,
) -> T {
    let fac = pair.phase * <T as From<f64>>::from(pair.s_red);
    match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w.
        0 => fac * T::einsum_ba_ab_realop(pair.w.as_ref().unwrap(), o),
        // With 1 zero (s_i = 0 for 1 i) we use P_i.
        1 => fac * T::einsum_ba_ab_realop(pair.p_i.as_ref().unwrap(), o),
        // Otherwise the matrix element is zero.
        _ => <T as From<f64>>::from(0.0),
    }
}

/// Calculate one body matrix elements using the generalised
/// Slater-Condon rules for a pair of determinants \Lambda and \Gamma,
/// with a scalar-valued one-body operator.
/// # Arguments:
/// - `o`: Operator to obtain matrix elements of.
/// - `pair`: Contains data concerning a pair of determinants.
/// # Returns:
/// - `T`: One-electron matrix element for the determinant pair, before external determinant excitation phase.
pub(in crate::noci) fn one_electron_scalar<T: NOCIScalar>(
    o: &Array2<T>,
    pair: &Pair<T>,
) -> T {
    let fac = pair.phase * <T as From<f64>>::from(pair.s_red);
    match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w.
        0 => fac * T::einsum_ba_ab(pair.w.as_ref().unwrap(), o),
        // With 1 zero (s_i = 0 for 1 i) we use P_i.
        1 => fac * T::einsum_ba_ab(pair.p_i.as_ref().unwrap(), o),
        // Otherwise the matrix element is zero.
        _ => <T as From<f64>>::from(0.0),
    }
}

/// Calculate two-body matrix elements for electrons of the same spin
/// using the generalised Slater-Condon rules for a pair of determinants
/// \Lambda and \Gamma.
/// # Arguments:
/// - `o`: Antisymmetrised two-electron operator tensor.
/// - `pair`: Contains data concerning a pair of determinants.
/// # Returns:
/// - `T`: Same-spin two-electron matrix element, before external determinant excitation phase.
pub(in crate::noci) fn two_electron_same<T: NOCIScalar>(
    o: &Array4<f64>,
    pair: &Pair<T>,
) -> T {
    let fac = pair.phase * <T as From<f64>>::from(pair.s_red);
    match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w on both sides.
        0 => {
            <T as From<f64>>::from(0.5)
                * fac
                * T::einsum_ba_abcd_cd_realop(pair.w.as_ref().unwrap(), o, pair.w.as_ref().unwrap())
        }
        // With 1 zero (s_i = 0 for one index i) we use P_i on one side.
        1 => {
            fac * T::einsum_ba_abcd_cd_realop(
                pair.p_i.as_ref().unwrap(),
                o,
                pair.w.as_ref().unwrap(),
            )
        }
        // with 2 zeros (s_i, s_j = 0 for two indices i, j) we use P_i on one side and P_j on the other.
        2 => {
            fac * T::einsum_ba_abcd_cd_realop(
                pair.p_i.as_ref().unwrap(),
                o,
                pair.p_j.as_ref().unwrap(),
            )
        }
        // Otherwise the matrix element is 0.
        _ => <T as From<f64>>::from(0.0),
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
/// - `T`: Opposite-spin two-electron matrix element, before external determinant excitation phase.
pub(in crate::noci) fn two_electron_diff<T: NOCIScalar>(
    o: &Array4<f64>,
    pa: &Pair<T>,
    pb: &Pair<T>,
) -> T {
    let fa = pa.phase * <T as From<f64>>::from(pa.s_red);
    let fb = pb.phase * <T as From<f64>>::from(pb.s_red);
    match (pa.zeros.len(), pb.zeros.len()) {
        // With no zeros (s_i != 0 for all i) for both spins we use munu_w on both sides.
        (0, 0) => {
            fa * fb * T::einsum_ba_abcd_cd_realop(pa.w.as_ref().unwrap(), o, pb.w.as_ref().unwrap())
        }
        // With one zero in beta spin only we use W^a and P_i^b.
        (0, 1) => {
            fa * fb
                * T::einsum_ba_abcd_cd_realop(pa.w.as_ref().unwrap(), o, pb.p_i.as_ref().unwrap())
        }
        // With one zero in alpha spin only we use P_i^a and W_b.
        (1, 0) => {
            fa * fb
                * T::einsum_ba_abcd_cd_realop(pa.p_i.as_ref().unwrap(), o, pb.w.as_ref().unwrap())
        }
        // With one zero in both spins we use P_i^a and P_i^b.
        (1, 1) => {
            fa * fb
                * T::einsum_ba_abcd_cd_realop(pa.p_i.as_ref().unwrap(), o, pb.p_i.as_ref().unwrap())
        }
        // Otherwise the matrix element is zero.
        _ => <T as From<f64>>::from(0.0),
    }
}
