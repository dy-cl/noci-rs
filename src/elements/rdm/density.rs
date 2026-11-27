// elements/rdm/density.rs

// External crate imports.
use ndarray::{Array1, Array2};
use rayon::prelude::*;

// Crate-root imports.
use crate::elements::{build_s_pair, occ_coeffs, pair_density};
use crate::{AoData, NOCIScalar};

/// Calculate the alpha and beta density matrices of a multireference NOCI state.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `states`: Determinant basis of the NOCI wavefunction.
/// - `c`: Coefficients of the NOCI wavefunction.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `space`: Determinant space whose transition densities are combined.
/// # Returns:
/// - `(Array2<T>, Array2<T>)`: Alpha and beta AO density matrices.
pub(crate) fn noci_density<T: NOCIScalar>(
    ao: &AoData,
    space: &crate::determinant::NOCISpace<T>,
    states: &[crate::determinant::NOCIIndex],
    c: &Array1<T>,
    tol: f64,
) -> (Array2<T>, Array2<T>) {
    let nao = ao.h.nrows();
    let nst = states.len();

    // `D^\sigma = \sum_{ij} c_i^* c_j \langle i|a_\sigma^\dagger a_\sigma|j\rangle`.
    // Partition bra states across threads and reduce their local AO densities.
    (0..nst)
        .into_par_iter()
        .map(|i| {
            let mut da_loc = Array2::<T>::zeros((nao, nao));
            let mut db_loc = Array2::<T>::zeros((nao, nao));

            let ldet = states[i];
            let lparent = space.parent_determinant(ldet);
            let (loa, lob) = space.occupations(ldet);
            let l_ca_occ = occ_coeffs(&lparent.ca, loa);
            let l_cb_occ = occ_coeffs(&lparent.cb, lob);

            for j in 0..nst {
                let gdet = states[j];
                let gparent = space.parent_determinant(gdet);
                let (goa, gob) = space.occupations(gdet);

                let g_ca_occ = occ_coeffs(&gparent.ca, goa);
                let g_cb_occ = occ_coeffs(&gparent.cb, gob);

                let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
                let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

                let rhoa = pair_density(&pa, nao);
                let rhob = pair_density(&pb, nao);

                // The spectator-spin overlap multiplies each spin's
                // transition density in the determinant product state.
                let cij = c[i].conj() * c[j];
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

/// Build the spin-squared matrix `\langle l|S^2|r\rangle` between two determinant lists, from which
/// a NOCI state `|\Psi\rangle = \sum_i c_i |i\rangle` has
/// `\langle S^2\rangle = \sum_{ij} c_i^* c_j \langle i|S^2|j\rangle / \sum_{ij} c_i^* c_j S_{ij}`.
/// Each determinant pair contributes
/// `\langle l|S^2|r\rangle = S_z(S_z + 1) S^\alpha S^\beta + S^\alpha \mathrm{Tr}(\rho^\beta S)
/// - \mathrm{Tr}(\rho^\alpha S \rho^\beta S)`, where `\rho^\sigma = S^\sigma \gamma^\sigma` is the
/// overlap-weighted transition density, so pairs with a zero spin overlap stay finite. For
/// `l = r` this reduces to the single-determinant formula used for SCF states.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `space`: Determinant space whose transition densities are combined.
/// - `left`: Bra determinants, the rows.
/// - `right`: Ket determinants, the columns.
/// - `tol`: Tolerance up to which a number is considered zero.
/// # Returns:
/// - `Array2<T>`: Spin-squared matrix `\langle l|S^2|r\rangle`.
pub(crate) fn build_noci_spin_square<T: NOCIScalar>(
    ao: &AoData,
    space: &crate::determinant::NOCISpace<T>,
    left: &[crate::determinant::NOCIIndex],
    right: &[crate::determinant::NOCIIndex],
    tol: f64,
) -> Array2<T> {
    let nao = ao.h.nrows();
    let s = ao.s.mapv(<T as From<f64>>::from);

    // Each row `l` of `\langle l|S^2|r\rangle` is evaluated independently.
    let rows: Vec<Vec<T>> = left
        .par_iter()
        .map(|&ldet| {
            let lparent = space.parent_determinant(ldet);
            let (loa, lob) = space.occupations(ldet);
            let l_ca_occ = occ_coeffs(&lparent.ca, loa);
            let l_cb_occ = occ_coeffs(&lparent.cb, lob);
            let sz = 0.5 * (loa.count_ones() as f64 - lob.count_ones() as f64);

            right
                .iter()
                .map(|&gdet| {
                    let gparent = space.parent_determinant(gdet);
                    let (goa, gob) = space.occupations(gdet);

                    let pa = build_s_pair(&l_ca_occ, &occ_coeffs(&gparent.ca, goa), &ao.s, tol);
                    let pb = build_s_pair(&l_cb_occ, &occ_coeffs(&gparent.cb, gob), &ao.s, tol);

                    let rhoa = pair_density(&pa, nao);
                    let rhob = pair_density(&pb, nao);

                    // `S_z(S_z + 1) S^\alpha S^\beta + S^\alpha \mathrm{Tr}(\rho^\beta S)
                    //  - \mathrm{Tr}(\rho^\alpha S \rho^\beta S)`.
                    let nb = rhob.dot(&s).diag().sum();
                    let cross = rhoa.dot(&s).dot(&rhob).dot(&s).diag().sum();
                    <T as From<f64>>::from(sz * (sz + 1.0)) * pa.s * pb.s + pa.s * nb - cross
                })
                .collect()
        })
        .collect();

    Array2::from_shape_fn((left.len(), right.len()), |(l, r)| rows[l][r])
}
