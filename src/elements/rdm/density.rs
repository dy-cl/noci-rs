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

                let det_phase = <T as From<f64>>::from(space.phase(ldet) * space.phase(gdet));

                // The spectator-spin overlap multiplies each spin's
                // transition density in the determinant product state.
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

/// Calculate the spin-squared expectation value `\langle S^2\rangle` of a multireference NOCI
/// state `|\Psi\rangle = \sum_i c_i |i\rangle`.
/// Each determinant pair contributes
/// `\langle i|S^2|j\rangle = S_z(S_z + 1) S^\alpha S^\beta + S^\alpha \mathrm{Tr}(\rho^\beta S)
/// - \mathrm{Tr}(\rho^\alpha S \rho^\beta S)`, where `\rho^\sigma = S^\sigma \gamma^\sigma` is the
/// overlap-weighted transition density, so pairs with a zero spin overlap stay finite. For
/// `i = j` this reduces to the single-determinant formula used for SCF states.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `space`: Determinant space whose transition densities are combined.
/// - `states`: Determinant basis of the NOCI wavefunction.
/// - `c`: Coefficients of the NOCI wavefunction.
/// - `tol`: Tolerance up to which a number is considered zero.
/// # Returns:
/// - `f64`: `\langle\Psi|S^2|\Psi\rangle / \langle\Psi|\Psi\rangle`.
pub(crate) fn noci_spin_square<T: NOCIScalar>(
    ao: &AoData,
    space: &crate::determinant::NOCISpace<T>,
    states: &[crate::determinant::NOCIIndex],
    c: &Array1<T>,
    tol: f64,
) -> f64 {
    let nao = ao.h.nrows();
    let s = ao.s.mapv(<T as From<f64>>::from);

    // Accumulate `\sum_{ij} c_i^* c_j \langle i|S^2|j\rangle` and the norm `\sum_{ij} c_i^* c_j S_{ij}`.
    let (s2, norm) = (0..states.len())
        .into_par_iter()
        .map(|i| {
            let mut s2 = <T as From<f64>>::from(0.0);
            let mut norm = <T as From<f64>>::from(0.0);

            let ldet = states[i];
            let lparent = space.parent_determinant(ldet);
            let (loa, lob) = space.occupations(ldet);
            let l_ca_occ = occ_coeffs(&lparent.ca, loa);
            let l_cb_occ = occ_coeffs(&lparent.cb, lob);
            let sz = 0.5 * (loa.count_ones() as f64 - lob.count_ones() as f64);

            for j in 0..states.len() {
                let gdet = states[j];
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
                let pair_s2 =
                    <T as From<f64>>::from(sz * (sz + 1.0)) * pa.s * pb.s + pa.s * nb - cross;

                let det_phase = <T as From<f64>>::from(space.phase(ldet) * space.phase(gdet));
                let cij = c[i].conj() * c[j] * det_phase;
                s2 += cij * pair_s2;
                norm += cij * pa.s * pb.s;
            }

            (s2, norm)
        })
        .reduce(
            || (<T as From<f64>>::from(0.0), <T as From<f64>>::from(0.0)),
            |(s2_a, norm_a), (s2_b, norm_b)| (s2_a + s2_b, norm_a + norm_b),
        );

    (s2 / norm).re()
}
