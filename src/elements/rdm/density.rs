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
            let lparent = space.parent(ldet);
            let (loa, lob) = space.occupations(ldet);
            let l_ca_occ = occ_coeffs(&lparent.ca, loa);
            let l_cb_occ = occ_coeffs(&lparent.cb, lob);

            for j in 0..nst {
                let gdet = states[j];
                let gparent = space.parent(gdet);
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
