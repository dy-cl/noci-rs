// noci/cache.rs
use ndarray::{Array2, Array4};

use crate::time_call;
use crate::{AoData, DetState};

use super::types::{FockMOCache, MOCache, NOCIScalar};
use crate::maths::{adjoint, real2_as};

/// Calculate maximum deviation from Hermitian orthonormality, `C^\dagger S C = I`.
/// # Arguments:
/// - `c`: MO coefficient matrix.
/// - `s`: AO overlap matrix.
/// # Returns:
/// - `f64`: Maximum absolute deviation from the identity.
fn hermitian_orthonormal_error<T: NOCIScalar>(
    c: &Array2<T>,
    s: &Array2<f64>,
) -> f64 {
    let smat = real2_as::<T>(s);
    let ov = adjoint(c).dot(&smat).dot(c);
    let mut err = 0.0;

    for i in 0..ov.nrows() {
        for j in 0..ov.ncols() {
            let target = if i == j {
                <T as From<f64>>::from(1.0)
            } else {
                <T as From<f64>>::from(0.0)
            };
            let d = (ov[(i, j)] - target).abs();
            if d > err {
                err = d;
            }
        }
    }

    err
}

/// Build MO-basis caches for all reference determinants.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `parents`: Reference NOCI determinants.
/// - `tol`: Tolerance for identifying Hermitian-orthonormal parent MOs.
/// # Returns:
/// - `Vec<MOCache<T>>`: MO-basis integrals for each parent.
pub fn build_mo_cache<T: NOCIScalar>(
    ao: &AoData,
    parents: &[DetState<T>],
    tol: f64,
) -> Vec<MOCache<T>> {
    time_call!(crate::timers::noci::add_build_mo_cache, {
        parents
            .iter()
            .map(|st| {
                let ca = st.ca.as_ref();
                let cb = st.cb.as_ref();
                let h = real2_as::<T>(&ao.h);

                let ha = adjoint(ca).dot(&h).dot(ca);
                let hb = adjoint(cb).dot(&h).dot(cb);

                let ca_conj = ca.mapv(|z| z.conj());
                let cb_conj = cb.mapv(|z| z.conj());

                let nmo_a = ca.ncols();
                let nmo_b = cb.ncols();
                let mut eri_aa_asym = Array4::<T>::zeros((nmo_a, nmo_a, nmo_a, nmo_a));
                let mut scratch =
                    T::new_eri_ao2mo_scratch(&ao.eri_asym, nmo_a, nmo_a, nmo_a, nmo_a);
                T::eri_ao2mo_hermitian_into(
                    &ao.eri_asym,
                    ca,
                    ca,
                    &ca_conj,
                    &ca_conj,
                    eri_aa_asym.view_mut(),
                    &mut scratch,
                );
                let mut eri_bb_asym = Array4::<T>::zeros((nmo_b, nmo_b, nmo_b, nmo_b));
                let mut scratch =
                    T::new_eri_ao2mo_scratch(&ao.eri_asym, nmo_b, nmo_b, nmo_b, nmo_b);
                T::eri_ao2mo_hermitian_into(
                    &ao.eri_asym,
                    cb,
                    cb,
                    &cb_conj,
                    &cb_conj,
                    eri_bb_asym.view_mut(),
                    &mut scratch,
                );
                let mut eri_ab_coul = Array4::<T>::zeros((nmo_a, nmo_a, nmo_b, nmo_b));
                let mut scratch =
                    T::new_eri_ao2mo_scratch(&ao.eri_coul, nmo_a, nmo_a, nmo_b, nmo_b);
                T::eri_ao2mo_hermitian_into(
                    &ao.eri_coul,
                    ca,
                    ca,
                    &cb_conj,
                    &cb_conj,
                    eri_ab_coul.view_mut(),
                    &mut scratch,
                );

                let orthogonal_slater_condon = hermitian_orthonormal_error(ca, &ao.s) < tol
                    && hermitian_orthonormal_error(cb, &ao.s) < tol;

                MOCache {
                    ha,
                    hb,
                    eri_aa_asym,
                    eri_bb_asym,
                    eri_ab_coul,
                    orthogonal_slater_condon,
                }
            })
            .collect()
    })
}

/// Build MO-basis Fock caches for all reference determinants.
/// # Arguments:
/// - `fa`: Spin-alpha Fock matrix in the AO basis.
/// - `fb`: Spin-beta Fock matrix in the AO basis.
/// - `parents`: Reference NOCI determinants.
/// - `s`: AO overlap matrix.
/// - `tol`: Tolerance for identifying Hermitian-orthonormal parent MOs.
/// # Returns:
/// - `Vec<FockMOCache<T>>`: MO basis Fock matrices for each parent.
pub fn build_fock_mo_cache<T: NOCIScalar>(
    fa: &Array2<T>,
    fb: &Array2<T>,
    parents: &[DetState<T>],
    s: &Array2<f64>,
    tol: f64,
) -> Vec<FockMOCache<T>> {
    time_call!(crate::timers::noci::add_build_fock_mo_cache, {
        parents
            .iter()
            .map(|st| {
                let ca = st.ca.as_ref();
                let cb = st.cb.as_ref();

                let fa_mo = adjoint(ca).dot(fa).dot(ca);
                let fb_mo = adjoint(cb).dot(fb).dot(cb);
                let orthogonal_slater_condon = hermitian_orthonormal_error(ca, s) < tol
                    && hermitian_orthonormal_error(cb, s) < tol;

                FockMOCache {
                    fa: fa_mo,
                    fb: fb_mo,
                    orthogonal_slater_condon,
                }
            })
            .collect()
    })
}
