// noci/cache.rs
use ndarray::{Array2, Array4};

use crate::{AoData, SCFState};

use crate::maths::eri_ao2mo;

pub struct MOCache {
    /// One-electron Hamiltonian in parent alpha MO basis.
    pub ha: Array2<f64>,
    /// One-electron Hamiltonian in parent beta MO basis.
    pub hb: Array2<f64>,
    /// Antisymmetrised same-spin ERIs in parent alpha MO basis.
    pub eri_aa_asym: Array4<f64>,
    /// Antisymmetrised same-spin ERIs in parent beta MO basis.
    pub eri_bb_asym: Array4<f64>,
    /// Coulomb different-spin ERIs in parent alpha/beta MO basis.
    pub eri_ab_coul: Array4<f64>,
}

pub struct FockMOCache {
    /// Spin-alpha Fock matrix in the parent alpha MO basis.
    pub fa: Array2<f64>,
    /// Spin-beta Fock matrix in the parent beta MO basis.
    pub fb: Array2<f64>,
}

/// Build MO-basis caches for all reference determinants.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `parents`: Reference NOCI determinants (parents of excited determinants).
/// # Returns:
/// - `Vec<MOCache>`: MO-basis integrals for each parent.
pub fn build_mo_cache(ao: &AoData, parents: &[SCFState]) -> Vec<MOCache> {
    parents.iter().map(|st| {
        let ca = st.ca.as_ref();
        let cb = st.cb.as_ref();

        let ha = st.ca.t().dot(&ao.h).dot(ca);
        let hb = st.cb.t().dot(&ao.h).dot(cb);

        let eri_aa_asym = eri_ao2mo(&ao.eri_asym, &st.ca, &st.ca, &st.ca, &st.ca).as_standard_layout().to_owned();
        let eri_bb_asym = eri_ao2mo(&ao.eri_asym, &st.cb, &st.cb, &st.cb, &st.cb).as_standard_layout().to_owned();
        let eri_ab_coul = eri_ao2mo(&ao.eri_coul, &st.ca, &st.ca, &st.cb, &st.cb).as_standard_layout().to_owned();

        MOCache {ha, hb, eri_aa_asym, eri_bb_asym, eri_ab_coul}
    }).collect()
}

/// Build MO-basis Fock caches for all reference determinants.
/// # Arguments:
/// - `fa`: Spin-alpha Fock matrix in the AO basis.
/// - `fb`: Spin-beta Fock matrix in the AO basis.
/// - `parents`: Reference NOCI determinants (parents of excited determinants).
/// # Returns:
/// - `Vec<FockMOCache>`: MO basis Fock matrices for each parent.
pub fn build_fock_mo_cache(fa: &Array2<f64>, fb: &Array2<f64>, parents: &[SCFState]) -> Vec<FockMOCache> {
    parents.iter().map(|st| {
        let ca = st.ca.as_ref();
        let cb = st.cb.as_ref();

        let fa_mo = ca.t().dot(fa).dot(ca);
        let fb_mo = cb.t().dot(fb).dot(cb);

        FockMOCache {fa: fa_mo, fb: fb_mo}
    }).collect()
}
