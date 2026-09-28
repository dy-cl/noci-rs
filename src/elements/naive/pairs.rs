// elements/naive/pairs.rs
//! Determinant-pair matrix elements from the generalised Slater–Condon rules.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::determinant::{NOCIIndex, NOCISpace};
use crate::elements::{FockData, NOCIData};
use crate::time_call;
use crate::{AoData, NOCIScalar};

// Parent/sibling imports.
use super::rules::{
    build_s_pair, occ_coeffs, one_electron, one_electron_scalar, two_electron_diff,
    two_electron_same,
};

/// Calculate the overlap matrix element between determinants x and w using
/// generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// # Returns:
/// - `T`: Overlap matrix element between `ldet` and `gdet`.
pub(crate) fn calculate_s_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_naive, {
        let lp = data.space.parent(ldet);
        let gp = data.space.parent(gdet);
        let (loa, lob) = data.space.occupations(ldet);
        let (goa, gob) = data.space.occupations(gdet);

        let l_ca_occ = occ_coeffs(&lp.ca, loa);
        let g_ca_occ = occ_coeffs(&gp.ca, goa);
        let l_cb_occ = occ_coeffs(&lp.cb, lob);
        let g_cb_occ = occ_coeffs(&gp.cb, gob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

        let det_phase = <T as From<f64>>::from(data.space.phase(ldet) * data.space.phase(gdet));
        det_phase * pa.s * pb.s
    })
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants x and w
/// using generalised Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `space`: Determinant space containing the bra and ket states.
/// - `tol`: Numerical tolerance for the determinant overlap.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`.
pub(crate) fn calculate_hs_pair_naive<T: NOCIScalar>(
    ao: &AoData,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair_naive, {
        // Per spin occupid coefficients.
        let lp = space.parent(ldet);
        let gp = space.parent(gdet);
        let (loa, lob) = space.occupations(ldet);
        let (goa, gob) = space.occupations(gdet);

        let l_ca_occ = occ_coeffs(&lp.ca, loa);
        let g_ca_occ = occ_coeffs(&gp.ca, goa);
        let l_cb_occ = occ_coeffs(&lp.cb, lob);
        let g_cb_occ = occ_coeffs(&gp.cb, gob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

        // Overlap matrix element for this pair.
        let s = pa.s * pb.s;

        let hnuc = match (pa.zeros.len(), pb.zeros.len()) {
            (0, 0) => <T as From<f64>>::from(ao.enuc) * s,
            _ => <T as From<f64>>::from(0.0),
        };

        let h1a = one_electron(&ao.h, &pa);
        let h1b = one_electron(&ao.h, &pb);
        let h1 = pb.s * h1a + pa.s * h1b;

        let h2aa = pb.s * two_electron_same(&ao.eri_asym, &pa);
        let h2bb = pa.s * two_electron_same(&ao.eri_asym, &pb);
        let h2ab = two_electron_diff(&ao.eri_coul, &pa, &pb);
        let h2 = h2aa + h2bb + h2ab;

        ((hnuc + h1 + h2), s)
    })
}

/// Calculate the Fock matrix element between determinants x and w using
/// generalised Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `fa`: NOCI Fock matrix spin alpha.
/// - `fb`: NOCI Fock matrix spin beta.
/// - `space`: Determinant space containing the bra and ket states.
/// - `tol`: Numerical tolerance for the determinant overlap.
/// # Returns:
/// - `T`: Fock matrix element between `ldet` and `gdet`.
pub(in crate::elements) fn calculate_f_pair_naive<T: NOCIScalar>(
    fa: &Array2<T>,
    fb: &Array2<T>,
    ao: &AoData,
    space: &NOCISpace<T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    tol: f64,
) -> T {
    time_call!(crate::timers::noci::add_calculate_f_pair_naive, {
        // Per spin occupid coefficients.
        let lp = space.parent(ldet);
        let gp = space.parent(gdet);
        let (loa, lob) = space.occupations(ldet);
        let (goa, gob) = space.occupations(gdet);

        let l_ca_occ = occ_coeffs(&lp.ca, loa);
        let g_ca_occ = occ_coeffs(&gp.ca, goa);
        let l_cb_occ = occ_coeffs(&lp.cb, lob);
        let g_cb_occ = occ_coeffs(&gp.cb, gob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

        pb.s * one_electron_scalar(fa, &pa) + pa.s * one_electron_scalar(fb, &pb)
    })
}

/// Calculate the shifted candidate-candidate matrix element using generalised
/// Slater-Condon rules.
/// # Arguments:
/// - `fock`: Spin-resolved Fock matrices in the AO basis.
/// - `data`: Authoritative NOCI space, AO overlap and numerical tolerance.
/// - `ldet`: State `a`.
/// - `gdet`: State `b`.
/// - `e0`: Zeroth-order energy shift.
/// # Returns:
/// - `T`: Shifted matrix element `M_{ab}`.
pub(in crate::elements) fn calculate_m_pair_naive<T: NOCIScalar>(
    fock: &FockData<'_, T>,
    data: &NOCIData<'_, T>,
    ldet: NOCIIndex,
    gdet: NOCIIndex,
    e0: f64,
) -> T {
    let space = data.space;
    let lp = space.parent(ldet);
    let gp = space.parent(gdet);
    let (loa, lob) = space.occupations(ldet);
    let (goa, gob) = space.occupations(gdet);

    let l_ca_occ = occ_coeffs(&lp.ca, loa);
    let g_ca_occ = occ_coeffs(&gp.ca, goa);
    let l_cb_occ = occ_coeffs(&lp.cb, lob);
    let g_cb_occ = occ_coeffs(&gp.cb, gob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    // Factorise the one-body matrix element over spin sectors, then apply
    // the reference shift:
    // `M_{ab} = F^\alpha_{ab}S^\beta_{ab} + S^\alpha_{ab}F^\beta_{ab} - E_0 S_{ab}`.
    let s = pa.s * pb.s;
    let f = pb.s * one_electron_scalar(fock.fa, &pa) + pa.s * one_electron_scalar(fock.fb, &pb);

    f - <T as From<f64>>::from(e0) * s
}
