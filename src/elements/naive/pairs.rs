// elements/naive/pairs.rs
//! Determinant-pair matrix elements from the generalised Slater–Condon rules.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::ExcitedState;
use crate::elements::{FockData, NOCIData};

// Parent/sibling imports.
use super::rules::{
    build_s_pair, occ_coeffs, one_electron, one_electron_scalar, two_electron_diff,
    two_electron_same,
};

/// Occupied MO coefficients of both spins of one prepared determinant in its parent frame.
/// # Arguments:
/// - `data`: Shared data holding the parent orbital frames.
/// - `det`: Prepared determinant.
/// # Returns:
/// - `(Array2<T>, Array2<T>)`: Alpha and beta occupied MO coefficients.
fn occupied_coefficients<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    det: &ExcitedState,
) -> (Array2<T>, Array2<T>) {
    let parent = &data.space.parents()[det.det.parent];
    (
        occ_coeffs(&parent.ca, det.det.alpha),
        occ_coeffs(&parent.cb, det.det.beta),
    )
}

/// Calculate the overlap matrix element between two prepared determinants using the generalised
/// Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `bra`: Prepared bra determinant x.
/// - `ket`: Prepared ket determinant w.
/// # Returns:
/// - `T`: Overlap matrix element between `bra` and `ket`.
pub(in crate::elements) fn calculate_s_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    bra: &ExcitedState,
    ket: &ExcitedState,
) -> T {
    let (l_ca_occ, l_cb_occ) = occupied_coefficients(data, bra);
    let (g_ca_occ, g_cb_occ) = occupied_coefficients(data, ket);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    let det_phase = T::from_real(bra.reduced.phase * ket.reduced.phase);
    det_phase * pa.s * pb.s
}

/// Calculate the Hamiltonian and overlap matrix elements between two prepared determinants using
/// the generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `bra`: Prepared bra determinant x.
/// - `ket`: Prepared ket determinant w.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between `bra` and `ket`.
pub(in crate::elements) fn calculate_hs_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    bra: &ExcitedState,
    ket: &ExcitedState,
) -> (T, T) {
    let ao = data.ao;
    let (l_ca_occ, l_cb_occ) = occupied_coefficients(data, bra);
    let (g_ca_occ, g_cb_occ) = occupied_coefficients(data, ket);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, data.tol);

    // Overlap matrix element for this pair.
    let s = pa.s * pb.s;

    // Nuclear repulsion contributes only when neither spin has a zero-overlap orbital pair.
    let hnuc = match (pa.zeros.len(), pb.zeros.len()) {
        (0, 0) => T::from_real(ao.enuc) * s,
        _ => T::from_real(0.0),
    };

    // `H_1 = S^\beta h^\alpha + S^\alpha h^\beta` from the one-electron Hamiltonian.
    let h1a = one_electron(&ao.h, &pa);
    let h1b = one_electron(&ao.h, &pb);
    let h1 = pb.s * h1a + pa.s * h1b;

    // Same-spin terms carry the other spin's overlap; the different-spin term couples both.
    let h2aa = pb.s * two_electron_same(&ao.eri_asym, &pa);
    let h2bb = pa.s * two_electron_same(&ao.eri_asym, &pb);
    let h2ab = two_electron_diff(&ao.eri_coul, &pa, &pb);
    let h2 = h2aa + h2bb + h2ab;

    ((hnuc + h1 + h2), s)
}

/// Calculate the Fock and overlap matrix elements between two prepared determinants using the
/// generalised Slater-Condon rules,
/// `F_{xw} = F^\alpha_{xw} S^\beta_{xw} + S^\alpha_{xw} F^\beta_{xw}` and
/// `S_{xw} = S^\alpha_{xw} S^\beta_{xw}`.
/// # Arguments:
/// - `fock`: Spin-resolved Fock matrices in the AO basis.
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `bra`: Prepared bra determinant x.
/// - `ket`: Prepared ket determinant w.
/// # Returns:
/// - `(T, T)`: Fock and overlap matrix elements between `bra` and `ket`.
pub(in crate::elements) fn calculate_fs_pair_naive<T: NOCIScalar>(
    fock: &FockData<'_, T>,
    data: &NOCIData<'_, T>,
    bra: &ExcitedState,
    ket: &ExcitedState,
) -> (T, T) {
    let (l_ca_occ, l_cb_occ) = occupied_coefficients(data, bra);
    let (g_ca_occ, g_cb_occ) = occupied_coefficients(data, ket);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    let f = pb.s * one_electron_scalar(fock.fa, &pa) + pa.s * one_electron_scalar(fock.fb, &pb);
    (f, pa.s * pb.s)
}
