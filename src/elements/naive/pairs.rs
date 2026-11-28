// elements/naive/pairs.rs
//! Determinant-pair matrix elements from the generalised Slater–Condon rules.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::DeterminantState;
use crate::elements::{DetPair, FockData, NOCIData, SpinTransition};

// Parent/sibling imports.
use super::rules::{
    build_s_pair, occ_coeffs, one_electron, one_electron_scalar, pair_density, two_electron_diff,
    two_electron_same,
};

/// Occupied MO coefficients of both spins of one determinant in its parent frame.
/// # Arguments:
/// - `data`: Shared data holding the parent orbital frames.
/// - `det`: Determinant.
/// # Returns:
/// - `(Array2<T>, Array2<T>)`: Alpha and beta occupied MO coefficients.
pub(super) fn occupied_coefficients<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    det: &DeterminantState,
) -> (Array2<T>, Array2<T>) {
    let parent = &data.space.parents()[det.parent];
    (
        occ_coeffs(&parent.ca, det.alpha),
        occ_coeffs(&parent.cb, det.beta),
    )
}

/// Calculate the overlap matrix element between two determinants using the generalised
/// Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `bra`: Bra determinant x.
/// - `ket`: Ket determinant w.
/// # Returns:
/// - `T`: Overlap matrix element between `bra` and `ket`.
pub(in crate::elements) fn calculate_s_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    bra: &DeterminantState,
    ket: &DeterminantState,
) -> T {
    let (l_ca_occ, l_cb_occ) = occupied_coefficients(data, bra);
    let (g_ca_occ, g_cb_occ) = occupied_coefficients(data, ket);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    // The occupied coefficients are in ascending orbital order, the determinant convention of
    // every evaluator, so no excitation phase applies.
    pa.s * pb.s
}

/// Calculate the Hamiltonian and overlap matrix elements between two determinants using
/// the generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `bra`: Bra determinant x.
/// - `ket`: Ket determinant w.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between `bra` and `ket`.
pub(in crate::elements) fn calculate_hs_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    bra: &DeterminantState,
    ket: &DeterminantState,
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

/// Calculate the Fock and overlap matrix elements between two determinants using the
/// generalised Slater-Condon rules,
/// `F_{xw} = F^\alpha_{xw} S^\beta_{xw} + S^\alpha_{xw} F^\beta_{xw}` and
/// `S_{xw} = S^\alpha_{xw} S^\beta_{xw}`.
/// # Arguments:
/// - `fock`: Spin-resolved Fock matrices in the AO basis.
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `bra`: Bra determinant x.
/// - `ket`: Ket determinant w.
/// # Returns:
/// - `(T, T)`: Fock and overlap matrix elements between `bra` and `ket`.
pub(in crate::elements) fn calculate_fs_pair_naive<T: NOCIScalar>(
    fock: &FockData<'_, T>,
    data: &NOCIData<'_, T>,
    bra: &DeterminantState,
    ket: &DeterminantState,
) -> (T, T) {
    let (l_ca_occ, l_cb_occ) = occupied_coefficients(data, bra);
    let (g_ca_occ, g_cb_occ) = occupied_coefficients(data, ket);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    let f = pb.s * one_electron_scalar(fock.fa, &pa) + pa.s * one_electron_scalar(fock.fb, &pb);
    (f, pa.s * pb.s)
}

/// Calculate the spin overlaps and covariant AO transition densities of two determinants using
/// the generalised Slater-Condon rules, for pairs of different parents without Wick intermediates.
/// The contravariant pair densities `W^\sigma` are lowered with the AO overlap,
/// `\tilde\Gamma^\sigma = \mathbf S W^\sigma \mathbf S`.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Bra and ket determinants.
/// # Returns:
/// - `SpinTransition<T>`: Spin overlaps and covariant transition densities.
pub(in crate::elements) fn calculate_spin_transition_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
) -> SpinTransition<T> {
    let nao = data.ao.h.nrows();
    let s = data.ao.s.mapv(<T as From<f64>>::from);

    let (l_ca_occ, l_cb_occ) = occupied_coefficients(data, &pair.bra);
    let (g_ca_occ, g_cb_occ) = occupied_coefficients(data, &pair.ket);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    // `\tilde\Gamma^\sigma = \mathbf S W^\sigma \mathbf S` from each spin's pair density.
    let ga = s.dot(&pair_density(&pa, nao)).dot(&s);
    let gb = s.dot(&pair_density(&pb, nao)).dot(&s);

    SpinTransition {
        s: (pa.s, pb.s),
        gamma: (Some(ga), Some(gb)),
    }
}
