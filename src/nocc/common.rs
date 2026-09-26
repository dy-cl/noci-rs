// nocc/common.rs

// External crate imports.
use ndarray::{Array2, Array4};

// Crate-root imports.
use crate::AoData;
use crate::nocc::space::{Excitation, ExcitationClass, Spaces};
use crate::nocc::terms::TensorFactor;
use crate::nocc::{Cumulants, RDM1};

/// Reference tensors needed to evaluate generated term tables.
pub(super) struct Tensors<'a> {
    /// Ao integrals.
    pub(super) ao: Option<&'a AoData>,
    /// Spin-free Fock matrix.
    pub(super) f: Option<&'a Array2<f64>>,
    /// Core, active, and virtual orbital-space maps.
    pub(super) spaces: &'a Spaces,
    /// Spin-free one-particle RDM.
    pub(super) gamma1: &'a RDM1<f64>,
    /// Spin-free active-space cumulants.
    pub(super) lambdas: &'a Cumulants<f64>,
    /// Spin-free single-excitation amplitude tensor.
    pub(super) t1: Option<&'a Array2<f64>>,
    /// Spin-free double-excitation amplitude tensor.
    pub(super) t2: Option<&'a Array4<f64>>,
}

/// Evaluate a Kronecker delta.
/// # Arguments:
/// - `p`: Left orbital index.
/// - `q`: Right orbital index.
/// # Returns:
/// - `f64`: `1.0` if the indices are equal, otherwise `0.0`.
pub(super) fn kronecker_delta(
    p: usize,
    q: usize,
) -> f64 {
    if p == q { 1.0 } else { 0.0 }
}

/// Evaluate an active-space hole density.
/// # Arguments:
/// - `gamma1`: Spin-free one-particle RDM.
/// - `p`: Upper orbital index.
/// - `q`: Lower orbital index.
/// # Returns:
/// - `f64`: `Theta^p_q = 2 delta^p_q - Gamma^p_q`.
pub(super) fn hole_density(
    gamma1: &RDM1<f64>,
    p: usize,
    q: usize,
) -> f64 {
    2.0 * kronecker_delta(p, q) - gamma1.data[p * gamma1.n + q]
}

/// Convert a global orbital index to an active-space index.
/// # Arguments:
/// - `spaces`: Orbital-space partitioning and index maps.
/// - `p`: Global orbital index.
/// # Returns:
/// - `usize`: Active-space index corresponding to `p`.
pub(super) fn active_index(
    spaces: &Spaces,
    p: usize,
) -> usize {
    spaces.active_map[p].expect("expected active orbital index")
}

/// Return orbitals belonging to a generated orbital-space id.
/// # Arguments:
/// - `spaces`: Orbital-space partitioning and index maps.
/// - `kind`: Generated orbital-space id.
/// # Returns:
/// - `&[usize]`: Orbital indices in the requested space.
/// # Panics
/// - Panics if `kind` does not identify a known orbital space.
pub(super) fn space_orbitals(
    spaces: &Spaces,
    kind: u8,
) -> &[usize] {
    match kind {
        0 => &spaces.core,
        1 => &spaces.active,
        2 => &spaces.virtuals,
        _ => panic!("unknown orbital space kind {kind}"),
    }
}

/// Convert class-local active indices to active-space tensor indices.
/// # Arguments:
/// - `spaces`: Orbital-space partitioning and index maps.
/// - `raw`: Class-local index ids.
/// - `idx`: Class-local orbital index values.
/// # Returns:
/// - `([usize; 4], usize)`: Active-space tensor indices and active length.
fn active_indices(
    spaces: &Spaces,
    raw: &[u16],
    idx: &[usize],
) -> ([usize; 4], usize) {
    let mut out = [0; 4];
    for (i, &id) in raw.iter().enumerate() {
        out[i] = active_index(spaces, idx[id as usize]);
    }
    (out, raw.len())
}

/// Return excitation indices in generated free-index order.
/// # Arguments:
/// - `ex`: Raw spin-free excitation.
/// # Returns:
/// - `([usize; 4], usize)`: Creation indices followed by annihilation indices and active length.
pub(super) fn excitation_indices(ex: Excitation) -> ([usize; 4], usize) {
    match ex {
        Excitation::Single { p, q } => ([p, q, 0, 0], 2),
        Excitation::Double { p, q, r, s } => ([p, q, r, s], 4),
    }
}

/// Return the generated term-table name for an excitation class.
/// # Arguments:
/// - `class`: Excitation class.
/// # Returns:
/// - `&'static str`: Generated excitation class name.
pub(super) fn class_name(class: ExcitationClass) -> &'static str {
    match class {
        ExcitationClass::CToA => "CToA",
        ExcitationClass::CToV => "CToV",
        ExcitationClass::AToA => "AToA",
        ExcitationClass::AToV => "AToV",
        ExcitationClass::CCToAA => "CCToAA",
        ExcitationClass::CCToAV => "CCToAV",
        ExcitationClass::CAToAA => "CAToAA",
        ExcitationClass::CAToAV => "CAToAV",
        ExcitationClass::CAToVA => "CAToVA",
        ExcitationClass::CAToVV => "CAToVV",
        ExcitationClass::AAToAA => "AAToAA",
        ExcitationClass::AAToAV => "AAToAV",
        ExcitationClass::AAToVV => "AAToVV",
        ExcitationClass::CCToVV => "CCToVV",
    }
}

/// Evaluate one generated tensor factor.
/// # Arguments:
/// - `tensor`: Generated tensor factor.
/// - `idx`: Local orbital index values.
/// - `tensors`: Runtime tensors needed by the generated-term evaluator.
/// # Returns:
/// - `f64`: Tensor element.
/// # Panics
/// - Panics if the tensor kind is unknown or a required runtime tensor is absent.
pub(super) fn evaluate_factor(
    tensor: &TensorFactor,
    idx: &[usize],
    tensors: &Tensors<'_>,
) -> f64 {
    let upper = &tensor.1;
    let lower = &tensor.2;

    // First element of tensor dictates what type it is.
    match tensor.0 {
        // Gamma_{i_l}^{i_u}.
        0 => {
            tensors.gamma1.data[idx[upper[0] as usize] * tensors.gamma1.n + idx[lower[0] as usize]]
        }
        // Theta_{i_l}^{i_u}.
        1 => hole_density(
            tensors.gamma1,
            idx[upper[0] as usize],
            idx[lower[0] as usize],
        ),
        // f_{i_l}^{i_u}.
        2 => tensors.f.unwrap()[(idx[upper[0] as usize], idx[lower[0] as usize])],
        // g_{i_l_1, i_l_2}^{i_u_1, i_u_2} = (i_u_1 i_l_1 | i_u_2 i_l_2), columns share an
        // electron and the integrals are stored in chemists' order.
        3 => {
            tensors.ao.unwrap().eri_coul[(
                idx[upper[0] as usize],
                idx[lower[0] as usize],
                idx[upper[1] as usize],
                idx[lower[1] as usize],
            )]
        }
        // Lambda_{i_l_1, i_l_2}^{i_u_1, i_u_2}.
        4 => {
            let (u, nu) = active_indices(tensors.spaces, upper, idx);
            let (l, nl) = active_indices(tensors.spaces, lower, idx);

            tensors.lambdas.lambda2.get(&u[..nu], &l[..nl])
        }
        // Lambda_{i_l_1, i_l_2, i_l_3}^{i_u_1, i_u_2, i_u_3}.
        5 => {
            let (u, nu) = active_indices(tensors.spaces, upper, idx);
            let (l, nl) = active_indices(tensors.spaces, lower, idx);

            tensors.lambdas.lambda3.get(&u[..nu], &l[..nl])
        }
        // Lambda_{i_l_1, i_l_2, i_l_3, i_l_4}^{i_u_1, i_u_2, i_u_3, i_u_4}.
        6 => {
            let (u, nu) = active_indices(tensors.spaces, upper, idx);
            let (l, nl) = active_indices(tensors.spaces, lower, idx);

            tensors.lambdas.lambda4.get(&u[..nu], &l[..nl])
        }
        // t_{i_l}^{i_u}.
        8 => tensors.t1.unwrap()[(idx[upper[0] as usize], idx[lower[0] as usize])],
        // t_{i_l_1, i_l_2}^{i_u_1, i_u_2}.
        9 => tensors.t2.unwrap()[(
            idx[upper[0] as usize],
            idx[upper[1] as usize],
            idx[lower[0] as usize],
            idx[lower[1] as usize],
        )],
        _ => panic!("unknown tensor kind {}", tensor.0),
    }
}
