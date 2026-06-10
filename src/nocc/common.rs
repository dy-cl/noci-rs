// nocc/common.rs

use crate::nocc::RDM1;
use crate::nocc::space::{Excitation, ExcitationClass, Spaces};

/// Evaluate a Kronecker delta.
/// # Arguments:
/// - `p`: Left orbital index.
/// - `q`: Right orbital index.
/// # Returns:
/// - `f64`: `1.0` if the indices are equal, otherwise `0.0`.
pub(super) fn delta(
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
pub(super) fn theta(
    gamma1: &RDM1<f64>,
    p: usize,
    q: usize,
) -> f64 {
    2.0 * delta(p, q) - gamma1.data[p * gamma1.n + q]
}

/// Convert a global orbital index to an active-space index.
/// # Arguments:
/// - `spaces`: Orbital-space partitioning and index maps.
/// - `p`: Global orbital index.
/// # Returns:
/// - `usize`: Active-space index corresponding to `p`.
pub(super) fn active(
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
pub(super) fn orbs(
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
/// - `Vec<usize>`: Active-space tensor indices.
pub(super) fn ids(
    spaces: &Spaces,
    raw: &[u16],
    idx: &[usize],
) -> Vec<usize> {
    raw.iter()
        .map(|&id| active(spaces, idx[id as usize]))
        .collect()
}

/// Return excitation indices in generated free-index order.
/// # Arguments:
/// - `ex`: Raw spin-free excitation.
/// # Returns:
/// - `Vec<usize>`: Creation indices followed by annihilation indices.
pub(super) fn unpack(ex: Excitation) -> Vec<usize> {
    match ex {
        Excitation::Single { p, q } => vec![p, q],
        Excitation::Double { p, q, r, s } => vec![p, q, r, s],
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
    }
}
