// nocc/equations/denominators.rs
//! Orbital-energy denominators that precondition the Newton update.

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::nocc::setup::ReferenceState;
use crate::nocc::space::Excitation;

/// Build the orbital-energy denominators of every excitation,
/// `\Delta^{pq}_{rs} = f^p_p + f^q_q - f^r_r - f^s_s` for `\hat E^{pq}_{rs}` and
/// `\Delta^p_q = f^p_p - f^q_q` for `\hat E^p_q`, from the generalised Fock diagonal.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `excitations`: Raw spin-free excitation list.
/// # Returns:
/// - `Array1<f64>`: One denominator per excitation.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eq. (61).
pub(in crate::nocc) fn orbital_denominators(
    reference: &ReferenceState<'_>,
    excitations: &[Excitation],
) -> Array1<f64> {
    let f = |p: usize| reference.fock[(p, p)];

    excitations
        .iter()
        .map(|&ex| match ex {
            Excitation::Single { p, q } => f(p) - f(q),
            Excitation::Double { p, q, r, s } => f(p) + f(q) - f(r) - f(s),
        })
        .collect()
}
