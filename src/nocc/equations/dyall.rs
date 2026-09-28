// nocc/equations/dyall.rs
//! Zeroth-order Dyall quantities for the GNOCC micro-iterations.
//!
//! The amplitude update approximates the change in the residual through the connected
//! zeroth-order coupling `\langle\Phi|\hat\tau_\mu^\dagger\hat H_0\hat\tau_\nu|\Phi\rangle_c` of the
//! Dyall Hamiltonian, preconditioned by generalised Fock orbital-energy differences.
//!
//! # References
//!
//! - Dyall, *J. Chem. Phys.* **102**, 4909 (1995),
//!   [doi:10.1063/1.469539](https://doi.org/10.1063/1.469539).
//! - Lee and Tew, *Spin-free Generalised Normal Ordered Coupled Cluster*, arXiv:2507.13472
//!   (2025), Eqs. (58)-(62).

// External crate imports.
use ndarray::{Array1, Array2};

// Crate-root imports.
use crate::nocc::setup::ReferenceState;
use crate::nocc::space::{Excitation, Spaces};
use crate::nocc::terms::{TermEvaluator, assemble_matrix, dyall_blocks};

/// Build the symmetric zeroth-order coupling matrix
/// `A_{\mu\nu} = \langle\Phi|\hat\tau_\mu^\dagger\hat H_0\hat\tau_\nu|\Phi\rangle_c` in the raw
/// excitation basis. The coupling vanishes between classes with different numbers of core
/// holes or virtual particles, exactly as the metric does, so it shares the metric blocks and,
/// like the metric, is evaluated without cumulant truncation.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// # Returns:
/// - `Array2<f64>`: Coupling matrix `A_{\mu\nu}`.
pub(in crate::nocc) fn dyall_matrix(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
) -> Array2<f64> {
    assemble_matrix(
        spaces,
        excitations,
        evaluator,
        TermEvaluator::exact_table_plan,
        dyall_blocks(),
        &reference.tensors(spaces, None),
    )
}

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
