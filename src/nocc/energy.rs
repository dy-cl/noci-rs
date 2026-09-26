// nocc/energy.rs

// Crate-root imports.
use crate::AoData;
use crate::nocc::contract::{FactorBlocks, TermEvaluator, evaluate_dense_table};
use crate::nocc::loader::{e1_terms, e2_terms};
use crate::nocc::reference::ReferenceState;
use crate::nocc::space::{DenseAmplitudes, ExcitationManifold};
use crate::nocc::{RDM1, RDM2};

/// Evaluate the reference energy from the one- and two-body RDMs,
/// `E_0 = E_{\text{nuc}} + \sum_{ab} h_{ab}\Gamma_{1,ba} + \tfrac12\sum_{abcd}(ab|cd)\Gamma_{2,bcad}`.
/// # Arguments:
/// - `ao`: Integrals in the NOCI natural-orbital basis.
/// - `gamma1`: Spin-free one-particle RDM.
/// - `gamma2`: Spin-free two-particle RDM.
/// # Returns:
/// - `f64`: Reference energy `\langle\Phi|\hat H|\Phi\rangle`.
pub(crate) fn reference_energy(
    ao: &AoData,
    gamma1: &RDM1<f64>,
    gamma2: &RDM2<f64>,
) -> f64 {
    let n1 = gamma1.n;
    let mut e1 = 0.0;
    for a in 0..n1 {
        for b in 0..n1 {
            e1 += ao.h[(a, b)] * gamma1.data[b * n1 + a];
        }
    }

    let n = gamma2.n;
    let mut e2 = 0.0;
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    let i = (((b * n + c) * n + a) * n) + d;
                    e2 += ao.eri_coul[(a, b, c, d)] * gamma2.data[i];
                }
            }
        }
    }

    ao.enuc + e1 + 0.5 * e2
}

/// Evaluate the correlation energy
/// `E - E_0 = \langle\Phi|\hat H\hat T|\Phi\rangle_c + \tfrac12\langle\Phi|\hat H\{\hat T\hat T\}|\Phi\rangle_c`.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `manifold`: Orbital spaces and raw excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `amplitudes`: Dense amplitude tensors of the current cluster operator.
/// # Returns:
/// - `f64`: Correlation energy.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eq. (35).
pub(crate) fn correlation_energy(
    reference: &ReferenceState<'_>,
    manifold: &ExcitationManifold<'_>,
    evaluator: &TermEvaluator,
    amplitudes: &DenseAmplitudes,
) -> f64 {
    let tensors = reference.tensors(manifold.spaces, Some(amplitudes));
    let tables = [e1_terms(), e2_terms()].map(|t| (t.terms.as_slice(), t.indices.as_slice()));
    let plans = tables.map(|t| evaluator.table_plan(t));
    let blocks = FactorBlocks::build_factor_blocks(
        &plans.iter().map(|p| p.as_ref()).collect::<Vec<_>>(),
        &tensors,
    );

    tables
        .iter()
        .zip(&plans)
        .map(|(&t, plan)| evaluate_dense_table(t, &[], plan, &blocks).data[0])
        .sum()
}
