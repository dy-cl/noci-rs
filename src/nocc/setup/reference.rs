// nocc/setup/reference.rs
//! Generalised-normal-ordered reference state, its generalised Fock matrix and energy.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::AoData;
use crate::nocc::cumulants::Cumulants;
use crate::nocc::rdm::{RDM1, RDM2};
use crate::nocc::space::{DenseAmplitudes, Spaces};
use crate::nocc::terms::Tensors;
use crate::scf::fock;

/// Generalised-normal-ordered reference: its Hamiltonian integrals and reduced quantities.
/// Every generated table depends on the reference only through these tensors.
pub(crate) struct ReferenceState<'a> {
    /// Integrals in the NOCI natural-orbital basis.
    pub(in crate::nocc) ao: &'a AoData,
    /// Generalised Fock matrix of the reference.
    pub(in crate::nocc) fock: Array2<f64>,
    /// Spin-free one-particle RDM.
    pub(in crate::nocc) gamma1: &'a RDM1<f64>,
    /// Spin-free active-space cumulants.
    pub(in crate::nocc) lambdas: &'a Cumulants<f64>,
}

impl<'a> ReferenceState<'a> {
    /// Build the reference state and its generalised Fock matrix.
    /// # Arguments:
    /// - `ao`: Integrals in the NOCI natural-orbital basis.
    /// - `gamma1`: Spin-free one-particle RDM.
    /// - `lambdas`: Spin-free active-space cumulants.
    /// # Returns:
    /// - `Self`: Reference state.
    pub(crate) fn new(
        ao: &'a AoData,
        gamma1: &'a RDM1<f64>,
        lambdas: &'a Cumulants<f64>,
    ) -> Self {
        Self {
            ao,
            fock: generalised_fock_matrix(ao, gamma1),
            gamma1,
            lambdas,
        }
    }

    /// Return the runtime tensors needed by the generated-term evaluator.
    /// # Arguments:
    /// - `spaces`: Core, active, and virtual orbital-space maps.
    /// - `amplitudes`: Dense amplitude tensors, or `None` for amplitude-free tables.
    /// # Returns:
    /// - `Tensors<'b>`: Borrowed evaluator tensors.
    pub(in crate::nocc) fn tensors<'b>(
        &'b self,
        spaces: &'b Spaces,
        amplitudes: Option<&'b DenseAmplitudes>,
    ) -> Tensors<'b> {
        Tensors {
            ao: self.ao,
            f: &self.fock,
            spaces,
            gamma1: self.gamma1,
            lambdas: self.lambdas,
            amplitudes,
        }
    }
}

/// Build the spin-free generalised Fock matrix of the reference,
/// `f^q_p = h^q_p + \sum_{rs}(g^{qs}_{pr} - \tfrac12 g^{qs}_{rp})\Gamma^r_s`.
/// # Arguments:
/// - `ao`: Integrals in the NOCI natural-orbital basis.
/// - `gamma1`: Spin-free one-particle RDM.
/// # Returns:
/// - `Array2<f64>`: Generalised Fock matrix.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eq. (18).
fn generalised_fock_matrix(
    ao: &AoData,
    gamma1: &RDM1<f64>,
) -> Array2<f64> {
    // Split the spin-free density equally between `\alpha` and `\beta`.
    let n = gamma1.n;
    let mut da = Array2::<f64>::zeros((n, n));
    let mut db = Array2::<f64>::zeros((n, n));

    for p in 0..n {
        for q in 0..n {
            let value = 0.5 * gamma1.data[p * n + q];
            da[(p, q)] = value;
            db[(p, q)] = value;
        }
    }

    fock(&ao.h, &ao.eri_coul, &da, &db).0
}

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
