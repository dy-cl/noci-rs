// nocc/reference.rs

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::AoData;
use crate::nocc::common::Tensors;
use crate::nocc::space::{DenseAmplitudes, Spaces};
use crate::nocc::{Cumulants, RDM1};
use crate::scf::fock;

/// Generalised-normal-ordered reference: its Hamiltonian integrals and reduced quantities.
/// Every generated table depends on the reference only through these tensors.
pub(crate) struct ReferenceState<'a> {
    /// Integrals in the NOCI natural-orbital basis.
    pub(crate) ao: &'a AoData,
    /// Generalised Fock matrix of the reference.
    pub(crate) fock: Array2<f64>,
    /// Spin-free one-particle RDM.
    pub(crate) gamma1: &'a RDM1<f64>,
    /// Spin-free active-space cumulants.
    pub(crate) lambdas: &'a Cumulants<f64>,
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
    pub(super) fn tensors<'b>(
        &'b self,
        spaces: &'b Spaces,
        amplitudes: Option<&'b DenseAmplitudes>,
    ) -> Tensors<'b> {
        Tensors {
            ao: Some(self.ao),
            f: Some(&self.fock),
            spaces,
            gamma1: self.gamma1,
            lambdas: self.lambdas,
            t1: amplitudes.map(|x| &x.t1),
            t2: amplitudes.map(|x| &x.t2),
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
pub(crate) fn generalised_fock_matrix(
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
