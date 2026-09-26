// nocc/context.rs

// External crate imports.
use ndarray::{Array1, Array2, Array4};

// Crate-root imports.
use crate::AoData;
use crate::nocc::common::Tensors;
use crate::nocc::contract::PlanCache;
use crate::nocc::space::{Excitation, Spaces};
use crate::nocc::{Cumulants, RDM1};
use crate::scf::fock;

/// Reference data shared by every evaluation of a generated GNOCC term table.
pub(crate) struct EvaluationContext<'a> {
    /// Integrals in the NOCI natural-orbital basis.
    pub(crate) ao: &'a AoData,
    /// Generalised Fock matrix of the reference.
    pub(crate) fock: Array2<f64>,
    /// Spin-free one-particle RDM.
    pub(crate) gamma1: &'a RDM1<f64>,
    /// Spin-free active-space cumulants.
    pub(crate) lambdas: &'a Cumulants<f64>,
    /// Core, active, and virtual orbital-space maps.
    pub(crate) spaces: &'a Spaces,
    /// Raw spin-free excitation list defining the amplitude ordering.
    pub(crate) excitations: &'a [Excitation],
    /// Dense-block plans of the term tables evaluated in this run.
    pub(crate) plans: PlanCache,
}

/// Dense spin-free amplitude tensors of one cluster operator.
pub(crate) struct DenseAmplitudes {
    /// Singles amplitudes `t^q_p`, stored as `[q, p]`.
    pub(crate) t1: Array2<f64>,
    /// Pair-symmetric doubles amplitudes `\bar t^{rs}_{pq}`, stored as `[r, s, p, q]`.
    pub(crate) t2: Array4<f64>,
}

impl<'a> EvaluationContext<'a> {
    /// Build the evaluation context of one reference.
    /// # Arguments:
    /// - `ao`: Integrals in the NOCI natural-orbital basis.
    /// - `gamma1`: Spin-free one-particle RDM.
    /// - `lambdas`: Spin-free active-space cumulants.
    /// - `spaces`: Core, active, and virtual orbital-space maps.
    /// - `excitations`: Raw spin-free excitation list.
    /// - `max_cumulant`: Highest cumulant rank kept in every term table.
    /// # Returns:
    /// - `Self`: Evaluation context with the generalised Fock matrix.
    pub(crate) fn new(
        ao: &'a AoData,
        gamma1: &'a RDM1<f64>,
        lambdas: &'a Cumulants<f64>,
        spaces: &'a Spaces,
        excitations: &'a [Excitation],
        max_cumulant: usize,
    ) -> Self {
        Self {
            ao,
            fock: generalised_fock_matrix(ao, gamma1),
            gamma1,
            lambdas,
            spaces,
            excitations,
            plans: PlanCache::new(max_cumulant),
        }
    }

    /// Build the dense amplitude tensors of one amplitude vector.
    /// The cluster operator is `\hat T = \sum_\mu t_\mu \hat\tau_\mu` over the excitation list.
    /// The generated tables use a pair-symmetric `\bar t` with
    /// `\hat T_2 = \tfrac12 \sum \bar t^{rs}_{pq} \hat E^{pq}_{rs}` over all orbitals, and
    /// `\hat E^{qp}_{sr} = \hat E^{pq}_{rs}`, so each amplitude enters both `X` and its pair swap
    /// `PX`: `\bar t_X = \bar t_{PX} = t_X + t_{PX}`, where `t_{PX}` is zero when the swapped
    /// excitation is not in the list.
    /// # Arguments:
    /// - `amplitudes`: Cluster amplitude vector in the same order as the excitation list.
    /// # Returns:
    /// - `DenseAmplitudes`: Dense `t_1` and `\bar t_2` tensors.
    pub(crate) fn dense_amplitudes(
        &self,
        amplitudes: &Array1<f64>,
    ) -> DenseAmplitudes {
        let n = self.gamma1.n;
        let mut t1 = Array2::<f64>::zeros((n, n));
        let mut t2 = Array4::<f64>::zeros((n, n, n, n));

        for (nu, &ex) in self.excitations.iter().enumerate() {
            match ex {
                Excitation::Single { p, q } => {
                    t1[(q, p)] = amplitudes[nu];
                }
                Excitation::Double { p, q, r, s } => {
                    t2[(r, s, p, q)] += amplitudes[nu];
                    t2[(s, r, q, p)] += amplitudes[nu];
                }
            }
        }

        DenseAmplitudes { t1, t2 }
    }

    /// Return the runtime tensors needed by the generated-term evaluator.
    /// # Arguments:
    /// - `amplitudes`: Dense amplitude tensors, or `None` for amplitude-free tables.
    /// # Returns:
    /// - `Tensors<'_>`: Borrowed evaluator tensors.
    pub(super) fn tensors<'b>(
        &'b self,
        amplitudes: Option<&'b DenseAmplitudes>,
    ) -> Tensors<'b> {
        Tensors {
            ao: Some(self.ao),
            f: Some(&self.fock),
            spaces: self.spaces,
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
