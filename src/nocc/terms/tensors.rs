// nocc/terms/tensors.rs
//! Runtime tensors of the generated tables and their element-wise evaluation.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::AoData;
use crate::nocc::cumulants::Cumulants;
use crate::nocc::rdm::RDM1;
use crate::nocc::space::{DenseAmplitudes, Excitation, Spaces};

/// Message of the panic when an amplitude-free table requests an amplitude tensor.
const AMPLITUDES: &str = "amplitude tensor requested by an amplitude-free table";

/// Tensor kind of a generated factor, with the ids used by the generated term tables.
/// Variants are declared in id order, so the derived ordering matches the ids.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum TensorKind {
    /// One-particle RDM `\Gamma^p_q`.
    Gamma1 = 0,
    /// Hole density `\Theta^p_q = 2\delta^p_q - \Gamma^p_q`.
    Theta = 1,
    /// Generalised Fock matrix `f^p_q`.
    Fock = 2,
    /// Two-electron integrals `g^{pq}_{rs}`.
    Eri = 3,
    /// Two-body cumulant `\Lambda_2`.
    Lambda2 = 4,
    /// Three-body cumulant `\Lambda_3`.
    Lambda3 = 5,
    /// Four-body cumulant `\Lambda_4`.
    Lambda4 = 6,
    /// Single-excitation amplitudes `t^p_q`.
    T1 = 8,
    /// Double-excitation amplitudes `t^{pq}_{rs}`.
    T2 = 9,
}

impl TensorKind {
    /// Every tensor kind, in id order.
    pub(super) const ALL: [Self; 9] = [
        Self::Gamma1,
        Self::Theta,
        Self::Fock,
        Self::Eri,
        Self::Lambda2,
        Self::Lambda3,
        Self::Lambda4,
        Self::T1,
        Self::T2,
    ];

    /// Tensor kind of every id, so generated ids convert in constant time.
    const BY_ID: [Option<Self>; 10] = {
        let mut table = [None; 10];
        let mut k = 0;
        while k < Self::ALL.len() {
            table[Self::ALL[k] as usize] = Some(Self::ALL[k]);
            k += 1;
        }
        table
    };

    /// Return the generated name of the tensor kind.
    /// # Arguments:
    /// - `self`: Tensor kind.
    /// # Returns:
    /// - `&'static str`: Name used by the generated term tables.
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Gamma1 => "Gamma1",
            Self::Theta => "Theta",
            Self::Fock => "f",
            Self::Eri => "g",
            Self::Lambda2 => "Lambda2",
            Self::Lambda3 => "Lambda3",
            Self::Lambda4 => "Lambda4",
            Self::T1 => "t1",
            Self::T2 => "t2",
        }
    }

    /// Convert a generated tensor-kind id.
    /// # Arguments:
    /// - `id`: Tensor-kind id from a generated term table.
    /// # Returns:
    /// - `Self`: Tensor kind.
    /// # Panics
    /// - Panics if `id` does not identify a known tensor kind.
    pub(super) fn from_id(id: u8) -> Self {
        Self::BY_ID
            .get(id as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("unknown tensor kind {id}"))
    }

    /// Return the rank `k` of a cumulant `\Lambda_k`, or zero for any other tensor.
    /// # Arguments:
    /// - `self`: Tensor kind.
    /// # Returns:
    /// - `usize`: Cumulant rank.
    pub(super) fn cumulant_rank(self) -> usize {
        match self {
            Self::Lambda2 => 2,
            Self::Lambda3 => 3,
            Self::Lambda4 => 4,
            _ => 0,
        }
    }
}

/// Orbital space of a generated index, with the ids used by the generated term tables.
/// Variants are declared in id order, so the derived ordering matches the ids.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum SpaceKind {
    /// Core orbitals.
    Core = 0,
    /// Active orbitals.
    Active = 1,
    /// Virtual orbitals.
    Virtual = 2,
}

impl SpaceKind {
    /// Every orbital space, in id order.
    pub(super) const ALL: [Self; 3] = [Self::Core, Self::Active, Self::Virtual];

    /// Orbital space of every id, so generated ids convert in constant time.
    const BY_ID: [Option<Self>; 3] = {
        let mut table = [None; 3];
        let mut k = 0;
        while k < Self::ALL.len() {
            table[Self::ALL[k] as usize] = Some(Self::ALL[k]);
            k += 1;
        }
        table
    };

    /// Return the generated name of the orbital space.
    /// # Arguments:
    /// - `self`: Orbital space.
    /// # Returns:
    /// - `&'static str`: Name used by the generated term tables.
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Active => "active",
            Self::Virtual => "virtual",
        }
    }

    /// Convert a generated orbital-space id.
    /// # Arguments:
    /// - `id`: Orbital-space id from a generated term table.
    /// # Returns:
    /// - `Self`: Orbital space.
    /// # Panics
    /// - Panics if `id` does not identify a known orbital space.
    pub(super) fn from_id(id: u8) -> Self {
        Self::BY_ID
            .get(id as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("unknown orbital space kind {id}"))
    }
}

/// Reference tensors needed to evaluate generated term tables.
pub(in crate::nocc) struct Tensors<'a> {
    /// Ao integrals.
    pub(in crate::nocc) ao: &'a AoData,
    /// Spin-free Fock matrix.
    pub(in crate::nocc) f: &'a Array2<f64>,
    /// Core, active, and virtual orbital-space maps.
    pub(in crate::nocc) spaces: &'a Spaces,
    /// Spin-free one-particle RDM.
    pub(in crate::nocc) gamma1: &'a RDM1<f64>,
    /// Spin-free active-space cumulants.
    pub(in crate::nocc) lambdas: &'a Cumulants<f64>,
    /// Dense amplitude tensors, or `None` for amplitude-free tables.
    pub(in crate::nocc) amplitudes: Option<&'a DenseAmplitudes>,
}

/// Evaluate a Kronecker delta.
/// # Arguments:
/// - `p`: Left orbital index.
/// - `q`: Right orbital index.
/// # Returns:
/// - `f64`: `1.0` if the indices are equal, otherwise `0.0`.
fn kronecker_delta(
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
fn hole_density(
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
fn active_index(
    spaces: &Spaces,
    p: usize,
) -> usize {
    spaces.active_map[p].expect("expected active orbital index")
}

/// Return the orbitals of one orbital space.
/// # Arguments:
/// - `spaces`: Orbital-space partitioning and index maps.
/// - `kind`: Orbital space.
/// # Returns:
/// - `&[usize]`: Orbital indices in the requested space.
pub(super) fn space_orbitals(
    spaces: &Spaces,
    kind: SpaceKind,
) -> &[usize] {
    match kind {
        SpaceKind::Core => &spaces.core,
        SpaceKind::Active => &spaces.active,
        SpaceKind::Virtual => &spaces.virtuals,
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

/// Evaluate one generated tensor factor.
/// # Arguments:
/// - `kind`: Tensor kind of the factor.
/// - `slots`: Upper then lower class-local index ids of the factor.
/// - `idx`: Local orbital index values.
/// - `tensors`: Runtime tensors needed by the generated-term evaluator.
/// # Returns:
/// - `f64`: Tensor element.
/// # Panics
/// - Panics if an amplitude tensor is requested from a table evaluated without amplitudes.
pub(super) fn evaluate_factor(
    kind: TensorKind,
    slots: (&[u16], &[u16]),
    idx: &[usize],
    tensors: &Tensors<'_>,
) -> f64 {
    let (upper, lower) = slots;

    match kind {
        // Gamma_{i_l}^{i_u}.
        TensorKind::Gamma1 => {
            tensors.gamma1.data[idx[upper[0] as usize] * tensors.gamma1.n + idx[lower[0] as usize]]
        }
        // Theta_{i_l}^{i_u}.
        TensorKind::Theta => hole_density(
            tensors.gamma1,
            idx[upper[0] as usize],
            idx[lower[0] as usize],
        ),
        // f_{i_l}^{i_u}.
        TensorKind::Fock => tensors.f[(idx[upper[0] as usize], idx[lower[0] as usize])],
        // g_{i_l_1, i_l_2}^{i_u_1, i_u_2} = (i_u_1 i_l_1 | i_u_2 i_l_2), columns share an
        // electron and the integrals are stored in chemists' order.
        TensorKind::Eri => {
            tensors.ao.eri_coul[(
                idx[upper[0] as usize],
                idx[lower[0] as usize],
                idx[upper[1] as usize],
                idx[lower[1] as usize],
            )]
        }
        // Lambda_{i_l_1, i_l_2}^{i_u_1, i_u_2}.
        TensorKind::Lambda2 => {
            let (u, nu) = active_indices(tensors.spaces, upper, idx);
            let (l, nl) = active_indices(tensors.spaces, lower, idx);

            tensors.lambdas.lambda2.get(&u[..nu], &l[..nl])
        }
        // Lambda_{i_l_1, i_l_2, i_l_3}^{i_u_1, i_u_2, i_u_3}.
        TensorKind::Lambda3 => {
            let (u, nu) = active_indices(tensors.spaces, upper, idx);
            let (l, nl) = active_indices(tensors.spaces, lower, idx);

            tensors.lambdas.lambda3.get(&u[..nu], &l[..nl])
        }
        // Lambda_{i_l_1, i_l_2, i_l_3, i_l_4}^{i_u_1, i_u_2, i_u_3, i_u_4}.
        TensorKind::Lambda4 => {
            let (u, nu) = active_indices(tensors.spaces, upper, idx);
            let (l, nl) = active_indices(tensors.spaces, lower, idx);

            tensors.lambdas.lambda4.get(&u[..nu], &l[..nl])
        }
        // t_{i_l}^{i_u}.
        TensorKind::T1 => {
            tensors.amplitudes.expect(AMPLITUDES).t1
                [(idx[upper[0] as usize], idx[lower[0] as usize])]
        }
        // t_{i_l_1, i_l_2}^{i_u_1, i_u_2}.
        TensorKind::T2 => {
            tensors.amplitudes.expect(AMPLITUDES).t2[(
                idx[upper[0] as usize],
                idx[upper[1] as usize],
                idx[lower[0] as usize],
                idx[lower[1] as usize],
            )]
        }
    }
}
