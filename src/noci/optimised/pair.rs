// noci/optimised/pair.rs
//! Cutoff-floored pair quantities of ResHF-cutoff: the signed overlap, the coupling and the
//! spin-resolved transition densities and Fock matrices of every ordered determinant pair.
//! Floored singular values keep `Q^{AB,\omega}` finite for (near-)orthogonal pairs, and the
//! coupling, overlap and resonating Fock matrices are all built from the same floored `Q`.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::elements::{build_s_pair_floored, occ_coeffs};
use crate::scf::fock_nonsymmetric;
use crate::{AoData, NOCIScalar, SCFState};

/// Cutoff-floored quantities of one ordered determinant pair `(A, B)`.
#[derive(Clone)]
pub(super) struct ResHFPair {
    /// Signed overlap `s_{AB} = \eta \prod_i \sigma^\alpha_i \prod_j \sigma^\beta_j`, floored.
    pub(super) s: f64,
    /// Coupling `H_{AB}`, including `s_{AB} E_{\text{nuc}}`.
    pub(super) h: f64,
    /// `Q^{AB,\alpha}` and `Q^{AB,\beta}`.
    pub(super) q: [Array2<f64>; 2],
    /// `F^{AB,\alpha}` and `F^{AB,\beta}`.
    pub(super) f: [Array2<f64>; 2],
}

impl ResHFPair {
    /// Quantities of the reversed pair `(B, A)` from those of `(A, B)`, which for real orbitals
    /// are `Q^{BA,\omega} = (Q^{AB,\omega})^T`, `F^{BA,\omega} = (F^{AB,\omega})^T`,
    /// `s_{BA} = s_{AB}` and `H_{BA} = H_{AB}`.
    /// # Arguments:
    /// - `self`: Quantities of the pair `(A, B)`.
    /// # Returns
    /// - `ResHFPair`: Quantities of the pair `(B, A)`.
    fn reversed(&self) -> ResHFPair {
        ResHFPair {
            s: self.s,
            h: self.h,
            q: [self.q[0].t().to_owned(), self.q[1].t().to_owned()],
            f: [self.f[0].t().to_owned(), self.f[1].t().to_owned()],
        }
    }
}

/// Floored overlap factor and transition density of one spin, `s^\omega` and
/// `Q^{AB,\omega} = B_o \Sigma^{-1} A_o^\dagger` with floored singular values `\Sigma`; see
/// [`build_s_pair_floored`]. A spin without electrons contributes `s^\omega = 1` and
/// `Q^{AB,\omega} = 0`.
/// # Arguments:
/// - `s_ao`: AO overlap matrix.
/// - `a_occ`: Occupied coefficients `C^A_o` of the bra determinant.
/// - `b_occ`: Occupied coefficients `C^B_o` of the ket determinant.
/// - `cutoff`: Relative singular value floor.
/// # Returns
/// - `(f64, Array2<f64>)`: Overlap factor `s^\omega` and transition density `Q^{AB,\omega}`.
fn spin_transition(
    s_ao: &Array2<f64>,
    a_occ: &Array2<f64>,
    b_occ: &Array2<f64>,
    cutoff: f64,
) -> (f64, Array2<f64>) {
    let nao = s_ao.nrows();
    if a_occ.ncols() == 0 {
        return (1.0, Array2::zeros((nao, nao)));
    }

    let pair = build_s_pair_floored(a_occ, b_occ, s_ao, cutoff);
    let q = pair.w.unwrap_or_else(|| Array2::zeros((nao, nao)));
    (pair.s, q)
}

/// Cutoff-floored ResHF quantities of the ordered pair `(A, B)`: per spin the floored
/// `s^\omega` and `Q^{AB,\omega}`, then `s_{AB} = s^\alpha s^\beta`,
/// `F^{AB,\omega} = h + J[Q^{AB,\alpha}] + J[Q^{AB,\beta}] - K[Q^{AB,\omega}]` and
/// `H_{AB} = s_{AB} [E_{\text{nuc}} + \frac{1}{2} \sum_\omega \mathrm{tr}((h + F^{AB,\omega}) Q^{AB,\omega})]`.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `a`: Bra determinant `A`.
/// - `b`: Ket determinant `B`.
/// - `cutoff`: Relative singular value floor; see [`build_s_pair_floored`].
/// # Returns
/// - `ResHFPair`: Floored overlap, coupling, transition densities and Fock matrices.
fn reshf_pair(
    ao: &AoData,
    a: &SCFState,
    b: &SCFState,
    cutoff: f64,
) -> ResHFPair {
    // Floored overlap factors and transition densities of both spins.
    let (sa, qa) = spin_transition(
        &ao.s,
        &occ_coeffs(&a.ca, a.oa),
        &occ_coeffs(&b.ca, b.oa),
        cutoff,
    );
    let (sb, qb) = spin_transition(
        &ao.s,
        &occ_coeffs(&a.cb, a.ob),
        &occ_coeffs(&b.cb, b.ob),
        cutoff,
    );

    // `F^{AB,\omega} = h + J[Q^\alpha + Q^\beta] - K[Q^\omega]` from the non-symmetric `Q`.
    let (fa, fb) = fock_nonsymmetric(&ao.h, &ao.eri_coul, &qa, &qb);

    // `H_{AB} = s_{AB} [E_{\text{nuc}} + \frac{1}{2} \sum_\omega \mathrm{tr}((h + F^\omega) Q^\omega)]`.
    let s = sa * sb;
    let e = ao.enuc
        + 0.5 * (f64::einsum_ba_ab(&(&ao.h + &fa), &qa) + f64::einsum_ba_ab(&(&ao.h + &fb), &qb));

    ResHFPair {
        s,
        h: s * e,
        q: [qa, qb],
        f: [fa, fb],
    }
}

/// Cutoff-floored ResHF quantities of every ordered pair of `dets`, indexed `[(a, b)]`. Pairs
/// `(A, B)` with `B \le A` are built directly and `(B, A)` follows from them by
/// [`ResHFPair::reversed`].
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `dets`: Determinants `A = 0, \ldots, n - 1`.
/// - `cutoff`: Relative singular value floor.
/// # Returns
/// - `Array2<ResHFPair>`: Quantities of all `n^2` ordered pairs.
pub(super) fn reshf_pairs(
    ao: &AoData,
    dets: &[SCFState],
    cutoff: f64,
) -> Array2<ResHFPair> {
    let n = dets.len();

    // Build the lower triangle and fill the upper triangle from the reversed pairs.
    let lower: Vec<Vec<ResHFPair>> = (0..n)
        .map(|a| {
            (0..=a)
                .map(|b| reshf_pair(ao, &dets[a], &dets[b], cutoff))
                .collect()
        })
        .collect();
    Array2::from_shape_fn((n, n), |(a, b)| {
        if b <= a {
            lower[a][b].clone()
        } else {
            lower[b][a].reversed()
        }
    })
}
