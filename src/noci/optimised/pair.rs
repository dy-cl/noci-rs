// noci/optimised/pair.rs
//! Cutoff-floored pair quantities of ResHF-cutoff: the signed overlap, the coupling and the
//! spin-resolved transition densities and Fock matrices of every ordered determinant pair.
//! Floored singular values keep `Q^{AB,\omega}` finite for (near-)orthogonal pairs, and the
//! coupling, overlap and resonating Fock matrices are all built from the same floored `Q`.

// External crate imports.
use ndarray::{Array1, Array2};

// Crate-root imports.
use crate::elements::nonorthogonalwicks::SameSpinBuild;
use crate::elements::{build_jk_reshf, occ_coeffs};
use crate::{AoData, SCFState};

/// Cutoff-floored quantities of one ordered determinant pair `(A, B)`.
pub struct ResHFPair {
    /// Signed overlap `s_{AB} = \eta \prod_i \sigma^\alpha_i \prod_j \sigma^\beta_j`, floored.
    pub s: f64,
    /// Coupling `H_{AB}`, including `s_{AB} E_{\text{nuc}}`.
    pub h: f64,
    /// `Q^{AB,\alpha}` and `Q^{AB,\beta}`.
    pub q: [Array2<f64>; 2],
    /// `F^{AB,\alpha}` and `F^{AB,\beta}`.
    pub f: [Array2<f64>; 2],
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

/// Cutoff-floored quantities of all ordered pairs of a determinant list, stored row-major with
/// index `a * n + b`.
pub struct ResHFPairs {
    /// Number of determinants `n`.
    n: usize,
    /// Quantities of pair `(A, B)` at index `a * n + b`.
    pairs: Vec<ResHFPair>,
}

impl ResHFPairs {
    /// Number of determinants `n` the pairs are formed from.
    /// # Arguments:
    /// - `self`: Pair quantities.
    /// # Returns
    /// - `usize`: Number of determinants.
    pub fn ndet(&self) -> usize {
        self.n
    }

    /// Quantities of the ordered pair `(A, B)`.
    /// # Arguments:
    /// - `self`: Pair quantities.
    /// - `a`: Bra determinant index `A`.
    /// - `b`: Ket determinant index `B`.
    /// # Returns
    /// - `&ResHFPair`: Quantities of the pair `(A, B)`.
    /// # Panics
    /// - Panics if `a` or `b` is not below `n`.
    pub fn pair(
        &self,
        a: usize,
        b: usize,
    ) -> &ResHFPair {
        assert!(a < self.n && b < self.n, "pair ({a}, {b}) out of range");
        &self.pairs[a * self.n + b]
    }

    /// Hamiltonian matrix `H_{AB}` built from the floored pair couplings.
    /// # Arguments:
    /// - `self`: Pair quantities.
    /// # Returns
    /// - `Array2<f64>`: Hamiltonian matrix, including nuclear repulsion.
    pub fn h(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.n, self.n), |(a, b)| self.pair(a, b).h)
    }

    /// Overlap matrix `S_{AB} = s_{AB}` built from the floored pair overlaps.
    /// # Arguments:
    /// - `self`: Pair quantities.
    /// # Returns
    /// - `Array2<f64>`: Overlap matrix.
    pub fn s(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.n, self.n), |(a, b)| self.pair(a, b).s)
    }
}

/// Floor singular values by yucca's rule: if `\sigma_{\min} / \sigma_{\text{ref}} < c`, set
/// `\sigma_i \leftarrow \max(\sigma_i, c\,\sigma_{\text{ref}})`, where `\sigma_{\text{ref}}` is
/// `\sigma_{\max}` for more than one orbital and `1` for a single orbital.
/// # Arguments:
/// - `sigma`: Singular values of an occupied orbital overlap matrix.
/// - `cutoff`: Relative floor `c`.
/// # Returns
/// - `Array1<f64>`: Floored singular values.
fn floor_singular_values(
    sigma: &Array1<f64>,
    cutoff: f64,
) -> Array1<f64> {
    let reference = if sigma.len() > 1 {
        sigma.fold(0.0, |m: f64, &x| m.max(x))
    } else {
        1.0
    };
    let smallest = sigma.fold(f64::INFINITY, |m: f64, &x| m.min(x));

    if smallest / reference < cutoff {
        sigma.mapv(|x| x.max(cutoff * reference))
    } else {
        sigma.clone()
    }
}

/// Floored overlap factor and transition density of one spin,
/// `s^\omega = \eta^\omega \prod_i \sigma_i` and `Q^{AB,\omega} = B_o \Sigma^{-1} A_o^\dagger`,
/// where `S^{AB,\omega}_{oo} = U \Sigma V^\dagger`, `A_o = C^A_o U`, `B_o = C^B_o V` and
/// `\eta^\omega = \det U \det V`. Unfloored, `Q^{AB,\omega} = C^B_o (S^{AB,\omega}_{oo})^{-1}
/// (C^A_o)^\dagger`.
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
    // A spin without electrons contributes `s^\omega = 1` and `Q^{AB,\omega} = 0`.
    let nao = s_ao.nrows();
    if a_occ.ncols() == 0 {
        return (1.0, Array2::zeros((nao, nao)));
    }

    // Pair the occupied orbitals by SVD; the ket-side orbitals `B_o` are returned first.
    let (sigma, b_tilde, a_tilde, eta) =
        SameSpinBuild::<f64>::perform_ortho_and_svd_and_rotate(s_ao, a_occ, b_occ);
    let sigma = floor_singular_values(&sigma, cutoff);

    // `Q = B_o \Sigma^{-1} A_o^\dagger`: scale each paired ket orbital by `1 / \sigma_i`.
    let mut b_scaled = b_tilde;
    for (mut col, &x) in b_scaled.columns_mut().into_iter().zip(sigma.iter()) {
        col /= x;
    }
    let q = b_scaled.dot(&a_tilde.t());

    (eta * sigma.product(), q)
}

/// `\mathrm{tr}(X Q) = \sum_{\mu\nu} X_{\mu\nu} Q_{\nu\mu}` for possibly non-symmetric matrices.
/// # Arguments:
/// - `x`: First matrix.
/// - `q`: Second matrix.
/// # Returns
/// - `f64`: Trace of the product.
fn trace_product(
    x: &Array2<f64>,
    q: &Array2<f64>,
) -> f64 {
    (x * &q.t()).sum()
}

/// Cutoff-floored ResHF quantities of the ordered pair `(A, B)`: per spin the floored
/// `s^\omega` and `Q^{AB,\omega}`, then `s_{AB} = s^\alpha s^\beta`,
/// `F^{AB,\omega} = h + J[Q^{AB,\alpha}] + J[Q^{AB,\beta}] - K[Q^{AB,\omega}]` and
/// `H_{AB} = s_{AB} [E_{\text{nuc}} + \frac{1}{2} \sum_\omega \mathrm{tr}((h + F^{AB,\omega}) Q^{AB,\omega})]`.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `a`: Bra determinant `A`.
/// - `b`: Ket determinant `B`.
/// - `cutoff`: Relative singular value floor; see [`floor_singular_values`].
/// # Returns
/// - `ResHFPair`: Floored overlap, coupling, transition densities and Fock matrices.
pub fn reshf_pair(
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

    // `F^{AB,\omega} = h + J^\alpha + J^\beta - K^\omega` from the full, non-symmetric `J` and `K`.
    let (ja, ka) = build_jk_reshf(&ao.eri_coul, &qa);
    let (jb, kb) = build_jk_reshf(&ao.eri_coul, &qb);
    let hj = &ao.h + &ja + &jb;
    let fa = &hj - &ka;
    let fb = &hj - &kb;

    // `H_{AB} = s_{AB} [E_{\text{nuc}} + \frac{1}{2} \sum_\omega \mathrm{tr}((h + F^\omega) Q^\omega)]`.
    let s = sa * sb;
    let e =
        ao.enuc + 0.5 * (trace_product(&(&ao.h + &fa), &qa) + trace_product(&(&ao.h + &fb), &qb));

    ResHFPair {
        s,
        h: s * e,
        q: [qa, qb],
        f: [fa, fb],
    }
}

/// Cutoff-floored ResHF quantities of every ordered pair of `dets`. Pairs `(A, B)` with
/// `B \le A` are built directly and `(B, A)` follows from them by [`ResHFPair::reversed`].
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `dets`: Determinants `A = 0, \ldots, n - 1`.
/// - `cutoff`: Relative singular value floor.
/// # Returns
/// - `ResHFPairs`: Quantities of all `n^2` ordered pairs.
pub fn reshf_pairs(
    ao: &AoData,
    dets: &[SCFState],
    cutoff: f64,
) -> ResHFPairs {
    let n = dets.len();

    // Build the lower triangle and fill the upper triangle from the reversed pairs.
    let mut pairs: Vec<Option<ResHFPair>> = (0..n * n).map(|_| None).collect();
    for a in 0..n {
        for b in 0..=a {
            let ab = reshf_pair(ao, &dets[a], &dets[b], cutoff);
            if b != a {
                pairs[b * n + a] = Some(ab.reversed());
            }
            pairs[a * n + b] = Some(ab);
        }
    }

    ResHFPairs {
        n,
        pairs: pairs
            .into_iter()
            .map(|p| p.expect("every pair is built"))
            .collect(),
    }
}
