// spin/rules.rs

// Standard library imports.
use std::sync::OnceLock;

// External crate imports.
use num_rational::Ratio;
use smallvec::SmallVec;

// Parent/sibling imports.
use super::gram::{self, Basis};

/// One spin-free replacement: a lower-index permutation and its coefficient.
pub(crate) type Replacement = (SmallVec<[u8; 4]>, Ratio<i64>);

/// Replacement table indexed by `[rank][upper spin bits][lower spin bits]`.
type Table = Vec<Vec<Vec<Vec<Replacement>>>>;

/// Return the spin-free replacement of one spin-orbital cumulant spin block.
/// For an `SU(2)`-invariant spin ensemble the spin-orbital cumulant of rank `k` is a linear
/// combination of lower-index permutations of the spin-free cumulant,
///
/// `\lambda^{p_1\sigma_1..p_k\sigma_k}_{q_1\tau_1..q_k\tau_k} = \sum_\rho c_\rho(\sigma,\tau)
/// \Lambda^{p_1..p_k}_{q_{\rho(1)}..q_{\rho(k)}},`
///
/// with `\Lambda = \sum_\sigma \lambda(\sigma, \sigma)`. The coefficients solve the spin Gram
/// system of `S_k` on two-state spins; for `k = 1` this is `\lambda = \tfrac12\Lambda\delta`.
/// # Arguments:
/// - `k`: Cumulant rank, `1..=4`.
/// - `upper`: Upper spin bits, bit `i` set for beta.
/// - `lower`: Lower spin bits, bit `i` set for beta.
/// # Returns:
/// - `&'static [Replacement]`: Nonzero permutations and coefficients.
pub(crate) fn cumulant(
    k: usize,
    upper: u8,
    lower: u8,
) -> &'static [Replacement] {
    static TABLE: OnceLock<Table> = OnceLock::new();

    &TABLE.get_or_init(table)[k][upper as usize][lower as usize]
}

/// Build every cumulant replacement up to rank four.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Table`: Replacements for every rank and spin pattern.
fn table() -> Table {
    (0..=4)
        .map(|k| {
            (0..1u8 << k)
                .map(|upper| (0..1u8 << k).map(|lower| solve(k, upper, lower)).collect())
                .collect()
        })
        .collect()
}

/// Solve one cumulant spin block.
/// # Arguments:
/// - `k`: Cumulant rank.
/// - `upper`: Upper spin bits.
/// - `lower`: Lower spin bits.
/// # Returns:
/// - `Vec<Replacement>`: Nonzero permutations and coefficients.
fn solve(
    k: usize,
    upper: u8,
    lower: u8,
) -> Vec<Replacement> {
    if k == 0 || upper.count_ones() != lower.count_ones() {
        return Vec::new();
    }

    let data = Basis::cached(k);
    let bit = |x: u8, i: usize| (x >> i) & 1;

    // Right-hand side: the signed spin pattern of every permutation.
    let b = data
        .ps
        .iter()
        .map(|p| {
            if (0..k).all(|i| bit(upper, i) == bit(lower, p[i])) {
                Ratio::from_integer(gram::sign(p))
            } else {
                Ratio::from_integer(0)
            }
        })
        .collect::<Vec<_>>();

    let x = gram::solve(data.g.clone(), b)
        .unwrap_or_else(|| panic!("inconsistent spin projection {k}"));

    data.ps
        .iter()
        .zip(x)
        .filter(|(_, c)| *c != Ratio::from_integer(0))
        .map(|(p, c)| (p.iter().map(|&i| i as u8).collect(), c))
        .collect()
}

/// Return the spin-free replacement of one antisymmetrised two-body spin block.
/// For `v^{p\mu_1 q\mu_2}_{r\lambda_1 s\lambda_2}` built from a spin-free two-body tensor `g` with
/// pair symmetry `g^{pq}_{rs} = g^{qp}_{sr}` (Lee and Tew Eqs. 22-24):
///
/// `v = g^{pq}_{rs}\delta_{\mu_1\lambda_1}\delta_{\mu_2\lambda_2} - g^{pq}_{sr}
/// \delta_{\mu_1\lambda_2}\delta_{\mu_2\lambda_1}.`
/// # Arguments:
/// - `upper`: Upper spin bits.
/// - `lower`: Lower spin bits.
/// # Returns:
/// - `Vec<Replacement>`: Direct and exchange lower permutations with their signs.
pub(crate) fn pair(
    upper: u8,
    lower: u8,
) -> Vec<Replacement> {
    let bit = |x: u8, i: usize| (x >> i) & 1;
    let mut out = Vec::new();

    if bit(upper, 0) == bit(lower, 0) && bit(upper, 1) == bit(lower, 1) {
        out.push((SmallVec::from_slice(&[0, 1]), Ratio::from_integer(1)));
    }
    if bit(upper, 0) == bit(lower, 1) && bit(upper, 1) == bit(lower, 0) {
        out.push((SmallVec::from_slice(&[1, 0]), Ratio::from_integer(-1)));
    }

    out
}
