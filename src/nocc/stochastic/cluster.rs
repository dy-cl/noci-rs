// nocc/stochastic/cluster.rs

// External crate imports.
use rand::Rng;

// Crate-root imports.
use crate::determinant::ExcitedDeterminant;
use crate::qmc::QmcRng;

// Parent/sibling imports.
use super::population::{Block, Excitor};

/// Select one cluster of a block for the sampled expansion
/// `\sum_s \frac{1}{s!}\sum_{\mu_1\ldots\mu_s} N_z \prod_k t_{z\mu_k}\,f(\mu_1,\ldots,\mu_s)`.
/// The size `s` is drawn with `p(s)=2^{-(s+1)}` for `s<s_{\max}` and `p(s_{\max})=2^{-s_{\max}}`,
/// and each of the `s` ordered excitors independently with probability `|n_\mu|/N_{\text{ex}}`,
/// so that the returned weight
///
/// `w = \frac{N_z}{s!\,p(s)\,A}\prod_k\frac{\operatorname{sign}(n_{\mu_k})N_{\text{ex}}}{N_z}`
///
/// makes the sum of `w f` over `A` attempts an unbiased estimate of the expansion, with
/// `t_{z\mu}=n_{z\mu}/N_z`. Repeated excitors are allowed. With the excitors distributed over
/// `P` ranks by a random assignment, each rank forms clusters from its own excitors only and its
/// weights carry `P^{d-1}`, the inverse probability that the cluster's `d` distinct excitors share
/// a rank, so the sum over ranks is unbiased over the assignment. A repeated excitor, which
/// contributes through the commutators of linked CCMC, always shares its own rank.
/// # Arguments:
/// - `block`: Block population.
/// - `smax`: Largest cluster size.
/// - `attempts`: Number of attempts `A` sharing the expansion on this rank.
/// - `ranks`: Number of ranks `P` holding the distributed excitors.
/// - `rng`: Random-number generator.
/// - `excitors`: Output excitors of the cluster.
/// # Returns:
/// - `f64`: Cluster weight `w`.
/// # References
/// - A. J. W. Thom, *Phys. Rev. Lett.* **105**, 263004 (2010),
///   [doi:10.1103/PhysRevLett.105.263004](https://doi.org/10.1103/PhysRevLett.105.263004).
/// - J. S. Spencer and A. J. W. Thom, *J. Chem. Phys.* **144**, 084108 (2016),
///   [doi:10.1063/1.4942173](https://doi.org/10.1063/1.4942173).
pub(super) fn select_cluster(
    block: &Block,
    smax: usize,
    attempts: f64,
    ranks: f64,
    rng: &mut QmcRng,
    excitors: &mut Vec<Excitor>,
) -> f64 {
    excitors.clear();
    let nex = block.nex();

    // Without excitors only the reference cluster exists.
    if nex == 0.0 || smax == 0 {
        return block.nref / (ranks * attempts);
    }

    // Draw the size from the truncated geometric distribution.
    let u = rng.r#gen::<f64>();
    let mut s = 0;
    let mut cumulative = 0.5;
    while s < smax && u >= cumulative {
        s += 1;
        cumulative += 0.5f64.powi(s as i32 + 1);
    }
    let ps = if s < smax {
        0.5f64.powi(s as i32 + 1)
    } else {
        0.5f64.powi(smax as i32)
    };

    // Draw ordered excitors proportionally to their population magnitudes.
    let mut weight = block.nref;
    for k in 1..=s {
        let target = rng.r#gen::<f64>() * nex;
        let i = block
            .cumulative
            .partition_point(|&c| c <= target)
            .min(block.excitors.len() - 1);
        let (excitor, n) = block.excitors[i];
        excitors.push(excitor);
        weight *= n.signum() * nex / block.nref / k as f64;
    }

    // Count the distinct excitors, which are the ones independently assigned to ranks.
    let distinct = (0..s)
        .filter(|&k| excitors[..k].iter().all(|e| e.det != excitors[k].det))
        .count();
    weight * ranks.powi(distinct as i32 - 1) / (ps * attempts)
}

/// Collapse a cluster onto the block reference, `\prod_k\hat\tau_{\mu_k}|{}^z\Psi\rangle`.
/// # Arguments:
/// - `reference`: Block reference.
/// - `excitors`: Cluster excitors, which commute.
/// # Returns:
/// - `Option<(ExcitedDeterminant, f64)>`: Collapsed determinant and sign, or `None` if the
///   product vanishes.
pub(super) fn collapse(
    reference: &ExcitedDeterminant,
    excitors: &[Excitor],
) -> Option<(ExcitedDeterminant, f64)> {
    excitors
        .iter()
        .try_fold((*reference, 1.0), |(det, sign), excitor| {
            let (next, s) = excitor.apply(&det)?;
            Some((next, sign * s))
        })
}
