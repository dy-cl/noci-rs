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
/// and each of the `s` ordered excitors independently with probability `|n_\mu|/N_{\text{ex}}`
/// from the block's alias table,
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
        let (excitor, n) = block.excitors[block.selection.sample(rng)];
        excitors.push(excitor);
        weight *= n.signum() * nex / block.nref / k as f64;
    }

    // Count the distinct excitors, which are the ones independently assigned to ranks.
    let distinct = (0..s)
        .filter(|&k| excitors[..k].iter().all(|e| e.det != excitors[k].det))
        .count();
    weight * ranks.powi(distinct as i32 - 1) / (ps * attempts)
}

/// Combinations of the truncated expansion sampled by the coupling between references, with
/// their weights. Writing `\hat T_x=\sum_j\hat T_{xj}` by excitation level, the unlinked expansion
/// `N_xe^{\hat T_x/N_x}=N_x\prod_j e^{\hat T_{xj}/N_x}` is a sum over combinations
/// `\eta=(\eta_1,\ldots,\eta_l)` of `\eta_j` excitors of level `j`, of size `s=\sum_j\eta_j`,
/// kept only when its excitation level `\sum_j j\eta_j` is within the expansion rank (truncated
/// selection). Drawing its excitors of level `j` with probability `|n_\mu|/L_j`, where `L_j` is
/// the population magnitude of that level on this rank, the combination contributes the weight
///
/// `a_\eta = |N_x|^{1-s}P^{s-1}\prod_j\frac{L_j^{\eta_j}}{\eta_j!},`
///
/// in which `P^{s-1}` is the inverse probability that its `s` excitors share a rank; a repeated
/// excitor collapses to zero, so contributing excitors are always distinct. The reference term is
/// `a_0=|N_x|/P`. Combinations of zero weight are omitted.
/// # Arguments:
/// - `levels`: Population magnitudes `L_j` of levels `j=1,\ldots,l` on this rank.
/// - `nref`: Reference population `N_x`.
/// - `expansion_rank`: Highest excitation level of a combination.
/// - `ranks`: Number of ranks `P` holding the distributed excitors.
/// # Returns:
/// - `(Vec<Vec<usize>>, Vec<f64>)`: Combinations `\eta` and their weights `a_\eta`.
/// # References
/// - C. J. C. Scott and A. J. W. Thom, *J. Chem. Phys.* **147**, 124105 (2017),
///   [doi:10.1063/1.4991795](https://doi.org/10.1063/1.4991795).
pub(super) fn coupling_combinations(
    levels: &[f64],
    nref: f64,
    expansion_rank: usize,
    ranks: f64,
) -> (Vec<Vec<usize>>, Vec<f64>) {
    // Every combination of counts per level within the expansion rank.
    let mut combinations = vec![(Vec::new(), 0usize)];
    for j in 1..=levels.len() {
        combinations = combinations
            .into_iter()
            .flat_map(|(eta, rank)| {
                (0..=(expansion_rank - rank) / j).map(move |count| {
                    let mut eta = eta.clone();
                    eta.push(count);
                    (eta, rank + j * count)
                })
            })
            .collect();
    }

    let factorial = |k: usize| (1..=k).map(|i| i as f64).product::<f64>();
    combinations
        .into_iter()
        .map(|(eta, _)| {
            let s = eta.iter().sum::<usize>() as i32;
            let weight = eta
                .iter()
                .zip(levels)
                .map(|(&count, &l)| l.powi(count as i32) / factorial(count))
                .product::<f64>()
                * nref.abs().powi(1 - s)
                * ranks.powi(s - 1);
            (eta, weight)
        })
        .filter(|(_, weight)| *weight > 0.0)
        .unzip()
}

/// Select one cluster of a block for the coupling between references by even selection: draw a
/// combination `\eta` with probability `a_\eta/A`, `A=\sum_\eta a_\eta`, then `\eta_j` ordered
/// excitors of each level `j` with probability `|n_\mu|/L_j`. Every cluster then carries the same
/// weight magnitude,
///
/// `w = \frac{A}{n_{\text{attempts}}}\operatorname{sign}(N_x)^{1-s}\prod_k\operatorname{sign}(n_{\mu_k}),`
///
/// so that no single sample of the coupling can outweigh another, however large the particle
/// ratio `N_{\text{ex}}/N_x`. The sum of `w f` over the attempts is an unbiased estimate of
/// `\sum_\eta a_\eta\langle\operatorname{sign}f\rangle_\eta`, the expansion of `N_xe^{\hat T_x/N_x}`.
/// # Arguments:
/// - `block`: Block population with its level and combination tables.
/// - `attempts`: Number of coupling attempts on this block on this rank.
/// - `rng`: Random-number generator.
/// - `excitors`: Output excitors of the cluster.
/// # Returns:
/// - `f64`: Cluster weight `w`, zero if the block has no reference population.
/// # References
/// - C. J. C. Scott and A. J. W. Thom, *J. Chem. Phys.* **147**, 124105 (2017),
///   [doi:10.1063/1.4991795](https://doi.org/10.1063/1.4991795).
pub(super) fn select_coupling_cluster(
    block: &Block,
    attempts: f64,
    rng: &mut QmcRng,
    excitors: &mut Vec<Excitor>,
) -> f64 {
    excitors.clear();
    let total = block.coupling.total();
    if total == 0.0 || block.nref == 0.0 {
        return 0.0;
    }

    let eta = &block.combinations[block.coupling.sample(rng)];
    let mut sign = 1.0;
    for ((indices, selection), &count) in block.levels.iter().zip(eta) {
        for _ in 0..count {
            let (excitor, n) = block.excitors[indices[selection.sample(rng)]];
            excitors.push(excitor);
            sign *= n.signum();
        }
    }
    if excitors.len().is_multiple_of(2) {
        sign *= block.nref.signum();
    }
    sign * total / attempts
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
