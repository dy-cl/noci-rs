// nocc/stochastic/linked.rs
//! Sampling of the same-reference term `N_z\langle{}^z\Psi|\hat\tau_{z\mu}^\dagger\bar H_z|{}^z\Psi\rangle`
//! with `\bar H_z = e^{-\hat T_z}\hat He^{\hat T_z}`.
//!
//! Within one reference the determinants are orthonormal and `\bar H_z` terminates after four
//! nested commutators. Because excitors commute, the commutator of a cluster `C` expands over
//! the subsets `L` of excitors placed left of `\hat H`,
//!
//! `[\ldots[\hat H,\hat\tau_1],\ldots,\hat\tau_s] = \sum_{L\subseteq C}(-1)^{|L|}\hat\tau_L\hat H\hat\tau_{C\setminus L},`
//!
//! so unlinked clusters cancel exactly. A projectee `\mu` is chosen by drawing one partition,
//! collapsing its right part to `|D_n\rangle`, drawing a Hamiltonian connection
//! `|D_n\rangle\to|D_m\rangle` and applying the left part to `|D_m\rangle`. Its generation
//! probability sums over every partition able to produce it. The diagonal projectee, the
//! collapsed cluster itself, is treated deterministically as death. Clusters whose excitors act
//! on a common orbital (conjoint clusters) can still contribute through partitions that place
//! them on different sides of `\hat H`.
//!
//! # References
//! - R. S. T. Franklin, J. S. Spencer, A. Zoccante and A. J. W. Thom, *J. Chem. Phys.* **144**,
//!   044111 (2016), [doi:10.1063/1.4940317](https://doi.org/10.1063/1.4940317).

// External crate imports.
use rand::Rng;

// Crate-root imports.
use crate::determinant::ExcitedDeterminant;
use crate::qmc::QmcRng;

// Parent/sibling imports.
use super::cluster::collapse;
use super::population::{Excitor, ReferenceSpaces, rank};
use super::state::{LinkedBatchedRequest, Target};

/// Queue the terms of
/// `V_\mu = \langle D_\mu|[\ldots[\hat H,\hat\tau_1],\ldots,\hat\tau_s]|{}^z\Psi\rangle`
/// `= \sum_{L}(-1)^{|L|}\langle\hat\tau_L^\dagger D_\mu|\hat H|\hat\tau_{C\setminus L}{}^z\Psi\rangle`
/// with the signs of each excitor action, each multiplied by `coefficient` and added to `target`
/// once evaluated. Partitions whose bra and ket differ by more than a double excitation vanish
/// and are not queued.
/// # Arguments:
/// - `reference`: Block reference `{}^z\Psi`.
/// - `excitors`: Cluster excitors.
/// - `projectee`: Projection determinant `|D_\mu\rangle`.
/// - `term`: Destination and coefficient of `V_\mu`.
/// - `requests`: Queued same-reference requests.
/// # Returns:
/// - `()`: Queues the surviving partitions.
fn commutator(
    reference: &ExcitedDeterminant,
    excitors: &[Excitor],
    projectee: &ExcitedDeterminant,
    term: (Target, f64),
    requests: &mut Vec<LinkedBatchedRequest>,
) {
    let (target, coefficient) = term;
    let s = excitors.len();
    for left in 0..1usize << s {
        // Right part acting on the reference.
        let right =
            (0..s)
                .filter(|k| left >> k & 1 == 0)
                .try_fold((*reference, 1.0), |(det, sign), k| {
                    let (next, sk) = excitors[k].apply(&det)?;
                    Some((next, sign * sk))
                });
        // Left part inverted on the projectee.
        let bra =
            (0..s)
                .filter(|k| left >> k & 1 == 1)
                .try_fold((*projectee, 1.0), |(det, sign), k| {
                    let (prev, sk) = excitors[k].invert(&det)?;
                    Some((prev, sign * sk))
                });
        if let (Some((ket, sr)), Some((bra, sl))) = (right, bra) {
            if (bra.alpha ^ ket.alpha).count_ones() + (bra.beta ^ ket.beta).count_ones() > 4 {
                continue;
            }
            let parity = if left.count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            requests.push(LinkedBatchedRequest {
                target,
                coefficient: coefficient * parity * sl * sr,
                bra,
                ket,
            });
        }
    }
}

/// Probability that the spawning step proposes projectee `\mu` from cluster `C`,
/// `P(\mu|C) = 2^{-s}\sum_{R\subseteq C} P_{\text{gen}}(\hat\tau_L^\dagger D_\mu|\hat\tau_R{}^z\Psi)`,
/// summing over the partitions whose right part collapses to `|D_n\rangle` and whose left part
/// inverts `|D_\mu\rangle` to a distinct single or double excitation `|D_m\rangle` of it.
/// # Arguments:
/// - `reference`: Block reference `{}^z\Psi`.
/// - `excitors`: Cluster excitors.
/// - `projectee`: Proposed projectee.
/// - `pgen`: Uniform connection probability.
/// # Returns:
/// - `f64`: Proposal probability.
fn proposal(
    reference: &ExcitedDeterminant,
    excitors: &[Excitor],
    projectee: &ExcitedDeterminant,
    pgen: f64,
) -> f64 {
    let s = excitors.len();
    let mut p = 0.0;
    for left in 0..1usize << s {
        let right = (0..s)
            .filter(|k| left >> k & 1 == 0)
            .try_fold(*reference, |det, k| Some(excitors[k].apply(&det)?.0));
        let bra = (0..s)
            .filter(|k| left >> k & 1 == 1)
            .try_fold(*projectee, |det, k| Some(excitors[k].invert(&det)?.0));
        if let (Some(n), Some(m)) = (right, bra) {
            let distance = (n.alpha ^ m.alpha).count_ones() + (n.beta ^ m.beta).count_ones();
            if distance != 0 && distance <= 4 {
                p += pgen;
            }
        }
    }
    p / (1usize << s) as f64
}

/// Perform one linked attempt for a selected cluster, queueing its matrix elements: the
/// reference projection `\langle{}^z\Psi|\hat H|\hat\tau_C{}^z\Psi\rangle` of `\bar H_z`, `-w V_C`
/// on the collapsed cluster (death), and `-w V_\mu/P(\mu|C)` on one sampled projectee
/// (spawning). Once evaluated the changes estimate `-N_zR_{z\mu}` before multiplication by
/// `\Delta\tau`.
/// # Arguments:
/// - `spaces`: Projection and expansion spaces of every reference block.
/// - `z`: Block of the cluster.
/// - `cluster`: Cluster excitors and weight `w`.
/// - `rng`: Random-number generator.
/// - `requests`: Queued same-reference requests.
/// # Returns:
/// - `()`: Queues the attempt's requests.
pub(super) fn linked_attempt(
    spaces: &ReferenceSpaces,
    z: usize,
    cluster: (&[Excitor], f64),
    rng: &mut QmcRng,
    requests: &mut Vec<LinkedBatchedRequest>,
) {
    let (excitors, weight) = cluster;
    let reference = &spaces.references[z];
    let key = |det: &ExcitedDeterminant| Target::Excitor((z, det.alpha, det.beta));

    // Reference projection and death on the collapsed cluster.
    if let Some((collapsed, sign)) = collapse(reference, excitors) {
        let r = rank(reference, &collapsed);
        if r <= 2 {
            requests.push(LinkedBatchedRequest {
                target: Target::Energy(z),
                coefficient: weight * sign,
                bra: *reference,
                ket: collapsed,
            });
        }
        if (1..=spaces.cluster_rank).contains(&r) {
            commutator(
                reference,
                excitors,
                &collapsed,
                (key(&collapsed), -weight),
                requests,
            );
        }
    }

    // Spawning: one partition, one Hamiltonian connection, then the left part.
    let s = excitors.len();
    let left = rng.gen_range(0..1usize << s);
    let Some(source) = (0..s)
        .filter(|k| left >> k & 1 == 0)
        .try_fold(*reference, |det, k| Some(excitors[k].apply(&det)?.0))
    else {
        return;
    };
    let Some((connection, pgen)) = spaces.generators[z].sample(rng) else {
        return;
    };
    let (ca, cb) = connection.child_of((source.alpha, source.beta), spaces.norb[z]);
    let target = ExcitedDeterminant {
        parent: reference.parent,
        alpha: ca,
        beta: cb,
    };
    let Some(projectee) = (0..s)
        .filter(|k| left >> k & 1 == 1)
        .try_fold(target, |det, k| Some(excitors[k].apply(&det)?.0))
    else {
        return;
    };
    let r = rank(reference, &projectee);
    if r == 0 || r > spaces.cluster_rank {
        return;
    }

    let p = proposal(reference, excitors, &projectee, pgen);
    commutator(
        reference,
        excitors,
        &projectee,
        (key(&projectee), -weight / p),
        requests,
    );
}
