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
use crate::determinant::{ExcitedDeterminant, SpinDeterminantState};
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{NOCIData, calculate_hs_excited};
use crate::qmc::QmcRng;
use crate::qmc::excitgen::OrthogonalUniformGenerator;

// Parent/sibling imports.
use super::cluster::collapse;
use super::population::{Excitor, ExcitorKey, rank};

/// Fixed data of one reference block for linked sampling.
pub(super) struct LinkedBlock<'a> {
    /// Block index `z`.
    pub(super) index: usize,
    /// Block reference determinant.
    pub(super) reference: ExcitedDeterminant,
    /// Number of spatial orbitals in the reference frame.
    pub(super) norb: usize,
    /// Highest excitation rank of the cluster operator.
    pub(super) cluster_rank: usize,
    /// Uniform Hamiltonian-connection generator of the reference frame.
    pub(super) generator: &'a OrthogonalUniformGenerator,
}

/// Evaluate `V_\mu = \langle D_\mu|[\ldots[\hat H,\hat\tau_1],\ldots,\hat\tau_s]|{}^z\Psi\rangle`
/// as `\sum_{L}(-1)^{|L|}\langle\hat\tau_L^\dagger D_\mu|\hat H|\hat\tau_{C\setminus L}{}^z\Psi\rangle`
/// with the signs of each excitor action.
/// # Arguments:
/// - `data`: NOCI data holding the parent frames and MO caches.
/// - `block`: Block reference data.
/// - `excitors`: Cluster excitors.
/// - `projectee`: Projection determinant `|D_\mu\rangle`.
/// - `scratch`: Wick scratch, unused by same-frame elements.
/// # Returns:
/// - `f64`: Nested-commutator matrix element.
fn commutator(
    data: &NOCIData<'_, f64>,
    block: &LinkedBlock<'_>,
    excitors: &[Excitor],
    projectee: &ExcitedDeterminant,
    scratch: &mut WickScratchSpin<f64>,
) -> f64 {
    let s = excitors.len();
    let mut value = 0.0;
    for left in 0..1usize << s {
        // Right part acting on the reference.
        let right = (0..s).filter(|k| left >> k & 1 == 0).try_fold(
            (block.reference, 1.0),
            |(det, sign), k| {
                let (next, sk) = excitors[k].apply(&det)?;
                Some((next, sign * sk))
            },
        );
        // Left part inverted on the projectee.
        let bra =
            (0..s)
                .filter(|k| left >> k & 1 == 1)
                .try_fold((*projectee, 1.0), |(det, sign), k| {
                    let (prev, sk) = excitors[k].invert(&det)?;
                    Some((prev, sign * sk))
                });
        if let (Some((ket, sr)), Some((bra, sl))) = (right, bra) {
            let h = calculate_hs_excited(data, (&bra, &ket), scratch).0;
            let parity = if left.count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            value += parity * sl * sr * h;
        }
    }
    value
}

/// Spin components of a determinant in its parent frame, giving its occupied and virtual
/// orbital lists.
/// # Arguments:
/// - `reference`: Block reference.
/// - `det`: Determinant.
/// - `norb`: Number of spatial orbitals.
/// # Returns:
/// - `(SpinDeterminantState, SpinDeterminantState)`: Alpha and beta components.
fn components(
    reference: &ExcitedDeterminant,
    det: &ExcitedDeterminant,
    norb: usize,
) -> (SpinDeterminantState, SpinDeterminantState) {
    (
        SpinDeterminantState::new(reference.alpha, det.alpha, norb),
        SpinDeterminantState::new(reference.beta, det.beta, norb),
    )
}

/// Probability that the spawning step proposes projectee `\mu` from cluster `C`,
/// `P(\mu|C) = 2^{-s}\sum_{R\subseteq C} P_{\text{gen}}(\hat\tau_L^\dagger D_\mu|\hat\tau_R{}^z\Psi)`,
/// summing over the partitions whose right part collapses to `|D_n\rangle` and whose left part
/// inverts `|D_\mu\rangle` to a distinct single or double excitation `|D_m\rangle` of it.
/// # Arguments:
/// - `block`: Block reference data.
/// - `excitors`: Cluster excitors.
/// - `projectee`: Proposed projectee.
/// - `pgen`: Uniform connection probability.
/// # Returns:
/// - `f64`: Proposal probability.
fn proposal(
    block: &LinkedBlock<'_>,
    excitors: &[Excitor],
    projectee: &ExcitedDeterminant,
    pgen: f64,
) -> f64 {
    let s = excitors.len();
    let mut p = 0.0;
    for left in 0..1usize << s {
        let right = (0..s)
            .filter(|k| left >> k & 1 == 0)
            .try_fold(block.reference, |det, k| Some(excitors[k].apply(&det)?.0));
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

/// Perform one linked attempt for a selected cluster: accumulate the reference projection
/// `\langle{}^z\Psi|\hat H|\hat\tau_C{}^z\Psi\rangle` of `\bar H_z`, apply death
/// `-\Delta\tau w V_C` on the collapsed cluster, and spawn `-\Delta\tau w V_\mu/P(\mu|C)` on one
/// sampled projectee.
/// # Arguments:
/// - `data`: NOCI data holding the parent frames and MO caches.
/// - `block`: Block reference data.
/// - `cluster`: Cluster excitors and weight `w`.
/// - `dt`: Time step `\Delta\tau`.
/// - `rng`: Random-number generator.
/// - `scratch`: Wick scratch, unused by same-frame elements.
/// - `out`: Spawned population changes.
/// # Returns:
/// - `f64`: Contribution `w\langle{}^z\Psi|\hat H|\hat\tau_C{}^z\Psi\rangle` to `N_z H^e_{zz}`.
pub(super) fn linked_attempt(
    data: &NOCIData<'_, f64>,
    block: &LinkedBlock<'_>,
    cluster: (&[Excitor], f64),
    dt: f64,
    rng: &mut QmcRng,
    scratch: &mut WickScratchSpin<f64>,
    out: &mut Vec<(ExcitorKey, f64)>,
) -> f64 {
    let (excitors, weight) = cluster;
    let reference = &block.reference;
    let key = |det: &ExcitedDeterminant| (block.index, det.alpha, det.beta);
    let mut energy = 0.0;

    // Reference projection and death on the collapsed cluster.
    if let Some((collapsed, sign)) = collapse(reference, excitors) {
        let r = rank(reference, &collapsed);
        if r <= 2 {
            energy = weight * sign * calculate_hs_excited(data, (reference, &collapsed), scratch).0;
        }
        if (1..=block.cluster_rank).contains(&r) {
            let v = commutator(data, block, excitors, &collapsed, scratch);
            out.push((key(&collapsed), -dt * weight * v));
        }
    }

    // Spawning: one partition, one Hamiltonian connection, then the left part.
    let s = excitors.len();
    let left = rng.gen_range(0..1usize << s);
    let Some(source) = (0..s)
        .filter(|k| left >> k & 1 == 0)
        .try_fold(*reference, |det, k| Some(excitors[k].apply(&det)?.0))
    else {
        return energy;
    };
    let Some((connection, pgen)) = block.generator.sample(rng) else {
        return energy;
    };
    let (alpha, beta) = components(reference, &source, block.norb);
    let (ca, cb) = connection.child_occupations(&alpha, &beta);
    let target = ExcitedDeterminant {
        parent: reference.parent,
        alpha: ca,
        beta: cb,
    };
    let Some(projectee) = (0..s)
        .filter(|k| left >> k & 1 == 1)
        .try_fold(target, |det, k| Some(excitors[k].apply(&det)?.0))
    else {
        return energy;
    };
    let r = rank(reference, &projectee);
    if r == 0 || r > block.cluster_rank {
        return energy;
    }

    let v = commutator(data, block, excitors, &projectee, scratch);
    if v != 0.0 {
        let p = proposal(block, excitors, &projectee, pgen);
        out.push((key(&projectee), -dt * weight * v / p));
    }
    energy
}
