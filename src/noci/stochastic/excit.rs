// noci/stochastic/excit.rs

// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::Rng;

// Crate-root imports.
use crate::determinant::{NOCIIndex, OrthogonalConnection};
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{
    MOCache, NOCIData, OrthogonalHamiltonianScratch, calculate_h_pairs_orthogonal_batched,
};
use crate::input::{ExcitationGen, Input};
use crate::qmc::QmcRng;
use crate::qmc::excitgen::{
    FrameSource, FrameSpin, OrthogonalUniformGenerator, PchbGenerator, PchbScratch,
};

// Parent/sibling imports.
use super::common::find_hs;
use super::state::{HeatBath, OverlapDerivativeSums, PropagationState, QMCRunInfo};

/// Proposal distribution for parent-orthogonal BApply connections.
pub(in crate::noci::stochastic) enum AuxiliaryGenerator {
    /// Uniform proposal over the system-wide connection table.
    Uniform(OrthogonalUniformGenerator),
    /// Precomputed heat-bath doubles and contracted-weight singles.
    Pchb(PchbGenerator),
}

impl AuxiliaryGenerator {
    /// Prepare source-dependent sampling state before all attempts from one source.
    /// The uniform proposal is source independent and needs no preparation.
    /// # Arguments:
    /// - `self`: Selected connection generator.
    /// - `source`: Retained source determinant `x`.
    /// - `data`: NOCI data holding the determinant space.
    /// - `scratch`: Worker-local per-source PCHB state.
    /// # Returns:
    /// - `()`: Updates `scratch` for PCHB generation.
    #[inline(always)]
    pub(in crate::noci::stochastic) fn prepare(
        &self,
        source: NOCIIndex,
        data: &NOCIData<'_, f64>,
        scratch: &mut PchbScratch,
    ) {
        if let Self::Pchb(generator) = self {
            let alpha = data.space.alpha(source);
            let beta = data.space.beta(source);
            let frame = FrameSource {
                parent: data.space.state(source).parent,
                reference: alpha.excitation.holes == 0 && beta.excitation.holes == 0,
                alpha: FrameSpin {
                    occupied: &alpha.occupied,
                    virtuals: &alpha.virtuals,
                },
                beta: FrameSpin {
                    occupied: &beta.occupied,
                    virtuals: &beta.virtuals,
                },
            };
            generator.prepare(&frame, scratch);
        }
    }

    /// Sample one source-relative connection and its exact `P_\mathrm{gen}(D|x)`.
    /// # Arguments:
    /// - `self`: Selected connection generator.
    /// - `scratch`: Per-source PCHB state from `prepare`.
    /// - `rng`: Thread-local random-number generator.
    /// # Returns:
    /// - `Option<(OrthogonalConnection, f64)>`: Connection and generation probability, or `None`
    ///   for an empty or rejected draw.
    #[inline(always)]
    pub(in crate::noci::stochastic) fn sample<R: Rng + ?Sized>(
        &self,
        scratch: &PchbScratch,
        rng: &mut R,
    ) -> Option<(OrthogonalConnection, f64)> {
        match self {
            Self::Uniform(generator) => generator.sample(rng),
            Self::Pchb(generator) => generator.sample(scratch, rng),
        }
    }
}

/// Construct the PCHB generator of every parent, installing the exact reference singles
/// `|H_{a\leftarrow i,0}|` of each retained parent reference evaluated with the orthogonal
/// Slater-Condon kernel.
/// # Arguments:
/// - `data`: NOCI data holding the determinant space and parent MO caches.
/// - `mocache`: Parent MO integral caches, one per parent reference.
/// # Returns:
/// - `PchbGenerator`: PCHB generator shared read-only by every worker.
pub(in crate::noci::stochastic) fn pchb_generator(
    data: &NOCIData<'_, f64>,
    mocache: &[MOCache<f64>],
) -> PchbGenerator {
    let mut generator = PchbGenerator::new(mocache);
    let mut scratch = OrthogonalHamiltonianScratch::new(mocache.len());

    // A parent reference is the retained determinant without parent-relative holes.
    for reference in (0..data.space.len()).map(NOCIIndex) {
        let parent = data.space.state(reference).parent;
        let (alpha, beta) = data.space.excitations(reference);
        if alpha.holes != 0 || beta.holes != 0 || parent >= mocache.len() {
            continue;
        }

        // Enumerate every reference single in source-rank form with its orbital labels.
        let alpha = data.space.alpha(reference);
        let beta = data.space.beta(reference);
        let mut labels = Vec::new();
        let mut pairs = Vec::new();
        for (x, &i) in alpha.occupied.iter().enumerate() {
            for (y, &a) in alpha.virtuals.iter().enumerate() {
                labels.push((true, i as usize, a as usize));
                pairs.push((
                    reference,
                    OrthogonalConnection::AlphaSingle {
                        occupied: x as u8,
                        virtual_: y as u8,
                    },
                ));
            }
        }
        for (x, &i) in beta.occupied.iter().enumerate() {
            for (y, &a) in beta.virtuals.iter().enumerate() {
                labels.push((false, i as usize, a as usize));
                pairs.push((
                    reference,
                    OrthogonalConnection::BetaSingle {
                        occupied: x as u8,
                        virtual_: y as u8,
                    },
                ));
            }
        }

        // Evaluate `H_{a\leftarrow i,0}` once and store magnitudes by orbital label.
        let mut values = vec![0.0; pairs.len()];
        calculate_h_pairs_orthogonal_batched(data, &pairs, &mut scratch, &mut values);
        let na = mocache[parent].ha.nrows();
        let nb = mocache[parent].hb.nrows();
        let mut exact_a = vec![0.0; na * na];
        let mut exact_b = vec![0.0; nb * nb];
        for (&(spin, i, a), value) in labels.iter().zip(values) {
            if spin {
                exact_a[i * na + a] = value.abs();
            } else {
                exact_b[i * nb + a] = value.abs();
            }
        }
        generator.set_reference_singles(parent, (exact_a, exact_b));
    }

    generator
}

/// Evaluate the shifted off-diagonal coupling
/// `T_{xw}(\Delta\tau) = H_{xw} - E_s(\Delta \tau) S_{xw}.`
/// # Arguments:
/// - `lambda`: Child determinant index `x`.
/// - `gamma`: Source determinant index `w`.
/// - `shift`: Current population-control shift `E_s`.
/// - `data`: Immutable stochastic propagation data.
/// - `scratch`: Scratch space for nonorthogonal Wick quantities.
/// # Returns:
/// - `f64`: Shifted coupling `T_{xw}`.
pub(in crate::noci::stochastic) fn coupling(
    lambda: usize,
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    scratch: &mut WickScratchSpin<f64>,
) -> f64 {
    let lambda_state = data.space.state(crate::determinant::NOCIIndex(lambda));
    let gamma_state = data.space.state(crate::determinant::NOCIIndex(gamma));
    let (lambda_oa, lambda_ob) = data
        .space
        .occupations(crate::determinant::NOCIIndex(lambda));
    let (gamma_oa, gamma_ob) = data.space.occupations(crate::determinant::NOCIIndex(gamma));

    if lambda_state.parent == gamma_state.parent
        && (lambda_oa ^ gamma_oa).count_ones() + (lambda_ob ^ gamma_ob).count_ones() > 4
    {
        return 0.0;
    }

    let (hxw, sxw) = find_hs(data, lambda, gamma, scratch);

    hxw - shift * sxw
}

/// Construct exact heat-bath excitation-generation data for determinant
/// `w. For every x \neq w, the heat-bath weight is`
/// `w_{xw} = |T_{xw}(\Delta \tau)|. The total weight is`
/// `W_w = \sum_{x \neq w}w_{xw}.`
/// # Arguments:
/// - `gamma`: Parent determinant index w.
/// - `shift`: `Current population-control shift E_s(\Delta \tau).`
/// - `data`: Immutable stochastic propagation data.
/// - `scratch`: Scratch space for nonorthogonal Wick quantities.
/// # Returns:
/// - `HeatBath`: Couplings and cumulative weights for sampling children.
/// # References
/// - Coupling-magnitude heat-bath proposal (adapted here by exact enumeration):
///   Holmes, Changlani, and Umrigar,
///   *J. Chem. Theory Comput.* **12**, 1561 (2016),
///   [doi:10.1021/acs.jctc.5b01170](https://doi.org/10.1021/acs.jctc.5b01170).
pub(in crate::noci::stochastic) fn init_heat_bath(
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    scratch: &mut WickScratchSpin<f64>,
) -> HeatBath {
    let ndets = data.space.len();
    // Total weight W_w = \sum_{x != w} |T_{x w}(\Delta \tau)|.
    let mut sumxw = 0.0_f64;
    // Cumulative weights A_n = \sum_{i = 1}^n |T_{i w}(\Delta \tau)|.
    let mut cumulatives: Vec<f64> = Vec::new();
    // Corresponding child indices to the cumulatives.
    let mut lambdas: Vec<usize> = Vec::new();
    // Signed shifted couplings T_{x w}(\Delta \tau).
    let mut ks: Vec<f64> = Vec::new();

    cumulatives.reserve(ndets - 1);
    lambdas.reserve(ndets - 1);
    ks.reserve(ndets - 1);

    for lambda in 0..ndets {
        if lambda == gamma {
            continue;
        }
        let k = coupling(lambda, gamma, shift, data, scratch);

        sumxw += k.abs();
        cumulatives.push(sumxw);
        lambdas.push(lambda);
        ks.push(k);
    }
    HeatBath {
        sumxw,
        cumulatives,
        lambdas,
        ks,
    }
}

/// Sample an off-diagonal child determinant from the exact heat-bath
/// `distribution. For nonzero total weight, P_{\mathrm{gen}}(xw)`
/// `= |T_{xw}(\Delta \tau)| / W_w, if W_w = 0,`
/// the function falls back to uniform sampling.
/// # Arguments:
/// - `gamma`: Parent determinant index `w`.
/// - `shift`: Current population-control shift `E_s`, used by the
///   uniform fallback.
/// - `data`: Immutable stochastic propagation data.
/// - `rng`: Random-number generator.
/// - `hb`: Exact heat-bath data constructed for the same determinant
///   and shift.
/// - `scratch`: Scratch space used for nonorthogonal Wick's quantities.
/// # Returns:
/// - `(f64, f64, usize)`: `Generation probability P_{\mathrm{Gen}}(x|w),`
///   `shifted coupling T_{xw}(\Delta \tau), and sampled child index x.`
pub(in crate::noci::stochastic) fn pgen_heat_bath(
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    rng: &mut QmcRng,
    hb: &HeatBath,
    scratch: &mut WickScratchSpin<f64>,
) -> (f64, f64, usize) {
    let ndets = data.space.len();
    // If \Sum_{x \neq w} |H_{xw} - E_s^S(\tau)S_{xw} (sumxw)
    // is zero (unsure how likely this is) then fallback to uniform distribution.
    if hb.sumxw == 0.0 {
        let mut lambda = rng.gen_range(0..(ndets - 1));
        if lambda >= gamma {
            lambda += 1;
        }
        let k = coupling(lambda, gamma, shift, data, scratch);
        let pgen = 1.0 / ((ndets - 1) as f64);
        return (pgen, k, lambda);
    }

    // We want P_{\text{gen}} = |H_{xw} - E_s^S(\tau)S_{xw}| /
    // \Sum_{x \neq w} |H_{xw} - E_s^S(\tau)S_{xw}|. We
    // choose a number (target) uniformly in \Sum_{x \neq w} |H_{xw} - E_s^S(\tau)S_{xw}
    // (sumxw) and define the sequence of cumulative sums:
    //      A_1 = |H_{1w} - E_s^S(\tau)S_{1w}|
    //      ..
    //      A_n = \sum_{i=1}^n |H_{iw} - E_s^S(\tau)S_{iw}|.
    // The probability that target is between A_{j-1} and A_{j} is:
    //     |H_{xw} - E_s^S(\tau)S_{xw}| /
    //     \Sum_{x \neq w} |H_{xw} - E_s^S(\tau)S_{xw}|,
    // which is exactly the distribution we want to sample. We can therefore add the A's
    // until we pass the target at which point the last tested x is the one chosen
    // with the correct probability, and we can compute k and pgen accordingly.
    let target = rng.gen_range(0.0..hb.sumxw);
    // Find first index where the cumulative sum is more than the target. Binary search returns
    // Result <usize, usize> where Ok(i) is element exactly equal to target and Err(i) is insertion
    // index where target would be inserted to keep array sorted. In both cases this is what we
    // want.
    let i = match hb
        .cumulatives
        .binary_search_by(|x| x.partial_cmp(&target).unwrap())
    {
        Ok(i) => i,
        Err(i) => i,
    };

    // Return P_{\text{gen}}, H_{xw} - E_s^S(\tau)S_{xw} and sampled child index.
    let lambda = hb.lambdas[i];
    let k = hb.ks[i];
    let pgen = k.abs() / hb.sumxw;
    (pgen, k, lambda)
}

/// Apply one report-level stochastic Newton update to the overlap mixture probability.
/// The accumulated derivatives estimate `M_2'(p)` and `M_2''(p)` for
/// `M_2(p) = \sum_w A_{wx}^2/q_p(w|x)`, so the Newton update is
/// `p_{\text{next}} = p - M_2'(p)/M_2''(p)`.
/// # Arguments:
/// - `state`: Current propagation state containing the mutable overlap mixture probability.
/// - `derivatives`: Report-local derivative sums to reduce and then clear.
/// - `input`: User input options.
/// - `run`: Rank-local propagation metadata.
/// - `world`: MPI communicator.
/// # Returns:
/// - `()`: Updates `state.overlap_weight` for the next report when the Newton step is finite.
pub(in crate::noci::stochastic) fn update_overlap_weight(
    state: &mut PropagationState,
    derivatives: &mut OverlapDerivativeSums,
    input: &Input,
    run: &QMCRunInfo,
    world: &impl Communicator,
) {
    let qmc = input.qmc.as_ref().unwrap();
    if !qmc.optimise_overlap_weight || qmc.excitation_gen != ExcitationGen::OverlapWeighted {
        *derivatives = OverlapDerivativeSums::default();
        return;
    }

    let local = [derivatives.gradient, derivatives.hessian];
    let mut global = [0.0; 2];
    if run.nranks == 1 {
        global = local;
    } else {
        world.all_reduce_into(&local, &mut global, SystemOperation::sum());
    }

    let gradient = global[0];
    let hessian = global[1];
    if gradient.is_finite() && hessian.is_finite() && hessian > 0.0 {
        // Convex stochastic Newton step:
        // \Delta p = -G/H, with G \approx M_2'(p) and H \approx M_2''(p).
        let delta = -gradient / hessian;
        if delta.is_finite() {
            let candidate = state.overlap_weight + delta;
            state.overlap_weight = if candidate < 0.0 {
                0.0
            } else if candidate >= 1.0 {
                1.0 - f64::EPSILON
            } else {
                candidate
            };
        }
    }

    *derivatives = OverlapDerivativeSums::default();
}
