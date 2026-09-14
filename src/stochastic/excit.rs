// stochastic/excit.rs
// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::Rng;

// Crate-root imports.
use crate::DetState;
use crate::input::{ExcitationGen, Input};
use crate::noci::{MOCache, NOCIData, OrthogonalDetState};
use crate::nonorthogonalwicks::WickScratchSpin;

// Parent/sibling imports.
use super::common::find_hs;
use super::state::{HeatBath, OverlapDerivativeSums, PropagationState, QMCRunInfo, QmcRng};

/// Return the orbital index of the `rank`th set bit.
/// Callers guarantee `rank < bits.count_ones()`.
/// # Arguments:
/// - `bits`: Orbital bit mask.
/// - `rank`: Zero-based rank among set bits.
/// # Returns
/// - `usize`: Selected orbital index.
#[inline(always)]
fn select_set_bit(
    mut bits: u128,
    mut rank: usize,
) -> usize {
    // Remove the `rank` lowest set bits. The next trailing set bit is the selected orbital.
    while rank != 0 {
        bits &= bits - 1;
        rank -= 1;
    }

    bits.trailing_zeros() as usize
}

/// Sample uniformly from all determinants coupled to `source` by the orthogonal-parent
/// one- and two-electron Hamiltonian.
/// The child classes are alpha singles, beta singles, alpha-alpha doubles, beta-beta doubles,
/// and alpha-beta doubles. Every distinct connected determinant has
/// `P_gen(D|x) = 1 / N_conn`.
/// # Arguments:
/// - `source`: NOCI source determinant `|Phi_x>`.
/// - `cache`: MO-basis integral cache defining the parent orbital dimensions.
/// - `rng`: Random-number generator.
/// # Returns
/// - `Option<(f64, OrthogonalDetState)>`: Generation probability and sampled orthogonal
///   determinant, or `None` when the source has no connected determinant.
pub(in crate::stochastic) fn pgen_orthogonal_uniform(
    source: &DetState<f64>,
    cache: &MOCache<f64>,
    rng: &mut QmcRng,
) -> Option<(f64, OrthogonalDetState)> {
    let nmoa = cache.ha.nrows();
    let nmob = cache.hb.nrows();

    // `V_\sigma = {0,\ldots,n_\mathrm{mo}^\sigma-1} \setminus O_\sigma`.
    // Mask unused bits above the parent MO dimension before complementing the occupation.
    let vira = !source.oa
        & if nmoa == 128 {
            u128::MAX
        } else {
            (1u128 << nmoa) - 1
        };
    let virb = !source.ob
        & if nmob == 128 {
            u128::MAX
        } else {
            (1u128 << nmob) - 1
        };

    let noa = source.oa.count_ones() as usize;
    let nob = source.ob.count_ones() as usize;
    let nva = vira.count_ones() as usize;
    let nvb = virb.count_ones() as usize;

    // `N_{\alpha1} = n_{o\alpha} n_{v\alpha}`,
    // `N_{\beta1} = n_{o\beta} n_{v\beta}`,
    // `N_{\alpha\alpha} = C(n_{o\alpha},2) C(n_{v\alpha},2)`,
    // `N_{\beta\beta} = C(n_{o\beta},2) C(n_{v\beta},2)`,
    // `N_{\alpha\beta} = n_{o\alpha}n_{v\alpha}n_{o\beta}n_{v\beta}`.
    let nas = noa * nva;
    let nbs = nob * nvb;

    let naa = if noa >= 2 && nva >= 2 {
        (noa * (noa - 1) / 2) * (nva * (nva - 1) / 2)
    } else {
        0
    };

    let nbb = if nob >= 2 && nvb >= 2 {
        (nob * (nob - 1) / 2) * (nvb * (nvb - 1) / 2)
    } else {
        0
    };

    let nab = nas * nbs;
    let nconnected = nas + nbs + naa + nbb + nab;

    if nconnected == 0 {
        return None;
    }

    // Select class `c` with `P(c) = N_c/N_\mathrm{conn}`. Uniform sampling inside the selected
    // class then gives `P_\mathrm{gen}(D|x) = 1/N_\mathrm{conn}` for every connected determinant.
    let mut class = rng.gen_range(0..nconnected);
    let mut oa = source.oa;
    let mut ob = source.ob;

    if class < nas {
        let i = select_set_bit(source.oa, rng.gen_range(0..noa));
        let a = select_set_bit(vira, rng.gen_range(0..nva));

        oa &= !(1u128 << i);
        oa |= 1u128 << a;
    } else {
        class -= nas;

        if class < nbs {
            let i = select_set_bit(source.ob, rng.gen_range(0..nob));
            let a = select_set_bit(virb, rng.gen_range(0..nvb));

            ob &= !(1u128 << i);
            ob |= 1u128 << a;
        } else {
            class -= nbs;

            if class < naa {
                // Ordered distinct ranks represent each unordered occupied pair twice and each
                // unordered virtual pair twice. The four representations therefore cancel
                // exactly, leaving every distinct same-spin double uniformly distributed.
                let ir = rng.gen_range(0..noa);
                let mut jr = rng.gen_range(0..noa - 1);
                if jr >= ir {
                    jr += 1;
                }

                let ar = rng.gen_range(0..nva);
                let mut br = rng.gen_range(0..nva - 1);
                if br >= ar {
                    br += 1;
                }

                let i = select_set_bit(source.oa, ir);
                let j = select_set_bit(source.oa, jr);
                let a = select_set_bit(vira, ar);
                let b = select_set_bit(vira, br);

                oa &= !((1u128 << i) | (1u128 << j));
                oa |= (1u128 << a) | (1u128 << b);
            } else {
                class -= naa;

                if class < nbb {
                    let ir = rng.gen_range(0..nob);
                    let mut jr = rng.gen_range(0..nob - 1);
                    if jr >= ir {
                        jr += 1;
                    }

                    let ar = rng.gen_range(0..nvb);
                    let mut br = rng.gen_range(0..nvb - 1);
                    if br >= ar {
                        br += 1;
                    }

                    let i = select_set_bit(source.ob, ir);
                    let j = select_set_bit(source.ob, jr);
                    let a = select_set_bit(virb, ar);
                    let b = select_set_bit(virb, br);

                    ob &= !((1u128 << i) | (1u128 << j));
                    ob |= (1u128 << a) | (1u128 << b);
                } else {
                    let i = select_set_bit(source.oa, rng.gen_range(0..noa));
                    let a = select_set_bit(vira, rng.gen_range(0..nva));
                    let j = select_set_bit(source.ob, rng.gen_range(0..nob));
                    let b = select_set_bit(virb, rng.gen_range(0..nvb));

                    oa &= !(1u128 << i);
                    oa |= 1u128 << a;

                    ob &= !(1u128 << j);
                    ob |= 1u128 << b;
                }
            }
        }
    }

    Some((
        1.0 / nconnected as f64,
        OrthogonalDetState {
            parent: source.parent,
            oa,
            ob,
        },
    ))
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
pub(in crate::stochastic) fn coupling(
    lambda: usize,
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    scratch: &mut WickScratchSpin<f64>,
) -> f64 {
    let lambda_det = &data.basis[lambda];
    let gamma_det = &data.basis[gamma];

    if lambda_det.parent == gamma_det.parent
        && (lambda_det.oa ^ gamma_det.oa).count_ones() + (lambda_det.ob ^ gamma_det.ob).count_ones()
            > 4
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
pub(in crate::stochastic) fn init_heat_bath(
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    scratch: &mut WickScratchSpin<f64>,
) -> HeatBath {
    let ndets = data.basis.len();
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
pub(in crate::stochastic) fn pgen_heat_bath(
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    rng: &mut QmcRng,
    hb: &HeatBath,
    scratch: &mut WickScratchSpin<f64>,
) -> (f64, f64, usize) {
    let ndets = data.basis.len();
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
pub(in crate::stochastic) fn update_overlap_weight(
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
