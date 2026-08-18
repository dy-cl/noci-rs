// stochastic/excit.rs
// External crate imports.
use rand::Rng;
use rand::rngs::SmallRng;

// Crate-root imports.
use crate::noci::NOCIData;
use crate::nonorthogonalwicks::WickScratchSpin;

// Parent/sibling imports.
use super::common::find_hs;
use super::state::HeatBath;

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
    rng: &mut SmallRng,
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
