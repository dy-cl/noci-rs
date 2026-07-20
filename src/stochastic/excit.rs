// stochastic/excit.rs
use rand::Rng;
use rand::rngs::SmallRng;

use super::state::HeatBath;
use crate::noci::NOCIData;
use crate::nonorthogonalwicks::WickScratchSpin;

use super::common::find_hs;

/// Evaluate the shifted off-diagonal coupling
/// T_{\Lambda\Gamma}(\Delta\tau) = H_{\Lambda\Gamma} - E_s(\Delta \tau) S_{\Lambda\Gamma}).
/// # Arguments:
/// - `hlg`: Hamiltonian matrix element H_{\Lambda\Gamma}.
/// - `slg`: Overlap matrix element S_{\Lambda\Gamma}.
/// - `shift`: Current population-control shift E_s(\Delta \tau).
/// # Returns:
/// - `f64`: Shifted coupling (T_{\Lambda\Gamma}(\Delta \tau).
/// # Arguments:
/// - `lambda`: Child determinant index \(\Lambda\).
/// - `gamma`: Source determinant index \(\Gamma\).
/// - `shift`: Current population-control shift \(E_s\).
/// - `data`: Immutable stochastic propagation data.
/// - `scratch`: Scratch space for nonorthogonal Wick quantities.
/// # Returns:
/// - `f64`: Shifted coupling \(T_{\Lambda\Gamma}\).
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
        && (lambda_det.oa ^ gamma_det.oa).count_ones()
            + (lambda_det.ob ^ gamma_det.ob).count_ones()
            > 4
    {
        return 0.0;
    }

    let (hlg, slg) = find_hs(data, lambda, gamma, scratch);

    hlg - shift * slg
}

/// Construct exact heat-bath excitation-generation data for determinant
/// \(\Gamma\). For every \Lambda \neq \Gamma, the heat-bath weight is
/// w_{\Lambda\Gamma} = |T_{\Lambda\Gamma}(\Delta \tau)|. The total weight is
/// W_\Gamma = \sum_{\Lambda \neq \Gamma}w_{\Lambda\Gamma}.
/// # Arguments:
/// - `gamma`: Parent determinant index \Gamma\.
/// - `shift`: Current population-control shift E_s(\Delta \tau).
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
    // Total weight W_Gamma = \sum_{\Lambda != \Gamma} |T_{\Lambda \Gamma}(\Delta \tau)|.
    let mut sumlg = 0.0_f64;
    // Cumulative weights A_n = \sum_{i = 1}^n |T_{i \Gamma}(\Delta \tau)|.
    let mut cumulatives: Vec<f64> = Vec::new();
    // Corresponding Lambda indices to the cumulatives.
    let mut lambdas: Vec<usize> = Vec::new();
    // Signed shifted couplings T_{\Lambda \Gamma}(\Delta \tau).
    let mut ks: Vec<f64> = Vec::new();

    cumulatives.reserve(ndets - 1);
    lambdas.reserve(ndets - 1);
    ks.reserve(ndets - 1);

    for lambda in 0..ndets {
        if lambda == gamma {
            continue;
        }
        let k = coupling(lambda, gamma, shift, data, scratch);

        sumlg += k.abs();
        cumulatives.push(sumlg);
        lambdas.push(lambda);
        ks.push(k);
    }
    HeatBath {
        sumlg,
        cumulatives,
        lambdas,
        ks,
    }
}

/// Sample an off-diagonal child determinant uniformly. The generation probability is
/// P_{\mathrm{Gen}}(\Lambda|\Gamma) = 1/(N_{\mathrm{det}}-1)\) for every \Lambda \neq \Gamma\.
/// # Arguments:
/// - `gamma`: Parent determinant index \Gamma.
/// - `shift`: Current population-control shift E_s(\Delta \tau).
/// - `data`: Immutable stochastic propagation data.
/// - `rng`: Random-number generator.
/// - `scratch`: Scratch space for nonorthogonal Wick quantities.
/// # Returns:
/// - `(f64, f64, usize)`: Generation probability P_{\mathrm{Gen}}(\Lambda|\Gamma),
///   shifted coupling T_{\Lambda\Gamma}(\Delta \tau), and sampled child index \Lambda.
pub(in crate::stochastic) fn pgen_uniform(
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    rng: &mut SmallRng,
    scratch: &mut WickScratchSpin<f64>,
) -> (f64, f64, usize) {
    let ndets = data.basis.len();
    // Sample Lambda uniformly from all dets except Gamma. If Lambda index is the same or
    // more than the Gamma index we map it back into the full index set.
    let mut lambda = rng.gen_range(0..(ndets - 1));
    if lambda >= gamma {
        lambda += 1;
    }

    let k = coupling(lambda, gamma, shift, data, scratch);
    // Uniform generation probability
    let pgen = 1.0 / ((ndets - 1) as f64);
    (pgen, k, lambda)
}

/// Sample an off-diagonal child determinant from the exact heat-bath
/// distribution. For nonzero total weight, P_{\mathrm{gen}}(\Lambda\Gamma)
/// = |T_{\Lambda\Gamma}(\Delta \tau)| / W_\Gamma, if W_\Gamma = 0,
/// the function falls back to uniform sampling.
/// # Arguments:
/// - `gamma`: Parent determinant index \(\Gamma\).
/// - `shift`: Current population-control shift \(E_s\), used by the
///   uniform fallback.
/// - `data`: Immutable stochastic propagation data.
/// - `rng`: Random-number generator.
/// - `hb`: Exact heat-bath data constructed for the same determinant
///   and shift.
/// - `scratch`: Scratch space used for nonorthogonal Wick's quantities.
/// # Returns:
/// - `(f64, f64, usize)`: Generation probability P_{\mathrm{Gen}}(\Lambda|\Gamma),
///   shifted coupling T_{\Lambda\Gamma}(\Delta \tau), and sampled child index \Lambda.
pub(in crate::stochastic) fn pgen_heat_bath(
    gamma: usize,
    shift: f64,
    data: &NOCIData<'_, f64>,
    rng: &mut SmallRng,
    hb: &HeatBath,
    scratch: &mut WickScratchSpin<f64>,
) -> (f64, f64, usize) {
    let ndets = data.basis.len();
    // If \Sum_{\Lambda \neq \Gamma} |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma} (sumlg)
    // is zero (unsure how likely this is) then fallback to uniform distribution.
    if hb.sumlg == 0.0 {
        let mut lambda = rng.gen_range(0..(ndets - 1));
        if lambda >= gamma {
            lambda += 1;
        }
        let k = coupling(lambda, gamma, shift, data, scratch);
        let pgen = 1.0 / ((ndets - 1) as f64);
        return (pgen, k, lambda);
    }

    // We want P_{\text{gen}} = |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}| /
    // \Sum_{\Lambda \neq \Gamma} |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}|. We
    // choose a number (target) uniformly in \Sum_{\Lambda \neq \Gamma} |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}
    // (sumlg) and define the sequence of cumulative sums:
    //      A_1 = |H_{1\Gamma} - E_s^S(\tau)S_{1\Gamma}|
    //      ..
    //      A_n = \sum_{i=1}^n |H_{i\Gamma} - E_s^S(\tau)S_{i\Gamma}|.
    // The probability that target is between A_{j-1} and A_{j} is:
    //     |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}| /
    //     \Sum_{\Lambda \neq \Gamma} |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}|,
    // which is exactly the distribution we want to sample. We can therefore add the A's
    // until we pass the target at which point the last tested \Lambda is the one chosen
    // with the correct probability, and we can compute k and pgen accordingly.
    let target = rng.gen_range(0.0..hb.sumlg);
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

    // Return P_{\text{gen}}, H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma} and index of Lambda.
    let lambda = hb.lambdas[i];
    let k = hb.ks[i];
    let pgen = k.abs() / hb.sumlg;
    (pgen, k, lambda)
}
