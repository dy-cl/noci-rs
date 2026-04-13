// stochastic/excit.rs
use rand::rngs::SmallRng;
use rand::Rng;

use crate::nonorthogonalwicks::WickScratchSpin;
use crate::noci::NOCIData;
use super::state::{Shifts, HeatBath};

use super::propagate::{find_hs, coupling};

/// Initialise exact heat-bath excitation generation data for determinant `gamma`.
/// # Arguments:
/// - `gamma`: Parent determinant index.
/// - `shifts`: Current non-overlap and overlap-transformed shifts.
/// - `data`: Immutable stochastic propagation data.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `HeatBath`: Precomputed heat-bath excitation generation data for determinant `gamma`.
pub(in crate::stochastic) fn init_heat_bath(gamma: usize, shifts: Shifts, data: &NOCIData<'_>, scratch: &mut WickScratchSpin) -> HeatBath {
    let ndets = data.basis.len();
    // \Sum_{\Lambda \neq \Gamma} |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma} 
    let mut sumlg = 0.0_f64;
    // Cumulative sums A_n = \sum_{i=1}^n |H_{i\Gamma} - E_s^S(\tau)S_{i\Gamma}|. 
    let mut cumulatives: Vec<f64> = Vec::new();
    // Corresponding Lambda indices to the cumulatives.
    let mut lambdas: Vec<usize> = Vec::new();
    // H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma} for a given Lambda. 
    let mut ks: Vec<f64> = Vec::new();

    cumulatives.reserve(ndets - 1);
    lambdas.reserve(ndets - 1);
    ks.reserve(ndets - 1);

    for lambda in 0..ndets {
        if lambda == gamma {continue;}
        let (hlg, slg) = find_hs(data, lambda, gamma, scratch);
        let k = coupling(hlg, slg, shifts.es_s, shifts.es, &data.input.prop_ref().propagator);

        sumlg += k.abs();
        cumulatives.push(sumlg);
        lambdas.push(lambda);
        ks.push(k);
    }
    HeatBath {sumlg, cumulatives, lambdas, ks}
}

/// Propose a determinant for spawning using a uniform excitation scheme.
/// # Arguments:
/// - `gamma`: Index of determinant being spawned from.
/// - `shifts`: Current non-overlap and overlap-transformed shifts.
/// - `data`: Immutable stochastic propagation data.
/// - `rng`: Random number generator.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `(f64, f64, usize)`: Generation probability, coupling, and selected determinant index.
pub(in crate::stochastic) fn pgen_uniform(gamma: usize, shifts: Shifts, data: &NOCIData<'_>, rng: &mut SmallRng, scratch: &mut WickScratchSpin) -> (f64, f64, usize) {
    let ndets = data.basis.len();
    // Sample Lambda uniformly from all dets except Gamma. If Lambda index is the same or
    // more than the Gamma index we map it back into the full index set.
    let mut lambda = rng.gen_range(0..(ndets - 1));
    if lambda >= gamma {lambda += 1;}

    let (hlg, slg) = find_hs(data, lambda, gamma, scratch);
    let k = coupling(hlg, slg, shifts.es_s, shifts.es, &data.input.prop_ref().propagator);
    // Uniform generation probability
    let pgen = 1.0 / ((ndets - 1) as f64);
    (pgen, k, lambda) 
}

/// Propose a determinant for spawning using the exact heat-bath excitation scheme.
/// # Arguments:
/// - `gamma`: Index of determinant being spawned from.
/// - `shifts`: Current non-overlap and overlap-transformed shifts.
/// - `data`: Immutable stochastic propagation data.
/// - `rng`: Random number generator.
/// - `hb`: Precomputed heat-bath excitation generation data.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `(f64, f64, usize)`: Generation probability, coupling, and selected determinant index.
pub(in crate::stochastic) fn pgen_heat_bath(gamma: usize, shifts: Shifts, data: &NOCIData<'_>, rng: &mut SmallRng, hb: &HeatBath, scratch: &mut WickScratchSpin) -> (f64, f64, usize) {
    let ndets = data.basis.len();
    // If \Sum_{\Lambda \neq \Gamma} |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma} (sumlg) 
    // is zero (unsure how likely this is) then fallback to uniform distribution.
    if hb.sumlg == 0.0 {
        let mut lambda = rng.gen_range(0..(ndets - 1));
        if lambda >= gamma {lambda += 1;}
        let (hlg, slg) = find_hs(data, lambda, gamma, scratch);
        let k = coupling(hlg, slg, shifts.es_s, shifts.es, &data.input.prop_ref().propagator);
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
    let i = match hb.cumulatives.binary_search_by(|x| x.partial_cmp(&target).unwrap()) {Ok(i) => i, Err(i) => i};

    // Return P_{\text{gen}}, H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma} and index of Lambda.
    let lambda = hb.lambdas[i];
    let k = hb.ks[i];
    let pgen = k.abs() / hb.sumlg;
    (pgen, k, lambda)
}

