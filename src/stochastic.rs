// stochastic.rs
use std::time::{Instant, Duration};

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use mpi::topology::Communicator;
use mpi::collective::SystemOperation;
use mpi::traits::*;
use rayon::prelude::*;
use rand_distr::{Binomial, Distribution};

use crate::AoData;
use crate::SCFState;
use crate::input::{Input, Propagator, ExcitationGen};
use crate::nonorthogonalwicks::{WicksView, WickScratchSpin};
use crate::noci::MOCache;

use crate::noci::{calculate_s_pair, calculate_hs_pair};
use crate::mpiutils::{owner, local_walkers, communicate_spawn_updates, gather_all_walkers};

// Storage for stochastic propagation timings.
#[derive(Default)]
pub struct StochStepTimings {
    pub initialse_walkers: Duration,
    pub spawn_death_collect: Duration,
    pub acc_pack: Duration,
    pub mpi_exchange_updates: Duration,
    pub unpack_acc_recieved: Duration,
    pub apply_delta: Duration,
    pub update_p: Duration,
    pub calc_populations: Duration,
    pub eproj: Duration,
}

// Storage for walker information.
pub struct Walkers {
    // Signed population vector length n determinants.
    pop: Vec<i64>,
    // List of indices with non-zero population.
    occ: Vec<usize>,
    // Position of index in `occ` or usize::MAX if not present, length n determinants.
    pos: Vec<usize>,
}

impl Walkers {
    /// Construct empty walker object for n determinants.
    /// # Arguments: 
    /// - `n`: Number of determinants. 
    /// # Returns
    /// - `Walkers`: Empty walker storage for `n` determinants.
    pub fn new(n: usize) -> Self {
        Self {
            pop: vec![0; n],
            occ: Vec::new(),
            pos: vec![usize::MAX; n],
        }
    }
    
    /// Return the current population of determinant i.
    /// # Arguments: 
    /// - `self`: Object containing information about current walkers.
    /// - `i`: Determinant index of choice.
    /// # Returns
    /// - `i64`: Current signed walker population on determinant `i`.
    pub fn get(&self, i: usize) -> i64 {
        self.pop[i]
    }
    
    /// Return a list of occupied determinant indices.
    /// # Arguments:
    /// - `self`: Object containing information about current walkers.
    /// # Returns
    /// - `&[usize]`: Slice of occupied determinant indices.
    pub fn occ(&self) -> &[usize] {
        &self.occ
    }

    /// Add dn (change in population) to determinant i, modifying pop, occ and pos as required
    /// # Arguments:
    /// - `self`: Object containing information about current walkers.
    /// - `i`: Determinant index of choice. 
    /// - `dn`: Change in population of determinant i.
    /// # Returns
    /// - `()`: Updates the walker storage in place.
    pub fn add(&mut self, i: usize, dn: i64) {
        // No changes to be made.
        if dn == 0 {return;}
       
        unsafe {
            // Update population vector.
            let pop = self.pop.get_unchecked_mut(i);
            let old = *pop;
            let new = old + dn;
            *pop = new;
            
            // If old population of determinant i was 0 and we are introducing population we must
            // add determinant i to the occupied list and store its position in pos.
            if old == 0 && new != 0 {
                let p = self.occ.len();
                // Position of determinant i in occupied list is the current end.
                *self.pos.get_unchecked_mut(i) = p;
                self.occ.push(i);
                return;
            }
            // If old population of determinant i was not 0 and we have removed population we must
            // remove i from the occupied list and remove its position from pos.
            if old != 0 && new == 0 {
                // Find where i is in occ. Should not return usize::MAX.
                let p = *self.pos.get_unchecked(i);
                // Pop the last occupied determinant index from occ.
                let last = self.occ.pop().unwrap_unchecked();
                
                // If the popped element is not i, then i was somewhere in the middle of occ and we
                // must move the popped element (last) to position p where i used to be. The position of  
                // last is then updated in the position vector. If the popped element is i then we do
                // nothing as we have directly removed it by popping.
                if last != i {
                    // move last into position p
                    *self.occ.get_unchecked_mut(p) = last;
                    *self.pos.get_unchecked_mut(last) = p;
                }

                // Position i is no longer occupied.
                *self.pos.get_unchecked_mut(i) = usize::MAX;
            }
        }
    }

    /// Compute the total walker population 1-norm.
    /// # Arguments:
    /// - `self`: Object containing information about current walkers.
    /// # Returns
    /// - `i64`: Total walker population 1-norm.
    pub fn norm(&self) -> i64 {
        self.occ.iter().map(|&i| self.pop[i].abs()).sum()
    }
}

// Storage for Monte Carlo state. 
pub struct MCState {
    pub walkers: Walkers,
    pub delta: Vec<i64>, // Changes to walker population of length n determinants.
    pub changed: Vec<usize>, // Indices for which delta[i] != 0, i.e., determinants with changed population.
    pub pg: Vec<f64>, // Incrementally updated p_{\Gamma} = \sum_{\Omega} S_{\Gamma\Omega} N_{\Omega}.
    pub rng: SmallRng,
    pub excitation_hist: Option<ExcitationHist> // Histogrammed samples of P_{\text{Spawn}} for excit gen diagnostics.
}

struct ProjectedEnergyUpdate {
    iref: usize,
    num: f64,
    den: f64,
}

// Storage for population update communication across ranks. Is also used for computation of
// \tilde{N}_w. In this case we interpret dn as the total population.
#[repr(C)]
#[derive(Copy, Clone, Equivalence)]
pub struct PopulationUpdate {
    pub det: u64,
    pub dn: i64,
}

// Storage for heat-bath excitation related quantities 
pub struct HeatBath {
    pub sumlg: f64, 
    pub cumulatives: Vec<f64>, 
    pub lambdas: Vec<usize>, 
    pub ks: Vec<f64>,
}

// Storage for histogrammed data 
pub struct ExcitationHist {
    pub logmin: f64,
    pub logmax: f64,
    pub noverflow_low: u64,
    pub noverflow_high: u64,
    pub counts: Vec<u64>,
    pub nbins: usize,
    pub ntotal: u64,
}

impl ExcitationHist {
    /// Constructor for ExcitationHist object. Creates ExcitationHist with chosen parameters.
    /// # Arguments:
    /// - `logmin`: Minimum histogram value on a logarithmic scale. 
    /// - `logmax`: Maximum histogram value on a logarithmic scale.
    /// - `nbins`: Number of histogram bins.
    /// # Returns
    /// - `ExcitationHist`: Empty histogram with the requested parameters.
    pub fn new(logmin: f64, logmax: f64, nbins: usize) -> Self {
        Self {logmin, logmax, noverflow_low: 0, noverflow_high: 0, counts: vec![0u64; nbins], nbins, ntotal: 0}
    }
    
    /// Add a computed spawning probability to the histogram.
    /// # Arguments: 
    /// `self`: ExcitationHist.
    /// - `pspawn`: Spawning probability as defined in population dynamics routines.
    /// # Returns
    /// - `()`: Updates the histogram in place.
    pub fn add(&mut self, pspawn: f64) {
        self.ntotal += 1;

        if !pspawn.is_finite() || pspawn <= 0.0 {self.noverflow_low += 1; return;}

        let logpspawn = pspawn.ln();

        if logpspawn < self.logmin {self.noverflow_low += 1; return;}
        if logpspawn >= self.logmax {self.noverflow_high += 1; return;}

        // Fractional position of logpspawn in histogram range.
        let t = (logpspawn - self.logmin) / (self.logmax - self.logmin);
        // Convert to bin units.
        let b = (t * self.nbins as f64) as usize;
        self.counts[b] += 1;
    }
}

/// Find matrix element S_{ij} from Wick's or naive path.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `basis`: Vector of all SCF states in the basis.
/// - `i`: Index of state i. 
/// - `j`: Index of state j.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `input`: User specified input options.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element S_{ij}.
fn find_s(ao: &AoData, basis: &[SCFState], i: usize, j: usize, tol: f64, input: &Input, wicks: Option<&WicksView>, 
          scratch: &mut WickScratchSpin) -> f64 {
    // Get the sorted pair of indices 
    let (a, b) = if i <= j {(i, j)} else {(j, i)};
    calculate_s_pair(ao, &basis[a], &basis[b], tol, input, wicks, Some(scratch))
}

/// Find matrix elements H_{ij} and S_{ij} from Wick's or naive path. 
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data. 
/// - `basis`: Vector of all SCF states in the basis.
/// - `i`: Index of state i. 
/// - `j`: Index of state j.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `input`: User specified input options.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `(f64, f64)`: Hamiltonian and overlap matrix elements H_{ij} and S_{ij}.
fn find_hs(ao: &AoData, basis: &[SCFState], i: usize, j: usize, tol: f64, input: &Input, wicks: Option<&WicksView>, 
           mocache: &[MOCache], scratch: &mut WickScratchSpin) -> (f64, f64) {
    // Get the sorted pair of indices 
    let (a, b) = if i <= j {(i, j)} else {(j, i)};
    calculate_hs_pair(ao, &basis[a], &basis[b], tol, input, mocache, wicks, Some(scratch))
}

/// Accumulate the population change dn for a determinant i into the per-iteration delta vector.
/// Note that the actual populations stored in mc.walkers are not yet changed here.
/// # Arguments:
/// - `mc`: Contains information about the current Monte Carlo state.
/// - `i`: Index of determinant i to be updated.
/// - `dn`: Population change on determinant i. 
/// # Returns
/// - `()`: Updates the accumulated population changes in place.
fn add_delta(mc: &mut MCState, i: usize, dn: i64) {
    if dn == 0 {return;}
    // If current delta for this determinant is zero this function being called is its first
    // modification for this iteration and so we record this fact in changed.
    if mc.delta[i] == 0 {
        mc.changed.push(i);
    }
    // Add the population change to delta.
    mc.delta[i] += dn;
}

/// Apply accumulated population changes from the mc.delta vector to the actual populations stored
/// in mc.walkers.
/// # Arguments:
/// - `mc`: Contains information about the current Monte Carlo state.
/// # Returns
/// - `Vec<PopulationUpdate>`: List of applied population updates for this iteration.
fn apply_delta(mc: &mut MCState) -> Vec<PopulationUpdate> {
    if mc.changed.is_empty() {return Vec::new();}
    let mut applied = Vec::with_capacity(mc.changed.len());
    // Consider only changed determinants.
    for &i in &mc.changed {
        // Total population change for this determinant for this iteration.
        let dn = mc.delta[i];
        // As we are applying the change we reset the delta.
        mc.delta[i] = 0;
        mc.walkers.add(i, dn);
        applied.push(PopulationUpdate { det: i as u64, dn });
    }
    // Reset list of changed determinants.
    mc.changed.clear();
    applied
}

/// For each entry in initial coefficient vector c0 calculate (c0_i / ||c||) * N_0 (initial population) 
/// and assign this value  as the initial walker population on this determinant. Sign of population is given 
/// by sign of c0_i.
/// # Arguments:
/// `c0`: `Vec<f64>`: Initial determinant coefficient vector.
/// `init_pop`: Number of walkers to start with.
/// `n`: Number of determinants.
/// - `ao`: Contains AO integrals and other system data. 
/// - `basis`: List of the full NOCI-QMC basis.
/// - `iref`: Index of the first reference determinant.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `input`: User specified input options.
/// - `wicks`: Intermediates required for evaluating matrix
/// elements using the extended non-orthogonal Wick's theorem.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `Walkers`: Initial walker population.
pub fn initialse_walkers(c0: &[f64], init_pop: i64, n: usize, ao: &AoData, basis: &[SCFState], iref: usize, tol: f64,
                         input: &Input, wicks: Option<&WicksView>, scratch: &mut WickScratchSpin) -> Walkers {
    let mut w = Walkers::new(n);

    // Ill-conditioning threshold.
    let threshold = 1e-6; 

    // Calculate 1-norm of initial coefficient vector.
    let norm1: f64 = c0.iter().map(|x| x.abs()).sum::<f64>();

    // Assign initial populations based on c0.
    for (i, &ci) in c0.iter().enumerate() {
        let ni = ((ci.abs() / norm1) * (init_pop as f64)).round() as i64;
        // Decide sign when not zero.
        if ni != 0 {
            let sgn = if ci >= 0.0 {1} else {-1};
            w.add(i, sgn * ni);
        }
    }
    
    // Calculate projected energy denominator \Sum_{\Gamma} N_{\Gamma} S_{\Gamma, iref}. If this
    // quantity is very small we have large amounts of overcompleteness and therefore
    // ill-conditioning in the NOCI-QMC overlap matrix.
    let mut den = 0.0; 
    for &gamma in w.occ() {
        let ngamma = w.get(gamma);
        // Calculate matrix element H_{\Gamma, \text{Reference}}, S_{\Gamma,\text{Reference}} 
        let sgr = find_s(ao, basis, gamma, iref, tol, input, wicks, scratch);
        den += (ngamma as f64) * sgr;
    }

    // If the NOCI-QMC overlap matrix is very ill-conditioned starting with all walkers on the RHF
    // reference rather than distributed according to c0 can prevent initial blow-ups in the
    // projected energy.
    if den.abs() < threshold {
        println!("NOCI-QMC overlap very ill-conditioned. Starting from reference index 0 (Denominator: {})", den);
        let mut w0 = Walkers::new(n);
        w0.add(0, init_pop);
        return w0;
    }
    w
}

/// Compute the off-diagonal coupling between determinants depending on which propagator is used.
/// # Arguments:
/// - `hlg`: Matrix element H_{\Lambda\Gamma}
/// - `slg`: Matrix element S_{\Lambda\Gamma}
/// - `es_s`: E_s^S(\tau) shift energy.
/// - `es`: E_s(\tau) shift energy.
/// - `prop`: Chosen propagator.
/// # Returns
/// - `f64`: Off-diagonal coupling used by the propagator.
fn coupling(hlg: f64, slg: f64, es_s: f64, es: f64, prop: &Propagator) -> f64 {
    match prop {
        Propagator::Unshifted => hlg - es_s * slg,
        Propagator::Shifted => hlg - es_s * slg,
        Propagator::DoublyShifted => hlg - es_s * slg,
        Propagator::DifferenceDoublyShiftedU1 => hlg - 0.5 * (es + es_s) * slg,
        Propagator::DifferenceDoublyShiftedU2 => hlg - es_s * slg,
    }
}

/// Initialise per-determinant heat-bath excitation generation quantities such that they can be
/// precomputed per-determinant rather than per walker. Calculates total weight
/// \Sum_{\Lambda \neq \Gamma} |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}|, cumulative sums
/// \sum_{i=1}^n |H_{i\Gamma} - E_s^S(\tau)S_{i\Gamma}|, the Lambda index corresponding to each
/// cumulative sum, and the couplings H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}.
/// # Arguments:
/// - `gamma`: Parent determinant index.
/// - `es_s`: E_s^S(\tau) shift energy.
/// - `es`: E_s(\tau) shift energy.
/// - `ao`: Contains AO integrals and other system data. 
/// - `basis`: List of the full NOCI-QMC basis.
/// - `input`: User specified input options.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `noci_reference_basis`: Only the reference basis determinants.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `HeatBath`: Precomputed heat-bath excitation generation data for determinant `gamma`.
fn init_heat_bath(gamma: usize, es_s: f64, es: f64, ao: &AoData, basis: &[SCFState], input: &Input, tol: f64, 
                  wicks: Option<&WicksView>, scratch: &mut WickScratchSpin, mocache: &[MOCache]) -> HeatBath {
    let ndets = basis.len();
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
        let (hlg, slg) = find_hs(ao, basis, lambda, gamma, tol, input, wicks, mocache, scratch);
        let k = coupling(hlg, slg, es_s, es, &input.prop_ref().propagator);

        sumlg += k.abs();
        cumulatives.push(sumlg);
        lambdas.push(lambda);
        ks.push(k);
    }
    HeatBath {sumlg, cumulatives, lambdas, ks}
}

/// Propose a determinant for spawning using a uniform excitation scheme, and calculate the probability 
/// P_{\text{gen}} that it was chosen.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data. 
/// - `basis`: List of the full NOCI-QMC basis.
/// - `gamma`: Index of determinant being spawned from.
/// - `es_s`: E_s^S(\tau) shift energy.
/// - `es`: E_s(\tau) shift energy.
/// - `input`: User specified input options.
/// - `rng`: Random number generator.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `(f64, f64, usize)`: Generation probability, coupling, and selected determinant index.
fn pgen_uniform(ao: &AoData, basis: &[SCFState], gamma: usize, es_s: f64, es: f64, input: &Input, rng: &mut SmallRng, tol: f64,
                wicks: Option<&WicksView>, scratch: &mut WickScratchSpin, mocache: &[MOCache]) -> (f64, f64, usize) {
    let ndets = basis.len();
    // Sample Lambda uniformly from all dets except Gamma. If Lambda index is the same or
    // more than the Gamma index we map it back into the full index set.
    let mut lambda = rng.gen_range(0..(ndets - 1));
    if lambda >= gamma {lambda += 1;}

    let (hlg, slg) = find_hs(ao, basis, lambda, gamma, tol, input, wicks, mocache, scratch);
    let k = coupling(hlg, slg, es_s, es, &input.prop_ref().propagator);
    // Uniform generation probability
    let pgen = 1.0 / ((ndets - 1) as f64);
    (pgen, k, lambda) 
}

/// Propose a determinant for spawning using a heat-bath excitation scheme, and calculate the probability 
/// P_{\text{gen}} that it was chosen. Warning: This implements the exact heat-bath excitation scheme rather than any 
/// approximate version, this means it is very very slow and should really only be used for benchmarking other 
/// excitation schemes.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data. 
/// - `basis`: List of the full NOCI-QMC basis.
/// - `gamma`: Index of determinant being spawned from.
/// - `es_s`: E_s^S(\tau) shift energy.
/// - `es`: E_s(\tau) shift energy.
/// - `input`: User specified input options.
/// - `rng`: Random number generator.
/// - `hb`: Precomputed heat-bath excitation generation data.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `(f64, f64, usize)`: Generation probability, coupling, and selected determinant index.
fn pgen_heat_bath (ao: &AoData, basis: &[SCFState], gamma: usize, es_s: f64, es: f64, input: &Input, rng: &mut SmallRng, hb: &HeatBath, tol: f64,
                   wicks: Option<&WicksView>, scratch: &mut WickScratchSpin, mocache: &[MOCache]) -> (f64, f64, usize) {

    let ndets = basis.len();

    // If \Sum_{\Lambda \neq \Gamma} |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma} (sumlg) 
    // is zero (unsure how likely this is) then fallback to uniform distribution.
    if hb.sumlg == 0.0 {
        let mut lambda = rng.gen_range(0..(ndets - 1));
        if lambda >= gamma {lambda += 1;}
        let (hlg, slg) = find_hs(ao, basis, lambda, gamma, tol, input, wicks, mocache, scratch);
        let k = coupling(hlg, slg, es_s, es, &input.prop_ref().propagator);
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

/// Perform off-diagonal spawning (amplitude transfer) step by calculating:
///     $P_{\text{Spawn}}(\Lambda|\Gamma) = \frac{\Delta\tau|H_{\Lambda\Gamma} - 
///     E_sS_{\Lambda\Gamma}|}{P_{\text{gen}}(\Lambda|\Gamma)}$
/// where if P_{\text{spawn}} > random float in [0, 1] we spawn a child walker onto determinant \Lambda 
/// with the same sign of its parent if H_{\Lambda\Gamma} > 0 and -sign if H_{\Lambda\Gamma} < 0. Furthermore, 
/// if P_{\text{spawn}} > 1 then we spawn floor(P_{\text{spawn}}) extra children and a final child with 
/// probability P_{\text{spawn}} - floor(P_{\text{spawn}}).
/// # Arguments
/// - `ao`: Contains AO integrals and other system data. 
/// - `basis`: List of the full NOCI-QMC basis.
/// - `gamma`: Index of determinant \Gamma. 
/// - `ngamma`: Walker population on determinant \Gamma.
/// - `es_s`: Overlap-transformed shift energy.
/// - `es`: Non-overlap transformed shift energy.
/// - `input`: User specified input options.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `noci_reference_basis`: Only the reference basis determinants.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `irank`: Number of current rank.
/// - `nranks`: Total number of ranks.
/// - `rng`: Random number generator.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `outlocal`: I64)>, locally owned population updates.
/// - `outremote`: Remotely owned population updates.
/// - `outsamples`: Collected spawning probability samples.
/// # Returns
/// - `()`: Appends spawning updates to the provided output buffers.
fn spawning(ao: &AoData, basis: &[SCFState], gamma: usize, ngamma: i64, es_s: f64, es: f64, input: &Input, tol: f64, wicks: Option<&WicksView>, 
            irank: usize, nranks: usize, rng: &mut SmallRng, scratch: &mut WickScratchSpin, mocache: &[MOCache], outlocal: &mut Vec<(usize, i64)>, outremote: &mut Vec<PopulationUpdate>, 
            outsamples: &mut Vec<f64>) {

    let parent_sign: i64 = if ngamma > 0 {1} else {-1};
    let nwalkers = ngamma.unsigned_abs();

    // Precompute per determinant heat-bath excitation generation quantities.
    let mut hb: Option<HeatBath> = None;
    if let ExcitationGen::HeatBath = input.qmc.as_ref().unwrap().excitation_gen {hb = Some(init_heat_bath(gamma, es_s, es, ao, basis, input, tol, wicks, scratch, mocache));}

    // Iterate over all walkers on state Gamma.
    for _ in 0..nwalkers {
        // Calculate generation probability, k = |H_{\Gamma\Lambda} - E_s^S(\tau)
        // S_{\Gamma\Lambda}| and the selected determinant for spawning via the selected excitation
        // generation scheme.
        let (pgen, k, lambda) = match input.qmc.as_ref().unwrap().excitation_gen {
            ExcitationGen::Uniform => pgen_uniform(ao, basis, gamma, es_s, es, input, rng, tol, wicks, scratch, mocache),
            ExcitationGen::HeatBath => pgen_heat_bath(ao, basis, gamma, es_s, es, input, rng, hb.as_ref().unwrap(), tol, wicks, scratch, mocache),
            ExcitationGen::ApproximateHeatBath => unimplemented!(),
        };

        // Calculate spawning probability.
        let pspawn = input.prop_ref().dt * k.abs() / pgen;

        // Store excitation sample if requested.
        if input.write.write_excitation_hist {outsamples.push(pspawn);}

        // Evaluate spawning outcomes
        let m = pspawn.floor() as i64;
        let frac = pspawn - (m as f64);
        let extra = if rng.gen_range(0.0..1.0) < frac {1} else {0};
        let nchildren = m + extra;
        if nchildren == 0 {
            continue;
        }

        let sign: i64 = if k > 0.0 {1} else {-1};
        let child_sign: i64 = -sign * parent_sign;
        let dn = child_sign * nchildren;
        
        // Apply spawwning population updates locally if the determinant to be updated is owned by
        // current thread, otherwise add the population update message to the send buffer.
        if nranks == 1 {
            outlocal.push((lambda, dn));
        } else {
            let destination = owner(lambda, nranks);
            if destination == irank {
                outlocal.push((lambda, dn));
            } else {
                outremote.push(PopulationUpdate {det: lambda as u64, dn});
            }
        }
    }
}

/// Perform diagonal death and cloning step by calculating:
///     $P_{\text{Death}}(\Lambda) = \Delta\tau(H_{\Lambda\Lambda}- 2E_s)$,
/// where which if P_{\text{death}} > 0 a walker dies with probability P_{\text{death}}, 
/// if P_{\text{death}} < 0 a walker is cloned with probability |P_{\text{death}}|. 
/// If P_{\text{death}} = 0, nothing will happen.  
/// # Arguments
/// - `ao`: Contains AO integrals and other system data. 
/// - `basis`: List of the full NOCI-QMC basis.
/// - `gamma`: Index of determinant \Gamma. 
/// - `ngamma`: Walker population on determinant \Gamma.
/// - `es`: Non-overlap transformed shift energy.
/// - `es_s`: Overlap-transformed shift energy.
/// - `input`: User specified input options.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `rng`: Random number generator.
/// - `scratch`: Scratch space for Wick's quantities.
/// - `out`: I64)>, output population updates.
/// # Returns
/// - `()`: Appends death/cloning updates to `out`.
fn death_cloning(ao: &AoData, basis: &[SCFState], gamma: usize, ngamma: i64, es: f64,  es_s: f64, input: &Input, tol: f64, 
                 wicks: Option<&WicksView>, mocache: &[MOCache], rng: &mut SmallRng, scratch: &mut WickScratchSpin, out: &mut Vec<(usize, i64)>) {
    
    // Calculate matrix elements H_{\Gamma\Gamma}, S_{\Gamma\Gamma}.
    let (hgg, sgg) = find_hs(ao, basis, gamma, gamma, tol, input, wicks, mocache, scratch);

    // Death probability.
    let pdeath = match input.prop_ref().propagator {
        Propagator::Unshifted => input.prop_ref().dt * (hgg - sgg * es_s),
        Propagator::Shifted => input.prop_ref().dt * (hgg - sgg * es_s - es_s),
        Propagator::DoublyShifted => input.prop_ref().dt * (hgg - sgg * es_s - es),
        Propagator::DifferenceDoublyShiftedU1 => input.prop_ref().dt * (hgg - sgg * 0.5 * (es_s + es) - (es - es_s)),
        Propagator::DifferenceDoublyShiftedU2 => input.prop_ref().dt * (hgg - sgg * es_s - (es - es_s)),
    };
    if pdeath == 0.0 {
        return;
    }
    let p = pdeath.abs();

    // Sign of parent walkers on state Gamma determines which way round death and cloning occur.
    let parent_sign = if ngamma > 0 {1} else {-1};
    
    let n = ngamma.abs();
    let m = p.floor() as i64;
    let frac = p - (m as f64);
   
    // Rather than iterate over all walkers we can sample binominal distribution.
    let extra = if frac > 0.0 {Binomial::new(n as u64, frac).unwrap().sample(rng) as i64} else {0};
    let nevents = n * m + extra;
    if nevents == 0 {return;}
    
    // Accumulate population change in delta.
    let dn = if pdeath > 0.0 {-parent_sign} else {parent_sign};
    out.push((gamma, dn * nevents));
}

/// Initialise the running projected-energy state
/// \frac{\sum_{\Gamma} N_{\Gamma} H_{\Gamma,\mathrm{ref}}}{\sum_{\Gamma} N_{\Gamma} S_{\Gamma,\mathrm{ref}}}
/// by performing the full initial sweep over the occupied determinant space.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `basis`: List of the full NOCI-QMC basis.
/// - `walkers`: Object containing information about current walkers.
/// - `iref`: Index of the determinant onto which the energy is projected.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `input`: User specified input options.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// # Returns
/// - `ProjectedEnergyUpdate`: Initial projected-energy numerator and denominator.
fn init_projected_energy(ao: &AoData, basis: &[SCFState], walkers: &Walkers, iref: usize, world: &impl Communicator, tol: f64, input: &Input,
                               wicks: Option<&WicksView>, mocache: &[MOCache]) -> ProjectedEnergyUpdate {
    let (num_local, den_local) = walkers.occ().par_iter().fold(|| (0.0_f64, 0.0_f64, WickScratchSpin::new()), |(mut num, mut den, mut scratch), &gamma| {
        let ngamma = walkers.get(gamma) as f64;
        let (hgr, sgr) = find_hs(ao, basis, gamma, iref, tol, input, wicks, mocache, &mut scratch);
        num += ngamma * hgr;
        den += ngamma * sgr;
        (num, den, scratch)
    }).map(|(num, den, _)| (num, den)).reduce(|| (0.0, 0.0), |(a_num, a_den), (b_num, b_den)| (a_num + b_num, a_den + b_den));

    let mut num = 0.0;
    let mut den = 0.0;
    world.all_reduce_into(&num_local, &mut num, SystemOperation::sum());
    world.all_reduce_into(&den_local, &mut den, SystemOperation::sum());

    ProjectedEnergyUpdate {iref, num, den}
}

/// Incrementally update the running projected-energy state using the net walker
/// population changes applied in the current iteration.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `basis`: List of the full NOCI-QMC basis.
/// - `d`: Net population changes applied in the current iteration.
/// - `pe`: Running projected-energy state to update in place.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `input`: User specified input options.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// # Returns
/// - `()`: Updates `pe` in place.
fn update_projected_energy(ao: &AoData, basis: &[SCFState], d: &[PopulationUpdate], pe: &mut ProjectedEnergyUpdate, world: &impl Communicator, tol: f64,
                                 input: &Input, wicks: Option<&WicksView>, mocache: &[MOCache]) {
    let iref = pe.iref;

    let (dnum_local, dden_local) = d.par_iter().fold(|| (0.0_f64, 0.0_f64, WickScratchSpin::new()), |(mut dnum, mut dden, mut scratch), up| {
        let gamma = up.det as usize;
        let dn = up.dn as f64;
        let (hgr, sgr) = find_hs(ao, basis, gamma, iref, tol, input, wicks, mocache, &mut scratch);
        dnum += dn * hgr;
        dden += dn * sgr;
        (dnum, dden, scratch)
    }).map(|(dnum, dden, _)| (dnum, dden)).reduce(|| (0.0, 0.0), |(a_num, a_den), (b_num, b_den)| (a_num + b_num, a_den + b_den));

    let mut dnum = 0.0;
    let mut dden = 0.0;
    world.all_reduce_into(&dnum_local, &mut dnum, SystemOperation::sum());
    world.all_reduce_into(&dden_local, &mut dden, SystemOperation::sum());

    pe.num += dnum;
    pe.den += dden;
}

/// Initialise the vector p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} N_{\Omega}. 
/// # Arguments:
/// - `start`: First determinant index owned by this rank.
/// - `end`: Final determinant index owned by this rank.
/// - `ao`: Contains AO integrals and other system data. 
/// - `basis`: List of the full NOCI-QMC basis.
/// - `walkers`: Object containing information about current walkers.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `input`: User specified input options.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `Vec<f64>`: Local portion of the overlap-transformed population vector p_{\Gamma}.
fn init_p(start: usize, end: usize, ao: &AoData, basis: &[SCFState], walkers: &Walkers, world: &impl Communicator, tol: f64, 
          input: &Input, wicks: Option<&WicksView>, scratch: &mut WickScratchSpin) -> Vec<f64> {

    let local: Vec<PopulationUpdate> = walkers.occ().iter().map(|&i| PopulationUpdate {det: i as u64, dn: walkers.get(i)}).collect();
    let global = gather_all_walkers(world, &local);

    let mut p = vec![0.0; end - start];
    for gamma in start..end {
        //p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} N_{\Omega}.
        let mut pgamma = 0.0;
        for entry in &global {
            // Here we use dn to mean total population.
            let nomega = entry.dn as f64;
            let omega = entry.det as usize;
            let sgo = find_s(ao, basis, gamma, omega, tol, input, wicks, scratch);
            pgamma += nomega * sgo;
        }
        p[gamma - start] = pgamma
    }
    p
}

/// Update the vector p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} N_{\Omega}.
/// # Arguments:
/// - `start`: First determinant index owned by this rank.
/// - `ao`: Contains AO integrals and other system data. 
/// - `basis`: List of the full NOCI-QMC basis.
/// - `world`: MPI communicator object (MPI_COMM_WORLD). 
/// - `plocal`: Vector p_{\Gamma} on this rank.
/// - `dlocal`: Local determinant population updates.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `input`: User specified input options.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// # Returns
/// - `()`: Updates `plocal` in place.
fn update_p(start: usize, ao: &AoData, basis: &[SCFState], world: &impl Communicator, plocal: &mut [f64], dlocal: &[PopulationUpdate], tol: f64,
            input: &Input, wicks: Option<&WicksView>) {

    let dglobal = gather_all_walkers(world, dlocal);

    // \Delta p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} \Delta N_{\Omega}.
    plocal.par_iter_mut().enumerate().for_each_init(WickScratchSpin::new, |scratch, (k, pgamma)| {
        let gamma = start + k;
        let mut dp = 0.0;
        for update in &dglobal {
            let omega = update.det as usize;
            dp += find_s(ao, basis, gamma, omega, tol, input, wicks, scratch) * update.dn as f64;
        }
        *pgamma += dp;
    });
}

/// Propagate according to the stochastic update equations for max_steps iterations.
/// # Arguments: 
/// - `c0`: Initial determinant coefficient vector to be translated into walker
///   populations.
/// - `ao`: Contains AO integrals and other system data. 
/// - `basis`: List of the full NOCI-QMC basis.
/// - `es`: Initial shift energy.
/// - `input`: User specified input options.
/// - `ref_indices`: Indices of the reference determinants embedded in the full space.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Intermediates required for evaluating matrix
///   elements using the extended non-orthogonal Wick's theorem.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// # Returns
/// - `(f64, Option<ExcitationHist>, StochStepTimings)`: Projected energy estimate, optional
///   excitation histogram, and stochastic propagation timings.
pub fn qmc_step(c0: &[f64], ao: &AoData, basis: &[SCFState], es: &mut f64, input: &mut Input, ref_indices: &[usize], world: &impl Communicator, tol: f64, 
                wicks: Option<&WicksView>, mocache: &[MOCache]) -> (f64, Option<ExcitationHist>, StochStepTimings) {
    
    // Timings.
    let mut d_spawn_death_collect = 0.0f64;
    let mut d_acc_pack = 0.0f64;
    let mut d_mpi_exchange_updates = 0.0f64;
    let mut d_unpack_acc_recieved = 0.0f64;
    let mut d_apply_delta = 0.0f64;
    let mut d_update_p = 0.0f64;
    let mut d_calc_populations = 0.0f64;
    let mut d_eproj = 0.0f64;

    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let ndets = basis.len();
    let start = (ndets * irank) / nranks;
    let end = (ndets * (irank + 1)) / nranks;

    let iref = 0;
    // Serial Wick's scratch.
    let mut scratch = WickScratchSpin::new();
    println!("Size of Wick's Scratch (MiB): {}", std::mem::size_of::<WickScratchSpin>() as f64 / (1024.0 * 1024.0));
    
    // Accumulated updates per Rayon thread. Local contains anything that happens on a
    // determinant under the control of the current thread, and remote anything that applies to
    // another thread's determinant.
    type LocalUpdates = Vec<(usize, i64)>;
    type RemoteUpdates = Vec<PopulationUpdate>;
    type Samples = Vec<f64>;

    type ThreadState = (LocalUpdates, RemoteUpdates, Samples, SmallRng, WickScratchSpin);
    println!("Size of per thread state (MiB): {}", std::mem::size_of::<ThreadState>() as f64 / (1024.0 * 1024.0));

    // Construct reference index mask.
    let mut isref = vec![false; basis.len()];
    for &i in ref_indices {isref[i] = true;}
    
    // If the number of references is 1 then the unshifted propagator is correct.
    if ref_indices.len() == 1 {
        if irank == 0 {println!("Number of references is 1. Setting propagator to unshifted.....");}
        input.prop_mut().propagator = Propagator::Unshifted;
    }

    // Unwrap QMC propagation specific options
    let qmc = input.qmc.as_ref().unwrap();
    let ndets = basis.len();

    // Initialise walker populations based on total initial population and c0
    let t0 = Instant::now(); 
    if irank == 0 {println!("Initialising walkers.....");}
    let w = initialse_walkers(c0, qmc.initial_population, ndets, ao, basis, iref, tol, input, wicks, &mut scratch);
    let w = local_walkers(w, irank, nranks);
    let d_initialise_walkers = t0.elapsed();

    // Flags activated once total walker population exceeds target population 
    let mut reached_sc = false;
    let mut reached_c = false;

    // Initialise RNG. If user provides a seed for deterministic runs we turn it into distinct per
    // rank seeds. Otherwise the seeds are random. Uses `Golden Ratio Hashing` via 0x9E3779B9. 
    let base: u64 = qmc.seed.unwrap_or_else(rand::random::<u64>);
    let seed = base.wrapping_add((irank as u64).wrapping_mul(0x9E3779B9));
    let rng = SmallRng::seed_from_u64(seed);

    // Initialise excitation generation histogram if requested.
    let excitation_hist = if input.write.write_excitation_hist {Some(ExcitationHist::new(-60.0, 1e-12, 100))} else {None};

    // Initialise Monte Carlo state. All population updates for a given iteration are accumulated within delta.
    let pg = init_p(start, end, ao, basis, &w, world, tol, input, wicks, &mut scratch);
    let mut mc = MCState {walkers: w, delta: vec![0; ndets], changed: Vec::new(), rng, pg, excitation_hist};

    // Project onto determinant with index zero (this is usually the first RHF reference).
    let mut pe = init_projected_energy(ao, basis, &mc.walkers, iref, world, tol, input, wicks, mocache);
    let eproj = pe.num / pe.den;

    // Initialise overlap-transformed shift E_s^S. This is distinct from es (E_s) in that E_s is
    // the shift updated using N_w, whilst E_s^S is updated using \tilde{N}_w.
    let mut es_s = *es;

    // Initialise overlap-transformed populations.
    let mut nwprevsc = 0.0;
    let mut nrefprevsc = 0.0; 
    let nwprevsc_local: f64 = mc.pg.iter().map(|x| x.abs()).sum();
    world.all_reduce_into(&nwprevsc_local, &mut nwprevsc, SystemOperation::sum());
    let nrefprevsc_local: f64 = mc.pg.iter().enumerate().filter(|(k, _)| isref[start + *k]).map(|(_, x)| x.abs()).sum();
    world.all_reduce_into(&nrefprevsc_local, &mut nrefprevsc, SystemOperation::sum());

    // Initialse non-overlap transformed populations.
    let mut nwprevc = qmc.initial_population;
    let mut nrefprevc: i64 = 0;
    let nrefprevc_local: i64 = mc.walkers.occ().iter().filter(|&&det| isref[det]).map(|&det| mc.walkers.get(det).abs() as i64).sum();
    world.all_reduce_into(&nrefprevc_local, &mut nrefprevc, SystemOperation::sum());

    // Print table header.
    let e0 = basis[0].e;
    let mut es_corr = 0.0;
    let mut es_s_corr = 0.0;
    if irank == 0 {
        println!("{}", "=".repeat(100));
        println!("{:<6} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}", 
                 "iter", "E", "Ecorr", "Shift (Es)", "Shift (EsS)", "Nw (||C||)", "Nref (||C||)", "Nw (||SC||)", "Nref(||SC||)");
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",  
                 0, eproj, eproj - e0, es_corr, es_s_corr, nwprevc as f64, nrefprevc as f64, nwprevsc, nrefprevsc);
    }
    
    // Per MPI rank send buffers.
    let mut send: Vec<Vec<PopulationUpdate>> = (0..nranks).map(|_| Vec::new()).collect();

    // Find maximum possible allocations for Wick's scratch allocation size.
    let maxex_a = basis.iter().map(|st| st.excitation.alpha.holes.len()).max().unwrap_or(0);
    let maxex_b = basis.iter().map(|st| st.excitation.beta.holes.len()).max().unwrap_or(0);
    let maxsame = 2 * maxex_a.max(maxex_b);
    let maxla = 2 * maxex_a;
    let maxlb = 2 * maxex_b;
    
    // Initialise early exit cache for printing iteration line.
    let mut eprojcur = pe.num / pe.den;
    let mut nwccur = qmc.initial_population;
    let mut nrefccur = nrefprevc;
    let mut nwsccur = nwprevsc;
    let mut nrefsccur = nrefprevsc;

    for it in 0..input.prop_ref().max_steps {
        // Function to initialise individual thread states.
        let initialise = || -> ThreadState {
            let tid = rayon::current_thread_index().unwrap_or(0) as u64;
            (Vec::new(), Vec::new(), Vec::new(), SmallRng::seed_from_u64(seed ^ tid ^ ((it as u64).wrapping_mul(0x9E3779B97F4A7C15))), 
             WickScratchSpin::with_sizes(maxsame, maxla, maxlb))
        };

        // Function to call spawning and death/cloning routines and accumulate population updates.
        let propagate =  |(mut loc, mut rem, mut samp, mut rng, mut scratch): ThreadState, &gamma: &usize| -> ThreadState {
            let ngamma = mc.walkers.get(gamma);
            if ngamma == 0 {return (loc, rem, samp, rng, scratch);}

            // Death and cloning is entirely local.
            death_cloning(ao, basis, gamma, ngamma, *es, es_s, input, tol, wicks, mocache, &mut rng, &mut scratch, &mut loc);
            // Spawning need not be local.
            spawning(ao, basis, gamma, ngamma, es_s, *es, input, tol, wicks, irank, nranks, &mut rng, &mut scratch, mocache, &mut loc, &mut rem, &mut samp);

            (loc, rem, samp, rng, scratch)
        };

        // Function to merge thread updates a and b.
        let merge = |mut a: ThreadState, mut b: ThreadState| -> ThreadState {
            a.0.append(&mut b.0);
            a.1.append(&mut b.1);
            a.2.append(&mut b.2);
            a
        };

        // Rayon parallised loop over occupied determinants.
        let occ = mc.walkers.occ();
        // Iterate over occ in parallel and give each thread its own accumulator in initialise and
        // propagate, and reduce them into a single total ThreadState. The empty type in .reduce is
        // what is reduced into by the threads.
        let t0 = Instant::now();
        let (local, remote, samples, _, _): ThreadState = occ.par_iter().fold(initialise, propagate)
                                            .reduce(|| (Vec::new(), Vec::new(), Vec::new(), SmallRng::seed_from_u64(0), WickScratchSpin::new()), merge);
        d_spawn_death_collect += t0.elapsed().as_secs_f64();

        let t0 = Instant::now();

        // Clear MPI send buffers from previous iteration if using MPI.
        if nranks > 1 {for buf in &mut send {buf.clear();}}

        // Apply Rayon thread local updates across all determinants and write excitation samples if requested.
        for (det, dn) in local {
            add_delta(&mut mc, det, dn);
        }
        if input.write.write_excitation_hist && let Some(hist) = mc.excitation_hist.as_mut() {
            for p in samples {
                hist.add(p);
            }
        }
        
        // Fill the MPI send buffer with updates needing to be communicated across ranks. 
        if nranks > 1 {
            for up in remote {
                let dest = owner(up.det as usize, nranks);
                send[dest].push(up);
            }
        }
        d_acc_pack += t0.elapsed().as_secs_f64();

        // Exchange population updates between ranks and add any recieved updates to local delta. 
        if nranks > 1 {
            let t0 = Instant::now();
            let recieved = communicate_spawn_updates(world, &send);
            d_mpi_exchange_updates += t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            for update in recieved {
                add_delta(&mut mc, update.det as usize, update.dn);
            }
            d_unpack_acc_recieved += t0.elapsed().as_secs_f64();
        }       

        // Early exit if there is no updates.
        let changed: i32 = (!mc.changed.is_empty()) as i32;
        let mut changedglobal: i32 = 0;
        world.all_reduce_into(&changed, &mut changedglobal, SystemOperation::max());
        if changedglobal == 0 {
            if irank == 0 {
                println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
                         it + 1, eprojcur, eprojcur - basis[0].e, es_corr, es_s_corr, nwccur as f64, nrefccur as f64, nwsccur, nrefsccur);
            }
            continue;
        }

        // Apply any local and recieved deltas. Annhilation is fully local process here.  
        // The change in population is also stored to update overlap-transformed walker population.
        let t0 = Instant::now();
        let d = apply_delta(&mut mc);
        d_apply_delta += t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        update_p(start, ao, basis, world, &mut mc.pg, &d, tol, input, wicks);
        d_update_p += t0.elapsed().as_secs_f64();
        
        let t0 = Instant::now();

        // Calculate non overlap-transformed walker population.
        let mut nwc = 0i64;
        let nwclocal = mc.walkers.norm() as i64;
        world.all_reduce_into(&nwclocal, &mut nwc, SystemOperation::sum());
        let mut nrefc: i64 = 0;
        let nrefc_local: i64 = mc.walkers.occ().iter().filter(|&&det| isref[det]).map(|&det| mc.walkers.get(det).abs() as i64).sum();
        world.all_reduce_into(&nrefc_local, &mut nrefc, SystemOperation::sum());

        // Calculate overlap-transformed walker population.
        let mut nwsc = 0.0;
        // Calculate overlap-transformed walker population as: 
        //\tilde{N}_w(\tau) = ||S_{\Gamma\Omega}C^\Omega(\tau)|| = \sum_{\Gamma} |p_{\Gamma}| N_{\Gamma}$
        let nwsc_local: f64 = mc.pg.iter().map(|x| x.abs()).sum();
        world.all_reduce_into(&nwsc_local, &mut nwsc, SystemOperation::sum());
        // Overlap-transformed reference walker population.
        let mut nrefsc = 0.0; 
        // Calculate overlap transformed reference only walker population as:
        //\tilde{N}_{w, refs}(\tau) = ||S_{\Gamma\Omega}C^\Omega(\tau)|| = \sum_{\Gamma, \Gamma \in refs} |p_{\Gamma}| N_{\Gamma}$
        let nrefsc_local: f64 = mc.pg.iter().enumerate().filter(|(k, _)| isref[start + *k]).map(|(_, x)| x.abs()).sum();
        world.all_reduce_into(&nrefsc_local, &mut nrefsc, SystemOperation::sum());
        d_calc_populations += t0.elapsed().as_secs_f64();

        // Cache populations for early exit printing.
        nwccur = nwc;
        nrefccur = nrefc;
        nwsccur = nwsc;
        nrefsccur = nrefsc;

        // Activate overlap-transformed and non-overlap transformed shifts.
        if !reached_c && (nwc > qmc.target_population) {
            reached_c = true;
            nwprevc = nwc;
        }
        if !reached_sc && (nwsc > qmc.target_population as f64) {
            reached_sc = true;
            nwprevsc = nwsc;
        }
        // Update shift once total populations have exceeded target population.
        if reached_c && (it + 1) % qmc.shift_update_freq == 0 {
            *es -= (qmc.shift_damping / (input.prop_ref().dt * (qmc.shift_update_freq as f64))) * (nwc as f64 / nwprevc as f64).ln();
            nwprevc = nwc;
        }
        if reached_sc && (it + 1) % qmc.shift_update_freq == 0 {
            es_s -= (qmc.shift_damping / (input.prop_ref().dt * (qmc.shift_update_freq as f64))) * (nwsc / nwprevsc).ln();
            nwprevsc = nwsc;
        }
        
        // Update energy.
        let t0 = Instant::now();
        update_projected_energy(ao, basis, &d, &mut pe, world, tol, input, wicks, mocache);
        let eproj = pe.num / pe.den;
        eprojcur = eproj;
        d_eproj += t0.elapsed().as_secs_f64();

        es_corr = if reached_c {*es - e0} else {0.0};
        es_s_corr = if reached_sc {es_s - e0} else {0.0};
   
        // Print table rows.
        if irank == 0 {println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", 
                       it + 1, eproj, eproj - basis[0].e, es_corr, es_s_corr, nwc as f64, nrefc as f64, nwsc, nrefsc);}
    }

    let timings = StochStepTimings {initialse_walkers: d_initialise_walkers, spawn_death_collect: Duration::from_secs_f64(d_spawn_death_collect), 
                                    acc_pack: Duration::from_secs_f64(d_acc_pack),  mpi_exchange_updates: Duration::from_secs_f64(d_mpi_exchange_updates), 
                                    unpack_acc_recieved: Duration::from_secs_f64(d_unpack_acc_recieved), apply_delta: Duration::from_secs_f64(d_apply_delta), 
                                    update_p: Duration::from_secs_f64(d_update_p), calc_populations: Duration::from_secs_f64(d_calc_populations), 
                                    eproj: Duration::from_secs_f64(d_eproj)};

    (eprojcur, mc.excitation_hist, timings)
}
