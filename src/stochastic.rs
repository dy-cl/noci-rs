// stochastic.rs
use std::time::{Instant, Duration};

use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use mpi::topology::Communicator;
use mpi::collective::SystemOperation;
use mpi::traits::*;
use rayon::prelude::*;
use rand_distr::{Binomial, Distribution};

use crate::AoData;
use crate::SCFState;
use crate::input::{Input, Propagator, ExcitationGen};
use crate::nonorthogonalwicks::WicksView;

use crate::noci::{calculate_s_pair_naive, calculate_hs_pair_naive, calculate_s_pair_wicks, calculate_hs_pair_wicks};
use crate::mpiutils::{owner, local_walkers, communicate_spawn_updates, gather_all_walkers};

// Storage for stochastic propagation timings.
#[derive(Default)]
pub struct StochStepTimings {
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
    ///    `n`: usize, number of determinants. 
    pub fn new(n: usize) -> Self {
        Self {
            pop: vec![0; n],
            occ: Vec::new(),
            pos: vec![usize::MAX; n],
        }
    }
    
    /// Return the current population of determinant i.
    /// # Arguments: 
    ///     `self`: Walkers, object containing information about current walkers.
    ///     `i`: usize, determinant index of choice.
    pub fn get(&self, i: usize) -> i64 {
        self.pop[i]
    }
    
    /// Return a list of occupied determinant indices.
    /// # Arguments:
    ///     `self`: Walkers, object containing information about current walkers.
    pub fn occ(&self) -> &[usize] {
        &self.occ
    }

    /// Add dn (change in population) to determinant i, modifying pop, occ and pos as required
    /// # Arguments:
    ///     `self`: Walkers, object containing information about current walkers.
    ///     `i`: usize, determinant index of choice. 
    ///     `dn`: i64, change in population of determinant i.
    pub fn add(&mut self, i: usize, dn: i64) {
        // No changes to be made.
        if dn == 0 {return;}
        
        // Update population vector.
        let old = self.pop[i];
        let new = old + dn;
        self.pop[i] = new;
        
        // If old population of determinant i was 0 and we are introducing population we must
        // add determinant i to the occupied list and store its position in pos.
        if old == 0 && new != 0 {
            // Position of determinant i in occupied list is the current end.
            self.pos[i] = self.occ.len();
            self.occ.push(i);
        // If old population of determinant i was not 0 and we have removed population we must
        // remove i from the occupied list and remove its position from pos.
        } else if old != 0 && new == 0 {
            // Find where i is in occ. Should not return usize::MAX.
            let p = self.pos[i];
            // Pop the last occupied determinant index from occ.
            let last = self.occ.pop().unwrap();

            // If the popped element is not i, then i was somewhere in the middle of occ and we
            // must move the popped element (last) to position p where i used to be. The position of  
            // last is then updated in the position vector. If the popped element is i then we do
            // nothing as we have directly removed it by popping.
            if last != i {
                self.occ[p] = last;
                self.pos[last] = p;
            }
            // Position i is no longer occupied.
            self.pos[i] = usize::MAX;
        }
    }

    /// Compute the total walker population 1-norm.
    /// # Arguments:
    ///     `self`: Walkers, object containing information about current walkers.
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
    ///     `logmin`: f64, minimum histogram value on a logarithmic scale. 
    ///     `logmax`: f64, maximum histogram value on a logarithmic scale.
    ///     `nbins`: usize, number of histogram bins.
    pub fn new(logmin: f64, logmax: f64, nbins: usize) -> Self {
        Self {logmin, logmax, noverflow_low: 0, noverflow_high: 0, counts: vec![0u64; nbins], nbins, ntotal: 0}
    }
    
    /// Add a computed spawning probability to the histogram.
    /// Arguments: 
    ///     self: ExcitationHist.
    ///     pspawn: f64, spawning probability as defined in population dynamics routines.
    pub fn addn(&mut self, pspawn: f64, n: u64) {
        if n == 0 {return;}
        self.ntotal += n;

        if !pspawn.is_finite() || pspawn <= 0.0 {self.noverflow_low += n; return;}

        let logpspawn = pspawn.ln();

        if logpspawn < self.logmin {self.noverflow_low += n; return;}
        if logpspawn >= self.logmax {self.noverflow_high += n; return;}

        // Fractional position of logpspawn in histogram range.
        let t = (logpspawn - self.logmin) / (self.logmax - self.logmin);
        // Convert to bin units.
        let b = (t * self.nbins as f64) as usize;
        self.counts[b] += n;
    }
}

struct GammaUpdates {
    local: Vec<(usize, i64)>,
    remote: Vec<PopulationUpdate>,
    samples: Vec<(f64, u64)>,
}

/// Find matrix element S_{ij} from Wick's or naive path.
/// # Arguments:
///     `ao`: AoData, contains AO integrals and other system data.
///     `basis`: Vec<SCFState>, vector of all SCF states in the basis.
///     `i`: usize, index of state i. 
///     `j`: usize, index of state k.
///     `input`: Input, user specified input options.
///     `noci_reference_basis`: [SCFState], only the reference basis determinants.
///     `wicks`: [Vec<WicksReferencePair>], intermediates required for evaluating matrix
///              elements using the extended non-orthogonal Wick's theorem.
fn find_s(ao: &AoData, basis: &[SCFState], i: usize, j: usize, tol: f64, input: &Input, noci_reference_basis: &[SCFState], wicks: Option<&WicksView>) -> f64 {
    // Get the sorted pair of indices 
    let (a, b) = if i <= j {(i, j)} else {(j, i)};

    if input.wicks.enabled {
        let w = wicks.unwrap();
        calculate_s_pair_wicks(basis, noci_reference_basis, a, b, tol, w)
    } else {
        calculate_s_pair_naive(ao, basis, a, b, tol)
    }
}

/// Find matrix elements H_{ij} and S_{ij} from Wick's or naive path. 
/// # Arguments:
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, vector of all SCF states in the basis.
///     `i`: usize, index of state i. 
///     `j`: usize, index of state k.
///     `input`: Input, user specified input options.
///     `noci_reference_basis`: [SCFState], only the reference basis determinants.
///     `wicks`: [Vec<WicksReferencePair>], intermediates required for evaluating matrix
///              elements using the extended non-orthogonal Wick's theorem.
fn find_hs(ao: &AoData, basis: &[SCFState], i: usize, j: usize, tol: f64, input: &Input, noci_reference_basis: &[SCFState], wicks: Option<&WicksView>) -> (f64, f64) {
    // Get the sorted pair of indices 
    let (a, b) = if i <= j {(i, j)} else {(j, i)};

    if input.wicks.enabled {
        let w = wicks.unwrap();
        calculate_hs_pair_wicks(ao, basis, noci_reference_basis, w, a, b, tol)
    } else {
        calculate_hs_pair_naive(ao, basis, a, b, tol)
    }
}

/// Accumulate the population change dn for a determinant i into the per-iteration delta vector.
/// Note that the actual populations stored in mc.walkers are not yet changed here.
/// # Arguments:
///     `mc`: MCState, contains information about the current Monte Carlo state.
///     `i`: usize, index of determinant i to be updated.
///     `dn`: i64, population change on determinant i. 
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
///     `mc`: MCState, contains information about the current Monte Carlo state.
fn apply_delta(mc: &mut MCState) -> Vec<PopulationUpdate> {
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
///     `c0`: Vec<f64>: Initial determinant coefficient vector.
///     `init_pop`: Number of walkers to start with.
///     `n`: Number of determinants.
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `iref`: usize, index of the first reference determinant.
///     `input`: Input, user specified input options.
///     `noci_reference_basis`: [SCFState], only the reference basis determinants.
///     `wicks`: [Vec<WicksReferencePair>], intermediates required for evaluating matrix
///              elements using the extended non-orthogonal Wick's theorem.
pub fn initialse_walkers(c0: &[f64], init_pop: i64, n: usize, ao: &AoData, basis: &[SCFState], iref: usize, tol: f64,
                         input: &Input, noci_reference_basis: &[SCFState], wicks: Option<&WicksView>) -> Walkers {
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
        let sgr = find_s(ao, basis, gamma, iref, tol, input, noci_reference_basis, wicks);
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
/// All currently implemented propagators have the same off-diagonal component:
/// |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}|.
/// # Arguments:
///      hlg: f64, matrix element H_{\Lambda\Gamma}
///      slg: f64, matrix element S_{\Lambda\Gamma}
///      es_s: f64, E_s^S(\tau) shift energy.
///      prop: Propagator, chosen propagator.
fn coupling(hlg: f64, slg: f64, es_s: f64, prop: &Propagator) -> f64 {
    // These are currently all the same, but for future-proofness if more propagators are
    // introduced this function is worth having.
    match prop {
        Propagator::Unshifted => hlg - es_s * slg,
        Propagator::Shifted => hlg - es_s * slg,
        Propagator::DoublyShifted => hlg - es_s * slg,
        Propagator::DifferenceDoublyShifted => hlg - es_s * slg,
    }
}

/// Initialise per-determinant heat-bath excitation generation quantities such that they can be
/// precomputed per-determinant rather than per walker. Calculates total weight
/// \Sum_{\Lambda \neq \Gamma} |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}|, cumulative sums
/// \sum_{i=1}^n |H_{i\Gamma} - E_s^S(\tau)S_{i\Gamma}|, the Lambda index corresponding to each
/// cumulative sum, and the couplings H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}.
/// # Arguments:
///     `gamma`: usize, parent determinant index.
///     `es_s`: f64, E_s^S(\tau) shift energy.
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `input`: Input, user specified input options.
///     `mc`: MCState, contains information about the current Monte Carlo state.
///     `noci_reference_basis`: [SCFState], only the reference basis determinants.
///     `wicks`: [Vec<WicksReferencePair>], intermediates required for evaluating matrix
///              elements using the extended non-orthogonal Wick's theorem.
fn init_heat_bath(gamma: usize, es_s: f64, ao: &AoData, basis: &[SCFState], input: &Input, tol: f64, 
                  noci_reference_basis: &[SCFState], wicks: Option<&WicksView>) -> HeatBath {
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
        let (hlg, slg) = find_hs(ao, basis, lambda, gamma, tol, input, noci_reference_basis, wicks);
        let k = coupling(hlg, slg, es_s, &input.prop.propagator);

        sumlg += k.abs();
        cumulatives.push(sumlg);
        lambdas.push(lambda);
        ks.push(k);
    }
    HeatBath {sumlg, cumulatives, lambdas, ks}
}

/// Perform off-diagonal spawning (amplitude transfer) step by calculating:
///     $P_{\text{Spawn}}(\Lambda|\Gamma) = \frac{\Delta\tau|H_{\Lambda\Gamma} - 
///     E_sS_{\Lambda\Gamma}|}{P_{\text{gen}}(\Lambda|\Gamma)}$
/// where if P_{\text{spawn}} > random float in [0, 1] we spawn a child walker onto determinant \Lambda 
/// with the same sign of its parent if H_{\Lambda\Gamma} > 0 and -sign if H_{\Lambda\Gamma} < 0. Furthermore, 
/// if P_{\text{spawn}} > 1 then we spawn floor(P_{\text{spawn}}) extra children and a final child with 
/// probability P_{\text{spawn}} - floor(P_{\text{spawn}}). This procedure can be made more
/// efficient (and is) by Drawing a count c for each candidate child \Lambda and applying the usual
/// spawning rules in one go.
/// # Arguments
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `gamma`: usize, index of determinant \Gamma. 
///     `ngamma`: i64, walker population on determinant \Gamma.
///     `dt`: f64, time-step.
///     `es_s`: f64, overlap-transformed shift energy.
///     `input`: Input, user specified input options.
///     `mc`: MCState, contains information about the current Monte Carlo state.
///     `irank`: usize, number of current rank.
///     `nranks`: usize, total number of ranks.
///     `send_buf`: Vec<Vec<PopulationUpdate>>, MPI send buffer which contains information about
///                 population updates that need to happen at determinants not owned by this rank.
///     `noci_reference_basis`: [SCFState], only the reference basis determinants.
///     `wicks`: [Vec<WicksReferencePair>], intermediates required for evaluating matrix
///              elements using the extended non-orthogonal Wick's theorem.
fn spawning(ao: &AoData, basis: &[SCFState], gamma: usize, ngamma: i64, es_s: f64, input: &Input, tol: f64, noci_reference_basis: &[SCFState], wicks: Option<&WicksView>,
            irank: usize, nranks: usize, rng: &mut SmallRng) -> (Vec<(usize, i64)>, Vec<PopulationUpdate>, Vec<(f64, u64)>) {

    let mut local: Vec<(usize, i64)> = Vec::new();
    let mut remote: Vec<PopulationUpdate> = Vec::new();
    let mut samples: Vec<(f64, u64)> = Vec::new();

    let mut nwalkers: u64 = ngamma.unsigned_abs();
    let ndets = basis.len();
    let mut nstates = ndets - 1;
    let parentsign: i64 = if ngamma > 0 {1} else {-1};

    let mut hb: Option<HeatBath> = None;
    let mut weight: f64 = 0.0;
    let mut hbi: usize = 0;
    if let ExcitationGen::HeatBath = input.qmc.as_ref().unwrap().excitation_gen {
        let hbinit = init_heat_bath(gamma, es_s, ao, basis, input, tol, noci_reference_basis, wicks);
        weight = hbinit.sumlg;
        hb = Some(hbinit);
    }
    
    // Iterate over candidate children.
    for lambda in 0..ndets {
        if lambda == gamma {continue;}
        if nwalkers == 0 {break;}
        
        // For each child determinant calculate the number of proposed spawnings c, the coupling
        // |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}| and the generation probability
        // P_{\text{gen}}(\Lambda|\Gamma). Proposed spawnings c_{\Lambda} are distributed in the
        // same manner as if we made a proposal for each walker.
        let (c, k, pgen) = match input.qmc.as_ref().unwrap().excitation_gen {
            // Uniform excitation generation considers all spawning events to be of equal
            // probability. P_{\text{gen}}(\Lambda|\Gamma) = 1 (N_{\text{dets}} - 1).
            ExcitationGen::Uniform => {
                // Probability next proposal lands on the current child determinant \Lambda.
                let p = 1.0 / (nstates as f64);
                // Uniform generation probability for any \Lambda != \Gamma.
                let pgen = 1.0 / (ndets as f64 - 1.0);
                // Number of spawning proposals for this \Lambda.
                let c: u64 = Binomial::new(nwalkers, p).unwrap().sample(rng);
                // Update remaining walkers and states.
                nwalkers -= c;
                nstates -= 1;
                // If there are no proposed spawnings for this \Lambds we can skip it.
                if c == 0 {continue;}
                let (hlg, slg) = find_hs(ao, basis, lambda, gamma, tol, input, noci_reference_basis, wicks);
                let k = coupling(hlg, slg, es_s, &input.prop.propagator);
                (c, k, pgen)
            }
            
            // Heat Bath excitation generation uses P_{\text{\gen}}(\Lambda|\Gamma) = |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}| 
            // / \sum_{\Omega \neq \Gamma} |H_{\Omega\Gamma} - E_s^S(\tau)S_{\Omega\Gamma}| such
            // that P_{\text{spawn}}(\Lambda|\Gamma) = \Delta\tau \sum_{\Omega \neq \Gamma} |H_{\Omega\Gamma} - E_s^S(\tau)S_{\Omega\Gamma}|
            // and so P_{\text{\gen}}(\Lambda|\Gamma)P_{\text{spawn}}(\Lambda|\Gamma) = \Delta\tau|H_{\Omega\Gamma} - E_s^S(\tau)S_{\Omega\Gamma}|,
            // meaning that states with strong coupling are more likely to have children spawned
            // upon them.
            ExcitationGen::HeatBath => {
                let hb = hb.as_ref().unwrap();
                // Current coupling term already computed by heat bath initialisation.
                let k = hb.ks[hbi];
                // Increment determinant count so we retrieve the correct coupling.
                hbi += 1;
                let w = k.abs();
                if weight <= 0.0 {break;}
                //P_{\text{\gen}}(\Lambda|\Gamma) = |H_{\Lambda\Gamma} - E_s^S(\tau)S_{\Lambda\Gamma}| 
                // / \sum_{\Omega \neq \Gamma} |H_{\Omega\Gamma} - E_s^S(\tau)S_{\Omega\Gamma}|
                let pgen = w / hb.sumlg;
                // At the beginning of the loop p is identical to pgen. But as weight is removed
                // they differ.
                let p = (w / weight).clamp(0.0, 1.0);
                // Number of spawning proposals for this \Lambda.
                let c: u64 = Binomial::new(nwalkers, p).unwrap().sample(rng);
                // Update remaining walkers and weights.
                nwalkers -= c;
                weight -= w;
                if w == 0.0 {continue;}
                if c == 0 {continue;}
                (c, k, pgen)
            }
        };

        let w = k.abs();
        let pspawn = input.prop.dt * w / pgen;
        if input.write.write_excitation_hist {samples.push((pspawn, c));}
        
        // Spawn floor(probability of spawning) children and additionally extra children with
        // probability: probability of spawning - floor(probability of spawning). 
        let m = pspawn.floor() as i64;
        let frac = pspawn - (m as f64);
        let extra = if frac > 0.0 {Binomial::new(c, frac).unwrap().sample(rng) as i64} else {0};

        let nchildren = (c as i64) * m + extra;
        if nchildren == 0 {continue;}

        let signk: i64 = if k > 0.0 {1} else {-1};
        let childsign: i64 = -signk * parentsign;
        let dn = childsign * nchildren;

        let destination = owner(lambda, nranks);
        if destination == irank {
            local.push((lambda, dn));
        } else {
            remote.push(PopulationUpdate { det: lambda as u64, dn });
        }
    }
    (local, remote, samples)
}

/// Perform diagonal death and cloning step by calculating:
///     $P_{\text{Death}}(\Lambda) = \Delta\tau(H_{\Lambda\Lambda}- 2E_s)$,
/// where which if P_{\text{death}} > 0 a walker dies with probability P_{\text{death}}, 
/// if P_{\text{death}} < 0 a walker is cloned with probability |P_{\text{death}}|. 
/// If P_{\text{death}} = 0, nothing will happen.  
/// # Arguments
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `gamma`: usize, index of determinant \Gamma. 
///     `ngamma`: i64, walker population on determinant \Gamma.
///     `es`: f64, non-overlap transformed shift energy.
///     `es_s`: f64, overlap-transformed shift energy.
///     `input`: Input, user specified input options.
fn death_cloning(ao: &AoData, basis: &[SCFState], gamma: usize, ngamma: i64, es: f64,  es_s: f64, input: &Input, tol: f64, noci_reference_basis: &[SCFState], 
                 wicks: Option<&WicksView>, rng: &mut SmallRng) -> Vec<(usize, i64)> {
    
    // Calculate matrix elements H_{\Gamma\Gamma}, S_{\Gamma\Gamma}.
    let (hgg, sgg) = find_hs(ao, basis, gamma, gamma, tol, input, noci_reference_basis, wicks);

    // Death probability.
    let pdeath = match input.prop.propagator {
        Propagator::Unshifted => input.prop.dt * (hgg - sgg * es_s),
        Propagator::Shifted => input.prop.dt * (hgg - sgg * es_s - es_s),
        Propagator::DoublyShifted => input.prop.dt * (hgg - sgg * es_s - es),
        Propagator::DifferenceDoublyShifted => input.prop.dt * (hgg - sgg * 0.5 * (es_s + es) - (es - es_s)),
    };
    let p = pdeath.abs();

    // Sign of parent walkers on state Gamma determines which way round death and cloning occur.
    let parent_sign = if ngamma > 0 {1} else {-1};
    
    let n = ngamma.abs();
    let m = p.floor() as i64;
    let frac = p - (m as f64);
   
    if pdeath == 0.0 {return Vec::new();}
    // Rather than iterate over all walkers we can sample binominal distribution.
    let extra = if frac > 0.0 {Binomial::new(n as u64, frac).unwrap().sample(rng) as i64} else {0};
    let nevents = n * m + extra;
    if nevents == 0 {return Vec::new();}
    
    // Accumulate population change in delta.
    let dn = if pdeath > 0.0 {-parent_sign} else {parent_sign};

    vec![(gamma, dn * nevents)]
}

/// Calculate projected energy as \frac{\sum_{H_{\Gamma, \text{Reference}}} N_{\Gamma}}{\sum_{S_{\Gamma, \text{Reference}}} N_{\Gamma}}.
/// # Arguments
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `walkers`: Walkers, object containing information about current walkers.
///     `iref`: usize, index of determinant we are projecting onto.
///     `world`: Communicator, MPI communicator object (MPI_COMM_WORLD).
///     `input`: Input, user specified input options.
///     `noci_reference_basis`: [SCFState], only the reference basis determinants.
///     `wicks`: [Vec<WicksReferencePair>], intermediates required for evaluating matrix
///              elements using the extended non-orthogonal Wick's theorem.
fn projected_energy(ao: &AoData, basis: &[SCFState], walkers: &Walkers, iref: usize, world: &impl Communicator, tol: f64,
                    input: &Input, noci_reference_basis: &[SCFState], wicks: Option<&WicksView>) -> f64 {
    
    // Calculate projected energy per Rayon thread.
    let (num, den) = walkers.occ().par_iter().fold(|| (0.0_f64, 0.0_f64), |(mut num, mut den), &gamma| {
        let ngamma = walkers.get(gamma) as f64;
        let (hgr, sgr) = find_hs(ao, basis, gamma, iref, tol, input, noci_reference_basis, wicks);
        num += ngamma * hgr;
        den += ngamma * sgr;
        (num, den)
    }).map(|(num, den)| (num, den)).reduce(|| (0.0, 0.0), |(a_num, a_den), (b_num, b_den)| (a_num + b_num, a_den + b_den));

    // Reduce contributions from each thread to get full projected energy across MPI ranks.
    let mut numglobal = 0.0;
    let mut denglobal = 0.0;
    world.all_reduce_into(&num, &mut numglobal, SystemOperation::sum());
    world.all_reduce_into(&den, &mut denglobal, SystemOperation::sum());

    numglobal / denglobal
}

/// Initialise the vector p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} N_{\Omega}. 
/// # Arguments:
///     `start`: usize, first determinant index owned by this rank.
///     `end`: usize, final determinant index owned by this rank.
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `walkers`: Walkers, object containing information about current walkers.
///     `world`: Communicator, MPI communicator object (MPI_COMM_WORLD).
///     `input`: Input, user specified input options.
///     `noci_reference_basis`: [SCFState], only the reference basis determinants.
///     `wicks`: [Vec<WicksReferencePair>], intermediates required for evaluating matrix
///              elements using the extended non-orthogonal Wick's theorem.
fn init_p(start: usize, end: usize, ao: &AoData, basis: &[SCFState], walkers: &Walkers, world: &impl Communicator, tol: f64, 
          input: &Input, noci_reference_basis: &[SCFState], wicks: Option<&WicksView>) -> Vec<f64> {

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
            let sgo = find_s(ao, basis, gamma, omega, tol, input, noci_reference_basis, wicks);
            pgamma += nomega * sgo;
        }
        p[gamma - start] = pgamma
    }
    p
}

/// Update the vector p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} N_{\Omega}.
/// # Arguments:
///     `start`: usize, first determinant index owned by this rank.
///     `end`: usize, final determinant index owned by this rank.
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `world`: Communicator, MPI communicator object (MPI_COMM_WORLD). 
///     `plocal`: [f64], vector p_{\Gamma} on this rank.
///     `dlocal`: [PopulationUpdate], local determinant population updates.
///     `input`: Input, user specified input options.
///     `noci_reference_basis`: [SCFState], only the reference basis determinants.
///     `wicks`: [Vec<WicksReferencePair>], intermediates required for evaluating matrix
///              elements using the extended non-orthogonal Wick's theorem.
fn update_p(start: usize, end: usize, ao: &AoData, basis: &[SCFState], world: &impl Communicator, plocal: &mut [f64], dlocal: &[PopulationUpdate], tol: f64,
            input: &Input, noci_reference_basis: &[SCFState], wicks: Option<&WicksView>) {

    let dglobal = gather_all_walkers(world, dlocal);

    plocal.par_iter_mut().enumerate().for_each_init(|| (), |_, (idx, pgamma)| {
        let gamma = start + idx;
        if gamma >= end {return;}
        
        //\Delta p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} \Delta N_{\Omega}.
        let mut dpgamma = 0.0_f64;
        for entry in &dglobal {
            // Here we use dn to mean population change.
            let dnomega = entry.dn as f64;
            let omega = entry.det as usize;
            let sgo = find_s(ao, basis, gamma, omega, tol, input, noci_reference_basis, wicks);
            dpgamma += dnomega * sgo;
        }
        *pgamma += dpgamma;
    });
}

/// Propagate according to the stochastic update equations for max_steps iterations.
/// # Arguments: 
///     `c0`: Vec<f64>, initial determinant coefficient vector to be translated into walker
///     populations.
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `es`: f64, initial shift energy.
///     `input`: Input, user specified input options.
///     `ref_indices`: [usize], indices of the reference determinants embedded in the full space.
///     `world`: Communicator, MPI communicator object (MPI_COMM_WORLD).
///     `noci_reference_basis`: [SCFState], only the reference basis determinants.
///     `wicks`: [Vec<WicksReferencePair>], intermediates required for evaluating matrix
///              elements using the extended non-orthogonal Wick's theorem.
pub fn step(c0: &[f64], ao: &AoData, basis: &[SCFState], es: &mut f64, input: &mut Input, ref_indices: &[usize], world: &impl Communicator, tol: f64, 
            noci_reference_basis: &[SCFState], wicks: Option<&WicksView>) -> (f64, Option<ExcitationHist>, StochStepTimings) {

    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let ndets = basis.len();
    let start = (ndets * irank) / nranks;
    let end = (ndets * (irank + 1)) / nranks;

    // Unwrap QMC propagation specific options
    let qmc = input.qmc.as_ref().unwrap();
    let ndets = basis.len();

    let iref = 0;

    // Construct reference index mask.
    let mut isref = vec![false; basis.len()];
    for &i in ref_indices {isref[i] = true;}
    
    // If the number of references is 1 then the unshifted propagator is correct.
    if ref_indices.len() == 1 {
        if irank == 0 {println!("Number of references is 1. Setting propagator to unshifted.....");}
        input.prop.propagator = Propagator::Unshifted;
    }

    // Initialise walker populations based on total initial population and c0.
    if irank == 0 {println!("Initialising walkers.....");}
    let w = initialse_walkers(c0, qmc.initial_population, ndets, ao, basis, iref, tol, input, noci_reference_basis, wicks);
    let w = local_walkers(w, irank, nranks);

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
    let pg = init_p(start, end, ao, basis, &w, world, tol, input, noci_reference_basis, wicks);
    let mut mc = MCState {walkers: w, delta: vec![0; ndets], changed: Vec::new(), rng, pg, excitation_hist};

    // Project onto determinant with index zero (this is usually the first RHF reference).
    let eproj = projected_energy(ao, basis, &mc.walkers, iref, world, tol, input, noci_reference_basis, wicks);

    // Initialise overlap-transformed shift E_s^S. This is distinct from es (E_s) in that E_s is
    // the shift updated using N_w, whilst E_s^S is updated using \tilde{N}_w.
    let mut es_s = *es;

    // Initialise populations.
    let mut nwscprev = 0.0;
    let nwscprev_local: f64 = mc.pg.iter().map(|x| x.abs()).sum();
    world.all_reduce_into(&nwscprev_local, &mut nwscprev, SystemOperation::sum());
    let mut nwcprev = qmc.initial_population;
    let mut nrefprev = 0.0; 
    let nrefprev_local: f64 = mc.pg.iter().enumerate().filter(|(k, _)| isref[start + *k]).map(|(_, x)| x.abs()).sum();
    world.all_reduce_into(&nrefprev_local, &mut nrefprev, SystemOperation::sum());

    // Print table header.
    let e0 = basis[0].e;
    let mut es_corr = 0.0;
    let mut es_s_corr = 0.0;
    if irank == 0 {
        println!("{}", "=".repeat(100));
        println!("{:<6} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}", 
                 "iter", "E", "Ecorr", "Shift (Es)", "Shift (EsS)", "Nw (||C||)", "Nw (||SC||)", "Nref(||SC||)");
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",  
                 0, eproj, eproj - e0, es_corr, es_s_corr, nwcprev as f64, nwscprev, nrefprev);
    }

    // Timings.
    let mut d_spawn_death_collect = 0.0f64;
    let mut d_acc_pack = 0.0f64;
    let mut d_mpi_exchange_updates = 0.0f64;
    let mut d_unpack_acc_recieved = 0.0f64;
    let mut d_apply_delta = 0.0f64;
    let mut d_update_p = 0.0f64;
    let mut d_calc_populations = 0.0f64;
    let mut d_eproj = 0.0f64;

    for it in 0..input.prop.max_steps {
        let t0 = Instant::now();

        // Rayon parallised QMC loop per MPI rank.
        let occ = mc.walkers.occ().to_vec();
        let updates: Vec<GammaUpdates> = occ.par_iter().map(|&gamma| {
            let ngamma = mc.walkers.get(gamma);
            if ngamma == 0 {return GammaUpdates {local: Vec::new(), remote: Vec::new(), samples: Vec::new()};}

            let s = seed ^ (it as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (gamma as u64).wrapping_mul(0xBF58476D1CE4E5B9) ^ (irank as u64).wrapping_mul(0x94D049BB133111EB);
            let mut rng = SmallRng::seed_from_u64(s);

            let mut local = Vec::new();
            let mut remote = Vec::new();
            let mut samples = Vec::new();
            
            // Death and cloning is entirely local.
            local.extend(death_cloning(ao, basis, gamma, ngamma, *es, es_s, input, tol, noci_reference_basis, wicks, &mut rng));
            // Spawning may or may not be local.
            let (l2, r2, s2) = spawning(ao, basis, gamma, ngamma, es_s, input, tol, noci_reference_basis, wicks, irank, nranks, &mut rng);
            local.extend(l2);
            remote.extend(r2);
            samples.extend(s2);

            GammaUpdates {local, remote, samples}
        }).collect();
        d_spawn_death_collect += t0.elapsed().as_secs_f64();
        
        let t0 = Instant::now();
        // Per MPI rank send buffers.
        let mut send: Vec<Vec<PopulationUpdate>> = (0..nranks).map(|_| Vec::new()).collect();
        // Fill the MPI rank quantities with the per Rayon thread data.
        for gu in updates {
            for (det, dn) in gu.local {
                add_delta(&mut mc, det, dn);
            }
            if input.write.write_excitation_hist && let Some(hist) = mc.excitation_hist.as_mut() {
                    for (p, n) in gu.samples {hist.addn(p, n);}
            }

            // Only need to update remote updates if we are using multiple MPI ranks,
            if nranks > 1 {
                for up in gu.remote {
                    let dest = owner(up.det as usize, nranks);
                    send[dest].push(up);
                }
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

        // Apply local and recieved deltas. Annhilation is fully local process here.  
        // The change in population is also stored to update overlap-transformed walker population.
        let t0 = Instant::now();
        let d = apply_delta(&mut mc);
        d_apply_delta += t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        update_p(start, end, ao, basis, world, &mut mc.pg, &d, tol, input, noci_reference_basis, wicks);
        d_update_p += t0.elapsed().as_secs_f64();
        
        let t0 = Instant::now();
        // Calculate non overlap-transformed walker population.
        let mut nwc = 0i64;
        let nwclocal = mc.walkers.norm() as i64;
        world.all_reduce_into(&nwclocal, &mut nwc, SystemOperation::sum());
        // Overlap-transformed walker population.
        let mut nwsc = 0.0;
        // Calculate overlap-transformed walker population as: 
        //\tilde{N}_w(\tau) = ||S_{\Gamma\Omega}C^\Omega(\tau)|| = \sum_{\Gamma} |p_{\Gamma}| N_{\Gamma}$
        let nwsc_local: f64 = mc.pg.iter().map(|x| x.abs()).sum();
        world.all_reduce_into(&nwsc_local, &mut nwsc, SystemOperation::sum());
        // Overlap-transformed reference walker population.
        let mut nref = 0.0; 
        // Calculate overlap transformed reference only walker population as:
        //\tilde{N}_{w, refs}(\tau) = ||S_{\Gamma\Omega}C^\Omega(\tau)|| = \sum_{\Gamma, \Gamma \in refs} |p_{\Gamma}| N_{\Gamma}$
        let nref_local: f64 = mc.pg.iter().enumerate().filter(|(k, _)| isref[start + *k]).map(|(_, x)| x.abs()).sum();
        world.all_reduce_into(&nref_local, &mut nref, SystemOperation::sum());
        d_calc_populations += t0.elapsed().as_secs_f64();
        
        // Activate overlap-transformed and non-overlap transformed shifts.
        if !reached_c && (nwc > qmc.target_population) {
            reached_c = true;
            nwcprev = nwc;
        }
        if !reached_sc && (nwsc > qmc.target_population as f64) {
            reached_sc = true;
            nwscprev = nwsc;
        }
        // Update shift once total populations have exceeded target population.
        if reached_c && (it + 1) % qmc.shift_update_freq == 0 {
            *es -= (qmc.shift_damping / (input.prop.dt * (qmc.shift_update_freq as f64))) * (nwc as f64 / nwcprev as f64).ln();
            nwcprev = nwc;
        }
        if reached_sc && (it + 1) % qmc.shift_update_freq == 0 {
            es_s -= (qmc.shift_damping / (input.prop.dt * (qmc.shift_update_freq as f64))) * (nwsc / nwscprev).ln();
            nwscprev = nwsc;
        }

        // Update energy. 
        let t0 = Instant::now();
        let iref = 0;
        let eproj = projected_energy(ao, basis, &mc.walkers, iref, world, tol, input, noci_reference_basis, wicks);
        d_eproj += t0.elapsed().as_secs_f64();
    
        es_corr = if reached_c {*es - e0} else {0.0};
        es_s_corr = if reached_sc {es_s - e0} else {0.0};

        // Print table rows.
        if irank == 0 {println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", 
                       it + 1, eproj, eproj - basis[0].e, es_corr, es_s_corr, nwc as f64, nwsc, nref);}
    }

    let timings = StochStepTimings {spawn_death_collect: Duration::from_secs_f64(d_spawn_death_collect), acc_pack: Duration::from_secs_f64(d_acc_pack), 
                                    mpi_exchange_updates: Duration::from_secs_f64(d_mpi_exchange_updates), unpack_acc_recieved: Duration::from_secs_f64(d_unpack_acc_recieved), 
                                    apply_delta: Duration::from_secs_f64(d_apply_delta), update_p: Duration::from_secs_f64(d_update_p), 
                                    calc_populations: Duration::from_secs_f64(d_calc_populations), eproj: Duration::from_secs_f64(d_eproj)};

    (eproj, mc.excitation_hist, timings)
}


