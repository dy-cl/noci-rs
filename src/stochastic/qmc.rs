// qmc.rs
use std::time::{Instant};
use std::path::Path;

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
use super::restart::RestartState;

use crate::noci::{calculate_s_pair, calculate_hs_pair};
use crate::mpiutils::{owner, local_walkers, communicate_spawn_updates, gather_all_walkers};
use super::restart::{read_restart_hdf5, write_restart_hdf5};

// Storage for QMC timings.
#[derive(Default, Clone)]
pub struct QMCTimings {
    pub initialise_walkers: f64,
    pub spawn_death_collect: f64,
    pub acc_pack: f64,
    pub mpi_exchange_updates: f64,
    pub unpack_acc_recieved: f64,
    pub apply_delta: f64,
    pub update_p: f64,
    pub calc_populations: f64,
    pub eproj: f64,
}

// Storage for immutable data required for stochastic propagation.
pub struct QMCData<'a> {
    pub ao: &'a AoData,
    pub basis: &'a [SCFState],
    pub input: &'a Input,
    pub wicks: Option<&'a WicksView>,
    pub mocache: &'a [MOCache],
    pub tol: f64,
}

// Storage for rank-local data layouts and metadata.
struct QMCRunInfo {
    irank: usize,
    nranks: usize,
    ndets: usize,
    start: usize,
    end: usize,
    iref: usize,
    base_seed: u64,
    rank_seed: u64,
}

// Storage for maximum size required for Wick's scratch.
struct ScratchSize {
    maxsame: usize,
    maxla: usize,
    maxlb: usize,
}

// Storage for Current shift values used during propagation.
#[derive(Clone, Copy)]
struct Shifts {
    es: f64,
    es_s: f64,
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
    // Walker distribution.
    pub walkers: Walkers,
    // Changes to walker population of length n determinants.
    pub delta: Vec<i64>,
    // Indices for which delta[i] != 0, i.e., determinants with changed population.
    pub changed: Vec<usize>,
    // Incrementally updated p_{\Gamma} = \sum_{\Omega} S_{\Gamma\Omega} N_{\Omega}.
    pub pg: Vec<f64>,
    // RNG.
    pub rng: SmallRng,
    // Histogrammed samples of P_{\text{Spawn}} for excit gen diagnostics.
    pub excitation_hist: Option<ExcitationHist> 
}

// Storage for the Incrementally updated projected-energy.
#[derive(Clone, Copy)]
struct ProjectedEnergyUpdate {
    iref: usize,
    num: f64,
    den: f64,
}

/// Storage for current walker populations in both the raw and overlap-transformed sense.
#[derive(Clone, Copy)]
struct PopulationStats {
    // Non-overlap transformed walker populations.
    nwc: i64,
    // Non-overlap transformed reference walker populations.
    nrefc: i64,
    // Overlap-transformed walker populations.
    nwsc: f64,
    // Overlap-transformed reference walker populations.
    nrefsc: f64,
}

impl PopulationStats {
    /// Construct raw and overlap-transformed walker population container.
    /// # Arguments:
    /// - `nwc`: Total non-overlap transformed walker population.
    /// - `nrefc`: Total non-overlap transformed reference walker population.
    /// - `nwsc`: Total overlap-transformed walker population.
    /// - `nrefsc`: Total overlap-transformed reference walker population.
    /// # Returns
    /// - `PopulationStats`: Walker population statistics.
    fn new(nwc: i64, nrefc: i64, nwsc: f64, nrefsc: f64) -> Self {
        Self {nwc, nrefc, nwsc, nrefsc}
    }
}

/// Storage for QMC step bookkeeping data.
struct PropagationState {
    // Full Monte Carlo state.
    mc: MCState,
    // Incrementally updated projected-energy.
    pe: ProjectedEnergyUpdate,
    // Overlap transformed shift.
    es_s: f64,
    // Walker populations at previous shift update.
    prev_pop: PopulationStats,
    // Current walker populations.
    cur_pop: PopulationStats,
    // From where did iterations begin (was a restart file used?).
    start_iter: usize,
    // Has the overlap-transformed population reached the target.
    reached_sc: bool,
    // Has the non-overlap-transformed population reached the target.
    reached_c: bool,
    // Current projected-energy.
    eprojcur: f64,
}

impl PropagationState {
    /// Construct propagation state from the Monte Carlo state, projected-energy,
    /// shift, and population bookkeeping quantities.
    /// # Arguments:
    /// - `mc`: Monte Carlo state.
    /// - `pe`: Incrementally updated projected-energy.
    /// - `es_s`: Overlap transformed shift.
    /// - `start_iter`: Iteration from which propagation begins.
    /// - `reached_sc`: Has the overlap-transformed population reached the target.
    /// - `reached_c`: Has the non-overlap-transformed population reached the target.
    /// - `prev_pop`: Walker populations at the previous shift update.
    /// # Returns
    /// - `PropagationState`: Initialised propagation state.
    fn new(mc: MCState, pe: ProjectedEnergyUpdate, es_s: f64, start_iter: usize, reached_sc: bool, reached_c: bool, prev_pop: PopulationStats) -> Self {
        let eprojcur = pe.num / pe.den;
        Self {mc, pe, es_s, prev_pop, cur_pop: prev_pop, start_iter, reached_sc, reached_c, eprojcur}
    }
    
    /// Construct propagation state for a run beginning from iteration zero.
    /// # Arguments:
    /// - `mc`: Monte Carlo state.
    /// - `pe`: Incrementally updated projected-energy.
    /// - `es_s`: Overlap transformed shift.
    /// - `prev_pop`: Initial walker populations.
    /// # Returns
    /// - `PropagationState`: Propagation state for a stochastic run from iteration zero.
    fn fresh(mc: MCState, pe: ProjectedEnergyUpdate, es_s: f64, prev_pop: PopulationStats) -> Self {
        Self::new(mc, pe, es_s, 0, false, false, prev_pop)
    }
    
    /// Construct propagation state for a run resumed from a restart file.
    /// # Arguments:
    /// - `mc`: Monte Carlo state.
    /// - `pe`: Incrementally updated projected-energy.
    /// - `es_s`: Overlap transformed shift.
    /// - `start_iter`: Iteration from which propagation resumes.
    /// - `reached_sc`: Has the overlap-transformed population reached the target.
    /// - `reached_c`: Has the non-overlap-transformed population reached the target.
    /// - `prev_pop`: Walker populations stored at the previous shift update.
    /// # Returns
    /// - `PropagationState`: Propagation state for a restarted stochastic run.
    fn restart(mc: MCState, pe: ProjectedEnergyUpdate, es_s: f64, start_iter: usize, reached_sc: bool, reached_c: bool, prev_pop: PopulationStats) -> Self {
        Self::new(mc, pe, es_s, start_iter, reached_sc, reached_c, prev_pop)
    }
}

// Storage for results of a single propagation step.
struct PropagationResult {
    // Population updates for determinants owned by current rank.
    local: Vec<(usize, i64)>,
    // Population updates for determinants owned by another rank.
    remote: Vec<PopulationUpdate>,
    // Excitation generation samples.
    samples: Vec<f64>,
}

// Storage for per thread propagation quantities.
struct ThreadPropagation {
    // Population changes generated by this thread that belong to determinants owned by current MPI rank.
    local: Vec<(usize, i64)>,
    // Population changes generated by this thread that belong to determinants owned by another MPI rank.
    remote: Vec<PopulationUpdate>,
    // Excitation generation samples.
    samples: Vec<f64>,
    // Thread local RNG.
    rng: SmallRng,
    // Per thread scratch space for extended non-orthogonal Wick's theorem.
    scratch: WickScratchSpin,
}

impl ThreadPropagation {
    /// Perform the diagonal death and cloning step for a parent determinant.
    /// # Arguments:
    /// - `gamma`: Index of parent determinant `\Gamma`.
    /// - `ngamma`: Walker population on determinant `\Gamma`.
    /// - `shifts`: Current non-overlap and overlap-transformed shifts.
    /// - `data`: Immutable stochastic propagation data.
    /// # Returns
    /// - `()`: Appends local death/cloning population updates to `self.local`.
    fn death_cloning(&mut self, gamma: usize, ngamma: i64, shifts: Shifts, data: &QMCData<'_>) {
        let (hgg, sgg) = find_hs(data, gamma, gamma, &mut self.scratch);

        let pdeath = match data.input.prop_ref().propagator {
            Propagator::Unshifted => data.input.prop_ref().dt * (hgg - sgg * shifts.es_s),
            Propagator::Shifted => data.input.prop_ref().dt * (hgg - sgg * shifts.es_s - shifts.es_s),
            Propagator::DoublyShifted => data.input.prop_ref().dt * (hgg - sgg * shifts.es_s - shifts.es),
            Propagator::DifferenceDoublyShiftedU1 => data.input.prop_ref().dt * (hgg - sgg * 0.5 * (shifts.es_s + shifts.es) - (shifts.es - shifts.es_s)),
            Propagator::DifferenceDoublyShiftedU2 => data.input.prop_ref().dt * (hgg - sgg * shifts.es_s - (shifts.es - shifts.es_s)),
        };
        if pdeath == 0.0 {
            return;
        }

        let p = pdeath.abs();
        let parent_sign = if ngamma > 0 {1} else {-1};
        let n = ngamma.abs();
        let m = p.floor() as i64;
        let frac = p - (m as f64);
        let extra = if frac > 0.0 {Binomial::new(n as u64, frac).unwrap().sample(&mut self.rng) as i64} else {0};
        let nevents = n * m + extra;
        if nevents == 0 {
            return;
        }

        let dn = if pdeath > 0.0 {-parent_sign} else {parent_sign};
        self.local.push((gamma, dn * nevents));
    }
    
    /// Perform the off-diagonal spawning step for a parent determinant.
    /// # Arguments:
    /// - `gamma`: Index of parent determinant `\Gamma`.
    /// - `ngamma`: Walker population on determinant `\Gamma`.
    /// - `shifts`: Current non-overlap and overlap-transformed shifts.
    /// - `data`: Immutable stochastic propagation data.
    /// - `run`: Rank-local run metadata.
    /// # Returns
    /// - `()`: Appends local and remote spawning updates to `self.local` and `self.remote`,
    ///   and excitation histogram samples to `self.samples`.
    fn spawning(&mut self, gamma: usize, ngamma: i64, shifts: Shifts, data: &QMCData<'_>, run: &QMCRunInfo) {
        let parent_sign = if ngamma > 0 {1} else {-1};
        let nwalkers = ngamma.unsigned_abs();

        let mut hb: Option<HeatBath> = None;
        if let ExcitationGen::HeatBath = data.input.qmc.as_ref().unwrap().excitation_gen {
            hb = Some(init_heat_bath(gamma, shifts, data, &mut self.scratch));
        }

        for _ in 0..nwalkers {
            let (pgen, k, lambda) = match data.input.qmc.as_ref().unwrap().excitation_gen {
                ExcitationGen::Uniform => pgen_uniform(gamma, shifts, data, &mut self.rng, &mut self.scratch),
                ExcitationGen::HeatBath => pgen_heat_bath(gamma, shifts, data, &mut self.rng, hb.as_ref().unwrap(), &mut self.scratch),
                ExcitationGen::ApproximateHeatBath => unimplemented!(),
            };

            let pspawn = data.input.prop_ref().dt * k.abs() / pgen;
            if data.input.write.write_excitation_hist {
                self.samples.push(pspawn);
            }

            let m = pspawn.floor() as i64;
            let frac = pspawn - (m as f64);
            let extra = if self.rng.gen_range(0.0..1.0) < frac {1} else {0};
            let nchildren = m + extra;
            if nchildren == 0 {
                continue;
            }

            let sign = if k > 0.0 {1} else {-1};
            let child_sign = -sign * parent_sign;
            let dn = child_sign * nchildren;

            if run.nranks == 1 {
                self.local.push((lambda, dn));
            } else {
                let destination = owner(lambda, run.nranks);
                if destination == run.irank {
                    self.local.push((lambda, dn));
                } else {
                    self.remote.push(PopulationUpdate {det: lambda as u64, dn});
                }
            }
        }
    }
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

// Storage for histogrammed data.
#[derive(Clone)]
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

/// Find overlap matrix element S_{ij}.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `i`: Index of state `i`.
/// - `j`: Index of state `j`.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element `S_{ij}`.
fn find_s(data: &QMCData<'_>, i: usize, j: usize, scratch: &mut WickScratchSpin) -> f64 {
    // Get the sorted pair of indices 
    let (a, b) = if i <= j {(i, j)} else {(j, i)};
    calculate_s_pair(data.ao, &data.basis[a], &data.basis[b], data.tol, data.input, data.wicks, Some(scratch))
}

/// Find Hamiltonian and overlap matrix elements H_{ij} and S_{ij}.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `i`: Index of state `i`.
/// - `j`: Index of state `j`.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `(f64, f64)`: Hamiltonian and overlap matrix elements `H_{ij}` and `S_{ij}`.
fn find_hs(data: &QMCData<'_>, i: usize, j: usize, scratch: &mut WickScratchSpin) -> (f64, f64) {
    // Get the sorted pair of indices 
    let (a, b) = if i <= j {(i, j)} else {(j, i)};
    calculate_hs_pair(data.ao, &data.basis[a], &data.basis[b], data.tol, data.input, data.mocache, data.wicks, Some(scratch))
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

/// For each entry in initial coefficient vector c0 calculate `(c0_i / ||c||) * N_0` and assign
/// this value as the initial walker population on determinant `i`. The sign of the population is
/// given by the sign of `c0_i`.
/// # Arguments:
/// - `c0`: Initial determinant coefficient vector.
/// - `init_pop`: Number of walkers to start with.
/// - `n`: Number of determinants.
/// - `data`: Immutable stochastic propagation data.
/// - `iref`: Index of the projected reference determinant.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `Walkers`: Initial walker population.
fn initialise_walkers(c0: &[f64], init_pop: i64, n: usize, data: &QMCData<'_>, iref: usize, scratch: &mut WickScratchSpin) -> Walkers {
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
        let sgr = find_s(data, gamma, iref, scratch);
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

/// Initialise exact heat-bath excitation generation data for determinant `gamma`.
/// # Arguments:
/// - `gamma`: Parent determinant index.
/// - `shifts`: Current non-overlap and overlap-transformed shifts.
/// - `data`: Immutable stochastic propagation data.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `HeatBath`: Precomputed heat-bath excitation generation data for determinant `gamma`.
fn init_heat_bath(gamma: usize, shifts: Shifts, data: &QMCData<'_>, scratch: &mut WickScratchSpin) -> HeatBath {
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
fn pgen_uniform(gamma: usize, shifts: Shifts, data: &QMCData<'_>, rng: &mut SmallRng, scratch: &mut WickScratchSpin) -> (f64, f64, usize) {
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
fn pgen_heat_bath(gamma: usize, shifts: Shifts, data: &QMCData<'_>, rng: &mut SmallRng, hb: &HeatBath, scratch: &mut WickScratchSpin) -> (f64, f64, usize) {
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

/// Initialise the running projected-energy state
/// `\frac{\sum_{\Gamma} N_{\Gamma} H_{\Gamma,\mathrm{ref}}}{\sum_{\Gamma} N_{\Gamma} S_{\Gamma,\mathrm{ref}}}`
/// by performing the full initial sweep over the occupied determinant space.
/// # Arguments:
/// - `walkers`: Object containing information about current walker distribution.
/// - `iref`: Index of the determinant onto which the energy is projected.
/// - `data`: Immutable stochastic propagation data.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `ProjectedEnergyUpdate`: Initial projected-energy numerator and denominator.
fn init_projected_energy(walkers: &Walkers, iref: usize, data: &QMCData<'_>, world: &impl Communicator) -> ProjectedEnergyUpdate {
    let (num_local, den_local) = walkers.occ().par_iter().fold(|| (0.0_f64, 0.0_f64, WickScratchSpin::new()), |(mut num, mut den, mut scratch), &gamma| {
        let ngamma = walkers.get(gamma) as f64;
        let (hgr, sgr) = find_hs(data, gamma, iref, &mut scratch);
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
/// - `d`: Net population changes applied in the current iteration.
/// - `pe`: Running projected-energy state to update in place.
/// - `data`: Immutable stochastic propagation data.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `()`: Updates `pe` in place.
fn update_projected_energy(d: &[PopulationUpdate], pe: &mut ProjectedEnergyUpdate, data: &QMCData<'_>, world: &impl Communicator) {
    let iref = pe.iref;

    let (dnum_local, dden_local) = d.par_iter().fold(|| (0.0_f64, 0.0_f64, WickScratchSpin::new()), |(mut dnum, mut dden, mut scratch), up| {
        let gamma = up.det as usize;
        let dn = up.dn as f64;
        let (hgr, sgr) = find_hs(data, gamma, iref, &mut scratch);
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

/// Initialise the vector `p_{\Gamma} = \sum_\Omega S_{\Gamma,\Omega} N_{\Omega}`.
/// # Arguments:
/// - `walkers`: Object containing information about current walker distribution.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `Vec<f64>`: Local portion of the overlap-transformed population vector `p_{\Gamma}`.
fn init_p(walkers: &Walkers, data: &QMCData<'_>, run: &QMCRunInfo, world: &impl Communicator, scratch: &mut WickScratchSpin) -> Vec<f64> {
    let local: Vec<PopulationUpdate> = walkers.occ().iter().map(|&i| PopulationUpdate {det: i as u64, dn: walkers.get(i)}).collect();
    let global = gather_all_walkers(world, &local);

    let mut p = vec![0.0; run.end - run.start];
    for gamma in run.start..run.end {
        //p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} N_{\Omega}.
        let mut pgamma = 0.0;
        for entry in &global {
            // Here we use dn to mean total population.
            let nomega = entry.dn as f64;
            let omega = entry.det as usize;
            let sgo = find_s(data, gamma, omega, scratch);
            pgamma += nomega * sgo;
        }
        p[gamma - run.start] = pgamma
    }
    p
}

/// Update the vector `p_{\Gamma} = \sum_\Omega S_{\Gamma,\Omega} N_{\Omega}`.
/// # Arguments:
/// - `plocal`: Local portion of `p_{\Gamma}` on this rank.
/// - `dlocal`: Local determinant population updates.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `()`: Updates `plocal` in place.
fn update_p(plocal: &mut [f64], dlocal: &[PopulationUpdate], data: &QMCData<'_>, run: &QMCRunInfo, world: &impl Communicator) {
    let dglobal = gather_all_walkers(world, dlocal);

    // \Delta p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} \Delta N_{\Omega}.
    plocal.par_iter_mut().enumerate().for_each_init(WickScratchSpin::new, |scratch, (k, pgamma)| {
        let gamma = run.start + k;
        let mut dp = 0.0;
        for update in &dglobal {
            let omega = update.det as usize;
            dp += find_s(data, gamma, omega, scratch) * update.dn as f64;
        }
        *pgamma += dp;
    });
}

/// Determine the maximum scratch sizes required for computation of matrix elements using extended
/// non-orthogonal Wick's theorem depending on the maximum excitation rank present in the basis.
/// # Arguments:
/// - `basis`: Full list of the NOCI-QMC basis.
/// # Returns
/// - `(usize, usize, usize)`: Maximum same-spin scratch size, alpha excitation size, and beta
///   excitation size.
fn max_scratch_sizes(basis: &[SCFState]) -> (usize, usize, usize) {
    let maxexa = basis.iter().map(|st| st.excitation.alpha.holes.len()).max().unwrap_or(0);
    let maxexb = basis.iter().map(|st| st.excitation.beta.holes.len()).max().unwrap_or(0);
    let maxsame = 2 * maxexa.max(maxexb);
    let maxla = 2 * maxexa;
    let maxlb = 2 * maxexb;
    (maxsame, maxla, maxlb)
}


/// Initialise projected-energy, walker distribution, `p_{\Gamma}` vector, and population totals
/// across ranks. Initialisation occurs either from an initial coefficient vector `c0` or from a
/// restart file.
/// # Arguments:
/// - `c0`: Initial determinant coefficient vector.
/// - `es`: Non-overlap transformed shift energy.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// - `isref`: Boolean mask specifying which determinants are reference determinants.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `PropagationState`: Initialised NOCI-QMC state with required bookkeeping parameters.
fn initialise_qmc_state(c0: &[f64], es: &mut f64, data: &QMCData<'_>, run: &QMCRunInfo, isref: &[bool], 
                        world: &impl Communicator, scratch: &mut WickScratchSpin) -> PropagationState {

    let qmc = data.input.qmc.as_ref().unwrap();
    // Use restart file if avaliable.
    if let Some(path) = data.input.write.read_restart.as_deref() {
        if run.irank == 0 {
            println!("Reading restart from {path}");
        }

        let rs = read_restart_hdf5(path, world, run.ndets).unwrap();
    
        // Read shifts and populations.
        *es = rs.es;
        let es_s = rs.es_s;

        let mc = MCState {
            walkers: rs.walkers, 
            delta: vec![0; run.ndets], 
            changed: Vec::new(), 
            rng: SmallRng::seed_from_u64(run.rank_seed),
            pg: rs.pg, 
            excitation_hist: rs.excitation_hist
        };
        
        // Initialise projected-energy.
        let pe = init_projected_energy(&mc.walkers, run.iref, data, world);
        
        let reached_c = rs.nwprevc >= qmc.target_population;
        let reached_sc = rs.nwprevsc >= qmc.target_population as f64;
        let start_iter = rs.iter + 1;
        let prev_pop = PopulationStats::new(rs.nwprevc, rs.nrefprevc, rs.nwprevsc, rs.nrefprevsc);
        PropagationState::restart(mc, pe, es_s, start_iter, reached_sc, reached_c, prev_pop)
    } else {
        if run.irank == 0 {
            println!("Initialising walkers.....");
        }
        
        // Initialise walker populations from initial coefficient vector.
        let w = initialise_walkers(c0, qmc.initial_population, run.ndets, data, run.iref, scratch);
        let w = local_walkers(w, run.irank, run.nranks);

        let excitation_hist = if data.input.write.write_excitation_hist {
            Some(ExcitationHist::new(-60.0, 1e-12, 100))
        } else {
            None
        };

        let pg = init_p(&w, data, run, world, scratch);
        let mc = MCState {walkers: w, delta: vec![0; run.ndets], changed: Vec::new(), rng: SmallRng::seed_from_u64(run.rank_seed), pg, excitation_hist};
        
        // Initialise projected-energy energy.
        let pe = init_projected_energy(&mc.walkers, run.iref, data, world);
        
        // Initialise overlap-transformed walker populations.
        let mut nwprevsc = 0.0;
        let nwprevsc_local: f64 = mc.pg.iter().map(|x| x.abs()).sum();
        world.all_reduce_into(&nwprevsc_local, &mut nwprevsc, SystemOperation::sum());
        let mut nrefprevsc = 0.0;
        let nrefprevsc_local: f64 = mc.pg.iter().enumerate().filter(|(k, _)| isref[run.start + *k]).map(|(_, x)| x.abs()).sum();
        world.all_reduce_into(&nrefprevsc_local, &mut nrefprevsc, SystemOperation::sum());
        
        // Initialise non-overlap-transformed walker populations.
        let nwprevc = qmc.initial_population;
        let mut nrefprevc = 0i64;
        let nrefprevc_local: i64 = mc.walkers.occ().iter().filter(|&&det| isref[det]).map(|&det| mc.walkers.get(det).abs()).sum();
        world.all_reduce_into(&nrefprevc_local, &mut nrefprevc, SystemOperation::sum());

        let prev_pop = PopulationStats::new(nwprevc, nrefprevc, nwprevsc, nrefprevsc);
        PropagationState::fresh(mc, pe, *es, prev_pop)
    }
}

/// Perform spawning and death/cloning steps over the currently occupied determinants.
/// # Arguments:
/// - `it`: Current iteration number.
/// - `mc`: Contains the current Monte Carlo state.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// - `scratchsize`: Maximum sizes required for per-thread Wick scratch space.
/// - `shifts`: Current non-overlap and overlap-transformed shifts.
/// - `timings`: Accumulated stochastic propagation timings.
/// # Returns
/// - `PropagationResult`: Local and remote population updates together with spawning probability samples.
fn propagate_iteration(it: usize, mc: &MCState, data: &QMCData<'_>, run: &QMCRunInfo, scratchsize: &ScratchSize, 
                       shifts: Shifts, timings: &mut QMCTimings) -> PropagationResult {

    let initialise = || -> ThreadPropagation {
        let tid = rayon::current_thread_index().unwrap_or(0) as u64;
        ThreadPropagation{
            local: Vec::new(),
            remote: Vec::new(),
            samples: Vec::new(),
            rng: SmallRng::seed_from_u64(run.rank_seed ^ tid ^ ((it as u64).wrapping_mul(0x9E3779B97F4A7C15))),
            scratch: WickScratchSpin::with_sizes(scratchsize.maxsame, scratchsize.maxla, scratchsize.maxlb),
        }
    };

    let propagate = |mut acc: ThreadPropagation, &gamma: &usize| -> ThreadPropagation {
        let ngamma = mc.walkers.get(gamma);
        if ngamma == 0 {
            return acc;
        }

        acc.death_cloning(gamma, ngamma, shifts, data);
        acc.spawning(gamma, ngamma, shifts, data, run);
        acc
    };

    let merge = |mut a: ThreadPropagation, mut b: ThreadPropagation| -> ThreadPropagation {
        a.local.append(&mut b.local);
        a.remote.append(&mut b.remote);
        a.samples.append(&mut b.samples);
        a
    };

    let t0 = Instant::now();
    let acc = mc.walkers.occ().par_iter().fold(initialise, propagate).reduce(
        || ThreadPropagation {
            local: Vec::new(),
            remote: Vec::new(),
            samples: Vec::new(),
            rng: SmallRng::seed_from_u64(0),
            scratch: WickScratchSpin::new(),
        },
        merge,
    );
    timings.spawn_death_collect += t0.elapsed().as_secs_f64();

    PropagationResult {
        local: acc.local,
        remote: acc.remote,
        samples: acc.samples,
    }
}

/// Accumulate thread-local updates into the global delta vector and exchange any remote
/// updates across MPI ranks.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `send`: Per-rank MPI send buffers.
/// - `prop`: Population updates generated in the spawning and death/cloning step.
/// - `input`: User specified input options.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `nranks`: Total number of MPI ranks.
/// - `timings`: Accumulated stochastic propagation timings.
/// # Returns
/// - `i32`: Global indicator for whether any population changes occurred on any rank.
fn accumulate_updates(mc: &mut MCState, send: &mut [Vec<PopulationUpdate>], prop: PropagationResult, input: &Input, world: &impl Communicator,
                      nranks: usize, timings: &mut QMCTimings) -> i32 {
    let t0 = Instant::now();

    if nranks > 1 {
        for buf in send.iter_mut() {
            buf.clear();
        }
    }

    for (det, dn) in prop.local {
        add_delta(mc, det, dn);
    }

    if input.write.write_excitation_hist && let Some(hist) = mc.excitation_hist.as_mut() {
        for p in prop.samples {
            hist.add(p);
        }
    }

    if nranks > 1 {
        for up in prop.remote {
            let dest = owner(up.det as usize, nranks);
            send[dest].push(up);
        }
    }

    timings.acc_pack += t0.elapsed().as_secs_f64();

    if nranks > 1 {
        let t0 = Instant::now();
        let received = communicate_spawn_updates(world, send);
        timings.mpi_exchange_updates += t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        for update in received {
            add_delta(mc, update.det as usize, update.dn);
        }
        timings.unpack_acc_recieved += t0.elapsed().as_secs_f64();
    }

    let changed = (!mc.changed.is_empty()) as i32;
    let mut changedglobal = 0;
    world.all_reduce_into(&changed, &mut changedglobal, SystemOperation::max());
    changedglobal
}

/// Compute the current non-overlap transformed and overlap-transformed walker populations.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `isref`: Boolean mask specifying which determinants are reference determinants.
/// - `run`: Rank-local run metadata.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `timings`: Accumulated stochastic propagation timings.
/// # Returns
/// - `PopulationStats`: Current total and reference populations in both representations.
fn compute_populations(mc: &MCState, isref: &[bool], run: &QMCRunInfo, world: &impl Communicator, timings: &mut QMCTimings) -> PopulationStats {
    let t0 = Instant::now();

    let nwclocal = mc.walkers.norm();
    let mut nwc = 0i64;
    world.all_reduce_into(&nwclocal, &mut nwc, SystemOperation::sum());

    let nrefc_local: i64 = mc.walkers.occ().iter().filter(|&&det| isref[det]).map(|&det| mc.walkers.get(det).abs()).sum();
    let mut nrefc = 0i64;
    world.all_reduce_into(&nrefc_local, &mut nrefc, SystemOperation::sum());

    let nwsc_local: f64 = mc.pg.iter().map(|x| x.abs()).sum();
    let mut nwsc = 0.0;
    world.all_reduce_into(&nwsc_local, &mut nwsc, SystemOperation::sum());

    let nrefsc_local: f64 = mc.pg.iter().enumerate().filter(|(k, _)| isref[run.start + *k]).map(|(_, x)| x.abs()).sum();
    let mut nrefsc = 0.0;
    world.all_reduce_into(&nrefsc_local, &mut nrefsc, SystemOperation::sum());

    timings.calc_populations += t0.elapsed().as_secs_f64();

    PopulationStats {nwc, nrefc, nwsc, nrefsc}
}

/// Cache the latest population statistics inside the propagation state for later
/// printing and possible early exiting.
/// # Arguments:
/// - `state`: Propagation state containing QMC stats.
/// - `stats`: Population statistics computed for the current iteration.
/// # Returns
/// - `()`: Updates cached population values in `state` in place.
fn cache_population_stats(state: &mut PropagationState, stats: &PopulationStats) {
    state.cur_pop.nwc = stats.nwc;
    state.cur_pop.nrefc = stats.nrefc;
    state.cur_pop.nwsc = stats.nwsc;
    state.cur_pop.nrefsc = stats.nrefsc;
}

/// Update the shift energies according to the current walker populations and shift.
/// # Arguments:
/// - `it`: Current iteration number.
/// - `stats`: Population statistics computed for the current iteration.
/// - `state`: Propagation state containing QMC stats.
/// - `es`: Non-overlap transformed shift energy.
/// - `input`: User specified input options.
/// # Returns
/// - `()`: Updates the shift energies and associated state variables in place.
fn update_shifts(it: usize, stats: &PopulationStats, state: &mut PropagationState, es: &mut f64, input: &Input) {
    let qmc = input.qmc.as_ref().unwrap();

    if !state.reached_c && stats.nwc > qmc.target_population {
        state.reached_c = true;
        state.prev_pop.nwc = stats.nwc;
    }

    if !state.reached_sc && stats.nwsc > qmc.target_population as f64 {
        state.reached_sc = true;
        state.prev_pop.nwsc = stats.nwsc;
    }

    if state.reached_c && (it + 1).is_multiple_of(qmc.shift_update_freq) {
        *es -= (qmc.shift_damping / (input.prop_ref().dt * (qmc.shift_update_freq as f64))) * (stats.nwc as f64 / state.prev_pop.nwc as f64).ln();
        state.prev_pop.nwc = stats.nwc;
    }

    if state.reached_sc && (it + 1).is_multiple_of(qmc.shift_update_freq) {
        state.es_s -= (qmc.shift_damping / (input.prop_ref().dt * (qmc.shift_update_freq as f64))) * (stats.nwsc / state.prev_pop.nwsc).ln();
        state.prev_pop.nwsc = stats.nwsc;
    }
}

/// Print the iteration table header on rank zero.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// # Returns
/// - `()`: Writes the table header to stdout on rank zero.
fn print_header(irank: usize) {
    if irank == 0 {
        println!("{}", "=".repeat(100));
        println!("{:<6} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
                 "iter", "E", "Ecorr", "Shift (Es)", "Shift (EsS)", "Nw (||C||)", "Nref (||C||)", "Nw (||SC||)", "Nref(||SC||)");
    }
}

/// Print the initial iteration line on rank zero.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `state`: Propagation state containing QMC stats.
/// - `e0`: Energy of the first basis determinant.
/// # Returns
/// - `()`: Writes the initial iteration line to stdout on rank zero.
fn print_initial_row(irank: usize, state: &PropagationState, e0: f64) {
    if irank == 0 {
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
                 0, state.eprojcur, state.eprojcur - e0, 0.0, 0.0, state.prev_pop.nwc as f64, state.prev_pop.nrefc as f64, 
                 state.prev_pop.nwsc, state.prev_pop.nrefsc);
    }
}

/// Print an iteration line using the cached population statistics stored in the run state.
/// This is used when no population changes occurred during an iteration and so we exit early.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `iter`: Iteration number to print.
/// - `state`: Propagation state containing QMC stats.
/// - `e0`: Energy of the first basis determinant.
/// - `es`: Non-overlap transformed shift energy.
/// # Returns
/// - `()`: Writes the cached iteration line to stdout on rank zero.
fn print_cached_row(irank: usize, iter: usize, state: &PropagationState, e0: f64, es: f64) {
    let es_corr = if state.reached_c {es - e0} else {0.0};
    let es_s_corr = if state.reached_sc {state.es_s - e0} else {0.0};

    if irank == 0 {
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
                 iter, state.eprojcur, state.eprojcur - e0, es_corr, es_s_corr, state.cur_pop.nwc as f64, 
                 state.cur_pop.nrefc as f64, state.cur_pop.nwsc, state.cur_pop.nrefsc);
    }
}

/// Print an iteration line using the current population statistics.
/// # Arguments:
/// - `irank`: Rank of the current MPI process.
/// - `iter`: Iteration number to print.
/// - `state`: Propagation state containing QMC stats.
/// - `stats`: Population statistics computed for the current iteration.
/// - `e0`: Energy of the first basis determinant.
/// - `es`: Non-overlap transformed shift energy.
/// # Returns
/// - `()`: Writes the current iteration line to stdout on rank zero.
fn print_row(irank: usize, iter: usize, state: &PropagationState, stats: &PopulationStats, e0: f64, es: f64) {
    let es_corr = if state.reached_c {es - e0} else {0.0};
    let es_s_corr = if state.reached_sc {state.es_s - e0} else {0.0};

    if irank == 0 {
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}",
                 iter, state.eprojcur, state.eprojcur - e0, es_corr, es_s_corr, stats.nwc as f64, stats.nrefc as f64, stats.nwsc, stats.nrefsc);
    }
}

/// Check for a `STOP` file, write a restart if required, and return early from the
/// stochastic propagation.
/// # Arguments:
/// - `it`: Current iteration number.
/// - `state`: Propagation state containing QMC bookkeeping data.
/// - `shifts`: Current non-overlap and overlap-transformed shifts.
/// - `run`: Rank-local run metadata.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `timings`: Accumulated stochastic propagation timings.
/// # Returns
/// - `Option<(f64, Option<ExcitationHist>, QMCTimings)>`: Final return values if a
///   stop was requested, otherwise `None`.
fn check_stop(it: usize, state: &mut PropagationState, shifts: Shifts, run: &QMCRunInfo, 
              world: &impl Communicator, timings: &QMCTimings) -> Option<(f64, Option<ExcitationHist>, QMCTimings)> {
    if !(it + 1).is_multiple_of(10) {
        return None;
    }

    let mut stop = 0;
    if run.irank == 0 && Path::new("STOP").exists() {
        stop = 1;
    }
    world.process_at_rank(0).broadcast_into(&mut stop);

    if stop == 0 {
        return None;
    }

    let rs = RestartState {
        iter: it,
        es: shifts.es,
        es_s: shifts.es_s,
        nwprevc: state.prev_pop.nwc,
        nrefprevc: state.prev_pop.nrefc,
        nwprevsc: state.prev_pop.nwsc,
        nrefprevsc: state.prev_pop.nrefsc,
        walkers: std::mem::replace(&mut state.mc.walkers, Walkers::new(run.ndets)),
        pg: std::mem::take(&mut state.mc.pg),
        excitation_hist: state.mc.excitation_hist.take(),
        base_seed: Some(run.base_seed),
    };

    write_restart_hdf5("RESTART.H5", world, &rs, run.start, run.end, run.ndets).unwrap();

    if run.irank == 0 {
        let _ = std::fs::remove_file("STOP");
        println!("STOP detected, Wrote RESTART.H5 and exiting");
    }

    Some((state.eprojcur, rs.excitation_hist, timings.clone()))
}

/// Propagate according to the stochastic update equations for `max_steps` iterations.
/// This routine initialises the NOCI-QMC state, performs the parallel spawning,
/// death/cloning, annihilation, population, shift, and projected-energy updates, and
/// optionally writes a restart file if a `STOP` file is detected.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: Initial determinant coefficient vector to be translated into walker populations.
/// - `es`: Non-overlap transformed shift energy.
/// - `ref_indices`: Indices of the reference determinants in the full basis.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `(f64, Option<ExcitationHist>, QMCTimings)`: Final projected energy estimate,
///   optional excitation histogram, and timing breakdown for the stochastic propagation.
pub fn qmc_step(data: &QMCData<'_>, c0: &[f64], es: &mut f64, ref_indices: &[usize], world: &impl Communicator) -> (f64, Option<ExcitationHist>, QMCTimings) {
    let qmc = data.input.qmc.as_ref().unwrap();

    // Set up data for stochastic run.
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let ndets = data.basis.len();
    let start = (ndets * irank) / nranks;
    let end = (ndets * (irank + 1)) / nranks;
    // Determine which of the determinant indices are reference states.
    let mut isref = vec![false; ndets];
    for &i in ref_indices {
        isref[i] = true;
    }

    let base_seed = qmc.seed.unwrap_or_else(rand::random);
    let rank_seed = base_seed.wrapping_add((irank as u64).wrapping_mul(0x9E3779B9));

    let run = QMCRunInfo {irank, nranks, ndets, start, end, iref: 0, base_seed, rank_seed};

    let scratchsize = {
        let (maxsame, maxla, maxlb) = max_scratch_sizes(data.basis);
        ScratchSize {maxsame, maxla, maxlb}
    };

    let mut timings = QMCTimings::default();
    let mut scratch = WickScratchSpin::new();

    let t0 = Instant::now();
    let mut state = initialise_qmc_state(c0, es, data, &run, &isref, world, &mut scratch);
    timings.initialise_walkers += t0.elapsed().as_secs_f64();

    println!("Size of Wick's Scratch (MiB): {}", std::mem::size_of::<WickScratchSpin>() as f64 / (1024.0 * 1024.0));
    type ThreadState = (Vec<(usize, i64)>, Vec<PopulationUpdate>, Vec<f64>, SmallRng, WickScratchSpin);
    println!("Size of per thread state (MiB): {}", std::mem::size_of::<ThreadState>() as f64 / (1024.0 * 1024.0));

    let mut send: Vec<Vec<PopulationUpdate>> = (0..nranks).map(|_| Vec::new()).collect();

    print_header(irank);
    print_initial_row(irank, &state, data.basis[0].e);

    for it in state.start_iter..data.input.prop_ref().max_steps {
        let shifts = Shifts {es: *es, es_s: state.es_s};
        let prop = propagate_iteration(it, &state.mc, data, &run, &scratchsize, shifts, &mut timings);

        let changedglobal = accumulate_updates(&mut state.mc, &mut send, prop, data.input, world, nranks, &mut timings);
        if changedglobal == 0 {
            print_cached_row(irank, it + 1, &state, data.basis[0].e, *es);
            continue;
        }
        
        let t0 = Instant::now();
        let d = apply_delta(&mut state.mc);
        timings.apply_delta += t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        update_p(&mut state.mc.pg, &d, data, &run, world);
        timings.update_p += t0.elapsed().as_secs_f64();

        let stats = compute_populations(&state.mc, &isref, &run, world, &mut timings);
        cache_population_stats(&mut state, &stats);
        update_shifts(it, &stats, &mut state, es, data.input);
        
        let t0 = Instant::now();
        update_projected_energy(&d, &mut state.pe, data, world);
        state.eprojcur = state.pe.num / state.pe.den;
        timings.eproj += t0.elapsed().as_secs_f64();
        
        let stopshifts = Shifts {es: *es, es_s: state.es_s};
        if let Some(ret) = check_stop(it, &mut state, stopshifts, &run, world, &timings) {
            return ret;
        }

        print_row(irank, it + 1, &state, &stats, data.basis[0].e, *es);
    }

    (state.eprojcur, state.mc.excitation_hist, timings)
}
