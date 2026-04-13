// stochastic/state.rs 
use rand::rngs::SmallRng;
use rand::Rng;
use mpi::traits::*;
use rand_distr::{Binomial, Distribution};

use crate::{AoData, SCFState};
use crate::nonorthogonalwicks::{WicksView, WickScratchSpin};
use crate::noci::MOCache;
use crate::input::{Input, Propagator, ExcitationGen};

use crate::mpiutils::owner;
use super::propagate::{find_hs};
use super::excit::{pgen_uniform, pgen_heat_bath, init_heat_bath};

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
pub(crate) struct QMCRunInfo {
    pub irank: usize,
    pub nranks: usize,
    pub ndets: usize,
    pub start: usize,
    pub end: usize,
    pub iref: usize,
    pub base_seed: u64,
    pub rank_seed: u64,
}

// Storage for maximum size required for Wick's scratch.
pub(crate) struct ScratchSize {
    pub(crate) maxsame: usize,
    pub(crate) maxla: usize,
    pub(crate) maxlb: usize,
}

// Storage for Current shift values used during propagation.
#[derive(Clone, Copy)]
pub(crate) struct Shifts {
    pub es: f64,
    pub es_s: f64,
}

// Storage for walker information.
pub(crate) struct Walkers {
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
    pub(crate) fn new(n: usize) -> Self {
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
    pub(crate) fn get(&self, i: usize) -> i64 {
        self.pop[i]
    }
    
    /// Return a list of occupied determinant indices.
    /// # Arguments:
    /// - `self`: Object containing information about current walkers.
    /// # Returns
    /// - `&[usize]`: Slice of occupied determinant indices.
    pub(crate) fn occ(&self) -> &[usize] {
        &self.occ
    }

    /// Add dn (change in population) to determinant i, modifying pop, occ and pos as required
    /// # Arguments:
    /// - `self`: Object containing information about current walkers.
    /// - `i`: Determinant index of choice. 
    /// - `dn`: Change in population of determinant i.
    /// # Returns
    /// - `()`: Updates the walker storage in place.
    pub(crate) fn add(&mut self, i: usize, dn: i64) {
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
    pub(crate) fn norm(&self) -> i64 {
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
    // Histogrammed samples of P_{\text{Spawn}} for excit gen diagnostics.
    pub excitation_hist: Option<ExcitationHist> 
}

// Storage for the Incrementally updated projected-energy.
#[derive(Clone, Copy)]
pub(crate) struct ProjectedEnergyUpdate {
    pub(crate) iref: usize,
    pub(crate) num: f64,
    pub(crate) den: f64,
}

/// Storage for current walker populations in both the raw and overlap-transformed sense.
#[derive(Clone, Copy)]
pub(crate) struct PopulationStats {
    // Non-overlap transformed walker populations.
    pub(crate) nwc: i64,
    // Non-overlap transformed reference walker populations.
    pub(crate) nrefc: i64,
    // Overlap-transformed walker populations.
    pub(crate) nwsc: f64,
    // Overlap-transformed reference walker populations.
    pub(crate) nrefsc: f64,
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
    pub(crate) fn new(nwc: i64, nrefc: i64, nwsc: f64, nrefsc: f64) -> Self {
        Self {nwc, nrefc, nwsc, nrefsc}
    }
}

/// Storage for QMC step bookkeeping data.
pub(crate) struct PropagationState {
    // Full Monte Carlo state.
    pub(crate) mc: MCState,
    // Incrementally updated projected-energy.
    pub(crate) pe: ProjectedEnergyUpdate,
    // Overlap transformed shift.
    pub(crate) es_s: f64,
    // Walker populations at previous shift update.
    pub(crate) prev_pop: PopulationStats,
    // Current walker populations.
    pub(crate) cur_pop: PopulationStats,
    // From where did iterations begin (was a restart file used?).
    pub(crate) start_iter: usize,
    // Has the overlap-transformed population reached the target.
    pub(crate) reached_sc: bool,
    // Has the non-overlap-transformed population reached the target.
    pub(crate) reached_c: bool,
    // Current projected-energy.
    pub(crate) eprojcur: f64,
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
    pub(crate) fn new(mc: MCState, pe: ProjectedEnergyUpdate, es_s: f64, start_iter: usize, reached_sc: bool, reached_c: bool, prev_pop: PopulationStats) -> Self {
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
    pub(crate) fn fresh(mc: MCState, pe: ProjectedEnergyUpdate, es_s: f64, prev_pop: PopulationStats) -> Self {
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
    pub(crate) fn restart(mc: MCState, pe: ProjectedEnergyUpdate, es_s: f64, start_iter: usize, reached_sc: bool, reached_c: bool, prev_pop: PopulationStats) -> Self {
        Self::new(mc, pe, es_s, start_iter, reached_sc, reached_c, prev_pop)
    }
}

// Storage for results of a single propagation step.
pub(crate) struct PropagationResult {
    // Population updates for determinants owned by current rank.
    pub(crate) local: Vec<(usize, i64)>,
    // Population updates for determinants owned by another rank.
    pub(crate) remote: Vec<PopulationUpdate>,
    // Excitation generation samples.
    pub(crate) samples: Vec<f64>,
}

// Storage for per thread propagation quantities.
pub(crate) struct ThreadPropagation {
    // Population changes generated by this thread that belong to determinants owned by current MPI rank.
    pub(crate) local: Vec<(usize, i64)>,
    // Population changes generated by this thread that belong to determinants owned by another MPI rank.
    pub(crate) remote: Vec<PopulationUpdate>,
    // Excitation generation samples.
    pub(crate) samples: Vec<f64>,
    // Thread local RNG.
    pub(crate) rng: SmallRng,
    // Per thread scratch space for extended non-orthogonal Wick's theorem.
    pub(crate) scratch: WickScratchSpin,
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
    pub(crate) fn death_cloning(&mut self, gamma: usize, ngamma: i64, shifts: Shifts, data: &QMCData<'_>) {
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
    pub(crate) fn spawning(&mut self, gamma: usize, ngamma: i64, shifts: Shifts, data: &QMCData<'_>, run: &QMCRunInfo) {
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
pub(crate) struct PopulationUpdate {
    pub det: u64,
    pub dn: i64,
}

// Storage for heat-bath excitation related quantities 
pub struct HeatBath {
    pub(crate) sumlg: f64, 
    pub(crate) cumulatives: Vec<f64>, 
    pub(crate) lambdas: Vec<usize>, 
    pub(crate) ks: Vec<f64>,
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
    pub(crate) fn new(logmin: f64, logmax: f64, nbins: usize) -> Self {
        Self {logmin, logmax, noverflow_low: 0, noverflow_high: 0, counts: vec![0u64; nbins], nbins, ntotal: 0}
    }
    
    /// Add a computed spawning probability to the histogram.
    /// # Arguments: 
    /// `self`: ExcitationHist.
    /// - `pspawn`: Spawning probability as defined in population dynamics routines.
    /// # Returns
    /// - `()`: Updates the histogram in place.
    pub(crate) fn add(&mut self, pspawn: f64) {
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

