// stochastic/state.rs
use mpi::traits::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::input::ExcitationGen;
use crate::noci::NOCIData;
use crate::nonorthogonalwicks::WickScratchSpin;

use super::excit::{coupling, init_heat_bath, pgen_heat_bath, pgen_uniform};
use super::propagate::{find_hs, stochastic_population_cutoff};

/// Storage for QMC timings.
#[derive(Default, Clone)]
pub struct QMCTimings {
    /// Time spent constructing the initial population.
    pub initialise_populations: f64,
    /// Time spent constructing sparse unbiased samples.
    pub sample_populations: f64,
    /// Time spent generating the pre-overlap population change.
    pub generate_population_changes: f64,
    /// Time spent accumulating thread-local changes and packing changes.
    pub acc_pack_updates: f64,
    /// Time spent exchanging spawned population changes between MPI ranks.
    pub exchange_population_changes: f64,
    /// Time spent adding received population changes to the local accumulator.
    pub unpack_population_changes: f64,
    /// Time spent draining the change accumulator into a sparse list.
    pub drain_population_changes: f64,
    /// Time spent applying the overlap-transformed change N \leftarrow N + S\Delta.
    pub apply_overlap_changes: f64,
    /// Time spent computing persistent and sampled population statistics.
    pub compute_population_stats: f64,
    /// Time spent computing the projected-energy numerator and denominator.
    pub compute_projected_energy: f64,
}

/// Storage for rank-local data layouts and metadata.
pub(in crate::stochastic) struct QMCRunInfo {
    /// MPI rank of the current process.
    pub(in crate::stochastic) irank: usize,
    /// Total number of MPI ranks in the run.
    pub(in crate::stochastic) nranks: usize,
    /// Total number of determinants in the stochastic basis.
    pub(in crate::stochastic) ndets: usize,
    /// Global determinant indices owned by this rank.
    pub(in crate::stochastic) owned: Vec<usize>,
    /// User- or randomly-selected base seed for the full run.
    pub(in crate::stochastic) base_seed: u64,
    /// Rank-specific seed derived from the base seed.
    pub(in crate::stochastic) rank_seed: u64,
    /// Reference-row Hamiltonian and overlap elements aligned with `owned`.
    pub(in crate::stochastic) reference_hs: Vec<(f64, f64)>,
    /// Cached diagonal Hamiltonian and overlap matrix elements for each determinant.
    pub(in crate::stochastic) diagonal_hs: Vec<(f64, f64)>,
}

/// Storage for maximum size required for Wick's scratch.
pub(in crate::stochastic) struct ScratchSize {
    /// Largest same-spin scratch dimension needed across the basis.
    pub(in crate::stochastic) maxsame: usize,
    /// Largest alpha-spin excitation scratch dimension needed across the basis.
    pub(in crate::stochastic) maxla: usize,
    /// Largest beta-spin excitation scratch dimension needed across the basis.
    pub(in crate::stochastic) maxlb: usize,
}

/// Storage for a sparse real population vector.
pub(crate) struct SparsePopulations {
    /// Signed real population vector over the full determinant space.
    pop: Vec<f64>,
    /// Determinant indices with nonzero population.
    occ: Vec<usize>,
    /// Position of each determinant in `occ`, or `usize::MAX` if unoccupied.
    pos: Vec<usize>,
}

impl SparsePopulations {
    /// Construct empty sparse population storage.
    /// # Arguments:
    /// - `n`: Number of determinants.
    /// # Returns:
    /// - `SparsePopulations`: Empty population storage.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            pop: vec![0.0; n],
            occ: Vec::new(),
            pos: vec![usize::MAX; n],
        }
    }

    /// Return the population on determinant `i`.
    /// # Arguments:
    /// - `self`: Sparse population storage.
    /// - `i`: Determinant index.
    /// # Returns:
    /// - `f64`: Signed real population.
    #[inline(always)]
    pub(crate) fn get(
        &self,
        i: usize,
    ) -> f64 {
        self.pop[i]
    }

    /// Return the occupied determinant indices.
    /// # Arguments:
    /// - `self`: Sparse population storage.
    /// # Returns:
    /// - `&[usize]`: Occupied determinant indices.
    #[inline(always)]
    pub(crate) fn occ(&self) -> &[usize] {
        &self.occ
    }

    /// Add a real population change to determinant `i`.
    /// # Arguments:
    /// - `self`: Sparse population storage.
    /// - `i`: Determinant index.
    /// - `dn`: Signed real population change.
    /// # Returns:
    /// - `()`: Updates the sparse population storage.
    pub(crate) fn add(
        &mut self,
        i: usize,
        dn: f64,
    ) {
        if dn == 0.0 {
            return;
        }

        unsafe {
            let pop = self.pop.get_unchecked_mut(i);
            let old = *pop;
            let new = old + dn;
            *pop = new;

            if old == 0.0 && new != 0.0 {
                let p = self.occ.len();
                *self.pos.get_unchecked_mut(i) = p;
                self.occ.push(i);
                return;
            }

            if old != 0.0 && new == 0.0 {
                let p = *self.pos.get_unchecked(i);
                let last = self.occ.pop().unwrap_unchecked();

                if last != i {
                    *self.occ.get_unchecked_mut(p) = last;
                    *self.pos.get_unchecked_mut(last) = p;
                }

                *self.pos.get_unchecked_mut(i) = usize::MAX;
            }
        }
    }

    /// Remove all populations while retaining allocated storage.
    /// # Arguments:
    /// - `self`: Sparse population storage.
    /// # Returns:
    /// - `()`: Clears every occupied population.
    pub(crate) fn clear(&mut self) {
        for i in self.occ.drain(..) {
            self.pop[i] = 0.0;
            self.pos[i] = usize::MAX;
        }
    }

    /// Compute the population 1-norm.
    /// # Arguments:
    /// - `self`: Sparse population storage.
    /// # Returns:
    /// - `f64`: Population 1-norm.
    pub(crate) fn norm(&self) -> f64 {
        self.occ.iter().map(|&i| self.pop[i].abs()).sum()
    }
}

/// Given a determinant index return which MPI rank owns it.
/// # Arguments
/// - `det`: Determinant index.
/// - `ndets`: Number of determinants (unused, kept for interface compatibility).
/// - `nranks`: Number of MPI ranks.
/// # Returns
/// - `usize`: MPI rank that owns the determinant.
#[inline(always)]
pub fn owner(
    det: usize,
    _ndets: usize,
    nranks: usize,
) -> usize {
    let mut x = det as u64;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    (x as usize) % nranks
}

/// Build the list of determinants owned by this rank.
/// # Arguments
/// - `irank`: Current MPI rank.
/// - `ndets`: Number of determinants.
/// - `nranks`: Number of MPI ranks.
/// # Returns
/// - `Vec<usize>`: Global determinant indices owned by this rank.
pub fn owned(
    irank: usize,
    ndets: usize,
    nranks: usize,
) -> Vec<usize> {
    let mut owned = Vec::new();

    for det in 0..ndets {
        if owner(det, ndets, nranks) == irank {
            owned.push(det);
        }
    }
    owned
}

/// Storage for the range-preserving Monte Carlo state.
pub(in crate::stochastic) struct MCState {
    /// Persistent local portion of the real population vector.
    pub(in crate::stochastic) populations: Vec<f64>,
    /// Temporary sparse real population vector used for spawning.
    pub(in crate::stochastic) sampled: SparsePopulations,
    /// Accumulated real population changes over the full determinant space.
    pub(in crate::stochastic) delta: Vec<f64>,
    /// Determinants for which `delta[i]` is nonzero.
    pub(in crate::stochastic) changed: Vec<usize>,
    /// Histogrammed spawning magnitudes.
    pub(in crate::stochastic) excitation_hist: Option<ExcitationHist>,
}

/// Storage for the current projected-energy numerator and denominator.
#[derive(Clone, Copy)]
pub(in crate::stochastic) struct ProjectedEnergyUpdate {
    /// Numerator of the projected-energy estimator.
    pub(in crate::stochastic) num: f64,
    /// Denominator of the projected-energy estimator.
    pub(in crate::stochastic) den: f64,
}

/// Storage for current persistent and sampled population statistics.
#[derive(Clone, Copy)]
pub(in crate::stochastic) struct PopulationStats {
    /// Persistent population 1-norm |N|_1.
    pub(in crate::stochastic) nw: f64,
    /// Persistent population 1-norm on the reference determinants.
    pub(in crate::stochastic) nref: f64,
    /// Sampled-population 1-norm |\tilde N|_1.
    pub(in crate::stochastic) nsampled: f64,
    /// Number of nonzero sampled populations | \tilde N|_0.
    pub(in crate::stochastic) nsampledo: i64,
}

impl PopulationStats {
    /// Construct population statistics.
    /// # Arguments:
    /// - `nw`: Persistent population 1-norm.
    /// - `nref`: Persistent reference population 1-norm.
    /// - `nsampled`: Sampled-population population 1-norm.
    /// - `nsampledo`: Number of sampled-population determinants.
    /// # Returns:
    /// - `PopulationStats`: Population statistics.
    pub(in crate::stochastic) fn new(
        nw: f64,
        nref: f64,
        nsampled: f64,
        nsampledo: i64,
    ) -> Self {
        Self {
            nw,
            nref,
            nsampled,
            nsampledo,
        }
    }
}

/// Storage for QMC propagation bookkeeping.
pub(in crate::stochastic) struct PropagationState {
    /// Full Monte Carlo state.
    pub(in crate::stochastic) mc: MCState,
    /// Current projected-energy numerator and denominator.
    pub(in crate::stochastic) pe: ProjectedEnergyUpdate,
    /// Population statistics at the previous shift update.
    pub(in crate::stochastic) prev_pop: PopulationStats,
    /// Current population statistics.
    pub(in crate::stochastic) cur_pop: PopulationStats,
    /// Report from which propagation begins.
    pub(in crate::stochastic) start_report: usize,
    /// Whether the persistent population has reached its target.
    pub(in crate::stochastic) reached: bool,
    /// Current projected energy.
    pub(in crate::stochastic) eprojcur: f64,
}

impl PropagationState {
    /// Construct a propagation state.
    /// # Arguments:
    /// - `mc`: Monte Carlo state.
    /// - `pe`: Projected-energy data.
    /// - `start_report`: Report from which propagation begins.
    /// - `reached`: Whether the target population has been reached.
    /// - `prev_pop`: Population statistics at the previous shift update.
    /// # Returns:
    /// - `PropagationState`: Initialised propagation state.
    pub(in crate::stochastic) fn new(
        mc: MCState,
        pe: ProjectedEnergyUpdate,
        start_report: usize,
        reached: bool,
        prev_pop: PopulationStats,
    ) -> Self {
        let eprojcur = pe.num / pe.den;

        Self {
            mc,
            pe,
            prev_pop,
            cur_pop: prev_pop,
            start_report,
            reached,
            eprojcur,
        }
    }
}

/// Storage for results of a single propagation step.
pub(in crate::stochastic) struct PropagationResult {
    /// Population updates for determinants owned by current rank.
    pub(in crate::stochastic) local: Vec<(usize, f64)>,
    /// Population updates for determinants owned by another rank.
    pub(in crate::stochastic) remote: Vec<PopulationUpdate>,
    /// Excitation generation samples.
    pub(in crate::stochastic) samples: Vec<f64>,
}

impl PropagationResult {
    /// Construct empty reusable propagation result storage.
    /// # Arguments:
    /// # Returns
    /// - `PropagationResult`: Empty propagation result storage.
    pub(in crate::stochastic) fn new() -> Self {
        Self {
            local: Vec::new(),
            remote: Vec::new(),
            samples: Vec::new(),
        }
    }

    /// Clear propagation results while retaining allocated storage.
    /// # Arguments:
    /// - `self`: Propagation result storage to clear.
    /// # Returns
    /// - `()`: Clears all result vectors without releasing their allocations.
    pub(in crate::stochastic) fn clear(&mut self) {
        self.local.clear();
        self.remote.clear();
        self.samples.clear();
    }
}

/// Storage for per thread propagation quantities.
pub(in crate::stochastic) struct ThreadPropagation {
    /// Population changes generated by this thread that belong to determinants owned by current MPI rank.
    pub(in crate::stochastic) local: Vec<(usize, f64)>,
    /// Population changes generated by this thread that belong to determinants owned by another MPI rank.
    pub(in crate::stochastic) remote: Vec<PopulationUpdate>,
    /// Excitation generation samples.
    pub(in crate::stochastic) samples: Vec<f64>,
    /// Thread local RNG.
    pub(in crate::stochastic) rng: SmallRng,
    /// Per thread scratch space for extended non-orthogonal Wick's theorem.
    pub(in crate::stochastic) wick_scratch: Box<WickScratchSpin<f64>>,
}

impl ThreadPropagation {
    /// Construct reusable per-thread propagation storage.
    /// # Arguments:
    /// - `seed`: Initial random-number generator seed.
    /// - `maxsame`: Maximum same-spin scratch dimension.
    /// - `maxla`: Maximum alpha-spin different-spin scratch dimension.
    /// - `maxlb`: Maximum beta-spin different-spin scratch dimension.
    /// # Returns
    /// - `ThreadPropagation`: Initialised per-thread propagation storage.
    pub(in crate::stochastic) fn with_sizes(
        seed: u64,
        maxsame: usize,
        maxla: usize,
        maxlb: usize,
    ) -> Self {
        Self {
            local: Vec::new(),
            remote: Vec::new(),
            samples: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
            wick_scratch: Box::new(WickScratchSpin::with_sizes(maxsame, maxla, maxlb)),
        }
    }

    /// Clear generated updates while retaining allocated storage and Wick scratch space.
    /// # Arguments:
    /// - `self`: Per-thread propagation storage to clear.
    /// # Returns
    /// - `()`: Clears generated updates without releasing their allocations.
    pub(in crate::stochastic) fn clear(&mut self) {
        self.local.clear();
        self.remote.clear();
        self.samples.clear();
    }

    /// Generate the diagonal real population change for one sampled determinant.
    /// # Arguments:
    /// - `gamma`: Parent determinant index.
    /// - `population`: Real sampled population on `gamma`.
    /// - `shift`: Current population-control shift.
    /// - `data`: Immutable stochastic propagation data.
    /// - `diagonal_hs`: Cached diagonal Hamiltonian and overlap elements.
    /// # Returns:
    /// - `()`: Appends the diagonal population change to `self.local`.
    pub(in crate::stochastic) fn diagonal_population_change(
        &mut self,
        gamma: usize,
        population: f64,
        shift: f64,
        data: &NOCIData<'_, f64>,
        diagonal_hs: &[(f64, f64)],
    ) {
        let (hgg, sgg) = diagonal_hs[gamma];
        let coupling = hgg - shift * sgg;
        let dn = -data.input.prop_ref().dt * coupling * population;

        if dn != 0.0 {
            self.local.push((gamma, dn));
        }
    }

    /// Generate off-diagonal real population changes from one sampled determinant.
    /// # Arguments:
    /// - `gamma`: Parent determinant index.
    /// - `population`: Real sampled population on `gamma`.
    /// - `shift`: Current population-control shift.
    /// - `data`: Immutable stochastic propagation data.
    /// - `run`: Rank-local propagation metadata.
    /// # Returns:
    /// - `()`: Appends local and remote real population changes.
    pub(in crate::stochastic) fn spawning(
        &mut self,
        gamma: usize,
        population: f64,
        shift: f64,
        data: &NOCIData<'_, f64>,
        run: &QMCRunInfo,
    ) {
        if population == 0.0 {
            return;
        }

        let qmc = data.input.qmc.as_ref().unwrap();
        let dt = data.input.prop_ref().dt;
        let write_excitation_hist = data.input.write.write_excitation_hist;

        let nattempts = population.abs().ceil().max(1.0) as usize;
        let parent_population = population / nattempts as f64;

        if let ExcitationGen::Uniform = qmc.excitation_gen
            && data.wicks.is_some()
        {
            let ndets = data.basis.len();
            let pgen = 1.0 / (ndets - 1) as f64;

            let sample_uniform = |rng: &mut SmallRng| -> usize {
                let mut lambda = rng.gen_range(0..ndets - 1);
                if lambda >= gamma {
                    lambda += 1;
                }
                lambda
            };

            let mut lambda = sample_uniform(&mut self.rng);

            for iattempt in 0..nattempts {
                let next = if iattempt + 1 < nattempts {
                    let next = sample_uniform(&mut self.rng);

                    Some(next)
                } else {
                    None
                };

                let (hlg, slg) = find_hs(data, lambda, gamma, self.wick_scratch.as_mut());

                let k = coupling(hlg, slg, shift);
                let raw = -dt * k * parent_population / pgen;

                if write_excitation_hist {
                    self.samples.push(raw.abs());
                }

                let dn = stochastic_population_cutoff(raw, qmc.spawn_cutoff, &mut self.rng);

                if dn != 0.0 {
                    if run.nranks == 1 {
                        self.local.push((lambda, dn));
                    } else {
                        let destination = owner(lambda, run.ndets, run.nranks);

                        if destination == run.irank {
                            self.local.push((lambda, dn));
                        } else {
                            self.remote.push(PopulationUpdate {
                                det: lambda as u64,
                                dn,
                            });
                        }
                    }
                }

                if let Some(next) = next {
                    lambda = next;
                }
            }

            return;
        }

        let heat_bath = if let ExcitationGen::HeatBath = qmc.excitation_gen {
            Some(init_heat_bath(
                gamma,
                shift,
                data,
                self.wick_scratch.as_mut(),
            ))
        } else {
            None
        };

        for _ in 0..nattempts {
            let (pgen, k, lambda) = match qmc.excitation_gen {
                ExcitationGen::Uniform => pgen_uniform(
                    gamma,
                    shift,
                    data,
                    &mut self.rng,
                    self.wick_scratch.as_mut(),
                ),
                ExcitationGen::HeatBath => pgen_heat_bath(
                    gamma,
                    shift,
                    data,
                    &mut self.rng,
                    heat_bath.as_ref().unwrap(),
                    self.wick_scratch.as_mut(),
                ),
                ExcitationGen::ApproximateHeatBath => {
                    unimplemented!()
                }
            };

            let raw = -dt * k * parent_population / pgen;

            if write_excitation_hist {
                self.samples.push(raw.abs());
            }

            let dn = stochastic_population_cutoff(raw, qmc.spawn_cutoff, &mut self.rng);

            if dn == 0.0 {
                continue;
            }

            if run.nranks == 1 {
                self.local.push((lambda, dn));
            } else {
                let destination = owner(lambda, run.ndets, run.nranks);

                if destination == run.irank {
                    self.local.push((lambda, dn));
                } else {
                    self.remote.push(PopulationUpdate {
                        det: lambda as u64,
                        dn,
                    });
                }
            }
        }
    }
}

/// Storage for a sparse real population change communicated across MPI ranks.
#[repr(C)]
#[derive(Copy, Clone, Equivalence)]
pub(crate) struct PopulationUpdate {
    /// Determinant index to which the population change applies.
    pub det: u64,
    /// Signed real population change.
    pub dn: f64,
}

/// Storage for heat-bath excitation related quantities.
pub(in crate::stochastic) struct HeatBath {
    /// Total absolute coupling weight over all allowed children.
    pub(in crate::stochastic) sumlg: f64,
    /// Cumulative absolute coupling weights used for inverse-CDF sampling.
    pub(in crate::stochastic) cumulatives: Vec<f64>,
    /// Child determinant indices corresponding to the cumulative weights.
    pub(in crate::stochastic) lambdas: Vec<usize>,
    /// Signed couplings associated with each child determinant.
    pub(in crate::stochastic) ks: Vec<f64>,
}

/// Storage for histogrammed data.
#[derive(Clone)]
pub struct ExcitationHist {
    /// Lower logarithmic bound of the histogram range.
    pub logmin: f64,
    /// Upper logarithmic bound of the histogram range.
    pub logmax: f64,
    /// Number of samples that fell below the histogram range.
    pub noverflow_low: u64,
    /// Number of samples that fell above the histogram range.
    pub noverflow_high: u64,
    /// Bin counts over the configured logarithmic range.
    pub counts: Vec<u64>,
    /// Total number of bins in the histogram.
    pub nbins: usize,
    /// Total number of samples processed by the histogram.
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
    pub(in crate::stochastic) fn new(
        logmin: f64,
        logmax: f64,
        nbins: usize,
    ) -> Self {
        Self {
            logmin,
            logmax,
            noverflow_low: 0,
            noverflow_high: 0,
            counts: vec![0u64; nbins],
            nbins,
            ntotal: 0,
        }
    }

    /// Add the absolute magnitude of an unrounded spawned population change.
    /// # Arguments:
    /// `self`: ExcitationHist.
    /// - `pspawn`: Spawning probability as defined in population dynamics routines.
    /// # Returns
    /// - `()`: Updates the histogram in place.
    pub(in crate::stochastic) fn add(
        &mut self,
        pspawn: f64,
    ) {
        self.ntotal += 1;

        if !pspawn.is_finite() || pspawn <= 0.0 {
            self.noverflow_low += 1;
            return;
        }

        let logpspawn = pspawn.ln();

        if logpspawn < self.logmin {
            self.noverflow_low += 1;
            return;
        }
        if logpspawn >= self.logmax {
            self.noverflow_high += 1;
            return;
        }

        // Fractional position of logpspawn in histogram range.
        let t = (logpspawn - self.logmin) / (self.logmax - self.logmin);
        // Convert to bin units.
        let b = (t * self.nbins as f64) as usize;
        self.counts[b] += 1;
    }
}

/// Reusable MPI scratch for walker-update collectives.
#[derive(Default)]
pub(crate) struct MPIScratch {
    /// Number of sparse updates contributed by each rank for all-gather.
    pub(crate) gather_counts: Vec<i32>,
    /// Displacements for gathered sparse updates.
    pub(crate) gather_displs: Vec<i32>,
    /// Reusable receive buffer for gathered sparse updates.
    pub(crate) gather_recv: Vec<PopulationUpdate>,
    /// Number of spawn updates sent to each rank.
    pub(crate) send_counts: Vec<i32>,
    /// Displacements into the contiguous spawn-send buffer for each rank.
    pub(crate) send_displacements: Vec<i32>,
    /// Number of spawn updates received from each rank.
    pub(crate) recv_counts: Vec<i32>,
    /// Displacements into the contiguous spawn-receive buffer for each rank.
    pub(crate) recv_displacements: Vec<i32>,
    /// Reusable contiguous send buffer for spawn exchange.
    pub(crate) send_contig: Vec<PopulationUpdate>,
    /// Reusable contiguous receive buffer for spawn exchange.
    pub(crate) recv_contig: Vec<PopulationUpdate>,
}

impl MPIScratch {
    /// Construct reusable MPI scratch buffers.
    /// # Arguments:
    /// - `nranks`: Number of MPI ranks.
    /// # Returns
    /// - `MPIScratch`: Scratch storage sized for the communicator.
    pub(in crate::stochastic) fn new(nranks: usize) -> Self {
        Self {
            gather_counts: vec![0; nranks],
            gather_displs: vec![0; nranks],
            gather_recv: Vec::new(),
            send_counts: vec![0; nranks],
            send_displacements: vec![0; nranks],
            recv_counts: vec![0; nranks],
            recv_displacements: vec![0; nranks],
            send_contig: Vec::new(),
            recv_contig: Vec::new(),
        }
    }
}
