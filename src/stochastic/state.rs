// stochastic/state.rs
// External crate imports.
use mpi::traits::*;
use rand::{Rng, SeedableRng};

// Crate-root imports.
use crate::input::{ExcitationGen, Propagator};
use crate::noci::{
    AuxiliaryIndex, AuxiliarySpace, NOCIData, NOCIIndex, OrthogonalHamiltonianScratch,
    OverlapFactors,
};
use crate::nonorthogonalwicks::WickScratchSpin;

// Parent/sibling imports.
use super::common::{find_h_orthogonal_batched, find_hs_batched};
use super::excit::{OrthogonalUniformGenerator, init_heat_bath, pgen_heat_bath};
use super::fri::{FriAmplitude, compress_sparse};
use super::overlapweighted::{OverlapProposal, OverlapWeightedGenerator};

/// Stable RNG used by seeded QMC streams.
pub(crate) type QmcRng = rand_xoshiro::Xoshiro256PlusPlus;

/// Population representation persisted by a stochastic propagator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::stochastic) enum PopulationRepresentation {
    /// Population stores signed coefficient coordinates.
    Coefficient,
    /// Population stores signed coordinates in `range(S)`.
    Range,
}

impl PopulationRepresentation {
    /// Return restart metadata spelling for this representation.
    /// # Arguments:
    /// - `self`: Population representation.
    /// # Returns:
    /// - `&'static str`: Stable representation name stored in HDF5 metadata.
    pub(in crate::stochastic) fn as_str(self) -> &'static str {
        match self {
            Self::Coefficient => "coefficient",
            Self::Range => "range",
        }
    }
}

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
    /// `Time spent applying the overlap-transformed change N \leftarrow N + S\Delta.`
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
    /// Deterministic compatibility hash of the ordered stochastic determinant basis.
    pub(in crate::stochastic) basis_hash: [u64; 2],
    /// MPI owner rank for each global determinant.
    pub(in crate::stochastic) det_owner: Vec<usize>,
    /// Global determinant indices owned by this rank.
    pub(in crate::stochastic) owned: Vec<usize>,
    /// User- or randomly-selected base seed for the full run.
    pub(in crate::stochastic) base_seed: u64,
    /// Rank-specific seed derived from the base seed.
    pub(in crate::stochastic) rank_seed: u64,
    /// Population representation used by this run.
    pub(in crate::stochastic) representation: PopulationRepresentation,
    /// Target population used by this run.
    pub(in crate::stochastic) target_population: f64,
    /// Projected-energy Hamiltonian and overlap contractions aligned with `owned`.
    pub(in crate::stochastic) projection_hs: Vec<(f64, f64)>,
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

/// Storage for the shifts used in one stochastic propagation cycle.
#[derive(Clone, Copy)]
pub(in crate::stochastic) struct ShiftSpec {
    /// Current non-overlap-transformed shift.
    pub(in crate::stochastic) es: f64,
    /// Current overlap-transformed shift.
    pub(in crate::stochastic) es_s: f64,
    /// Propagator approximation.
    pub(in crate::stochastic) propagator: Propagator,
}

impl ShiftSpec {
    /// Construct the S-apply shift specification.
    /// # Arguments:
    /// - `es`: Population-control shift `E_s`.
    /// # Returns:
    /// - `ShiftSpec`: Shift specification for `H - E_s S`.
    pub(in crate::stochastic) fn s_apply(es: f64) -> Self {
        Self {
            es,
            es_s: es,
            propagator: Propagator::SApply,
        }
    }

    /// Return the shift multiplying the overlap matrix.
    /// # Arguments:
    /// - `self`: Shift specification.
    /// # Returns:
    /// - `f64`: Shift `E_s^S` in `H - E_s^S S`.
    pub(in crate::stochastic) fn overlap_shift(&self) -> f64 {
        match self.propagator {
            Propagator::Unshifted => self.es_s,
            Propagator::Shifted => self.es_s,
            Propagator::DoublyShifted => self.es_s,
            Propagator::DifferenceDoublyShiftedU1 => 0.5 * (self.es + self.es_s),
            Propagator::DifferenceDoublyShiftedU2 => self.es_s,
            Propagator::SApply | Propagator::BApply => self.es,
        }
    }

    /// Return the shift multiplying the identity.
    /// # Arguments:
    /// - `self`: Shift specification.
    /// # Returns:
    /// - `f64`: Shift `E_s^I` in `H - E_s^S S - E_s^I I`.
    pub(in crate::stochastic) fn identity_shift(&self) -> f64 {
        match self.propagator {
            Propagator::Unshifted => 0.0,
            Propagator::Shifted => self.es_s,
            Propagator::DoublyShifted => self.es,
            Propagator::DifferenceDoublyShiftedU1 => self.es - self.es_s,
            Propagator::DifferenceDoublyShiftedU2 => self.es - self.es_s,
            Propagator::SApply | Propagator::BApply => 0.0,
        }
    }

    /// Evaluate the off-diagonal shifted coupling.
    /// # Arguments:
    /// - `h`: Hamiltonian matrix element `H_{xw}`.
    /// - `s`: Overlap matrix element `S_{xw}`.
    /// # Returns:
    /// - `f64`: `H_{xw} - E_s^S S_{xw}`.
    pub(in crate::stochastic) fn coupling(
        &self,
        h: f64,
        s: f64,
    ) -> f64 {
        h - self.overlap_shift() * s
    }

    /// Evaluate the diagonal residual.
    /// # Arguments:
    /// - `h`: Hamiltonian matrix element `H_{ww}`.
    /// - `s`: Overlap matrix element `S_{ww}`.
    /// # Returns:
    /// - `f64`: `H_{ww} - E_s^S S_{ww} - E_s^I`.
    pub(in crate::stochastic) fn diagonal_residual(
        &self,
        h: f64,
        s: f64,
    ) -> f64 {
        h - self.overlap_shift() * s - self.identity_shift()
    }
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

    /// Insert a non-zero population into an empty slot after `clear`.
    /// # Arguments:
    /// - `self`: Sparse population storage.
    /// - `i`: Determinant index currently absent from `occ`.
    /// - `population`: Non-zero signed real population.
    /// # Returns:
    /// - `()`: Adds the determinant to `occ` and writes its population.
    pub(crate) fn insert_nonzero(
        &mut self,
        i: usize,
        population: f64,
    ) {
        let p = self.occ.len();
        self.pop[i] = population;
        self.pos[i] = p;
        self.occ.push(i);
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
/// - `nranks`: Number of MPI ranks.
/// # Returns
/// - `usize`: MPI rank that owns the determinant.
#[inline(always)]
pub(crate) fn owner(
    det: usize,
    nranks: usize,
) -> usize {
    (mix_owner_key(det as u64) as usize) % nranks
}

/// Map one orthogonal determinant to a deterministic MPI owner.
/// # Arguments:
/// - `det`: Deterministic flattened auxiliary determinant identity.
/// - `nranks`: Number of MPI ranks.
/// # Returns:
/// - `usize`: MPI rank owning the complete coalesced orthogonal amplitude.
pub(in crate::stochastic) fn auxiliary_owner(
    det: AuxiliaryIndex,
    nranks: usize,
) -> usize {
    (mix_owner_key(det.0 as u64) as usize) % nranks
}

/// Apply the shared deterministic avalanche mix used by stochastic ownership maps.
/// # Arguments:
/// - `x`: Unsigned key word.
/// # Returns:
/// - `u64`: Mixed key word.
#[inline(always)]
fn mix_owner_key(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
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
    /// `Persistent population 1-norm |N|_1.`
    pub(in crate::stochastic) nw: f64,
    /// Persistent population 1-norm on the reference determinants.
    pub(in crate::stochastic) nref: f64,
    /// `Sampled-population 1-norm |\tilde N|_1.`
    pub(in crate::stochastic) nsampled: f64,
    /// `Number of nonzero sampled populations | \tilde N|_0.`
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
    /// Optional report-level heavy-ball velocity `V_r` in the rank-local population layout.
    pub(in crate::stochastic) momentum: Option<Vec<f64>>,
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
    /// Current overlap-weighted mixture probability `p`.
    pub(in crate::stochastic) overlap_weight: f64,
}

impl PropagationState {
    /// Construct a propagation state.
    /// # Arguments:
    /// - `mc`: Monte Carlo state.
    /// - `momentum`: Optional report-level heavy-ball velocity `V_r` in the rank-local population
    ///   layout.
    /// - `pe`: Projected-energy data.
    /// - `start_report`: Report from which propagation begins.
    /// - `reached`: Whether the target population has been reached.
    /// - `prev_pop`: Population statistics at the previous shift update.
    /// - `overlap_weight`: Current overlap-weighted mixture probability `p`.
    /// # Returns:
    /// - `PropagationState`: Initialised propagation state.
    pub(in crate::stochastic) fn new(
        mc: MCState,
        momentum: Option<Vec<f64>>,
        pe: ProjectedEnergyUpdate,
        start_report: usize,
        reached: bool,
        prev_pop: PopulationStats,
        overlap_weight: f64,
    ) -> Self {
        let eprojcur = pe.num / pe.den;

        Self {
            mc,
            momentum,
            pe,
            prev_pop,
            cur_pop: prev_pop,
            start_report,
            reached,
            eprojcur,
            overlap_weight,
        }
    }
}

/// Storage for results of a single propagation step.
pub(in crate::stochastic) struct NOCIPropagationResult {
    /// Population updates for determinants owned by current rank.
    pub(in crate::stochastic) local: Vec<(usize, f64)>,
    /// Population updates for determinants owned by another rank, grouped with their destination rank.
    pub(in crate::stochastic) remote: Vec<(usize, NOCIPopulationUpdate)>,
    /// Excitation generation samples.
    pub(in crate::stochastic) samples: Vec<f64>,
    /// Report-local adaptive overlap-weight derivative sums.
    pub(in crate::stochastic) overlap_derivatives: OverlapDerivativeSums,
}

/// Results of one orthogonal propagation cycle, separated by physical ownership.
pub(in crate::stochastic) struct AuxiliaryPropagationResult {
    /// Physical residual updates owned by this rank.
    pub(in crate::stochastic) local: Vec<AuxiliaryPopulationUpdate>,
    /// Physical residual updates owned by another rank and their destination.
    pub(in crate::stochastic) remote: Vec<(usize, AuxiliaryPopulationUpdate)>,
    /// Excitation generation samples.
    pub(in crate::stochastic) samples: Vec<f64>,
}

impl AuxiliaryPropagationResult {
    /// Construct empty cycle-local orthogonal propagation results.
    /// # Arguments:
    /// # Returns:
    /// - `Self`: Reusable local, remote, and sample buffers.
    pub(in crate::stochastic) fn new() -> Self {
        Self {
            local: Vec::new(),
            remote: Vec::new(),
            samples: Vec::new(),
        }
    }

    /// Clear cycle-local results while retaining their capacity.
    /// # Arguments:
    /// - `self`: Reusable propagation results.
    /// # Returns:
    /// - `()`: Empties each result buffer.
    pub(in crate::stochastic) fn clear(&mut self) {
        self.local.clear();
        self.remote.clear();
        self.samples.clear();
    }
}

/// Report-local derivative sums for adaptive overlap-weight optimisation.
#[derive(Clone, Copy, Default)]
pub(in crate::stochastic) struct OverlapDerivativeSums {
    /// Stochastic estimate of `M2'(p)`.
    pub(in crate::stochastic) gradient: f64,
    /// Stochastic estimate of `M2''(p)`.
    pub(in crate::stochastic) hessian: f64,
}

impl OverlapDerivativeSums {
    /// Add another pair of overlap-weight derivative sums.
    /// # Arguments:
    /// - `self`: Accumulator to update.
    /// - `other`: Derivative sums to add.
    /// # Returns:
    /// - `()`: Adds `other` in place.
    pub(in crate::stochastic) fn add(
        &mut self,
        other: &Self,
    ) {
        self.gradient += other.gradient;
        self.hessian += other.hessian;
    }
}

impl NOCIPropagationResult {
    /// Construct empty reusable propagation result storage.
    /// # Arguments:
    /// # Returns
    /// - `NOCIPropagationResult`: Empty propagation result storage.
    pub(in crate::stochastic) fn new() -> Self {
        Self {
            local: Vec::new(),
            remote: Vec::new(),
            samples: Vec::new(),
            overlap_derivatives: OverlapDerivativeSums::default(),
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
        self.overlap_derivatives = OverlapDerivativeSums::default();
    }
}

/// Batched off-diagonal spawn request with a known generation probability.
struct NOCIBatchedSpawnRequest {
    /// Child determinant index.
    child: usize,
    /// Parent determinant index.
    parent: usize,
    /// Per-attempt sampled parent population.
    parent_population: f64,
    /// Exact total generation probability for this determinant pair.
    pgen: f64,
}

/// Batched orthogonal spawn request with a relative connection and known proposal probability.
struct AuxiliaryBatchedSpawnRequest {
    /// Relative connection-table index.
    connection: usize,
    /// Retained source determinant index.
    source: NOCIIndex,
    /// Per-attempt sampled source population.
    parent_population: f64,
    /// Exact uniform generation probability for the connection.
    pgen: f64,
}
/// Report-block derivative of the pre-overlap change with respect to the shift.
/// SApply stores `B = \partial\Delta/\partial E_s = \sum_a dt S\tilde N^{(a)}`;
/// BApply stores `B = \partial\chi/\partial E_s = \sum_a dt\tilde N^{(a)}`.
pub(in crate::stochastic) struct ShiftTangent {
    /// Dense tangent amplitudes indexed by global determinant.
    pub(in crate::stochastic) values: Vec<f64>,
    /// Determinants touched on an MPI-owned sparse path.
    pub(in crate::stochastic) changed: Vec<usize>,
    /// Tangent contributions owned by another MPI rank.
    pub(in crate::stochastic) remote: Vec<(usize, NOCIPopulationUpdate)>,
}

impl ShiftTangent {
    /// Construct zeroed shift-tangent storage.
    /// # Arguments:
    /// - `ndets`: Initial dense tangent length.
    /// # Returns:
    /// - `Self`: Empty tangent accumulator.
    pub(in crate::stochastic) fn new(ndets: usize) -> Self {
        // Both retained-space tangents can occupy most determinant rows before report FRI.
        Self {
            values: vec![0.0; ndets],
            changed: Vec::new(),
            remote: Vec::new(),
        }
    }

    /// Ensure dense tangent storage covers every determinant.
    /// # Arguments:
    /// - `ndets`: Required determinant-space size.
    /// # Returns:
    /// - `()`: Resizes the dense vector when necessary.
    pub(in crate::stochastic) fn prepare(
        &mut self,
        ndets: usize,
    ) {
        // Ordinary walkers do not allocate this storage; both range propagators prepare it.
        if self.values.len() < ndets {
            self.values.resize(ndets, 0.0);
        }
    }

    /// Add one tangent contribution.
    /// # Arguments:
    /// - `det`: Global determinant index.
    /// - `dn`: Tangent amplitude to add.
    /// - `record_changed`: Whether to record first touches for sparse MPI draining.
    /// # Returns:
    /// - `()`: Accumulates the tangent contribution.
    pub(in crate::stochastic) fn add(
        &mut self,
        det: usize,
        dn: f64,
        record_changed: bool,
    ) {
        if dn == 0.0 {
            return;
        }
        // The MPI path records first touches; the single-rank path omits this bookkeeping because B
        // is nearly dense and is reduced with one determinant-major pass.
        if record_changed && self.values[det] == 0.0 {
            self.changed.push(det);
        }
        self.values[det] += dn;
    }

    /// Drain touched tangent entries into sparse population updates.
    /// # Arguments:
    /// - `updates`: Reusable sparse output buffer.
    /// # Returns:
    /// - `()`: Clears touched entries and fills `updates`.
    pub(in crate::stochastic) fn take_sparse(
        &mut self,
        updates: &mut Vec<NOCIPopulationUpdate>,
    ) {
        updates.clear();
        // Convert only touched components of `B = \partial\Delta/\partial E_s` and clear them
        // while draining so the same storage is ready for the next report.
        for det in self.changed.drain(..) {
            let dn = self.values[det];
            self.values[det] = 0.0;
            if dn != 0.0 {
                updates.push(NOCIPopulationUpdate {
                    det: det as u64,
                    dn,
                });
            }
        }
    }
}

/// Storage for per thread propagation quantities.
pub(in crate::stochastic) struct NOCIThreadPropagation {
    /// Population changes generated by this thread that belong to determinants owned by current MPI rank.
    pub(in crate::stochastic) local: Vec<(usize, f64)>,
    /// Population changes generated by this thread that belong to another MPI rank, grouped with destination.
    pub(in crate::stochastic) remote: Vec<(usize, NOCIPopulationUpdate)>,
    /// Report-level SApply shift tangent accumulated by this worker.
    pub(in crate::stochastic) shift_tangent: ShiftTangent,
    /// Excitation generation samples.
    pub(in crate::stochastic) samples: Vec<f64>,
    /// Thread local RNG.
    pub(in crate::stochastic) rng: QmcRng,
    /// Batched off-diagonal spawn requests accumulated over one worker propagation iteration.
    spawn_requests: Vec<NOCIBatchedSpawnRequest>,
    /// Raw off-diagonal spawn events awaiting one worker-batch pivotal compression.
    raw_spawn_updates: Vec<NOCIPopulationUpdate>,
    /// Canonically ordered determinant pairs corresponding to `spawn_requests`.
    spawn_pairs: Vec<(usize, usize)>,
    /// Hamiltonian and overlap elements corresponding to `spawn_requests`.
    spawn_hs: Vec<(f64, f64)>,
    /// Per thread scratch space for extended non-orthogonal Wick's theorem.
    pub(in crate::stochastic) wick_scratch: Box<WickScratchSpin<f64>>,
    /// Iteration-local adaptive overlap-weight derivative sums.
    pub(in crate::stochastic) overlap_derivatives: OverlapDerivativeSums,
}

/// Reusable per-thread propagation storage for parent-orthogonal BApply spawning.
pub(in crate::stochastic) struct AuxiliaryThreadPropagation {
    /// Physical residual updates owned by this MPI rank.
    pub(in crate::stochastic) local: Vec<AuxiliaryPopulationUpdate>,
    /// Physical residual updates owned by another rank and their destination.
    pub(in crate::stochastic) remote: Vec<(usize, AuxiliaryPopulationUpdate)>,
    /// Report-level retained-space shift tangent accumulated by this worker.
    pub(in crate::stochastic) shift_tangent: ShiftTangent,
    /// Excitation-generation histogram samples.
    pub(in crate::stochastic) samples: Vec<f64>,
    /// Thread-local random-number generator.
    pub(in crate::stochastic) rng: QmcRng,
    /// Batched relative connection requests for one stochastic iteration.
    spawn_requests: Vec<AuxiliaryBatchedSpawnRequest>,
    /// Raw off-diagonal auxiliary events awaiting one worker-batch pivotal compression.
    raw_spawn_updates: Vec<AuxiliaryPopulationUpdate>,
    /// Compact source and relative connection indices aligned with requests.
    spawn_pairs: Vec<(NOCIIndex, usize)>,
    /// Parent-orthogonal Hamiltonian results aligned with spawn requests.
    spawn_h: Vec<f64>,
    /// Reusable numerical parent-and-sector grouping storage for orthogonal H batches.
    orthogonal_scratch: OrthogonalHamiltonianScratch,
}

/// Access the shared shift-tangent accumulator of either propagation worker.
pub(in crate::stochastic) trait TangentWorker {
    /// Borrow this worker's report-level retained-space tangent.
    /// # Arguments:
    /// - `self`: Persistent propagation worker.
    /// # Returns:
    /// - `&mut ShiftTangent`: Shared report tangent storage.
    fn tangent(&mut self) -> &mut ShiftTangent;
}

impl TangentWorker for NOCIThreadPropagation {
    /// Borrow the SApply source-path tangent `dt S\tilde N`.
    /// # Arguments:
    /// - `self`: SApply worker.
    /// # Returns:
    /// - `&mut ShiftTangent`: Worker tangent accumulator.
    fn tangent(&mut self) -> &mut ShiftTangent {
        &mut self.shift_tangent
    }
}

impl TangentWorker for AuxiliaryThreadPropagation {
    /// Borrow the BApply source-path tangent `dt\tilde N`.
    /// # Arguments:
    /// - `self`: BApply worker.
    /// # Returns:
    /// - `&mut ShiftTangent`: Worker tangent accumulator.
    fn tangent(&mut self) -> &mut ShiftTangent {
        &mut self.shift_tangent
    }
}

impl NOCIThreadPropagation {
    /// Construct reusable per-thread propagation storage.
    /// # Arguments:
    /// - `seed`: Initial random-number generator seed.
    /// - `maxsame`: Maximum same-spin scratch dimension.
    /// - `maxla`: Maximum alpha-spin different-spin scratch dimension.
    /// - `maxlb`: Maximum beta-spin different-spin scratch dimension.
    /// # Returns
    /// - `NOCIThreadPropagation`: Initialised per-thread propagation storage.
    pub(in crate::stochastic) fn with_sizes(
        seed: u64,
        maxsame: usize,
        maxla: usize,
        maxlb: usize,
    ) -> Self {
        // Initialise B at zero length so ordinary walker propagators never allocate dense N_det
        // storage. SApply expands it on first propagation and retains allocation thereafter.
        Self {
            local: Vec::new(),
            remote: Vec::new(),
            shift_tangent: ShiftTangent::new(0),
            samples: Vec::new(),
            rng: QmcRng::seed_from_u64(seed),
            spawn_requests: Vec::new(),
            raw_spawn_updates: Vec::new(),
            spawn_pairs: Vec::new(),
            spawn_hs: Vec::new(),
            wick_scratch: Box::new(WickScratchSpin::with_sizes(maxsame, maxla, maxlb)),
            overlap_derivatives: OverlapDerivativeSums::default(),
        }
    }

    /// Clear generated updates while retaining allocated storage and Wick scratch space.
    /// # Arguments:
    /// - `self`: Per-thread propagation storage to clear.
    /// # Returns
    /// - `()`: Clears generated updates without releasing their allocations.
    pub(in crate::stochastic) fn clear(&mut self) {
        // These buffers are cycle-local. Do not clear shift_tangent because
        // `B = \sum_{a=1}^{ncycles} dB^{(a)}` accumulates across complete report.
        self.local.clear();
        self.remote.clear();
        self.samples.clear();
        self.spawn_requests.clear();
        self.raw_spawn_updates.clear();
        self.spawn_pairs.clear();
        self.spawn_hs.clear();
        self.overlap_derivatives = OverlapDerivativeSums::default();
    }

    /// Resolve all batched spawn requests accumulated by this worker during one propagation
    /// iteration. Matrix elements are evaluated together before ordinary spawning, pivotal FRI and
    /// ownership logic. For SApply, the same realised overlap elements also supply
    /// `dB_w = dt S_{wx}\tilde N_x/p_gen(w|x)`.
    /// # Arguments:
    /// - `shift`: Current population-control shift.
    /// - `accumulate_shift_tangent`: Whether to differentiate spawning with respect to `E_s`.
    /// - `data`: Immutable stochastic propagation data.
    /// - `run`: Rank-local propagation metadata.
    /// - `overlap`: Overlap factors, proposal generator, mixture weight, and optimisation flag.
    /// # Returns:
    /// - `()`: Appends resolved local and remote population changes.
    pub(in crate::stochastic) fn resolve_batched_spawning(
        &mut self,
        shift: ShiftSpec,
        accumulate_shift_tangent: bool,
        data: &NOCIData<'_, f64>,
        run: &QMCRunInfo,
        overlap: (
            Option<&OverlapFactors>,
            Option<&OverlapWeightedGenerator>,
            f64,
            bool,
        ),
    ) {
        let (overlap_factors, overlap_generator, overlap_weight, optimise_overlap_weight) = overlap;

        if self.spawn_requests.is_empty() && self.raw_spawn_updates.is_empty() {
            return;
        }

        let qmc = data.input.qmc.as_ref().unwrap();
        let dt = data.input.prop_ref().dt;
        let write_excitation_hist = data.input.write.write_excitation_hist;

        self.spawn_pairs.clear();
        for request in &self.spawn_requests {
            let pair = if request.child <= request.parent {
                (request.child, request.parent)
            } else {
                (request.parent, request.child)
            };
            self.spawn_pairs.push(pair);
        }

        self.spawn_hs.clear();
        self.spawn_hs.resize(self.spawn_pairs.len(), (0.0, 0.0));
        find_hs_batched(
            data,
            &self.spawn_pairs,
            self.wick_scratch.as_mut(),
            &mut self.spawn_hs,
        );

        for i in 0..self.spawn_requests.len() {
            let request = &self.spawn_requests[i];
            let (h, s) = self.spawn_hs[i];

            let raw = if accumulate_shift_tangent {
                // Both SApply physical spawning and shift tangent contain
                // `dt \tilde N_x/p_gen(w|x)`, so evaluate this importance factor once.
                let scale = dt * request.parent_population / request.pgen;

                // For `d\Delta_w = -dt(H_{wx}-E_sS_{wx})\tilde N_x/p_gen`,
                // `dB_w = \partial(d\Delta_w)/\partial E_s`
                // `     = +dt S_{wx}\tilde N_x/p_gen`; positive sign comes from differentiating
                // `-(-E_sS_{wx})`. Accumulate raw B before physical spawn FRI because stochastic
                // branch decisions are not differentiated.
                let tangent_dn = s * scale;
                if run.nranks == 1 {
                    self.shift_tangent.add(request.child, tangent_dn, false);
                } else if run.det_owner[request.child] == run.irank {
                    self.shift_tangent.add(request.child, tangent_dn, true);
                } else if tangent_dn != 0.0 {
                    self.shift_tangent.remote.push((
                        run.det_owner[request.child],
                        NOCIPopulationUpdate {
                            det: request.child as u64,
                            dn: tangent_dn,
                        },
                    ));
                }

                let coupling = shift.coupling(h, s);
                -coupling * scale
            } else {
                // Ordinary walker propagation has no shift tangent, so evaluate only its sampled
                // physical update.
                let coupling = shift.coupling(h, s);
                -dt * coupling * request.parent_population / request.pgen
            };

            if optimise_overlap_weight {
                let q_u = 1.0 / (data.space.len() - 1) as f64;
                // For p > 0, recover d/q_p from the realised q_p:
                // d/q_p = (q_p - q_U)/(p q_p), where d = q_S - q_U.
                // At p = 0, q_p = q_U and q_S must be evaluated explicitly.
                let score = if overlap_weight > 0.0 {
                    (request.pgen - q_u) / (overlap_weight * request.pgen)
                } else if let Some(generator) = overlap_generator {
                    let overlap_factors =
                        overlap_factors.expect("overlap-weighted factors must be present");
                    let q_s = generator.overlap_probability(
                        request.parent,
                        request.child,
                        overlap_factors,
                    );
                    (q_s - q_u) / q_u
                } else {
                    0.0
                };
                // With raw = A/q_p, accumulate unbiased estimators
                // g = -raw^2 d/q_p and h = 2 raw^2 (d/q_p)^2.
                let raw2 = raw * raw;
                self.overlap_derivatives.gradient += -raw2 * score;
                self.overlap_derivatives.hessian += 2.0 * raw2 * score * score;
            }

            if write_excitation_hist {
                self.samples.push(raw.abs());
            }

            self.raw_spawn_updates.push(NOCIPopulationUpdate {
                det: request.child as u64,
                dn: raw,
            });
        }

        self.spawn_requests.clear();
        self.spawn_pairs.clear();
        self.spawn_hs.clear();

        // Compress complete event batch before ownership routing; duplicate children remain distinct.
        compress_sparse(
            &mut self.raw_spawn_updates,
            qmc.fri.spawn_cutoff,
            &mut self.rng,
        );
        for update in &self.raw_spawn_updates {
            let child = update.det as usize;
            if run.nranks == 1 {
                self.local.push((child, update.dn));
            } else {
                let destination = run.det_owner[child];

                if destination == run.irank {
                    self.local.push((child, update.dn));
                } else {
                    self.remote.push((destination, *update));
                }
            }
        }
        self.raw_spawn_updates.clear();
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
        shift: ShiftSpec,
        data: &NOCIData<'_, f64>,
        diagonal_hs: &[(f64, f64)],
    ) {
        let (hgg, sgg) = diagonal_hs[gamma];
        let coupling = shift.diagonal_residual(hgg, sgg);
        let dn = -data.input.prop_ref().dt * coupling * population;

        if dn != 0.0 {
            self.local.push((gamma, dn));
        }
    }

    /// Generate off-diagonal real population changes from one sampled determinant.
    /// Each of `n_attempts = ceil(|N_gamma|)` trials carries population
    /// `N_gamma/n_attempts`, so a sampled coupling contributes the unbiased weight
    /// `-dt K_{lambda gamma} N_gamma/(n_attempts P_gen(lambda|gamma))`.
    /// # Arguments:
    /// - `gamma`: Parent determinant index.
    /// - `population`: Real sampled population on `gamma`.
    /// - `shift`: Current population-control shift.
    /// - `data`: Immutable stochastic propagation data.
    /// - `overlap`: Optional overlap factors and generator with current mixture weight.
    /// # Returns:
    /// - `()`: Appends unresolved off-diagonal spawn events to this worker's batch.
    pub(in crate::stochastic) fn spawning(
        &mut self,
        gamma: usize,
        population: f64,
        shift: ShiftSpec,
        data: &NOCIData<'_, f64>,
        overlap: (
            Option<&OverlapFactors>,
            Option<&OverlapWeightedGenerator>,
            f64,
        ),
    ) {
        // Resolve optional overlap-weighted proposal data and discard empty parents early.
        let (overlap_factors, overlap_generator, overlap_weight) = overlap;

        if population == 0.0 {
            return;
        }

        // Split the parent population over independent trials without changing total weight.
        let qmc = data.input.qmc.as_ref().unwrap();
        let dt = data.input.prop_ref().dt;
        let write_excitation_hist = data.input.write.write_excitation_hist;

        let nattempts = population.abs().ceil().max(1.0) as usize;
        let parent_population = population / nattempts as f64;

        // Uniform generation samples every determinant except the parent with equal probability.
        if let ExcitationGen::Uniform = qmc.excitation_gen {
            let ndets = data.space.len();
            let pgen = 1.0 / (ndets - 1) as f64;

            for _ in 0..nattempts {
                let mut lambda = self.rng.gen_range(0..ndets - 1);
                if lambda >= gamma {
                    lambda += 1;
                }

                self.spawn_requests.push(NOCIBatchedSpawnRequest {
                    child: lambda,
                    parent: gamma,
                    parent_population,
                    pgen,
                });
            }

            return;
        }

        // Overlap-weighted generation samples the configured overlap/uniform mixture exactly.
        if let ExcitationGen::OverlapWeighted = qmc.excitation_gen {
            let ndets = data.space.len();
            let generator = overlap_generator.expect("overlap-weighted generator must be present");
            let overlap_factors =
                overlap_factors.expect("overlap-weighted factors must be present");

            for _ in 0..nattempts {
                if self.rng.r#gen::<f64>() < overlap_weight {
                    // Draw from the overlap channel, whose sampler returns the full mixture pgen.
                    match generator.sample_overlap(
                        gamma,
                        overlap_factors,
                        overlap_weight,
                        &mut self.rng,
                    ) {
                        OverlapProposal::Valid { child, pgen } => {
                            self.spawn_requests.push(NOCIBatchedSpawnRequest {
                                child,
                                parent: gamma,
                                parent_population,
                                pgen,
                            });
                        }
                        OverlapProposal::Null => {}
                    }
                } else {
                    // Draw uniformly, then evaluate the full mixture probability for reweighting.
                    let mut lambda = self.rng.gen_range(0..ndets - 1);
                    if lambda >= gamma {
                        lambda += 1;
                    }
                    let pgen = generator.mixture_probability(
                        gamma,
                        lambda,
                        overlap_factors,
                        overlap_weight,
                    );

                    self.spawn_requests.push(NOCIBatchedSpawnRequest {
                        child: lambda,
                        parent: gamma,
                        parent_population,
                        pgen,
                    });
                }
            }

            return;
        }

        // Cache parent-specific heat-bath normalisation once for all attempts from this parent.
        let heat_bath = if let ExcitationGen::HeatBath = qmc.excitation_gen {
            Some(init_heat_bath(
                gamma,
                shift.overlap_shift(),
                data,
                self.wick_scratch.as_mut(),
            ))
        } else {
            None
        };

        // Generate and retain raw importance-sampled events for later batching and coalescing.
        for _ in 0..nattempts {
            let (pgen, k, lambda) = match qmc.excitation_gen {
                ExcitationGen::HeatBath => pgen_heat_bath(
                    gamma,
                    shift.overlap_shift(),
                    data,
                    &mut self.rng,
                    heat_bath.as_ref().unwrap(),
                    self.wick_scratch.as_mut(),
                ),
                ExcitationGen::ApproximateHeatBath => {
                    unimplemented!()
                }
                ExcitationGen::Uniform => unreachable!(),
                ExcitationGen::OverlapWeighted => unreachable!(),
            };

            // Apply `1/P_gen` so the expected spawned change equals the exact propagator action.
            let raw = -dt * k * parent_population / pgen;

            if write_excitation_hist {
                self.samples.push(raw.abs());
            }

            // Preserve individual events until the common compression/communication stage.
            self.raw_spawn_updates.push(NOCIPopulationUpdate {
                det: lambda as u64,
                dn: raw,
            });
        }
    }
}

impl AuxiliaryThreadPropagation {
    /// Construct reusable storage for orthogonal residual generation `\chi=-dt(\hat H-E_s)BN`.
    /// # Arguments:
    /// - `seed`: Initial thread-local random-number seed.
    /// - `nparents`: Number of source parent references for numerical request grouping.
    /// # Returns
    /// - `Self`: Empty report and iteration buffers for BApply spawning.
    pub(in crate::stochastic) fn new(
        seed: u64,
        nparents: usize,
    ) -> Self {
        Self {
            local: Vec::new(),
            remote: Vec::new(),
            shift_tangent: ShiftTangent::new(0),
            samples: Vec::new(),
            rng: QmcRng::seed_from_u64(seed),
            spawn_requests: Vec::new(),
            raw_spawn_updates: Vec::new(),
            spawn_pairs: Vec::new(),
            spawn_h: Vec::new(),
            orthogonal_scratch: OrthogonalHamiltonianScratch::new(nparents),
        }
    }

    /// Clear one cycle's generated orthogonal updates while retaining the report tangent.
    /// # Arguments:
    /// - `self`: Persistent worker storage.
    /// # Returns:
    /// - `()`: Resets cycle-local buffers for reuse.
    pub(in crate::stochastic) fn clear(&mut self) {
        self.local.clear();
        self.remote.clear();
        self.samples.clear();
        self.spawn_requests.clear();
        self.raw_spawn_updates.clear();
        self.spawn_pairs.clear();
        self.spawn_h.clear();
    }

    /// Route a realised `\chi_D^P` update to its physical determinant owner.
    /// # Arguments:
    /// - `det`: Arbitrary physical determinant `D^P`.
    /// - `dn`: Realised orthogonal residual amplitude.
    /// - `run`: Current rank and MPI ownership metadata.
    /// # Returns:
    /// - `()`: Appends a local or remote update.
    fn route_update(
        &mut self,
        det: AuxiliaryIndex,
        dn: f64,
        run: &QMCRunInfo,
    ) {
        let update = AuxiliaryPopulationUpdate::new(det, dn);
        let peer = auxiliary_owner(det, run.nranks);
        if peer == run.irank {
            self.local.push(update);
        } else {
            self.remote.push((peer, update));
        }
    }

    /// Generate the diagonal orthogonal residual `\chi_x=-dt(H_{xx}-E_s)\tilde N_x`.
    /// # Arguments:
    /// - `source`: Retained source determinant index `x`.
    /// - `population`: Sampled real population `\tilde N_x`.
    /// - `shift`: Current physical shift `E_s`.
    /// - `data`: Shared input containing the timestep.
    /// - `auxiliary`: Auxiliary determinant space used to index the orthogonal update.
    /// - `run`: Rank-local metadata containing cached diagonal `H_{xx}`.
    /// # Returns
    /// - `()`: Appends a nonzero diagonal orthogonal update.
    pub(in crate::stochastic) fn diagonal_population_change(
        &mut self,
        source: usize,
        population: f64,
        shift: f64,
        data: &NOCIData<'_, f64>,
        auxiliary: &AuxiliarySpace,
        run: &QMCRunInfo,
    ) {
        let hxx = run.diagonal_hs[source].0;
        let dn = -data.input.prop_ref().dt * (hxx - shift) * population;

        if dn != 0.0 {
            let det = auxiliary.embed_noci(data.space, NOCIIndex(source));
            self.route_update(det, dn, run);
        }
    }

    /// Sample uniform relative connections for the off-diagonal residual
    /// `\chi_D=-dt H_{Dx}\tilde N_x/P_\mathrm{gen}(D|x)`.
    /// # Arguments:
    /// - `source`: Retained source determinant index `x`.
    /// - `population`: Sampled real population `\tilde N_x`.
    /// - `generator`: Persistent system-wide orthogonal connection topology.
    /// # Returns
    /// - `()`: Appends unresolved batched spawn requests.
    pub(in crate::stochastic) fn spawning(
        &mut self,
        source: usize,
        population: f64,
        generator: &OrthogonalUniformGenerator,
    ) {
        if population == 0.0 {
            return;
        }

        // Split a large signed population over `\lceil|\tilde N_x|\rceil`
        // independent connection draws while preserving its total weight.
        let nattempts = population.abs().ceil().max(1.0) as usize;
        let parent_population = population / nattempts as f64;
        for _ in 0..nattempts {
            if let Some((connection, pgen)) = generator.sample(&mut self.rng) {
                self.spawn_requests.push(AuxiliaryBatchedSpawnRequest {
                    connection,
                    source: NOCIIndex(source),
                    parent_population,
                    pgen,
                });
            }
        }
    }

    /// Resolve and evaluate `H_{Dx}` for all requests before pivotal spawn compression.
    /// Each raw event `\chi_D=-dt H_{Dx}\tilde N_x/P_\mathrm{gen}(D|x)` constructs its connected
    /// `O'_\sigma=(O_\sigma\setminus I_\sigma)\cup A_\sigma` key, then the complete worker batch is
    /// compressed without coalescing duplicate keys.
    /// # Arguments:
    /// - `data`: Shared NOCI basis, parent MO caches, timestep, and FRI configuration.
    /// - `generator`: Persistent system-wide orthogonal connection topology.
    /// - `auxiliary`: Canonical auxiliary determinant space.
    /// - `run`: MPI ownership metadata for realised physical determinants.
    /// # Returns
    /// - `()`: Appends realised orthogonal updates and clears iteration-local batch buffers.
    pub(in crate::stochastic) fn resolve_batched_spawning(
        &mut self,
        data: &NOCIData<'_, f64>,
        generator: &OrthogonalUniformGenerator,
        auxiliary: &AuxiliarySpace,
        run: &QMCRunInfo,
    ) {
        if self.spawn_requests.is_empty() && self.raw_spawn_updates.is_empty() {
            return;
        }

        // Gather all requested orthogonal couplings for one batched Hamiltonian
        // evaluation before constructing individual spawn amplitudes.
        self.spawn_pairs.clear();
        self.spawn_pairs.extend(
            self.spawn_requests
                .iter()
                .map(|request| (request.source, request.connection)),
        );
        self.spawn_h.clear();
        self.spawn_h.resize(self.spawn_requests.len(), 0.0);
        find_h_orthogonal_batched(
            data,
            generator,
            &self.spawn_pairs,
            &mut self.orthogonal_scratch,
            &mut self.spawn_h,
        );

        let qmc = data.input.qmc.as_ref().unwrap();
        let dt = data.input.prop_ref().dt;
        let record_samples = data.input.write.write_excitation_hist;

        // Each raw event contributes `-\Delta t\, H_{Dx}\tilde N_x / P_{\text{gen}}(D|x)` to its
        // connected auxiliary determinant, before pivotal compression.
        for i in 0..self.spawn_requests.len() {
            let request = &self.spawn_requests[i];
            let raw = -dt * self.spawn_h[i] * request.parent_population / request.pgen;
            if record_samples {
                self.samples.push(raw.abs());
            }

            let child = auxiliary.connected(
                data.space,
                request.source,
                generator.connections()[request.connection],
            );
            self.raw_spawn_updates
                .push(AuxiliaryPopulationUpdate::new(child, raw));
        }

        self.spawn_requests.clear();
        self.spawn_pairs.clear();
        self.spawn_h.clear();

        // Preserve event-level spawn-cutoff semantics by pivotalising before duplicate coalescing.
        compress_sparse(
            &mut self.raw_spawn_updates,
            qmc.fri.spawn_cutoff,
            &mut self.rng,
        );

        // Route compressed events to their owning MPI ranks, then release
        // iteration-local batch storage.
        for i in 0..self.raw_spawn_updates.len() {
            let update = self.raw_spawn_updates[i];
            self.route_update(update.index(), update.dn, run);
        }
        self.raw_spawn_updates.clear();
    }
}

/// Storage for a sparse real population change communicated across MPI ranks.
#[repr(C)]
#[derive(Copy, Clone, Equivalence)]
pub(crate) struct NOCIPopulationUpdate {
    /// Determinant index to which the population change applies.
    pub det: u64,
    /// Signed real population change.
    pub dn: f64,
}

/// Storage for one sparse parent-orthogonal population change communicated across MPI ranks.
#[repr(C)]
#[derive(Copy, Clone, Equivalence)]
pub(crate) struct AuxiliaryPopulationUpdate {
    /// Deterministic flattened auxiliary determinant index.
    pub(crate) det: u64,
    /// Signed orthogonal-space residual amplitude.
    pub(crate) dn: f64,
}

impl AuxiliaryPopulationUpdate {
    /// Encode one sparse `\chi_D^P` update for MPI transport.
    /// # Arguments:
    /// - `det`: Deterministic flattened auxiliary identity of `D^P`.
    /// - `dn`: Signed realised residual amplitude.
    /// # Returns
    /// - `Self`: MPI-safe orthogonal population update.
    pub(crate) fn new(
        det: AuxiliaryIndex,
        dn: f64,
    ) -> Self {
        Self {
            det: det.0 as u64,
            dn,
        }
    }

    /// Decode the determinant key of one sparse `\chi_D^P` update.
    /// # Arguments:
    /// - `self`: MPI-safe orthogonal population update.
    /// # Returns
    /// - `AuxiliaryIndex`: Flattened auxiliary determinant defining `D^P`.
    pub(crate) fn index(&self) -> AuxiliaryIndex {
        AuxiliaryIndex(self.det as usize)
    }
}

impl FriAmplitude for AuxiliaryPopulationUpdate {
    /// Return one orthogonal-space residual amplitude `\chi_D^P`.
    /// # Arguments:
    /// - `self`: Sparse orthogonal population update.
    /// # Returns
    /// - `f64`: Signed residual amplitude.
    fn amplitude(&self) -> f64 {
        self.dn
    }

    /// Replace one orthogonal-space residual amplitude `\chi_D^P`.
    /// # Arguments:
    /// - `self`: Sparse orthogonal population update.
    /// - `amplitude`: Replacement signed residual amplitude.
    /// # Returns
    /// - `()`: Updates the stored amplitude.
    fn set_amplitude(
        &mut self,
        amplitude: f64,
    ) {
        self.dn = amplitude;
    }
}

/// Storage for heat-bath excitation related quantities.
pub(in crate::stochastic) struct HeatBath {
    /// Total absolute coupling weight over all allowed children.
    pub(in crate::stochastic) sumxw: f64,
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
pub(crate) struct NOCIMPIScratch {
    /// Number of sparse updates contributed by each rank for all-gather.
    pub(crate) gather_counts: Vec<i32>,
    /// Displacements for gathered sparse updates.
    pub(crate) gather_displs: Vec<i32>,
    /// Reusable receive buffer for gathered sparse updates.
    pub(crate) gather_recv: Vec<NOCIPopulationUpdate>,
    /// Number of spawn updates sent to each rank.
    pub(crate) send_counts: Vec<i32>,
    /// Displacements into the contiguous spawn-send buffer for each rank.
    pub(crate) send_displacements: Vec<i32>,
    /// Number of spawn updates received from each rank.
    pub(crate) recv_counts: Vec<i32>,
    /// Displacements into the contiguous spawn-receive buffer for each rank.
    pub(crate) recv_displacements: Vec<i32>,
    /// Reusable contiguous send buffer for spawn exchange.
    pub(crate) send_contig: Vec<NOCIPopulationUpdate>,
    /// Reusable remote spawn updates with destination ranks.
    pub(crate) send_ranked: Vec<(usize, NOCIPopulationUpdate)>,
    /// Reusable contiguous receive buffer for spawn exchange.
    pub(crate) recv_contig: Vec<NOCIPopulationUpdate>,
}

/// Reusable MPI scratch for parent-orthogonal residual collectives.
#[derive(Default)]
pub(crate) struct AuxiliaryMPIScratch {
    /// Number of sparse orthogonal updates contributed by each rank for all-gather.
    pub(crate) gather_counts: Vec<i32>,
    /// Displacements for gathered sparse orthogonal updates.
    pub(crate) gather_displs: Vec<i32>,
    /// Reusable receive buffer for gathered sparse orthogonal updates.
    pub(crate) gather_recv: Vec<AuxiliaryPopulationUpdate>,
    /// Number of orthogonal updates sent to each deterministic owner rank.
    pub(crate) send_counts: Vec<i32>,
    /// Displacements into the contiguous orthogonal send buffer.
    pub(crate) send_displacements: Vec<i32>,
    /// Number of orthogonal updates received from each rank.
    pub(crate) recv_counts: Vec<i32>,
    /// Displacements into the contiguous orthogonal receive buffer.
    pub(crate) recv_displacements: Vec<i32>,
    /// Reusable contiguous orthogonal send buffer.
    pub(crate) send_contig: Vec<AuxiliaryPopulationUpdate>,
    /// Reusable orthogonal updates paired with deterministic owner ranks.
    pub(crate) send_ranked: Vec<(usize, AuxiliaryPopulationUpdate)>,
    /// Reusable contiguous orthogonal receive buffer.
    pub(crate) recv_contig: Vec<AuxiliaryPopulationUpdate>,
}

impl NOCIMPIScratch {
    /// Construct reusable MPI scratch buffers.
    /// # Arguments:
    /// - `nranks`: Number of MPI ranks.
    /// # Returns
    /// - `NOCIMPIScratch`: Scratch storage sized for the communicator.
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
            send_ranked: Vec::new(),
            recv_contig: Vec::new(),
        }
    }
}

impl AuxiliaryMPIScratch {
    /// Construct reusable MPI storage for distributed `\chi_D^P` updates.
    /// # Arguments:
    /// - `nranks`: Number of MPI ranks.
    /// # Returns
    /// - `Self`: Empty orthogonal communication buffers sized for the communicator.
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
            send_ranked: Vec::new(),
            recv_contig: Vec::new(),
        }
    }
}
