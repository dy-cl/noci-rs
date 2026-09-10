// input/qmc.rs

// Standard library imports.
use std::str::FromStr;

// Parent/sibling imports.
use super::SNOCIStorage;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ExcitationGen {
    Uniform,
    HeatBath,
    ApproximateHeatBath,
    OverlapWeighted,
}

impl FromStr for ExcitationGen {
    type Err = String;

    /// Parse excitation generator from input string.
    /// # Arguments:
    /// - `s`: String specifying the excitation generator.
    /// # Returns:
    /// - `Result<Self, Self::Err>`: Parsed excitation generator if valid string, otherwise error message.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "uniform" => Ok(Self::Uniform),
            "heat-bath" => Ok(Self::HeatBath),
            "approximate-heat-bath" => Ok(Self::ApproximateHeatBath),
            "overlap-weighted" => Ok(Self::OverlapWeighted),
            _ => Err(format!("invalid excitation generator: {s}")),
        }
    }
}

impl Default for ExcitationGen {
    /// Return default excitation generator.
    /// # Returns:
    /// - `Self`: Default excitation generator choice.
    fn default() -> Self {
        Self::Uniform
    }
}

pub struct QMCOptions {
    /// Initial persistent population 1-norm.
    pub initial_population: f64,
    /// Target persistent population 1-norm.
    pub target_population: f64,
    /// FRI configuration for each stochastic compression site.
    pub fri: FriOptions,
    /// Shift damping factor.
    pub shift_damping: f64,
    /// Number of QMC cycles per report block.
    pub ncycles: usize,
    /// Number of report blocks.
    pub nreports: usize,
    /// Excitation generator choice.
    pub excitation_gen: ExcitationGen,
    /// Storage strategy for persistent overlap factor tables.
    pub factor_tables: SNOCIStorage,
    /// Mixture weight for the factorised-overlap excitation proposal.
    pub overlap_weight: f64,
    /// Whether to optimise the overlap mixture weight during propagation.
    pub optimise_overlap_weight: bool,
    /// Optional RNG seed.
    pub seed: Option<u64>,
}

/// FRI policies fixed by stochastic compression site.
#[derive(Clone, Copy)]
pub struct FriOptions {
    /// Fixed cutoff for sampling persistent populations.
    pub population_cutoff: f64,
    /// Fixed cutoff for individual spawned population changes.
    pub spawn_cutoff: f64,
    /// Per-MPI-rank target NNZ for the physical pre-overlap report vector.
    pub pre_overlap_target_nnz: usize,
    /// Per-MPI-rank target NNZ for the DirectOverlap shift tangent.
    pub shift_tangent_target_nnz: usize,
}

impl Default for FriOptions {
    /// Return explicit default FRI policies for every compression site.
    /// # Returns:
    /// - `Self`: Fixed cutoffs `1.0` and `0.25`, with per-rank report targets
    ///   `2048` and `1024` for the physical and shift-tangent vectors.
    fn default() -> Self {
        // Population sampling and individual spawning use fixed amplitude cutoffs, while the
        // report-level vectors use adaptive cutoffs determined from
        // `M(c) = \sum_i min(1, |x_i|/c)`.
        Self {
            population_cutoff: 1.0,
            spawn_cutoff: 0.25,
            pre_overlap_target_nnz: 2048,
            shift_tangent_target_nnz: 1024,
        }
    }
}

impl Default for QMCOptions {
    /// Return default stochastic QMC options.
    /// # Returns:
    /// - `Self`: Default population, propagation, excitation, and FRI configuration.
    fn default() -> Self {
        // Keep global excitation-generator default uniform. Parsing changes only an omitted
        // DirectOverlap generator to overlap-weighted because that path already builds overlap
        // factors needed by its explicit metric action.
        Self {
            initial_population: 100.0,
            target_population: 100000.0,
            fri: FriOptions::default(),
            shift_damping: 5e-4,
            ncycles: 10,
            nreports: 1000,
            excitation_gen: ExcitationGen::default(),
            factor_tables: SNOCIStorage::RAM,
            overlap_weight: 0.0,
            optimise_overlap_weight: false,
            seed: None,
        }
    }
}

pub struct NOCCMCOptions {}

impl Default for NOCCMCOptions {
    /// Return default NOCCMC options.
    /// # Returns:
    /// - `Self`: NOCCMC options.
    fn default() -> Self {
        Self {}
    }
}
