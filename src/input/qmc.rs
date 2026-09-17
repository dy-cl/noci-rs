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
    /// Shift-control activation population and, when restoring is enabled, persistent target.
    pub target_population: f64,
    /// Total number of determinants retained in the projected-energy trial state.
    /// `None` uses exactly the number of NOCI reference determinants.
    pub n_projected: Option<usize>,
    /// FRI configuration for each stochastic compression site.
    pub fri: FriOptions,
    /// Damping `\zeta` of the population-control Newton update.
    pub shift_damping: f64,
    /// Dimensionless target-restoring strength `\kappa` for range propagators.
    pub population_restoring: f64,
    /// Dimensionless report-level heavy-ball coefficient `\beta` for BApply.
    pub momentum_beta: f64,
    /// Number of QMC cycles per report block.
    pub ncycles: usize,
    /// Number of report blocks.
    pub nreports: usize,
    /// Excitation generator choice.
    pub excitation_gen: ExcitationGen,
    /// Storage strategy for persistent overlap factor tables.
    pub factor_tables: SNOCIStorage,
    /// SApply storage strategy for persistent overlap factors and proposal CDFs.
    pub sapply_factor_tables: SNOCIStorage,
    /// BApply storage strategy for persistent overlap factors.
    pub bapply_factor_tables: SNOCIStorage,
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
    /// Per-MPI-rank target NNZ for a range-propagator shift tangent.
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
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Self`: Default population, projection, propagation, excitation, and FRI configuration.
    fn default() -> Self {
        // Resolve the default projection size from the actual reference count during QMC setup.
        Self {
            initial_population: 100.0,
            target_population: 100000.0,
            n_projected: None,
            fri: FriOptions::default(),
            shift_damping: 5e-4,
            population_restoring: 0.0,
            momentum_beta: 0.0,
            ncycles: 10,
            nreports: 1000,
            excitation_gen: ExcitationGen::default(),
            factor_tables: SNOCIStorage::RAM,
            sapply_factor_tables: SNOCIStorage::RAM,
            bapply_factor_tables: SNOCIStorage::RAM,
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
