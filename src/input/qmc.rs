// input/qmc.rs

// Standard library imports.
use std::str::FromStr;

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
    /// Minimum sampled persistent-population magnitude.
    pub sampling_cutoff1: f64,
    /// Minimum sampled pre-overlap population-change magnitude.
    pub sampling_cutoff2: f64,
    /// Minimum spawned population-change magnitude.
    pub spawn_cutoff: f64,
    /// Shift damping factor.
    pub shift_damping: f64,
    /// Number of QMC cycles per report block.
    pub ncycles: usize,
    /// Number of report blocks.
    pub nreports: usize,
    /// Excitation generator choice.
    pub excitation_gen: ExcitationGen,
    /// Mixture weight for the factorised-overlap excitation proposal.
    pub overlap_weight: f64,
    /// Whether to optimise the overlap mixture weight during propagation.
    pub optimise_overlap_weight: bool,
    /// Optional RNG seed.
    pub seed: Option<u64>,
}

impl Default for QMCOptions {
    fn default() -> Self {
        Self {
            initial_population: 100.0,
            target_population: 100000.0,
            sampling_cutoff1: 0.0,
            sampling_cutoff2: 0.0,
            spawn_cutoff: 0.0,
            shift_damping: 5e-4,
            ncycles: 10,
            nreports: 1000,
            excitation_gen: ExcitationGen::default(),
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
