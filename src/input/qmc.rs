// input/qmc.rs

use std::str::FromStr;

pub enum ExcitationGen {
    Uniform,
    HeatBath,
    ApproximateHeatBath,
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
    /// Initial walker population.
    pub initial_population: i64,
    /// Target walker population.
    pub target_population: i64,
    /// Shift damping factor.
    pub shift_damping: f64,
    /// Number of QMC cycles per report block.
    pub ncycles: usize,
    /// Number of report blocks.
    pub nreports: usize,
    /// Excitation generator choice.
    pub excitation_gen: ExcitationGen,
    /// Optional RNG seed.
    pub seed: Option<u64>,
}

impl Default for QMCOptions {
    /// Return default QMC propagation options.
    /// # Returns:
    /// - `Self`: QMC propagation options with default population and shift settings.
    fn default() -> Self {
        Self {
            initial_population: 100,
            target_population: 100000,
            shift_damping: 5e-4,
            ncycles: 10,
            nreports: 1000,
            excitation_gen: ExcitationGen::default(),
            seed: None,
        }
    }
}
