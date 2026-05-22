// input/state.rs

use std::str::FromStr;

pub enum Spin {
    Alpha,
    Beta,
    Both,
}

impl Spin {
    /// Return excitation spin as input string.
    /// # Returns:
    /// - `&'static str`: String representation used in input parsing.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Alpha => "alpha",
            Self::Beta => "beta",
            Self::Both => "both",
        }
    }
}

impl FromStr for Spin {
    type Err = String;

    /// Parse excitation spin from input string.
    /// # Arguments:
    /// - `s`: String specifying the excitation spin.
    /// # Returns:
    /// - `Result<Self, Self::Err>`: Parsed spin if valid string, otherwise error message.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "alpha" => Ok(Self::Alpha),
            "beta" => Ok(Self::Beta),
            "both" => Ok(Self::Both),
            _ => Err(format!("invalid excitation spin: {s}")),
        }
    }
}

impl Default for Spin {
    /// Return default excitation spin.
    /// # Returns:
    /// - `Self`: Default excitation spin choice.
    fn default() -> Self {
        Self::Both
    }
}

pub struct SCFExcitation {
    /// Spin channel to excite.
    pub spin: Spin,
    /// Occupied orbital offset.
    pub occ: i32,
    /// Virtual orbital offset.
    pub vir: i32,
}

impl Default for SCFExcitation {
    /// Return default SCF excitation.
    /// # Returns:
    /// - `Self`: Excitation with no excitation.
    fn default() -> Self {
        Self {
            spin: Spin::default(),
            occ: 0,
            vir: 0,
        }
    }
}

pub struct SpinBias {
    /// Atom-resolved spin-bias pattern.
    pub pattern: Vec<i8>,
    /// Spin-bias polarization strength.
    pub pol: f64,
}

impl Default for SpinBias {
    /// Return default spin bias options.
    /// # Returns:
    /// - `Self`: Spin bias with empty pattern and zero polarization.
    fn default() -> Self {
        Self {
            pattern: Vec::new(),
            pol: 0.0,
        }
    }
}

pub struct SpatialBias {
    /// Atom-resolved spatial-bias pattern.
    pub pattern: Vec<i8>,
    /// Spatial-bias polarization strength.
    pub pol: f64,
}

impl Default for SpatialBias {
    /// Return default spatial bias options.
    /// # Returns:
    /// - `Self`: Spatial bias with empty pattern and zero polarization.
    fn default() -> Self {
        Self {
            pattern: Vec::new(),
            pol: 0.0,
        }
    }
}

pub struct StateRecipe {
    /// User-visible state label.
    pub label: String,
    /// Optional spin-density bias.
    pub spin_bias: Option<SpinBias>,
    /// Optional spatial-density bias.
    pub spatial_bias: Option<SpatialBias>,
    /// Optional MOM excitation.
    pub scfexcitation: Option<SCFExcitation>,
    /// Real partner label used for h-SCF gating.
    pub partner: Option<String>,
    /// Whether state enters NOCI basis.
    pub noci: bool,
    /// Whether recipe generates h-SCF state.
    pub holomorphic: bool,
}

impl Default for StateRecipe {
    /// Return default state recipe.
    /// # Returns:
    /// - `Self`: State recipe with empty label and no bias or excitation. We assume that a state
    ///   is desired to be used within NOCI by default.
    fn default() -> Self {
        Self {
            label: String::new(),
            spin_bias: None,
            spatial_bias: None,
            scfexcitation: None,
            partner: None,
            noci: true,
            holomorphic: false,
        }
    }
}

pub struct Metadynamics {
    /// Number of RHF states requested.
    pub nstates_rhf: usize,
    /// Number of UHF states requested.
    pub nstates_uhf: usize,
    /// Spin-bias polarization strength.
    pub spinpol: f64,
    /// Spatial-bias polarization strength.
    pub spatialpol: f64,
    /// Metadynamics bias strength.
    pub lambda: f64,
    /// Generated RHF state labels.
    pub labels_rhf: Vec<String>,
    /// Generated UHF state labels.
    pub labels_uhf: Vec<String>,
    /// Successful RHF spatial-bias patterns.
    pub spatial_patterns_rhf: Vec<Option<Vec<i8>>>,
    /// Successful UHF spin-bias patterns.
    pub spin_patterns_uhf: Vec<Option<Vec<i8>>>,
    /// Maximum number of biased-SCF attempts.
    pub max_attempts: usize,
}

impl Default for Metadynamics {
    /// Return default metadynamics options.
    /// # Returns:
    /// - `Self`: Metadynamics options with no requested states and empty labels and patterns.
    fn default() -> Self {
        Self {
            nstates_rhf: 0,
            nstates_uhf: 0,
            spinpol: 0.0,
            spatialpol: 0.0,
            lambda: 0.0,
            labels_rhf: Vec::new(),
            labels_uhf: Vec::new(),
            spatial_patterns_rhf: Vec::new(),
            spin_patterns_uhf: Vec::new(),
            max_attempts: 100,
        }
    }
}

pub enum StateType {
    Mom(Vec<StateRecipe>),
    Metadynamics(Metadynamics),
}

impl Default for StateType {
    /// Return default state specification.
    /// # Returns:
    /// - `Self`: Default empty MOM state list.
    fn default() -> Self {
        Self::Mom(Vec::new())
    }
}
