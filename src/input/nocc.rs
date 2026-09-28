// input/nocc.rs

// Standard library imports.
use std::str::FromStr;

/// Weighting of the FOIS metric before canonical orthogonalisation.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum FoisWeighting {
    /// Unit weights on excitations with `|h_\mu|` above the coupling threshold and zero weight
    /// on all others, so the unweighted metric of the Hamiltonian-coupled excitations is
    /// orthogonalised.
    Coupled,
    /// Hamiltonian weights `h_\mu`, orthogonalising `\tilde S = hSh`.
    Hamiltonian,
}

impl FoisWeighting {
    /// Return FOIS weighting as input string.
    /// # Arguments:
    /// - `self`: FOIS weighting.
    /// # Returns:
    /// - `&'static str`: String representation used in input parsing and printing.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Coupled => "coupled",
            Self::Hamiltonian => "hamiltonian",
        }
    }
}

impl FromStr for FoisWeighting {
    type Err = String;

    /// Parse FOIS weighting from input string.
    /// # Arguments:
    /// - `s`: String specifying the FOIS weighting.
    /// # Returns:
    /// - `Result<Self, Self::Err>`: Parsed FOIS weighting if valid string, otherwise error message.
    /// # Errors
    /// - Returns an error if `s` does not name a supported FOIS weighting.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "coupled" => Ok(Self::Coupled),
            "hamiltonian" => Ok(Self::Hamiltonian),
            _ => Err(format!("invalid FOIS weighting: {s}")),
        }
    }
}

pub struct NOCCMCOptions {
    /// Natural-occupation tolerance for the active space: orbitals within this distance of two
    /// or zero electrons are core or virtual, and all others active.
    pub active_space_tol: f64,
    /// Highest cumulant rank kept in the energy and residual equations; terms with higher-rank
    /// cumulants are dropped. The metric and zeroth-order coupling are always evaluated exactly.
    pub max_cumulant: usize,
    /// Weighting of the FOIS metric before canonical orthogonalisation.
    pub fois_weighting: FoisWeighting,
    /// Hamiltonian coupling threshold: with coupled weighting, excitations with `|h_\mu|` at or
    /// below it are excluded from the FOIS.
    pub fois_coupling_tol: f64,
    /// Eigenvalue threshold of the weighted FOIS metric; directions below it are discarded as
    /// redundant.
    pub fois_tol: f64,
    /// Maximum number of amplitude macro-iterations.
    pub max_macro: usize,
    /// Maximum number of micro-iterations per amplitude update.
    pub max_micro: usize,
    /// Convergence threshold on the FOIS residual norm `\lVert Y^\dagger R\rVert`.
    pub residual_tol: f64,
    /// Convergence threshold on the linearised update equation residual.
    pub micro_tol: f64,
    /// Level shift `\eta` added to the orbital-energy denominators.
    pub level_shift: f64,
    /// Number of vectors kept in each DIIS subspace.
    pub diis_space: usize,
}

impl Default for NOCCMCOptions {
    /// Return default NOCCMC options.
    /// # Returns:
    /// - `Self`: NOCCMC options.
    fn default() -> Self {
        Self {
            active_space_tol: 1e-6,
            max_cumulant: 4,
            fois_weighting: FoisWeighting::Coupled,
            fois_coupling_tol: 1e-8,
            fois_tol: 1e-8,
            max_macro: 100,
            max_micro: 200,
            residual_tol: 1e-8,
            micro_tol: 1e-10,
            level_shift: 0.5,
            diis_space: 8,
        }
    }
}
