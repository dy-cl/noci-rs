// input/nocc.rs

// Standard library imports.
use std::str::FromStr;

// Parent/sibling imports.
use super::gmres::GMRESOptions;

/// Linear solver of each GNOCC Newton step `Y^\dagger J Y\,\delta c = -Y^\dagger R`.
#[derive(Clone, Copy)]
pub enum NOCCNewton {
    /// Build the FOIS Jacobian `Y^\dagger J Y` explicitly and solve it directly.
    Dense,
    /// Solve matrix-free by preconditioned GMRES on Jacobian-vector products.
    GMRES,
}

impl NOCCNewton {
    /// Return the Newton solver as input string.
    /// # Arguments:
    /// - `self`: Newton solver selection.
    /// # Returns:
    /// - `&'static str`: String representation used in input parsing.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::GMRES => "gmres",
        }
    }
}

impl FromStr for NOCCNewton {
    type Err = String;

    /// Parse the Newton solver from input string.
    /// # Arguments:
    /// - `s`: String specifying the Newton solver.
    /// # Returns:
    /// - `Result`: Parsed Newton solver if valid string, otherwise error message.
    /// # Errors
    /// - Returns an error if `s` does not name a supported Newton solver.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "dense" => Ok(Self::Dense),
            "gmres" => Ok(Self::GMRES),
            _ => Err(format!("invalid NOCC Newton solver: {s}")),
        }
    }
}

/// Preconditioner of the GMRES Newton solver.
#[derive(Clone, Copy)]
pub enum NOCCPreconditioner {
    /// Apply the inverse of the exact linear FOIS Jacobian `Y^\dagger J_1 Y`, the Jacobian at
    /// `t = 0`, built once per run.
    Linear,
    /// Apply the inverse diagonal `[Y^\dagger S D Y]_{ii}` of the level-shifted orbital-energy
    /// denominators.
    Diagonal,
}

impl NOCCPreconditioner {
    /// Return the preconditioner as input string.
    /// # Arguments:
    /// - `self`: Preconditioner selection.
    /// # Returns:
    /// - `&'static str`: String representation used in input parsing.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Diagonal => "diagonal",
        }
    }
}

impl FromStr for NOCCPreconditioner {
    type Err = String;

    /// Parse the preconditioner from input string.
    /// # Arguments:
    /// - `s`: String specifying the preconditioner.
    /// # Returns:
    /// - `Result`: Parsed preconditioner if valid string, otherwise error message.
    /// # Errors
    /// - Returns an error if `s` does not name a supported preconditioner.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "linear" => Ok(Self::Linear),
            "diagonal" => Ok(Self::Diagonal),
            _ => Err(format!("invalid NOCC preconditioner: {s}")),
        }
    }
}

pub struct NOCCMCOptions {
    /// Natural-occupation tolerance for the active space: orbitals within this distance of two
    /// or zero electrons are core or virtual, and all others active.
    pub active_space_tol: f64,
    /// Highest cumulant rank kept in the energy and residual equations; terms with higher-rank
    /// cumulants are dropped. The metric is always evaluated exactly.
    pub max_cumulant: usize,
    /// Eigenvalue threshold of the FOIS metric; directions below it are discarded. The default
    /// keeps the canonical orthogonaliser's singular values `\lambda^{-1/2}` below 5, the pruning of
    /// Lee, Tew and Huynh, since near-null directions amplify the cumulant-truncation error.
    pub fois_tol: f64,
    /// Maximum number of amplitude macro-iterations.
    pub max_macro: usize,
    /// Convergence threshold on the FOIS residual norm `\lVert Y^\dagger R\rVert`.
    pub residual_tol: f64,
    /// Level shift `\eta` added to the orbital-energy denominators.
    pub level_shift: f64,
    /// Whether to solve the holomorphic amplitude equations with complex amplitudes, which
    /// continue the solution onto the complex plane where no real solution exists.
    pub holomorphic: bool,
    /// Linear solver of each Newton step. The dense solver holds intermediates for a batch of
    /// directions at once, so it suits only small active spaces.
    pub newton: NOCCNewton,
    /// Preconditioner of the GMRES Newton solver.
    pub preconditioner: NOCCPreconditioner,
    /// GMRES options of each Newton step of the GMRES solver, which is solved to a tenth of
    /// `\lVert Y^\dagger R\rVert` and never more tightly than `res_tol`.
    pub gmres: GMRESOptions,
}

impl Default for NOCCMCOptions {
    /// Return default NOCCMC options.
    /// # Returns:
    /// - `Self`: NOCCMC options.
    fn default() -> Self {
        Self {
            active_space_tol: 1e-6,
            max_cumulant: 4,
            fois_tol: 4e-2,
            max_macro: 100,
            residual_tol: 1e-8,
            level_shift: 0.5,
            holomorphic: true,
            newton: NOCCNewton::GMRES,
            preconditioner: NOCCPreconditioner::Linear,
            gmres: GMRESOptions {
                max_iter: 200,
                restart: 200,
                res_tol: 1e-10,
            },
        }
    }
}
