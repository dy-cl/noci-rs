// input/nocc.rs

// Parent/sibling imports.
use super::gmres::GMRESOptions;

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
    /// GMRES options of each Newton step, which is solved to a tenth of
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
            gmres: GMRESOptions {
                max_iter: 200,
                restart: 200,
                res_tol: 1e-10,
            },
        }
    }
}
