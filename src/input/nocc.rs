// input/nocc.rs

pub struct NOCCMCOptions {
    /// Natural-occupation tolerance for the active space: orbitals within this distance of two
    /// or zero electrons are core or virtual, and all others active.
    pub active_space_tol: f64,
    /// Highest cumulant rank kept in the energy and residual equations; terms with higher-rank
    /// cumulants are dropped. The metric and zeroth-order coupling are always evaluated exactly.
    pub max_cumulant: usize,
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
    /// Whether to solve the holomorphic amplitude equations with complex amplitudes, which
    /// continue the solution onto the complex plane where no real solution exists.
    pub holomorphic: bool,
}

impl Default for NOCCMCOptions {
    /// Return default NOCCMC options.
    /// # Returns:
    /// - `Self`: NOCCMC options.
    fn default() -> Self {
        Self {
            active_space_tol: 1e-6,
            max_cumulant: 4,
            fois_tol: 1e-8,
            max_macro: 100,
            max_micro: 200,
            residual_tol: 1e-8,
            micro_tol: 1e-10,
            level_shift: 0.5,
            diis_space: 8,
            holomorphic: true,
        }
    }
}
