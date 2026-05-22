// input/scf.rs

pub struct DiisOptions {
    /// Maximum DIIS subspace size.
    pub space: usize,
}

impl Default for DiisOptions {
    /// Return default DIIS options.
    /// # Returns:
    /// - `Self`: DIIS options with default subspace size.
    fn default() -> Self {
        Self { space: 8 }
    }
}

pub struct SCFInfo {
    /// Maximum number of SCF iterations.
    pub max_cycle: i32,
    /// SCF energy convergence tolerance.
    pub e_tol: f64,
    /// FDS-SDF commutator convergence tolerance.
    pub fds_sdf_tol: f64,
    /// Density duplicate and collapse tolerance.
    pub d_tol: f64,
    /// DIIS acceleration options.
    pub diis: DiisOptions,
    /// Whether PySCF should run FCI.
    pub do_fci: bool,
    /// Holomorphic SCF options.
    pub h: HSCFOptions,
}

/// Options controlling holomorphic SCF optimisation.
pub struct HSCFOptions {
    /// Maximum number of h-SCF quasi-Newton iterations.
    pub max_cycle: usize,
    /// Convergence threshold for the occupied-virtual gradient norm.
    pub g_tol: f64,
    /// Threshold for accepting SR1 secant updates.
    pub sr1_tol: f64,
    /// Minimum orbital-energy gap used in energy weighting.
    pub denom_tol: f64,
    /// Maximum occupied-virtual step norm.
    pub max_step: f64,
    /// Maximum number of backtracking line-search steps.
    pub line_steps: usize,
    /// Multiplicative shrink factor for line-search steps.
    pub line_shrink: f64,
    /// Maximum number of stored secant pairs.
    pub history: usize,
}

impl Default for HSCFOptions {
    /// Return default h-SCF quasi-Newton options.
    /// # Returns:
    /// - `Self`: h-SCF quasi-Newton options with conservative convergence settings.
    fn default() -> Self {
        Self {
            max_cycle: 100,
            g_tol: 1e-10,
            sr1_tol: 1e-12,
            denom_tol: 1e-10,
            max_step: 0.5,
            line_steps: 12,
            line_shrink: 0.5,
            history: 20,
        }
    }
}

impl Default for SCFInfo {
    /// Return default SCF options.
    /// # Returns:
    /// - `Self`: SCF options with standard convergence and DIIS settings.
    fn default() -> Self {
        Self {
            max_cycle: 10_000,
            e_tol: 1e-12,
            fds_sdf_tol: 1e-8,
            d_tol: 1e-4,
            diis: DiisOptions::default(),
            do_fci: false,
            h: HSCFOptions::default(),
        }
    }
}
