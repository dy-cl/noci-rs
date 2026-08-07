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
        }
    }
}
