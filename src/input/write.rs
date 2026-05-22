// input/write.rs

pub struct WriteOptions {
    /// Whether to print verbose progress.
    pub verbose: bool,
    /// Whether to write deterministic coefficients.
    pub write_deterministic_coeffs: bool,
    /// Whether to write orbital coefficients.
    pub write_orbitals: bool,
    /// Whether to write excitation history.
    pub write_excitation_hist: bool,
    /// Whether to write matrix data.
    pub write_matrices: bool,
    /// Output directory.
    pub write_dir: String,
    /// Optional restart output path.
    pub write_restart: Option<String>,
    /// Optional restart input path.
    pub read_restart: Option<String>,
}

impl Default for WriteOptions {
    /// Return default output options.
    /// # Returns:
    /// - `Self`: Output options with all optional writes disabled.
    fn default() -> Self {
        Self {
            verbose: true,
            write_deterministic_coeffs: false,
            write_orbitals: false,
            write_excitation_hist: false,
            write_matrices: false,
            write_dir: "outputs/".to_string(),
            write_restart: None,
            read_restart: None,
        }
    }
}
