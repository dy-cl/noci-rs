// input/det.rs

pub struct DeterministicOptions {
    /// Maximum deterministic propagation steps.
    pub max_steps: usize,
    /// Whether to use dynamic shift.
    pub dynamic_shift: bool,
    /// Dynamic-shift damping factor.
    pub dynamic_shift_alpha: f64,
    /// Deterministic energy convergence tolerance.
    pub e_tol: f64,
    /// Relative cutoff for deterministic projector overlap eigenvalues.
    pub projector_eps: f64,
    /// Number of dominant canonical states to print.
    pub canonical_states_n: usize,
    /// Number of original-basis terms to print for each canonical state.
    pub canonical_terms_m: usize,
}

impl Default for DeterministicOptions {
    /// Return default deterministic propagation options.
    /// # Returns:
    /// - `Self`: Deterministic propagation options with dynamic shift enabled.
    fn default() -> Self {
        Self {
            max_steps: 10000,
            dynamic_shift: true,
            dynamic_shift_alpha: 0.1,
            e_tol: 1e-10,
            projector_eps: 1e-12,
            canonical_states_n: 10,
            canonical_terms_m: 10,
        }
    }
}
