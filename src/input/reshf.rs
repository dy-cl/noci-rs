// input/reshf.rs

/// Options of resonating Hartree-Fock (ResHF) reference optimisation.
pub struct ResHFOptions {
    /// State weights `w_I` of `E_{\text{SA}} = \sum_I w_I E_I`, lowest root first.
    pub weights: Vec<f64>,
    /// Relative floor on the singular values of each occupied orbital overlap matrix
    /// (ResHF-cutoff).
    pub cutoff: f64,
}

impl Default for ResHFOptions {
    /// Return default ResHF options: the ground state alone and a floor of `10^{-8}`.
    /// # Returns:
    /// - `Self`: Default ResHF options.
    fn default() -> Self {
        Self {
            weights: vec![1.0],
            cutoff: 1e-8,
        }
    }
}
