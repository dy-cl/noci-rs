// input/reshf.rs

/// Determinants whose orbitals ResHF optimises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResHFSpace {
    /// The NOCI reference states alone.
    References,
    /// Every reference and its excitations of the `excit` orders, each as its own determinant.
    Excitations,
}

/// Options of resonating Hartree-Fock (ResHF) reference optimisation.
pub struct ResHFOptions {
    /// State weights `w_I` of `E_{\text{SA}} = \sum_I w_I E_I`, lowest root first.
    pub weights: Vec<f64>,
    /// Relative floor on the singular values of each occupied orbital overlap matrix
    /// (ResHF-cutoff).
    pub cutoff: f64,
    /// Determinants to optimise.
    pub space: ResHFSpace,
}

impl Default for ResHFOptions {
    /// Return default ResHF options: the ground state alone, a floor of `10^{-8}` and the
    /// references only.
    /// # Returns:
    /// - `Self`: Default ResHF options.
    fn default() -> Self {
        Self {
            weights: vec![1.0],
            cutoff: 1e-8,
            space: ResHFSpace::References,
        }
    }
}
