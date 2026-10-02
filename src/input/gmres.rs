// input/gmres.rs

pub struct GMRESOptions {
    /// Maximum GMRES iterations.
    pub max_iter: usize,
    /// GMRES restart dimension.
    pub restart: usize,
    /// GMRES residual RMS tolerance.
    pub res_tol: f64,
}

impl Default for GMRESOptions {
    /// Return default GMRES options.
    /// # Returns:
    /// - `Self`: GMRES options with default iteration limit, restart and residual tolerance.
    fn default() -> Self {
        Self {
            max_iter: 100,
            restart: 200,
            res_tol: 1e-8,
        }
    }
}
