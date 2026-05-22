// input/snoci.rs

use std::str::FromStr;

pub struct GMRESOptions {
    /// Maximum GMRES iterations.
    pub max_iter: usize,
    /// GMRES residual tolerance.
    pub res_tol: f64,
    /// Metric singular-value tolerance.
    pub metric_tol: f64,
    /// GMRES restart dimension.
    pub restart: usize,
    /// Whether to use full projected operator.
    pub full_m: bool,
}

impl Default for GMRESOptions {
    /// Return default GMRES options.
    /// # Returns:
    /// - `Self`: GMRES options with default iteration limit and residual tolerance.
    fn default() -> Self {
        Self {
            max_iter: 100,
            res_tol: 1e-8,
            metric_tol: 1e-8,
            restart: 200,
            full_m: false,
        }
    }
}

#[derive(Clone, Copy)]
pub enum SNOCIPreconditioner {
    Diag,
    Woodbury,
}

impl SNOCIPreconditioner {
    /// Return SNOCI preconditioner as input string.
    /// # Returns:
    /// - `&'static str`: String representation used in input parsing.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Diag => "diag",
            Self::Woodbury => "woodbury",
        }
    }
}

impl FromStr for SNOCIPreconditioner {
    type Err = String;

    /// Parse SNOCI preconditioner from input string.
    /// # Arguments:
    /// - `s`: String specifying the SNOCI preconditioner.
    /// # Returns:
    /// - `Result`: Parsed preconditioner if valid string, otherwise error message.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "diag" => Ok(Self::Diag),
            "woodbury" => Ok(Self::Woodbury),
            _ => Err(format!("invalid SNOCI preconditioner: {s}")),
        }
    }
}

impl Default for SNOCIPreconditioner {
    /// Return default SNOCI preconditioner.
    /// # Returns:
    /// - `Self`: Default SNOCI preconditioner.
    fn default() -> Self {
        Self::Woodbury
    }
}

pub struct SNOCIOptions {
    /// Selection denominator shift.
    pub sigma: f64,
    /// Candidate residual tolerance.
    pub tol: f64,
    /// Imaginary shifts for complex SNOCI.
    pub imag_shifts: Vec<f64>,
    /// Maximum SNOCI macro-iterations.
    pub max_iter: usize,
    /// Maximum determinants added per iteration.
    pub max_add: usize,
    /// Maximum adaptive-space dimension.
    pub max_dim: usize,
    /// Linear-solve preconditioner.
    pub preconditioner: SNOCIPreconditioner,
    /// Inner GMRES options.
    pub gmres: GMRESOptions,
}

impl Default for SNOCIOptions {
    /// Return default SNOCI options.
    /// # Returns:
    /// - `Self`: SNOCI options with default selection and GMRES parameters.
    fn default() -> Self {
        Self {
            sigma: 1e-6,
            tol: 1e-8,
            imag_shifts: vec![0.0],
            max_iter: 100,
            max_add: 1,
            max_dim: 100,
            preconditioner: SNOCIPreconditioner::default(),
            gmres: GMRESOptions::default(),
        }
    }
}
