// input/jmnocc.rs

// Standard library imports.
use std::str::FromStr;

// Parent/sibling imports.
use super::qmc::QMCOptions;

/// Amplitude update of the deterministic JM-NOCCMC propagation.
#[derive(Clone, Copy)]
pub enum JMPropagator {
    /// Plain imaginary-time step `t_{y\mu} \leftarrow t_{y\mu} - \Delta\tau R_{y\mu}`, the update
    /// sampled by stochastic JM-NOCCMC.
    Plain,
    /// Damped Jacobi step `t_{y\mu} \leftarrow t_{y\mu} - \alpha R_{y\mu} / D_{y\mu}` with the
    /// diagonal denominators `D_{y\mu} = H_{\mu\mu} - H_{yy}` of each reference.
    Diagonal,
}

impl JMPropagator {
    /// Return the propagator as input string.
    /// # Arguments:
    /// - `self`: Propagator selection.
    /// # Returns:
    /// - `&'static str`: String representation used in input parsing.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Diagonal => "diagonal",
        }
    }
}

impl FromStr for JMPropagator {
    type Err = String;

    /// Parse the propagator from input string.
    /// # Arguments:
    /// - `s`: String specifying the propagator.
    /// # Returns:
    /// - `Result`: Parsed propagator if valid string, otherwise error message.
    /// # Errors
    /// - Returns an error if `s` does not name a supported propagator.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "plain" => Ok(Self::Plain),
            "diagonal" => Ok(Self::Diagonal),
            _ => Err(format!("invalid JM-NOCCMC propagator: {s}")),
        }
    }
}

pub struct JMNOCCMCOptions {
    /// Highest excitation rank of each reference's cluster operator, `2` for CCSD.
    pub cluster_rank: usize,
    /// Highest excitation rank of the expansion of each `e^{T_x}|x\rangle`, or `None` for every
    /// rank. The single-reference terms need `cluster_rank + 2`, so lower values are rejected;
    /// higher values truncate only the overlaps between different references, which is exact
    /// only for the complete expansion. It plays the role of the largest cluster size sampled
    /// by stochastic propagation.
    pub expansion_rank: Option<usize>,
    /// Amplitude update.
    pub propagator: JMPropagator,
    /// Imaginary-time step of the plain propagator.
    pub dt: f64,
    /// Damping of the diagonal propagator.
    pub damping: f64,
    /// Maximum number of propagation steps.
    pub max_steps: usize,
    /// Convergence threshold on the largest residual `\max |R_{y\mu}|`.
    pub residual_tol: f64,
    /// Whether to report the stability of the linearised update at convergence.
    pub diagnostics: bool,
    /// Stochastic propagation options. When present the plain update is sampled by coupled
    /// cluster Monte Carlo with time step `dt` instead of being propagated deterministically.
    pub qmc: Option<QMCOptions>,
    /// Memory, in report blocks, of the running effective overlap and Hamiltonian from which
    /// stochastic propagation takes `G`; each block is forgotten by `\lambda = 1 - 1/\text{memory}`.
    pub memory_effective: f64,
}

impl Default for JMNOCCMCOptions {
    /// Return default JM-NOCCMC options.
    /// # Returns:
    /// - `Self`: JM-NOCCMC options with CCSD clusters and the plain propagator.
    fn default() -> Self {
        Self {
            cluster_rank: 2,
            expansion_rank: None,
            propagator: JMPropagator::Plain,
            dt: 0.1,
            damping: 0.5,
            max_steps: 10000,
            residual_tol: 1e-9,
            diagnostics: false,
            qmc: None,
            memory_effective: 100.0,
        }
    }
}
