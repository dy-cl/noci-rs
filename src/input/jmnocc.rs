// input/jmnocc.rs

// Parent/sibling imports.
use super::qmc::QMCOptions;

pub struct JMNOCCMCOptions {
    /// Highest excitation rank of each reference's cluster operator, `2` for CCSD.
    pub cluster_rank: usize,
    /// Highest excitation rank of the expansion of each `e^{T_x}|x\rangle`, or `None` for every
    /// rank. The single-reference terms need `cluster_rank + 2`, so lower values are rejected;
    /// higher values truncate only the overlaps between different references, which is exact
    /// only for the complete expansion. It plays the role of the largest cluster size sampled
    /// by stochastic propagation.
    pub expansion_rank: Option<usize>,
    /// Imaginary-time step of the deterministic and stochastic propagations.
    pub dt: f64,
    /// Maximum number of deterministic propagation steps.
    pub max_steps: usize,
    /// Deterministic convergence threshold on the largest residual `\max |R_{x\mu}|`.
    pub residual_tol: f64,
    /// Deterministic convergence threshold on the change in projected energy per step.
    pub e_tol: f64,
    /// Whether the deterministic shift follows the population,
    /// `E_s \leftarrow E_s - (\alpha/\Delta\tau)\Delta\ln N_w`.
    pub dynamic_shift: bool,
    /// Damping `\alpha` of the deterministic dynamic shift.
    pub dynamic_shift_alpha: f64,
    /// Whether to report the stability of the linearised update at convergence.
    pub diagnostics: bool,
    /// Stochastic propagation options. When present the populations are sampled by coupled
    /// cluster Monte Carlo with time step `dt` instead of being propagated deterministically.
    pub qmc: Option<QMCOptions>,
    /// Memory, in report blocks, of the running effective overlap and Hamiltonian from which
    /// stochastic propagation takes `G`; each block is forgotten by `\lambda = 1 - 1/\text{memory}`.
    pub memory_effective: f64,
}

impl Default for JMNOCCMCOptions {
    /// Return default JM-NOCCMC options.
    /// # Returns:
    /// - `Self`: JM-NOCCMC options with CCSD clusters.
    fn default() -> Self {
        Self {
            cluster_rank: 2,
            expansion_rank: None,
            dt: 0.1,
            max_steps: 10000,
            residual_tol: 1e-9,
            e_tol: 1e-10,
            dynamic_shift: true,
            dynamic_shift_alpha: 0.1,
            diagnostics: false,
            qmc: None,
            memory_effective: 100.0,
        }
    }
}
