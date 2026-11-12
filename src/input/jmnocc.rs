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
    /// Deterministic propagation options. When present the populations are propagated exactly.
    pub det: Option<JMDeterministicOptions>,
    /// Stochastic propagation options. When present the populations are sampled by coupled
    /// cluster Monte Carlo with time step `dt`.
    pub qmc: Option<QMCOptions>,
    /// Memory, in report blocks, of the running effective overlap and Hamiltonian from which
    /// stochastic propagation takes `G`; each block is forgotten by `\lambda = 1 - 1/\text{memory}`.
    pub memory_effective: f64,
}

pub struct JMDeterministicOptions {
    /// Maximum number of deterministic propagation steps.
    pub max_steps: usize,
    /// Convergence threshold on the largest residual `\max |R_{x\mu}|`.
    pub residual_tol: f64,
    /// Convergence threshold on the change in projected energy per step.
    pub e_tol: f64,
    /// Whether the shift follows the population,
    /// `E_s \leftarrow E_s - (\alpha/\Delta\tau)\Delta\ln N_w`.
    pub dynamic_shift: bool,
    /// Damping `\alpha` of the dynamic shift.
    pub dynamic_shift_alpha: f64,
    /// Whether to diagnose the Jacobian at the initial and final amplitudes and write the history
    /// of the amplitude error along its right singular vectors. Builds the overlap between every
    /// pair of expansion determinants, so it is meant for small systems.
    pub diagnostics: bool,
    /// Relative cutoff `\epsilon` below which a singular value `\sigma_i < \epsilon\sigma_{\max}`
    /// of the Jacobian is counted as near-null.
    pub null_eps: f64,
    /// Number of lowest Jacobian singular values printed with their external fractions.
    pub null_n: usize,
}

impl Default for JMNOCCMCOptions {
    /// Return default JM-NOCCMC options.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Self`: JM-NOCCMC options with CCSD clusters and no propagation selected.
    fn default() -> Self {
        Self {
            cluster_rank: 2,
            expansion_rank: None,
            dt: 0.1,
            det: None,
            qmc: None,
            memory_effective: 100.0,
        }
    }
}

impl Default for JMDeterministicOptions {
    /// Return default deterministic JM-NOCCMC options.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Self`: Deterministic JM-NOCCMC options with a dynamic shift and no diagnostics.
    fn default() -> Self {
        Self {
            max_steps: 10000,
            residual_tol: 1e-9,
            e_tol: 1e-10,
            dynamic_shift: true,
            dynamic_shift_alpha: 0.1,
            diagnostics: false,
            null_eps: 1e-8,
            null_n: 10,
        }
    }
}
