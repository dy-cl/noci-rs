// nocc/stochastic/effective.rs

// External crate imports.
use ndarray::Array2;
use ndarray_linalg::Inverse;

/// Accumulated intermediate-normalised effective overlap `S^e_{zx}=\langle{}^z\Psi|\Phi_x\rangle`
/// and Hamiltonian `H^e_{zx}=\langle{}^z\Psi|\hat H|\Phi_x\rangle`. Each step's samples are
/// divided by that step's reference populations `N_x` before accumulation, so the ratio
/// estimate `G=(S^e)^{-1}H^e` is independent of the population growth.
pub(super) struct Effective {
    /// Accumulated `S^e`.
    pub(super) se: Array2<f64>,
    /// Accumulated `H^e`.
    pub(super) he: Array2<f64>,
}

impl Effective {
    /// Construct zero accumulators for `M` references.
    /// # Arguments:
    /// - `m`: Number of references.
    /// # Returns:
    /// - `Effective`: Zero accumulators.
    pub(super) fn new(m: usize) -> Self {
        Self {
            se: Array2::zeros((m, m)),
            he: Array2::zeros((m, m)),
        }
    }

    /// Add another accumulation.
    /// # Arguments:
    /// - `self`: Accumulators.
    /// - `other`: Accumulation to add.
    /// # Returns:
    /// - `()`: Adds `other` in place.
    pub(super) fn add(
        &mut self,
        other: &Effective,
    ) {
        self.se += &other.se;
        self.he += &other.he;
    }

    /// Effective Hamiltonian `G=(S^e)^{-1}H^e` of the accumulation.
    /// # Arguments:
    /// - `self`: Accumulators.
    /// # Returns:
    /// - `Option<Array2<f64>>`: `G`, or `None` if `S^e` is singular or `G` is not finite.
    pub(super) fn g(&self) -> Option<Array2<f64>> {
        let g = self.se.inv().ok()?.dot(&self.he);
        g.iter().all(|v| v.is_finite()).then_some(g)
    }
}

/// Effective Hamiltonian of the unnormalised correlated references
/// `|\tilde\Phi_x\rangle = N_x|\Phi_x\rangle`,
/// `\tilde G_{xy} = G_{xy}N_y/N_x`, whose row sums `g_x=\sum_y\tilde G_{xy}=(GN)_x/N_x` are the
/// growth rates of the reference populations.
/// # Arguments:
/// - `g`: Intermediate-normalised effective Hamiltonian `G`.
/// - `nref`: Reference populations `N_x`.
/// # Returns:
/// - `Array2<f64>`: Scaled effective Hamiltonian `\tilde G`.
pub(super) fn scaled(
    g: &Array2<f64>,
    nref: &[f64],
) -> Array2<f64> {
    Array2::from_shape_fn(g.dim(), |(x, y)| g[(x, y)] * nref[y] / nref[x])
}
