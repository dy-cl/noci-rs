// nocc/stochastic/effective.rs

// External crate imports.
use ndarray::Array2;
use ndarray_linalg::Inverse;

// Crate-root imports.
use crate::nocc::jm_energies;

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

    /// Scale the accumulation, as when forgetting old report blocks.
    /// # Arguments:
    /// - `self`: Accumulators.
    /// - `factor`: Scale factor.
    /// # Returns:
    /// - `()`: Scales both accumulators in place.
    pub(super) fn scale(
        &mut self,
        factor: f64,
    ) {
        self.se *= factor;
        self.he *= factor;
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

/// Smallest number of bins accepted when reading the error plateau of a reblocking.
const MINBINS: usize = 32;

/// Ground-state energy, the lowest real eigenvalue of `G=(S^e)^{-1}H^e`, of an accumulation.
/// # Arguments:
/// - `effective`: Accumulated `S^e` and `H^e`.
/// # Returns:
/// - `Option<f64>`: Ground-state energy, or `None` if `G` is unavailable.
pub(super) fn ground_energy(effective: &Effective) -> Option<f64> {
    effective.g().map(|g| jm_energies(&g)[0].re)
}

/// Ratio estimate of the ground-state energy from report blocks with a reblocked jackknife error.
/// The blocks are merged pairwise into `n` bins `b` at each level; with the total `S` and the
/// leave-one-bin-out energies `E_b = E(S - S_b)`, the jackknife error is
/// `\sigma^2 = \frac{n-1}{n}\sum_b(E_b-\bar E)^2`. The error is taken as the largest over levels with
/// at least `MINBINS` bins, where correlated blocks have become independent.
/// # Arguments:
/// - `blocks`: Report-block accumulations in time order.
/// # Returns:
/// - `Option<(f64, f64)>`: Energy and its error, or `None` with too few blocks.
/// # References
/// - H. Flyvbjerg and H. G. Petersen, *J. Chem. Phys.* **91**, 461 (1989),
///   [doi:10.1063/1.457480](https://doi.org/10.1063/1.457480).
pub(super) fn reblocked_energy(blocks: &[Effective]) -> Option<(f64, f64)> {
    let m = blocks.first()?.se.nrows();
    let mut total = Effective::new(m);
    for block in blocks {
        total.add(block);
    }
    let energy = ground_energy(&total)?;

    // Pairwise merge the blocks level by level and evaluate the jackknife error at each.
    let mut bins = blocks
        .iter()
        .map(|b| Effective {
            se: b.se.clone(),
            he: b.he.clone(),
        })
        .collect::<Vec<_>>();
    let mut error = None::<f64>;
    while bins.len() >= MINBINS {
        let n = bins.len();
        let energies = bins
            .iter()
            .map(|b| {
                ground_energy(&Effective {
                    se: &total.se - &b.se,
                    he: &total.he - &b.he,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let mean = energies.iter().sum::<f64>() / n as f64;
        let variance =
            energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() * (n - 1) as f64 / n as f64;
        error = Some(error.map_or(variance.sqrt(), |e| e.max(variance.sqrt())));

        bins = bins
            .chunks_exact(2)
            .map(|pair| {
                let mut merged = Effective::new(m);
                merged.add(&pair[0]);
                merged.add(&pair[1]);
                merged
            })
            .collect();
    }
    Some((energy, error?))
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
