// diis.rs
//! Direct inversion in the iterative subspace for SCF acceleration.
//!
//! # References
//!
//! - Hamilton and Pulay, *J. Chem. Phys.* **84**, 5728 (1986),
//!   [doi:10.1063/1.449880](https://doi.org/10.1063/1.449880).

// External crate imports.
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{EighInto, UPLO};
use rayon::prelude::*;

/// DIIS storage for SCF states. Stores the Fock and error matrices of every channel, for
/// example the two spins of one determinant, which are extrapolated with common coefficients.
pub struct Diis {
    /// Size of the DIIS subspace, i.e. the number of past SCF iterations retained.
    m: usize,
    /// History of Fock matrices, one list of channels per iteration.
    f_hist: Vec<Vec<Array2<f64>>>,
    /// History of error matrices, one list of channels per iteration.
    e_hist: Vec<Vec<Array2<f64>>>,
}

impl Diis {
    /// Constructor for DIIS object, initialises object with DIIS space m.
    /// History of Fock and error matrices are initialised as vectors with length m.
    /// # Arguments
    /// - `m`: Size of the DIIS space.
    /// # Returns
    /// - `Diis`: Empty DIIS storage with capacity for `m` previous iterations.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            f_hist: Vec::with_capacity(m),
            e_hist: Vec::with_capacity(m),
        }
    }

    /// `Calculate the orthonormalised DIIS error. R = F^s D^s S - S D^s F^s, then`
    /// `R' = X^T R X where X = S^{-1/2}.`
    /// # Arguments:
    /// - `f`: Spin specific Fock matrix.
    /// - `d`: Spin specific density matrix.
    /// - `s`: AO overlap matrix.
    /// - `x`: Precomputed Loewdin orthgonaliser.
    /// # Returns
    /// - `Array2<f64>`: Orthonormalised DIIS error matrix.
    fn build_error(
        f: &Array2<f64>,
        d: &Array2<f64>,
        s: &Array2<f64>,
        x: &Array2<f64>,
    ) -> Array2<f64> {
        let r = f.dot(d).dot(s) - s.dot(d).dot(f);
        x.t().dot(&r).dot(x)
    }

    /// Add Fock matrices and DIIS error matrices from current SCF cycle to history.
    /// # Arguments:
    /// - `f`: Fock matrix of each channel, e.g. `[F^\alpha, F^\beta]`.
    /// - `d`: Density matrix of each channel, in the same order as `f`.
    /// - `s`: AO overlap matrix.
    /// - `x`: Precomputed Loewdin orthgonaliser.
    /// # Returns
    /// - `()`: Updates the stored Fock and error history in place.
    pub fn push(
        &mut self,
        f: &[&Array2<f64>],
        d: &[&Array2<f64>],
        s: &Array2<f64>,
        x: &Array2<f64>,
    ) {
        let r_prime: Vec<Array2<f64>> = f
            .par_iter()
            .zip(d.par_iter())
            .map(|(f, d)| Self::build_error(f, d, s, x))
            .collect();

        self.f_hist.push(f.iter().map(|&f| f.clone()).collect());
        self.e_hist.push(r_prime);

        // Cap DIIS subspace at size m.
        if self.f_hist.len() > self.m {
            self.f_hist.remove(0);
            self.e_hist.remove(0);
        }
    }

    /// `Calculate squared norm of the most recent errors, \sum_c ||E^c||^2 over channels c.`
    /// # Arguments:
    /// - `self`: Contains DIIS data.
    /// # Returns
    /// - `Option<f64>`: Squared norm of the most recent error matrices of all channels, or
    ///   `None` if no error history is stored.
    pub fn last_error_norm2(&self) -> Option<f64> {
        // If our subspace size is currently zero we cannot calculate an error.
        let e = self.e_hist.last()?;
        Some(e.iter().fold(0.0, |acc, e| acc + (e * e).sum()))
    }

    /// Use past Fock matrices and solutions to augmented linear system to extrapolate.
    /// `Calculates the extrapolated DIIS Fock matrix F_{DIIS}^c = \sum_i^m c_i F_i^c of each channel c.`
    /// # Arguments
    /// - `self`: Contains DIIS data.
    /// # Returns
    /// - `Option<Vec<Array2<f64>>>`: Extrapolated DIIS Fock matrix of each channel, or `None` if
    ///   insufficient or ill-conditioned history prevents extrapolation.
    pub fn extrapolate_fock(&self) -> Option<Vec<Array2<f64>>> {
        let m = self.e_hist.len();

        // With less than 2 cycles of history we cannot extrapolate to DIIS Fock.
        if m < 2 {
            return None;
        }

        // Construct augmented matrix of DIIS residuals.
        //  B_{ij}^c = \sum_{pq}(E^c_i)_{pq}(E^c_j)_{pq} for each channel c.
        //  Main matrix elements are \sum_c B_{ij}^c, summed in channel order from zero,
        //  augmentation applies H[0, 0] = 0, H[0, i] = 1 for all i..m, and
        //  H[i, 0] = 1 for all i..m.
        let mut h = Array2::<f64>::zeros((m + 1, m + 1));
        h.slice_mut(s![0, 1..]).fill(1.0);
        h.slice_mut(s![1.., 0]).fill(1.0);
        for i in 0..m {
            for j in 0..=i {
                let bij = self.e_hist[i]
                    .iter()
                    .zip(&self.e_hist[j])
                    .fold(0.0, |acc, (ei, ej)| acc + (ei * ej).sum());
                h[(i + 1, j + 1)] = bij;
                h[(j + 1, i + 1)] = bij;
            }
        }

        // Form RHS of the Pulay system g = [1, \mathbf{0}]^T. We are solving
        // for [\lambda, c_1, .., c_m] with constraint 1^T c = 1.
        let mut g = Array1::<f64>::zeros(m + 1);
        g[0] = 1.0;

        // Diagonalise augmeneted matrix.
        let (w, v) = h.clone().eigh_into(UPLO::Lower).unwrap();

        // Drop egienvectors with small eigenvalues to keep well conditioned.
        let tol = 1e-13;
        let keep: Vec<usize> = (0..w.len()).filter(|&k| w[k].abs() > tol).collect();
        if keep.is_empty() {
            return None;
        }

        // Calculate coefficients V \Lambda^{-1} V^T g which are eigenvectors of
        // the pseudo-inverse H^{-1} (i.e., not truly inverse due to the dropping
        // of tiny eigenvalues). Doing so reduces linear dependence in B.
        let mut c_aug = Array1::<f64>::zeros(m + 1);
        for &k in &keep {
            let vk = v.slice(s![.., k]);
            let alpha = vk.dot(&g) / w[k];
            for i in 0..c_aug.len() {
                c_aug[i] += vk[i] * alpha;
            }
        }

        // Ignore Lagrange multiplier \lambda.
        let mut c = c_aug.slice(s![1..]).to_owned();

        // Normalise coefficients such that \sum_i c_i = 1.
        let n: f64 = c.sum();
        if n.abs() > 0.0 {
            c.mapv_inplace(|x| x / n);
        }

        // Extrapolate to get DIIS Fock matrix of each channel.
        // F_{DIIS}^c = \sum_i^m c_i F_i^c.
        let f_diis = (0..self.f_hist[0].len())
            .map(|k| {
                let mut f = self.f_hist[0][k].clone() * c[0];
                for i in 1..m {
                    f = f + self.f_hist[i][k].clone() * c[i];
                }
                f
            })
            .collect();
        Some(f_diis)
    }
}
