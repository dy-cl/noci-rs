// maths/gmres.rs
//! Restarted right-preconditioned GMRES for general linear systems.
//!
//! # References
//!
//! - Saad and Schultz, *SIAM J. Sci. Stat. Comput.* **7**, 856 (1986),
//!   [doi:10.1137/0907058](https://doi.org/10.1137/0907058).

#[cfg(feature = "gpu")]
mod device;

// Crate-visible type re-exports.
#[cfg(feature = "gpu")]
pub(crate) use device::DeviceKrylov;

// Standard library imports.
use std::time::Instant;

// External crate imports.
use ndarray::{Array1, Array2, Axis, s};
use num_complex::Complex64;
use rayon::prelude::*;

// Crate-root imports.
use crate::NOCIScalar;

const SMALL: f64 = 1e-14_f64;
const PRINT_STRIDE: usize = 1usize;
/// Vector entries per Rayon task. Fixed, so every reduction sums in the same order whatever the
/// thread count.
const CHUNK: usize = 4096;

/// Result of a GMRES linear solve.
pub struct GMRESResult<V> {
    /// Approximate solution vector.
    pub x: V,
    /// Root-mean-square residual norm.
    pub residual_rms: f64,
    /// Number of GMRES iterations performed.
    pub iterations: usize,
    /// Whether the residual reached the requested tolerance.
    pub converged: bool,
}

/// Vector storage and arithmetic on which GMRES runs: host arrays, or vectors resident on a GPU.
/// Every operation GMRES performs on a vector of the system's length goes through this trait;
/// the small Hessenberg least-squares problem always stays on the host.
pub trait KrylovSpace<T: NOCIScalar> {
    /// Vector of the system's length.
    type Vector;
    /// Orthonormal Arnoldi basis of one restart cycle.
    type Basis;

    /// Zero vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `n`: Vector length.
    /// # Returns:
    /// - `Self::Vector`: Zero vector of length `n`.
    fn zeros(
        &self,
        n: usize,
    ) -> Self::Vector;

    /// Length of a vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Vector.
    /// # Returns:
    /// - `usize`: Number of entries of `v`.
    fn length(
        &self,
        v: &Self::Vector,
    ) -> usize;

    /// Copy of a vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Vector to copy.
    /// # Returns:
    /// - `Self::Vector`: Independent copy of `v`.
    fn copy(
        &self,
        v: &Self::Vector,
    ) -> Self::Vector;

    /// Conjugated inner product `x^\dagger y`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `x`: Left vector.
    /// - `y`: Right vector.
    /// # Returns:
    /// - `T`: Inner product `x^\dagger y`.
    fn inner_product(
        &self,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> T;

    /// Euclidean norm.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Vector to norm.
    /// # Returns:
    /// - `f64`: Euclidean norm of `v`.
    fn vector_norm(
        &self,
        v: &Self::Vector,
    ) -> f64;

    /// Scale a vector in place, `v \leftarrow a v`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `a`: Scale factor.
    /// - `v`: Vector to scale.
    /// # Returns:
    /// - `()`: Scales `v` in place.
    fn scale(
        &self,
        a: T,
        v: &mut Self::Vector,
    );

    /// Add a scaled vector in place, `y \leftarrow y + a x`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `a`: Scale factor.
    /// - `x`: Vector to add.
    /// - `y`: Vector to update.
    /// # Returns:
    /// - `()`: Updates `y` in place.
    fn scaled_add(
        &self,
        a: T,
        x: &Self::Vector,
        y: &mut Self::Vector,
    );

    /// Start the Arnoldi basis of a restart cycle with `q_0 = r / \beta`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `r`: Residual at the start of the cycle.
    /// - `beta`: Norm of `r`.
    /// - `capacity`: Largest number of basis vectors the cycle can hold.
    /// # Returns:
    /// - `Self::Basis`: Basis holding `q_0`.
    fn basis(
        &self,
        r: &Self::Vector,
        beta: f64,
        capacity: usize,
    ) -> Self::Basis;

    /// Normalise an orthogonalised Arnoldi vector and append it to the basis,
    /// `q_{k+1} = w / \lVert w\rVert`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `w`: Orthogonalised Arnoldi vector.
    /// - `norm`: Norm of `w`.
    /// # Returns:
    /// - `()`: Appends `q_{k+1}` to `q`.
    fn extend(
        &self,
        q: &mut Self::Basis,
        w: Self::Vector,
        norm: f64,
    );

    /// Apply the right preconditioner to one basis vector, `z_k = M^{-1} q_k`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `k`: Index of the basis vector.
    /// - `precondition`: Right preconditioner `M^{-1}`.
    /// # Returns:
    /// - `Self::Vector`: Preconditioned vector `z_k`.
    fn precondition<P>(
        &self,
        q: &Self::Basis,
        k: usize,
        precondition: &P,
    ) -> Self::Vector
    where
        P: Fn(&Self::Vector) -> Self::Vector;

    /// Orthogonalise an Arnoldi vector against the first `k + 1` basis vectors by classical
    /// Gram-Schmidt applied twice, `h = Q^\dagger w`, `w \leftarrow w - Q h`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `w`: Arnoldi vector to orthogonalise in place.
    /// - `k`: Current Arnoldi iteration in the restart cycle.
    /// # Returns:
    /// - `Vec<T>`: Hessenberg column `h_{jk}`, `j \le k`, the sum of both passes.
    fn orthogonalise(
        &self,
        q: &Self::Basis,
        w: &mut Self::Vector,
        k: usize,
    ) -> Vec<T>;

    /// Combine basis vectors, `u = \sum_j y_j q_j`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `y`: Expansion coefficients, one per leading basis vector.
    /// - `n`: Vector length.
    /// # Returns:
    /// - `Self::Vector`: Combination `u`.
    fn combine(
        &self,
        q: &Self::Basis,
        y: &[T],
        n: usize,
    ) -> Self::Vector;
}

/// Host vectors `Array1<T>`, with every reduction summed over fixed chunks in parallel.
pub struct HostKrylov;

/// Storage for a single restarted Arnoldi cycle. The preconditioned directions
/// `z_j = M^{-1} q_j` are not stored: the preconditioner is linear, so a correction
/// `\sum_j y_j z_j` is formed as `M^{-1} \sum_j y_j q_j`.
struct ArnoldiCycle<T: NOCIScalar, S: KrylovSpace<T>> {
    /// Orthonormal Arnoldi basis `q_j`.
    q: S::Basis,
    /// Upper Hessenberg matrix after Givens rotations.
    h: Array2<T>,
    /// Rotated residual right-hand side.
    g: Array1<T>,
    /// Number of Arnoldi iterations completed in the current cycle.
    kfinal: usize,
}

/// Parameters for a single restarted Arnoldi cycle.
struct ArnoldiParams<'a> {
    /// Maximum number of Arnoldi iterations in this restart cycle.
    inner_max: usize,
    /// GMRES restart cycle index.
    restart_id: usize,
    /// Total number of GMRES iterations before this cycle.
    total_iter: usize,
    /// Square-root of the vector length.
    rms: f64,
    /// Wall-time for GMRES.
    gmres_start: &'a Instant,
}

/// Iteration table printed by `gmres`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GMRESPrint {
    /// No output.
    Silent,
    /// Residual estimates, true residuals and timings.
    Residual,
    /// As `Residual`, with the Hylleraas functional `-\Re(b^\dagger x) - \Re(x^\dagger r)` of the
    /// solution at every restart, where the true residual `r` is at hand.
    Hylleraas,
}

/// Print the GMRES iteration table header.
/// # Arguments:
/// - `print`: Table layout.
/// # Returns:
/// - `()`: Prints the GMRES iteration header to standard output.
fn print_gmres_header(print: GMRESPrint) {
    println!();
    println!("  GMRES solve");
    if print == GMRESPrint::Hylleraas {
        println!("  {}", "-".repeat(148));
        println!(
            "  {:>8} {:>8} {:>16} {:>16} {:>18} {:>18} {:>16} {:>16}",
            "restart",
            "iter",
            "Res (est.)",
            "Res (true)",
            "E2 (Hyl)",
            "dE (Hyl-Proj)",
            "Apply / s",
            "Elapsed / s"
        );
    } else {
        println!("  {}", "-".repeat(110));
        println!(
            "  {:>8} {:>8} {:>16} {:>16} {:>16} {:>16}",
            "restart", "iter", "Res (est.)", "Res (true)", "Apply / s", "Elapsed / s"
        );
    }
}

/// Print a single GMRES iteration summary line.
/// # Arguments:
/// - `print`: Table layout.
/// - `restart_id`: GMRES restart cycle index.
/// - `iter`: Total GMRES iteration index.
/// - `residual_est`: Arnoldi/Givens residual estimate for the current Krylov solve.
/// - `apply_secs`: Time spent applying the matrix-free operator on this iteration.
/// - `elapsed_secs`: Total elapsed GMRES wall time.
/// # Returns:
/// - `()`: Prints the GMRES iteration summary to standard output.
fn print_gmres_iteration(
    print: GMRESPrint,
    restart_id: usize,
    iter: usize,
    residual_est: f64,
    apply_secs: f64,
    elapsed_secs: f64,
) {
    if print == GMRESPrint::Hylleraas {
        println!(
            "  {:>8} {:>8} {:>16.8e} {:>16} {:>18} {:>18} {:>16.6} {:>16.6}",
            restart_id, iter, residual_est, "-", "-", "-", apply_secs, elapsed_secs
        );
    } else {
        println!(
            "  {:>8} {:>8} {:>16.8e} {:>16} {:>16.6} {:>16.6}",
            restart_id, iter, residual_est, "-", apply_secs, elapsed_secs
        );
    }
}

/// Print a single GMRES restart summary line using the true residual, with the Hylleraas
/// functional of the solution in the `Hylleraas` layout.
/// # Arguments:
/// - `space`: Vector space of the solve.
/// - `print`: Table layout.
/// - `restart_id`: GMRES restart cycle index.
/// - `iter`: Total GMRES iteration index after the restart.
/// - `system`: Right-hand side `b`, solution `x` and its true residual `r = b - A x`.
/// - `rms`: Square-root of the vector length.
/// - `elapsed_secs`: Total elapsed GMRES wall time.
/// # Returns:
/// - `()`: Prints the GMRES restart summary to standard output.
fn print_gmres_restart_summary<S, T>(
    space: &S,
    print: GMRESPrint,
    restart_id: usize,
    iter: usize,
    system: (&S::Vector, &S::Vector, &S::Vector),
    rms: f64,
    elapsed_secs: f64,
) where
    S: KrylovSpace<T>,
    T: NOCIScalar + Into<Complex64>,
{
    let (b, x, r) = system;
    let residual_true = space.vector_norm(r) / rms;
    match print {
        GMRESPrint::Silent => {}
        GMRESPrint::Residual => println!(
            "  {:>8} {:>8} {:>16} {:>16.8e} {:>16} {:>16.6}",
            restart_id, iter, "-", residual_true, "-", elapsed_secs
        ),
        GMRESPrint::Hylleraas => {
            // `E_{2,\text{Hyl}} = -\Re(b^\dagger x) - \Re(x^\dagger r)`; the residual term is the
            // difference from the projected second-order energy.
            let b_dot_x: Complex64 = space.inner_product(b, x).into();
            let x_dot_r: Complex64 = space.inner_product(x, r).into();
            let delta_hyl_proj = -x_dot_r.re;
            println!(
                "  {:>8} {:>8} {:>16} {:>16.8e} {:>18.10e} {:>18.10e} {:>16} {:>16.6}",
                restart_id,
                iter,
                "-",
                residual_true,
                -b_dot_x.re + delta_hyl_proj,
                delta_hyl_proj,
                "-",
                elapsed_secs
            );
        }
    }
}

/// Build the true residual `b - A x`.
/// # Arguments:
/// - `space`: Vector space of the solve.
/// - `apply`: Matrix-vector product callback.
/// - `b`: Right-hand side vector.
/// - `x`: Current solution vector.
/// # Returns:
/// - `S::Vector`: True residual vector.
fn true_residual<S, F, T>(
    space: &S,
    apply: &mut F,
    b: &S::Vector,
    x: &S::Vector,
) -> S::Vector
where
    S: KrylovSpace<T>,
    F: FnMut(&S::Vector) -> S::Vector,
    T: NOCIScalar,
{
    let mut r = apply(x);
    space.scale(T::from_real(-1.0), &mut r);
    space.scaled_add(T::from_real(1.0), b, &mut r);
    r
}

impl<T: NOCIScalar> KrylovSpace<T> for HostKrylov {
    type Vector = Array1<T>;
    type Basis = Vec<Array1<T>>;

    /// Zero vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `n`: Vector length.
    /// # Returns:
    /// - `Array1<T>`: Zero vector of length `n`.
    fn zeros(
        &self,
        n: usize,
    ) -> Array1<T> {
        Array1::from_elem(n, T::from_real(0.0))
    }

    /// Length of a vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Vector.
    /// # Returns:
    /// - `usize`: Number of entries of `v`.
    fn length(
        &self,
        v: &Array1<T>,
    ) -> usize {
        v.len()
    }

    /// Copy of a vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Vector to copy.
    /// # Returns:
    /// - `Array1<T>`: Independent copy of `v`.
    fn copy(
        &self,
        v: &Array1<T>,
    ) -> Array1<T> {
        v.clone()
    }

    /// Conjugated inner product `x^\dagger y`, summing fixed chunks in parallel and the chunk sums in order.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `x`: Left vector.
    /// - `y`: Right vector.
    /// # Returns:
    /// - `T`: Inner product `x^\dagger y`.
    /// # Panics
    /// - If either vector is not contiguous, which no GMRES vector is.
    fn inner_product(
        &self,
        x: &Array1<T>,
        y: &Array1<T>,
    ) -> T {
        let zero = T::from_real(0.0);

        // `\sum_c \sum_{i \in c} x_i^* y_i`, one partial sum per fixed chunk `c`.
        let parts = x
            .as_slice()
            .unwrap()
            .par_chunks(CHUNK)
            .zip(y.as_slice().unwrap().par_chunks(CHUNK))
            .map(|(xc, yc)| {
                xc.iter()
                    .zip(yc)
                    .fold(zero, |acc, (&xi, &yi)| acc + xi.conj() * yi)
            })
            .collect::<Vec<_>>();

        parts.into_iter().fold(zero, |acc, part| acc + part)
    }

    /// Euclidean norm, summing fixed chunks in parallel and the chunk sums in order.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Vector to norm.
    /// # Returns:
    /// - `f64`: Euclidean norm of `v`.
    /// # Panics
    /// - If `v` is not contiguous, which no GMRES vector is.
    fn vector_norm(
        &self,
        v: &Array1<T>,
    ) -> f64 {
        // `\lVert v\rVert^2 = \sum_c \sum_{i \in c} |v_i|^2`, one partial sum per fixed chunk `c`.
        let parts = v
            .as_slice()
            .unwrap()
            .par_chunks(CHUNK)
            .map(|c| c.iter().map(|&x| x.abs() * x.abs()).sum::<f64>())
            .collect::<Vec<_>>();

        parts.into_iter().sum::<f64>().sqrt()
    }

    /// Scale a vector in place, `v \leftarrow a v`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `a`: Scale factor.
    /// - `v`: Vector to scale.
    /// # Returns:
    /// - `()`: Scales `v` in place.
    fn scale(
        &self,
        a: T,
        v: &mut Array1<T>,
    ) {
        v.par_mapv_inplace(|vi| a * vi);
    }

    /// Add a scaled vector in place, `y \leftarrow y + a x`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `a`: Scale factor.
    /// - `x`: Vector to add.
    /// - `y`: Vector to update.
    /// # Returns:
    /// - `()`: Updates `y` in place.
    fn scaled_add(
        &self,
        a: T,
        x: &Array1<T>,
        y: &mut Array1<T>,
    ) {
        y.scaled_add(a, x);
    }

    /// Start the Arnoldi basis of a restart cycle with `q_0 = r / \beta`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `r`: Residual at the start of the cycle.
    /// - `beta`: Norm of `r`.
    /// - `capacity`: Largest number of basis vectors the cycle can hold.
    /// # Returns:
    /// - `Vec<Array1<T>>`: Basis holding `q_0`.
    fn basis(
        &self,
        r: &Array1<T>,
        beta: f64,
        capacity: usize,
    ) -> Vec<Array1<T>> {
        let mut q = Vec::with_capacity(capacity);
        q.push(r.mapv(|ri| ri / T::from_real(beta)));
        q
    }

    /// Normalise an orthogonalised Arnoldi vector and append it to the basis.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `w`: Orthogonalised Arnoldi vector.
    /// - `norm`: Norm of `w`.
    /// # Returns:
    /// - `()`: Appends `w / norm` to `q`.
    fn extend(
        &self,
        q: &mut Vec<Array1<T>>,
        mut w: Array1<T>,
        norm: f64,
    ) {
        w.par_mapv_inplace(|wi| wi / T::from_real(norm));
        q.push(w);
    }

    /// Apply the right preconditioner to one basis vector, `z_k = M^{-1} q_k`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `k`: Index of the basis vector.
    /// - `precondition`: Right preconditioner `M^{-1}`.
    /// # Returns:
    /// - `Array1<T>`: Preconditioned vector `z_k`.
    fn precondition<P>(
        &self,
        q: &Vec<Array1<T>>,
        k: usize,
        precondition: &P,
    ) -> Array1<T>
    where
        P: Fn(&Array1<T>) -> Array1<T>,
    {
        precondition(&q[k])
    }

    /// Orthogonalise an Arnoldi vector against the first `k + 1` basis vectors by classical Gram-Schmidt applied twice; each pass is one sweep for every product and one for the update.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `w`: Arnoldi vector to orthogonalise in place.
    /// - `k`: Current Arnoldi iteration in the restart cycle.
    /// # Returns:
    /// - `Vec<T>`: Hessenberg column `h_{jk}`, `j \le k`, the sum of both passes.
    fn orthogonalise(
        &self,
        q: &Vec<Array1<T>>,
        w: &mut Array1<T>,
        k: usize,
    ) -> Vec<T> {
        let mut h = vec![T::from_real(0.0); k + 1];

        for _ in 0..2 {
            // `h = Q^\dagger w` in one sweep over fixed chunks of `w`, each read once against the
            // same chunk of every basis vector, `p_{cj} = (\sum_{i \in c} q_{ji} w_i^*)^*`.
            let parts = w
                .axis_chunks_iter(Axis(0), CHUNK)
                .into_par_iter()
                .enumerate()
                .map(|(c, wc)| {
                    let start = c * CHUNK;
                    let wc = wc.mapv(|wi| wi.conj());
                    q[..=k]
                        .iter()
                        .map(|qj| qj.slice(s![start..start + wc.len()]).dot(&wc).conj())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            // `h_j = \sum_c p_{cj}`, summed in chunk order whatever the thread count.
            let mut hk = vec![T::from_real(0.0); k + 1];
            for part in parts {
                for (hj, pj) in hk.iter_mut().zip(part) {
                    *hj += pj;
                }
            }

            // `w \leftarrow w - Q h` in a second sweep, each chunk of `w` updated in place by the
            // same chunk of every basis vector.
            w.axis_chunks_iter_mut(Axis(0), CHUNK)
                .into_par_iter()
                .enumerate()
                .for_each(|(c, mut wc)| {
                    let start = c * CHUNK;
                    for (qj, &hj) in q[..=k].iter().zip(&hk) {
                        wc.scaled_add(-hj, &qj.slice(s![start..start + wc.len()]));
                    }
                });

            for (hj, &pass) in h.iter_mut().zip(&hk) {
                *hj += pass;
            }
        }

        h
    }

    /// Combine basis vectors, `u = \sum_j y_j q_j`; each fixed chunk of `u` accumulates the same chunk of every basis vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `y`: Expansion coefficients, one per leading basis vector.
    /// - `n`: Vector length.
    /// # Returns:
    /// - `Array1<T>`: Combination `u`.
    fn combine(
        &self,
        q: &Vec<Array1<T>>,
        y: &[T],
        n: usize,
    ) -> Array1<T> {
        let mut u = Array1::from_elem(n, T::from_real(0.0));
        u.axis_chunks_iter_mut(Axis(0), CHUNK)
            .into_par_iter()
            .enumerate()
            .for_each(|(c, mut uc)| {
                let start = c * CHUNK;
                for (qj, &yj) in q.iter().zip(y) {
                    uc.scaled_add(yj, &qj.slice(s![start..start + uc.len()]));
                }
            });
        u
    }
}

/// Apply all previous Givens rotations to the current Hessenberg column.
/// # Arguments:
/// - `h`: Hessenberg matrix to update.
/// - `cs`: Cosines of previous Givens rotations.
/// - `sn`: Sines of previous Givens rotations.
/// - `k`: Current Arnoldi iteration in the restart cycle.
/// # Returns:
/// - `()`: Updates the current column of `h` in place.
fn apply_previous_givens<T: NOCIScalar>(
    h: &mut Array2<T>,
    cs: &[f64],
    sn: &[T],
    k: usize,
) {
    for j in 0..k {
        let csj = T::from_real(cs[j]);
        let temp = csj * h[(j, k)] + sn[j] * h[(j + 1, k)];
        h[(j + 1, k)] = -sn[j].conj() * h[(j, k)] + csj * h[(j + 1, k)];
        h[(j, k)] = temp;
    }
}

/// Build and apply the next Givens rotation.
/// # Arguments:
/// - `h`: Hessenberg matrix to update.
/// - `cs`: Cosines of Givens rotations.
/// - `sn`: Sines of Givens rotations.
/// - `g`: Rotated residual right-hand side.
/// - `k`: Current Arnoldi iteration in the restart cycle.
/// # Returns:
/// - `()`: Updates `h`, `cs`, `sn`, and `g` in place.
fn apply_current_givens<T: NOCIScalar>(
    h: &mut Array2<T>,
    cs: &mut [f64],
    sn: &mut [T],
    g: &mut Array1<T>,
    k: usize,
) {
    let x = h[(k, k)];
    let y = h[(k + 1, k)];
    let ax = x.abs();
    let ay = y.abs();
    let denom = (ax * ax + ay * ay).sqrt();

    // Choose the complex Givens rotation that annihilates `H_{k+1,k}`;
    // use the identity when both inputs are numerically zero.
    if denom > SMALL {
        cs[k] = ax / denom;
        sn[k] = if ax > SMALL {
            T::from_real(cs[k]) * y.conj() / x.conj()
        } else {
            T::from_real(1.0)
        };
    } else {
        cs[k] = 1.0;
        sn[k] = T::from_real(0.0);
    }

    let csk = T::from_real(cs[k]);
    let snk = sn[k];

    // Apply the same unitary 2x2 rotation to the Hessenberg column and
    // residual right-hand side so the least-squares problem stays equivalent.
    let h0 = h[(k, k)];
    let h1 = h[(k + 1, k)];
    h[(k, k)] = csk * h0 + snk * h1;
    h[(k + 1, k)] = -snk.conj() * h0 + csk * h1;

    let g0 = g[k];
    let g1 = g[k + 1];
    g[k] = csk * g0 + snk * g1;
    g[k + 1] = -snk.conj() * g0 + csk * g1;
}

/// Run one restarted Arnoldi cycle for a callback-defined right-preconditioned GMRES operator.
/// # Arguments:
/// - `space`: Vector space of the solve.
/// - `apply`: Matrix-vector product callback.
/// - `precondition`: Right-preconditioner callback.
/// - `rtrue`: True residual at the start of the restart cycle.
/// - `params`: Parameters for the current Arnoldi cycle.
/// - `tol`: Residual RMS tolerance.
/// - `print`: Table layout of each printed Arnoldi iteration.
/// # Returns:
/// - `ArnoldiCycle`: Krylov basis, Hessenberg matrix, rotated residual vector, and final inner iteration count.
fn run_arnoldi_cycle<S, F, P, T>(
    space: &S,
    apply: &mut F,
    precondition: &P,
    rtrue: &S::Vector,
    params: &ArnoldiParams<'_>,
    tol: f64,
    print: GMRESPrint,
) -> ArnoldiCycle<T, S>
where
    S: KrylovSpace<T>,
    F: FnMut(&S::Vector) -> S::Vector,
    P: Fn(&S::Vector) -> S::Vector,
    T: NOCIScalar + Into<Complex64>,
{
    let beta = space.vector_norm(rtrue);
    let mut q = space.basis(rtrue, beta, params.inner_max + 1);

    let mut h_rot =
        Array2::<T>::from_elem((params.inner_max + 1, params.inner_max), T::from_real(0.0));
    let mut cs = vec![0.0; params.inner_max];
    let mut sn = vec![T::from_real(0.0); params.inner_max];
    let mut g = Array1::<T>::from_elem(params.inner_max + 1, T::from_real(0.0));
    g[0] = T::from_real(beta);

    let mut kfinal = 0usize;

    for k in 0..params.inner_max {
        // Apply the right preconditioner first so Arnoldi sees A P^{-1}.
        let z = space.precondition(&q, k, precondition);

        // Apply the expensive matrix-free operator.
        let t_apply = Instant::now();
        let mut w = apply(&z);
        let apply_secs = t_apply.elapsed().as_secs_f64();

        // Orthogonalise `A M^{-1} q_k` against the basis, filling the raw Hessenberg column.
        let column = space.orthogonalise(&q, &mut w, k);
        for (j, &hjk) in column.iter().enumerate() {
            h_rot[(j, k)] += hjk;
        }

        // Normalise the new direction and append it to the Krylov basis if non-zero.
        let h_next = space.vector_norm(&w);
        h_rot[(k + 1, k)] = T::from_real(h_next);
        if h_next > SMALL {
            space.extend(&mut q, w, h_next);
        }

        // Update the small least-squares problem with Givens rotations.
        apply_previous_givens(&mut h_rot, &cs, &sn, k);
        apply_current_givens(&mut h_rot, &mut cs, &mut sn, &mut g, k);

        kfinal = k + 1;

        // Cheap GMRES residual estimate from the rotated least-squares RHS.
        let residual_est = g[k + 1].abs() / params.rms;
        let iter = params.total_iter + k + 1;

        if print != GMRESPrint::Silent
            && (k == 0 || iter.is_multiple_of(PRINT_STRIDE) || residual_est <= tol)
        {
            print_gmres_iteration(
                print,
                params.restart_id,
                iter,
                residual_est,
                apply_secs,
                params.gmres_start.elapsed().as_secs_f64(),
            );
        }

        if residual_est <= tol || h_next <= SMALL {
            break;
        }
    }
    ArnoldiCycle {
        q,
        h: h_rot,
        g,
        kfinal,
    }
}

/// Update the solution vector with the right-preconditioned Krylov correction,
/// `x \leftarrow x + M^{-1} \sum_j y_j q_j`.
/// # Arguments:
/// - `space`: Vector space of the solve.
/// - `x`: Solution vector to update in place.
/// - `q`: Arnoldi basis vectors.
/// - `y`: Krylov expansion coefficients.
/// - `precondition`: Right preconditioner `M^{-1}`.
/// - `n`: Vector length.
/// # Returns:
/// - `()`: Updates `x` in place.
fn update_solution<S, P, T>(
    space: &S,
    x: &mut S::Vector,
    q: &S::Basis,
    y: &Array1<T>,
    precondition: &P,
    n: usize,
) where
    S: KrylovSpace<T>,
    P: Fn(&S::Vector) -> S::Vector,
    T: NOCIScalar,
{
    if y.is_empty() {
        return;
    }

    let u = space.combine(q, y.as_slice().expect("coefficients are contiguous"), n);
    let correction = precondition(&u);
    space.scaled_add(T::from_real(1.0), &correction, x);
}

/// Solve the small upper-triangular least-squares problem after Arnoldi.
/// # Arguments:
/// - `h`: Rotated Hessenberg matrix.
/// - `g`: Rotated residual right-hand side.
/// - `kfinal`: Number of Arnoldi iterations completed in the current cycle.
/// # Returns:
/// - `Array1<T>`: Least-squares coefficients in the Krylov basis.
fn back_solve<T: NOCIScalar>(
    h: &Array2<T>,
    g: &Array1<T>,
    kfinal: usize,
) -> Array1<T> {
    let mut y = Array1::<T>::from_elem(kfinal, T::from_real(0.0));

    // Back-substitute through the rotated upper-triangular Hessenberg block;
    // zero pivots yield zero coefficients for dependent Krylov directions.
    for ii in 0..kfinal {
        let i = kfinal - 1 - ii;
        let mut rhs = g[i];

        for j in (i + 1)..kfinal {
            rhs -= h[(i, j)] * y[j];
        }

        y[i] = if h[(i, i)].abs() > SMALL {
            rhs / h[(i, i)]
        } else {
            T::from_real(0.0)
        };
    }

    y
}

/// Solve a linear system using restarted GMRES with a callback-defined right preconditioner.
/// # Arguments:
/// - `space`: Vector space of the solve, such as `HostKrylov` for host arrays.
/// - `apply`: Matrix-vector product callback.
/// - `precondition`: Linear right preconditioner `M^{-1}`. Linearity lets each cycle keep only
///   the Arnoldi basis and precondition the combined correction once.
/// - `system`: Right-hand side `b` and the initial guess `x_0`, zero when `None`.
/// - `restart`: Krylov dimension of each restart cycle.
/// - `max_iter`: Maximum total number of iterations.
/// - `tol`: Residual RMS tolerance, accepted only on the true residual.
/// - `print`: Iteration table to print, if any.
/// # Returns:
/// - `GMRESResult<S::Vector>`: Approximate solution vector together with final residual RMS,
///   number of iterations performed, and convergence flag.
#[allow(clippy::too_many_arguments)]
pub fn gmres<S, F, P, T>(
    space: &S,
    mut apply: F,
    precondition: P,
    system: (&S::Vector, Option<&S::Vector>),
    restart: usize,
    max_iter: usize,
    tol: f64,
    print: GMRESPrint,
) -> GMRESResult<S::Vector>
where
    S: KrylovSpace<T>,
    F: FnMut(&S::Vector) -> S::Vector,
    P: Fn(&S::Vector) -> S::Vector,
    T: NOCIScalar + Into<Complex64>,
{
    let gmres_start = Instant::now();
    let (b, x0) = system;
    let n = space.length(b);
    let mut x = match x0 {
        Some(x0) => space.copy(x0),
        None => space.zeros(n),
    };

    if print != GMRESPrint::Silent {
        print_gmres_header(print);
    }

    // Empty systems are already solved.
    if n == 0 {
        return GMRESResult {
            x,
            residual_rms: 0.0,
            iterations: 0,
            converged: true,
        };
    }

    let rms = (n as f64).sqrt();

    // The true residual of the initial guess is `b - A x_0`, which is `b` itself for `x_0 = 0`.
    let mut rtrue = match x0 {
        Some(_) => true_residual(space, &mut apply, b, &x),
        None => space.copy(b),
    };
    let mut residual_rms = space.vector_norm(&rtrue) / rms;

    if print != GMRESPrint::Silent {
        print_gmres_restart_summary(
            space,
            print,
            0,
            0,
            (b, &x, &rtrue),
            rms,
            gmres_start.elapsed().as_secs_f64(),
        );
    }

    // Accept the initial guess if it already satisfies the true residual tolerance.
    if residual_rms <= tol {
        return GMRESResult {
            x,
            residual_rms,
            iterations: 0,
            converged: true,
        };
    }

    let mut total_iter = 0usize;
    let mut restart_id = 0usize;

    while total_iter < max_iter {
        let beta = space.vector_norm(&rtrue);

        // Stop if the residual is numerically zero.
        if beta <= SMALL {
            residual_rms = beta / rms;

            if print != GMRESPrint::Silent {
                print_gmres_restart_summary(
                    space,
                    print,
                    restart_id,
                    total_iter,
                    (b, &x, &rtrue),
                    rms,
                    gmres_start.elapsed().as_secs_f64(),
                );
            }

            return GMRESResult {
                x,
                residual_rms,
                iterations: total_iter,
                converged: residual_rms <= tol,
            };
        }

        // Build one Krylov subspace for the right-preconditioned operator A P^{-1}.
        let inner_max = restart.min(max_iter - total_iter);
        let arnoldi_params = ArnoldiParams {
            inner_max,
            restart_id,
            total_iter,
            rms,
            gmres_start: &gmres_start,
        };

        let cycle = run_arnoldi_cycle(
            space,
            &mut apply,
            &precondition,
            &rtrue,
            &arnoldi_params,
            tol,
            print,
        );

        // Solve the small least-squares problem in the Krylov basis.
        let y = back_solve(&cycle.h, &cycle.g, cycle.kfinal);

        // Apply the right-preconditioned Krylov correction.
        update_solution(space, &mut x, &cycle.q, &y, &precondition, n);

        total_iter += cycle.kfinal;

        // Recompute the true residual after each restart as the Arnoldi residual is only an estimate.
        rtrue = true_residual(space, &mut apply, b, &x);
        residual_rms = space.vector_norm(&rtrue) / rms;

        if print != GMRESPrint::Silent {
            print_gmres_restart_summary(
                space,
                print,
                restart_id,
                total_iter,
                (b, &x, &rtrue),
                rms,
                gmres_start.elapsed().as_secs_f64(),
            );
        }

        restart_id += 1;

        // Only the true residual is accepted as final convergence.
        if residual_rms <= tol {
            return GMRESResult {
                x,
                residual_rms,
                iterations: total_iter,
                converged: true,
            };
        }
    }

    GMRESResult {
        x,
        residual_rms,
        iterations: total_iter,
        converged: residual_rms <= tol,
    }
}
