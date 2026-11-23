// maths/gmres.rs
//! Restarted right-preconditioned GMRES for general linear systems.
//!
//! # References
//!
//! - Saad and Schultz, *SIAM J. Sci. Stat. Comput.* **7**, 856 (1986),
//!   [doi:10.1137/0907058](https://doi.org/10.1137/0907058).

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
pub struct GMRESResult<T: NOCIScalar> {
    /// Approximate solution vector.
    pub x: Array1<T>,
    /// Root-mean-square residual norm.
    pub residual_rms: f64,
    /// Number of GMRES iterations performed.
    pub iterations: usize,
    /// Whether the residual reached the requested tolerance.
    pub converged: bool,
}

/// Storage for a single restarted Arnoldi cycle. The preconditioned directions
/// `z_j = M^{-1} q_j` are not stored: the preconditioner is linear, so a correction
/// `\sum_j y_j z_j` is formed as `M^{-1} \sum_j y_j q_j`.
struct ArnoldiCycle<T: NOCIScalar> {
    /// Orthonormal Arnoldi basis `q_j`.
    q: Vec<Array1<T>>,
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
/// - `print`: Table layout.
/// - `restart_id`: GMRES restart cycle index.
/// - `iter`: Total GMRES iteration index after the restart.
/// - `system`: Right-hand side `b`, solution `x` and its true residual `r = b - A x`.
/// - `rms`: Square-root of the vector length.
/// - `elapsed_secs`: Total elapsed GMRES wall time.
/// # Returns:
/// - `()`: Prints the GMRES restart summary to standard output.
fn print_gmres_restart_summary<T: NOCIScalar + Into<Complex64>>(
    print: GMRESPrint,
    restart_id: usize,
    iter: usize,
    system: (&Array1<T>, &Array1<T>, &Array1<T>),
    rms: f64,
    elapsed_secs: f64,
) {
    let (b, x, r) = system;
    let residual_true = calculate_residual_rms(r, rms);
    match print {
        GMRESPrint::Silent => {}
        GMRESPrint::Residual => println!(
            "  {:>8} {:>8} {:>16} {:>16.8e} {:>16} {:>16.6}",
            restart_id, iter, "-", residual_true, "-", elapsed_secs
        ),
        GMRESPrint::Hylleraas => {
            // `E_{2,\text{Hyl}} = -\Re(b^\dagger x) - \Re(x^\dagger r)`; the residual term is the
            // difference from the projected second-order energy.
            let b_dot_x: Complex64 = inner_product(b, x).into();
            let x_dot_r: Complex64 = inner_product(x, r).into();
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
/// - `apply`: Matrix-vector product callback.
/// - `b`: Right-hand side vector.
/// - `x`: Current solution vector.
/// # Returns:
/// - `Array1<T>`: True residual vector.
fn true_residual<F, T>(
    apply: &mut F,
    b: &Array1<T>,
    x: &Array1<T>,
) -> Array1<T>
where
    F: FnMut(&Array1<T>) -> Array1<T>,
    T: NOCIScalar,
{
    let mut r = apply(x);
    r.par_mapv_inplace(|axi| -axi);
    r.zip_mut_with(b, |ri, &bi| *ri += bi);
    r
}

/// Compute the conjugated inner product of two scalar vectors, `x^\dagger y`, summing fixed
/// chunks in parallel and the chunk sums in order.
/// # Arguments:
/// - `x`: Left vector.
/// - `y`: Right vector.
/// # Returns:
/// - `T`: Inner product `x^H y`.
/// # Panics
/// - Panics if either vector is not contiguous, which no GMRES vector is.
pub fn inner_product<T: NOCIScalar>(
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

/// Compute the Euclidean norm of a scalar vector, summing fixed chunks in parallel and the chunk
/// sums in order.
/// # Arguments:
/// - `v`: Vector to norm.
/// # Returns:
/// - `f64`: Euclidean norm.
/// # Panics
/// - Panics if `v` is not contiguous, which no GMRES vector is.
fn vector_norm<T: NOCIScalar>(v: &Array1<T>) -> f64 {
    // `\lVert v\rVert^2 = \sum_c \sum_{i \in c} |v_i|^2`, one partial sum per fixed chunk `c`.
    let parts = v
        .as_slice()
        .unwrap()
        .par_chunks(CHUNK)
        .map(|c| c.iter().map(|&x| x.abs() * x.abs()).sum::<f64>())
        .collect::<Vec<_>>();

    parts.into_iter().sum::<f64>().sqrt()
}

/// Compute the RMS norm of a residual vector.
/// # Arguments:
/// - `r`: Residual vector.
/// - `rms`: Square-root of the vector length.
/// # Returns:
/// - `f64`: RMS residual norm.
fn calculate_residual_rms<T: NOCIScalar>(
    r: &Array1<T>,
    rms: f64,
) -> f64 {
    vector_norm(r) / rms
}

/// Orthogonalise an Arnoldi vector against the existing Krylov basis by classical Gram-Schmidt
/// applied twice, `h = Q^\dagger w`, `w \leftarrow w - Q h`, accumulating both passes in the same
/// Hessenberg column. Each pass is one sweep for every product and one for the update.
/// # Arguments:
/// - `q`: Existing Krylov basis vectors.
/// - `h`: Hessenberg matrix to update.
/// - `w`: Arnoldi vector to orthogonalise in place.
/// - `k`: Current Arnoldi iteration in the restart cycle.
/// # Returns:
/// - `()`: Updates `h` and `w` in place.
fn orthogonalise_arnoldi_vector<T: NOCIScalar>(
    q: &[Array1<T>],
    h: &mut Array2<T>,
    w: &mut Array1<T>,
    k: usize,
) {
    for _ in 0..2 {
        // `h = Q^\dagger w` in one sweep over fixed chunks of `w`, each read once against the same
        // chunk of every basis vector, `p_{cj} = (\sum_{i \in c} q_{ji} w_i^*)^*`.
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

        // `w \leftarrow w - Q h` in a second sweep, each chunk of `w` updated in place by the same
        // chunk of every basis vector.
        w.axis_chunks_iter_mut(Axis(0), CHUNK)
            .into_par_iter()
            .enumerate()
            .for_each(|(c, mut wc)| {
                let start = c * CHUNK;
                for (qj, &hj) in q[..=k].iter().zip(&hk) {
                    wc.scaled_add(-hj, &qj.slice(s![start..start + wc.len()]));
                }
            });

        for (j, &hjk) in hk.iter().enumerate() {
            h[(j, k)] += hjk;
        }
    }
}

/// Normalise and append a new Arnoldi vector if it is non-zero.
/// # Arguments:
/// - `q`: Krylov basis vectors.
/// - `h`: Hessenberg matrix to update.
/// - `w`: Orthogonalised Arnoldi vector.
/// - `k`: Current Arnoldi iteration in the restart cycle.
/// # Returns:
/// - `f64`: Norm of the candidate next Arnoldi vector.
fn extend_arnoldi_basis<T: NOCIScalar>(
    q: &mut Vec<Array1<T>>,
    h: &mut Array2<T>,
    w: Array1<T>,
    k: usize,
) -> f64 {
    let h_next = vector_norm(&w);
    h[(k + 1, k)] = T::from_real(h_next);

    if h_next > SMALL {
        let mut w = w;
        w.par_mapv_inplace(|wi| wi / T::from_real(h_next));
        q.push(w);
    }

    h_next
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
/// - `apply`: Matrix-vector product callback.
/// - `precondition`: Right-preconditioner callback.
/// - `rtrue`: True residual at the start of the restart cycle.
/// - `params`: Parameters for the current Arnoldi cycle.
/// - `tol`: Residual RMS tolerance.
/// - `print`: Table layout of each printed Arnoldi iteration.
/// # Returns:
/// - `ArnoldiCycle`: Krylov basis, Hessenberg matrix, rotated residual vector, and final inner iteration count.
fn run_arnoldi_cycle<F, P, T>(
    apply: &mut F,
    precondition: &P,
    rtrue: &Array1<T>,
    params: &ArnoldiParams<'_>,
    tol: f64,
    print: GMRESPrint,
) -> ArnoldiCycle<T>
where
    F: FnMut(&Array1<T>) -> Array1<T>,
    P: Fn(&Array1<T>) -> Array1<T>,
    T: NOCIScalar + Into<Complex64>,
{
    let beta = vector_norm(rtrue);

    let mut q: Vec<Array1<T>> = Vec::with_capacity(params.inner_max + 1);
    q.push(rtrue.mapv(|ri| ri / T::from_real(beta)));

    let mut h_rot =
        Array2::<T>::from_elem((params.inner_max + 1, params.inner_max), T::from_real(0.0));
    let mut cs = vec![0.0; params.inner_max];
    let mut sn = vec![T::from_real(0.0); params.inner_max];
    let mut g = Array1::<T>::from_elem(params.inner_max + 1, T::from_real(0.0));
    g[0] = T::from_real(beta);

    let mut kfinal = 0usize;

    for k in 0..params.inner_max {
        // Apply the right preconditioner first so Arnoldi sees A P^{-1}.
        let z = precondition(&q[k]);

        // Apply the expensive matrix-free operator.
        let t_apply = Instant::now();
        let aq = apply(&z);
        let apply_secs = t_apply.elapsed().as_secs_f64();

        let mut w = aq;

        // Orthogonalise `A M^{-1} q_k` against the basis, filling the raw Hessenberg column.
        orthogonalise_arnoldi_vector(&q, &mut h_rot, &mut w, k);

        // Normalise the new direction and append it to the Krylov basis if non-zero.
        let h_next = extend_arnoldi_basis(&mut q, &mut h_rot, w, k);

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
/// - `x`: Solution vector to update in place.
/// - `q`: Arnoldi basis vectors.
/// - `y`: Krylov expansion coefficients.
/// - `precondition`: Right preconditioner `M^{-1}`.
/// # Returns:
/// - `()`: Updates `x` in place.
fn update_solution<T, P>(
    x: &mut Array1<T>,
    q: &[Array1<T>],
    y: &Array1<T>,
    precondition: &P,
) where
    T: NOCIScalar,
    P: Fn(&Array1<T>) -> Array1<T>,
{
    if y.is_empty() {
        return;
    }

    // `u = \sum_j y_j q_j`, each fixed chunk of `u` accumulating the same chunk of every basis
    // vector.
    let mut u = Array1::from_elem(x.len(), T::from_real(0.0));
    u.axis_chunks_iter_mut(Axis(0), CHUNK)
        .into_par_iter()
        .enumerate()
        .for_each(|(c, mut uc)| {
            let start = c * CHUNK;
            for (qj, &yj) in q.iter().zip(y) {
                uc.scaled_add(yj, &qj.slice(s![start..start + uc.len()]));
            }
        });

    let correction = precondition(&u);
    x.zip_mut_with(&correction, |xi, &ci| *xi += ci);
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
/// - `apply`: Matrix-vector product callback.
/// - `precondition`: Linear right preconditioner `M^{-1}`. Linearity lets each cycle keep only
///   the Arnoldi basis and precondition the combined correction once.
/// - `system`: Right-hand side `b` and the initial guess `x_0`, zero when `None`.
/// - `restart`: Krylov dimension of each restart cycle.
/// - `max_iter`: Maximum total number of iterations.
/// - `tol`: Residual RMS tolerance, accepted only on the true residual.
/// - `print`: Iteration table to print, if any.
/// # Returns:
/// - `GMRESResult<T>`: Approximate solution vector together with final residual RMS, number of
///   iterations performed, and convergence flag.
pub fn gmres<F, P, T>(
    mut apply: F,
    precondition: P,
    system: (&Array1<T>, Option<&Array1<T>>),
    restart: usize,
    max_iter: usize,
    tol: f64,
    print: GMRESPrint,
) -> GMRESResult<T>
where
    F: FnMut(&Array1<T>) -> Array1<T>,
    P: Fn(&Array1<T>) -> Array1<T>,
    T: NOCIScalar + Into<Complex64>,
{
    let gmres_start = Instant::now();
    let (b, x0) = system;
    let n = b.len();
    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Array1::<T>::from_elem(n, T::from_real(0.0)),
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
        Some(_) => true_residual(&mut apply, b, &x),
        None => b.clone(),
    };
    let mut residual_rms = calculate_residual_rms(&rtrue, rms);

    if print != GMRESPrint::Silent {
        print_gmres_restart_summary(
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
        let beta = vector_norm(&rtrue);

        // Stop if the residual is numerically zero.
        if beta <= SMALL {
            residual_rms = beta / rms;

            if print != GMRESPrint::Silent {
                print_gmres_restart_summary(
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
        update_solution(&mut x, &cycle.q, &y, &precondition);

        total_iter += cycle.kfinal;

        // Recompute the true residual after each restart as the Arnoldi residual is only an estimate.
        rtrue = true_residual(&mut apply, b, &x);
        residual_rms = calculate_residual_rms(&rtrue, rms);

        if print != GMRESPrint::Silent {
            print_gmres_restart_summary(
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
