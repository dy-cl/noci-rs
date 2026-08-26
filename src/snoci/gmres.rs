// snoci/gmres.rs

// Standard library imports.
use std::time::Instant;

// External crate imports.
use mpi::topology::Communicator;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

// Crate-root imports.
use crate::noci::NOCIScalar;
use crate::{input::GMRESOptions, time_call};

// Parent/sibling imports.
use super::{ArnoldiCycle, ArnoldiParams, GMRESResult};

const SMALL: f64 = 1e-14_f64;
const PRINT_STRIDE: usize = 1usize;

/// Print the GMRES iteration table header.
/// # Arguments:
/// None.
/// # Returns:
/// - `()`: Prints the GMRES iteration header to standard output.
fn print_gmres_header() {
    println!();
    println!("  GMRES solve");
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
}

/// Print a single GMRES iteration summary line.
/// # Arguments:
/// - `restart_id`: GMRES restart cycle index.
/// - `iter`: Total GMRES iteration index.
/// - `residual_est`: Arnoldi/Givens residual estimate for the current Krylov solve.
/// - `apply_secs`: Time spent applying the matrix-free operator on this iteration.
/// - `elapsed_secs`: Total elapsed GMRES wall time.
/// # Returns:
/// - `()`: Prints the GMRES iteration summary to standard output.
fn print_gmres_iteration(
    restart_id: usize,
    iter: usize,
    residual_est: f64,
    e2_hyl: f64,
    delta_hyl_proj: f64,
    apply_secs: f64,
    elapsed_secs: f64,
) {
    println!(
        "  {:>8} {:>8} {:>16.8e} {:>16} {:>18.10e} {:>18.10e} {:>16.6} {:>16.6}",
        restart_id, iter, residual_est, "-", e2_hyl, delta_hyl_proj, apply_secs, elapsed_secs
    );
}

/// Print a single GMRES restart summary line using the true residual.
/// # Arguments:
/// - `restart_id`: GMRES restart cycle index.
/// - `iter`: Total GMRES iteration index after the restart.
/// - `residual_true`: True residual RMS after updating the solution.
/// - `elapsed_secs`: Total elapsed GMRES wall time.
/// # Returns:
/// - `()`: Prints the GMRES restart summary to standard output.
fn print_gmres_restart_summary(
    restart_id: usize,
    iter: usize,
    residual_true: f64,
    elapsed_secs: f64,
) {
    println!(
        "  {:>8} {:>8} {:>16} {:>16.8e} {:>18} {:>18} {:>16} {:>16.6}",
        restart_id, iter, "-", residual_true, "-", "-", "-", elapsed_secs
    );
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
    let ax = apply(x);
    Array1::from_iter(b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi))
}

/// Compute the conjugated inner product of two scalar vectors.
/// # Arguments:
/// - `x`: Left vector.
/// - `y`: Right vector.
/// # Returns:
/// - `T`: Inner product `x^H y`.
pub fn inner_product<T: NOCIScalar>(
    x: &Array1<T>,
    y: &Array1<T>,
) -> T {
    x.iter()
        .zip(y.iter())
        .fold(T::from_real(0.0), |acc, (&xi, &yi)| acc + xi.conj() * yi)
}

/// Compute the Euclidean norm of a scalar vector.
/// # Arguments:
/// - `v`: Vector to norm.
/// # Returns:
/// - `f64`: Euclidean norm.
fn vector_norm<T: NOCIScalar>(v: &Array1<T>) -> f64 {
    v.iter()
        .map(|&x| {
            let ax = x.abs();
            ax * ax
        })
        .sum::<f64>()
        .sqrt()
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

/// Orthogonalise an Arnoldi vector against the existing Krylov basis.
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
        for j in 0..=k {
            let hjk = inner_product(&q[j], w);
            h[(j, k)] += hjk;
            for i in 0..w.len() {
                w[i] -= hjk * q[j][i];
            }
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
        q.push(w.mapv(|wi| wi / T::from_real(h_next)));
    }

    h_next
}

/// Compute current GMRES Hylleraas diagnostics using Arnoldi data already generated this iteration.
/// # Arguments:
/// - `b`: Right-hand side vector.
/// - `x_start`: Solution at the start of the restart cycle.
/// - `q`: Arnoldi basis.
/// - `z_basis`: Cached right-preconditioned Krylov basis.
/// - `h_raw`: Raw Arnoldi Hessenberg matrix.
/// - `h_rot`: Rotated Hessenberg matrix.
/// - `g`: Rotated residual right-hand side.
/// - `kfinal`: Number of completed Arnoldi iterations in the current cycle.
/// - `beta`: Norm of the restart-cycle initial residual.
/// # Returns:
/// - `(f64, f64)`: `E2_Hyl` and `E2_Hyl - E2_Proj`.
fn hylleraas_diagnostic<T: NOCIScalar + Into<Complex64>>(
    params: &ArnoldiParams<'_, T>,
    q: &[Array1<T>],
    z_basis: &[Array1<T>],
    h_raw: &Array2<T>,
    h_rot: &Array2<T>,
    g: &Array1<T>,
    beta: f64,
) -> (f64, f64) {
    let kfinal = z_basis.len();
    let y = back_solve(h_rot, g, kfinal);

    let mut a = params.x_start.clone();
    for j in 0..kfinal {
        for i in 0..a.len() {
            a[i] += y[j] * z_basis[j][i];
        }
    }

    let mut rho = Array1::<T>::from_elem(kfinal + 1, T::from_real(0.0));
    rho[0] = T::from_real(beta);
    for j in 0..kfinal {
        for i in 0..=kfinal {
            rho[i] -= h_raw[(i, j)] * y[j];
        }
    }

    let mut r = Array1::<T>::from_elem(q[0].len(), T::from_real(0.0));
    for i in 0..=kfinal {
        for n in 0..r.len() {
            r[n] += rho[i] * q[i][n];
        }
    }

    let b_dot_a: Complex64 = inner_product(params.b, &a).into();
    let a_dot_r: Complex64 = inner_product(&a, &r).into();
    let e2_proj = -b_dot_a.re;
    let delta_hyl_proj = -a_dot_r.re;
    (e2_proj + delta_hyl_proj, delta_hyl_proj)
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
/// - `opts`: GMRES options controlling restart size, iteration limit, and residual tolerance.
/// # Returns:
/// - `ArnoldiCycle`: Krylov basis, Hessenberg matrix, rotated residual vector, and final inner iteration count.
fn run_arnoldi_cycle<F, P, T>(
    apply: &mut F,
    precondition: &P,
    rtrue: &Array1<T>,
    params: &ArnoldiParams<'_, T>,
    opts: &GMRESOptions,
    print_iterations: bool,
) -> ArnoldiCycle<T>
where
    F: FnMut(&Array1<T>) -> Array1<T>,
    P: Fn(&Array1<T>) -> Array1<T>,
    T: NOCIScalar + Into<Complex64>,
{
    let beta = vector_norm(rtrue);

    let mut q: Vec<Array1<T>> = Vec::with_capacity(params.inner_max + 1);
    q.push(rtrue.mapv(|ri| ri / T::from_real(beta)));

    let mut z_basis: Vec<Array1<T>> = Vec::with_capacity(params.inner_max);
    let mut h_raw =
        Array2::<T>::from_elem((params.inner_max + 1, params.inner_max), T::from_real(0.0));
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
        z_basis.push(z.clone());

        // Apply the expensive matrix-free operator.
        let t_apply = Instant::now();
        let aq = apply(&z);
        let apply_secs = t_apply.elapsed().as_secs_f64();

        let mut w = aq;

        // Modified Gram-Schmidt: orthogonalise A P^{-1} q_k against previous q vectors.
        orthogonalise_arnoldi_vector(&q, &mut h_raw, &mut w, k);

        // Normalise the new direction and append it to the Krylov basis if non-zero.
        let h_next = extend_arnoldi_basis(&mut q, &mut h_raw, w, k);

        for i in 0..=(k + 1) {
            h_rot[(i, k)] = h_raw[(i, k)];
        }

        // Update the small least-squares problem with Givens rotations.
        apply_previous_givens(&mut h_rot, &cs, &sn, k);
        apply_current_givens(&mut h_rot, &mut cs, &mut sn, &mut g, k);

        kfinal = k + 1;

        // Cheap GMRES residual estimate from the rotated least-squares RHS.
        let residual_est = g[k + 1].abs() / params.rms;
        let iter = params.total_iter + k + 1;

        if print_iterations
            && (k == 0 || iter.is_multiple_of(PRINT_STRIDE) || residual_est <= opts.res_tol)
        {
            let (e2_hyl, delta_hyl_proj) =
                hylleraas_diagnostic(params, &q, &z_basis, &h_raw, &h_rot, &g, beta);
            print_gmres_iteration(
                params.restart_id,
                iter,
                residual_est,
                e2_hyl,
                delta_hyl_proj,
                apply_secs,
                params.gmres_start.elapsed().as_secs_f64(),
            );
        }

        if residual_est <= opts.res_tol || h_next <= SMALL {
            break;
        }
    }
    ArnoldiCycle {
        z: z_basis,
        h: h_rot,
        g,
        kfinal,
    }
}

/// Update the solution vector using the cached right-preconditioned Krylov basis.
/// # Arguments:
/// - `x`: Solution vector to update in place.
/// - `z_basis`: Cached right-preconditioned Krylov basis vectors.
/// - `y`: Krylov expansion coefficients.
/// # Returns:
/// - `()`: Updates `x` in place.
fn update_solution<T>(
    x: &mut Array1<T>,
    z_basis: &[Array1<T>],
    y: &Array1<T>,
) where
    T: NOCIScalar,
{
    for j in 0..y.len() {
        for i in 0..x.len() {
            x[i] += y[j] * z_basis[j][i];
        }
    }
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
/// - `precondition`: Right-preconditioner callback.
/// - `b`: Right-hand side vector.
/// - `opts`: GMRES options controlling restart size, iteration limit, and residual tolerance.
/// # Returns:
/// - `GMRES`: Approximate solution vector together with final residual RMS, number of
///   iterations performed, and convergence flag.
pub(in crate::snoci) fn gmres<F, P, T>(
    mut apply: F,
    precondition: P,
    b: &Array1<T>,
    opts: &GMRESOptions,
    world: &impl Communicator,
) -> GMRESResult<T>
where
    F: FnMut(&Array1<T>) -> Array1<T>,
    P: Fn(&Array1<T>) -> Array1<T>,
    T: NOCIScalar + Into<Complex64>,
{
    time_call!(crate::timers::snoci::add_gmres, {
        let gmres_start = Instant::now();
        let n = b.len();
        let mut x = Array1::<T>::from_elem(n, T::from_real(0.0));

        if world.rank() == 0 {
            print_gmres_header();
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

        // Start from the zero vector and compute the true residual.
        let mut rtrue = true_residual(&mut apply, b, &x);
        let mut residual_rms = calculate_residual_rms(&rtrue, rms);

        if world.rank() == 0 {
            print_gmres_restart_summary(0, 0, residual_rms, gmres_start.elapsed().as_secs_f64());
        }

        // Accept the zero initial guess if it already satisfies the true residual tolerance.
        if residual_rms <= opts.res_tol {
            return GMRESResult {
                x,
                residual_rms,
                iterations: 0,
                converged: true,
            };
        }

        let mut total_iter = 0usize;
        let mut restart_id = 0usize;

        while total_iter < opts.max_iter {
            let beta = vector_norm(&rtrue);

            // Stop if the residual is numerically zero.
            if beta <= SMALL {
                residual_rms = beta / rms;

                if world.rank() == 0 {
                    print_gmres_restart_summary(
                        restart_id,
                        total_iter,
                        residual_rms,
                        gmres_start.elapsed().as_secs_f64(),
                    );
                }

                return GMRESResult {
                    x,
                    residual_rms,
                    iterations: total_iter,
                    converged: residual_rms <= opts.res_tol,
                };
            }

            // Build one Krylov subspace for the right-preconditioned operator A P^{-1}.
            let inner_max = opts.restart.min(opts.max_iter - total_iter);
            let arnoldi_params = ArnoldiParams {
                inner_max,
                b,
                x_start: &x,
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
                opts,
                world.rank() == 0,
            );

            // Solve the small least-squares problem in the Krylov basis.
            let y = back_solve(&cycle.h, &cycle.g, cycle.kfinal);

            // Apply the right-preconditioned Krylov correction.
            update_solution(&mut x, &cycle.z, &y);

            total_iter += cycle.kfinal;

            // Recompute the true residual after each restart as the Arnoldi residual is only an estimate.
            rtrue = true_residual(&mut apply, b, &x);
            residual_rms = calculate_residual_rms(&rtrue, rms);

            if world.rank() == 0 {
                print_gmres_restart_summary(
                    restart_id,
                    total_iter,
                    residual_rms,
                    gmres_start.elapsed().as_secs_f64(),
                );
            }

            restart_id += 1;

            // Only the true residual is accepted as final convergence.
            if residual_rms <= opts.res_tol {
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
            converged: residual_rms <= opts.res_tol,
        }
    })
}
