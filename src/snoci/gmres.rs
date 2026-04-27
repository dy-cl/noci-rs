// snoci/gmres.rs

use std::time::Instant;

use ndarray::{Array1, Array2};

use crate::time_call;
use super::GMRES;

const SMALL: f64 = 1e-14_f64;
const DIAG_FLOOR_REL: f64 = 1e-12_f64;
const PRINT_STRIDE: usize = 1usize;

/// Storage for a single restarted Arnoldi cycle.
/// # Fields:
/// - `q`: Orthonormal Krylov vectors.
/// - `h`: Upper Hessenberg matrix after Givens rotations.
/// - `g`: Rotated residual right-hand side.
/// - `kfinal`: Number of Arnoldi vectors generated in the current cycle.
struct ArnoldiCycle {
    q: Vec<Array1<f64>>,
    h: Array2<f64>,
    g: Array1<f64>,
    kfinal: usize,
}

/// Print the GMRES iteration table header.
/// # Arguments:
/// None.
/// # Returns:
/// - `()`: Prints the GMRES iteration header to standard output.
fn print_gmres_header() {
    println!("  {}", "-".repeat(98));
    println!("  {:>8} {:>8} {:>16} {:>16} {:>16} {:>16}", "restart", "iter", "Res (est.)", "Res (true)", "Apply / s", "Elapsed / s");
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
fn print_gmres_iteration(restart_id: usize, iter: usize, residual_est: f64, apply_secs: f64, elapsed_secs: f64) {
    println!("  {:>8} {:>8} {:>16.8e} {:>16} {:>16.6} {:>16.6}", restart_id, iter, residual_est, "-", apply_secs, elapsed_secs);
}

/// Print a single GMRES restart summary line using the true residual.
/// # Arguments:
/// - `restart_id`: GMRES restart cycle index.
/// - `iter`: Total GMRES iteration index after the restart.
/// - `residual_true`: True residual RMS after updating the solution.
/// - `elapsed_secs`: Total elapsed GMRES wall time.
/// # Returns:
/// - `()`: Prints the GMRES restart summary to standard output.
fn print_gmres_restart_summary(restart_id: usize, iter: usize, residual_true: f64, elapsed_secs: f64) {
    println!("  {:>8} {:>8} {:>16} {:>16.8e} {:>16} {:>16.6}", restart_id, iter, "-", residual_true, "-", elapsed_secs);
}

/// Build the inverse diagonal used for right Jacobi preconditioning.
/// # Arguments:
/// - `diag`: Optional diagonal of the matrix-free operator.
/// # Returns:
/// - `Option<Array1<f64>>`: Optional inverse diagonal with small entries safely floored.
fn build_right_jacobi_inverse(diag: Option<&Array1<f64>>) -> Option<Array1<f64>> {
    diag.map(|d| {
        let dmax = d.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
        let dfloor = (DIAG_FLOOR_REL * dmax).max(SMALL);
        Array1::from_iter(d.iter().map(|&x| if x.abs() > dfloor {1.0 / x} else {1.0}))
    })
}

/// Apply the optional right Jacobi preconditioner to a vector.
/// # Arguments:
/// - `v`: Vector to precondition.
/// - `dinv`: Optional inverse diagonal.
/// # Returns:
/// - `Array1<f64>`: Preconditioned vector.
fn apply_preconditioner(v: &Array1<f64>, dinv: Option<&Array1<f64>>) -> Array1<f64> {
    match dinv {
        Some(d) => Array1::from_iter(v.iter().zip(d.iter()).map(|(&vi, &di)| vi * di)),
        None => v.clone(),
    }
}

/// Build the true residual `b - A x`.
/// # Arguments:
/// - `apply`: Matrix-vector product callback.
/// - `b`: Right-hand side vector.
/// - `x`: Current solution vector.
/// # Returns:
/// - `Array1<f64>`: True residual vector.
fn true_residual<F>(apply: &F, b: &Array1<f64>, x: &Array1<f64>) -> Array1<f64>
where F: Fn(&Array1<f64>) -> Array1<f64> {
    let mut r = b.clone();
    r.scaled_add(-1.0, &apply(x));
    r
}

/// Compute the RMS norm of a residual vector.
/// # Arguments:
/// - `r`: Residual vector.
/// - `rms`: Square-root of the vector length.
/// # Returns:
/// - `f64`: RMS residual norm.
fn calculate_residual_rms(r: &Array1<f64>, rms: f64) -> f64 {
    r.dot(r).sqrt() / rms
}

/// Orthogonalise an Arnoldi vector against the existing Krylov basis.
/// # Arguments:
/// - `q`: Existing Krylov basis vectors.
/// - `h`: Hessenberg matrix to update.
/// - `w`: Arnoldi vector to orthogonalise in place.
/// - `k`: Current Arnoldi iteration in the restart cycle.
/// # Returns:
/// - `()`: Updates `h` and `w` in place.
fn orthogonalise_arnoldi_vector(q: &[Array1<f64>], h: &mut Array2<f64>, w: &mut Array1<f64>, k: usize) {
    for j in 0..=k {
        h[(j, k)] = q[j].dot(w);
        w.scaled_add(-h[(j, k)], &q[j]);
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
fn extend_arnoldi_basis(q: &mut Vec<Array1<f64>>, h: &mut Array2<f64>, w: Array1<f64>, k: usize) -> f64 {
    let h_next = w.dot(&w).sqrt();
    h[(k + 1, k)] = h_next;

    if h_next > SMALL {
        q.push(w.mapv(|wi| wi / h_next));
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
fn apply_previous_givens(h: &mut Array2<f64>, cs: &[f64], sn: &[f64], k: usize) {
    for j in 0..k {
        let temp = cs[j] * h[(j, k)] + sn[j] * h[(j + 1, k)];
        h[(j + 1, k)] = -sn[j] * h[(j, k)] + cs[j] * h[(j + 1, k)];
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
fn apply_current_givens(h: &mut Array2<f64>, cs: &mut [f64], sn: &mut [f64], g: &mut Array1<f64>, k: usize) {
    let hk = h[(k, k)];
    let hk1 = h[(k + 1, k)];
    let denom = (hk * hk + hk1 * hk1).sqrt();

    if denom > SMALL {
        cs[k] = hk / denom;
        sn[k] = hk1 / denom;
    } else {
        cs[k] = 1.0;
        sn[k] = 0.0;
    }

    h[(k, k)] = cs[k] * hk + sn[k] * hk1;
    h[(k + 1, k)] = 0.0;
    g[k + 1] = -sn[k] * g[k];
    g[k] *= cs[k];
}

/// Run one restarted Arnoldi cycle for the right-preconditioned GMRES operator.
/// # Arguments:
/// - `apply`: Matrix-vector product callback.
/// - `dinv`: Optional inverse diagonal for right Jacobi preconditioning.
/// - `rtrue`: True residual at the start of the restart cycle.
/// - `inner_max`: Maximum number of Arnoldi iterations in this restart cycle.
/// - `restart_id`: GMRES restart cycle index.
/// - `total_iter`: Total number of GMRES iterations before this cycle.
/// - `rms`: Square-root of the vector length.
/// - `tol`: Arnoldi/Givens residual estimate convergence tolerance.
/// - `gmres_start`: Wall-time origin for GMRES.
/// # Returns:
/// - `ArnoldiCycle`: Krylov basis, Hessenberg matrix, rotated residual vector, and final inner iteration count.
fn run_arnoldi_cycle<F>(apply: &F, dinv: Option<&Array1<f64>>, rtrue: &Array1<f64>, inner_max: usize, restart_id: usize, 
                        total_iter: usize, rms: f64, tol: f64, gmres_start: &Instant) -> ArnoldiCycle
where F: Fn(&Array1<f64>) -> Array1<f64> {
    let beta = rtrue.dot(rtrue).sqrt();

    let mut q: Vec<Array1<f64>> = Vec::with_capacity(inner_max + 1);
    q.push(rtrue.mapv(|ri| ri / beta));

    let mut h = Array2::<f64>::zeros((inner_max + 1, inner_max));
    let mut cs = vec![0.0; inner_max];
    let mut sn = vec![0.0; inner_max];
    let mut g = Array1::<f64>::zeros(inner_max + 1);
    g[0] = beta;

    let mut kfinal = 0usize;

    for k in 0..inner_max {
        let z = apply_preconditioner(&q[k], dinv);

        let t_apply = Instant::now();
        let aq = apply(&z);
        let apply_secs = t_apply.elapsed().as_secs_f64();

        let mut w = aq;

        orthogonalise_arnoldi_vector(&q, &mut h, &mut w, k);
        let h_next = extend_arnoldi_basis(&mut q, &mut h, w, k);

        apply_previous_givens(&mut h, &cs, &sn, k);
        apply_current_givens(&mut h, &mut cs, &mut sn, &mut g, k);

        kfinal = k + 1;

        let residual_est = g[k + 1].abs() / rms;
        let iter = total_iter + k + 1;

        if k == 0 || iter.is_multiple_of(PRINT_STRIDE) || residual_est <= tol {
            print_gmres_iteration(restart_id, iter, residual_est, apply_secs, gmres_start.elapsed().as_secs_f64());
        }

        if residual_est <= tol || h_next <= SMALL {
            break;
        }
    }

    ArnoldiCycle {q, h, g, kfinal}
}

/// Solve the small upper-triangular least-squares problem after Arnoldi.
/// # Arguments:
/// - `h`: Rotated Hessenberg matrix.
/// - `g`: Rotated residual right-hand side.
/// - `kfinal`: Number of Arnoldi iterations completed.
/// # Returns:
/// - `Array1<f64>`: Least-squares coefficients in the Krylov basis.
fn back_solve(h: &Array2<f64>, g: &Array1<f64>, kfinal: usize) -> Array1<f64> {
    let mut y = Array1::<f64>::zeros(kfinal);

    for ii in 0..kfinal {
        let i = kfinal - 1 - ii;
        let mut rhs = g[i];

        for j in (i + 1)..kfinal {
            rhs -= h[(i, j)] * y[j];
        }

        y[i] = if h[(i, i)].abs() > SMALL {rhs / h[(i, i)]} else {0.0};
    }

    y
}

/// Update the solution vector using the right-preconditioned Krylov basis.
/// # Arguments:
/// - `x`: Solution vector to update in place.
/// - `q`: Krylov basis vectors.
/// - `y`: Krylov expansion coefficients.
/// - `dinv`: Optional inverse diagonal for right Jacobi preconditioning.
/// # Returns:
/// - `()`: Updates `x` in place.
fn update_solution(x: &mut Array1<f64>, q: &[Array1<f64>], y: &Array1<f64>, dinv: Option<&Array1<f64>>) {
    for j in 0..y.len() {
        let z = apply_preconditioner(&q[j], dinv);
        x.scaled_add(y[j], &z);
    }
}

/// Solve a linear system using restarted GMRES with an operator callback.
/// Uses optional right Jacobi preconditioning.
/// # Arguments:
/// - `apply`: Matrix-vector product callback.
/// - `diag`: Optional diagonal used for right Jacobi preconditioning.
/// - `b`: Right-hand side vector.
/// - `restart`: Maximum Krylov subspace size before restart.
/// - `max_iter`: Maximum total GMRES iterations.
/// - `tol`: True residual RMS convergence tolerance.
/// # Returns:
/// - `GMRES`: Approximate solution vector together with final residual RMS, number of
///   iterations performed, and convergence flag.
pub(in crate::snoci) fn gmres<F>(apply: F, diag: Option<&Array1<f64>>, b: &Array1<f64>, restart: usize, max_iter: usize, tol: f64) -> GMRES
where F: Fn(&Array1<f64>) -> Array1<f64> {
    time_call!(crate::timers::snoci::add_gmres, {
        let gmres_start = Instant::now();
        let n = b.len();
        let mut x = Array1::<f64>::zeros(n);

        print_gmres_header();

        if n == 0 {
            return GMRES {x, residual_rms: 0.0, iterations: 0, converged: true};
        }

        let restart = restart.max(1).min(n);
        let rms = (n as f64).sqrt();
        let dinv = build_right_jacobi_inverse(diag);

        let mut rtrue = true_residual(&apply, b, &x);
        let mut residual_rms = calculate_residual_rms(&rtrue, rms);

        print_gmres_restart_summary(0, 0, residual_rms, gmres_start.elapsed().as_secs_f64());

        if residual_rms <= tol {
            return GMRES {x, residual_rms, iterations: 0, converged: true};
        }

        let mut total_iter = 0usize;
        let mut restart_id = 0usize;

        while total_iter < max_iter {
            let beta = rtrue.dot(&rtrue).sqrt();

            if beta <= SMALL {
                residual_rms = beta / rms;
                print_gmres_restart_summary(restart_id, total_iter, residual_rms, gmres_start.elapsed().as_secs_f64());
                return GMRES {x, residual_rms, iterations: total_iter, converged: residual_rms <= tol};
            }

            let inner_max = restart.min(max_iter - total_iter);
            let cycle = run_arnoldi_cycle(&apply, dinv.as_ref(), &rtrue, inner_max, restart_id, total_iter, rms, tol, &gmres_start);
            let y = back_solve(&cycle.h, &cycle.g, cycle.kfinal);

            update_solution(&mut x, &cycle.q, &y, dinv.as_ref());

            total_iter += cycle.kfinal;

            rtrue = true_residual(&apply, b, &x);
            residual_rms = calculate_residual_rms(&rtrue, rms);

            print_gmres_restart_summary(restart_id, total_iter, residual_rms, gmres_start.elapsed().as_secs_f64());

            restart_id += 1;

            if residual_rms <= tol {
                return GMRES {x, residual_rms, iterations: total_iter, converged: true};
            }
        }

        GMRES {x, residual_rms, iterations: total_iter, converged: residual_rms <= tol}
    })
}
