// nocc/deterministic/solver.rs
//! Inexact Newton solution of the GNOCC amplitude equations in the FOIS coordinates.
//!
//! Each macro-iteration evaluates the energy and the residual `R_\mu` in the linearly dependent
//! excitation basis and projects the residual onto the FOIS, `Y^\dagger R`. The amplitude change
//! `\delta t = Y\,\delta c` then solves the Newton equation
//!
//! `Y^\dagger J Y\,\delta c = -Y^\dagger R`
//!
//! with the exact Jacobian `J = \partial R / \partial t`, either directly from the explicit FOIS
//! Jacobian or matrix-free by GMRES on exact Jacobian-vector products. GMRES is preconditioned by
//! the exact linear Jacobian `Y^\dagger J_1 Y`, built once, or by level-shifted orbital-energy
//! denominators. A non-monotone backtracking line search on `\lVert Y^\dagger R\rVert` scales
//! each step, so a step along a near-singular Jacobian direction cannot carry the iteration away
//! from the root, while moderate increases still let it leave a local minimum of the norm, such
//! as the real axis near a fold. Since `\delta t` lies in the FOIS, the amplitudes stay there without a separate
//! projection.
//!
//! # References
//!
//! - Lee and Tew, *Spin-free Generalised Normal Ordered Coupled Cluster*, arXiv:2507.13472
//!   (2025), Secs. II E-H.
//! - Dembo, Eisenstat and Steihaug, *SIAM J. Numer. Anal.* **19**, 400 (1982),
//!   [doi:10.1137/0719025](https://doi.org/10.1137/0719025).

// Standard library imports.
use std::collections::VecDeque;

// External crate imports.
use ndarray::{Array1, Array2, Zip};
use ndarray_linalg::{Inverse, SVD, Solve};
use num_complex::Complex64;

// Crate-root imports.
use crate::NOCIScalar;
use crate::input::{NOCCMCOptions, NOCCNewton, NOCCPreconditioner};
use crate::maths::gmres::{GMRESPrint, gmres};
use crate::maths::parallel_matvec;
use crate::nocc::equations::{
    COLUMNS, correlation_energy, jacobian_columns, orbital_denominators, residual_vector,
};
use crate::nocc::setup::ReferenceState;
use crate::nocc::space::{Excitation, FoisBasis, Spaces, dense_amplitudes};
use crate::nocc::terms::TermEvaluator;

/// Norm of the imaginary amplitude vector that starts a holomorphic run.
const SEED: f64 = 1e-3;

/// Relative tolerance of each inexact Newton step.
const FORCING: f64 = 1e-1;

/// Fraction of the predicted decrease of `\lVert Y^\dagger R\rVert` a step must achieve.
const ARMIJO: f64 = 1e-4;

/// Largest number of step halvings in one line search.
const HALVINGS: usize = 10;

/// Number of recent residual norms whose largest a step must improve on.
const MEMORY: usize = 5;

/// Converged or final state of the amplitude equations.
pub(crate) struct AmplitudeSolution {
    /// Correlation energy `E - E_0`, its real part in a holomorphic run.
    pub(crate) correlation_energy: f64,
    /// Imaginary part of the correlation energy, zero unless the run is holomorphic.
    pub(crate) imaginary_energy: f64,
    /// Norm of the FOIS residual `\lVert Y^\dagger R\rVert`.
    pub(crate) residual_norm: f64,
    /// Number of macro-iterations performed.
    pub(crate) iterations: usize,
    /// Whether the residual norm fell below the tolerance.
    pub(crate) converged: bool,
    /// Final amplitude vector, complex in a holomorphic run.
    pub(crate) amplitudes: Array1<Complex64>,
}

/// Solve the GNOCC amplitude equations by inexact Newton iteration.
/// Starting from `t = 0` or a continued guess, each macro-iteration evaluates `R_\mu` and `E - E_0` and adds the
/// Newton step `\delta t = Y\,\delta c`. A holomorphic run iterates complex amplitudes in the
/// same equations, without complex conjugation.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `fois`: FOIS basis data.
/// - `start`: Reference energy, used only for printing total energies, and the starting
///   amplitudes continued from the previous geometry, if any.
/// - `options`: Iteration limits, tolerances, level shift and holomorphic switch.
/// # Returns:
/// - `AmplitudeSolution`: Final energy and convergence state.
pub(crate) fn solve_amplitudes(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
    fois: &FoisBasis,
    start: (f64, Option<&Array1<Complex64>>),
    options: &NOCCMCOptions,
) -> AmplitudeSolution {
    if options.holomorphic {
        iterate_amplitudes::<Complex64>(
            reference,
            spaces,
            excitations,
            evaluator,
            fois,
            start,
            options,
        )
    } else {
        iterate_amplitudes::<f64>(
            reference,
            spaces,
            excitations,
            evaluator,
            fois,
            start,
            options,
        )
    }
}

/// Run the Newton iteration in the amplitude scalar type `T`, real or, for a holomorphic run,
/// complex. A holomorphic run adds a small imaginary amplitude vector inside the FOIS to its
/// start, so the iteration can follow the solution onto the complex plane where no real solution
/// exists; where one does, the imaginary part decays.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `fois`: FOIS basis data.
/// - `start`: Reference energy, used only for printing total energies, and the starting
///   amplitudes continued from the previous geometry, if any.
/// - `options`: Iteration limits, tolerances and level shift.
/// # Returns:
/// - `AmplitudeSolution`: Final energy and convergence state.
/// # Panics
/// - Panics if a holomorphic run is iterated in real arithmetic.
fn iterate_amplitudes<T: NOCIScalar + Into<Complex64>>(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
    fois: &FoisBasis,
    start: (f64, Option<&Array1<Complex64>>),
    options: &NOCCMCOptions,
) -> AmplitudeSolution {
    let (e0, guess) = start;
    let y = &fois.y;
    let n = excitations.len();
    let yt = y.t().to_owned();

    // The dense solver and the linear preconditioner apply the Jacobian to every FOIS direction
    // `v_c = Y e_c`, in batches of at most `COLUMNS` directions with the Jacobian tables planned
    // for that batch size.
    let width = COLUMNS.min(y.ncols());
    let batch = TermEvaluator::new(options.max_cumulant, spaces, width);
    let linear = matches!(options.preconditioner, NOCCPreconditioner::Linear);
    let directions = match options.newton {
        NOCCNewton::GMRES if !linear => Vec::new(),
        _ => (0..y.ncols())
            .map(|c| dense_amplitudes(spaces, excitations, &y.column(c).to_owned()))
            .collect::<Vec<_>>(),
    };

    // GMRES preconditioner `M^{-1}`: the inverse linear FOIS Jacobian `[Y^\dagger J_1 Y]^{-1}`, the
    // exact Jacobian at `t = 0` independent of the amplitudes, or the inverse diagonal
    // `M_{ii} = [Y^\dagger S D Y]_{ii}` with the shifted denominators `D = \Delta + \eta`.
    let inverse = match (options.newton, options.preconditioner) {
        (NOCCNewton::Dense, _) => Array2::zeros((0, 0)),
        (NOCCNewton::GMRES, NOCCPreconditioner::Linear) => {
            let zero = dense_amplitudes(spaces, excitations, &Array1::<f64>::zeros(n));
            let columns = jacobian_columns(
                reference,
                spaces,
                excitations,
                &batch,
                &zero,
                &directions,
                &[1],
            );
            yt.dot(&columns)
                .inv()
                .expect("singular linear FOIS Jacobian")
        }
        (NOCCNewton::GMRES, NOCCPreconditioner::Diagonal) => {
            let denominators = orbital_denominators(reference, excitations) + options.level_shift;
            let sy = fois.metric.dot(y);
            Array2::from_diag(&Array1::from_iter((0..y.ncols()).map(|i| {
                1.0 / (0..n)
                    .map(|nu| sy[(nu, i)] * denominators[nu] * y[(nu, i)])
                    .sum::<f64>()
            })))
        }
    };
    let precondition = |x: &Array1<T>| real_operator(|w| inverse.dot(w), x);

    // Parts `R_k` of the residual of the requested amplitude orders `k` at amplitudes `t`.
    let residual_at = |t: &Array1<T>, orders: &[usize]| {
        let dense = dense_amplitudes(spaces, excitations, t);
        residual_vector(reference, spaces, excitations, evaluator, &dense, orders)
    };

    // Start from the continued guess, or `t = 0`, its real part in a real run.
    let mut amplitudes = match guess {
        Some(t) if options.holomorphic => {
            t.mapv(|z| <T as From<f64>>::from(z.re) + T::from_imag(z.im))
        }
        Some(t) => t.mapv(|z| <T as From<f64>>::from(z.re)),
        None => Array1::<T>::zeros(n),
    };
    if options.holomorphic {
        // A fixed, structureless direction `u_k = \{0.618 (k + 1)\} - \tfrac12` in the FOIS
        // coordinates, mapped to amplitudes `Y u` and scaled to norm `SEED` there.
        let u =
            Array1::from_iter((0..y.ncols()).map(|k| ((k + 1) as f64 * 0.618034).fract() - 0.5));
        let seed = y.dot(&u);
        let seed = &seed * (SEED / seed.dot(&seed).sqrt());
        amplitudes = amplitudes + seed.mapv(T::from_imag);
    }
    let mut previous = <T as From<f64>>::from(0.0);
    let mut solution = AmplitudeSolution {
        correlation_energy: 0.0,
        imaginary_energy: 0.0,
        residual_norm: f64::INFINITY,
        iterations: 0,
        converged: false,
        amplitudes: Array1::zeros(n),
    };

    // The step columns report the GMRES iterations and linear residual, or the extreme singular
    // values of the dense FOIS Jacobian, then the line-search step length.
    let (first, second) = match options.newton {
        NOCCNewton::Dense => ("s_min", "s_max"),
        NOCCNewton::GMRES => ("gmres", "||gmres||"),
    };
    println!("{}", "=".repeat(100));
    println!("GNOCC amplitude iterations");
    if options.holomorphic {
        println!(
            "{:>4} {:>20} {:>14} {:>20} {:>12} {:>12} {:>12} {:>9}",
            "i", "Re E", "Im E", "Re dE", "||Y^T R||", first, second, "step"
        );
    } else {
        println!(
            "{:>4} {:>20} {:>20} {:>12} {:>12} {:>12} {:>9}",
            "i", "E", "dE", "||Y^T R||", first, second, "step"
        );
    }

    // The residual at the step accepted by the line search, reused by the next iteration, and the
    // recent residual norms the line search compares against.
    let mut ahead: Option<Array1<T>> = None;
    let mut recent = VecDeque::<f64>::with_capacity(MEMORY);

    for iteration in 0..options.max_macro {
        // Energy and FOIS residual `R_i = \sum_\mu Y^\dagger_{i\mu} R_\mu` at the current amplitudes.
        let dense = dense_amplitudes(spaces, excitations, &amplitudes);
        let residual = ahead
            .take()
            .unwrap_or_else(|| residual_at(&amplitudes, &[0, 1, 2]));
        let ecorr = correlation_energy(reference, spaces, evaluator, &dense);
        let rfois = real_operator(|v| y.t().dot(v), &residual);
        let norm = vector_norm(&rfois);

        let converged = norm < options.residual_tol;
        solution = AmplitudeSolution {
            correlation_energy: ecorr.re(),
            imaginary_energy: ecorr.im(),
            residual_norm: norm,
            iterations: iteration,
            converged,
            amplitudes: amplitudes.mapv(Into::into),
        };
        let row = if options.holomorphic {
            format!(
                "{:>4} {:>20.12} {:>+14.6e} {:>20.12e} {:>12.4e}",
                iteration,
                e0 + ecorr.re(),
                ecorr.im(),
                ecorr.re() - previous.re(),
                norm
            )
        } else {
            format!(
                "{:>4} {:>20.12} {:>20.12e} {:>12.4e}",
                iteration,
                e0 + ecorr.re(),
                ecorr.re() - previous.re(),
                norm
            )
        };
        previous = ecorr;
        if converged {
            println!("{row} {:>12} {:>12} {:>9}", "-", "-", "-");
            break;
        }

        // Solve `Y^\dagger J Y \delta c = -Y^\dagger R`; `\delta t = Y \delta c` stays in the FOIS.
        let step = match options.newton {
            NOCCNewton::Dense => {
                // `[Y^\dagger J Y]_{ic} = \sum_\mu Y_{\mu i} [J v_c]_\mu` from every column at once.
                let columns = jacobian_columns(
                    reference,
                    spaces,
                    excitations,
                    &batch,
                    &dense,
                    &directions,
                    &[1, 2],
                );
                let jacobian = real_matrix_operator(&yt, &columns);
                let (step, smin, smax) = dense_newton_step(&rfois, &jacobian);
                (step, format!("{:>12.4e} {:>12.4e}", smin, smax))
            }
            NOCCNewton::GMRES => {
                // The residual is quadratic in the amplitudes, `R = R_0 + R_1(t) + R_2(t, t)`, so
                // the exact Jacobian action is `J v = R_1(v) + [R_2(t + v) - R_2(t - v)] / 2`. The
                // Newton system is solved in the FOIS coordinates, `v = Y c` and `Y^\dagger J Y c`.
                let jacobian = |c: &Array1<T>| {
                    let v = real_operator(|w| parallel_matvec(y, w), c);
                    let linear = residual_at(&v, &[1]);
                    let plus = residual_at(&(&amplitudes + &v), &[2]);
                    let minus = residual_at(&(&amplitudes - &v), &[2]);
                    let jv = linear + (plus - minus).mapv(|x| x * <T as From<f64>>::from(0.5));
                    real_operator(|w| parallel_matvec(&yt, w), &jv)
                };
                let (step, count, micro) = newton_step(&rfois, &jacobian, &precondition, options);
                (step, format!("{:>12} {:>12.4e}", count, micro))
            }
        };
        let (step, stats) = step;

        // Non-monotone backtracking line search: halve the step `\delta t = \alpha Y \delta c`
        // until `\lVert Y^\dagger R(t + \delta t)\rVert \le (1 - c\alpha) \max_k \lVert Y^\dagger R(t_k)\rVert`
        // over the last `MEMORY` iterates, keeping the last trial after `HALVINGS` halvings.
        if recent.len() == MEMORY {
            recent.pop_front();
        }
        recent.push_back(norm);
        let bound = recent.iter().copied().fold(0.0, f64::max);
        let delta = real_operator(|w| y.dot(w), &step);
        let mut alpha = 1.0;
        for halving in 0..=HALVINGS {
            let trial = &amplitudes + &delta.mapv(|x| x * <T as From<f64>>::from(alpha));
            let r = residual_at(&trial, &[0, 1, 2]);
            let f = vector_norm(&real_operator(|v| y.t().dot(v), &r));
            if f <= (1.0 - ARMIJO * alpha) * bound || halving == HALVINGS {
                amplitudes = trial;
                ahead = Some(r);
                break;
            }
            alpha *= 0.5;
        }
        println!("{row} {stats} {:>9.3e}", alpha);
    }

    solution
}

/// Solve the Newton equation `Y^\dagger J Y\,\delta c = -Y^\dagger R` in the FOIS coordinates by
/// restarted GMRES, right-preconditioned by `M^{-1}`. The inexact step is solved
/// only to the fraction `FORCING` of `\lVert Y^\dagger R\rVert`, and never beyond the GMRES
/// residual RMS tolerance.
/// # Arguments:
/// - `rfois`: FOIS residual `Y^\dagger R`.
/// - `jacobian`: Action of the Jacobian `Y^\dagger J Y` on a vector of FOIS coordinates.
/// - `precondition`: Action of the preconditioner `M^{-1}`.
/// - `options`: GMRES iteration limit, restart and tolerance.
/// # Returns:
/// - `(Array1<T>, usize, f64)`: Coordinate change `\delta c`, the number of GMRES iterations
///   performed and the final norm of `Y^\dagger R + Y^\dagger J Y\,\delta c`.
fn newton_step<T: NOCIScalar + Into<Complex64>>(
    rfois: &Array1<T>,
    jacobian: &impl Fn(&Array1<T>) -> Array1<T>,
    precondition: &impl Fn(&Array1<T>) -> Array1<T>,
    options: &NOCCMCOptions,
) -> (Array1<T>, usize, f64) {
    // GMRES measures the residual RMS, so the relative tolerance is scaled by `\sqrt{n}`.
    let rms = (rfois.len() as f64).sqrt();
    let tol = (FORCING * vector_norm(rfois) / rms).max(options.gmres.res_tol);
    let b = rfois.mapv(|x| -x);
    let result = gmres(
        jacobian,
        precondition,
        &b,
        options.gmres.restart,
        options.gmres.max_iter,
        tol,
        GMRESPrint::Residual,
    );

    (result.x, result.iterations, result.residual_rms * rms)
}

/// Solve the Newton equation `Y^\dagger J Y\,\delta c = -Y^\dagger R` directly from the dense
/// FOIS Jacobian by LU factorisation, reporting its extreme singular values; a vanishing
/// `\sigma_{\min}` marks the fold where the real root is lost.
/// # Arguments:
/// - `rfois`: FOIS residual `Y^\dagger R`.
/// - `jacobian`: Dense FOIS Jacobian `Y^\dagger J Y`.
/// # Returns:
/// - `(Array1<T>, f64, f64)`: Coordinate change `\delta c` and the smallest and largest singular
///   values of `Y^\dagger J Y`.
/// # Panics
/// - Panics if the Jacobian is exactly singular or its singular values cannot be computed.
fn dense_newton_step<T: NOCIScalar>(
    rfois: &Array1<T>,
    jacobian: &Array2<T>,
) -> (Array1<T>, f64, f64) {
    let (_, sigma, _) = jacobian
        .svd(false, false)
        .expect("FOIS Jacobian SVD failed");
    let smax = sigma.iter().copied().fold(0.0, f64::max);
    let smin = sigma.iter().copied().fold(f64::INFINITY, f64::min);

    let b = rfois.mapv(|x| -x);
    let step = jacobian.solve(&b).expect("singular FOIS Jacobian");

    (step, smin, smax)
}

/// Apply a real linear operator to every column of a real or complex matrix,
/// `A X = A \Re X + i A \Im X`.
/// # Arguments:
/// - `a`: Real linear operator.
/// - `x`: Matrix to transform.
/// # Returns:
/// - `Array2<T>`: Transformed matrix.
fn real_matrix_operator<T: NOCIScalar>(
    a: &Array2<f64>,
    x: &Array2<T>,
) -> Array2<T> {
    let re = a.dot(&x.mapv(|z| z.re()));
    let im = a.dot(&x.mapv(|z| z.im()));
    Zip::from(&re)
        .and(&im)
        .map_collect(|&p, &q| <T as From<f64>>::from(p) + T::from_imag(q))
}

/// Apply a real linear operator to a real or complex vector, `A x = A \Re x + i A \Im x`; a
/// vector with no imaginary part takes one application.
/// # Arguments:
/// - `operator`: Real linear operator.
/// - `x`: Vector to transform.
/// # Returns:
/// - `Array1<T>`: Transformed vector.
fn real_operator<T: NOCIScalar>(
    operator: impl Fn(&Array1<f64>) -> Array1<f64>,
    x: &Array1<T>,
) -> Array1<T> {
    let re = operator(&x.mapv(|z| z.re()));
    let im = x.mapv(|z| z.im());
    if im.iter().all(|&v| v == 0.0) {
        return re.mapv(<T as From<f64>>::from);
    }
    let im = operator(&im);
    Zip::from(&re)
        .and(&im)
        .map_collect(|&a, &b| <T as From<f64>>::from(a) + T::from_imag(b))
}

/// Return the Euclidean norm `\lVert x\rVert = (\sum_\mu |x_\mu|^2)^{1/2}` of a real or complex
/// vector.
/// # Arguments:
/// - `x`: Vector.
/// # Returns:
/// - `f64`: Norm of `x`.
fn vector_norm<T: NOCIScalar>(x: &Array1<T>) -> f64 {
    let re = x.mapv(|z| z.re());
    let im = x.mapv(|z| z.im());
    (re.dot(&re) + im.dot(&im)).sqrt()
}
