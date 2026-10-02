// nocc/deterministic/solver.rs
//! Inexact Newton solution of the GNOCC amplitude equations in the FOIS coordinates.
//!
//! Each macro-iteration evaluates the energy and the residual `R_\mu` in the linearly dependent
//! excitation basis and projects the residual onto the FOIS, `Y^\dagger R`. The amplitude change
//! `\delta t = Y\,\delta c` then solves the Newton equation
//!
//! `Y^\dagger J Y\,\delta c = -Y^\dagger R`
//!
//! with the exact Jacobian `J = \partial R / \partial t`, by GMRES preconditioned with
//! level-shifted orbital-energy denominators. Since `\delta t` lies in the FOIS, the amplitudes
//! stay there without a separate projection.
//!
//! # References
//!
//! - Lee and Tew, *Spin-free Generalised Normal Ordered Coupled Cluster*, arXiv:2507.13472
//!   (2025), Secs. II E-H.
//! - Dembo, Eisenstat and Steihaug, *SIAM J. Numer. Anal.* **19**, 400 (1982),
//!   [doi:10.1137/0719025](https://doi.org/10.1137/0719025).

// External crate imports.
use ndarray::{Array1, Zip};
use num_complex::Complex64;

// Crate-root imports.
use crate::NOCIScalar;
use crate::input::NOCCMCOptions;
use crate::maths::gmres::{GMRESPrint, gmres};
use crate::maths::parallel_matvec;
use crate::nocc::equations::{correlation_energy, orbital_denominators, residual_vector};
use crate::nocc::setup::ReferenceState;
use crate::nocc::space::{Excitation, FoisBasis, Spaces, dense_amplitudes};
use crate::nocc::terms::TermEvaluator;

/// Norm of the imaginary amplitude vector that starts a holomorphic run.
const SEED: f64 = 1e-3;

/// Relative tolerance of each inexact Newton step.
const FORCING: f64 = 1e-1;

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
}

/// Solve the GNOCC amplitude equations by inexact Newton iteration.
/// Starting from `t = 0`, each macro-iteration evaluates `R_\mu` and `E - E_0` and adds the
/// Newton step `\delta t = Y\,\delta c`. A holomorphic run iterates complex amplitudes in the
/// same equations, without complex conjugation.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `fois`: FOIS basis data.
/// - `e0`: Reference energy, used only for printing total energies.
/// - `options`: Iteration limits, tolerances, level shift and holomorphic switch.
/// # Returns:
/// - `AmplitudeSolution`: Final energy and convergence state.
pub(crate) fn solve_amplitudes(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
    fois: &FoisBasis,
    e0: f64,
    options: &NOCCMCOptions,
) -> AmplitudeSolution {
    if options.holomorphic {
        iterate_amplitudes::<Complex64>(
            reference,
            spaces,
            excitations,
            evaluator,
            fois,
            e0,
            options,
        )
    } else {
        iterate_amplitudes::<f64>(reference, spaces, excitations, evaluator, fois, e0, options)
    }
}

/// Run the Newton iteration in the amplitude scalar type `T`, real or, for a holomorphic run,
/// complex. A holomorphic run starts from a small imaginary amplitude vector inside the FOIS, so
/// the iteration can follow the solution onto the complex plane where no real solution exists;
/// where one does, the imaginary part decays.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `fois`: FOIS basis data.
/// - `e0`: Reference energy, used only for printing total energies.
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
    e0: f64,
    options: &NOCCMCOptions,
) -> AmplitudeSolution {
    let y = &fois.y;
    let n = excitations.len();
    let yt = y.t().to_owned();

    // Jacobi preconditioner of the Newton system in the orthonormal FOIS coordinates,
    // `M_{ii} = [Y^\dagger S D Y]_{ii}` with the shifted denominators `D = \Delta + \eta`, so
    // `M = (\Delta + \eta) I` when the denominators are uniform.
    let denominators = orbital_denominators(reference, excitations) + options.level_shift;
    let sy = fois.metric.dot(y);
    let preconditioner = Array1::from_iter((0..y.ncols()).map(|i| {
        (0..n)
            .map(|nu| sy[(nu, i)] * denominators[nu] * y[(nu, i)])
            .sum()
    }));

    // Parts `R_k` of the residual of the requested amplitude orders `k` at amplitudes `t`.
    let residual_at = |t: &Array1<T>, orders: &[usize]| {
        let dense = dense_amplitudes(spaces, excitations, t);
        residual_vector(reference, spaces, excitations, evaluator, &dense, orders)
    };

    let mut amplitudes = Array1::<T>::zeros(n);
    if options.holomorphic {
        // A fixed, structureless direction `u_k = \{0.618 (k + 1)\} - \tfrac12` in the FOIS
        // coordinates, mapped to amplitudes `Y u` and scaled to norm `SEED` there.
        let u =
            Array1::from_iter((0..y.ncols()).map(|k| ((k + 1) as f64 * 0.618034).fract() - 0.5));
        let seed = y.dot(&u);
        let seed = &seed * (SEED / seed.dot(&seed).sqrt());
        amplitudes = seed.mapv(T::from_imag);
    }
    let mut previous = <T as From<f64>>::from(0.0);
    let mut solution = AmplitudeSolution {
        correlation_energy: 0.0,
        imaginary_energy: 0.0,
        residual_norm: f64::INFINITY,
        iterations: 0,
        converged: false,
    };

    println!("{}", "=".repeat(100));
    println!("GNOCC amplitude iterations");
    if options.holomorphic {
        println!(
            "{:>4} {:>20} {:>14} {:>20} {:>12} {:>7} {:>12}",
            "i", "Re E", "Im E", "Re dE", "||Y^T R||", "gmres", "||gmres||"
        );
    } else {
        println!(
            "{:>4} {:>20} {:>20} {:>12} {:>7} {:>12}",
            "i", "E", "dE", "||Y^T R||", "gmres", "||gmres||"
        );
    }

    for iteration in 0..options.max_macro {
        // Energy and FOIS residual `R_i = \sum_\mu Y^\dagger_{i\mu} R_\mu` at the current amplitudes.
        let dense = dense_amplitudes(spaces, excitations, &amplitudes);
        let residual = residual_vector(
            reference,
            spaces,
            excitations,
            evaluator,
            &dense,
            &[0, 1, 2],
        );
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
            println!("{row} {:>7} {:>12}", "-", "-");
            break;
        }

        // The residual is quadratic in the amplitudes, `R = R_0 + R_1(t) + R_2(t, t)`, so the
        // exact Jacobian action is `J v = R_1(v) + [R_2(t + v) - R_2(t - v)] / 2`. The Newton
        // system is solved in the FOIS coordinates, `v = Y c` and `Y^\dagger J Y c`.
        let jacobian = |c: &Array1<T>| {
            let v = real_operator(|w| parallel_matvec(y, w), c);
            let linear = residual_at(&v, &[1]);
            let plus = residual_at(&(&amplitudes + &v), &[2]);
            let minus = residual_at(&(&amplitudes - &v), &[2]);
            let jv = linear + (plus - minus).mapv(|x| x * <T as From<f64>>::from(0.5));
            real_operator(|w| parallel_matvec(&yt, w), &jv)
        };

        // Solve `Y^\dagger J Y \delta c = -Y^\dagger R`; `\delta t = Y \delta c` stays in the FOIS.
        let (step, count, micro) = newton_step(&rfois, &jacobian, &preconditioner, options);
        println!("{row} {:>7} {:>12.4e}", count, micro);
        amplitudes = &amplitudes + &real_operator(|w| y.dot(w), &step);
    }

    solution
}

/// Solve the Newton equation `Y^\dagger J Y\,\delta c = -Y^\dagger R` in the FOIS coordinates by
/// restarted GMRES, right-preconditioned by the diagonal `M_{ii}`. The inexact step is solved
/// only to the fraction `FORCING` of `\lVert Y^\dagger R\rVert`, and never beyond the GMRES
/// residual RMS tolerance.
/// # Arguments:
/// - `rfois`: FOIS residual `Y^\dagger R`.
/// - `jacobian`: Action of the Jacobian `Y^\dagger J Y` on a vector of FOIS coordinates.
/// - `preconditioner`: Diagonal preconditioner `M_{ii}`.
/// - `options`: GMRES iteration limit, restart and tolerance.
/// # Returns:
/// - `(Array1<T>, usize, f64)`: Coordinate change `\delta c`, the number of GMRES iterations
///   performed and the final norm of `Y^\dagger R + Y^\dagger J Y\,\delta c`.
fn newton_step<T: NOCIScalar + Into<Complex64>>(
    rfois: &Array1<T>,
    jacobian: &impl Fn(&Array1<T>) -> Array1<T>,
    preconditioner: &Array1<f64>,
    options: &NOCCMCOptions,
) -> (Array1<T>, usize, f64) {
    // GMRES measures the residual RMS, so the relative tolerance is scaled by `\sqrt{n}`.
    let rms = (rfois.len() as f64).sqrt();
    let tol = (FORCING * vector_norm(rfois) / rms).max(options.gmres.res_tol);
    let precondition = |x: &Array1<T>| {
        Zip::from(x)
            .and(preconditioner)
            .map_collect(|&e, &d| e / <T as From<f64>>::from(d))
    };
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
