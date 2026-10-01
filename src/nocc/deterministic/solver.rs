// nocc/deterministic/solver.rs
//! Macro-micro iterative solution of the GNOCC amplitude equations.
//!
//! Each macro-iteration evaluates the energy and the residual `R_\mu` in the linearly dependent
//! excitation basis and projects the residual onto the FOIS, `\tilde R = Y Y^\dagger R`. The
//! amplitude change then solves the linearised equation
//!
//! `\tilde R_\sigma + \sum_{i\mu\nu} Y_{\sigma i}Y^\dagger_{i\mu}
//! \langle\Phi|\hat\tau_\mu^\dagger\hat H_0\hat\tau_\nu|\Phi\rangle_c\,\delta t_\nu = 0`
//!
//! by micro-iterations preconditioned with level-shifted orbital-energy denominators, and is
//! projected by `P = Y Y^\dagger S` so the amplitudes stay in the FOIS. DIIS accelerates both
//! the macro- and micro-iterations.
//!
//! # References
//!
//! - Lee and Tew, *Spin-free Generalised Normal Ordered Coupled Cluster*, arXiv:2507.13472
//!   (2025), Secs. II F-H, Eqs. (54)-(64).
//! - Pulay, *Chem. Phys. Lett.* **73**, 393 (1980),
//!   [doi:10.1016/0009-2614(80)80396-4](https://doi.org/10.1016/0009-2614(80)80396-4).

// Standard library imports.
use std::collections::VecDeque;

// External crate imports.
use ndarray::{Array1, Array2, Zip};
use ndarray_linalg::Solve;
use num_complex::Complex64;

// Crate-root imports.
use crate::NOCIScalar;
use crate::input::NOCCMCOptions;
use crate::maths::parallel_matvec;
use crate::nocc::equations::{
    correlation_energy, dyall_matrix, orbital_denominators, residual_vector,
};
use crate::nocc::setup::ReferenceState;
use crate::nocc::space::{Excitation, FoisBasis, Spaces, dense_amplitudes, project_onto_fois};
use crate::nocc::terms::TermEvaluator;

/// Norm of the imaginary amplitude vector that starts a holomorphic run.
const SEED: f64 = 1e-3;

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

/// Direct inversion in the iterative subspace for vector sequences. Products of error vectors are
/// bilinear, without complex conjugation, so the extrapolation stays holomorphic in complex
/// amplitudes.
struct VectorDiis<T> {
    /// Maximum number of stored vectors.
    capacity: usize,
    /// Stored trial vectors.
    vectors: VecDeque<Array1<T>>,
    /// Error vector of every trial vector.
    errors: VecDeque<Array1<T>>,
}

impl<T: NOCIScalar> VectorDiis<T> {
    /// Build an empty DIIS subspace.
    /// # Arguments:
    /// - `capacity`: Maximum number of stored vectors.
    /// # Returns:
    /// - `Self`: Empty subspace.
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            vectors: VecDeque::new(),
            errors: VecDeque::new(),
        }
    }

    /// Add one trial vector and its error vector, discarding the oldest pair when full.
    /// # Arguments:
    /// - `vector`: Trial vector.
    /// - `error`: Error vector of the trial vector.
    /// # Returns:
    /// - `()`: Mutates the subspace.
    fn push_trial(
        &mut self,
        vector: Array1<T>,
        error: Array1<T>,
    ) {
        if self.vectors.len() == self.capacity {
            self.vectors.pop_front();
            self.errors.pop_front();
        }
        self.vectors.push_back(vector);
        self.errors.push_back(error);
    }

    /// Extrapolate the stored vectors, `x = \sum_i c_i x_i`, minimising `\lVert\sum_i c_i e_i\rVert`
    /// subject to `\sum_i c_i = 1`.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Option<Array1<T>>`: Extrapolated vector, or `None` with fewer than two vectors or a
    ///   singular DIIS system.
    fn extrapolate_vector(&self) -> Option<Array1<T>> {
        let m = self.vectors.len();
        if m < 2 {
            return None;
        }

        // Augmented system `[B, -1; -1, 0][c; \lambda] = [0; -1]`, `B_{ij} = e_i \cdot e_j`.
        let one = <T as From<f64>>::from(1.0);
        let mut b = Array2::<T>::zeros((m + 1, m + 1));
        for i in 0..m {
            for j in 0..m {
                b[(i, j)] = self.errors[i].dot(&self.errors[j]);
            }
            b[(i, m)] = -one;
            b[(m, i)] = -one;
        }
        let mut rhs = Array1::<T>::zeros(m + 1);
        rhs[m] = -one;

        let c = b.solve(&rhs).ok()?;
        let mut out = Array1::<T>::zeros(self.vectors[0].len());
        for (ci, x) in c.iter().zip(&self.vectors) {
            out.scaled_add(*ci, x);
        }

        Some(out)
    }
}

/// Solve the GNOCC amplitude equations by macro-micro iteration.
/// Starting from `t = 0`, each macro-iteration evaluates `R_\mu` and `E - E_0`, solves the
/// linearised update equation for `\delta t` in micro-iterations, projects it with `P`, and
/// updates the amplitudes with DIIS extrapolation. A holomorphic run iterates complex
/// amplitudes in the same equations, without complex conjugation.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `fois`: Weighted FOIS basis data.
/// - `e0`: Reference energy, used only for printing total energies.
/// - `options`: Iteration limits, tolerances, level shift, DIIS space and holomorphic switch.
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

/// Run the macro-micro iteration in the amplitude scalar type `T`, real or, for a holomorphic run,
/// complex. A holomorphic run starts from a small imaginary amplitude vector inside the FOIS, so
/// the iteration can follow the solution onto the complex plane where no real solution exists;
/// where one does, the imaginary part decays.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `fois`: Weighted FOIS basis data.
/// - `e0`: Reference energy, used only for printing total energies.
/// - `options`: Iteration limits, tolerances, level shift and DIIS space.
/// # Returns:
/// - `AmplitudeSolution`: Final energy and convergence state.
/// # Panics
/// - Panics if a holomorphic run is iterated in real arithmetic.
fn iterate_amplitudes<T: NOCIScalar>(
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

    // Fixed parts of the update: the zeroth-order coupling `A`, applied as `Y Y^\dagger A`, and
    // the shifted denominators `\Delta_\nu + \eta`.
    let dyall = dyall_matrix(reference, spaces, excitations, evaluator);
    let yt = y.t().to_owned();
    let jacobian = |x: &Array1<T>| {
        real_operator(
            |v| parallel_matvec(y, &parallel_matvec(&yt, &parallel_matvec(&dyall, v))),
            x,
        )
    };
    let denominators = orbital_denominators(reference, excitations) + options.level_shift;

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
    let mut diis = VectorDiis::new(options.diis_space);
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
            "i", "Re E", "Im E", "Re dE", "||Y^T R||", "micro", "||micro||"
        );
    } else {
        println!(
            "{:>4} {:>20} {:>20} {:>12} {:>7} {:>12}",
            "i", "E", "dE", "||Y^T R||", "micro", "||micro||"
        );
    }

    for iteration in 0..options.max_macro {
        // Energy and FOIS residual `R_i = \sum_\mu Y^\dagger_{i\mu} R_\mu` at the current amplitudes.
        let dense = dense_amplitudes(spaces, excitations, &amplitudes);
        let residual = residual_vector(reference, spaces, excitations, evaluator, &dense);
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

        // Solve the linearised update for `\tilde R = Y Y^\dagger R` and project it onto the FOIS.
        let rtilde = real_operator(|v| y.dot(v), &rfois);
        let (step, count, micro) = micro_iterations(&rtilde, &jacobian, &denominators, options);
        println!("{row} {:>7} {:>12.4e}", count, micro);
        let step = real_operator(|v| project_onto_fois(fois, v), &step);

        // DIIS over the updated amplitudes with the projected step as error vector.
        let next = &amplitudes + &step;
        diis.push_trial(next.clone(), step);
        amplitudes = diis.extrapolate_vector().unwrap_or(next);
    }

    solution
}

/// Solve the linearised update equation `\tilde R + Y Y^\dagger A\,\delta t = 0` iteratively.
/// Each step subtracts `\delta^{(2)} t_\nu = (\tilde R_\nu + [Y Y^\dagger A\,\delta t]_\nu) /
/// (\Delta_\nu + \eta)`, accelerated by DIIS on the steps.
/// # Arguments:
/// - `rtilde`: FOIS-projected residual `\tilde R`.
/// - `jacobian`: Action of the zeroth-order coupling `Y Y^\dagger A` on a vector.
/// - `denominators`: Shifted orbital-energy denominators `\Delta_\nu + \eta`.
/// - `options`: Micro-iteration limit, tolerance and DIIS space.
/// # Returns:
/// - `(Array1<T>, usize, f64)`: Amplitude change `\delta t` in the raw excitation basis, the
///   number of micro-iterations performed and the final norm of `\tilde R + Y Y^\dagger A\,\delta t`.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eqs. (58)-(62).
fn micro_iterations<T: NOCIScalar>(
    rtilde: &Array1<T>,
    jacobian: &impl Fn(&Array1<T>) -> Array1<T>,
    denominators: &Array1<f64>,
    options: &NOCCMCOptions,
) -> (Array1<T>, usize, f64) {
    let mut step = Array1::<T>::zeros(rtilde.len());
    let mut diis = VectorDiis::new(options.diis_space);
    let mut count = 0;
    let mut norm = vector_norm(rtilde);

    for iteration in 0..options.max_micro {
        let error = rtilde + &jacobian(&step);
        norm = vector_norm(&error);
        count = iteration;
        if norm < options.micro_tol {
            break;
        }

        let change = Zip::from(&error)
            .and(denominators)
            .map_collect(|&e, &d| e / <T as From<f64>>::from(d));
        let next = &step - &change;
        diis.push_trial(next.clone(), change);
        step = diis.extrapolate_vector().unwrap_or(next);
    }

    (step, count, norm)
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
