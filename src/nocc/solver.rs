// nocc/solver.rs
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
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

// Crate-root imports.
use crate::input::NOCCMCOptions;
use crate::nocc::context::EvaluationContext;
use crate::nocc::dyall::{dyall_matrix, orbital_denominators};
use crate::nocc::energy::correlation_energy;
use crate::nocc::residual::residual_vector;
use crate::nocc::space::{FoisBasis, metric_projector};

/// Converged or final state of the amplitude equations.
pub(crate) struct AmplitudeSolution {
    /// Correlation energy `E - E_0`.
    pub(crate) correlation_energy: f64,
    /// Norm of the FOIS residual `\lVert Y^\dagger R\rVert`.
    pub(crate) residual_norm: f64,
    /// Number of macro-iterations performed.
    pub(crate) iterations: usize,
    /// Whether the residual norm fell below the tolerance.
    pub(crate) converged: bool,
}

/// Direct inversion in the iterative subspace for vector sequences.
struct VectorDiis {
    /// Maximum number of stored vectors.
    capacity: usize,
    /// Stored trial vectors.
    vectors: VecDeque<Array1<f64>>,
    /// Error vector of every trial vector.
    errors: VecDeque<Array1<f64>>,
}

impl VectorDiis {
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
        vector: Array1<f64>,
        error: Array1<f64>,
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
    /// - `Option<Array1<f64>>`: Extrapolated vector, or `None` with fewer than two vectors or a
    ///   singular DIIS system.
    fn extrapolate_vector(&self) -> Option<Array1<f64>> {
        let m = self.vectors.len();
        if m < 2 {
            return None;
        }

        // Augmented system `[B, -1; -1, 0][c; \lambda] = [0; -1]`, `B_{ij} = e_i \cdot e_j`.
        let mut b = Array2::<f64>::zeros((m + 1, m + 1));
        for i in 0..m {
            for j in 0..m {
                b[(i, j)] = self.errors[i].dot(&self.errors[j]);
            }
            b[(i, m)] = -1.0;
            b[(m, i)] = -1.0;
        }
        let mut rhs = Array1::<f64>::zeros(m + 1);
        rhs[m] = -1.0;

        let c = b.solve(&rhs).ok()?;
        let mut out = Array1::<f64>::zeros(self.vectors[0].len());
        for (ci, x) in c.iter().zip(&self.vectors) {
            out.scaled_add(*ci, x);
        }

        Some(out)
    }
}

/// Solve the GNOCC amplitude equations by macro-micro iteration.
/// Starting from `t = 0`, each macro-iteration evaluates `R_\mu` and `E - E_0`, solves the
/// linearised update equation for `\delta t` in micro-iterations, projects it with `P`, and
/// updates the amplitudes with DIIS extrapolation.
/// # Arguments:
/// - `ctx`: Reference evaluation context.
/// - `fois`: Weighted FOIS basis data.
/// - `e0`: Reference energy, used only for printing total energies.
/// - `options`: Iteration limits, tolerances, level shift and DIIS space.
/// # Returns:
/// - `AmplitudeSolution`: Final energy and convergence state.
pub(crate) fn solve_amplitudes(
    ctx: &EvaluationContext<'_>,
    fois: &FoisBasis,
    e0: f64,
    options: &NOCCMCOptions,
) -> AmplitudeSolution {
    let y = &fois.y;
    let n = ctx.excitations.len();

    // Fixed parts of the update: `Y Y^\dagger A`, the projector `P` and the shifted
    // denominators `\Delta_\nu + \eta`.
    let jacobian = y.dot(&y.t()).dot(&dyall_matrix(ctx));
    let projector = metric_projector(fois);
    let denominators = orbital_denominators(ctx) + options.level_shift;

    let mut amplitudes = Array1::<f64>::zeros(n);
    let mut diis = VectorDiis::new(options.diis_space);
    let mut previous = 0.0;
    let mut solution = AmplitudeSolution {
        correlation_energy: 0.0,
        residual_norm: f64::INFINITY,
        iterations: 0,
        converged: false,
    };

    println!("{}", "=".repeat(100));
    println!("GNOCC amplitude iterations");
    println!(
        "{:>4} {:>20} {:>20} {:>12} {:>7} {:>12}",
        "i", "E", "dE", "||Y^T R||", "micro", "||micro||"
    );

    for iteration in 0..options.max_macro {
        // Energy and FOIS residual `R_i = \sum_\mu Y^\dagger_{i\mu} R_\mu` at the current amplitudes.
        let dense = ctx.dense_amplitudes(&amplitudes);
        let residual = residual_vector(ctx, &dense);
        let ecorr = correlation_energy(ctx, &dense);
        let rfois = y.t().dot(&residual);
        let norm = rfois.dot(&rfois).sqrt();

        let converged = norm < options.residual_tol;
        solution = AmplitudeSolution {
            correlation_energy: ecorr,
            residual_norm: norm,
            iterations: iteration,
            converged,
        };
        let row = format!(
            "{:>4} {:>20.12} {:>20.12e} {:>12.4e}",
            iteration,
            e0 + ecorr,
            ecorr - previous,
            norm
        );
        previous = ecorr;
        if converged {
            println!("{row} {:>7} {:>12}", "-", "-");
            break;
        }

        // Solve the linearised update for `\tilde R = Y Y^\dagger R` and project it onto the FOIS.
        let rtilde = y.dot(&rfois);
        let (step, count, micro) = micro_iterations(&rtilde, &jacobian, &denominators, options);
        println!("{row} {:>7} {:>12.4e}", count, micro);
        let step = projector.dot(&step);

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
/// - `jacobian`: Zeroth-order coupling `Y Y^\dagger A`.
/// - `denominators`: Shifted orbital-energy denominators `\Delta_\nu + \eta`.
/// - `options`: Micro-iteration limit, tolerance and DIIS space.
/// # Returns:
/// - `(Array1<f64>, usize, f64)`: Amplitude change `\delta t` in the raw excitation basis, the
///   number of micro-iterations performed and the final norm of `\tilde R + Y Y^\dagger A\,\delta t`.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eqs. (58)-(62).
fn micro_iterations(
    rtilde: &Array1<f64>,
    jacobian: &Array2<f64>,
    denominators: &Array1<f64>,
    options: &NOCCMCOptions,
) -> (Array1<f64>, usize, f64) {
    let mut step = Array1::<f64>::zeros(rtilde.len());
    let mut diis = VectorDiis::new(options.diis_space);
    let mut count = 0;
    let mut norm = rtilde.dot(rtilde).sqrt();

    for iteration in 0..options.max_micro {
        let error = rtilde + &jacobian.dot(&step);
        norm = error.dot(&error).sqrt();
        count = iteration;
        if norm < options.micro_tol {
            break;
        }

        let change = &error / denominators;
        let next = &step - &change;
        diis.push_trial(next.clone(), change);
        step = diis.extrapolate_vector().unwrap_or(next);
    }

    (step, count, norm)
}
