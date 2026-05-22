// scf/h/step.rs

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve;
use num_complex::Complex64;

use crate::AoData;
use crate::input::HSCFOptions;
use crate::scf::DensityMode;

use super::tangent::{geodesic_step, pack, unpack};
use super::types::SecantPair;
use crate::scf::{density, fock, orbital_gradient};

/// Solve the complex-symmetric SR1 quasi-Newton equation in energy-weighted coordinates.
/// # Arguments:
/// - `hist`: Stored unweighted secant pairs.
/// - `g`: Alpha- and beta-spin occupied-virtual gradients.
/// - `eps`: Alpha- and beta-spin pseudo-canonical orbital energies.
/// - `nocc`: Number of occupied alpha- and beta-spin orbitals.
/// - `opts`: h-SCF quasi-Newton options.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Unweighted alpha- and beta-spin orbital steps.
pub(crate) fn sr1_step(
    hist: &[SecantPair],
    g: (&Array2<Complex64>, &Array2<Complex64>),
    eps: (&Array1<Complex64>, &Array1<Complex64>),
    nocc: (usize, usize),
    opts: &HSCFOptions,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let (ga, gb) = g;
    let (epsa, epsb) = eps;
    let (na, nb) = nocc;

    // Convert current gradient into energy-weighted coordinates as
    // \bar g_{ai} = g_{ai} / \sqrt{\Delta{ai}} such that the true Hessian is
    // closer to the identity.
    let gpa = weight_by_gap(ga, epsa, na, opts.denom_tol, false);
    let gpb = weight_by_gap(gb, epsb, nb, opts.denom_tol, false);

    let n = gpa.len() + gpb.len();

    // Begin initial guess of the approximate Hessian as identity.
    let mut b = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        b[(i, i)] = Complex64::new(1.0, 0.0);
    }

    // For every secant pair stored update the approximated Hessian.
    for pair in hist {
        // Convert previous step into energy-weighted coordinates as \bar s_{ai} = s_{ai} \sqrt{\Delta_{ai}}.
        let sa = weight_by_gap(&pair.sa, epsa, na, opts.denom_tol, true);
        let sb = weight_by_gap(&pair.sb, epsb, nb, opts.denom_tol, true);

        // Convert previous gradient change into energy-weighted coordinates as \bar y_{ai} = y_{ai} / \sqrt{\Delta_{ai}}.
        let ya = weight_by_gap(&pair.ya, epsa, na, opts.denom_tol, false);
        let yb = weight_by_gap(&pair.yb, epsb, nb, opts.denom_tol, false);

        let s = pack(&sa, &sb);
        let y = pack(&ya, &yb);

        // Calculate residual error in current prediction.
        let r = &y - &b.dot(&s);

        // Update Hessian approximation as B_{k + 1} = B_k + (r_k r_k^T) / r_k^T s_k.
        let denom = r.dot(&s);
        if denom.norm() > opts.sr1_tol {
            let col = r.view().insert_axis(Axis(1));
            let row = r.view().insert_axis(Axis(0));
            b = b + col.dot(&row).mapv(|z| z / denom);
        }
    }

    // Solve B \bar p = - \bar g for energy-weighted occupied-virtual rotation amplitudes.
    let rhs = pack(&gpa, &gpb).mapv(|z| -z);
    let p = b.solve_into(rhs.clone()).unwrap_or(rhs);
    let (pa_bar, pb_bar) = unpack(&p, (ga.nrows(), ga.ncols()), (gb.nrows(), gb.ncols()));

    // Convert the solution back to unweighted occupied-virtual rotation coordinates.
    (
        weight_by_gap(&pa_bar, epsa, na, opts.denom_tol, false),
        weight_by_gap(&pb_bar, epsb, nb, opts.denom_tol, false),
    )
}

/// Backtrack along the complex-orthogonal geodesic and minimise gradient norm.
/// # Arguments:
/// - `c`: Current alpha- and beta-spin MO coefficient matrices.
/// - `ao`: Contains AO integrals and metadata.
/// - `nocc`: Number of occupied alpha- and beta-spin orbitals.
/// - `p`: Alpha- and beta-spin occupied-virtual steps.
/// - `g0`: Current occupied-virtual gradient norm.
/// - `opts`: h-SCF quasi-Newton options.
/// # Returns:
/// - `(f64, Array2<Complex64>, Array2<Complex64>)`: Step length and updated alpha/beta orbitals.
pub(crate) fn line_search(
    c: (&Array2<Complex64>, &Array2<Complex64>),
    ao: &AoData,
    nocc: (usize, usize),
    p: (&Array2<Complex64>, &Array2<Complex64>),
    g0: f64,
    opts: &HSCFOptions,
) -> (f64, Array2<Complex64>, Array2<Complex64>) {
    let (ca, cb) = c;
    let (na, nb) = nocc;
    let (pa, pb) = p;

    // Start with the full step length.
    let mut alpha = 1.0;
    let mut best = (0.0, ca.clone(), cb.clone(), g0);

    // Try a sequence of increasingly smaller step lengths.
    for _ in 0..opts.line_steps {
        // Try both directions along the geodesic.
        for sign in [1.0, -1.0] {
            let cat = geodesic_step(ca, pa, na, alpha * sign);
            let cbt = geodesic_step(cb, pb, nb, alpha * sign);

            let da = density(&cat, na, DensityMode::Holomorphic);
            let db = density(&cbt, nb, DensityMode::Holomorphic);

            let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);

            // Calculate orbital gradient g_{ai} = 2 \sum_{\mu\nu} C_a^\mu F_{\mu\nu} C_i^\nu.
            let ga = orbital_gradient(&cat, &fa, na, DensityMode::Holomorphic);
            let gb = orbital_gradient(&cbt, &fb, nb, DensityMode::Holomorphic);

            // Compare candidate steps using the real diagnostic norm of the h-SCF gradient.
            let g = (ga.iter().map(|z| z.norm_sqr()).sum::<f64>()
                + gb.iter().map(|z| z.norm_sqr()).sum::<f64>())
            .sqrt();

            if g < best.3 {
                best = (alpha * sign, cat.clone(), cbt.clone(), g);
            }
            if g < g0 {
                return (alpha * sign, cat, cbt);
            }
        }
        alpha *= opts.line_shrink;
    }

    (best.0, best.1, best.2)
}

/// Build and solve a finite-difference local Newton equation for stalled h-SCF iterations.
/// # Arguments:
/// - `ca`: Current alpha-spin MO coefficients.
/// - `cb`: Current beta-spin MO coefficients.
/// - `ao`: AO data.
/// - `na`: Number of occupied alpha-spin orbitals.
/// - `nb`: Number of occupied beta-spin orbitals.
/// - `ga`: Current alpha-spin occupied-virtual gradient.
/// - `gb`: Current beta-spin occupied-virtual gradient.
/// # Returns:
/// - `Option<(Array2<Complex64>, Array2<Complex64>)>`: Newton step if the linear solve succeeds.
pub(crate) fn finite_difference_newton_step(
    ca: &Array2<Complex64>,
    cb: &Array2<Complex64>,
    ao: &AoData,
    na: usize,
    nb: usize,
    ga: &Array2<Complex64>,
    gb: &Array2<Complex64>,
) -> Option<(Array2<Complex64>, Array2<Complex64>)> {
    let g0 = pack(ga, gb);
    let h = finite_difference_hessian(ca, cb, ao, na, nb, ga, gb);
    let rhs = g0.mapv(|z| -z);
    let p = h.solve_into(rhs).ok()?;
    Some(unpack(
        &p,
        (ga.nrows(), ga.ncols()),
        (gb.nrows(), gb.ncols()),
    ))
}

/// Build a finite-difference internal h-SCF Hessian in occupied-virtual coordinates.
/// # Arguments:
/// - `ca`: Current alpha-spin MO coefficients.
/// - `cb`: Current beta-spin MO coefficients.
/// - `ao`: AO data.
/// - `na`: Number of occupied alpha-spin orbitals.
/// - `nb`: Number of occupied beta-spin orbitals.
/// - `ga`: Current alpha-spin occupied-virtual gradient.
/// - `gb`: Current beta-spin occupied-virtual gradient.
/// # Returns:
/// - `Array2<Complex64>`: Finite-difference Jacobian of the h-SCF gradient.
pub(crate) fn finite_difference_hessian(
    ca: &Array2<Complex64>,
    cb: &Array2<Complex64>,
    ao: &AoData,
    na: usize,
    nb: usize,
    ga: &Array2<Complex64>,
    gb: &Array2<Complex64>,
) -> Array2<Complex64> {
    let g0 = pack(ga, gb);
    let n = g0.len();
    let eps = 1.0e-4;
    let mut h = Array2::<Complex64>::zeros((n, n));

    for j in 0..n {
        let mut va = Array2::<Complex64>::zeros(ga.raw_dim());
        let mut vb = Array2::<Complex64>::zeros(gb.raw_dim());

        if j < ga.len() {
            for (k, x) in va.iter_mut().enumerate() {
                if k == j {
                    *x = Complex64::new(eps, 0.0);
                    break;
                }
            }
        } else {
            let jb = j - ga.len();
            for (k, x) in vb.iter_mut().enumerate() {
                if k == jb {
                    *x = Complex64::new(eps, 0.0);
                    break;
                }
            }
        }

        let cat = geodesic_step(ca, &va, na, 1.0);
        let cbt = geodesic_step(cb, &vb, nb, 1.0);

        let da = density(&cat, na, DensityMode::Holomorphic);
        let db = density(&cbt, nb, DensityMode::Holomorphic);

        let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
        let gt_a = orbital_gradient(&cat, &fa, na, DensityMode::Holomorphic);
        let gt_b = orbital_gradient(&cbt, &fb, nb, DensityMode::Holomorphic);

        let dg = (pack(&gt_a, &gt_b) - &g0).mapv(|z| z / eps);
        h.column_mut(j).assign(&dg);
    }

    h
}

/// Apply or remove pseudo-canonical orbital-gap weighting:
///     \sqrt{\Delta_{ai}} = \sqrt{\tilde{F}_{aa} - \tilde{F}_{ii}}.
/// # Arguments:
/// - `x`: Occupied-virtual block.
/// - `eps`: Pseudo-canonical orbital energies.
/// - `nocc`: Number of occupied orbitals.
/// - `tol`: Minimum allowed gap magnitude.
/// - `multiply`: If true multiply by the gap square root, otherwise divide.
/// # Returns:
/// - `Array2<Complex64>`: Weighted or unweighted occupied-virtual block.
fn weight_by_gap(
    x: &Array2<Complex64>,
    eps: &Array1<Complex64>,
    nocc: usize,
    tol: f64,
    multiply: bool,
) -> Array2<Complex64> {
    let mut y = x.clone();

    for a in 0..x.nrows() {
        for i in 0..x.ncols() {
            // \Delta_{ai} = \tilde{F}_{aa} - \tilde{F}_{ii}.
            let mut gap = eps[nocc + a] - eps[i];

            if gap.norm() < tol {
                gap = Complex64::new(tol, 0.0);
            }

            let w = gap.sqrt();
            y[(a, i)] = if multiply {
                x[(a, i)] * w
            } else {
                x[(a, i)] / w
            };
        }
    }

    y
}

/// Limit combined alpha/beta occupied-virtual step norm.
/// # Arguments:
/// - `pa`: Alpha-spin occupied-virtual step.
/// - `pb`: Beta-spin occupied-virtual step.
/// - `max_step`: Maximum allowed combined step norm.
/// # Returns:
/// - `()`: Updates `pa` and `pb` in place.
pub(crate) fn limit_step(
    pa: &mut Array2<Complex64>,
    pb: &mut Array2<Complex64>,
    max_step: f64,
) {
    // ||p|| = \sqrt{\sum_{ai} |p_{ai}^\alpha|^2 + \sum_{ai} |p_{ai}^\beta|^2}.
    let n = step_norm(pa, pb);

    // If the proposed step size found by the SR1 solve is too big scale it down.
    if n > max_step && n > 0.0 {
        let scale = max_step / n;
        pa.mapv_inplace(|z| z * scale);
        pb.mapv_inplace(|z| z * scale);
    }
}

/// Combined alpha/beta occupied-virtual step norm.
/// # Arguments:
/// - `pa`: Alpha-spin occupied-virtual step.
/// - `pb`: Beta-spin occupied-virtual step.
/// # Returns:
/// - `f64`: Euclidean norm of the complex alpha/beta step blocks.
pub(crate) fn step_norm(
    pa: &Array2<Complex64>,
    pb: &Array2<Complex64>,
) -> f64 {
    (pa.iter().map(|z| z.norm_sqr()).sum::<f64>() + pb.iter().map(|z| z.norm_sqr()).sum::<f64>())
        .sqrt()
}
