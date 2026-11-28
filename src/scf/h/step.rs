// scf/h/step.rs

// External crate imports.
use ndarray::{Array1, Array2, Axis, stack};
use ndarray_linalg::Solve;
use num_complex::Complex64;

// Crate-root imports.
use crate::AoData;
use crate::maths::gmres::{GMRESPrint, HostKrylov, gmres};
use crate::scf::DensityMode;
use crate::scf::{density, fock_lambda, orbital_gradient};

// Parent/sibling imports.
use super::tangent::{geodesic_step, pack, unpack};
use super::types::SecantPair;

const SR1_TOL: f64 = 1e-8;
const DENOM_TOL: f64 = 1e-8;
const MAX_STEP: f64 = 0.5;
const LINE_STEPS: usize = 100;
const LINE_SHRINK: f64 = 0.5;

/// Solve the complex-symmetric SR1 quasi-Newton equation in energy-weighted coordinates.
/// # Arguments:
/// - `hist`: Stored unweighted secant pairs.
/// - `g`: Alpha- and beta-spin occupied-virtual gradients.
/// - `eps`: Alpha- and beta-spin pseudo-canonical orbital energies.
/// - `nocc`: Number of occupied alpha- and beta-spin orbitals.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Unweighted alpha- and beta-spin orbital steps.
pub(crate) fn sr1_step(
    hist: &[SecantPair],
    g: (&Array2<Complex64>, &Array2<Complex64>),
    eps: (&Array1<Complex64>, &Array1<Complex64>),
    nocc: (usize, usize),
) -> (Array2<Complex64>, Array2<Complex64>) {
    let (ga, gb) = g;
    let (epsa, epsb) = eps;
    let (na, nb) = nocc;

    // Convert current gradient into energy-weighted coordinates as
    // \bar g_{ai} = g_{ai} / \sqrt{\Delta{ai}} such that the true Hessian is
    // closer to the identity.
    let gpa = weight_by_gap(ga, epsa, na, DENOM_TOL, false);
    let gpb = weight_by_gap(gb, epsb, nb, DENOM_TOL, false);

    // The SR1 approximation `B_m = I + \sum_j r_j r_j^T / d_j` is kept as its columns `r_j` and
    // denominators `d_j`, never as an `n \times n` matrix. For every pair in turn,
    // `r_j = y_j - B_{j - 1} s_j` with `B_{j - 1} s_j = s_j + \sum_{i < j} r_i (r_i^T s_j) / d_i`
    // and `d_j = r_j^T s_j`, skipping the update when `d_j` is small.
    let mut rs: Vec<Array1<Complex64>> = Vec::new();
    let mut ds: Vec<Complex64> = Vec::new();
    for pair in hist {
        // Convert previous step into energy-weighted coordinates as \bar s_{ai} = s_{ai} \sqrt{\Delta_{ai}}.
        let sa = weight_by_gap(&pair.sa, epsa, na, DENOM_TOL, true);
        let sb = weight_by_gap(&pair.sb, epsb, nb, DENOM_TOL, true);

        // Convert previous gradient change into energy-weighted coordinates as \bar y_{ai} = y_{ai} / \sqrt{\Delta_{ai}}.
        let ya = weight_by_gap(&pair.ya, epsa, na, DENOM_TOL, false);
        let yb = weight_by_gap(&pair.yb, epsb, nb, DENOM_TOL, false);

        let s = pack(&sa, &sb);
        let y = pack(&ya, &yb);

        // Calculate residual error in current prediction, `r = y - B s`.
        let mut r = &y - &s;
        for (ri, &di) in rs.iter().zip(&ds) {
            r.scaled_add(-ri.dot(&s) / di, ri);
        }

        let denom = r.dot(&s);
        if denom.norm() > SR1_TOL {
            rs.push(r);
            ds.push(denom);
        }
    }

    // Solve B \bar p = - \bar g for energy-weighted occupied-virtual rotation amplitudes by the
    // Woodbury identity `B^{-1} = I - R (D + R^T R)^{-1} R^T`, with the columns `r_j` in `R` and
    // `D = \mathrm{diag}(d_j)`. `B` is singular exactly when `D + R^T R` is, and then the step is
    // `-\bar g`.
    let g = pack(&gpa, &gpb);
    let rhs = g.mapv(|z| -z);
    let p = if rs.is_empty() {
        rhs
    } else {
        let views = rs.iter().map(|r| r.view()).collect::<Vec<_>>();
        let r = stack(Axis(1), &views).unwrap();
        let m = Array2::from_diag(&Array1::from(ds)) + r.t().dot(&r);

        match m.solve_into(r.t().dot(&g)) {
            Ok(c) => -(g - r.dot(&c)),
            Err(_) => rhs,
        }
    };

    let (pa_bar, pb_bar) = unpack(&p, (ga.nrows(), ga.ncols()), (gb.nrows(), gb.ncols()));

    // Convert the solution back to unweighted occupied-virtual rotation coordinates.
    (
        weight_by_gap(&pa_bar, epsa, na, DENOM_TOL, false),
        weight_by_gap(&pb_bar, epsb, nb, DENOM_TOL, false),
    )
}

/// Backtrack along the complex-orthogonal geodesic and minimise gradient norm.
/// # Arguments:
/// - `c`: Current alpha- and beta-spin MO coefficient matrices.
/// - `ao`: Contains AO integrals and metadata.
/// - `nocc`: Number of occupied alpha- and beta-spin orbitals.
/// - `p`: Alpha- and beta-spin occupied-virtual steps.
/// - `g0`: Current occupied-virtual gradient norm.
/// - `lambda`: Current holomorphic continuation parameter `\lambda`.
/// # Returns:
/// - `(f64, Array2<Complex64>, Array2<Complex64>)`: Step length and updated alpha/beta orbitals.
pub(crate) fn line_search(
    c: (&Array2<Complex64>, &Array2<Complex64>),
    ao: &AoData,
    nocc: (usize, usize),
    p: (&Array2<Complex64>, &Array2<Complex64>),
    g0: f64,
    lambda: Complex64,
) -> (f64, Array2<Complex64>, Array2<Complex64>) {
    let (ca, cb) = c;
    let (na, nb) = nocc;
    let (pa, pb) = p;

    // Start with the full step length.
    let mut alpha = 1.0;
    let mut best = (0.0, ca.clone(), cb.clone(), g0);

    // Try a sequence of increasingly smaller step lengths.
    for _ in 0..LINE_STEPS {
        // Try both directions along the geodesic.
        for sign in [1.0, -1.0] {
            let cat = geodesic_step(ca, pa, na, alpha * sign);
            let cbt = geodesic_step(cb, pb, nb, alpha * sign);

            let da = density(&cat, na, DensityMode::Holomorphic);
            let db = density(&cbt, nb, DensityMode::Holomorphic);

            let (fa, fb) = fock_lambda(&ao.h, &ao.eri_coul, &da, &db, lambda);

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
        alpha *= LINE_SHRINK;
    }

    (best.0, best.1, best.2)
}

/// Solve the local Newton equation `H p = -g` of a stalled h-SCF iteration by restarted GMRES.
/// Every Hessian product is one forward difference of the gradient along the orbital geodesic,
/// `H v \approx [g(C(\epsilon \hat v)) - g(C)] \lVert v \rVert / \epsilon` with
/// `\hat v = v / \lVert v \rVert`, so it costs one Fock build, and the orbital gaps
/// `\Delta_{ai} = \tilde F_{aa} - \tilde F_{ii}` precondition the solve. The step is solved only to
/// a modest relative accuracy, since the line search judges it.
/// # Arguments:
/// - `c`: Current `\alpha`- and `\beta`-spin MO coefficients.
/// - `ao`: AO data.
/// - `nocc`: Numbers of occupied `\alpha`- and `\beta`-spin orbitals.
/// - `g`: Current `\alpha`- and `\beta`-spin occupied-virtual gradients.
/// - `eps`: `\alpha`- and `\beta`-spin pseudo-canonical orbital energies.
/// - `lambda`: Current holomorphic continuation parameter `\lambda`.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: `\alpha`- and `\beta`-spin Newton steps.
pub(crate) fn finite_difference_newton_step(
    c: (&Array2<Complex64>, &Array2<Complex64>),
    ao: &AoData,
    nocc: (usize, usize),
    g: (&Array2<Complex64>, &Array2<Complex64>),
    eps: (&Array1<Complex64>, &Array1<Complex64>),
    lambda: Complex64,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let (ca, cb) = c;
    let (na, nb) = nocc;
    let (ga, gb) = g;
    let (epsa, epsb) = eps;
    let shapes = ((ga.nrows(), ga.ncols()), (gb.nrows(), gb.ncols()));
    let g0 = pack(ga, gb);

    // Finite-difference displacement `\epsilon`, Krylov dimension and iteration limit, and the
    // residual of the solve relative to `g`.
    let step = 1.0e-4;
    let krylov = 64;
    let relative = 1e-3;

    // `H v` from the gradient at orbitals displaced by `\epsilon \hat v` along the geodesic, so
    // the trial MO frames keep their orthonormality.
    let hessian = |v: &Array1<Complex64>| {
        let norm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Array1::zeros(v.len());
        }

        let (va, vb) = unpack(&v.mapv(|z| z * (step / norm)), shapes.0, shapes.1);
        let cat = geodesic_step(ca, &va, na, 1.0);
        let cbt = geodesic_step(cb, &vb, nb, 1.0);

        // Holomorphic densities and Fock matrices at the displaced orbitals.
        let da = density(&cat, na, DensityMode::Holomorphic);
        let db = density(&cbt, nb, DensityMode::Holomorphic);
        let (fa, fb) = fock_lambda(&ao.h, &ao.eri_coul, &da, &db, lambda);

        // `H v = [g(C(\epsilon \hat v)) - g(C)] \lVert v \rVert / \epsilon`.
        let gt_a = orbital_gradient(&cat, &fa, na, DensityMode::Holomorphic);
        let gt_b = orbital_gradient(&cbt, &fb, nb, DensityMode::Holomorphic);

        (pack(&gt_a, &gt_b) - &g0).mapv(|z| z * (norm / step))
    };

    // `v_{ai} / \Delta_{ai}`, dividing twice by `\sqrt{\Delta_{ai}}`.
    let precondition = |v: &Array1<Complex64>| {
        let (va, vb) = unpack(v, shapes.0, shapes.1);
        let gap = |x: &Array2<Complex64>, eps: &Array1<Complex64>, nocc: usize| {
            let x = weight_by_gap(x, eps, nocc, DENOM_TOL, false);
            weight_by_gap(&x, eps, nocc, DENOM_TOL, false)
        };

        pack(&gap(&va, epsa, na), &gap(&vb, epsb, nb))
    };

    // GMRES accepts a root-mean-square residual, here `relative` times that of `g`.
    let gnorm = g0.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let tol = relative * gnorm / (g0.len() as f64).sqrt();

    let rhs = g0.mapv(|z| -z);
    let result = gmres(
        &HostKrylov,
        hessian,
        precondition,
        (&rhs, None),
        krylov,
        krylov,
        tol,
        GMRESPrint::Silent,
    );

    unpack(&result.x, shapes.0, shapes.1)
}

/// Apply or remove pseudo-canonical orbital-gap weighting:
///     `\sqrt{\Delta_{ai}} = \sqrt{\tilde{F}_{aa} - \tilde{F}_{ii}}.`
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
) {
    // ||p|| = \sqrt{\sum_{ai} |p_{ai}^\alpha|^2 + \sum_{ai} |p_{ai}^\beta|^2}.
    let n = step_norm(pa, pb);

    // If the proposed step size found by the SR1 solve is too big scale it down.
    if n > MAX_STEP {
        let scale = MAX_STEP / n;
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
