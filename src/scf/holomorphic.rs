// scf/holomorphic.rs

use std::sync::Arc;

use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::Solve;
use num_complex::Complex64;

use crate::{AoData, Excitation, ExcitationSpin, HSCFState, SCFState};
use crate::input::{HSCFOptions, Input, StateRecipe};
use crate::scf::DensityMode;

use crate::maths::{symmetric_evp_complex, real2_as, matrix_exp_complex};
use super::{density, energy, fock, orbital_energies, orbital_gradient};
use super::print::print_header_h;
use crate::utils::print_array2_indexed;

/// Stored quasi-Newton secant pair in the current local tangent basis.
#[derive(Clone, Debug)]
struct SecantPair {
    /// Previous alpha-spin accepted step in unweighted occupied-virtual rotation coordinates.
    sa: Array2<Complex64>,
    /// Previous beta-spin accepted step in unweighted occupied-virtual rotation coordinates.
    sb: Array2<Complex64>,
    /// Previous alpha-spin gradient change in unweighted occupied-virtual coordinates.
    ya: Array2<Complex64>,
    /// Previous beta-spin gradient change in unweighted occupied-virtual coordinates.
    yb: Array2<Complex64>,
}

/// Spin block being pseudo-canonicalised.
#[derive(Clone, Copy, Debug)]
enum SpinBlock {
    /// Alpha-spin orbital block.
    Alpha,
    /// Beta-spin orbital block.
    Beta,
}

/// Build a complex h-SCF state from a real SCF seed.
/// # Arguments:
/// - `seed`: Real SCF state used as the initial h-SCF determinant.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: User input specifications.
/// - `label`: Label for the h-SCF state.
/// - `noci_basis`: Whether this state is intended for a later NOCI basis.
/// - `i`: State index.
/// # Returns:
/// - `Option<HSCFState>`: Converged h-SCF state if optimisation succeeds.
pub fn hscf_from_real_state(seed: &SCFState, ao: &AoData, input: &Input, label: &str, noci_basis: bool, i: usize) -> Option<HSCFState> {
    // Put occupied orbitals first because the h-SCF local coordinates assume this ordering.
    let idx_a: Vec<usize> = (0..seed.ca.ncols()).filter(|&p| ((seed.oa >> p) & 1u128) == 1).chain((0..seed.ca.ncols()).filter(|&p| ((seed.oa >> p) & 1u128) == 0)).collect();
    let idx_b: Vec<usize> = (0..seed.cb.ncols()).filter(|&p| ((seed.ob >> p) & 1u128) == 1).chain((0..seed.cb.ncols()).filter(|&p| ((seed.ob >> p) & 1u128) == 0)).collect();

    let ca = real2_as::<Complex64>(&seed.ca).select(Axis(1), &idx_a);
    let cb = real2_as::<Complex64>(&seed.cb).select(Axis(1), &idx_b);

    hscf_cycle(&ca, &cb, ao, input, label, noci_basis, i)
}

/// Run a holomorphic unrestricted SCF quasi-Newton optimisation.
/// # Arguments:
/// - `ca0`: Initial alpha-spin MO coefficients ordered as occupied then virtual.
/// - `cb0`: Initial beta-spin MO coefficients ordered as occupied then virtual.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: User input specifications.
/// - `label`: Label for the h-SCF state.
/// - `noci_basis`: Whether this state is intended for a later NOCI basis.
/// - `i`: State index.
/// # Returns:
/// - `Option<HSCFState>`: Converged h-SCF state if optimisation succeeds.
pub fn hscf_cycle(ca0: &Array2<Complex64>, cb0: &Array2<Complex64>, ao: &AoData, input: &Input, label: &str, noci_basis: bool, i: usize) -> Option<HSCFState> {
    let na = usize::try_from(ao.nelec[0]).unwrap(); let nb = usize::try_from(ao.nelec[1]).unwrap();
    let opts = &input.scf.h;
    let mut ca = ca0.clone(); let mut cb = cb0.clone();
    let mut hist: Vec<SecantPair> = Vec::new();

    print_header_h(input, label);

    let mut g_prev: Option<(Array2<Complex64>, Array2<Complex64>)> = None;
    let mut step_prev: Option<(Array2<Complex64>, Array2<Complex64>)> = None;

    for iter in 0..opts.max_cycle {
        let da = density(&ca, na, DensityMode::Holomorphic);
        let db = density(&cb, nb, DensityMode::Holomorphic);
        let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);

        let mut extra_a: Vec<&mut Array2<Complex64>> = Vec::new();
        if let Some((sa, _)) = step_prev.as_mut() {
            extra_a.push(sa);
        }
        if let Some((ga, _)) = g_prev.as_mut() {
            extra_a.push(ga);
        }
        pseudo_canonicalise(&mut ca, &fa, na, &mut hist, SpinBlock::Alpha, &mut extra_a);

        let mut extra_b: Vec<&mut Array2<Complex64>> = Vec::new();
        if let Some((_, sb)) = step_prev.as_mut() {
            extra_b.push(sb);
        }
        if let Some((_, gb)) = g_prev.as_mut() {
            extra_b.push(gb);
        }
        pseudo_canonicalise(&mut cb, &fb, nb, &mut hist, SpinBlock::Beta, &mut extra_b);

        let da = density(&ca, na, DensityMode::Holomorphic);
        let db = density(&cb, nb, DensityMode::Holomorphic);
        let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
        let e = energy(&ao.h, ao.enuc, &da, &db, &fa, &fb);
        let epsa = orbital_energies(&ca, &fa, DensityMode::Holomorphic);
        let epsb = orbital_energies(&cb, &fb, DensityMode::Holomorphic);

        // Calculate g_{ai} = 2 \sum_{\mu\nu} C_a^\mu F_{\mu\nu} C_i^\nu.
        let ga = orbital_gradient(&ca, &fa, na, DensityMode::Holomorphic);
        let gb = orbital_gradient(&cb, &fb, nb, DensityMode::Holomorphic);

        // Use the Euclidean Frobenius norm only as a real convergence diagnostic.
        let gnorm = (ga.iter().map(|z| z.norm_sqr()).sum::<f64>() + gb.iter().map(|z| z.norm_sqr()).sum::<f64>()).sqrt();

        if gnorm < opts.g_tol {
            if input.write.verbose {
                println!("{:4} {:16.10} {:+16.10}i {:12.4e} {:>12} {:>12}", iter, e.re, e.im, gnorm, "-", "-");
            }
            return Some(finalise(ca, cb, ao, input, label, noci_basis, i, true));
        }

        if let (Some((sa, sb)), Some((gpa, gpb))) = (step_prev.take(), g_prev.take()) {
            let ya = &ga - &gpa;
            let yb = &gb - &gpb;

            hist.push(SecantPair {sa, sb, ya, yb});
            if hist.len() > opts.history {
                hist.remove(0);
            }
        }

        let (mut pa, mut pb) = sr1_step(&hist, &ga, &gb, &epsa, &epsb, na, nb, opts);
        limit_step(&mut pa, &mut pb, opts.max_step);
        let pnorm = step_norm(&pa, &pb);

        let (alpha, ca_new, cb_new) = line_search(&ca, &cb, ao, na, nb, &pa, &pb, gnorm, opts);
        if input.write.verbose {
            println!("{:4} {:16.10} {:+16.10}i {:12.4e} {:12.4e} {:12.4e}", iter, e.re, e.im, gnorm, alpha, pnorm);
        }
        if alpha == 0.0 {
            if input.write.verbose {
                println!("h-SCF line search found no improving step.");
            }
            finalise(ca_new, cb_new, ao, input, label, noci_basis, i, false);
            return None;
        }

        let pa_acc = pa.mapv(|z| z * alpha);
        let pb_acc = pb.mapv(|z| z * alpha);

        // Store the accepted unweighted displacement and gradient for the next SR1 secant pair.
        step_prev = Some((pa_acc, pb_acc));
        g_prev = Some((ga, gb));
        ca = ca_new; cb = cb_new;
    }

    finalise(ca, cb, ao, input, label, noci_basis, i, false);
    None
}

/// Pseudo-canonicalise occupied and virtual spaces for one spin block.
/// # Arguments:
/// - `c`: MO coefficient matrix ordered as occupied then virtual.
/// - `f`: Spin Fock matrix.
/// - `nocc`: Number of occupied orbitals.
/// - `hist`: Stored SR1 secant pairs transformed into the new tangent basis.
/// - `spin`: Spin block being transformed.
/// - `extra`: Additional tangent matrices transformed into the new tangent basis.
/// # Returns:
/// - `Array1<Complex64>`: Occupied followed by virtual pseudo-canonical orbital energies.
fn pseudo_canonicalise(c: &mut Array2<Complex64>, f: &Array2<Complex64>, nocc: usize, hist: &mut [SecantPair], spin: SpinBlock, 
                       extra: &mut [&mut Array2<Complex64>]) -> Array1<Complex64> {
    let n = c.ncols();
    // Transform Fock matrix into MO basis such that we have o-o, o-v, v-o, v-v blocks.
    let fmo = c.t().dot(f).dot(c);
    // Diagonalise o-o and v-v blocks.
    let (eo, uo) = symmetric_evp_complex(&fmo.slice(s![0..nocc, 0..nocc]).to_owned());
    let (ev, uv) = symmetric_evp_complex(&fmo.slice(s![nocc..n, nocc..n]).to_owned());

    // Rotate orbitals within occupied and virtual spaces as 
    // C_o = C_o U_o, C_v = C_v U_v where U are eigenvectors.
    let cocc = c.slice(s![.., 0..nocc]).to_owned().dot(&uo);
    let cvir = c.slice(s![.., nocc..n]).to_owned().dot(&uv);
    c.slice_mut(s![.., 0..nocc]).assign(&cocc);
    c.slice_mut(s![.., nocc..n]).assign(&cvir);

    // Transform the stored SR1 secant-pair history into the new pseudo-canonical tangent basis.
    for pair in hist.iter_mut() {
        match spin {
            SpinBlock::Alpha => {
                // Transform the previous alpha-spin step as s_ai -> (U_v^T s U_o)_ai.
                pair.sa = uv.t().dot(&pair.sa).dot(&uo);
                // Transform the previous alpha-spin gradient change as y_ai -> (U_v^T y U_o)_ai.
                pair.ya = uv.t().dot(&pair.ya).dot(&uo);
            }
            SpinBlock::Beta => {
                // Transform the previous beta-spin step as s_ai -> (U_v^T s U_o)_ai.
                pair.sb = uv.t().dot(&pair.sb).dot(&uo);
                // Transform the previous beta-spin gradient change as y_ai -> (U_v^T y U_o)_ai.
                pair.yb = uv.t().dot(&pair.yb).dot(&uo);
            }
        }
    }

    // Transform any additional tangent-space matrices into the new pseudo-canonical basis.
    for x in extra.iter_mut() {
        // Transform the extra occupied-virtual block as X_ai -> (U_v^T X U_o)_ai.
        **x = uv.t().dot(&**x).dot(&uo);
    }

    // Return orbital energies.
    let mut eps = Array1::<Complex64>::zeros(n);
    eps.slice_mut(s![0..nocc]).assign(&eo);
    eps.slice_mut(s![nocc..n]).assign(&ev);
    eps
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
fn weight_by_gap(x: &Array2<Complex64>, eps: &Array1<Complex64>, nocc: usize, tol: f64, multiply: bool) -> Array2<Complex64> {
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

/// Solve the complex-symmetric SR1 quasi-Newton equation in energy-weighted coordinates.
/// # Arguments:
/// - `hist`: Stored unweighted secant pairs.
/// - `ga`: Alpha-spin occupied-virtual gradient.
/// - `gb`: Beta-spin occupied-virtual gradient.
/// - `epsa`: Alpha-spin pseudo-canonical orbital energies.
/// - `epsb`: Beta-spin pseudo-canonical orbital energies.
/// - `na`: Number of occupied alpha-spin orbitals.
/// - `nb`: Number of occupied beta-spin orbitals.
/// - `opts`: h-SCF quasi-Newton options.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Unweighted alpha- and beta-spin orbital steps.
fn sr1_step(hist: &[SecantPair], ga: &Array2<Complex64>, gb: &Array2<Complex64>, epsa: &Array1<Complex64>, epsb: &Array1<Complex64>, 
            na: usize, nb: usize, opts: &HSCFOptions) -> (Array2<Complex64>, Array2<Complex64>) {

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

        // Calculate residiual error in current prediction.
        let r = &y - &b.dot(&s);

        // Update Hessian approximation as 
        // B_{k + 1} = B_k + (r_k r_k^T) / r_k^T s_k
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
    (weight_by_gap(&pa_bar, epsa, na, opts.denom_tol, false), weight_by_gap(&pb_bar, epsb, nb, opts.denom_tol, false))
}

/// Limit combined alpha/beta occupied-virtual step norm.
/// # Arguments:
/// - `pa`: Alpha-spin occupied-virtual step.
/// - `pb`: Beta-spin occupied-virtual step.
/// - `max_step`: Maximum allowed combined step norm.
/// # Returns:
/// - `()`: Updates `pa` and `pb` in place.
fn limit_step(pa: &mut Array2<Complex64>, pb: &mut Array2<Complex64>, max_step: f64) {
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
fn step_norm(pa: &Array2<Complex64>, pb: &Array2<Complex64>) -> f64 {
    (pa.iter().map(|z| z.norm_sqr()).sum::<f64>() + pb.iter().map(|z| z.norm_sqr()).sum::<f64>()).sqrt()
}

/// Backtrack along the complex-orthogonal geodesic and minimise gradient norm.
/// # Arguments:
/// - `ca`: Current alpha-spin MO coefficients.
/// - `cb`: Current beta-spin MO coefficients.
/// - `ao`: Contains AO integrals and metadata.
/// - `na`: Number of occupied alpha-spin orbitals.
/// - `nb`: Number of occupied beta-spin orbitals.
/// - `pa`: Alpha-spin occupied-virtual step.
/// - `pb`: Beta-spin occupied-virtual step.
/// - `g0`: Current occupied-virtual gradient norm.
/// - `opts`: h-SCF quasi-Newton options.
/// # Returns:
/// - `(f64, Array2<Complex64>, Array2<Complex64>)`: Step length and updated alpha/beta orbitals.
fn line_search(ca: &Array2<Complex64>, cb: &Array2<Complex64>, ao: &AoData, na: usize, nb: usize, pa: &Array2<Complex64>, 
               pb: &Array2<Complex64>, g0: f64, opts: &HSCFOptions) -> (f64, Array2<Complex64>, Array2<Complex64>) {

    // Start with the full step length.
    let mut alpha = 1.0;
    let mut best = (0.0, ca.clone(), cb.clone(), g0);
    
    // Try a sequence of increasingly smaller step lengths.
    for _ in 0..opts.line_steps {
        // Try both directions along the geodesic.
        for sign in [1.0, -1.0] {
            // Apply the orbital rotation update.
            let cat = geodesic_step(ca, pa, na, alpha * sign);
            let cbt = geodesic_step(cb, pb, nb, alpha * sign);

            let da = density(&cat, na, DensityMode::Holomorphic);
            let db = density(&cbt, nb, DensityMode::Holomorphic);

            let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);

            // Calculate orbital gradient g_{ai} = 2 \sum_{\mu\nu} C_a^\mu F_{\mu\nu} C_i^\nu.
            let ga = orbital_gradient(&cat, &fa, na, DensityMode::Holomorphic);
            let gb = orbital_gradient(&cbt, &fb, nb, DensityMode::Holomorphic);

            // Compare candidate steps using the real diagnostic norm of the h-SCF gradient.
            let g = (ga.iter().map(|z| z.norm_sqr()).sum::<f64>() + gb.iter().map(|z| z.norm_sqr()).sum::<f64>()).sqrt();

            if g < best.3 {best = (alpha * sign, cat.clone(), cbt.clone(), g);}
            if g < g0 {return (alpha * sign, cat, cbt);}
        }
        alpha *= opts.line_shrink;
    }
    (best.0, best.1, best.2)
}

/// Apply a complex-orthogonal occupied-virtual geodesic step.
/// # Arguments:
/// - `c`: Current MO coefficient matrix ordered as occupied then virtual.
/// - `p`: Occupied-virtual step block with shape `(nvir, nocc)`.
/// - `nocc`: Number of occupied orbitals.
/// - `alpha`: Step length.
/// # Returns:
/// - `Array2<Complex64>`: Updated MO coefficient matrix.
fn geodesic_step(c: &Array2<Complex64>, p: &Array2<Complex64>, nocc: usize, alpha: f64) -> Array2<Complex64> {
    let n = c.ncols(); 
    let nvir = n - nocc;

    let mut k = Array2::<Complex64>::zeros((n, n));

    // Construct the \exp(0 -\alpha_k p_k^T \\ \alpha_k p_k) matrix exponential.
    for a in 0..nvir {
        for i in 0..nocc {
            let z = p[(a, i)] * alpha;
            k[(nocc + a, i)] = z;
            k[(i, nocc + a)] = -z;
        }
    }
    c.dot(&matrix_exp_complex(&k))
}

/// Apply a imaginary occupied-virtual petrubation to initialise a complex h-SCF branch.
/// # Arguments:
/// - `c`: MO coefficient matrix ordered as occupied then virtual.
/// - `nocc`: Number of occupied orbitals.
/// - `theta`: Complex rotation amplitude applied to corresponding occupied-virtual pairs.
/// # Returns:
/// - `Array2<Complex64>`: Kicked MO coefficient matrix.
fn perturb_ov(c: &Array2<Complex64>, nocc: usize, theta: Complex64) -> Array2<Complex64> {
    let n = c.ncols(); 
    let nrot = nocc.min(n - nocc);

    if nrot == 0 || theta.norm() == 0.0 {
        return c.clone();
    }

    let mut p = Array2::<Complex64>::zeros((n - nocc, nocc));
    for i in 0..nrot {
        p[(i, i)] = theta;
    }

    geodesic_step(c, &p, nocc, 1.0)
}

/// Construct final h-SCF state from optimised complex orbitals.
/// # Arguments:
/// - `ca`: Final alpha-spin MO coefficients.
/// - `cb`: Final beta-spin MO coefficients.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: User input specifications.
/// - `label`: Label for the h-SCF state.
/// - `noci_basis`: Whether this state is intended for a later NOCI basis.
/// - `i`: State index.
/// - `converged`: Whether the h-SCF gradient threshold was reached.
/// # Returns:
/// - `HSCFState`: Final h-SCF determinant state.
fn finalise(ca: Array2<Complex64>, cb: Array2<Complex64>, ao: &AoData, input: &Input, label: &str, noci_basis: bool, i: usize, converged: bool) -> HSCFState {
    let na = usize::try_from(ao.nelec[0]).unwrap(); let nb = usize::try_from(ao.nelec[1]).unwrap();

    let da = density(&ca, na, DensityMode::Holomorphic);
    let db = density(&cb, nb, DensityMode::Holomorphic);

    let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
    let e = energy(&ao.h, ao.enuc, &da, &db, &fa, &fb);
    
    // Calculate g_{ai} = 2 \sum_{\mu\nu} C_a^\mu F_{\mu\nu} C_i^\nu.
    let ga = orbital_gradient(&ca, &fa, na, DensityMode::Holomorphic);
    let gb = orbital_gradient(&cb, &fb, nb, DensityMode::Holomorphic);

    let gnorm = (ga.iter().map(|z| z.norm_sqr()).sum::<f64>() + gb.iter().map(|z| z.norm_sqr()).sum::<f64>()).sqrt();

    if input.write.verbose {
        println!("h-SCF {}: E = {} {:+}i, ||g_ov|| = {:.4e}", if converged {"converged"} else {"not converged"}, e.re, e.im, gnorm);
        println!("{}", "-".repeat(100));
        println!("Complex coefficients ca:");
        print_array2_indexed(&ca);
        println!("Complex coefficients cb:");
        print_array2_indexed(&cb);
    }
    if input.write.write_orbitals {println!("Complex h-SCF orbital HDF5 writing is not implemented yet.");}

    let excitation = Excitation {alpha: ExcitationSpin {holes: vec![], parts: vec![]}, beta: ExcitationSpin {holes: vec![], parts: vec![]}};

    // Occupy the first `na` and `nb` orbitals because h-SCF keeps occupied orbitals first throughout.
    let oa = (0..na).fold(0u128, |bits, j| bits | (1u128 << j));
    let ob = (0..nb).fold(0u128, |bits, j| bits | (1u128 << j));

    HSCFState {
        e, 
        oa, 
        ob, 
        pha: 1.0,
        phb: 1.0, 
        ca: Arc::new(ca), 
        cb: Arc::new(cb), 
        da: Arc::new(da), 
        db: Arc::new(db), 
        label: label.to_string(), 
        noci_basis, parent: i, 
        excitation
    }
}

/// Pack alpha and beta tangent blocks into one vector.
/// # Arguments:
/// - `a`: Alpha-spin tangent block.
/// - `b`: Beta-spin tangent block.
/// # Returns:
/// - `Array1<Complex64>`: Concatenated vector.
fn pack(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array1<Complex64> {
    Array1::from_iter(a.iter().chain(b.iter()).copied())
}

/// Unpack one vector into alpha and beta tangent blocks.
/// # Arguments:
/// - `x`: Packed tangent vector.
/// - `adim`: Alpha-spin block dimensions.
/// - `bdim`: Beta-spin block dimensions.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Alpha- and beta-spin tangent blocks.
fn unpack(x: &Array1<Complex64>, adim: (usize, usize), bdim: (usize, usize)) -> (Array2<Complex64>, Array2<Complex64>) {
    let na = adim.0 * adim.1;
    let mut a = Array2::<Complex64>::zeros(adim);
    let mut b = Array2::<Complex64>::zeros(bdim);
    for (dst, src) in a.iter_mut().zip(x.slice(s![0..na]).iter()) {*dst = *src;}
    for (dst, src) in b.iter_mut().zip(x.slice(s![na..]).iter()) {*dst = *src;}
    (a, b)
}

/// Build initial h-SCF orbitals from a real SCF seed and state recipe.
/// # Arguments:
/// - `seed`: Real SCF seed state.
/// - `recipe`: State construction recipe.
/// - `ao`: Contains electron counts and AO metadata.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Alpha and beta h-SCF initial orbitals.
pub fn h_seed_orbitals(seed: &SCFState, recipe: &StateRecipe, ao: &AoData) -> (Array2<Complex64>, Array2<Complex64>) {
    let na = usize::try_from(ao.nelec[0]).unwrap(); let nb = usize::try_from(ao.nelec[1]).unwrap();

    // Reorder alpha orbitals so occupied columns are first, followed by virtual columns.
    let idx_a: Vec<usize> = (0..seed.ca.ncols())
        .filter(|&p| ((seed.oa >> p) & 1u128) == 1)
        .chain((0..seed.ca.ncols()).filter(|&p| ((seed.oa >> p) & 1u128) == 0))
        .collect();
    let mut ca = real2_as::<Complex64>(&seed.ca).select(Axis(1), &idx_a);

    // Reorder beta orbitals so occupied columns are first, followed by virtual columns.
    let idx_b: Vec<usize> = (0..seed.cb.ncols())
        .filter(|&p| ((seed.ob >> p) & 1u128) == 1)
        .chain((0..seed.cb.ncols()).filter(|&p| ((seed.ob >> p) & 1u128) == 0))
        .collect();
    let mut cb = real2_as::<Complex64>(&seed.cb).select(Axis(1), &idx_b);

    if let Some(sb) = &recipe.spin_bias {
        let sgn = sb.pattern.iter().copied().find(|&x| x != 0).unwrap_or(1) as f64;
        let theta = Complex64::new(0.0, sgn * sb.pol.abs().max(0.05));
        ca = perturb_ov(&ca, na, theta);
        cb = perturb_ov(&cb, nb, -theta);
    }

    if let Some(spb) = &recipe.spatial_bias {
        let sgn = spb.pattern.iter().copied().find(|&x| x != 0).unwrap_or(1) as f64;
        let theta = Complex64::new(0.0, sgn * spb.pol.abs().max(0.05));
        ca = perturb_ov(&ca, na, theta);
        cb = perturb_ov(&cb, nb, theta);
    }

    (ca, cb)
}
