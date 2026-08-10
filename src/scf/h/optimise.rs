// scf/h/optimise.rs

// External crate imports.
use ndarray::Array2;
use num_complex::Complex64;

// Crate-root imports.
use crate::input::Input;
use crate::scf::DensityMode;
use crate::scf::print::print_header_h;
use crate::scf::{density, energy, fock_lambda, orbital_gradient};
use crate::{AoData, HSCFState};

// Parent/sibling imports.
use super::canonical::pseudo_canonicalise;
use super::finalise::finalise;
use super::step::{finite_difference_newton_step, limit_step, line_search, sr1_step, step_norm};
use super::types::{HSCFRunData, SecantPair, SpinBlock};

const G_TOL: f64 = 1e-6;
const HISTORY: usize = 20;

/// Run a holomorphic unrestricted SCF quasi-Newton optimisation.
/// # Arguments:
/// - `ca0`: Initial alpha-spin MO coefficients ordered as occupied then virtual.
/// - `cb0`: Initial beta-spin MO coefficients ordered as occupied then virtual.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: User input specifications.
/// - `run`: Immutable data for this h-SCF optimisation, including the state label,
///   NOCI status, parent index, and electron-electron scaling.
/// # Returns:
/// - `Option<HSCFState>`: Converged h-SCF state if optimisation succeeds.
pub(crate) fn hscf_cycle(
    ca0: &Array2<Complex64>,
    cb0: &Array2<Complex64>,
    ao: &AoData,
    input: &Input,
    run: HSCFRunData<'_>,
) -> Option<HSCFState> {
    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();

    // Use local copies of orbital coefficients.
    let mut ca = ca0.clone();
    let mut cb = cb0.clone();

    // History of the SR1 orbital displacements and gradient differences
    // to improve approximation of Hessian quantities.
    let mut hist: Vec<SecantPair> = Vec::new();

    print_header_h(input, run.label);

    // Retain previous gradient and orbital displacement to construct
    // secant pairs which are stored in `hist`.
    let mut g_prev: Option<(Array2<Complex64>, Array2<Complex64>)> = None;
    let mut step_prev: Option<(Array2<Complex64>, Array2<Complex64>)> = None;

    // Quantities used to monitor whether the orbital gradient descent
    // has stagnated, at which point a finite-difference step is used to
    // escape.
    let mut best_gnorm = f64::INFINITY;
    let mut stagnant = 0usize;

    for iter in 0..input.scf.max_cycle {
        let da = density(&ca, na, DensityMode::Holomorphic);
        let db = density(&cb, nb, DensityMode::Holomorphic);

        let (fa, fb) = fock_lambda(&ao.h, &ao.eri_coul, &da, &db, run.lambda);

        // Pseudo canonicalise alpha occupied and virtual subspaces seperately.
        // The stored secant tangent-space quantities must undergo the same transform.
        let mut extra_a: Vec<&mut Array2<Complex64>> = Vec::new();
        if let Some((sa, _)) = step_prev.as_mut() {
            extra_a.push(sa);
        }
        if let Some((ga, _)) = g_prev.as_mut() {
            extra_a.push(ga);
        }
        let epsa = pseudo_canonicalise(&mut ca, &fa, na, &mut hist, SpinBlock::Alpha, &mut extra_a);

        // Pseudo canonicalise beta occupied and virtual subspaces seperately.
        // The stored secant tangent-space quantities must undergo the same transform.
        let mut extra_b: Vec<&mut Array2<Complex64>> = Vec::new();
        if let Some((_, sb)) = step_prev.as_mut() {
            extra_b.push(sb);
        }
        if let Some((_, gb)) = g_prev.as_mut() {
            extra_b.push(gb);
        }
        let epsb = pseudo_canonicalise(&mut cb, &fb, nb, &mut hist, SpinBlock::Beta, &mut extra_b);

        let e = energy(&ao.h, ao.enuc, &da, &db, &fa, &fb);

        // Calculate g_{ai} = 2 \sum_{\mu\nu} C_a^\mu F_{\mu\nu} C_i^\nu.
        let (ga, gb) = rayon::join(
            || orbital_gradient(&ca, &fa, na, DensityMode::Holomorphic),
            || orbital_gradient(&cb, &fb, nb, DensityMode::Holomorphic),
        );

        // Use the Euclidean Frobenius norm only as a real convergence diagnostic.
        let gnorm = (ga.iter().map(|z| z.norm_sqr()).sum::<f64>()
            + gb.iter().map(|z| z.norm_sqr()).sum::<f64>())
        .sqrt();

        if gnorm < G_TOL {
            if input.write.verbose >= 1 {
                println!(
                    "{:4} {:16.10} {:+16.10}i {:12.4e} {:>12} {:>12}",
                    iter, e.re, e.im, gnorm, "-", "-"
                );
            }
            return Some(finalise(ca, cb, ao, input, run));
        }

        // Store the new secant pair:
        // `s_k = \alpha_k p_k`,
        // `y_k` = g_{k + 1} - g_k.
        if let (Some((sa, sb)), Some((gpa, gpb))) = (step_prev.take(), g_prev.take()) {
            let ya = &ga - &gpa;
            let yb = &gb - &gpb;

            hist.push(SecantPair { sa, sb, ya, yb });
            // Only keep the desired number of historical steps.
            if hist.len() > HISTORY {
                hist.remove(0);
            }
        }

        // Iteration is not going in a useful direction if norm falls by
        // less than some amount, currently set very arbitrarily.
        if gnorm < best_gnorm * 0.95 {
            best_gnorm = gnorm;
            stagnant = 0;
        } else {
            stagnant += 1;
        }

        let use_fd_newton = stagnant >= 8;

        // After an unacceptable amount of useless steps deploy finite-difference.
        if use_fd_newton && !hist.is_empty() {
            hist.clear();
            stagnant = 0;
            if input.write.verbose >= 1 {
                println!("h-SCF progress stalled; using finite-difference Newton rescue step.");
            }
        }
        let (mut pa, mut pb) = if use_fd_newton {
            finite_difference_newton_step((&ca, &cb), ao, (na, nb), (&ga, &gb), run.lambda)
                .unwrap_or_else(|| sr1_step(&hist, (&ga, &gb), (&epsa, &epsb), (na, nb)))
        } else {
            sr1_step(&hist, (&ga, &gb), (&epsa, &epsb), (na, nb))
        };

        // Limit the total occupied-virtual rotation before the line search so
        // that a poor approximation to the Hessian cannot produce a huge orbital step.
        limit_step(&mut pa, &mut pb);
        let pnorm = step_norm(&pa, &pb);

        // Backtrack along proposed direction and accept a step only when reducing the
        // orbital gradient.
        let (alpha, ca_new, cb_new) =
            line_search((&ca, &cb), ao, (na, nb), (&pa, &pb), gnorm, run.lambda);

        if input.write.verbose >= 1 {
            println!(
                "{:4} {:16.10} {:+16.10}i {:12.4e} {:12.4e} {:12.4e}",
                iter, e.re, e.im, gnorm, alpha, pnorm
            );
        }

        if alpha == 0.0 {
            if input.write.verbose >= 1 {
                println!("h-SCF line search stalled; trying finite-difference Newton rescue.");
            }

            hist.clear();
            stagnant = 0;

            if let Some((mut pa_fd, mut pb_fd)) =
                finite_difference_newton_step((&ca, &cb), ao, (na, nb), (&ga, &gb), run.lambda)
            {
                limit_step(&mut pa_fd, &mut pb_fd);

                let (alpha_fd, ca_fd, cb_fd) = line_search(
                    (&ca, &cb),
                    ao,
                    (na, nb),
                    (&pa_fd, &pb_fd),
                    gnorm,
                    run.lambda,
                );

                if alpha_fd != 0.0 {
                    if input.write.verbose >= 1 {
                        println!(
                            "h-SCF finite-difference Newton rescue accepted: alpha = {:.4e}",
                            alpha_fd
                        );
                    }

                    step_prev = Some((pa_fd.mapv(|z| z * alpha_fd), pb_fd.mapv(|z| z * alpha_fd)));

                    g_prev = Some((ga, gb));

                    ca = ca_fd;
                    cb = cb_fd;

                    continue;
                }
            }

            if input.write.verbose >= 1 {
                println!("h-SCF finite-difference Newton rescue also failed.");
            }

            finalise(ca_new, cb_new, ao, input, run);
            return None;
        }

        let pa_acc = pa.mapv(|z| z * alpha);
        let pb_acc = pb.mapv(|z| z * alpha);

        // Store the accepted unweighted displacement and gradient for the next SR1 secant pair.
        step_prev = Some((pa_acc, pb_acc));
        g_prev = Some((ga, gb));
        ca = ca_new;
        cb = cb_new;
    }

    finalise(ca, cb, ao, input, run);
    None
}
