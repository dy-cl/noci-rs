// scf/h/optimise.rs

use ndarray::Array2;
use num_complex::Complex64;

use crate::input::Input;
use crate::scf::DensityMode;
use crate::{AoData, HSCFState};

use super::canonical::pseudo_canonicalise;
use super::finalise::finalise;
use super::step::{finite_difference_newton_step, limit_step, line_search, sr1_step, step_norm};
use super::types::{HSCFRunData, SecantPair, SpinBlock};
use crate::scf::print::print_header_h;
use crate::scf::{density, energy, fock, orbital_energies, orbital_gradient};

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
pub fn hscf_cycle(
    ca0: &Array2<Complex64>,
    cb0: &Array2<Complex64>,
    ao: &AoData,
    input: &Input,
    label: &str,
    noci_basis: bool,
    i: usize,
) -> Option<HSCFState> {
    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();
    let opts = &input.scf.h;
    let mut ca = ca0.clone();
    let mut cb = cb0.clone();
    let mut hist: Vec<SecantPair> = Vec::new();
    let run = HSCFRunData {
        label,
        noci_basis,
        parent: i,
    };

    print_header_h(input, label);

    let mut g_prev: Option<(Array2<Complex64>, Array2<Complex64>)> = None;
    let mut step_prev: Option<(Array2<Complex64>, Array2<Complex64>)> = None;
    let mut best_gnorm = f64::INFINITY;
    let mut stagnant = 0usize;

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
        let gnorm = (ga.iter().map(|z| z.norm_sqr()).sum::<f64>()
            + gb.iter().map(|z| z.norm_sqr()).sum::<f64>())
        .sqrt();

        if gnorm < opts.g_tol {
            if input.write.verbose {
                println!(
                    "{:4} {:16.10} {:+16.10}i {:12.4e} {:>12} {:>12}",
                    iter, e.re, e.im, gnorm, "-", "-"
                );
            }
            return Some(finalise(ca, cb, ao, input, run));
        }

        if let (Some((sa, sb)), Some((gpa, gpb))) = (step_prev.take(), g_prev.take()) {
            let ya = &ga - &gpa;
            let yb = &gb - &gpb;

            hist.push(SecantPair { sa, sb, ya, yb });
            if hist.len() > opts.history {
                hist.remove(0);
            }
        }

        if gnorm < best_gnorm * 0.95 {
            best_gnorm = gnorm;
            stagnant = 0;
        } else {
            stagnant += 1;
        }

        let use_fd_newton = stagnant >= 8;
        if use_fd_newton && !hist.is_empty() {
            hist.clear();
            stagnant = 0;
            if input.write.verbose {
                println!("h-SCF progress stalled; using finite-difference Newton rescue step.");
            }
        }

        let (mut pa, mut pb) = if use_fd_newton {
            finite_difference_newton_step(&ca, &cb, ao, na, nb, &ga, &gb)
                .unwrap_or_else(|| sr1_step(&hist, (&ga, &gb), (&epsa, &epsb), (na, nb), opts))
        } else {
            sr1_step(&hist, (&ga, &gb), (&epsa, &epsb), (na, nb), opts)
        };

        limit_step(&mut pa, &mut pb, opts.max_step);
        let pnorm = step_norm(&pa, &pb);

        let (alpha, ca_new, cb_new) =
            line_search((&ca, &cb), ao, (na, nb), (&pa, &pb), gnorm, opts);

        if input.write.verbose {
            println!(
                "{:4} {:16.10} {:+16.10}i {:12.4e} {:12.4e} {:12.4e}",
                iter, e.re, e.im, gnorm, alpha, pnorm
            );
        }

        if alpha == 0.0 {
            if input.write.verbose {
                println!("h-SCF line search found no improving step.");
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
