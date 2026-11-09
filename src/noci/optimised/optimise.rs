// noci/optimised/optimise.rs
//! Self-consistent ResHF loop: alternate the CI step with a Roothaan step on every
//! determinant's resonating Fock matrices, accelerated by one DIIS history shared by all
//! determinants. The determinants are coupled through the CI expansion, so a common set of
//! extrapolation coefficients keeps their updates consistent.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::elements::occ_coeffs;
use crate::input::{Input, ResHFOptions};
use crate::scf::{
    DensityMode, Diis, occ_first, orbital_gradient, print_header, print_iteration, roothaan_step,
};
use crate::{AoData, SCFState};

// Parent/sibling imports.
use super::ci::reshf_solve;
use super::fock::reshf_fock;
use super::pair::reshf_pairs;

/// Overlap eigenvalue below which the CI step projects a direction out, the tolerance the
/// geometry driver uses for reference NOCI.
const NOCITOL: f64 = 1e-8;

/// Pair weight `D_{AA}` below which determinant `A` keeps its orbitals: it then barely enters
/// the averaged states, so `F^{A,\omega}` is close to zero and diagonalising it would scramble
/// the orbitals.
const WEIGHTTOL: f64 = 1e-10;

/// Optimise the orbitals of every NOCI state by ResHF-cutoff, making
/// `E_{\text{SA}} = \sum_I w_I E_I` stationary. Each iteration builds the floored pair
/// quantities, solves `H c = S c E`, builds the resonating Fock matrices and takes one
/// Roothaan step per determinant, all accelerated by one DIIS history whose channels are the
/// two spins of every determinant. Weights and cutoff come from `options`; `e_tol`,
/// `fds_sdf_tol`, `max_cycle` and the DIIS space from `input.scf`. Convergence needs `|E - E_{\text{old}}| <` `e_tol` and
/// `\sqrt{\sum_A \|X^\dagger (F^A P^A S - S P^A F^A) X\|^2} <` `fds_sdf_tol`.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `options`: ResHF state weights and singular value floor.
/// - `states`: SCF states; those with `noci_basis` are optimised, the others are kept as given.
/// # Returns
/// - `Vec<SCFState>`: Every input state, with the NOCI states replaced in place by their
///   converged (or last) ResHF determinants and labels kept.
pub fn reshf_optimise(
    ao: &AoData,
    input: &Input,
    options: &ResHFOptions,
    states: &[SCFState],
) -> Vec<SCFState> {
    let verbose = input.write.verbose >= 1;

    // Optimise the NOCI states in their original order, occupied orbitals first.
    let indices: Vec<usize> = (0..states.len())
        .filter(|&i| states[i].noci_basis)
        .collect();
    let mut dets: Vec<SCFState> = indices.iter().map(|&i| occ_first(&states[i])).collect();

    // One DIIS history for all determinants, with channels `F^{0,\alpha}, F^{0,\beta}, F^{1,\alpha}, \ldots`.
    let mut diis = Diis::new(input.scf.diis.space);

    if verbose {
        println!("{}", "=".repeat(100));
        println!("ResHF optimisation of {} determinants", dets.len());
    }
    print_header(input, None);

    let mut e_old = f64::INFINITY;
    let mut iter = 0;
    let (e, converged) = loop {
        // CI step on the floored pair quantities, then the resonating Fock matrices.
        let pairs = reshf_pairs(ao, &dets, options.cutoff);
        let h = pairs.map(|p| p.h);
        let s = pairs.map(|p| p.s);
        let expansion = reshf_solve(&h, &s, &options.weights, NOCITOL);
        let e: f64 = options
            .weights
            .iter()
            .zip(expansion.energies.iter())
            .map(|(w, e)| w * e)
            .sum();
        let fock = reshf_fock(&ao.s, &pairs, &expansion);

        // Residual of the ResHF condition, `X^\dagger (F^A P^A S - S P^A F^A) X` for every
        // determinant and spin, as channels of the shared DIIS history.
        let f: Vec<&Array2<f64>> = fock.iter().flatten().collect();
        let d: Vec<&Array2<f64>> = dets.iter().flat_map(|d| [&*d.da, &*d.db]).collect();
        diis.push(&f, &d, &ao.s, &ao.x);
        let err = diis.last_error_norm2().unwrap_or(f64::INFINITY).sqrt();
        let d_e = (e - e_old).abs();

        // Occupied-virtual blocks of the resonating Fock matrices, the orbital gradient.
        let gnorm = dets
            .iter()
            .zip(&fock)
            .flat_map(|(det, f)| {
                let ga = orbital_gradient(
                    &det.ca,
                    &f[0],
                    det.oa.count_ones() as usize,
                    DensityMode::Hermitian,
                );
                let gb = orbital_gradient(
                    &det.cb,
                    &f[1],
                    det.ob.count_ones() as usize,
                    DensityMode::Hermitian,
                );
                [ga, gb]
            })
            .map(|g| g.iter().map(|z| z * z).sum::<f64>())
            .sum::<f64>()
            .sqrt();

        if verbose {
            print_iteration(iter, e, d_e, err, gnorm);
        }

        // Stop once converged or out of iterations, keeping the determinants `E` belongs to.
        iter += 1;
        if d_e < input.scf.e_tol && err < input.scf.fds_sdf_tol {
            break (e, true);
        }
        if iter >= input.scf.max_cycle {
            break (e, false);
        }

        // Roothaan step on each weighted determinant's extrapolated Fock matrices.
        let f = diis
            .extrapolate_fock()
            .unwrap_or_else(|| fock.into_iter().flatten().collect());
        for (a, det) in dets.iter_mut().enumerate() {
            if expansion.pair_weights[(a, a)] < WEIGHTTOL {
                continue;
            }

            // Follow each spin's occupied space by MOM and rebuild the determinant's own
            // densities, Fock matrices and SCF energy, keeping occupied orbitals first.
            let nocc = (det.oa.count_ones() as usize, det.ob.count_ones() as usize);
            let c_occ = (occ_coeffs(&det.ca, det.oa), occ_coeffs(&det.cb, det.ob));
            let step = roothaan_step(
                ao,
                (&f[2 * a], &f[2 * a + 1]),
                nocc,
                (true, true),
                (Some(&c_occ.0), Some(&c_occ.1)),
                None,
            );
            *det = occ_first(&step.into_state(&det.label, true));
        }
        e_old = e;
    };

    if verbose {
        println!("ResHF energy: {e}");
    }
    if !converged {
        println!(
            "Warning: ResHF not converged after {iter} iterations; using the last determinants."
        );
    }

    // Put the optimised determinants back in place of the NOCI states.
    let mut out = states.to_vec();
    for (i, det) in indices.into_iter().zip(dets) {
        out[i] = det;
    }
    out
}
