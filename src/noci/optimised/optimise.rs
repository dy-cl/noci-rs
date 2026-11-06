// noci/optimised/optimise.rs
//! Self-consistent ResHF loop: alternate the CI step with a Roothaan step on every
//! determinant's resonating Fock matrices, accelerated by one DIIS history shared by all
//! determinants, as in yucca. The determinants are coupled through the CI expansion, so a
//! common set of extrapolation coefficients keeps their updates consistent.

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use ndarray::{Array2, s};

// Crate-root imports.
use crate::elements::occ_coeffs;
use crate::input::Input;
use crate::maths::general_evp_x;
use crate::scf::{DensityMode, Diis, density, energy, fock, mom_select, occ_first, occvec_to_bits};
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

/// Converged (or last) ResHF determinants and their energy.
pub struct ResHFSolution {
    /// Every input state, with the optimised NOCI states replaced in place and labels kept.
    pub states: Vec<SCFState>,
    /// `E^{\text{ResHF}} = \sum_I w_I E_I`, including nuclear repulsion.
    pub energy: f64,
    /// Number of CI steps performed.
    pub iterations: usize,
    /// Whether the energy and commutator criteria of `input.scf` were met.
    pub converged: bool,
}

/// Block-diagonal matrix `\mathrm{diag}(M_0, M_1, \ldots)`, which lets one DIIS history extrapolate
/// the matrices of all determinants with common coefficients.
/// # Arguments:
/// - `blocks`: Diagonal blocks, in determinant order.
/// # Returns
/// - `Array2<f64>`: The block-diagonal matrix.
fn block_diagonal(blocks: &[&Array2<f64>]) -> Array2<f64> {
    let rows: usize = blocks.iter().map(|m| m.nrows()).sum();
    let cols: usize = blocks.iter().map(|m| m.ncols()).sum();
    let mut out = Array2::<f64>::zeros((rows, cols));

    // Place each block below and to the right of the previous one.
    let (mut r, mut c) = (0, 0);
    for m in blocks {
        out.slice_mut(s![r..r + m.nrows(), c..c + m.ncols()])
            .assign(m);
        r += m.nrows();
        c += m.ncols();
    }
    out
}

/// Diagonal block `a` of a block-diagonal matrix of `n \times n` blocks.
/// # Arguments:
/// - `m`: Block-diagonal matrix.
/// - `a`: Block index.
/// - `n`: Dimension of each block.
/// # Returns
/// - `Array2<f64>`: Block `a`.
fn diagonal_block(
    m: &Array2<f64>,
    a: usize,
    n: usize,
) -> Array2<f64> {
    m.slice(s![a * n..(a + 1) * n, a * n..(a + 1) * n])
        .to_owned()
}

/// Roothaan step for one spin: solve `F C = S C \epsilon` and occupy the `nocc` orbitals that
/// overlap most with the previous occupied space (MOM).
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `f`: Resonating Fock matrix `F^{A,\omega}`.
/// - `c_occ_old`: Previous occupied coefficients `C^A_o`.
/// - `nocc`: Number of occupied orbitals.
/// # Returns
/// - `(Array2<f64>, u128)`: New MO coefficients and their occupation bitstring.
fn occupy(
    ao: &AoData,
    f: &Array2<f64>,
    c_occ_old: &Array2<f64>,
    nocc: usize,
) -> (Array2<f64>, u128) {
    let (_, c) = general_evp_x(f, &ao.x);
    let idx = mom_select(c_occ_old, &c, &ao.s, nocc);
    (c, occvec_to_bits(&idx))
}

/// New orbitals of one determinant from its resonating Fock matrices, with occupied orbitals
/// first, `P^\sigma`, `F^\sigma` and its own energy `E` rebuilt as `scf_cycle` does.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `det`: Current determinant.
/// - `fa`: Resonating Fock matrix `F^{A,\alpha}`, possibly DIIS-extrapolated.
/// - `fb`: Resonating Fock matrix `F^{A,\beta}`, possibly DIIS-extrapolated.
/// # Returns
/// - `SCFState`: The updated determinant, with the same label.
fn reshf_step(
    ao: &AoData,
    det: &SCFState,
    fa: &Array2<f64>,
    fb: &Array2<f64>,
) -> SCFState {
    let na = det.oa.count_ones() as usize;
    let nb = det.ob.count_ones() as usize;

    // Diagonalise each spin's resonating Fock matrix and follow the occupied space by MOM.
    let (ca, oa) = occupy(ao, fa, &occ_coeffs(&det.ca, det.oa), na);
    let (cb, ob) = occupy(ao, fb, &occ_coeffs(&det.cb, det.ob), nb);
    let st = occ_first(&SCFState {
        e: det.e,
        oa,
        ob,
        ca: Arc::new(ca),
        cb: Arc::new(cb),
        da: det.da.clone(),
        db: det.db.clone(),
        label: det.label.clone(),
        noci_basis: true,
    });

    // Rebuild `P^\sigma`, `F^\sigma` and the determinant's own SCF energy.
    let da = density(&st.ca, na, DensityMode::Hermitian);
    let db = density(&st.cb, nb, DensityMode::Hermitian);
    let (fa_own, fb_own) = fock(&ao.h, &ao.eri_coul, &da, &db);
    let e = energy(&ao.h, ao.enuc, &da, &db, &fa_own, &fb_own);

    SCFState {
        e,
        da: Arc::new(da),
        db: Arc::new(db),
        ..st
    }
}

/// Optimise the orbitals of every NOCI state by ResHF-cutoff, making
/// `E_{\text{SA}} = \sum_I w_I E_I` stationary. Each iteration builds the floored pair
/// quantities, solves `H c = S c E`, builds the resonating Fock matrices and takes one
/// Roothaan step per determinant, all accelerated by one DIIS history of the block-diagonal
/// Fock and error matrices. Weights and cutoff come from `input.reshf`
/// (defaults if absent); `e_tol`, `fds_sdf_tol`, `max_cycle` and the DIIS space from
/// `input.scf`. Convergence needs `|E - E_{\text{old}}| <` `e_tol` and
/// `\sqrt{\sum_A \|X^\dagger (F^A P^A S - S P^A F^A) X\|^2} <` `fds_sdf_tol`.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `states`: SCF states; those with `noci_basis` are optimised, the others are kept as given.
/// # Returns
/// - `ResHFSolution`: The states with optimised NOCI determinants and the ResHF energy.
pub fn reshf_optimise(
    ao: &AoData,
    input: &Input,
    states: &[SCFState],
) -> ResHFSolution {
    let options = input.reshf.clone().unwrap_or_default();
    let max_cycle = usize::try_from(input.scf.max_cycle).unwrap_or(0);
    let verbose = input.write.verbose >= 1;

    // Optimise the NOCI states in their original order, occupied orbitals first.
    let indices: Vec<usize> = (0..states.len())
        .filter(|&i| states[i].noci_basis)
        .collect();
    let mut dets: Vec<SCFState> = indices.iter().map(|&i| occ_first(&states[i])).collect();

    // One DIIS history for all determinants, on block-diagonal matrices whose error
    // `X^\dagger (F P S - S P F) X` is block diagonal too.
    let nao = ao.s.nrows();
    let mut diis = Diis::new(input.scf.diis.space);
    let s_all = block_diagonal(&vec![&ao.s; dets.len()]);
    let x_all = block_diagonal(&vec![&ao.x; dets.len()]);

    if verbose {
        println!("{}", "=".repeat(100));
        println!("ResHF optimisation of {} determinants", dets.len());
        println!("{:>4} {:>12} {:>12} {:>12}", "i", "E", "dE", "‖FPS - SPF‖");
    }

    let mut e_old = f64::INFINITY;
    let mut iterations = 0;
    let (e, converged) = loop {
        // CI step on the floored pair quantities, then the resonating Fock matrices.
        let pairs = reshf_pairs(ao, &dets, options.cutoff);
        let expansion = reshf_solve(pairs.h(), pairs.s(), &options.weights, NOCITOL);
        let e: f64 = options
            .weights
            .iter()
            .zip(expansion.energies.iter())
            .map(|(w, e)| w * e)
            .sum();
        let fock = reshf_fock(ao, &pairs, &expansion);

        // Residual of the ResHF condition, `X^\dagger (F^A P^A S - S P^A F^A) X` for every
        // determinant, stacked block-diagonally into the shared DIIS history.
        let fa: Vec<&Array2<f64>> = fock.iter().map(|f| &f[0]).collect();
        let fb: Vec<&Array2<f64>> = fock.iter().map(|f| &f[1]).collect();
        let da: Vec<&Array2<f64>> = dets.iter().map(|d| &*d.da).collect();
        let db: Vec<&Array2<f64>> = dets.iter().map(|d| &*d.db).collect();
        diis.push(
            &block_diagonal(&fa),
            &block_diagonal(&fb),
            &block_diagonal(&da),
            &block_diagonal(&db),
            &s_all,
            &x_all,
        );
        let err = diis.last_error_norm2().unwrap_or(f64::INFINITY).sqrt();
        let d_e = (e - e_old).abs();

        if verbose {
            println!("{:4} {:12.6} {:12.4e} {:12.4e}", iterations, e, d_e, err);
        }

        // Stop once converged or out of iterations, keeping the determinants `E` belongs to.
        iterations += 1;
        if d_e < input.scf.e_tol && err < input.scf.fds_sdf_tol {
            break (e, true);
        }
        if iterations >= max_cycle {
            break (e, false);
        }

        // Roothaan step on each weighted determinant's block of the extrapolated Fock matrices.
        let extrapolated = diis.extrapolate_fock();
        for (a, det) in dets.iter_mut().enumerate() {
            if expansion.pair_weights[(a, a)] < WEIGHTTOL {
                continue;
            }
            let (fa, fb) = match &extrapolated {
                Some((fa, fb)) => (diagonal_block(fa, a, nao), diagonal_block(fb, a, nao)),
                None => (fock[a][0].clone(), fock[a][1].clone()),
            };
            *det = reshf_step(ao, det, &fa, &fb);
        }
        e_old = e;
    };

    if verbose {
        println!("ResHF energy: {e}");
    }
    if !converged {
        println!(
            "Warning: ResHF not converged after {iterations} iterations; using the last determinants."
        );
    }

    // Put the optimised determinants back in place of the NOCI states.
    let mut out = states.to_vec();
    for (i, det) in indices.into_iter().zip(dets) {
        out[i] = det;
    }

    ResHFSolution {
        states: out,
        energy: e,
        iterations,
        converged,
    }
}
