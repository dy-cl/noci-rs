// noci/matrix.rs
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use super::types::{DetPair, FockData, MOCache, NOCIData, NOCIScalar, ScatterValue};
use crate::input::Input;
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use crate::time_call;
use crate::{AoData, DetState};

use super::fock::compare_f_pair_wicks_naive;
use super::hs::compare_hs_pair_wicks_naive;
use crate::maths::general_evp;
use crate::noci::{calculate_f_pair, calculate_hs_pair, calculate_s_pair};
use crate::utils::print_array2_indexed;
use crate::write::write_hs_matrices;

/// Evaluate an arbitrary determinant-pair quantity given a closure `o`
/// which computes `U` for the pair. The closure may evaluate, for example,
/// Hamiltonian, overlap, or Fock matrix elements.
/// # Arguments:
/// - `left`: First set of determinants.
/// - `right`: Second set of determinants.
/// - `input`: User specified input options.
/// - `symmetric`: Whether only the upper triangle should be evaluated.
/// - `o`: closure for determinant-pair evaluation.
/// # Returns:
/// - `(Vec<(usize, usize, U)>, Duration)`: Evaluated matrix elements with
///   their indices and the wall time for the evaluation.
/// # Type Parameters:
/// - `O`: `Fn(&DetState<T>, &DetState<T>, Option<&mut WickScratchSpin<T>>) -> U` and `Sync`.
/// - `U`: Required to be `Send`.
fn calculate_matrix_elements<T, U, O>(
    left: &[DetState<T>],
    right: &[DetState<T>],
    input: &Input,
    symmetric: bool,
    o: O,
) -> (Vec<(usize, usize, U)>, Duration)
where
    T: NOCIScalar,
    U: Send,
    O: Fn(&DetState<T>, &DetState<T>, Option<&mut WickScratchSpin<T>>) -> U + Sync,
{
    let nl = left.len();
    let nr = right.len();

    // Build list of all upper-triangle and diagonal pairs \Lambda, \Gamma.
    let pairs: Vec<(usize, usize)> = if symmetric {
        (0..nl).flat_map(|i| (i..nr).map(move |j| (i, j))).collect()
    } else {
        (0..nl).flat_map(|i| (0..nr).map(move |j| (i, j))).collect()
    };

    let t0 = Instant::now();

    let use_wicks_scratch = input.wicks.enabled;
    let vals = if use_wicks_scratch {
        pairs
            .par_iter()
            .map_init(WickScratchSpin::<T>::new, |scratch, &(i, j)| {
                (i, j, o(&left[i], &right[j], Some(scratch)))
            })
            .collect()
    } else {
        pairs
            .par_iter()
            .map(|&(i, j)| (i, j, o(&left[i], &right[j], None)))
            .collect()
    };

    let dt = t0.elapsed();

    (vals, dt)
}

/// Scatter matrix elements into 2D Array.
/// # Arguments:
/// - `vals`: Usize, U)>, matrix elements and indices.
/// - `nl`: Length of determinant set 1.
/// - `nr`: Length of determinant set 2.
/// - `symmetric`: Whether symmetry should be used to fill the lower triangle.
/// # Returns:
/// - `U::Output`: Scattered dense matrix or matrix pair.
/// # Type Parameters:
/// - `U`: Type implementing `ScatterValue`.
fn scatter_matrix_elements<U>(
    vals: Vec<(usize, usize, U)>,
    nl: usize,
    nr: usize,
    symmetric: bool,
) -> U::Output
where
    U: ScatterValue + Copy,
{
    let mut out = U::zeros(nl, nr);
    for (i, j, val) in vals {
        U::write(&mut out, i, j, val);
        if symmetric && i != j {
            U::write(&mut out, j, i, val.mirror());
        }
    }
    out
}

/// Construct the full NOCI Fock matrix using either the generalised
/// Slater-Condon rules or extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `left`: First set of determinants.
/// - `right`: Second set of determinants.
/// - `symmetric`: Whether the matrix is symmetric.
/// # Returns:
/// - `(Array2<T>, Duration)`: NOCI Fock matrix and matrix-build time.
pub(crate) fn build_noci_fock<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    left: &[DetState<T>],
    right: &[DetState<T>],
    symmetric: bool,
) -> (Array2<T>, Duration) {
    time_call!(crate::timers::noci::add_build_full_fock, {
        let nl = left.len();
        let nr = right.len();

        if data.input.wicks.enabled && data.input.wicks.compare {
            let (vals, dt) = calculate_matrix_elements(
                left,
                right,
                data.input,
                symmetric,
                |ldet, gdet, scratch| {
                    compare_f_pair_wicks_naive(
                        data,
                        fock,
                        DetPair::new(ldet, gdet),
                        scratch.unwrap(),
                    )
                },
            );

            let mut td = 0.0;
            let mut fvals = Vec::with_capacity(vals.len());
            for (i, j, (f, d)) in vals {
                fvals.push((i, j, f));
                td += d;
            }
            println!("Total naive–wicks discrepancy (Fock): {:.6e}", td);
            let f = scatter_matrix_elements(fvals, nl, nr, symmetric);
            return (f, dt);
        }

        let (vals, dt) =
            calculate_matrix_elements(left, right, data.input, symmetric, |ldet, gdet, scratch| {
                calculate_f_pair(data, fock, DetPair::new(ldet, gdet), scratch)
            });

        let f = scatter_matrix_elements(vals, nl, nr, symmetric);
        (f, dt)
    })
}

/// Form the full overlap matrix using either the generalised Slater-Condon
/// rules or extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `left`: First set of determinants.
/// - `right`: Second set of determinants.
/// - `symmetric`: Whether the matrix is symmetric.
/// # Returns:
/// - `(Array2<T>, Duration)`: The overlap matrix and the matrix-build time.
pub fn build_noci_s<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    left: &[DetState<T>],
    right: &[DetState<T>],
    symmetric: bool,
) -> (Array2<T>, Duration) {
    time_call!(crate::timers::noci::add_build_full_overlap, {
        let nl = left.len();
        let nr = right.len();

        let (vals, dt) =
            calculate_matrix_elements(left, right, data.input, symmetric, |ldet, gdet, scratch| {
                calculate_s_pair(data, DetPair::new(ldet, gdet), scratch)
            });

        let s = scatter_matrix_elements(vals, nl, nr, symmetric);
        (s, dt)
    })
}

/// Form the full Hamiltonian and overlap matrices using either the
/// generalised Slater-Condon rules or extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `left`: First set of determinants.
/// - `right`: Second set of determinants.
/// - `symmetric`: Whether the matrices are symmetric.
/// # Returns:
/// - `(Array2<T>, Array2<T>, Duration)`: The Hamiltonian matrix, overlap matrix,
///   and matrix-build time.
pub fn build_noci_hs<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    left: &[DetState<T>],
    right: &[DetState<T>],
    symmetric: bool,
) -> (Array2<T>, Array2<T>, Duration) {
    time_call!(crate::timers::noci::add_build_full_hs, {
        let nl = left.len();
        let nr = right.len();

        if data.input.wicks.enabled && data.input.wicks.compare {
            let (vals, dt) = calculate_matrix_elements(
                left,
                right,
                data.input,
                symmetric,
                |ldet, gdet, scratch| {
                    compare_hs_pair_wicks_naive(data, DetPair::new(ldet, gdet), scratch.unwrap())
                },
            );

            let mut td = 0.0;
            let mut hsvals = Vec::with_capacity(vals.len());
            for (i, j, (hs, d)) in vals {
                hsvals.push((i, j, hs));
                td += d;
            }
            println!(
                "Total naive–wicks discrepancy (Hamiltonian and overlap): {:.6e}",
                td
            );
            let (h, s) = scatter_matrix_elements(hsvals, nl, nr, symmetric);
            return (h, s, dt);
        }

        let (vals, dt) =
            calculate_matrix_elements(left, right, data.input, symmetric, |ldet, gdet, scratch| {
                calculate_hs_pair(data, DetPair::new(ldet, gdet), scratch)
            });

        let (h, s) = scatter_matrix_elements(vals, nl, nr, symmetric);

        if data.input.write.write_matrices {
            write_hs_matrices(&data.input.write.write_dir, &h, &s);
        }
        (h, s, dt)
    })
}

/// Calculate the NOCI ground-state energy by solving the generalised
/// eigenvalue problem for the NOCI Hamiltonian and overlap matrices.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `scfstates`: Vector of all SCF states used in the NOCI basis.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `wicks`: Optional precomputed Wick's intermediates.
/// # Returns:
/// - `(f64, Array1<T>, Duration)`: The lowest NOCI eigenvalue, its coefficient
///   vector in the NOCI basis, and the time spent building the Hamiltonian/overlap matrices.
pub fn calculate_noci_energy<T: NOCIScalar>(
    ao: &AoData,
    input: &Input,
    scfstates: &[DetState<T>],
    tol: f64,
    mocache: &[MOCache<T>],
    wicks: Option<&WicksView<T>>,
) -> (f64, Array1<T>, Duration) {
    let data = NOCIData::new(ao, scfstates, input, tol, wicks).withmocache(mocache);
    let (h, s, d_hs) = build_noci_hs(&data, scfstates, scfstates, true);

    println!("{}", "=".repeat(100));
    println!("NOCI-reference Hamiltonian:");
    print_array2_indexed(&h);
    println!("NOCI-reference Overlap:");
    print_array2_indexed(&s);
    println!("Shifted NOCI-reference Hamiltonian");
    let h_shift = &h - &s.mapv(|x| scfstates[0].e * x);
    print_array2_indexed(&h_shift);

    let (evals, c) = general_evp(&h, &s, true, tol);
    println!("GEVP eigenvalues in NOCI-reference basis: {}", evals);

    let c0 = c.column(0).to_owned();
    (evals[0], c0, d_hs)
}
