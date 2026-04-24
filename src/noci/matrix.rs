// noci/matrix.rs
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::{AoData, SCFState};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use crate::input::Input;
use super::types::{MOCache, NOCIData, FockData, DetPair, ScatterValue};
use crate::time_call;

use crate::utils::print_array2;
use crate::noci::{calculate_f_pair, calculate_s_pair, calculate_hs_pair};
use crate::maths::general_evp_real;
use crate::write::write_hs_matrices;
use super::hs::compare_hs_pair_wicks_naive;

/// Evaluate an arbitrary determinant-pair quantity given a closure `o`
/// which computes `T` for the pair. The closure may evaluate, for example,
/// Hamiltonian, overlap, or Fock matrix elements.
/// # Arguments:
/// - `left`: First set of determinants.
/// - `right`: Second set of determinants.
/// - `input`: User specified input options.
/// - `symmetric`: Whether only the upper triangle should be evaluated.
/// - `o`: closure for determinant-pair evaluation.
/// # Returns:
/// - `(Vec<(usize, usize, T)>, Duration)`: Evaluated matrix elements with
///   their indices and the wall time for the evaluation.
/// # Type Parameters:
/// - `O`: &SCFState, Option<&mut WickScratchSpin>) -> T` and `Sync`.
/// - `T`: Required to be `Send`.
fn calculate_matrix_elements<T, O> (left: &[SCFState], right: &[SCFState], input: &Input, symmetric: bool, o: O) -> (Vec<(usize, usize, T)>, Duration)
    where T: Send, O: Fn(&SCFState, &SCFState, Option<&mut WickScratchSpin>) -> T + Sync {

    let nl = left.len();
    let nr = right.len();
    
    // Build list of all upper-triangle and diagonal pairs \Lambda, \Gamma.
    let pairs: Vec<(usize, usize)> = if symmetric {(0..nl).flat_map(|i| (i..nr).map(move |j| (i, j))).collect()} 
    else {(0..nl).flat_map(|i| (0..nr).map(move |j| (i, j))).collect()};
    
    let t0 = Instant::now();
    let vals = if input.wicks.enabled {
        pairs.par_iter().map_init(WickScratchSpin::new, |scratch, &(i, j)| {(i, j, o(&left[i], &right[j], Some(scratch)))}).collect()
    } else {
         pairs.par_iter().map(|&(i, j)| (i, j, o(&left[i], &right[j], None))).collect()
    };
    let dt = t0.elapsed();

    (vals, dt)
}

/// Scatter matrix elements into 2D Array.
/// # Arguments:
/// - `vals`: Usize, T)>, matrix elements and indices.
/// - `nl`: Length of determinant set 1.
/// - `nr`: Length of determinant set 2.
/// - `symmetric`: Whether symmetry should be used to fill the lower triangle.
/// # Returns:
/// - `T::Output`: Scattered dense matrix or matrix pair.
/// # Type Parameters:
/// - `T`: F64)`.
fn scatter_matrix_elements<T>(vals: Vec<(usize, usize, T)>, nl: usize, nr: usize, symmetric: bool) -> T::Output 
    where T: ScatterValue + Copy {
    let mut out = T::zeros(nl, nr);
    for (i, j, val) in vals {
        T::write(&mut out, i, j, val);
        if symmetric && i != j {
            T::write(&mut out, j, i, val);
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
/// - `(Array2<f64>, Duration)`: NOCI Fock matrix and matrix-build time.
pub(crate) fn build_noci_fock(data: &NOCIData<'_>, fock: &FockData<'_>, left: &[SCFState], right: &[SCFState], symmetric: bool) -> (Array2<f64>, Duration) {
    time_call!(crate::timers::noci::add_build_full_fock, {
        let nl = left.len();
        let nr = right.len();

        let (vals, dt) = calculate_matrix_elements(left, right, data.input, symmetric, |ldet, gdet, scratch| {
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
/// - `(Array2<f64>, Duration)`: The overlap matrix and the matrix-build time.
pub fn build_noci_s(data: &NOCIData<'_>, left: &[SCFState], right: &[SCFState], symmetric: bool) -> (Array2<f64>, Duration) {
    time_call!(crate::timers::noci::add_build_full_overlap, {
        let nl = left.len();
        let nr = right.len();

        let (vals, dt) = calculate_matrix_elements(left, right, data.input, symmetric, |ldet, gdet, scratch| {
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
/// - `(Array2<f64>, Array2<f64>, Duration)`: The Hamiltonian matrix, overlap matrix,
///   and matrix-build time.
pub fn build_noci_hs(data: &NOCIData<'_>, left: &[SCFState], right: &[SCFState], symmetric: bool) -> (Array2<f64>, Array2<f64>, Duration) {
    time_call!(crate::timers::noci::add_build_full_hs, {
        let nl = left.len();
        let nr = right.len();

        if data.input.wicks.compare {
            let (vals, dt) = calculate_matrix_elements(left, right, data.input, symmetric, |ldet, gdet, scratch| {
                compare_hs_pair_wicks_naive(data, DetPair::new(ldet, gdet), scratch.unwrap())
            });

            let mut td = 0.0;
            let mut hsvals = Vec::with_capacity(vals.len());
            for (i, j, (hs, d)) in vals {
                hsvals.push((i, j, hs));
                td += d;
            }
            println!("Total naive–wicks discrepancy: {:.6e}", td);
            let (h, s) = scatter_matrix_elements(hsvals, nl, nr, symmetric);
            return (h, s, dt);
        }

        let (vals, dt) = calculate_matrix_elements(left, right, data.input, symmetric, |ldet, gdet, scratch| {
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
/// - `(f64, Array1<f64>, Duration)`: The lowest NOCI eigenvalue, its coefficient
///   vector in the NOCI basis, and the time spent building the Hamiltonian/overlap matrices.
pub fn calculate_noci_energy(ao: &AoData, input: &Input, scfstates: &[SCFState], tol: f64, mocache: &[MOCache], wicks: Option<&WicksView>) -> (f64, Array1<f64>, Duration) {
    let data = NOCIData::new(ao, scfstates, input, tol, wicks).withmocache(mocache);
    let (h, s, d_hs) = build_noci_hs(&data, scfstates, scfstates, true);

    println!("{}", "=".repeat(100));
    println!("NOCI-reference Hamiltonian:");
    print_array2(&h);
    println!("NOCI-reference Overlap:");
    print_array2(&s);
    println!("Shifted NOCI-reference Hamiltonian");
    let h_shift = &h.map(|z: &f64| z) - scfstates[0].e * &s;
    print_array2(&h_shift);
    let (evals, c) = general_evp_real(&h, &s, true, f64::EPSILON);
    println!("GEVP eigenvalues in NOCI-reference basis: {}", evals);

    let c0 = c.column(0).to_owned();
    (evals[0], c0, d_hs)
}
