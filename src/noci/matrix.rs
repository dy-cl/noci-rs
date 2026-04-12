// noci/matrix.rs
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::{AoData, SCFState};
use crate::noci::{MOCache, FockMOCache};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use crate::input::Input;

use crate::utils::print_array2;
use crate::noci::{calculate_f_pair, calculate_s_pair, calculate_hs_pair};
use super::hs::{calculate_hs_pair_naive, calculate_hs_pair_wicks};
use crate::maths::general_evp_real;
use crate::write::write_hs_matrices;

// Trait which defines how returned determinant-pair quatity should be scattered into matrices.
// Used such that we can have generic scatter functions which return 1 or 2 matrices.
trait ScatterValue: Sized {
    type Output;
    /// Construct zero initialised output. 
    /// # Arguments:
    /// - `nl`: Length of determinant set 1.
    /// - `nr`: Length of determinant set 2.
    fn zeros(nl: usize, nr: usize) -> Self::Output;
    /// Write a value into the output at indices i, j.
    /// # Arguments:
    /// - `out`: Output container to write into.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: Determinant-pair value to scatter.
    fn write(out: &mut Self::Output, i: usize, j: usize, val: Self);
}

impl ScatterValue for f64 {
    type Output = Array2<f64>;
    /// Construct zero initialised output. 
    /// # Arguments:
    /// - `nl`: Length of determinant set 1.
    /// - `nr`: Length of determinant set 2.
    fn zeros(nl: usize, nr: usize) -> Self::Output {
        Array2::<f64>::zeros((nl, nr))
    }
    /// Write scalar value into matrix position (i, j).
    /// # Arguments:
    /// - `out`: Output matrix.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: Matrix element value.
    fn write(out: &mut Self::Output, i: usize, j: usize, val: Self) {
        out[(i, j)] = val;
    }
}

impl ScatterValue for (f64, f64) {
    type Output = (Array2<f64>, Array2<f64>);

    /// Construct zero initialised output. 
    /// # Arguments:
    /// - `nl`: Length of determinant set 1.
    /// - `nr`: Length of determinant set 2.
    fn zeros(nl: usize, nr: usize) -> Self::Output {
        (Array2::<f64>::zeros((nl, nr)), Array2::<f64>::zeros((nl, nr)))
    }

    /// Write scalar value into matrix position (i, j) in both matrices.
    /// # Arguments:
    /// - `out`: `Array2<f64>`), output matrices.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: F64), matrix element values.
    fn write(out: &mut Self::Output, i: usize, j: usize, val: Self) {
        out.0[(i, j)] = val.0;
        out.1[(i, j)] = val.1;
    }
}

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
/// - `ao`: Contains AO integrals and other system data.
/// - `left`: First set of determinants.
/// - `right`: Second set of determinants.
/// - `fa`: NOCI Fock matrix spin alpha.
/// - `fb`: NOCI Fock matrix spin beta.
/// - `fock_mocache`: MO-basis Fock integral caches.
/// - `wicks`: Optional precomputed Wick's intermediates.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `symmetric`: Whether the matrix is symmetric.
/// - `input`: User specified input options.
/// # Returns:
/// - `Array2<f64>`: NOCI Fock matrix.
/// - `Duration`: Matrix-build time.
pub fn build_noci_fock(ao: &AoData, left: &[SCFState], right: &[SCFState], fa: &Array2<f64>, fb: &Array2<f64>, fock_mocache: &[FockMOCache], 
                       wicks: Option<&WicksView>, tol: f64, symmetric: bool, input: &Input) -> (Array2<f64>, Duration) {
    let nl = left.len();
    let nr = right.len();

    let wicks = if input.wicks.enabled || input.wicks.compare {Some(wicks.expect("Wick's requested but found wicks: None"))} else {None};

    let (vals, dt) = calculate_matrix_elements(left, right, input, symmetric, |ldet, gdet, scratch| {
        calculate_f_pair(fa, fb, ao, ldet, gdet, tol, input, fock_mocache, wicks, scratch)
    });
    let f = scatter_matrix_elements(vals, nl, nr, symmetric);
    (f, dt)
}

/// Form the full overlap matrix using either the generalised Slater-Condon
/// rules or extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `left`: First set of determinants.
/// - `right`: Second set of determinants.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Optional precomputed Wick's intermediates.
/// - `symmetric`: Whether the matrix is symmetric.
/// # Returns:
/// - `(Array2<f64>, Duration)`: The overlap matrix and the matrix-build time.
pub fn build_noci_s(ao: &AoData, input: &Input, left: &[SCFState], right: &[SCFState], tol: f64, wicks: Option<&WicksView>, symmetric: bool) 
                     -> (Array2<f64>, Duration) {
    let nl = left.len();
    let nr = right.len();

    let wicks = if input.wicks.enabled || input.wicks.compare {Some(wicks.expect("Wick's requested but found wicks: None"))} else {None};

    let (vals, dt) = calculate_matrix_elements(left, right, input, symmetric, |ldet, gdet, scratch| {
        calculate_s_pair(ao, ldet, gdet, tol, input, wicks, scratch)
    });
    let s = scatter_matrix_elements(vals, nl, nr, symmetric);
    (s, dt)
}

/// Form the full Hamiltonian and overlap matrices using either the
/// generalised Slater-Condon rules or extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `left`: First set of determinants.
/// - `right`: Second set of determinants.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Optional precomputed Wick's intermediates.
/// - `symmetric`: Whether the matrices are symmetric.
/// # Returns:
/// - `(Array2<f64>, Array2<f64>, Duration)`: The Hamiltonian matrix, overlap matrix,
///   and matrix-build time.
pub fn build_noci_hs(ao: &AoData, input: &Input, left: &[SCFState], right: &[SCFState], tol: f64, mocache: &[MOCache], 
                     wicks: Option<&WicksView>, symmetric: bool)  -> (Array2<f64>, Array2<f64>, Duration) {
    let nl = left.len();
    let nr = right.len();

    let wicks = if input.wicks.enabled || input.wicks.compare {Some(wicks.expect("Wick's requested but found wicks: None"))} else {None};

    if input.wicks.compare {
        let (vals, dt) = calculate_matrix_elements(left, right, input, symmetric, |ldet, gdet, scratch| {
            let (hn, sn) = calculate_hs_pair_naive(ao, ldet, gdet, tol);
            let (hw, sw) = calculate_hs_pair_wicks(ao, ldet, gdet, tol, wicks.unwrap(), scratch.unwrap());
            ((hw, sw), (hn - hw).abs() + (sn - sw).abs())
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

    let (vals, dt) = calculate_matrix_elements(left, right, input, symmetric, |ldet, gdet, scratch| {
        calculate_hs_pair(ao, ldet, gdet, tol, input, mocache, wicks, scratch)
    });

    let (h, s) = scatter_matrix_elements(vals, nl, nr, symmetric);

    if input.write.write_matrices {
        write_hs_matrices(&input.write.write_dir, &h, &s);
    }
    (h, s, dt)
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
    let (h, s, d_hs) = build_noci_hs(ao, input, scfstates, scfstates, tol, mocache, wicks, true);
    
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

    // Assumes columns of c are energy ordered eigenvectors
    let c0 = c.column(0).to_owned(); 
    (evals[0], c0, d_hs)
}
