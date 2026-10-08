// noci/matrix.rs

// Standard library imports.
use std::time::{Duration, Instant};

// External crate imports.
use ndarray::{Array1, Array2};
use rayon::prelude::*;

// Crate-root imports.
use crate::AoData;
use crate::NOCIScalar;
use crate::determinant::{NOCIIndex, NOCISpace};
use crate::elements::compare_f_pair_wicks_naive;
use crate::elements::compare_hs_pair_wicks_naive;
use crate::elements::nonorthogonalwicks::{WickScratchSpin, WicksView};
use crate::elements::{DetPair, FockData, MOCache, NOCIData};
use crate::elements::{calculate_f_pair, calculate_hs_pair, calculate_s_pair};
use crate::input::Input;
use crate::maths::general_evp;
use crate::time_call;
use crate::utils::print_array2_indexed;
use crate::write::write_hs_matrices;

/// Evaluate an arbitrary determinant-pair quantity given a closure `o`
/// which computes `U` for a batch of pairs. The closure may evaluate, for example,
/// Hamiltonian, overlap, or Fock matrix elements. Each matrix row is passed as one batch so
/// the matrix-element dispatchers can packetise its pairs for SIMD evaluation.
/// # Arguments:
/// - `left`: First set of determinants.
/// - `right`: Second set of determinants.
/// - `input`: User specified input options.
/// - `symmetric`: Whether only the upper triangle should be evaluated.
/// - `zero`: Initial value for each row output buffer.
/// - `o`: closure for batched determinant-pair evaluation.
/// # Returns:
/// - `(Vec<(usize, usize, U)>, Duration)`: Evaluated matrix elements with
///   their indices and the wall time for the evaluation.
/// # Type Parameters:
/// - `O`: Matrix-element callback over determinant pairs, Wick scratch, and row outputs.
/// - `U`: Required to be `Send` and `Copy`.
fn calculate_matrix_elements<T, U, O>(
    left: &[NOCIIndex],
    right: &[NOCIIndex],
    input: &Input,
    symmetric: bool,
    zero: U,
    o: O,
) -> (Vec<(usize, usize, U)>, Duration)
where
    T: NOCIScalar,
    U: Send + Sync + Copy,
    O: Fn(&[DetPair], Option<&mut WickScratchSpin<T>>, &mut [U]) + Sync,
{
    let nl = left.len();
    let nr = right.len();

    let t0 = Instant::now();

    // Evaluate each row's upper-triangle and diagonal, or full, columns as one batch.
    let use_wicks_scratch = input.wicks.enabled;
    let rows: Vec<Vec<(usize, usize, U)>> = (0..nl)
        .into_par_iter()
        .map_init(
            || {
                (
                    use_wicks_scratch.then(WickScratchSpin::<T>::new),
                    Vec::new(),
                    Vec::new(),
                )
            },
            |(scratch, pairs, values), i| {
                let start = if symmetric { i } else { 0 };
                pairs.clear();
                pairs.extend(
                    right[start..]
                        .iter()
                        .map(|&gdet| DetPair::new(left[i], gdet)),
                );
                values.clear();
                values.resize(pairs.len(), zero);
                o(pairs, scratch.as_mut(), values);
                (start..nr)
                    .zip(values.iter())
                    .map(|(j, &value)| (i, j, value))
                    .collect()
            },
        )
        .collect();
    let vals = rows.into_iter().flatten().collect();

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
    left: &[NOCIIndex],
    right: &[NOCIIndex],
    symmetric: bool,
) -> (Array2<T>, Duration) {
    time_call!(crate::timers::noci::add_build_full_fock, {
        let nl = left.len();
        let nr = right.len();

        // In comparison mode, build with Wick's theorem while accumulating
        // discrepancies against the generalised Slater-Condon elements.
        if data.input.wicks.enabled && data.input.wicks.compare {
            let zero = (T::from_real(0.0), (0.0, 0.0));
            let (vals, dt) = calculate_matrix_elements(
                left,
                right,
                data.input,
                symmetric,
                zero,
                |pairs, scratch, out| {
                    let scratch = scratch.unwrap();
                    for (&pair, value) in pairs.iter().zip(out) {
                        *value = compare_f_pair_wicks_naive(data, fock, pair, scratch);
                    }
                },
            );

            let mut td = 0.0;
            let mut md = 0.0;
            let mut fvals = Vec::with_capacity(vals.len());
            for (i, j, (f, (d, m))) in vals {
                fvals.push((i, j, f));
                td += d;
                md = f64::max(md, m);
            }
            println!(
                "Total naive–wicks discrepancy (Fock): {:.6e}; max element: {:.6e}",
                td, md
            );
            let f = scatter_matrix_elements(fvals, nl, nr, symmetric);
            return (f, dt);
        }

        // Otherwise evaluate each determinant pair once and scatter its
        // result into the requested symmetric or rectangular matrix.
        let zero = T::from_real(0.0);
        let (vals, dt) = calculate_matrix_elements(
            left,
            right,
            data.input,
            symmetric,
            zero,
            |pairs, scratch, out| calculate_f_pair(data, fock, pairs, scratch, out),
        );

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
pub(crate) fn build_noci_s<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    left: &[NOCIIndex],
    right: &[NOCIIndex],
    symmetric: bool,
) -> (Array2<T>, Duration) {
    time_call!(crate::timers::noci::add_build_full_overlap, {
        let nl = left.len();
        let nr = right.len();

        let zero = T::from_real(0.0);
        let (vals, dt) = calculate_matrix_elements(
            left,
            right,
            data.input,
            symmetric,
            zero,
            |pairs, scratch, out| calculate_s_pair(data, pairs, scratch, out),
        );

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
    left: &[NOCIIndex],
    right: &[NOCIIndex],
    symmetric: bool,
) -> (Array2<T>, Array2<T>, Duration) {
    time_call!(crate::timers::noci::add_build_full_hs, {
        let nl = left.len();
        let nr = right.len();

        // Compare both H and S pair elements before scattering the Wick values.
        if data.input.wicks.enabled && data.input.wicks.compare {
            let zero = ((T::from_real(0.0), T::from_real(0.0)), (0.0, 0.0));
            let (vals, dt) = calculate_matrix_elements(
                left,
                right,
                data.input,
                symmetric,
                zero,
                |pairs, scratch, out| {
                    let scratch = scratch.unwrap();
                    for (&pair, value) in pairs.iter().zip(out) {
                        *value = compare_hs_pair_wicks_naive(data, pair, scratch);
                    }
                },
            );

            let mut td = 0.0;
            let mut md = 0.0;
            let mut hsvals = Vec::with_capacity(vals.len());
            for (i, j, (hs, (d, m))) in vals {
                hsvals.push((i, j, hs));
                td += d;
                md = f64::max(md, m);
            }
            println!(
                "Total naive–wicks discrepancy (Hamiltonian and overlap): {:.6e}; max element: {:.6e}",
                td, md
            );
            let (h, s) = scatter_matrix_elements(hsvals, nl, nr, symmetric);
            return (h, s, dt);
        }

        // Assemble ordinary H and S pair elements with the selected evaluator.
        let zero = (T::from_real(0.0), T::from_real(0.0));
        let (vals, dt) = calculate_matrix_elements(
            left,
            right,
            data.input,
            symmetric,
            zero,
            |pairs, scratch, out| calculate_hs_pair(data, pairs, scratch, out),
        );

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
/// - `space`: Authoritative retained determinant topology and parent orbital frames.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `wicks`: Optional precomputed Wick's intermediates.
/// # Returns:
/// - `(f64, Array1<T>, Duration)`: The lowest NOCI eigenvalue, its coefficient
///   vector in the NOCI basis, and the time spent building the Hamiltonian/overlap matrices.
pub fn calculate_noci_energy<T: NOCIScalar>(
    ao: &AoData,
    input: &Input,
    space: &NOCISpace<T>,
    tol: f64,
    mocache: &[MOCache<T>],
    wicks: Option<&WicksView<T>>,
) -> (f64, Array1<T>, Duration) {
    let data = NOCIData::new(ao, space, input, tol, wicks).withmocache(mocache);
    let indices = (0..space.len()).map(NOCIIndex).collect::<Vec<_>>();
    let (h, s, d_hs) = build_noci_hs(&data, &indices, &indices, true);

    // The shifted matrix is diagnostic; the physical energy comes from
    // solving the unshifted generalised eigenproblem `Hc = ESc`.
    let h_shift = &h - &s.mapv(|x| space.parents[0].e * x);
    if input.write.verbose >= 2 {
        println!("{}", "=".repeat(100));
        println!("NOCI-reference Hamiltonian:");
        print_array2_indexed(&h);
        println!("NOCI-reference Overlap:");
        print_array2_indexed(&s);
        println!("Shifted NOCI-reference Hamiltonian");
        print_array2_indexed(&h_shift);
    }

    let (evals, c) = general_evp(&h, &s, true, tol);
    if input.write.verbose >= 1 {
        println!("GEVP eigenvalues in NOCI-reference basis: {}", evals);
    }

    let c0 = c.column(0).to_owned();
    (evals[0], c0, d_hs)
}

/// Trait which defines how returned determinant-pair quantities should be scattered into matrices.
pub(in crate::noci) trait ScatterValue: Sized + Copy {
    type Output;

    /// Construct zero initialised output.
    /// # Arguments:
    /// - `nl`: Length of determinant set 1.
    /// - `nr`: Length of determinant set 2.
    /// # Returns:
    /// - `Self::Output`: Zero initialised output container.
    fn zeros(
        nl: usize,
        nr: usize,
    ) -> Self::Output;

    /// Write a value into the output at indices i, j.
    /// # Arguments:
    /// - `out`: Output container to write into.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: Matrix element value.
    /// # Returns
    /// - `()`: Writes the matrix element into `out`.
    fn write(
        out: &mut Self::Output,
        i: usize,
        j: usize,
        val: Self,
    );

    /// Value to write into the mirrored Hermitian position.
    /// # Arguments:
    /// - `self`: Matrix element value.
    /// # Returns:
    /// - `Self`: Complex-conjugated mirrored value.
    fn mirror(self) -> Self;
}

impl<T: NOCIScalar> ScatterValue for T {
    type Output = Array2<T>;

    /// Construct zero initialised matrix.
    /// # Arguments:
    /// - `nl`: Number of rows.
    /// - `nr`: Number of columns.
    /// # Returns:
    /// - `Array2<T>`: Zero initialised matrix.
    fn zeros(
        nl: usize,
        nr: usize,
    ) -> Self::Output {
        Array2::<T>::zeros((nl, nr))
    }

    /// Write scalar value into matrix at row `i` and column `j`.
    /// # Arguments:
    /// - `out`: Matrix to write into.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: Value to write.
    /// # Returns
    /// - `()`: Writes the scalar into the matrix.
    fn write(
        out: &mut Self::Output,
        i: usize,
        j: usize,
        val: Self,
    ) {
        out[(i, j)] = val;
    }

    /// Return Hermitian mirrored value for the lower triangle.
    /// # Arguments:
    /// - `self`: Matrix element value.
    /// # Returns:
    /// - `Self`: Complex conjugated matrix element value.
    fn mirror(self) -> Self {
        self.conj()
    }
}

impl<T: NOCIScalar> ScatterValue for (T, T) {
    type Output = (Array2<T>, Array2<T>);

    /// Construct pair of zero initialised matrices.
    /// # Arguments:
    /// - `nl`: Number of rows.
    /// - `nr`: Number of columns.
    /// # Returns:
    /// - `(Array2<T>, Array2<T>)`: Pair of zero initialised matrices.
    fn zeros(
        nl: usize,
        nr: usize,
    ) -> Self::Output {
        (Array2::<T>::zeros((nl, nr)), Array2::<T>::zeros((nl, nr)))
    }

    /// Write pair of scalar values into pair of matrices at row `i` and column `j`.
    /// # Arguments:
    /// - `out`: Pair of matrices to write into.
    /// - `i`: Row index.
    /// - `j`: Column index.
    /// - `val`: Pair of values to write.
    /// # Returns
    /// - `()`: Writes both scalars into their matrices.
    fn write(
        out: &mut Self::Output,
        i: usize,
        j: usize,
        val: Self,
    ) {
        out.0[(i, j)] = val.0;
        out.1[(i, j)] = val.1;
    }

    /// Return Hermitian mirrored values for the lower triangle.
    /// # Arguments:
    /// - `self`: Pair of matrix element values.
    /// # Returns:
    /// - `Self`: Pair of complex conjugated matrix element values.
    fn mirror(self) -> Self {
        (self.0.conj(), self.1.conj())
    }
}
