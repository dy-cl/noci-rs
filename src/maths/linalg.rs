// maths/linalg.rs

use crate::StateScalar;
use ndarray::{Array1, Array2, Axis, ShapeBuilder};
use ndarray_linalg::{Eig, Eigh, EighInto, Inverse, UPLO};
use num_complex::Complex64;
use rayon::prelude::*;

/// Compute the dot product of two real contiguous vectors.
/// This kernel uses multiple accumulators so LLVM can more easily vectorise
/// the reduction in hot loops over short-to-medium rows.
/// # Arguments:
/// - `x`: First real vector.
/// - `y`: Second real vector.
/// # Returns
/// - `f64`: Dot product \(\sum_i x_i y_i\), truncated to the shorter input length.
#[inline(always)]
pub fn dot_f64(
    x: &[f64],
    y: &[f64],
) -> f64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: Runtime feature detection above guarantees that AVX instructions are available.
            return unsafe { dot_f64_avx(x, y) };
        }
    }

    dot_f64_scalar(x, y)
}

/// Compute a scalar dot product of two real contiguous vectors.
/// # Arguments:
/// - `x`: First real vector.
/// - `y`: Second real vector.
/// # Returns
/// - `f64`: Dot product \(\sum_i x_i y_i\), truncated to the shorter input length.
#[inline(always)]
fn dot_f64_scalar(
    x: &[f64],
    y: &[f64],
) -> f64 {
    let n = x.len().min(y.len());
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;

    let mut i = 0usize;
    while i + 4 <= n {
        s0 += x[i] * y[i];
        s1 += x[i + 1] * y[i + 1];
        s2 += x[i + 2] * y[i + 2];
        s3 += x[i + 3] * y[i + 3];
        i += 4;
    }

    let mut tail = 0.0;
    while i < n {
        tail += x[i] * y[i];
        i += 1;
    }

    (s0 + s1) + (s2 + s3) + tail
}

/// Compute the dot product of two real contiguous vectors using AVX vector lanes.
/// # Arguments:
/// - `x`: First real vector.
/// - `y`: Second real vector.
/// # Returns
/// - `f64`: Dot product \(\sum_i x_i y_i\), truncated to the shorter input length.
#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx")]
unsafe fn dot_f64_avx(
    x: &[f64],
    y: &[f64],
) -> f64 {
    use std::arch::x86::*;

    let n = x.len().min(y.len());
    let mut acc = _mm256_setzero_pd();
    let mut i = 0usize;

    while i + 4 <= n {
        let xv = unsafe { _mm256_loadu_pd(x.as_ptr().add(i)) };
        let yv = unsafe { _mm256_loadu_pd(y.as_ptr().add(i)) };
        let prod = _mm256_mul_pd(xv, yv);
        acc = _mm256_add_pd(acc, prod);
        i += 4;
    }

    let mut lanes = [0.0; 4];
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
    let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);

    while i < n {
        total += x[i] * y[i];
        i += 1;
    }

    total
}

/// Compute the dot product of two real contiguous vectors using AVX vector lanes.
/// # Arguments:
/// - `x`: First real vector.
/// - `y`: Second real vector.
/// # Returns
/// - `f64`: Dot product \(\sum_i x_i y_i\), truncated to the shorter input length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_f64_avx(
    x: &[f64],
    y: &[f64],
) -> f64 {
    use std::arch::x86_64::*;

    let n = x.len().min(y.len());
    let mut acc = _mm256_setzero_pd();
    let mut i = 0usize;

    while i + 4 <= n {
        let xv = unsafe { _mm256_loadu_pd(x.as_ptr().add(i)) };
        let yv = unsafe { _mm256_loadu_pd(y.as_ptr().add(i)) };
        let prod = _mm256_mul_pd(xv, yv);
        acc = _mm256_add_pd(acc, prod);
        i += 4;
    }

    let mut lanes = [0.0; 4];
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
    let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);

    while i < n {
        total += x[i] * y[i];
        i += 1;
    }

    total
}

/// Hermitian adjoint of a matrix.
/// # Arguments:
/// - `a`: Matrix to conjugate transpose.
/// # Returns
/// - `Array2<T>`: Hermitian adjoint `a^\dagger`.
pub fn adjoint<T: StateScalar>(a: &Array2<T>) -> Array2<T> {
    a.t().mapv(|z| z.conj())
}

/// Convert a real vector to scalar type `T`.
/// # Arguments:
/// - `a`: Real vector.
/// # Returns
/// - `Array1<T>`: Vector with entries promoted to `T`.
pub fn real1_as<T: StateScalar>(a: &Array1<f64>) -> Array1<T> {
    a.mapv(T::from_real)
}

/// Convert a real matrix to scalar type `T`.
/// # Arguments:
/// - `a`: Real matrix.
/// # Returns
/// - `Array2<T>`: Matrix with entries promoted to `T`.
pub fn real2_as<T: StateScalar>(a: &Array2<f64>) -> Array2<T> {
    a.mapv(T::from_real)
}

/// Return positive eigenvalue subspace of a Hermitian matrix.
/// # Arguments:
/// - `s`: Hermitian matrix, uses only the lower triangle.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns
/// - `(Array1<f64>, Array2<T>)`: Positive eigenvalues and their associated eigenvectors.
pub fn positive_subspace<T: StateScalar>(
    s: &Array2<T>,
    tol: f64,
) -> (Array1<f64>, Array2<T>) {
    let (lambdas, evecs) = hermitian_eigh(s, UPLO::Lower);

    let pos: Vec<usize> = lambdas
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| if x > tol { Some(i) } else { None })
        .collect();

    let vals = Array1::from_iter(pos.iter().map(|&i| lambdas[i]));
    let mut vecs = Array2::<T>::zeros((s.nrows(), pos.len()));

    for (j, &i) in pos.iter().enumerate() {
        vecs.column_mut(j).assign(&evecs.column(i));
    }

    (vals, vecs)
}

/// Loewdin symmetric orthogonalizer, computes `X = S^{-1/2}`.
/// If `project` is true, returns the rectangular orthogonalizer `X = U_+ Lambda_+^{-1/2}`.
/// If `project` is false, returns the square orthogonalizer `X = U Lambda^{-1/2} U^\dagger`.
/// # Arguments:
/// - `s`: Hermitian matrix, uses only the lower triangle.
/// - `project`: Whether or not to project to non-zero positive subspace of `s`.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns
/// - `Array2<T>`: Orthogonalizer.
pub fn loewdin_x<T: StateScalar>(
    s: &Array2<T>,
    project: bool,
    tol: f64,
) -> Array2<T> {
    if project {
        let (vals, vecs) = positive_subspace(s, tol);
        let d = Array2::from_diag(&Array1::from_iter(
            vals.iter().map(|&x| T::from_real(1.0 / x.sqrt())),
        ));
        vecs.dot(&d)
    } else {
        let (vals, vecs) = hermitian_eigh(s, UPLO::Lower);
        let d = Array2::from_diag(&Array1::from_iter(
            vals.iter().map(|&x| T::from_real(1.0 / x.sqrt())),
        ));
        vecs.dot(&d).dot(&adjoint(&vecs))
    }
}

/// Solve the Hermitian generalized eigenproblem `H C = S C e`.
/// # Arguments:
/// - `h`: Hermitian Hamiltonian matrix, uses only the lower triangle.
/// - `s`: Hermitian overlap matrix, uses only the lower triangle.
/// - `project`: Whether or not to project to non-zero positive subspace of `s`.
/// - `tol`: Tolerance for whether an overlap eigenvalue is considered zero.
/// # Returns
/// - `(Array1<f64>, Array2<T>)`: Eigenvalues and generalized eigenvectors.
pub fn general_evp<T: StateScalar>(
    h: &Array2<T>,
    s: &Array2<T>,
    project: bool,
    tol: f64,
) -> (Array1<f64>, Array2<T>) {
    let x = loewdin_x(s, project, tol);
    let ht = adjoint(&x).dot(h).dot(&x);
    let (epsilon, u) = hermitian_eigh(&ht, UPLO::Lower);
    let c = x.dot(&u);
    (epsilon, c)
}

/// Solve the Hermitian generalized eigenproblem `H C = S C e` with pre-computed `x`.
/// # Arguments:
/// - `h`: Hermitian Hamiltonian matrix, uses only the lower triangle.
/// - `x`: Precomputed symmetric orthogonaliser `s^{-1/2}`.
/// # Returns
/// - `(Array1<f64>, Array2<T>)`: Eigenvalues and generalized eigenvectors.
pub fn general_evp_x(
    h: &Array2<f64>,
    x: &Array2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let ht = adjoint(x).dot(h).dot(x);
    let (epsilon, u) = hermitian_eigh(&ht, UPLO::Lower);
    let c = x.dot(&u);
    (epsilon, c)
}

/// Diagonalise a Hermitian matrix using explicit column-major storage.
/// This avoids transposing complex C-order matrices before the LAPACK call.
/// # Arguments:
/// - `a`: Hermitian matrix.
/// - `uplo`: Triangle of the matrix used for diagonalisation.
/// # Returns:
/// - `(Array1<f64>, Array2<T>)`: Eigenvalues and eigenvectors.
fn hermitian_eigh<T: StateScalar>(
    a: &Array2<T>,
    uplo: UPLO,
) -> (Array1<f64>, Array2<T>) {
    let mut af = Array2::<T>::zeros((a.nrows(), a.ncols()).f());
    af.assign(a);

    af.eigh_into(uplo).unwrap()
}

/// Solve a symmetric positive-semidefinite linear system by eigenvalue projection.
///
/// This computes the minimum-norm pseudoinverse solution `x = A^+ b` by
/// discarding eigenvalues below `tol`.
///
/// # Arguments:
/// - `a`: Real symmetric positive-semidefinite matrix.
/// - `b`: Right-hand-side vector.
/// - `tol`: Eigenvalue cutoff for the pseudoinverse.
///
/// # Returns:
/// - `Array1<f64>`: Projected pseudoinverse solution.
pub fn solve_pseudoinverse(
    a: &Array2<f64>,
    b: &Array1<f64>,
    tol: f64,
) -> Array1<f64> {
    let (evals, evecs) = a
        .clone()
        .eigh(UPLO::Lower)
        .expect("PSD pseudoinverse diagonalisation failed");

    let mut x = Array1::zeros(b.len());

    for k in 0..evals.len() {
        if evals[k] <= tol {
            continue;
        }

        let col = evecs.column(k);
        let coeff = col.dot(b) / evals[k];

        for i in 0..x.len() {
            x[i] += col[i] * coeff;
        }
    }

    x
}

/// Diagonalise a complex-symmetric matrix and transpose-normalise eigenvectors.
/// This is for holomorphic SCF blocks, not for Hermitian NOCI matrices.
/// # Arguments:
/// - `a`: Complex-symmetric occupied or virtual Fock block.
/// # Returns
/// - `(Array1<Complex64>, Array2<Complex64>)`: Eigenvalues and transpose-normalised eigenvectors.
pub fn symmetric_evp_complex(a: &Array2<Complex64>) -> (Array1<Complex64>, Array2<Complex64>) {
    if a.nrows() == 0 {
        return (Array1::zeros(0), Array2::zeros((0, 0)));
    }

    if a.nrows() == 1 {
        return (
            Array1::from_vec(vec![a[(0, 0)]]),
            Array2::from_elem((1, 1), Complex64::new(1.0, 0.0)),
        );
    }

    let (vals, vecs) = a.eig().unwrap();
    let mut order: Vec<usize> = (0..vals.len()).collect();
    order.sort_by(|&i, &j| {
        vals[i]
            .re
            .partial_cmp(&vals[j].re)
            .unwrap()
            .then(vals[i].im.partial_cmp(&vals[j].im).unwrap())
    });

    let mut e = Array1::<Complex64>::zeros(vals.len());
    let mut u = Array2::<Complex64>::zeros(vecs.raw_dim());

    for (k, &i) in order.iter().enumerate() {
        e[k] = vals[i];

        u.column_mut(k).assign(&vecs.column(i));
    }

    transpose_orthonormalize_columns(&mut u);

    (e, u)
}

/// Modified Gram-Schmidt orthonormalisation in the transpose metric.
/// # Arguments:
/// - `u`: Matrix whose columns are orthonormalised in place so that `U^T U = I`.
/// # Returns:
/// - `()`: Updates `u` in place.
fn transpose_orthonormalize_columns(u: &mut Array2<Complex64>) {
    let nrows = u.nrows();
    let ncols = u.ncols();

    for j in 0..ncols {
        let mut col = u.column(j).to_owned();

        for i in 0..j {
            let qi = u.column(i).to_owned();
            let proj = qi.dot(&col);
            col = &col - &qi.mapv(|z| z * proj);
        }

        let mut nrm = col.dot(&col).sqrt();
        if nrm.norm() <= 1e-14 {
            col.fill(Complex64::new(0.0, 0.0));
            for k in 0..nrows {
                col[k] = Complex64::new(1.0, 0.0);
                for i in 0..j {
                    let qi = u.column(i).to_owned();
                    let proj = qi.dot(&col);
                    col = &col - &qi.mapv(|z| z * proj);
                }
                nrm = col.dot(&col).sqrt();
                if nrm.norm() > 1e-14 {
                    break;
                }
                col[k] = Complex64::new(0.0, 0.0);
            }
        }

        if nrm.norm() > 1e-14 {
            col.mapv_inplace(|z| z / nrm);
        }
        u.column_mut(j).assign(&col);
    }
}

/// Compute a dense complex matrix exponential through diagonalisation.
/// # Arguments:
/// - `a`: Complex square matrix.
/// # Returns
/// - `Array2<Complex64>`: Matrix exponential.
pub fn matrix_exp_complex(a: &Array2<Complex64>) -> Array2<Complex64> {
    if a.nrows() == 0 {
        return Array2::zeros((0, 0));
    }

    let (vals, vecs) = a.eig().unwrap();
    let d = Array2::from_diag(&vals.mapv(|z| z.exp()));
    vecs.dot(&d).dot(&vecs.inv().unwrap())
}

/// Reorthonormalise complex orbitals in a transpose metric.
/// This is for holomorphic SCF, where the metric condition is `C^T S C = I`.
/// # Arguments:
/// - `c`: MO coefficient matrix.
/// - `s`: Real AO overlap matrix defining the metric.
/// # Returns
/// - `Array2<Complex64>`: Coefficients transformed so that `C^T S C = I`.
pub fn complex_metric_orthonormalize(
    c: &Array2<Complex64>,
    s: &Array2<f64>,
) -> Array2<Complex64> {
    let sc = real2_as::<Complex64>(s);
    let m = c.t().dot(&sc).dot(c);
    let (vals, vecs) = m.eig().unwrap();
    let d = Array2::from_diag(&vals.mapv(|z| Complex64::new(1.0, 0.0) / z.sqrt()));
    c.dot(&vecs.dot(&d).dot(&vecs.inv().unwrap()))
}

/// Maximum Hermitian error of a matrix.
/// # Arguments:
/// - `a`: Matrix to check.
/// # Returns
/// - `f64`: Maximum `|a_ij - a_ji^*|`.
pub fn max_hermitian_error<T: StateScalar>(a: &Array2<T>) -> f64 {
    let mut err = 0.0;

    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let d = (a[(i, j)] - a[(j, i)].conj()).abs();
            if d > err {
                err = d;
            }
        }
    }
    err
}

/// Calculate a matrix-vector product `H C = U` in parallel.
/// # Arguments  
/// - `h`: Matrix.
/// - `c`: Vector.
/// # Returns
/// - `Array1<T>`: Matrix-vector product.
pub fn parallel_matvec<T: StateScalar>(
    h: &Array2<T>,
    c: &Array1<T>,
) -> Array1<T> {
    let result: Vec<T> = h
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| {
            let mut acc = T::from_real(0.0);
            for (&hij, &cj) in row.iter().zip(c.iter()) {
                acc += hij * cj;
            }
            acc
        })
        .collect();

    Array1::from_vec(result)
}
