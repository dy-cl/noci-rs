// maths/linalg.rs 

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Eig, Eigh, Inverse, UPLO};
use num_complex::Complex64;
use rayon::prelude::*;
use crate::StateScalar;

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
pub fn positive_subspace<T: StateScalar>(s: &Array2<T>, tol: f64) -> (Array1<f64>, Array2<T>) {
    let (lambdas, evecs) = s.eigh(UPLO::Lower).unwrap();

    let pos: Vec<usize> = lambdas
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| if x > tol {Some(i)} else {None})
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
pub fn loewdin_x<T: StateScalar>(s: &Array2<T>, project: bool, tol: f64) -> Array2<T> {
    if project {
        let (vals, vecs) = positive_subspace(s, tol);
        let d = Array2::from_diag(&Array1::from_iter(vals.iter().map(|&x| T::from_real(1.0 / x.sqrt()))));
        vecs.dot(&d)
    } else {
        let (vals, vecs) = s.eigh(UPLO::Lower).unwrap();
        let d = Array2::from_diag(&Array1::from_iter(vals.iter().map(|&x| T::from_real(1.0 / x.sqrt()))));
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
pub fn general_evp<T: StateScalar>(h: &Array2<T>, s: &Array2<T>, project: bool, tol: f64) -> (Array1<f64>, Array2<T>) {
    let x = loewdin_x(s, project, tol);
    let ht = adjoint(&x).dot(h).dot(&x);
    let (epsilon, u) = ht.eigh(UPLO::Lower).unwrap();
    let c = x.dot(&u);
    (epsilon, c)
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
        return (Array1::from_vec(vec![a[(0, 0)]]), Array2::from_elem((1, 1), Complex64::new(1.0, 0.0)));
    }

    let (vals, vecs) = a.eig().unwrap();
    let mut order: Vec<usize> = (0..vals.len()).collect();
    order.sort_by(|&i, &j| vals[i].re.partial_cmp(&vals[j].re).unwrap().then(vals[i].im.partial_cmp(&vals[j].im).unwrap()));

    let mut e = Array1::<Complex64>::zeros(vals.len());
    let mut u = Array2::<Complex64>::zeros(vecs.raw_dim());

    for (k, &i) in order.iter().enumerate() {
        e[k] = vals[i];

        let mut col = vecs.column(i).to_owned();
        let nrm = col.dot(&col).sqrt();
        if nrm.norm() > 1e-14 {
            col.mapv_inplace(|z| z / nrm);
        }
        u.column_mut(k).assign(&col);
    }

    (e, u)
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
pub fn complex_metric_orthonormalize(c: &Array2<Complex64>, s: &Array2<f64>) -> Array2<Complex64> {
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
pub fn parallel_matvec<T: StateScalar>(h: &Array2<T>, c: &Array1<T>) -> Array1<T> {
    let result: Vec<T> = h.axis_iter(Axis(0)).into_par_iter().map(|row| {
        let mut acc = T::from_real(0.0);
        for (&hij, &cj) in row.iter().zip(c.iter()) {
            acc += hij * cj;
        }
        acc
    }).collect();

    Array1::from_vec(result)
}
