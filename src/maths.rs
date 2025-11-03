// maths.rs
use ndarray::{Array1, Array2, Array4};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex64;

/// Loewdin symmetric orthogonalizer, computes X = S^{-1/2} for real/complex matrices.
/// # Arguments 
/// `s`: Array2, Hermitian matrix, uses only the lower triangle. Often AO basis overlap 
/// matrix is s here.
/// `project`: Bool, whether or not to project to non-zero positive subspace of S.
/// `tol`: Float, tolerance for whether a number is considered zero.
pub fn loewdin_x_real(s: &Array2<f64>, project: bool, tol: f64) -> Array2<f64> {
    // S = U \Lambda U^T
    let (lambdas, evecs) = s.eigh(UPLO::Lower).unwrap();
    // \Lambda^{-1/2}
    let invsqrt: Array1<f64> = lambdas.mapv(|x| {
        if project {
            if x > tol {1.0 / x.sqrt()} else {0.0}
        } else {
            1.0 / x.sqrt()
        }
    });
    let d = Array2::from_diag(&invsqrt);
    // X = U \Lambda^{-1/2} U^T
    evecs.dot(&d).dot(&evecs.t())
}

pub fn loewdin_x_complex(s: &Array2<Complex64>, project: bool, tol: f64) -> Array2<Complex64> {
    // S = U \Lambda U^{\dagger}
    let (lambdas, u) = s.eigh(UPLO::Lower).unwrap(); 
    // \Lambda^{-1/2}
    let invsqrt: Array1<Complex64> = lambdas.mapv(|x| { 
        if project {
            if x > tol {Complex64::new(1.0 / x.sqrt(), 0.0)} else {Complex64::new(0.0, 0.0)}
        } else {
           Complex64::new(1.0 / x.sqrt(), 0.0) 
        }
    });
    let d = Array2::from_diag(&invsqrt);
    // X = U \Lambda^{-1/2} U^{\dagger}
    let u_dag = u.t().map(|z| z.conj());
    u.dot(&d).dot(&u_dag)
}

/// Solve the real/complex generalized eigenproblem F C = S C e using the Loewdin orthogonalizer.
/// # Arguments
///     `f`: Array2, Hermitian matrix, uses only the lower triangle. Often Fock matrix 
///     is f here.
///     `s`: Array2, Hermitian matrix, uses only the lower triangle. Often AO basis overlap 
///     matrix is s here.
///     `project`: Bool, whether or not to project to non-zero positive subspace of S.
///     `tol`: Float, tolerance for whether a number is considered zero.
pub fn general_evp_real(f: &Array2<f64>, s: &Array2<f64>, project: bool, tol: f64) 
                        -> (Array1<f64>, Array2<f64>) {
    // X = S^{-1/2}
    let x = loewdin_x_real(s, project, tol); 
    // \tilde{F} = X^T F X.
    let ft = x.t().dot(f).dot(&x);
    // \tilde{F} U = \epsilon U.
    let (epsilon, u) = ft.eigh(UPLO::Lower).unwrap();
    // C = X U. 
    let c = x.dot(&u);
    (epsilon, c)
}

pub fn general_evp_complex(f: &Array2<Complex64>, s: &Array2<Complex64>, project: bool, 
                           tol: f64) -> (Array1<f64>, Array2<Complex64>) {
    // X = S^{-1/2}
    let x = loewdin_x_complex(s, project, tol); 
    // \tilde{F} = X^{\dagger} F X.
    let xt = x.t().map(|z| z.conj());
    let ft = xt.dot(f).dot(&x);
    // \tilde{F} U = \epsilon U.
    let (epsilon, u) = ft.eigh(UPLO::Lower).unwrap();
    // C = X U. 
    let c = x.dot(&u);
    (epsilon, c)
}

/// Calculate Einstein summation of matrices `g` and `h` as \sum_{a,b} g_{b,a} h_{ab}. 
/// Assumes `g` and `h` are of identical shape.
/// # Arguments 
///     `g`: Matrix 1. 
///     `h`: Matrix 2.
pub fn einsum_ba_ab(g: &Array2<Complex64>, h: &Array2<f64>) -> Complex64 {
    let n = g.nrows();
    let mut acc = Complex64::new(0.0, 0.0);
    for a in 0..n {
        for b in 0..n {
            acc += g[(b, a)] * Complex64::new(h[(a, b)], 0.0);
        }
    }

    acc
}

/// Calculate Einstein summation of matrices `g` and `h` and 4D tensor `t` as 
/// \sum_{a,c}\sum_{b,d} g_{b,a} t_{a,c,b,d} h_{d,c}. Assumes `g`, `h` and `t` all 
/// have axes of equal length.
/// # Arguments
///     `g`: Matrix 1. 
///     `t`: 4D tensor.
///     `h`: Matrix 2.
pub fn einsum_ba_acbd_dc(g: &Array2<Complex64>, t: &Array4<f64>, h: &Array2<Complex64>)
                         -> Complex64 {
    let n = g.nrows();
    let mut acc = Complex64::new(0.0, 0.0);

    for a in 0..n {
        for c in 0..n {
            for b in 0..n {
                for d in 0..n {
                    acc += g[(b, a)] * Complex64::new(t[(a, c, b, d)], 0.0) * h[(d, c)];
                }
            }
        }
    }
    
    acc
}


