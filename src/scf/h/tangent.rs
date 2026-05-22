// scf/h/tangent.rs

use ndarray::{Array1, Array2, s};
use num_complex::Complex64;

use crate::maths::matrix_exp_complex;

/// Apply a complex-orthogonal occupied-virtual geodesic step.
/// # Arguments:
/// - `c`: Current MO coefficient matrix ordered as occupied then virtual.
/// - `p`: Occupied-virtual step block with shape `(nvir, nocc)`.
/// - `nocc`: Number of occupied orbitals.
/// - `alpha`: Step length.
/// # Returns:
/// - `Array2<Complex64>`: Updated MO coefficient matrix.
pub(crate) fn geodesic_step(
    c: &Array2<Complex64>,
    p: &Array2<Complex64>,
    nocc: usize,
    alpha: f64,
) -> Array2<Complex64> {
    let n = c.ncols();
    let nvir = n - nocc;

    let mut k = Array2::<Complex64>::zeros((n, n));

    // Construct the \exp(0 -\alpha_k p_k^T \\ \alpha_k p_k) matrix exponential.
    for a in 0..nvir {
        for i in 0..nocc {
            let z = p[(a, i)] * alpha;
            k[(nocc + a, i)] = z;
            k[(i, nocc + a)] = -z;
        }
    }

    c.dot(&matrix_exp_complex(&k))
}

/// Pack alpha and beta tangent blocks into one vector.
/// # Arguments:
/// - `a`: Alpha-spin tangent block.
/// - `b`: Beta-spin tangent block.
/// # Returns:
/// - `Array1<Complex64>`: Concatenated vector.
pub(crate) fn pack(
    a: &Array2<Complex64>,
    b: &Array2<Complex64>,
) -> Array1<Complex64> {
    Array1::from_iter(a.iter().chain(b.iter()).copied())
}

/// Unpack one vector into alpha and beta tangent blocks.
/// # Arguments:
/// - `x`: Packed tangent vector.
/// - `adim`: Alpha-spin block dimensions.
/// - `bdim`: Beta-spin block dimensions.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Alpha- and beta-spin tangent blocks.
pub(crate) fn unpack(
    x: &Array1<Complex64>,
    adim: (usize, usize),
    bdim: (usize, usize),
) -> (Array2<Complex64>, Array2<Complex64>) {
    let na = adim.0 * adim.1;
    let mut a = Array2::<Complex64>::zeros(adim);
    let mut b = Array2::<Complex64>::zeros(bdim);

    for (dst, src) in a.iter_mut().zip(x.slice(s![0..na]).iter()) {
        *dst = *src;
    }
    for (dst, src) in b.iter_mut().zip(x.slice(s![na..]).iter()) {
        *dst = *src;
    }

    (a, b)
}
