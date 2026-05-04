// scf/occupation.rs

use ndarray::{Array1, Array2};

/// Calculate MO occupancies as the diagonal of a density matrix in the MO basis.
/// `T = C^T S D S C`.
/// # Arguments
/// - `c`: Spin MO coefficient matrix.
/// - `d`: Spin density matrix.
/// - `s`: AO overlap matrix.
/// # Returns
/// - `Array1<f64>`: MO occupancies thresholded to zero or one.
pub(crate) fn mo_occupancies(c: &Array2<f64>, d: &Array2<f64>, s: &Array2<f64>) -> Array1<f64> {
    let t = c.t().dot(s).dot(d).dot(s).dot(c);
    let diag = t.diag().to_owned();
    diag.mapv(|x| if x > 0.5 {1.0} else {0.0})
}

/// Convert occupied MO indices to a bitstring.
/// # Arguments
/// - `idx`: Occupied MO indices.
/// # Returns
/// - `u128`: Occupation bitstring.
pub(crate) fn occvec_to_bits(idx: &[usize]) -> u128 {
    let mut bits = 0u128;
    for &i in idx {bits |= 1u128 << i;}
    bits
}
