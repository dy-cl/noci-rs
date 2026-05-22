// scf/h/distance.rs

use num_complex::Complex64;

use crate::maths::real2_as;
use crate::{HSCFState, SCFState};

/// Distance between an h-SCF candidate and a real seed using holomorphic densities.
/// # Arguments:
/// - `h`: Holomorphic candidate.
/// - `seed`: Real seed state.
/// # Returns:
/// - `f64`: Frobenius norm of the alpha/beta density difference.
pub(crate) fn h_density_distance_from_real(
    h: &HSCFState,
    seed: &SCFState,
) -> f64 {
    let da0 = real2_as::<Complex64>(&seed.da);
    let db0 = real2_as::<Complex64>(&seed.db);

    h.da.iter()
        .zip(da0.iter())
        .map(|(x, y)| (*x - *y).norm_sqr())
        .chain(
            h.db.iter()
                .zip(db0.iter())
                .map(|(x, y)| (*x - *y).norm_sqr()),
        )
        .sum::<f64>()
        .sqrt()
}
