// scf/h/seed.rs

use ndarray::{Array2, Axis};
use num_complex::Complex64;

use crate::maths::{complex_metric_orthonormalize, real2_as};
use crate::scf::spin_occupation;
use crate::{AoData, SCFState};

/// Build occupied-first complex orbitals from a real SCF seed.
/// # Arguments:
/// - `seed`: Real SCF state used as the orbital source.
/// - `ao`: AO data containing the overlap matrix.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Alpha and beta complex orbitals ordered occupied first.
pub(crate) fn complex_orbitals_from_real(
    seed: &SCFState,
    ao: &AoData,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let occ = spin_occupation(seed);
    let idx_a = occ.alpha_occupied_first();
    let idx_b = occ.beta_occupied_first();

    let ca = real2_as::<Complex64>(&seed.ca).select(Axis(1), &idx_a);
    let cb = real2_as::<Complex64>(&seed.cb).select(Axis(1), &idx_b);

    (
        complex_metric_orthonormalize(&ca, &ao.s),
        complex_metric_orthonormalize(&cb, &ao.s),
    )
}
