// scf/h/seed.rs

use ndarray::{Array2, Axis};
use num_complex::Complex64;

use crate::input::StateRecipe;
use crate::maths::{complex_metric_orthonormalize, real2_as};
use crate::scf::spin_occupation;
use crate::{AoData, SCFState};

use super::perturb::{hessian_mode_perturbation, perturb_ov};
use super::tangent::geodesic_step;

/// Build initial h-SCF orbitals from a real SCF seed and state recipe.
/// # Arguments:
/// - `seed`: Real SCF seed state.
/// - `recipe`: State construction recipe.
/// - `ao`: Contains electron counts and AO metadata.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Alpha and beta h-SCF initial orbitals.
pub fn h_seed_orbitals(
    seed: &SCFState,
    recipe: &StateRecipe,
    ao: &AoData,
) -> (Array2<Complex64>, Array2<Complex64>) {
    h_seed_orbitals_with_scale(seed, recipe, ao, 0.05)
}

/// Build initial h-SCF orbitals from a real SCF seed and state recipe with a recovery scale.
/// # Arguments:
/// - `seed`: Real SCF seed state.
/// - `recipe`: State construction recipe.
/// - `ao`: Contains electron counts and AO metadata.
/// - `scale`: Minimum imaginary rotation amplitude.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Alpha and beta h-SCF initial orbitals.
pub(crate) fn h_seed_orbitals_with_scale(
    seed: &SCFState,
    recipe: &StateRecipe,
    ao: &AoData,
    scale: f64,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();

    let (mut ca, mut cb) = complex_orbitals_from_real(seed, ao);

    if let Some(sb) = &recipe.spin_bias {
        let sgn = sb.pattern.iter().copied().find(|&x| x != 0).unwrap_or(1) as f64;
        let theta = Complex64::new(0.0, sgn * scale);
        if let Some((pa, pb)) = hessian_mode_perturbation(&ca, &cb, ao, na, nb, theta) {
            ca = geodesic_step(&ca, &pa, na, 1.0);
            cb = geodesic_step(&cb, &pb, nb, 1.0);
        } else {
            ca = perturb_ov(&ca, na, theta);
            cb = perturb_ov(&cb, nb, -theta);
        }
    }

    if let Some(spb) = &recipe.spatial_bias {
        let sgn = spb.pattern.iter().copied().find(|&x| x != 0).unwrap_or(1) as f64;
        let theta = Complex64::new(0.0, sgn * scale);
        if let Some((pa, pb)) = hessian_mode_perturbation(&ca, &cb, ao, na, nb, theta) {
            ca = geodesic_step(&ca, &pa, na, 1.0);
            cb = geodesic_step(&cb, &pb, nb, 1.0);
        } else {
            ca = perturb_ov(&ca, na, theta);
            cb = perturb_ov(&cb, nb, theta);
        }
    }

    (ca, cb)
}

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
