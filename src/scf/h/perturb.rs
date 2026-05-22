// scf/h/perturb.rs

use ndarray::Array2;
use ndarray_linalg::Eig;
use num_complex::Complex64;

use crate::AoData;
use crate::scf::DensityMode;

use super::canonical::pseudo_canonicalise;
use super::step::finite_difference_hessian;
use super::tangent::{geodesic_step, unpack};
use super::types::{SecantPair, SpinBlock};
use crate::scf::{density, fock, orbital_gradient};

/// Build a complex recovery kick along the softest h-SCF Hessian mode.
/// # Arguments:
/// - `ca0`: Initial alpha-spin MO coefficients ordered as occupied then virtual.
/// - `cb0`: Initial beta-spin MO coefficients ordered as occupied then virtual.
/// - `ao`: AO data.
/// - `na`: Number of occupied alpha-spin orbitals.
/// - `nb`: Number of occupied beta-spin orbitals.
/// - `theta`: Imaginary rotation amplitude including the requested branch sign.
/// # Returns:
/// - `Option<(Array2<Complex64>, Array2<Complex64>)>`: Alpha and beta occupied-virtual rotations.
pub(crate) fn hessian_mode_perturbation(
    ca0: &Array2<Complex64>,
    cb0: &Array2<Complex64>,
    ao: &AoData,
    na: usize,
    nb: usize,
    theta: Complex64,
) -> Option<(Array2<Complex64>, Array2<Complex64>)> {
    if theta.norm() == 0.0 {
        return None;
    }

    let mut ca = ca0.clone();
    let mut cb = cb0.clone();
    let mut hist: Vec<SecantPair> = Vec::new();
    let mut extra: Vec<&mut Array2<Complex64>> = Vec::new();

    let da = density(&ca, na, DensityMode::Holomorphic);
    let db = density(&cb, nb, DensityMode::Holomorphic);
    let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
    pseudo_canonicalise(&mut ca, &fa, na, &mut hist, SpinBlock::Alpha, &mut extra);
    pseudo_canonicalise(&mut cb, &fb, nb, &mut hist, SpinBlock::Beta, &mut extra);

    let da = density(&ca, na, DensityMode::Holomorphic);
    let db = density(&cb, nb, DensityMode::Holomorphic);
    let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
    let ga = orbital_gradient(&ca, &fa, na, DensityMode::Holomorphic);
    let gb = orbital_gradient(&cb, &fb, nb, DensityMode::Holomorphic);

    let h = finite_difference_hessian(&ca, &cb, ao, na, nb, &ga, &gb);
    if h.nrows() == 0 {
        return None;
    }

    let (vals, vecs) = h.eig().ok()?;
    let mode = vals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())?
        .0;

    let mut v = vecs.column(mode).to_owned();
    let nrm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if nrm <= 1.0e-14 {
        return None;
    }
    v.mapv_inplace(|z| z / nrm);

    let phase = -theta;
    let p = v.mapv(|z| z * phase);
    Some(unpack(
        &p,
        (ga.nrows(), ga.ncols()),
        (gb.nrows(), gb.ncols()),
    ))
}

/// Apply an imaginary occupied-virtual perturbation to initialise a complex h-SCF branch.
/// # Arguments:
/// - `c`: MO coefficient matrix ordered as occupied then virtual.
/// - `nocc`: Number of occupied orbitals.
/// - `theta`: Complex rotation amplitude applied to corresponding occupied-virtual pairs.
/// # Returns:
/// - `Array2<Complex64>`: Kicked MO coefficient matrix.
pub(crate) fn perturb_ov(
    c: &Array2<Complex64>,
    nocc: usize,
    theta: Complex64,
) -> Array2<Complex64> {
    let n = c.ncols();
    let nvir = n - nocc;

    if nocc == 0 || nvir == 0 || theta.norm() == 0.0 {
        return c.clone();
    }

    let mut p = Array2::<Complex64>::zeros((nvir, nocc));
    p[(0, nocc - 1)] = theta;

    geodesic_step(c, &p, nocc, 1.0)
}
