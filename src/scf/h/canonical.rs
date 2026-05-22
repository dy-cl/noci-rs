// scf/h/canonical.rs

use std::sync::Arc;

use ndarray::{Array1, Array2, Axis, s};
use num_complex::Complex64;

use crate::HSCFState;
use crate::maths::{adjoint, loewdin_x, real2_as, symmetric_evp_complex};
use crate::scf::spin_occupation;

use super::types::{SecantPair, SpinBlock};

/// Hermitian-orthonormalise one spin orbital basis while preserving occupied and virtual subspaces.
/// # Arguments:
/// - `c`: Spin MO coefficient matrix.
/// - `occ`: Occupied MO indices.
/// - `virt`: Virtual MO indices.
/// - `s`: AO overlap matrix.
/// - `tol`: Tolerance below which a metric eigenvalue is treated as singular.
/// # Returns:
/// - `Array2<Complex64>`: MO coefficients satisfying `C^\dagger S C = I` within the occupied
///   and virtual spaces and `C_o^\dagger S C_v = 0`.
fn hermitian_orthonormalise_spin_basis(
    c: &Array2<Complex64>,
    occ: &[usize],
    virt: &[usize],
    s: &Array2<f64>,
    tol: f64,
) -> Array2<Complex64> {
    let smat = real2_as::<Complex64>(s);
    let mut out = Array2::<Complex64>::zeros(c.raw_dim());

    let c_occ = c.select(Axis(1), occ);
    let c_occ = if occ.is_empty() {
        c_occ
    } else {
        let soo = adjoint(&c_occ).dot(&smat).dot(&c_occ);
        c_occ.dot(&loewdin_x(&soo, false, tol))
    };

    if !occ.is_empty() {
        for (k, &p) in occ.iter().enumerate() {
            out.column_mut(p).assign(&c_occ.column(k));
        }
    }

    let c_vir = c.select(Axis(1), virt);
    let c_vir = if virt.is_empty() {
        c_vir
    } else {
        let c_vir_perp = if occ.is_empty() {
            c_vir
        } else {
            let sov = adjoint(&c_occ).dot(&smat).dot(&c_vir);
            c_vir - c_occ.dot(&sov)
        };

        let svv = adjoint(&c_vir_perp).dot(&smat).dot(&c_vir_perp);
        c_vir_perp.dot(&loewdin_x(&svv, false, tol))
    };

    if !virt.is_empty() {
        for (k, &p) in virt.iter().enumerate() {
            out.column_mut(p).assign(&c_vir.column(k));
        }
    }

    out
}

/// Convert one h-SCF state to the ordinary Hermitian orbital convention required by NOCI/Wick.
/// h-SCF orbitals are optimised with the holomorphic constraint `C^T S C = I`, while NOCI matrix
/// elements and Wick's theorem use the ordinary Hermitian bra-ket metric. This constructs a
/// post-SCF orbital representation satisfying `C^\dagger S C = I` without changing the occupied
/// or virtual subspaces.
/// # Arguments:
/// - `st`: h-SCF determinant state to normalise in place.
/// - `s`: AO overlap matrix.
/// # Returns:
/// - `()`: Updates the stored alpha and beta orbital coefficients in place.
pub fn normalise_hermitian(
    st: &mut HSCFState,
    s: &Array2<f64>,
) {
    let occ = spin_occupation(st);
    let tol = 1.0e-12;

    let ca = hermitian_orthonormalise_spin_basis(&st.ca, &occ.occ_alpha, &occ.virt_alpha, s, tol);
    let cb = hermitian_orthonormalise_spin_basis(&st.cb, &occ.occ_beta, &occ.virt_beta, s, tol);

    st.ca = Arc::new(ca);
    st.cb = Arc::new(cb);
}

/// Pseudo-canonicalise occupied and virtual spaces for one spin block.
/// # Arguments:
/// - `c`: MO coefficient matrix ordered as occupied then virtual.
/// - `f`: Spin Fock matrix.
/// - `nocc`: Number of occupied orbitals.
/// - `hist`: Stored SR1 secant pairs transformed into the new tangent basis.
/// - `spin`: Spin block being transformed.
/// - `extra`: Additional tangent matrices transformed into the new tangent basis.
/// # Returns:
/// - `Array1<Complex64>`: Occupied followed by virtual pseudo-canonical orbital energies.
pub(crate) fn pseudo_canonicalise(
    c: &mut Array2<Complex64>,
    f: &Array2<Complex64>,
    nocc: usize,
    hist: &mut [SecantPair],
    spin: SpinBlock,
    extra: &mut [&mut Array2<Complex64>],
) -> Array1<Complex64> {
    let n = c.ncols();

    // Transform Fock matrix into MO basis such that we have o-o, o-v, v-o, v-v blocks.
    let fmo = c.t().dot(f).dot(c);

    // Diagonalise o-o and v-v blocks.
    let (eo, uo) = symmetric_evp_complex(&fmo.slice(s![0..nocc, 0..nocc]).to_owned());
    let (ev, uv) = symmetric_evp_complex(&fmo.slice(s![nocc..n, nocc..n]).to_owned());

    // Rotate orbitals within occupied and virtual spaces as C_o = C_o U_o, C_v = C_v U_v.
    let cocc = c.slice(s![.., 0..nocc]).to_owned().dot(&uo);
    let cvir = c.slice(s![.., nocc..n]).to_owned().dot(&uv);
    c.slice_mut(s![.., 0..nocc]).assign(&cocc);
    c.slice_mut(s![.., nocc..n]).assign(&cvir);

    // Transform the stored SR1 secant-pair history into the new pseudo-canonical tangent basis.
    for pair in hist.iter_mut() {
        match spin {
            SpinBlock::Alpha => {
                // Transform the previous alpha-spin step as s_ai -> (U_v^T s U_o)_ai.
                pair.sa = uv.t().dot(&pair.sa).dot(&uo);

                // Transform the previous alpha-spin gradient change as y_ai -> (U_v^T y U_o)_ai.
                pair.ya = uv.t().dot(&pair.ya).dot(&uo);
            }
            SpinBlock::Beta => {
                // Transform the previous beta-spin step as s_ai -> (U_v^T s U_o)_ai.
                pair.sb = uv.t().dot(&pair.sb).dot(&uo);

                // Transform the previous beta-spin gradient change as y_ai -> (U_v^T y U_o)_ai.
                pair.yb = uv.t().dot(&pair.yb).dot(&uo);
            }
        }
    }

    // Transform any additional tangent-space matrices into the new pseudo-canonical basis.
    for x in extra.iter_mut() {
        // Transform the extra occupied-virtual block as X_ai -> (U_v^T X U_o)_ai.
        **x = uv.t().dot(&**x).dot(&uo);
    }

    let mut eps = Array1::<Complex64>::zeros(n);
    eps.slice_mut(s![0..nocc]).assign(&eo);
    eps.slice_mut(s![nocc..n]).assign(&ev);
    eps
}
