// nonorthogonalwicks/eval/rdm2diff.rs

use ndarray::{Array2, Array4};

use super::super::scratch::WickScratch;
use super::super::view::WicksPairView;
use super::helpers::{
    construct_determinant_indices_gen, det_slice, extend_rdm_d, for_each_m_combination,
};
use crate::Excitation;
use crate::maths::{build_d, mix_columns};
use crate::noci::NOCIScalar;
use crate::time_call;

/// Calculate the different-spin contribution to the spin-free two-body RDM matrix element:
/// {}^{\Lambda\Gamma}D_{\alpha\beta}^{pq}{}_{rs}
/// + {}^{\Lambda\Gamma}D_{\beta\alpha}^{pq}{}_{rs}.
///   This evaluates the two products
///   \langle a^\dagger_{p\alpha} a_{r\alpha} \rangle
///   \langle a^\dagger_{q\beta} a_{s\beta} \rangle
///   and
///   \langle a^\dagger_{p\beta} a_{r\beta} \rangle
///   \langle a^\dagger_{q\alpha} a_{s\alpha} \rangle.
/// # Arguments:
/// - `w`: Same-spin and different-spin Wick's reference pair intermediates.
/// - `l_ex`: Excitation for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Excitation for |{}^\Gamma \Psi\rangle.
/// - `ca`: Left and right alpha orbital coefficients in the current physical basis.
/// - `cb`: Left and right beta orbital coefficients in the current physical basis.
/// - `scratch`: Different-spin scratch space and prepared same-spin alpha/beta scratch spaces.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `Array4<T>`: Different-spin contribution to the spin-free two-body RDM.
#[inline(always)]
pub(crate) fn lg_rdm2_diff<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    l_ex: &Excitation,
    g_ex: &Excitation,
    ca: (&Array2<T>, &Array2<T>),
    cb: (&Array2<T>, &Array2<T>),
    scratch: (&mut WickScratch<T>, &WickScratch<T>, &WickScratch<T>),
    tol: f64,
) -> Array4<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_rdm2_diff, {
        let (diff, a, b) = scratch;
        let _ = diff;
        let _ = a;
        let _ = b;

        let (l_ca, g_ca) = ca;
        let (l_cb, g_cb) = cb;
        let n = l_ca.nrows();
        let zero = <T as From<f64>>::from(0.0);
        let pref = w.aa.phase
            * <T as From<f64>>::from(w.aa.tilde_s_prod)
            * w.bb.phase
            * <T as From<f64>>::from(w.bb.tilde_s_prod);

        let mut out = Array4::<T>::zeros((n, n, n, n));

        let mut rows_a_base = Vec::new();
        let mut cols_a_base = Vec::new();
        let mut rows_b_base = Vec::new();
        let mut cols_b_base = Vec::new();

        construct_determinant_indices_gen(
            &l_ex.alpha,
            &g_ex.alpha,
            w.aa.nmo,
            &mut rows_a_base,
            &mut cols_a_base,
        );

        construct_determinant_indices_gen(
            &l_ex.beta,
            &g_ex.beta,
            w.bb.nmo,
            &mut rows_b_base,
            &mut cols_b_base,
        );

        let la = rows_a_base.len();
        let lb = rows_b_base.len();
        let dima = la + 1;
        let dimb = lb + 1;

        let xa0 = w.aa.x(0);
        let ya0 = w.aa.y(0);
        let xa1 = w.aa.x(1);
        let ya1 = w.aa.y(1);
        let xb0 = w.bb.x(0);
        let yb0 = w.bb.y(0);
        let xb1 = w.bb.x(1);
        let yb1 = w.bb.y(1);
        let xa0p = extend_rdm_d(&xa0, l_ca, g_ca, w.aa.nmo);
        let ya0p = extend_rdm_d(&ya0, l_ca, g_ca, w.aa.nmo);
        let xa1p = extend_rdm_d(&xa1, l_ca, g_ca, w.aa.nmo);
        let ya1p = extend_rdm_d(&ya1, l_ca, g_ca, w.aa.nmo);
        let xb0p = extend_rdm_d(&xb0, l_cb, g_cb, w.bb.nmo);
        let yb0p = extend_rdm_d(&yb0, l_cb, g_cb, w.bb.nmo);
        let xb1p = extend_rdm_d(&xb1, l_cb, g_cb, w.bb.nmo);
        let yb1p = extend_rdm_d(&yb1, l_cb, g_cb, w.bb.nmo);
        let xa0p = xa0p.view();
        let ya0p = ya0p.view();
        let xa1p = xa1p.view();
        let ya1p = ya1p.view();
        let xb0p = xb0p.view();
        let yb0p = yb0p.view();
        let xb1p = xb1p.view();
        let yb1p = yb1p.view();

        let mut rows_a = Vec::with_capacity(dima);
        let mut cols_a = Vec::with_capacity(dima);
        let mut rows_b = Vec::with_capacity(dimb);
        let mut cols_b = Vec::with_capacity(dimb);

        let mut deta0 = vec![zero; dima * dima];
        let mut deta1 = vec![zero; dima * dima];
        let mut detam = vec![zero; dima * dima];
        let mut detb0 = vec![zero; dimb * dimb];
        let mut detb1 = vec![zero; dimb * dimb];
        let mut detbm = vec![zero; dimb * dimb];

        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        rows_a.clear();
                        cols_a.clear();
                        rows_b.clear();
                        cols_b.clear();

                        rows_a.push(2 * w.aa.nmo + p);
                        rows_a.extend_from_slice(rows_a_base.as_slice());
                        cols_a.push(2 * w.aa.nmo + r);
                        cols_a.extend_from_slice(cols_a_base.as_slice());

                        rows_b.push(2 * w.bb.nmo + q);
                        rows_b.extend_from_slice(rows_b_base.as_slice());
                        cols_b.push(2 * w.bb.nmo + s);
                        cols_b.extend_from_slice(cols_b_base.as_slice());

                        build_d(
                            &mut deta0,
                            dima,
                            &xa0p,
                            &ya0p,
                            rows_a.as_slice(),
                            cols_a.as_slice(),
                        );
                        build_d(
                            &mut deta1,
                            dima,
                            &xa1p,
                            &ya1p,
                            rows_a.as_slice(),
                            cols_a.as_slice(),
                        );
                        build_d(
                            &mut detb0,
                            dimb,
                            &xb0p,
                            &yb0p,
                            rows_b.as_slice(),
                            cols_b.as_slice(),
                        );
                        build_d(
                            &mut detb1,
                            dimb,
                            &xb1p,
                            &yb1p,
                            rows_b.as_slice(),
                            cols_b.as_slice(),
                        );

                        let mut dab = zero;
                        for_each_m_combination(dima, w.aa.m, |bits_a| {
                            mix_columns(
                                detam.as_mut_slice(),
                                deta0.as_slice(),
                                deta1.as_slice(),
                                dima,
                                bits_a,
                            );

                            if let Some(da) = det_slice(detam.as_slice(), dima) {
                                if da.abs() <= tol {
                                    return;
                                }

                                for_each_m_combination(dimb, w.bb.m, |bits_b| {
                                    mix_columns(
                                        detbm.as_mut_slice(),
                                        detb0.as_slice(),
                                        detb1.as_slice(),
                                        dimb,
                                        bits_b,
                                    );

                                    if let Some(db) = det_slice(detbm.as_slice(), dimb)
                                        && db.abs() > tol
                                    {
                                        dab += da * db;
                                    }
                                });
                            }
                        });

                        rows_a.clear();
                        cols_a.clear();
                        rows_b.clear();
                        cols_b.clear();

                        rows_a.push(2 * w.aa.nmo + q);
                        rows_a.extend_from_slice(rows_a_base.as_slice());
                        cols_a.push(2 * w.aa.nmo + s);
                        cols_a.extend_from_slice(cols_a_base.as_slice());

                        rows_b.push(2 * w.bb.nmo + p);
                        rows_b.extend_from_slice(rows_b_base.as_slice());
                        cols_b.push(2 * w.bb.nmo + r);
                        cols_b.extend_from_slice(cols_b_base.as_slice());

                        build_d(
                            &mut deta0,
                            dima,
                            &xa0p,
                            &ya0p,
                            rows_a.as_slice(),
                            cols_a.as_slice(),
                        );
                        build_d(
                            &mut deta1,
                            dima,
                            &xa1p,
                            &ya1p,
                            rows_a.as_slice(),
                            cols_a.as_slice(),
                        );
                        build_d(
                            &mut detb0,
                            dimb,
                            &xb0p,
                            &yb0p,
                            rows_b.as_slice(),
                            cols_b.as_slice(),
                        );
                        build_d(
                            &mut detb1,
                            dimb,
                            &xb1p,
                            &yb1p,
                            rows_b.as_slice(),
                            cols_b.as_slice(),
                        );

                        let mut dba = zero;
                        for_each_m_combination(dima, w.aa.m, |bits_a| {
                            mix_columns(
                                detam.as_mut_slice(),
                                deta0.as_slice(),
                                deta1.as_slice(),
                                dima,
                                bits_a,
                            );

                            if let Some(da) = det_slice(detam.as_slice(), dima) {
                                if da.abs() <= tol {
                                    return;
                                }

                                for_each_m_combination(dimb, w.bb.m, |bits_b| {
                                    mix_columns(
                                        detbm.as_mut_slice(),
                                        detb0.as_slice(),
                                        detb1.as_slice(),
                                        dimb,
                                        bits_b,
                                    );

                                    if let Some(db) = det_slice(detbm.as_slice(), dimb)
                                        && db.abs() > tol
                                    {
                                        dba += da * db;
                                    }
                                });
                            }
                        });

                        out[(p, q, r, s)] = pref * (dab + dba);
                    }
                }
            }
        }

        out
    })
}
