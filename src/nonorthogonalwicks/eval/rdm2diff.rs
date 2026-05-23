// nonorthogonalwicks/eval/rdm2diff.rs

use ndarray::Array4;

use super::super::scratch::WickScratch;
use super::super::view::WicksPairView;
use super::helpers::{construct_determinant_indices_gen, det_slice, for_each_m_combination};
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
/// - `diff`: Different-spin scratch space.
/// - `a`: Prepared same-spin alpha scratch space.
/// - `b`: Prepared same-spin beta scratch space.
/// - `tol`: Tolerance for singularity handling in determinant evaluation.
/// # Returns
/// - `Array4<T>`: Different-spin contribution to the spin-free two-body RDM.
#[inline(always)]
pub(crate) fn lg_rdm2_diff<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    l_ex: &Excitation,
    g_ex: &Excitation,
    diff: &mut WickScratch<T>,
    a: &WickScratch<T>,
    b: &WickScratch<T>,
    tol: f64,
) -> Array4<T> {
    time_call!(crate::timers::nonorthogonalwicks::add_lg_rdm2_diff, {
        let _ = diff;
        let _ = a;
        let _ = b;

        let n = w.aa.nmo;
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

                        rows_a.push(p);
                        rows_a.extend_from_slice(rows_a_base.as_slice());
                        cols_a.push(w.aa.nmo + r);
                        cols_a.extend_from_slice(cols_a_base.as_slice());

                        rows_b.push(q);
                        rows_b.extend_from_slice(rows_b_base.as_slice());
                        cols_b.push(w.bb.nmo + s);
                        cols_b.extend_from_slice(cols_b_base.as_slice());

                        build_d(
                            &mut deta0,
                            dima,
                            &xa0,
                            &ya0,
                            rows_a.as_slice(),
                            cols_a.as_slice(),
                        );
                        build_d(
                            &mut deta1,
                            dima,
                            &xa1,
                            &ya1,
                            rows_a.as_slice(),
                            cols_a.as_slice(),
                        );
                        build_d(
                            &mut detb0,
                            dimb,
                            &xb0,
                            &yb0,
                            rows_b.as_slice(),
                            cols_b.as_slice(),
                        );
                        build_d(
                            &mut detb1,
                            dimb,
                            &xb1,
                            &yb1,
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

                        rows_a.push(q);
                        rows_a.extend_from_slice(rows_a_base.as_slice());
                        cols_a.push(w.aa.nmo + s);
                        cols_a.extend_from_slice(cols_a_base.as_slice());

                        rows_b.push(p);
                        rows_b.extend_from_slice(rows_b_base.as_slice());
                        cols_b.push(w.bb.nmo + r);
                        cols_b.extend_from_slice(cols_b_base.as_slice());

                        build_d(
                            &mut deta0,
                            dima,
                            &xa0,
                            &ya0,
                            rows_a.as_slice(),
                            cols_a.as_slice(),
                        );
                        build_d(
                            &mut deta1,
                            dima,
                            &xa1,
                            &ya1,
                            rows_a.as_slice(),
                            cols_a.as_slice(),
                        );
                        build_d(
                            &mut detb0,
                            dimb,
                            &xb0,
                            &yb0,
                            rows_b.as_slice(),
                            cols_b.as_slice(),
                        );
                        build_d(
                            &mut detb1,
                            dimb,
                            &xb1,
                            &yb1,
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
