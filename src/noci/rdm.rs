// noci/rdm.rs

use ndarray::{Array1, Array2, Array4};

use crate::nonorthogonalwicks::{
    WickScratchSpin, lg_overlap, lg_rdm1, lg_rdm2_diff, lg_rdm2_same, prepare_same,
};

use super::naive::{build_s_pair, occ_coeffs, pair_density};
use super::types::{DetPair, NOCIData, NOCIScalar};

/// Build spin-free one and two body RDMs for a NOCI reference state:
/// \Gamma^p_q = 1 / \mathcal N \sum_{xw} c_x^L c_w^R
///     \langle {}^x \Psi | \hat E^p_q | {}^w \Psi \rangle,
/// \Gamma^{pq}_{rs} = 1 / \mathcal N \sum_{xw} c_x^L c_w^R
///     \langle {}^x \Psi | \hat E^{pq}_{rs} | {}^w \Psi \rangle,
/// with \mathcal N = \sum_{xw} c_x^L c_w^R
///     \langle {}^x \Psi | {}^w \Psi \rangle.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `coeff_l`: Left NOCI coefficient vector.
/// - `coeff_r`: Right NOCI coefficient vector.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, Array2<T>, Array4<T>)`: Reference norm, spin-free 1-RDM and spin-free 2-RDM.
pub(crate) fn build_spin_free_rdms_12<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    coeff_l: &Array1<T>,
    coeff_r: &Array1<T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, Array2<T>, Array4<T>) {
    assert_eq!(coeff_l.len(), data.basis.len());
    assert_eq!(coeff_r.len(), data.basis.len());

    let norb = data.ao.h.nrows();
    let mut norm = <T as From<f64>>::from(0.0);
    let mut gamma1 = Array2::<T>::zeros((norb, norb));
    let mut gamma2 = Array4::<T>::zeros((norb, norb, norb, norb));
    let mut scratch = scratch;
    let mut td = 0.0;

    for x in 0..data.basis.len() {
        for w in 0..data.basis.len() {
            let pair = DetPair::new(&data.basis[x], &data.basis[w]);
            let weight = coeff_l[x] * coeff_r[w];

            let (sxw, g1xw, g2xw) = if data.input.wicks.compare {
                let ((sxw, g1xw, g2xw), d) = compare_spin_free_rdm12_pair_wicks_naive(
                    data,
                    pair,
                    scratch.as_deref_mut().unwrap(),
                );

                td += d;
                (sxw, g1xw, g2xw)
            } else {
                calculate_spin_free_rdm12_pair(data, pair, scratch.as_deref_mut())
            };

            norm += weight * sxw;
            gamma1.scaled_add(weight, &g1xw);
            gamma2.scaled_add(weight, &g2xw);
        }
    }

    if data.input.wicks.compare {
        println!("Total naive–wicks discrepancy (spin-free RDMs): {:.6e}", td);
    }

    gamma1.mapv_inplace(|x| x / norm);
    gamma2.mapv_inplace(|x| x / norm);

    (norm, gamma1, gamma2)
}

/// Calculate spin-free one and two body RDM matrix elements for a determinant pair:
/// {}^{xw}\Gamma^p_q = \langle {}^x \Psi | \hat E^p_q | {}^w \Psi \rangle,
/// {}^{xw}\Gamma^{pq}_{rs} = \langle {}^x \Psi | \hat E^{pq}_{rs} | {}^w \Psi \rangle,
/// where \hat E is the spin-summed excitation operator.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose transition RDMs are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, Array2<T>, Array4<T>)`: Pair overlap, spin-free 1-RDM and spin-free 2-RDM.
pub(crate) fn calculate_spin_free_rdm12_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, Array2<T>, Array4<T>) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    if ldet.parent != gdet.parent && data.input.wicks.enabled {
        calculate_spin_free_rdm12_pair_wicks(
            data,
            pair,
            scratch.expect("Wick scratch required for spin-free RDM evaluation"),
        )
    } else {
        calculate_spin_free_rdm12_pair_naive(data, pair)
    }
}

/// Compare naive and Wick's calculation of spin-free RDM matrix elements.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be compared.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `((T, Array2<T>, Array4<T>), f64)`: Pair overlap, spin-free 1-RDM and spin-free
///   2-RDM from Wick's path, and the total discrepancy from the naive path.
pub(in crate::noci) fn compare_spin_free_rdm12_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> ((T, Array2<T>, Array4<T>), f64) {
    let (sn, g1n, g2n) = calculate_spin_free_rdm12_pair_naive(data, pair);
    let (sw, g1w, g2w) = calculate_spin_free_rdm12_pair_wicks(data, pair, scratch);

    let mut diff = (sn - sw).abs();

    for (n, w) in g1n.iter().zip(g1w.iter()) {
        diff += (*n - *w).abs();
    }

    for (n, w) in g2n.iter().zip(g2w.iter()) {
        diff += (*n - *w).abs();
    }

    ((sw, g1w, g2w), diff)
}

/// Calculate spin-free one and two body RDM matrix elements using generalised
/// Slater-Condon rules:
/// {}^{xw}S = d_{xw} {}^{xw}S_\alpha {}^{xw}S_\beta,
/// {}^{xw}\Gamma^p_q = d_{xw} (
///     {}^{xw}S_\beta {}^{xw}D_\alpha{}^p_q
///   + {}^{xw}S_\alpha {}^{xw}D_\beta{}^p_q),
///     {}^{xw}\Gamma^{pq}_{rs} = d_{xw} (
///     {}^{xw}S_\beta {}^{xw}D_{\alpha\alpha}^{pq}{}_{rs}
///   + {}^{xw}S_\alpha {}^{xw}D_{\beta\beta}^{pq}{}_{rs}
///   + {}^{xw}D_\alpha{}^p_r {}^{xw}D_\beta{}^q_s
///   + {}^{xw}D_\beta{}^p_r {}^{xw}D_\alpha{}^q_s).
///     Here d_{xw} is the external determinant excitation phase and
///     D_\alpha and D_\beta are the overlap-weighted spin-block densities.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// # Returns:
/// - `(T, Array2<T>, Array4<T>)`: Pair overlap, spin-free 1-RDM and spin-free 2-RDM.
fn calculate_spin_free_rdm12_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
) -> (T, Array2<T>, Array4<T>) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;
    let norb = data.ao.h.nrows();

    let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
    let half = <T as From<f64>>::from(0.5);
    let sxw = det_phase * pa.s * pb.s;

    let da = pair_density(&pa, norb);
    let db = pair_density(&pb, norb);

    let mut gamma1 = Array2::<T>::zeros((norb, norb));
    gamma1.scaled_add(det_phase * pb.s, &da);
    gamma1.scaled_add(det_phase * pa.s, &db);

    let mut gamma2 = Array4::<T>::zeros((norb, norb, norb, norb));

    for (spin_pair, other_s) in [(&pa, pb.s), (&pb, pa.s)] {
        let fac = det_phase * other_s * spin_pair.phase * <T as From<f64>>::from(spin_pair.s_red);

        let (a, b) = match spin_pair.zeros.len() {
            0 => (spin_pair.w.as_ref().unwrap(), spin_pair.w.as_ref().unwrap()),
            1 => (
                spin_pair.p_i.as_ref().unwrap(),
                spin_pair.w.as_ref().unwrap(),
            ),
            2 => (
                spin_pair.p_i.as_ref().unwrap(),
                spin_pair.p_j.as_ref().unwrap(),
            ),
            _ => continue,
        };

        for p in 0..norb {
            for q in 0..norb {
                for r in 0..norb {
                    for s in 0..norb {
                        gamma2[(p, q, r, s)] += half
                            * fac
                            * (a[(p, r)] * b[(q, s)] - a[(p, s)] * b[(q, r)]
                                + b[(p, r)] * a[(q, s)]
                                - b[(p, s)] * a[(q, r)]);
                    }
                }
            }
        }
    }

    for p in 0..norb {
        for q in 0..norb {
            for r in 0..norb {
                for s in 0..norb {
                    gamma2[(p, q, r, s)] +=
                        det_phase * (da[(p, r)] * db[(q, s)] + db[(p, r)] * da[(q, s)]);
                }
            }
        }
    }

    (sxw, gamma1, gamma2)
}

/// Calculate spin-free one and two body RDM matrix elements using extended
/// non-orthogonal Wick's theorem:
/// {}^{xw}\Gamma^p_q = d_\alpha d_\beta (
///     {}^{xw}S_\beta {}^{xw}D_\alpha{}^p_q
///   + {}^{xw}S_\alpha {}^{xw}D_\beta{}^p_q),
///     {}^{xw}\Gamma^{pq}_{rs} = d_\alpha d_\beta (
///     {}^{xw}S_\beta {}^{xw}D_{\alpha\alpha}^{pq}{}_{rs}
///   + {}^{xw}S_\alpha {}^{xw}D_{\beta\beta}^{pq}{}_{rs}
///   + {}^{xw}D_{\alpha\beta}^{pq}{}_{rs}
///   + {}^{xw}D_{\beta\alpha}^{pq}{}_{rs}).
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, Array2<T>, Array4<T>)`: Pair overlap, spin-free 1-RDM and spin-free 2-RDM.
fn calculate_spin_free_rdm12_pair_wicks<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> (T, Array2<T>, Array4<T>) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;
    let norb = data.ao.h.nrows();

    let wicks = data.wicks.unwrap();
    let w = wicks.pair(ldet.parent, gdet.parent);

    let ex_la = &ldet.excitation.alpha;
    let ex_ga = &gdet.excitation.alpha;
    let ex_lb = &ldet.excitation.beta;
    let ex_gb = &gdet.excitation.beta;

    let la = ex_la.holes.len() + ex_ga.holes.len();
    let lb = ex_lb.holes.len() + ex_gb.holes.len();

    let dosa = w.aa.m <= la;
    let dosb = w.bb.m <= lb;
    let do1a = w.aa.m <= la + 1;
    let do1b = w.bb.m <= lb + 1;
    let do2aa = w.aa.m <= la + 2;
    let do2bb = w.bb.m <= lb + 2;
    let do2ab = w.aa.m <= la + 1 && w.bb.m <= lb + 1;

    let pha = <T as From<f64>>::from(ldet.pha * gdet.pha);
    let phb = <T as From<f64>>::from(ldet.phb * gdet.phb);
    let det_phase = pha * phb;

    let mut sa = <T as From<f64>>::from(0.0);
    let mut sb = <T as From<f64>>::from(0.0);

    if dosa || do1a || do2aa || do2ab {
        prepare_same(&w.aa, ex_la, ex_ga, &mut scratch.aa);

        if dosa {
            sa = lg_overlap(&w.aa, ex_la, ex_ga, &mut scratch.aa);
        }
    }

    if dosb || do1b || do2bb || do2ab {
        prepare_same(&w.bb, ex_lb, ex_gb, &mut scratch.bb);

        if dosb {
            sb = lg_overlap(&w.bb, ex_lb, ex_gb, &mut scratch.bb);
        }
    }

    let sxw = det_phase * sa * sb;
    let mut gamma1 = Array2::<T>::zeros((norb, norb));
    let mut gamma2 = Array4::<T>::zeros((norb, norb, norb, norb));

    let g1a = if do1a {
        let x = lg_rdm1(
            &w.aa,
            ex_la,
            ex_ga,
            ldet.ca.as_ref(),
            gdet.ca.as_ref(),
            &mut scratch.aa,
            data.tol,
        );
        Some(x)
    } else {
        None
    };
    let g1b = if do1b {
        let x = lg_rdm1(
            &w.bb,
            ex_lb,
            ex_gb,
            ldet.cb.as_ref(),
            gdet.cb.as_ref(),
            &mut scratch.bb,
            data.tol,
        );
        Some(x)
    } else {
        None
    };

    if sb.abs() > data.tol
        && let Some(g1a) = g1a.as_ref()
    {
        gamma1.scaled_add(det_phase * sb, g1a);
    }

    if sa.abs() > data.tol
        && let Some(g1b) = g1b.as_ref()
    {
        gamma1.scaled_add(det_phase * sa, g1b);
    }

    if sb.abs() > data.tol && do2aa {
        if sa.abs() > data.tol
            && let Some(g1a) = g1a.as_ref()
        {
            let scale = det_phase * sb / sa;

            for p in 0..norb {
                for q in 0..norb {
                    for r in 0..norb {
                        for s in 0..norb {
                            gamma2[(p, q, r, s)] +=
                                scale * (g1a[(p, r)] * g1a[(q, s)] - g1a[(p, s)] * g1a[(q, r)]);
                        }
                    }
                }
            }
        } else {
            let g2aa = lg_rdm2_same(
                &w.aa,
                ex_la,
                ex_ga,
                ldet.ca.as_ref(),
                gdet.ca.as_ref(),
                &mut scratch.aa,
                data.tol,
            );
            gamma2.scaled_add(det_phase * sb, &g2aa);
        }
    }

    if sa.abs() > data.tol && do2bb {
        if sb.abs() > data.tol
            && let Some(g1b) = g1b.as_ref()
        {
            let scale = det_phase * sa / sb;

            for p in 0..norb {
                for q in 0..norb {
                    for r in 0..norb {
                        for s in 0..norb {
                            gamma2[(p, q, r, s)] +=
                                scale * (g1b[(p, r)] * g1b[(q, s)] - g1b[(p, s)] * g1b[(q, r)]);
                        }
                    }
                }
            }
        } else {
            let g2bb = lg_rdm2_same(
                &w.bb,
                ex_lb,
                ex_gb,
                ldet.cb.as_ref(),
                gdet.cb.as_ref(),
                &mut scratch.bb,
                data.tol,
            );
            gamma2.scaled_add(det_phase * sa, &g2bb);
        }
    }

    if do2ab {
        let g2ab = lg_rdm2_diff(
            &w,
            &ldet.excitation,
            &gdet.excitation,
            (ldet.ca.as_ref(), gdet.ca.as_ref()),
            (ldet.cb.as_ref(), gdet.cb.as_ref()),
            (&mut scratch.diff, &scratch.aa, &scratch.bb),
            data.tol,
        );
        gamma2.scaled_add(det_phase, &g2ab);
    }

    (sxw, gamma1, gamma2)
}
