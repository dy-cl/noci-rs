// nocc/rdm/rdm2.rs

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::noci::{DetPair, NOCIData, NOCIScalar, build_s_pair, occ_coeffs, pair_density};
use crate::nonorthogonalwicks::{
    WickScratchSpin, prepare_same, xw_overlap, xw_rdmk_same_prepared_batched,
};

/// `Spin-free two-body RDM stored as \Gamma[p, q, r, s].`
pub(crate) struct RDM2<T> {
    pub n: usize,
    pub data: Vec<T>,
}

/// Build the spin-free two-body RDM for a NOCI reference state.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `coeff_l`: Left NOCI coefficient vector.
/// - `coeff_r`: Right NOCI coefficient vector.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM2<T>)`: Reference norm and spin-free two-body RDM.
pub(crate) fn rdm2<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    coeff_l: &Array1<T>,
    coeff_r: &Array1<T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, RDM2<T>) {
    let n = data.ao.h.nrows();
    let mut norm = <T as From<f64>>::from(0.0);
    let mut gamma = RDM2 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(4)],
    };
    let mut scratch = scratch;
    let mut td = 0.0;
    let mut md = 0.0;

    for x in 0..data.basis.len() {
        for w in 0..data.basis.len() {
            let pair = DetPair::new(&data.basis[x], &data.basis[w]);
            let weight = coeff_l[x] * coeff_r[w];

            let (sxw, gxw) = if data.input.wicks.enabled && data.input.wicks.compare {
                let ((sxw, gxw), (d, m)) =
                    compare_rdm2_pair_wicks_naive(data, pair, scratch.as_deref_mut().unwrap());

                td += d;
                md = f64::max(md, m);
                (sxw, gxw)
            } else {
                rdm2_pair(data, pair, scratch.as_deref_mut())
            };

            norm += weight * sxw;

            for i in 0..gamma.data.len() {
                gamma.data[i] += weight * gxw.data[i];
            }
        }
    }

    if data.input.wicks.enabled && data.input.wicks.compare {
        println!(
            "Total naive–wicks discrepancy (spin-free 2-RDM): {:.6e}; max element: {:.6e}",
            td, md
        );
    }

    for v in gamma.data.iter_mut() {
        *v /= norm;
    }

    (norm, gamma)
}

/// Calculate a spin-free two-body RDM matrix element block for a determinant pair.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose transition RDM is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM2<T>)`: Pair overlap and spin-free two-body RDM.
fn rdm2_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, RDM2<T>) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    if ldet.parent != gdet.parent && data.input.wicks.enabled {
        rdm2_pair_wicks(
            data,
            pair,
            scratch.expect("Wick scratch required for spin-free 2-RDM evaluation"),
        )
    } else {
        rdm2_pair_naive(data, pair)
    }
}

/// Compare naive and Wick's calculation of spin-free two-body RDM matrix elements.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be compared.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `((T, RDM2<T>), (f64, f64))`: Pair overlap and spin-free 2-RDM from Wick's
///   path, total discrepancy from the naive path, and max elementwise discrepancy.
fn compare_rdm2_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> ((T, RDM2<T>), (f64, f64)) {
    let (sn, g2n) = rdm2_pair_naive(data, pair);
    let (sw, g2w) = rdm2_pair_wicks(data, pair, scratch);

    let mut total = (sn - sw).abs();
    let mut max_element = 0.0;

    for (n, w) in g2n.data.iter().zip(g2w.data.iter()) {
        let diff = (*n - *w).abs();
        total += diff;
        max_element = f64::max(max_element, diff);
    }

    ((sw, g2w), (total, max_element))
}

/// Calculate spin-free two-body RDM matrix elements using generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// # Returns:
/// - `(T, RDM2<T>)`: Pair overlap and spin-free two-body RDM.
fn rdm2_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
) -> (T, RDM2<T>) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;
    let n = data.ao.h.nrows();

    let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
    let half = <T as From<f64>>::from(0.5);
    let sxw = det_phase * pa.s * pb.s;

    let da = pair_density(&pa, n);
    let db = pair_density(&pb, n);
    let mut gamma = RDM2 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(4)],
    };

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

        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        let i = (((p * n + q) * n + r) * n) + s;
                        gamma.data[i] += half
                            * fac
                            * (a[(p, r)] * b[(q, s)] - a[(p, s)] * b[(q, r)]
                                + b[(p, r)] * a[(q, s)]
                                - b[(p, s)] * a[(q, r)]);
                    }
                }
            }
        }
    }

    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let i = (((p * n + q) * n + r) * n) + s;
                    gamma.data[i] +=
                        det_phase * (da[(p, r)] * db[(q, s)] + db[(p, r)] * da[(q, s)]);
                }
            }
        }
    }

    (sxw, gamma)
}

/// Calculate spin-free two-body RDM matrix elements using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM2<T>)`: Pair overlap and spin-free two-body RDM.
fn rdm2_pair_wicks<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> (T, RDM2<T>) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;
    let n = data.ao.h.nrows();

    let wicks = data.wicks.unwrap();
    let w = wicks.pair(ldet.parent, gdet.parent);

    let ex_la = &ldet.excitation.alpha;
    let ex_ga = &gdet.excitation.alpha;
    let ex_lb = &ldet.excitation.beta;
    let ex_gb = &gdet.excitation.beta;

    let la = ex_la.holes.count_ones() as usize + ex_ga.holes.count_ones() as usize;
    let lb = ex_lb.holes.count_ones() as usize + ex_gb.holes.count_ones() as usize;

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

    if dosa {
        prepare_same(&w.aa, ex_la, ex_ga, &mut scratch.aa);
        sa = xw_overlap(&w.aa, ex_la, ex_ga, &mut scratch.aa);
    }

    if dosb {
        prepare_same(&w.bb, ex_lb, ex_gb, &mut scratch.bb);
        sb = xw_overlap(&w.bb, ex_lb, ex_gb, &mut scratch.bb);
    }

    let sxw = det_phase * sa * sb;
    let mut gamma = RDM2 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(4)],
    };

    let requests1: Vec<_> = (0..n)
        .flat_map(|p| (0..n).map(move |q| ([p], [q])))
        .collect();
    let g1a = if do1a {
        let mut values = vec![<T as From<f64>>::from(0.0); requests1.len()];
        xw_rdmk_same_prepared_batched::<T, 1>(
            &w.aa,
            (ex_la, ex_ga),
            (ldet.ca.as_ref(), gdet.ca.as_ref()),
            &requests1,
            &mut scratch.aa,
            data.tol,
            &mut values,
        );
        Some(values)
    } else {
        None
    };

    let g1b = if do1b {
        let mut values = vec![<T as From<f64>>::from(0.0); requests1.len()];
        xw_rdmk_same_prepared_batched::<T, 1>(
            &w.bb,
            (ex_lb, ex_gb),
            (ldet.cb.as_ref(), gdet.cb.as_ref()),
            &requests1,
            &mut scratch.bb,
            data.tol,
            &mut values,
        );
        Some(values)
    } else {
        None
    };

    if sb.abs() > data.tol && do2aa {
        if sa.abs() > data.tol
            && let Some(g1a) = g1a.as_ref()
        {
            let scale = det_phase * sb / sa;

            for p in 0..n {
                for q in 0..n {
                    for r in 0..n {
                        for s in 0..n {
                            let i = (((p * n + q) * n + r) * n) + s;
                            gamma.data[i] += scale
                                * (g1a[p * n + r] * g1a[q * n + s]
                                    - g1a[p * n + s] * g1a[q * n + r]);
                        }
                    }
                }
            }
        } else {
            let requests2: Vec<_> = (0..n)
                .flat_map(|p| {
                    (0..n).flat_map(move |q| {
                        (0..n).flat_map(move |r| (0..n).map(move |s| ([p, q], [r, s])))
                    })
                })
                .collect();
            let mut g2aa = vec![<T as From<f64>>::from(0.0); requests2.len()];
            xw_rdmk_same_prepared_batched::<T, 2>(
                &w.aa,
                (ex_la, ex_ga),
                (ldet.ca.as_ref(), gdet.ca.as_ref()),
                &requests2,
                &mut scratch.aa,
                data.tol,
                &mut g2aa,
            );

            for (value, g2) in gamma.data.iter_mut().zip(g2aa) {
                *value += det_phase * sb * g2;
            }
        }
    }

    if sa.abs() > data.tol && do2bb {
        if sb.abs() > data.tol
            && let Some(g1b) = g1b.as_ref()
        {
            let scale = det_phase * sa / sb;

            for p in 0..n {
                for q in 0..n {
                    for r in 0..n {
                        for s in 0..n {
                            let i = (((p * n + q) * n + r) * n) + s;
                            gamma.data[i] += scale
                                * (g1b[p * n + r] * g1b[q * n + s]
                                    - g1b[p * n + s] * g1b[q * n + r]);
                        }
                    }
                }
            }
        } else {
            let requests2: Vec<_> = (0..n)
                .flat_map(|p| {
                    (0..n).flat_map(move |q| {
                        (0..n).flat_map(move |r| (0..n).map(move |s| ([p, q], [r, s])))
                    })
                })
                .collect();
            let mut g2bb = vec![<T as From<f64>>::from(0.0); requests2.len()];
            xw_rdmk_same_prepared_batched::<T, 2>(
                &w.bb,
                (ex_lb, ex_gb),
                (ldet.cb.as_ref(), gdet.cb.as_ref()),
                &requests2,
                &mut scratch.bb,
                data.tol,
                &mut g2bb,
            );

            for (value, g2) in gamma.data.iter_mut().zip(g2bb) {
                *value += det_phase * sa * g2;
            }
        }
    }

    if do2ab && let (Some(g1a), Some(g1b)) = (g1a.as_ref(), g1b.as_ref()) {
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        let i = (((p * n + q) * n + r) * n) + s;
                        gamma.data[i] += det_phase
                            * (g1a[p * n + r] * g1b[q * n + s] + g1b[p * n + r] * g1a[q * n + s]);
                    }
                }
            }
        }
    }

    (sxw, gamma)
}
