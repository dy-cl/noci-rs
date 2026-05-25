// noci/rdm/rdm4.rs

use ndarray::Array1;

use crate::nonorthogonalwicks::{WickScratchSpin, lg_overlap, prepare_same};

use super::common::{spin_assignment_rdm_element, spin_assignment_rdm_element_naive};
use crate::noci::naive::{build_s_pair, occ_coeffs};
use crate::noci::types::{DetPair, NOCIData, NOCIScalar};

/// Active-space spin-free four-body RDM stored as \Gamma[p, q, r, s, t, u, v, w].
pub(crate) struct RDM4<T> {
    pub n: usize,
    pub data: Vec<T>,
}

/// Build the active-space spin-free four-body RDM for a NOCI reference state.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `coeff_l`: Left NOCI coefficient vector.
/// - `coeff_r`: Right NOCI coefficient vector.
/// - `active`: Active orbital indices in the RDM basis.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM4<T>)`: Reference norm and active-space spin-free four-body RDM.
pub(crate) fn rdm4<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    coeff_l: &Array1<T>,
    coeff_r: &Array1<T>,
    active: &[usize],
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, RDM4<T>) {
    let n = active.len();
    let mut norm = <T as From<f64>>::from(0.0);
    let mut gamma = RDM4 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(8)],
    };
    let mut scratch = scratch;
    let mut td = 0.0;
    let mut md = 0.0;

    for x in 0..data.basis.len() {
        for w in 0..data.basis.len() {
            let pair = DetPair::new(&data.basis[x], &data.basis[w]);
            let weight = coeff_l[x] * coeff_r[w];

            let (sxw, gxw) = if data.input.wicks.enabled && data.input.wicks.compare {
                let ((sxw, gxw), (d, m)) = compare_rdm4_pair_wicks_naive(
                    data,
                    pair,
                    active,
                    scratch.as_deref_mut().unwrap(),
                );

                td += d;
                md = f64::max(md, m);
                (sxw, gxw)
            } else if data.input.wicks.enabled {
                rdm4_pair_wicks(data, pair, active, scratch.as_deref_mut().unwrap())
            } else {
                rdm4_pair_naive(data, pair, active)
            };

            norm += weight * sxw;

            for i in 0..gamma.data.len() {
                gamma.data[i] += weight * gxw.data[i];
            }
        }
    }

    if data.input.wicks.enabled && data.input.wicks.compare {
        println!(
            "Total naive–wicks discrepancy (active spin-free 4-RDM): {:.6e}; max element: {:.6e}",
            td, md
        );
    }

    for v in gamma.data.iter_mut() {
        *v /= norm;
    }

    (norm, gamma)
}

/// Compare naive and Wick's calculation of active-space spin-free four-body RDM matrix elements.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be compared.
/// - `active`: Active orbital indices in the RDM basis.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `((T, RDM4<T>), (f64, f64))`: Pair overlap and active-space spin-free 4-RDM
///   from Wick's path, total discrepancy from the naive path, and max elementwise
///   discrepancy.
fn compare_rdm4_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    active: &[usize],
    scratch: &mut WickScratchSpin<T>,
) -> ((T, RDM4<T>), (f64, f64)) {
    let (sn, g4n) = rdm4_pair_naive(data, pair, active);
    let (sw, g4w) = rdm4_pair_wicks(data, pair, active, scratch);

    let mut total = (sn - sw).abs();
    let mut max_element = 0.0;

    for (n, w) in g4n.data.iter().zip(g4w.data.iter()) {
        let diff = (*n - *w).abs();
        total += diff;
        max_element = f64::max(max_element, diff);
    }

    ((sw, g4w), (total, max_element))
}

/// Calculate active-space spin-free four-body RDM matrix elements by determinant expansion.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// - `active`: Active orbital indices in the RDM basis.
/// # Returns:
/// - `(T, RDM4<T>)`: Pair overlap and active-space spin-free four-body RDM.
fn rdm4_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    active: &[usize],
) -> (T, RDM4<T>) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;
    let n = active.len();

    let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
    let sxw = det_phase * pa.s * pb.s;
    let mut gamma = RDM4 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(8)],
    };

    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    for e in 0..n {
                        for f in 0..n {
                            for g in 0..n {
                                for h in 0..n {
                                    let ps = [active[a], active[b], active[c], active[d]];
                                    let qs = [active[e], active[f], active[g], active[h]];
                                    let mut val = <T as From<f64>>::from(0.0);

                                    for mask in 0..16 {
                                        val +=
                                            spin_assignment_rdm_element_naive(pair, &ps, &qs, mask);
                                    }

                                    let i = (((((((a * n + b) * n + c) * n + d) * n + e) * n + f)
                                        * n
                                        + g)
                                        * n)
                                        + h;
                                    gamma.data[i] = det_phase * val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (sxw, gamma)
}

/// Calculate active-space spin-free four-body RDM matrix elements using Wick's theorem.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose transition RDM is to be evaluated.
/// - `active`: Active orbital indices in the RDM basis.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM4<T>)`: Pair overlap and active-space spin-free four-body RDM.
fn rdm4_pair_wicks<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    active: &[usize],
    scratch: &mut WickScratchSpin<T>,
) -> (T, RDM4<T>) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;
    let n = active.len();

    let wicks = data.wicks.unwrap();
    let w = wicks.pair(ldet.parent, gdet.parent);

    prepare_same(
        &w.aa,
        &ldet.excitation.alpha,
        &gdet.excitation.alpha,
        &mut scratch.aa,
    );
    prepare_same(
        &w.bb,
        &ldet.excitation.beta,
        &gdet.excitation.beta,
        &mut scratch.bb,
    );

    let sa = lg_overlap(
        &w.aa,
        &ldet.excitation.alpha,
        &gdet.excitation.alpha,
        &mut scratch.aa,
    );
    let sb = lg_overlap(
        &w.bb,
        &ldet.excitation.beta,
        &gdet.excitation.beta,
        &mut scratch.bb,
    );

    let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
    let sxw = det_phase * sa * sb;
    let mut gamma = RDM4 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(8)],
    };

    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    for e in 0..n {
                        for f in 0..n {
                            for g in 0..n {
                                for h in 0..n {
                                    let ps = [active[a], active[b], active[c], active[d]];
                                    let qs = [active[e], active[f], active[g], active[h]];
                                    let mut val = <T as From<f64>>::from(0.0);

                                    for mask in 0..16 {
                                        val += spin_assignment_rdm_element(
                                            data, pair, &ps, &qs, mask, scratch,
                                        );
                                    }

                                    let i = (((((((a * n + b) * n + c) * n + d) * n + e) * n + f)
                                        * n
                                        + g)
                                        * n)
                                        + h;
                                    gamma.data[i] = det_phase * val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (sxw, gamma)
}
