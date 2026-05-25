// noci/rdm/rdm1.rs

use ndarray::Array1;

use crate::nonorthogonalwicks::{WickScratchSpin, lg_overlap, lg_rdm1, prepare_same};

use super::super::naive::{build_s_pair, occ_coeffs, pair_density};
use super::super::types::{DetPair, NOCIData, NOCIScalar};

/// Spin-free one-body RDM stored as \Gamma[p, q].
pub(crate) struct RDM1<T> {
    pub n: usize,
    pub data: Vec<T>,
}

/// Build the spin-free one-body RDM for a NOCI reference state.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `coeff_l`: Left NOCI coefficient vector.
/// - `coeff_r`: Right NOCI coefficient vector.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM1<T>)`: Reference norm and spin-free one-body RDM.
pub(crate) fn rdm1<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    coeff_l: &Array1<T>,
    coeff_r: &Array1<T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, RDM1<T>) {
    let n = data.ao.h.nrows();
    let mut norm = <T as From<f64>>::from(0.0);
    let mut gamma = RDM1 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n * n],
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
                    compare_rdm1_pair_wicks_naive(data, pair, scratch.as_deref_mut().unwrap());

                td += d;
                md = f64::max(md, m);
                (sxw, gxw)
            } else {
                rdm1_pair(data, pair, scratch.as_deref_mut())
            };

            norm += weight * sxw;

            for i in 0..gamma.data.len() {
                gamma.data[i] += weight * gxw.data[i];
            }
        }
    }

    if data.input.wicks.enabled && data.input.wicks.compare {
        println!(
            "Total naive–wicks discrepancy (spin-free 1-RDM): {:.6e}; max element: {:.6e}",
            td, md
        );
    }

    for v in gamma.data.iter_mut() {
        *v /= norm;
    }

    (norm, gamma)
}

/// Calculate a spin-free one-body RDM matrix element block for a determinant pair.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose transition RDM is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM1<T>)`: Pair overlap and spin-free one-body RDM.
fn rdm1_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, RDM1<T>) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    if ldet.parent != gdet.parent && data.input.wicks.enabled {
        rdm1_pair_wicks(
            data,
            pair,
            scratch.expect("Wick scratch required for spin-free 1-RDM evaluation"),
        )
    } else {
        rdm1_pair_naive(data, pair)
    }
}

/// Compare naive and Wick's calculation of spin-free one-body RDM matrix elements.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be compared.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `((T, RDM1<T>), (f64, f64))`: Pair overlap and spin-free 1-RDM from Wick's
///   path, total discrepancy from the naive path, and max elementwise discrepancy.
fn compare_rdm1_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> ((T, RDM1<T>), (f64, f64)) {
    let (sn, g1n) = rdm1_pair_naive(data, pair);
    let (sw, g1w) = rdm1_pair_wicks(data, pair, scratch);

    let mut total = (sn - sw).abs();
    let mut max_element = 0.0;

    for (n, w) in g1n.data.iter().zip(g1w.data.iter()) {
        let diff = (*n - *w).abs();
        total += diff;
        max_element = f64::max(max_element, diff);
    }

    ((sw, g1w), (total, max_element))
}

/// Calculate spin-free one-body RDM matrix elements using generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// # Returns:
/// - `(T, RDM1<T>)`: Pair overlap and spin-free one-body RDM.
fn rdm1_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
) -> (T, RDM1<T>) {
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
    let sxw = det_phase * pa.s * pb.s;

    let da = pair_density(&pa, n);
    let db = pair_density(&pb, n);
    let mut gamma = RDM1 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n * n],
    };

    for p in 0..n {
        for q in 0..n {
            let i = p * n + q;
            gamma.data[i] = det_phase * (pb.s * da[(p, q)] + pa.s * db[(p, q)]);
        }
    }

    (sxw, gamma)
}

/// Calculate spin-free one-body RDM matrix elements using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM1<T>)`: Pair overlap and spin-free one-body RDM.
fn rdm1_pair_wicks<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> (T, RDM1<T>) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;
    let n = data.ao.h.nrows();

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

    let pha = <T as From<f64>>::from(ldet.pha * gdet.pha);
    let phb = <T as From<f64>>::from(ldet.phb * gdet.phb);
    let det_phase = pha * phb;

    let mut sa = <T as From<f64>>::from(0.0);
    let mut sb = <T as From<f64>>::from(0.0);

    if dosa || do1a {
        prepare_same(&w.aa, ex_la, ex_ga, &mut scratch.aa);

        if dosa {
            sa = lg_overlap(&w.aa, ex_la, ex_ga, &mut scratch.aa);
        }
    }

    if dosb || do1b {
        prepare_same(&w.bb, ex_lb, ex_gb, &mut scratch.bb);

        if dosb {
            sb = lg_overlap(&w.bb, ex_lb, ex_gb, &mut scratch.bb);
        }
    }

    let sxw = det_phase * sa * sb;
    let mut gamma = RDM1 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n * n],
    };

    if sb.abs() > data.tol && do1a {
        let g1a = lg_rdm1(
            &w.aa,
            ex_la,
            ex_ga,
            ldet.ca.as_ref(),
            gdet.ca.as_ref(),
            &mut scratch.aa,
            data.tol,
        );

        for p in 0..n {
            for q in 0..n {
                let i = p * n + q;
                gamma.data[i] += det_phase * sb * g1a[(p, q)];
            }
        }
    }

    if sa.abs() > data.tol && do1b {
        let g1b = lg_rdm1(
            &w.bb,
            ex_lb,
            ex_gb,
            ldet.cb.as_ref(),
            gdet.cb.as_ref(),
            &mut scratch.bb,
            data.tol,
        );

        for p in 0..n {
            for q in 0..n {
                let i = p * n + q;
                gamma.data[i] += det_phase * sa * g1b[(p, q)];
            }
        }
    }

    (sxw, gamma)
}
