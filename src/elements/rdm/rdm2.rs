// elements/rdm/rdm2.rs

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{DetPair, NOCIData};

// Parent/sibling imports.
use super::super::naive::rdm2_pair_naive;
use super::super::nonorthogonalwicks::rdm2_pair_wicks;

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

    // Accumulate coefficient-weighted pair RDMs and the reference norm
    // `\sum_{xw} c_x^L c_w^R S_{xw}` before normalisation.
    for x in 0..data.space.len() {
        for w in 0..data.space.len() {
            let pair = DetPair::new(
                crate::determinant::NOCIIndex(x),
                crate::determinant::NOCIIndex(w),
            );
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

    // Convert the unnormalised pair sum into the state RDM.
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
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, RDM2<T>) {
    let lstate = data.space.state(pair.ldet);
    let gstate = data.space.state(pair.gdet);

    if lstate.parent != gstate.parent && data.input.wicks.enabled {
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
    pair: DetPair,
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
