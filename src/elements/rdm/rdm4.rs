// elements/rdm/rdm4.rs

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{DetPair, NOCIData};

// Parent/sibling imports.
use super::super::naive::rdm4_pair_naive;
use super::super::nonorthogonalwicks::rdm4_pair_wicks;

/// `Active-space spin-free four-body RDM stored as \Gamma[p, q, r, s, t, u, v, w].`
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

    // Sum each determinant-pair contribution with weight `c_x^L c_w^R`,
    // recording `\sum_{xw} c_x^L c_w^R S_{xw}` for final normalisation.
    for x in 0..data.space.len() {
        for w in 0..data.space.len() {
            let pair = DetPair::new(
                crate::determinant::NOCIIndex(x),
                crate::determinant::NOCIIndex(w),
            );
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

    // Divide the accumulated active-space RDM by the reference norm.
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
    pair: DetPair,
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
