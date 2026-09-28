// elements/rdm/rdm3.rs

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{DetPair, NOCIData};

// Parent/sibling imports.
use super::super::naive::rdm3_pair_naive;
use super::super::nonorthogonalwicks::rdm3_pair_wicks;

/// `Active-space spin-free three-body RDM stored as \Gamma[p, q, r, s, t, u].`
pub(crate) struct RDM3<T> {
    pub n: usize,
    pub data: Vec<T>,
}

/// Build the active-space spin-free three-body RDM for a NOCI reference state.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `coeff_l`: Left NOCI coefficient vector.
/// - `coeff_r`: Right NOCI coefficient vector.
/// - `active`: Active orbital indices in the RDM basis.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM3<T>)`: Reference norm and active-space spin-free three-body RDM.
pub(crate) fn rdm3<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    coeff_l: &Array1<T>,
    coeff_r: &Array1<T>,
    active: &[usize],
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, RDM3<T>) {
    let n = active.len();
    let mut norm = <T as From<f64>>::from(0.0);
    let mut gamma = RDM3 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(6)],
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
                let ((sxw, gxw), (d, m)) = compare_rdm3_pair_wicks_naive(
                    data,
                    pair,
                    active,
                    scratch.as_deref_mut().unwrap(),
                );

                td += d;
                md = f64::max(md, m);
                (sxw, gxw)
            } else if data.input.wicks.enabled {
                rdm3_pair_wicks(data, pair, active, scratch.as_deref_mut().unwrap())
            } else {
                rdm3_pair_naive(data, pair, active)
            };

            norm += weight * sxw;

            for i in 0..gamma.data.len() {
                gamma.data[i] += weight * gxw.data[i];
            }
        }
    }

    if data.input.wicks.enabled && data.input.wicks.compare {
        println!(
            "Total naive–wicks discrepancy (active spin-free 3-RDM): {:.6e}; max element: {:.6e}",
            td, md
        );
    }

    // Divide the accumulated active-space RDM by the reference norm.
    for v in gamma.data.iter_mut() {
        *v /= norm;
    }

    (norm, gamma)
}

/// Compare naive and Wick's calculation of active-space spin-free three-body RDM matrix elements.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be compared.
/// - `active`: Active orbital indices in the RDM basis.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `((T, RDM3<T>), (f64, f64))`: Pair overlap and active-space spin-free 3-RDM
///   from Wick's path, total discrepancy from the naive path, and max elementwise
///   discrepancy.
fn compare_rdm3_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    active: &[usize],
    scratch: &mut WickScratchSpin<T>,
) -> ((T, RDM3<T>), (f64, f64)) {
    let (sn, g3n) = rdm3_pair_naive(data, pair, active);
    let (sw, g3w) = rdm3_pair_wicks(data, pair, active, scratch);

    let mut total = (sn - sw).abs();
    let mut max_element = 0.0;

    for (n, w) in g3n.data.iter().zip(g3w.data.iter()) {
        let diff = (*n - *w).abs();
        total += diff;
        max_element = f64::max(max_element, diff);
    }

    ((sw, g3w), (total, max_element))
}
