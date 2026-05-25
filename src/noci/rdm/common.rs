// noci/rdm/common.rs

use ndarray::Array2;

use crate::maths::det_occupied_minor;
use crate::noci::naive::occ_coeffs;
use crate::noci::types::{DetPair, NOCIData, NOCIScalar};
use crate::nonorthogonalwicks::{WickScratchSpin, lg_rdm_same_element};

/// Split creation and annihilation indices by spin assignment mask.
/// # Arguments:
/// - `ps`: Creation indices in the full RDM basis.
/// - `qs`: Annihilation indices in the full RDM basis.
/// - `mask`: Spin assignment mask, where bit `i` selects beta for operator `i`.
/// # Returns:
/// - `(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>)`: Alpha creation, alpha annihilation,
///   beta creation, and beta annihilation indices.
fn split_spin_assignment(
    ps: &[usize],
    qs: &[usize],
    mask: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut pa = Vec::new();
    let mut qa = Vec::new();
    let mut pb = Vec::new();
    let mut qb = Vec::new();

    for i in 0..ps.len() {
        if (mask >> i) & 1 == 0 {
            pa.push(ps[i]);
            qa.push(qs[i]);
        } else {
            pb.push(ps[i]);
            qb.push(qs[i]);
        }
    }

    (pa, qa, pb, qb)
}

/// Calculate one spin-assignment contribution to a spin-free RDM element using Wick's theorem.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose transition RDM element is to be evaluated.
/// - `ps`: Creation indices in the full RDM basis.
/// - `qs`: Annihilation indices in the full RDM basis.
/// - `mask`: Spin assignment mask, where bit `i` selects beta for operator `i`.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Spin-assignment contribution to the spin-free RDM element.
pub(super) fn spin_assignment_rdm_element<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    ps: &[usize],
    qs: &[usize],
    mask: usize,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    let ldet = pair.ldet;
    let gdet = pair.gdet;
    let wicks = data.wicks.unwrap();
    let w = wicks.pair(ldet.parent, gdet.parent);
    let zero = <T as From<f64>>::from(0.0);

    let (pa, qa, pb, qb) = split_spin_assignment(ps, qs, mask);

    if pa.len() > ldet.oa.count_ones() as usize || pa.len() > gdet.oa.count_ones() as usize {
        return zero;
    }

    if pb.len() > ldet.ob.count_ones() as usize || pb.len() > gdet.ob.count_ones() as usize {
        return zero;
    }

    let la = ldet.excitation.alpha.holes.len() + gdet.excitation.alpha.holes.len();
    let lb = ldet.excitation.beta.holes.len() + gdet.excitation.beta.holes.len();

    let va = if w.aa.m <= la + pa.len() {
        lg_rdm_same_element(
            &w.aa,
            (&ldet.excitation.alpha, &gdet.excitation.alpha),
            (ldet.ca.as_ref(), gdet.ca.as_ref()),
            (&pa, &qa),
            &mut scratch.aa,
            data.tol,
        )
    } else {
        zero
    };

    let vb = if w.bb.m <= lb + pb.len() {
        lg_rdm_same_element(
            &w.bb,
            (&ldet.excitation.beta, &gdet.excitation.beta),
            (ldet.cb.as_ref(), gdet.cb.as_ref()),
            (&pb, &qb),
            &mut scratch.bb,
            data.tol,
        )
    } else {
        zero
    };

    va * vb
}

/// Calculate one spin-assignment contribution to a spin-free RDM element by determinant expansion.
/// # Arguments:
/// - `pair`: Pair of determinants whose transition RDM element is to be evaluated.
/// - `ps`: Creation indices in the full RDM basis.
/// - `qs`: Annihilation indices in the full RDM basis.
/// - `mask`: Spin assignment mask, where bit `i` selects beta for operator `i`.
/// # Returns:
/// - `T`: Spin-assignment contribution to the spin-free RDM element.
pub(super) fn spin_assignment_rdm_element_naive<T: NOCIScalar>(
    pair: DetPair<'_, T>,
    ps: &[usize],
    qs: &[usize],
    mask: usize,
) -> T {
    let ldet = pair.ldet;
    let gdet = pair.gdet;
    let zero = <T as From<f64>>::from(0.0);

    let (pa, qa, pb, qb) = split_spin_assignment(ps, qs, mask);
    let nela = ldet.oa.count_ones() as usize;
    let nelb = ldet.ob.count_ones() as usize;

    if pa.len() > nela || pa.len() > gdet.oa.count_ones() as usize {
        return zero;
    }

    if pb.len() > nelb || pb.len() > gdet.ob.count_ones() as usize {
        return zero;
    }

    let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

    let va = same_spin_rdm_element_naive(&l_ca_occ, &g_ca_occ, nela, &pa, &qa);
    let vb = same_spin_rdm_element_naive(&l_cb_occ, &g_cb_occ, nelb, &pb, &qb);

    va * vb
}

/// Calculate a same-spin RDM element by explicit determinant expansion.
/// # Arguments:
/// - `l_c`: Left determinant orbital coefficients in an orthonormal RDM basis.
/// - `g_c`: Right determinant orbital coefficients in an orthonormal RDM basis.
/// - `nel`: Number of electrons in the spin block.
/// - `ps`: Creation indices in the RDM basis.
/// - `qs`: Annihilation indices in the RDM basis.
/// # Returns:
/// - `T`: Same-spin RDM element for the supplied creation and annihilation indices.
fn same_spin_rdm_element_naive<T: NOCIScalar>(
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    nel: usize,
    ps: &[usize],
    qs: &[usize],
) -> T {
    let zero = <T as From<f64>>::from(0.0);
    let one = <T as From<f64>>::from(1.0);
    let minus_one = <T as From<f64>>::from(-1.0);

    if ps.len() != qs.len() || ps.len() > nel {
        return zero;
    }

    let norb = g_c.nrows();

    let mut acc = zero;
    let limit = 1u128 << norb;

    for ket in 0..limit {
        if ket.count_ones() as usize != nel {
            continue;
        }

        let cg = det_occupied_minor(g_c, ket, nel);
        let mut bra = ket;
        let mut phase = one;
        let mut valid = true;

        for &q in qs {
            if ((bra >> q) & 1) == 0 {
                valid = false;
                break;
            }

            if (bra & ((1u128 << q) - 1)).count_ones() % 2 == 1 {
                phase *= minus_one;
            }
            bra &= !(1u128 << q);
        }

        if !valid {
            continue;
        }

        for &p in ps.iter().rev() {
            if ((bra >> p) & 1) == 1 {
                valid = false;
                break;
            }

            if (bra & ((1u128 << p) - 1)).count_ones() % 2 == 1 {
                phase *= minus_one;
            }
            bra |= 1u128 << p;
        }

        if valid {
            let cl = det_occupied_minor(l_c, bra, nel);
            acc += phase * cl * cg;
        }
    }

    acc
}
