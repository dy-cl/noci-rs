// nocc/rdm/common.rs

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::maths::det_occupied_minor_dynamic;
use crate::noci::{NOCIScalar, occ_coeffs};

/// Determinant data required by the NOCC RDM evaluators, resolved from the authoritative
/// NOCI space.
pub(super) struct RDMDeterminantView<'a, T: NOCIScalar> {
    /// Parent-reference index used to select ordered Wick intermediates.
    pub(super) parent: usize,
    /// Alpha-spin occupation of the retained determinant.
    pub(super) oa: u128,
    /// Beta-spin occupation of the retained determinant.
    pub(super) ob: u128,
    /// Alpha-spin parent orbital coefficients.
    pub(super) ca: &'a std::sync::Arc<Array2<T>>,
    /// Beta-spin parent orbital coefficients.
    pub(super) cb: &'a std::sync::Arc<Array2<T>>,
    /// Alpha- and beta-spin excitations relative to the parent determinant.
    pub(super) excitation: crate::Excitation,
    /// Alpha-spin fermionic excitation phase.
    pub(super) pha: f64,
    /// Beta-spin fermionic excitation phase.
    pub(super) phb: f64,
}

/// Resolve a retained determinant into the orbital and excitation data required by the RDM code.
/// # Arguments:
/// - `space`: Authoritative retained NOCI determinant space.
/// - `index`: Retained determinant index to resolve.
/// # Returns
/// - `RDMDeterminantView<'a, T>`: Borrowed orbital data and copied determinant metadata.
pub(super) fn resolve_rdm_determinant<'a, T: NOCIScalar>(
    space: &'a crate::noci::NOCISpace<T>,
    index: crate::noci::NOCIIndex,
) -> RDMDeterminantView<'a, T> {
    let state = space.state(index);
    let parent = space.parent(index);
    let alpha = space.alpha(index);
    let beta = space.beta(index);

    RDMDeterminantView {
        parent: state.parent,
        oa: alpha.occupation,
        ob: beta.occupation,
        ca: &parent.ca,
        cb: &parent.cb,
        excitation: crate::Excitation {
            alpha: alpha.excitation,
            beta: beta.excitation,
        },
        pha: alpha.reduced.phase,
        phb: beta.reduced.phase,
    }
}

/// Split creation and annihilation indices by spin assignment mask.
/// # Arguments:
/// - `ps`: Creation indices in the full RDM basis.
/// - `qs`: Annihilation indices in the full RDM basis.
/// - `mask`: Spin assignment mask, where bit `i` selects beta for operator `i`.
/// # Returns
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

/// Calculate one spin-assignment contribution to a spin-free RDM element by determinant expansion.
/// # Arguments:
/// - `ldet`: Resolved bra determinant data.
/// - `gdet`: Resolved ket determinant data.
/// - `ps`: Creation indices in the full RDM basis.
/// - `qs`: Annihilation indices in the full RDM basis.
/// - `mask`: Spin assignment mask, where bit `i` selects beta for operator `i`.
/// # Returns
/// - `T`: Spin-assignment contribution to the spin-free RDM element.
pub(super) fn spin_assignment_rdm_element_naive<T: NOCIScalar>(
    ldet: &RDMDeterminantView<'_, T>,
    gdet: &RDMDeterminantView<'_, T>,
    ps: &[usize],
    qs: &[usize],
    mask: usize,
) -> T {
    let zero = <T as From<f64>>::from(0.0);

    // Fixed spin assignments factor into independent `\alpha` and `\beta`
    // determinant matrix elements; the spin-free RDM sums these assignments.
    let (pa, qa, pb, qb) = split_spin_assignment(ps, qs, mask);
    let nela = ldet.oa.count_ones() as usize;
    let nelb = ldet.ob.count_ones() as usize;

    if pa.len() > nela || pa.len() > gdet.oa.count_ones() as usize {
        return zero;
    }

    if pb.len() > nelb || pb.len() > gdet.ob.count_ones() as usize {
        return zero;
    }

    let l_ca_occ = occ_coeffs(ldet.ca.as_ref(), ldet.oa);
    let g_ca_occ = occ_coeffs(gdet.ca.as_ref(), gdet.oa);
    let l_cb_occ = occ_coeffs(ldet.cb.as_ref(), ldet.ob);
    let g_cb_occ = occ_coeffs(gdet.cb.as_ref(), gdet.ob);

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
/// # Returns
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

    // Expand the ket Slater determinant over occupation bitstrings; each
    // coefficient is the corresponding occupied-orbital minor of `g_c`.
    let mut acc = zero;
    let limit = 1u128 << norb;

    for ket in 0..limit {
        if ket.count_ones() as usize != nel {
            continue;
        }

        let cg = det_occupied_minor_dynamic(g_c, ket, nel);
        let mut bra = ket;
        let mut phase = one;
        let mut valid = true;

        // Apply the annihilators in the supplied order. An operator crossing
        // each occupied orbital below `q` contributes one fermionic minus sign.
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

        // Creation acts in reverse index order on the intermediate bitstring;
        // the same occupied-below parity determines its sign.
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

        // Contract the resulting bra configuration with its determinant minor.
        if valid {
            let cl = det_occupied_minor_dynamic(l_c, bra, nel);
            acc += phase * cl * cg;
        }
    }

    acc
}
