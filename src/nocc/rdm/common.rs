// nocc/rdm/common.rs

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::ExcitationSpin;
use crate::maths::det_occupied_minor_dynamic;
use crate::noci::{NOCIScalar, occ_coeffs};
use crate::nonorthogonalwicks::{SameSpinView, WickScratch, xw_rdmk_same_prepared_batched};

/// Largest number of transition-density requests evaluated in one batch.
const RDMBATCH: usize = 1 << 16;

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

/// Evaluate every same-spin transition-density element of ranks `0..=rank` over the active
/// orbitals, `{}^{xw}\Gamma_\sigma^{p_1\cdots p_k}{}_{q_1\cdots q_k}`, with the rank-zero element
/// the spin-sector overlap. Each rank is evaluated by the batched Wick evaluator, which builds the
/// external-basis contractions once per batch.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates, prepared for this determinant pair.
/// - `ex`: Excitations defining the bra and ket determinants in this spin sector.
/// - `coeff`: Bra- and ket-reference orbital coefficients in this spin sector.
/// - `active`: Active orbital indices in the RDM basis.
/// - `overlap`: Spin-sector overlap.
/// - `rank`: Highest rank, at most four.
/// - `scratch`: Prepared scratch space for this spin sector.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// # Returns
/// - `Vec<Vec<T>>`: Row-major elements of every rank over active positions, upper then lower.
/// # Panics
/// - Panics if `rank` exceeds four.
#[allow(clippy::too_many_arguments)]
pub(super) fn same_spin_transition_rdms<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    active: &[usize],
    overlap: T,
    rank: usize,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Vec<Vec<T>> {
    let mut out = vec![vec![overlap]];
    for k in 1..=rank {
        let values = match k {
            1 => same_spin_rank_elements::<T, 1>(w, ex, coeff, active, scratch, tol),
            2 => same_spin_rank_elements::<T, 2>(w, ex, coeff, active, scratch, tol),
            3 => same_spin_rank_elements::<T, 3>(w, ex, coeff, active, scratch, tol),
            4 => same_spin_rank_elements::<T, 4>(w, ex, coeff, active, scratch, tol),
            _ => panic!("same-spin transition densities are available up to rank four"),
        };
        out.push(values);
    }
    out
}

/// Evaluate every same-spin rank-`K` transition-density element over the active orbitals in
/// batches.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates, prepared for this determinant pair.
/// - `ex`: Excitations defining the bra and ket determinants in this spin sector.
/// - `coeff`: Bra- and ket-reference orbital coefficients in this spin sector.
/// - `active`: Active orbital indices in the RDM basis.
/// - `scratch`: Prepared scratch space for this spin sector.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// # Returns
/// - `Vec<T>`: Row-major elements over active positions `p_1\cdots p_K q_1\cdots q_K`.
fn same_spin_rank_elements<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    active: &[usize],
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> Vec<T> {
    let n = active.len();
    let size = n.pow(2 * K as u32);
    let mut values = vec![<T as From<f64>>::from(0.0); size];

    // Decode each flat position into its creation and annihilation orbitals.
    let request = |flat: usize| {
        let (mut p, mut q) = ([0usize; K], [0usize; K]);
        let mut r = flat;
        for k in (0..K).rev() {
            q[k] = active[r % n];
            r /= n;
        }
        for k in (0..K).rev() {
            p[k] = active[r % n];
            r /= n;
        }
        (p, q)
    };

    let mut requests = Vec::with_capacity(RDMBATCH.min(size));
    for (b, chunk) in values.chunks_mut(RDMBATCH).enumerate() {
        requests.clear();
        requests.extend((0..chunk.len()).map(|i| request(b * RDMBATCH + i)));
        xw_rdmk_same_prepared_batched::<T, K>(w, ex, coeff, &requests, scratch, tol, chunk);
    }

    values
}

/// Combine spin-sector transition densities into one spin-free element,
/// `\Gamma^{p_1\cdots p_k}_{q_1\cdots q_k} = \sum_{A}
/// \Gamma_\alpha^{\mathbf p_A}{}_{\mathbf q_A}\,\Gamma_\beta^{\mathbf p_{\bar A}}{}_{\mathbf q_{\bar A}}`,
/// summed over the sets `A` of operator pairs assigned `\alpha` spin, with each spin subsequence
/// keeping the external operator order and the rank-zero factors the spin-sector overlaps.
/// # Arguments:
/// - `alpha`: Alpha-spin transition densities of every rank, from `same_spin_transition_rdms`.
/// - `beta`: Beta-spin transition densities of every rank.
/// - `n`: Number of active orbitals.
/// - `upper`: Active positions of the creation indices.
/// - `lower`: Active positions of the annihilation indices.
/// # Returns
/// - `T`: Spin-free transition-density element.
pub(super) fn spin_free_element<T: NOCIScalar>(
    alpha: &[Vec<T>],
    beta: &[Vec<T>],
    n: usize,
    upper: &[usize],
    lower: &[usize],
) -> T {
    let k = upper.len();
    let mut value = <T as From<f64>>::from(0.0);
    for mask in 0..1usize << k {
        let (mut ua, mut la, mut ub, mut lb) = (0, 0, 0, 0);
        for i in 0..k {
            if mask & (1 << i) != 0 {
                ua = ua * n + upper[i];
                la = la * n + lower[i];
            } else {
                ub = ub * n + upper[i];
                lb = lb * n + lower[i];
            }
        }
        let ka = mask.count_ones();
        let kb = k as u32 - ka;
        value += alpha[ka as usize][ua * n.pow(ka) + la] * beta[kb as usize][ub * n.pow(kb) + lb];
    }
    value
}
