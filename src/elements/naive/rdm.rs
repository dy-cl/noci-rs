// elements/naive/rdm.rs
//! Transition reduced density matrices of determinant pairs from the generalised
//! Slater–Condon rules.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::{DetPair, NOCIData};
use crate::maths::det_occupied_minor_dynamic;

// Parent/sibling imports.
use super::super::rdm::{RDM1, RDM2, RDM3, RDM4, RDMDeterminantView, resolve_rdm_determinant};
use super::rules::{build_s_pair, occ_coeffs, pair_density};

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
fn spin_assignment_rdm_element_naive<T: NOCIScalar>(
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

/// Calculate spin-free one-body RDM matrix elements using generalised Slater-Condon rules.
/// Forms `Gamma^p_q = S_beta rho^alpha[p,q] + S_alpha rho^beta[p,q]`, including the
/// determinant excitation phase, from the two spin-resolved transition densities.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// # Returns:
/// - `(T, RDM1<T>)`: Pair overlap and spin-free one-body RDM.
pub(in crate::elements) fn rdm1_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
) -> (T, RDM1<T>) {
    // Resolve the determinant pair and external AO-basis tensor dimension.
    let ldet = resolve_rdm_determinant(data.space, pair.ldet);
    let gdet = resolve_rdm_determinant(data.space, pair.gdet);
    let n = data.ao.h.nrows();

    // Extract occupied MO coefficients for each bra/ket spin sector.
    let l_ca_occ = occ_coeffs(ldet.ca.as_ref(), ldet.oa);
    let g_ca_occ = occ_coeffs(gdet.ca.as_ref(), gdet.oa);
    let l_cb_occ = occ_coeffs(ldet.cb.as_ref(), ldet.ob);
    let g_cb_occ = occ_coeffs(gdet.cb.as_ref(), gdet.ob);

    // Build nonorthogonal spin overlaps and their transition-density intermediates.
    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
    let sxw = det_phase * pa.s * pb.s;

    // Transform spin transition densities to the AO basis and combine their spin complements.
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

/// Calculate spin-free two-body RDM matrix elements using generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// # Returns:
/// - `(T, RDM2<T>)`: Pair overlap and spin-free two-body RDM.
pub(in crate::elements) fn rdm2_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
) -> (T, RDM2<T>) {
    let ldet = resolve_rdm_determinant(data.space, pair.ldet);
    let gdet = resolve_rdm_determinant(data.space, pair.gdet);
    let n = data.ao.h.nrows();

    let l_ca_occ = occ_coeffs(ldet.ca.as_ref(), ldet.oa);
    let g_ca_occ = occ_coeffs(gdet.ca.as_ref(), gdet.oa);
    let l_cb_occ = occ_coeffs(ldet.cb.as_ref(), ldet.ob);
    let g_cb_occ = occ_coeffs(gdet.cb.as_ref(), gdet.ob);

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

    // Same-spin terms are antisymmetrised products `A_{pr} B_{qs} - A_{ps} B_{qr}`.
    // Zero, one, or two zero singular values select `W`, `P_i`, or `P_j`
    // transition co-densities in the generalised Slater-Condon expansion.
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

    // Opposite-spin terms factor into `\alpha` and `\beta` one-body densities:
    // `\Gamma_{pqrs} += D^\alpha_{pr} D^\beta_{qs} + D^\beta_{pr} D^\alpha_{qs}`.
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

/// Calculate active-space spin-free three-body RDM matrix elements by determinant expansion.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// - `active`: Active orbital indices in the RDM basis.
/// # Returns:
/// - `(T, RDM3<T>)`: Pair overlap and active-space spin-free three-body RDM.
pub(in crate::elements) fn rdm3_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    active: &[usize],
) -> (T, RDM3<T>) {
    let ldet = resolve_rdm_determinant(data.space, pair.ldet);
    let gdet = resolve_rdm_determinant(data.space, pair.gdet);
    let n = active.len();

    let l_ca_occ = occ_coeffs(ldet.ca.as_ref(), ldet.oa);
    let g_ca_occ = occ_coeffs(gdet.ca.as_ref(), gdet.oa);
    let l_cb_occ = occ_coeffs(ldet.cb.as_ref(), ldet.ob);
    let g_cb_occ = occ_coeffs(gdet.cb.as_ref(), gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
    let sxw = det_phase * pa.s * pb.s;
    let mut gamma = RDM3 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(6)],
    };

    // Sum all `2^3` `\alpha`/`\beta` assignments for each active-index sextuplet;
    // the assignment helper includes the fermionic operator signs.
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    for e in 0..n {
                        for f in 0..n {
                            let ps = [active[a], active[b], active[c]];
                            let qs = [active[d], active[e], active[f]];
                            let mut val = <T as From<f64>>::from(0.0);

                            for mask in 0..8 {
                                val +=
                                    spin_assignment_rdm_element_naive(&ldet, &gdet, &ps, &qs, mask);
                            }

                            let i = (((((a * n + b) * n + c) * n + d) * n + e) * n) + f;
                            gamma.data[i] = det_phase * val;
                        }
                    }
                }
            }
        }
    }

    (sxw, gamma)
}

/// Calculate active-space spin-free four-body RDM matrix elements by determinant expansion.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// - `active`: Active orbital indices in the RDM basis.
/// # Returns:
/// - `(T, RDM4<T>)`: Pair overlap and active-space spin-free four-body RDM.
pub(in crate::elements) fn rdm4_pair_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    active: &[usize],
) -> (T, RDM4<T>) {
    let ldet = resolve_rdm_determinant(data.space, pair.ldet);
    let gdet = resolve_rdm_determinant(data.space, pair.gdet);
    let n = active.len();

    let l_ca_occ = occ_coeffs(ldet.ca.as_ref(), ldet.oa);
    let g_ca_occ = occ_coeffs(gdet.ca.as_ref(), gdet.oa);
    let l_cb_occ = occ_coeffs(ldet.cb.as_ref(), ldet.ob);
    let g_cb_occ = occ_coeffs(gdet.cb.as_ref(), gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &data.ao.s, data.tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &data.ao.s, data.tol);

    let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
    let sxw = det_phase * pa.s * pb.s;
    let mut gamma = RDM4 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(8)],
    };

    // Sum all `2^4` `\alpha`/`\beta` assignments for each active-index octuplet;
    // the assignment helper includes the fermionic operator signs.
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
                                        val += spin_assignment_rdm_element_naive(
                                            &ldet, &gdet, &ps, &qs, mask,
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
