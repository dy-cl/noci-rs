// elements/nonorthogonalwicks/rdm.rs
//! Transition reduced density matrices of determinant pairs from the extended
//! nonorthogonal Wick theorem.

// External crate imports.
use ndarray::Array2;
use rayon::prelude::*;

// Crate-root imports.
use crate::elements::{DetPair, NOCIData};
use crate::{ExcitationSpin, NOCIScalar};

// Parent/sibling imports.
use super::super::rdm::{RDM1, RDM2, RDM3, RDM4, resolve_rdm_determinant};
use super::{
    SameSpinView, WickScratch, WickScratchSpin, prepare_same, xw_overlap,
    xw_rdmk_same_prepared_batched,
};

/// Largest number of transition-density requests evaluated in one batch.
const RDMBATCH: usize = 1 << 16;

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
fn same_spin_transition_rdms<T: NOCIScalar>(
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
fn spin_free_element<T: NOCIScalar>(
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

/// Calculate spin-free one-body RDM matrix elements using extended non-orthogonal Wick's theorem.
/// Evaluates `Gamma^p_q = Gamma^{alpha,p}_q S_beta + S_alpha Gamma^{beta,p}_q`
/// from batched rank-one fundamental contractions.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM1<T>)`: Pair overlap and spin-free one-body RDM.
pub(in crate::elements) fn rdm1_pair_wicks<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    scratch: &mut WickScratchSpin<T>,
) -> (T, RDM1<T>) {
    // Resolve the determinant pair, AO dimension, and ordered parent-pair Wick data.
    let ldet = resolve_rdm_determinant(data.space, pair.ldet);
    let gdet = resolve_rdm_determinant(data.space, pair.gdet);
    let n = data.ao.h.nrows();

    let wicks = data.wicks.unwrap();
    let w = wicks.pair(ldet.parent, gdet.parent);

    // Cache spin excitations and determine which zero-overlap distributions can contribute.
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

    // Prepare surviving spin overlap determinants and retain their excitation phases.
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

    // Allocate the AO-basis spin-free tensor and its complete rank-one request batch.
    let sxw = det_phase * sa * sb;
    let mut gamma = RDM1 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n * n],
    };
    let requests: Vec<_> = (0..n)
        .flat_map(|p| (0..n).map(move |q| ([p], [q])))
        .collect();

    // Add alpha transition density weighted by the complementary beta overlap.
    if sb.abs() > data.tol && do1a {
        let mut g1a = vec![<T as From<f64>>::from(0.0); requests.len()];
        xw_rdmk_same_prepared_batched::<T, 1>(
            &w.aa,
            (ex_la, ex_ga),
            (ldet.ca.as_ref(), gdet.ca.as_ref()),
            &requests,
            &mut scratch.aa,
            data.tol,
            &mut g1a,
        );

        for (value, g1) in gamma.data.iter_mut().zip(g1a) {
            *value += det_phase * sb * g1;
        }
    }

    // Add the symmetric beta contribution weighted by the alpha overlap.
    if sa.abs() > data.tol && do1b {
        let mut g1b = vec![<T as From<f64>>::from(0.0); requests.len()];
        xw_rdmk_same_prepared_batched::<T, 1>(
            &w.bb,
            (ex_lb, ex_gb),
            (ldet.cb.as_ref(), gdet.cb.as_ref()),
            &requests,
            &mut scratch.bb,
            data.tol,
            &mut g1b,
        );

        for (value, g1) in gamma.data.iter_mut().zip(g1b) {
            *value += det_phase * sa * g1;
        }
    }

    (sxw, gamma)
}

/// Calculate spin-free two-body RDM matrix elements using extended non-orthogonal Wick's theorem.
///
/// The returned tensor is the spin sum
/// `Gamma^{pq}_{rs} = sum_{sigma,tau}<a^+_{p sigma} a^+_{q tau}
/// a_{s tau} a_{r sigma}>`. Same-spin blocks are antisymmetrized products when
/// their one-body transition densities are nonsingular; mixed-spin blocks are
/// direct products of the alpha and beta one-body transition densities.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose RDM matrix elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM2<T>)`: Pair overlap and spin-free two-body RDM.
pub(in crate::elements) fn rdm2_pair_wicks<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    scratch: &mut WickScratchSpin<T>,
) -> (T, RDM2<T>) {
    // Resolve the determinant pair, parent Wick data, and excitation ranks.
    let ldet = resolve_rdm_determinant(data.space, pair.ldet);
    let gdet = resolve_rdm_determinant(data.space, pair.gdet);
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

    // Prepare each spin sector only when its overlap contraction can survive.
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

    // Allocate the spin-free transition tensor and retain the phased pair overlap.
    let sxw = det_phase * sa * sb;
    let mut gamma = RDM2 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(4)],
    };

    // Batch all one-body transition densities used by the product reconstructions.
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

    // Add the alpha-alpha block, using its antisymmetrized one-body product when stable.
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
            // Evaluate Gamma2 directly when division by the alpha overlap is singular.
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

    // Add the beta-beta block by the spin-mirrored construction.
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
            // Evaluate Gamma2 directly when division by the beta overlap is singular.
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

    // Add both alpha-beta orderings; unlike same-spin blocks these have no exchange term.
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

/// Calculate active-space spin-free three-body RDM matrix elements using Wick's theorem.
///
/// This evaluates `Gamma^{pqr}_{stu} = sum_{sigma,tau,upsilon}
/// <a^+_{p sigma} a^+_{q tau} a^+_{r upsilon}
/// a_{u upsilon} a_{t tau} a_{s sigma}>` by summing its eight alpha/beta
/// assignments while preserving the external operator order within each spin sector.
/// Known issue: for determinant pairs whose alpha and beta overlaps have opposite signs, such as
/// RHF with UHF, individual pair contributions differ from the naive expansion by up to order
/// one, although the pair `(x, w)` and `(w, x)` contributions cancel exactly. The coefficient-
/// weighted RDM of one real state, the only current use, therefore matches the naive result to
/// machine precision; transition RDMs with different bra and ket coefficients would be affected.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose transition RDM is to be evaluated.
/// - `active`: Active orbital indices in the RDM basis.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM3<T>)`: Pair overlap and active-space spin-free three-body RDM.
pub(in crate::elements) fn rdm3_pair_wicks<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    active: &[usize],
    scratch: &mut WickScratchSpin<T>,
) -> (T, RDM3<T>) {
    // Resolve the determinant pair and its parent-pair Wick intermediates.
    let ldet = resolve_rdm_determinant(data.space, pair.ldet);
    let gdet = resolve_rdm_determinant(data.space, pair.gdet);
    let n = active.len();

    let wicks = data.wicks.unwrap();
    let w = wicks.pair(ldet.parent, gdet.parent);

    // Prepare the alpha and beta contraction determinants for repeated RDM queries.
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

    // Evaluate spin-sector overlaps and the determinant excitation phase.
    let sa = xw_overlap(
        &w.aa,
        &ldet.excitation.alpha,
        &gdet.excitation.alpha,
        &mut scratch.aa,
    );
    let sb = xw_overlap(
        &w.bb,
        &ldet.excitation.beta,
        &gdet.excitation.beta,
        &mut scratch.bb,
    );

    let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
    let sxw = det_phase * sa * sb;

    // Allocate the active-space tensor and shared arguments for mixed-spin contractions.
    let mut gamma = RDM3 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(6)],
    };
    let coeff = (
        (ldet.ca.as_ref(), gdet.ca.as_ref()),
        (ldet.cb.as_ref(), gdet.cb.as_ref()),
    );

    // Evaluate every same-spin transition density once per rank and combine them into each
    // spin-free element over all `2^3` spin assignments.
    let alpha = same_spin_transition_rdms(
        &w.aa,
        (&ldet.excitation.alpha, &gdet.excitation.alpha),
        coeff.0,
        active,
        sa,
        3,
        &mut scratch.aa,
        data.tol,
    );
    let beta = same_spin_transition_rdms(
        &w.bb,
        (&ldet.excitation.beta, &gdet.excitation.beta),
        coeff.1,
        active,
        sb,
        3,
        &mut scratch.bb,
        data.tol,
    );
    let nk = n.pow(3);
    gamma
        .data
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat, value)| {
            let (mut upper, mut lower) = ([0usize; 3], [0usize; 3]);
            let (mut u, mut l) = (flat / nk, flat % nk);
            for k in (0..3).rev() {
                upper[k] = u % n;
                lower[k] = l % n;
                u /= n;
                l /= n;
            }
            *value = det_phase * spin_free_element(&alpha, &beta, n, &upper, &lower);
        });

    (sxw, gamma)
}

/// Calculate active-space spin-free four-body RDM matrix elements using Wick's theorem.
///
/// This evaluates `Gamma^{pqrs}_{tuvw} = sum_{sigma_1...sigma_4}
/// <a^+_{p sigma_1}...a^+_{s sigma_4} a_{w sigma_4}...a_{t sigma_1}>`
/// by summing all sixteen alpha/beta assignments while preserving the external
/// operator order within each spin sector.
/// Known issue: for determinant pairs whose alpha and beta overlaps have opposite signs, such as
/// RHF with UHF, individual pair contributions differ from the naive expansion by up to order
/// one, although the pair `(x, w)` and `(w, x)` contributions cancel exactly. The coefficient-
/// weighted RDM of one real state, the only current use, therefore matches the naive result to
/// machine precision; transition RDMs with different bra and ket coefficients would be affected.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose transition RDM is to be evaluated.
/// - `active`: Active orbital indices in the RDM basis.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, RDM4<T>)`: Pair overlap and active-space spin-free four-body RDM.
pub(in crate::elements) fn rdm4_pair_wicks<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    active: &[usize],
    scratch: &mut WickScratchSpin<T>,
) -> (T, RDM4<T>) {
    // Resolve the determinant pair and its parent-pair Wick intermediates.
    let ldet = resolve_rdm_determinant(data.space, pair.ldet);
    let gdet = resolve_rdm_determinant(data.space, pair.gdet);
    let n = active.len();

    let wicks = data.wicks.unwrap();
    let w = wicks.pair(ldet.parent, gdet.parent);

    // Prepare both spin contraction determinants for repeated rank-four queries.
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

    // Evaluate spin-sector overlaps and the determinant excitation phase.
    let sa = xw_overlap(
        &w.aa,
        &ldet.excitation.alpha,
        &gdet.excitation.alpha,
        &mut scratch.aa,
    );
    let sb = xw_overlap(
        &w.bb,
        &ldet.excitation.beta,
        &gdet.excitation.beta,
        &mut scratch.bb,
    );

    let det_phase = <T as From<f64>>::from((ldet.pha * gdet.pha) * (ldet.phb * gdet.phb));
    let sxw = det_phase * sa * sb;

    // Allocate the active-space tensor and shared arguments for mixed-spin contractions.
    let mut gamma = RDM4 {
        n,
        data: vec![<T as From<f64>>::from(0.0); n.pow(8)],
    };
    let coeff = (
        (ldet.ca.as_ref(), gdet.ca.as_ref()),
        (ldet.cb.as_ref(), gdet.cb.as_ref()),
    );

    // Evaluate every same-spin transition density once per rank and combine them into each
    // spin-free element over all `2^4` spin assignments.
    let alpha = same_spin_transition_rdms(
        &w.aa,
        (&ldet.excitation.alpha, &gdet.excitation.alpha),
        coeff.0,
        active,
        sa,
        4,
        &mut scratch.aa,
        data.tol,
    );
    let beta = same_spin_transition_rdms(
        &w.bb,
        (&ldet.excitation.beta, &gdet.excitation.beta),
        coeff.1,
        active,
        sb,
        4,
        &mut scratch.bb,
        data.tol,
    );
    let nk = n.pow(4);
    gamma
        .data
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat, value)| {
            let (mut upper, mut lower) = ([0usize; 4], [0usize; 4]);
            let (mut u, mut l) = (flat / nk, flat % nk);
            for k in (0..4).rev() {
                upper[k] = u % n;
                lower[k] = l % n;
                u /= n;
                l /= n;
            }
            *value = det_phase * spin_free_element(&alpha, &beta, n, &upper, &lower);
        });

    (sxw, gamma)
}
