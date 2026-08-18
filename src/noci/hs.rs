// noci/hs.rs
// Crate-root imports.
use crate::basis::excitation_phase;
use crate::nonorthogonalwicks::{
    WickScratchSpin, WicksView, xw_hamiltonian_overlap_prepared,
    xw_hamiltonian_overlap_prepared_batched,
};
use crate::time_call;
use crate::{AoData, DetState};

// Parent/sibling imports.
use super::naive::{build_s_pair, occ_coeffs, one_electron, two_electron_diff, two_electron_same};
use super::overlap::calculate_s_pair_orthogonal;
use super::types::{DetPair, MOCache, NOCIData, NOCIScalar};

/// Wrapper function which dispatches to Hamiltonian and overlap matrix-element evaluation routines
/// depending on user input and properties of the determinant pair involved. If the determinant
/// pair have the same Hermitian-orthonormal parents we may use the standard Slater-Condon rules,
/// if not we can either use generalised Slater-Condon rules or extended non-orthogonal Wick's
/// theorem to evaluate the matrix element.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose Hamiltonian and overlap matrix elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between the determinant pair.
pub(crate) fn calculate_hs_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair, {
        let ldet = pair.ldet;
        let gdet = pair.gdet;

        if ldet.parent == gdet.parent
            && let Some(mocache) = data.mocache
        {
            let cache = &mocache[ldet.parent];
            if cache.orthogonal_slater_condon {
                return calculate_hs_pair_orthogonal(data.ao, cache, ldet, gdet);
            }
        }

        if data.input.wicks.enabled {
            calculate_hs_pair_wicks(
                data.ao,
                ldet,
                gdet,
                data.tol,
                data.wicks.unwrap(),
                scratch.unwrap(),
            )
        } else {
            calculate_hs_pair_naive(data.ao, ldet, gdet, data.tol)
        }
    })
}

/// Calculate batched Hamiltonian and overlap matrix elements using extended nonorthogonal Wick's
/// theorem. The determinant pairs are canonically ordered before this routine is called.
/// Same-parent Slater-Condon cases are handled here; all CPU-specific batching is delegated to the
/// nonorthogonal Wick evaluator.
/// # Arguments:
/// - `data`: Shared real NOCI data with precomputed Wick intermediates.
/// - `pairs`: Canonically ordered determinant-index pairs `(a, b)` with `a <= b`.
/// - `scratch`: Reusable Wick workspace for generic-rank evaluation.
/// - `out`: Hamiltonian and overlap results in the same order as `pairs`.
/// # Returns:
/// - `()`: Writes every requested `(H, S)` pair into `out`.
pub(crate) fn calculate_hs_pairs_wicks_batched(
    data: &NOCIData<'_, f64>,
    pairs: &[(usize, usize)],
    scratch: &mut WickScratchSpin<f64>,
    out: &mut [(f64, f64)],
) {
    let wicks = data.wicks.unwrap();

    // Resolve same-parent matrix elements whose Slater-Condon structure is known before entering
    // the nonorthogonal Wick batching layer.
    for (i, &(a, b)) in pairs.iter().enumerate() {
        let ldet = &data.basis[a];
        let gdet = &data.basis[b];

        if ldet.parent != gdet.parent {
            continue;
        }

        if (ldet.oa ^ gdet.oa).count_ones() + (ldet.ob ^ gdet.ob).count_ones() > 4 {
            out[i] = (0.0, 0.0);
            continue;
        }

        if let Some(mocache) = data.mocache {
            let cache = &mocache[ldet.parent];
            if cache.orthogonal_slater_condon {
                out[i] = calculate_hs_pair_orthogonal(data.ao, cache, ldet, gdet);
            }
        }
    }

    // Each call below owns one ordered reference pair. The evaluator itself decides whether the
    // matching determinant requests are handled by AVX-512, AVX2/FMA, or the scalar path.
    for lp in 0..wicks.nref {
        for gp in 0..wicks.nref {
            if lp == gp
                && let Some(mocache) = data.mocache
                && mocache[lp].orthogonal_slater_condon
            {
                continue;
            }

            let w = wicks.pair(lp, gp);
            xw_hamiltonian_overlap_prepared_batched(
                &w,
                data.basis,
                pairs,
                lp,
                gp,
                data.ao.enuc,
                scratch,
                data.tol,
                out,
            );
        }
    }
}

/// Compare naive and Wick's calculation of matrix elements to ensure consistency.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose matrix elements are to be compared.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `((T, T), (f64, f64))`: Hamiltonian and overlap matrix elements between
///   the determinant pair, total discrepancy between the naive and Wick's path,
///   and max elementwise discrepancy.
pub(in crate::noci) fn compare_hs_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair<'_, T>,
    scratch: &mut WickScratchSpin<T>,
) -> ((T, T), (f64, f64)) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    let (hn, sn) = calculate_hs_pair_naive(data.ao, ldet, gdet, data.tol);
    let (hw, sw) =
        calculate_hs_pair_wicks(data.ao, ldet, gdet, data.tol, data.wicks.unwrap(), scratch);

    let hdiff = (hn - hw).abs();
    let sdiff = (sn - sw).abs();
    ((hw, sw), (hdiff + sdiff, f64::max(hdiff, sdiff)))
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants x and w using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `cache`: MO-basis one and two-electron integral cache for the shared parent determinant.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`.
fn calculate_hs_pair_orthogonal<T: NOCIScalar>(
    ao: &AoData,
    cache: &MOCache<T>,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair_orthogonal, {
        let xa = ldet.oa ^ gdet.oa;
        let xb = ldet.ob ^ gdet.ob;

        let ra = (xa.count_ones() as usize) / 2;
        let rb = (xb.count_ones() as usize) / 2;

        let s = calculate_s_pair_orthogonal(ldet, gdet);

        if ra > 2 || rb > 2 || ra + rb > 2 {
            return (<T as From<f64>>::from(0.0), s);
        }

        let mut holesa = [0usize; 2];
        let mut partsa = [0usize; 2];
        let mut holesb = [0usize; 2];
        let mut partsb = [0usize; 2];

        if ra > 0 {
            let mut bits = gdet.oa & !ldet.oa;
            let mut k = 0;
            while bits != 0 {
                holesa[k] = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                k += 1;
            }

            let mut bits = ldet.oa & !gdet.oa;
            let mut k = 0;
            while bits != 0 {
                partsa[k] = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                k += 1;
            }
        }

        if rb > 0 {
            let mut bits = gdet.ob & !ldet.ob;
            let mut k = 0;
            while bits != 0 {
                holesb[k] = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                k += 1;
            }

            let mut bits = ldet.ob & !gdet.ob;
            let mut k = 0;
            while bits != 0 {
                partsb[k] = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                k += 1;
            }
        }

        let phase = <T as From<f64>>::from(
            excitation_phase(gdet.oa, &holesa[..ra], &partsa[..ra])
                * excitation_phase(gdet.ob, &holesb[..rb], &partsb[..rb]),
        );

        if ra == 0 && rb == 0 {
            let mut h = <T as From<f64>>::from(ao.enuc);

            let mut bits = ldet.oa;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                h += cache.ha[(i, i)];
            }

            let mut bits = ldet.ob;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                h += cache.hb[(i, i)];
            }

            let mut bits_i = ldet.oa;
            while bits_i != 0 {
                let i = bits_i.trailing_zeros() as usize;
                bits_i &= bits_i - 1;

                let mut bits_j = ldet.oa;
                while bits_j != 0 {
                    let j = bits_j.trailing_zeros() as usize;
                    bits_j &= bits_j - 1;
                    h += <T as From<f64>>::from(0.5) * cache.eri_aa_asym[(i, i, j, j)];
                }
            }

            let mut bits_i = ldet.ob;
            while bits_i != 0 {
                let i = bits_i.trailing_zeros() as usize;
                bits_i &= bits_i - 1;

                let mut bits_j = ldet.ob;
                while bits_j != 0 {
                    let j = bits_j.trailing_zeros() as usize;
                    bits_j &= bits_j - 1;
                    h += <T as From<f64>>::from(0.5) * cache.eri_bb_asym[(i, i, j, j)];
                }
            }

            let mut bits_i = ldet.oa;
            while bits_i != 0 {
                let i = bits_i.trailing_zeros() as usize;
                bits_i &= bits_i - 1;

                let mut bits_j = ldet.ob;
                while bits_j != 0 {
                    let j = bits_j.trailing_zeros() as usize;
                    bits_j &= bits_j - 1;
                    h += cache.eri_ab_coul[(i, i, j, j)];
                }
            }

            return (h, s);
        }

        if ra == 1 && rb == 0 {
            let i = holesa[0];
            let a = partsa[0];

            let mut h = cache.ha[(a, i)];

            let mut bits = ldet.oa & gdet.oa;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                h += cache.eri_aa_asym[(a, i, j, j)];
            }

            let mut bits = ldet.ob & gdet.ob;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                h += cache.eri_ab_coul[(a, i, j, j)];
            }

            return (phase * h, s);
        }

        if ra == 0 && rb == 1 {
            let i = holesb[0];
            let a = partsb[0];

            let mut h = cache.hb[(a, i)];

            let mut bits = ldet.ob & gdet.ob;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                h += cache.eri_bb_asym[(a, i, j, j)];
            }

            let mut bits = ldet.oa & gdet.oa;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                h += cache.eri_ab_coul[(j, j, i, a)];
            }

            return (phase * h, s);
        }

        if ra == 2 && rb == 0 {
            let i = holesa[0];
            let j = holesa[1];
            let a = partsa[0];
            let b = partsa[1];
            return (phase * cache.eri_aa_asym[(a, i, j, b)], s);
        }

        if ra == 0 && rb == 2 {
            let i = holesb[0];
            let j = holesb[1];
            let a = partsb[0];
            let b = partsb[1];
            return (phase * cache.eri_bb_asym[(a, i, j, b)], s);
        }

        if ra == 1 && rb == 1 {
            let i = holesa[0];
            let j = holesb[0];
            let a = partsa[0];
            let b = partsb[0];
            return (phase * cache.eri_ab_coul[(a, i, j, b)], s);
        }
        (<T as From<f64>>::from(0.0), s)
    })
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants x and w
/// using generalised Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`.
pub(in crate::noci) fn calculate_hs_pair_naive<T: NOCIScalar>(
    ao: &AoData,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair_naive, {
        // Per spin occupid coefficients.
        let l_ca_occ = occ_coeffs(&ldet.ca, ldet.oa);
        let g_ca_occ = occ_coeffs(&gdet.ca, gdet.oa);
        let l_cb_occ = occ_coeffs(&ldet.cb, ldet.ob);
        let g_cb_occ = occ_coeffs(&gdet.cb, gdet.ob);

        let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
        let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

        // Overlap matrix element for this pair.
        let s = pa.s * pb.s;

        let hnuc = match (pa.zeros.len(), pb.zeros.len()) {
            (0, 0) => <T as From<f64>>::from(ao.enuc) * s,
            _ => <T as From<f64>>::from(0.0),
        };

        let h1a = one_electron(&ao.h, &pa);
        let h1b = one_electron(&ao.h, &pb);
        let h1 = pb.s * h1a + pa.s * h1b;

        let h2aa = pb.s * two_electron_same(&ao.eri_asym, &pa);
        let h2bb = pa.s * two_electron_same(&ao.eri_asym, &pb);
        let h2ab = two_electron_diff(&ao.eri_coul, &pa, &pb);
        let h2 = h2aa + h2bb + h2ab;

        ((hnuc + h1 + h2), s)
    })
}

/// Calculate both the Hamiltonian and overlap matrix elements between
/// determinants x and w using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: Bra-reference state x.
/// - `gdet`: Ket-reference state w.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Precomputed Wick's intermediates.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements for the pair.
pub(in crate::noci) fn calculate_hs_pair_wicks<T: NOCIScalar>(
    ao: &AoData,
    ldet: &DetState<T>,
    gdet: &DetState<T>,
    tol: f64,
    wicks: &WicksView<T>,
    scratch: &mut WickScratchSpin<T>,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair_wicks, {
        let w = wicks.pair(ldet.parent, gdet.parent);
        let excitation_phase = (ldet.pha * gdet.pha) * (ldet.phb * gdet.phb);

        xw_hamiltonian_overlap_prepared(
            &w,
            &ldet.excitation,
            &gdet.excitation,
            &ldet.excitation_cache,
            &gdet.excitation_cache,
            excitation_phase,
            ao.enuc,
            scratch,
            tol,
        )
    })
}
