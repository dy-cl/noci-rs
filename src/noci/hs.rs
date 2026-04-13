// noci/hs.rs
use crate::{AoData, SCFState};
use crate::noci::cache::MOCache;
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use crate::input::Input;

use crate::basis::excitation_phase;
use crate::nonorthogonalwicks::{prepare_same, lg_overlap, lg_h1, lg_h2_same, lg_h2_diff};
use super::naive::{occ_coeffs, build_s_pair, one_electron, two_electron_same, two_electron_diff};
use super::overlap::calculate_s_pair_orthogonal;

/// Wrapper function which dispatches to Hamiltonian and overlap matrix-element evaluation routines
/// depending on user input and properties of the determinant pair involved. If the determinant
/// pair have the same parents we may use the standard Slater-Condon rules, if not we can either
/// use generalised Slater-Condon rules or extended non-orthogonal Wick's theorem to evaluate the
/// matrix element.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// - `tol`: Tolerance for a number being zero.
/// - `input`: User defined input options.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `wicks`: View to the intermediates required for non-orthogonal Wick's theorem.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(f64, f64)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`.
pub fn calculate_hs_pair(ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64, input: &Input, mocache: &[MOCache], 
                         wicks: Option<&WicksView>, scratch: Option<&mut WickScratchSpin>) -> (f64, f64) {
    if ldet.parent == gdet.parent {
        calculate_hs_pair_orthogonal(ao, &mocache[ldet.parent], ldet, gdet)
    } else if input.wicks.enabled {
        calculate_hs_pair_wicks(ao, ldet, gdet, tol, wicks.unwrap(), scratch.unwrap())
    } else {
        calculate_hs_pair_naive(ao, ldet, gdet, tol)
    }
}

/// Compare naive and Wick's calculation of matrix elements to ensure consistency.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// - `tol`: Tolerance for a number being zero.
/// - `wicks`: View to the intermediates required for non-orthogonal Wick's theorem.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `((f64, f64), f64)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`, and
///   the total discrepancy between the naive and Wick's path.
pub(crate) fn compare_hs_pair_wicks_naive(ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64, wicks: &WicksView, scratch: &mut WickScratchSpin) -> ((f64, f64), f64) {
    let (hn, sn) = calculate_hs_pair_naive(ao, ldet, gdet, tol);
    let (hw, sw) = calculate_hs_pair_wicks(ao, ldet, gdet, tol, wicks, scratch);
    ((hw, sw), (hn - hw).abs() + (sn - sw).abs())
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants \Lambda and \Gamma using
/// standard Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `cache`: MO-basis one and two-electron integral cache for the shared parent determinant.
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// # Returns:
/// - `(f64, f64)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`.
pub(crate) fn calculate_hs_pair_orthogonal(ao: &AoData, cache: &MOCache, ldet: &SCFState, gdet: &SCFState) -> (f64, f64) {
    let xa = ldet.oa ^ gdet.oa;
    let xb = ldet.ob ^ gdet.ob;

    let ra = (xa.count_ones() as usize) / 2;
    let rb = (xb.count_ones() as usize) / 2;

    let s = calculate_s_pair_orthogonal(ldet, gdet);

    if ra > 2 || rb > 2 || ra + rb > 2 {
        return (0.0, s);
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

    let phase = excitation_phase(gdet.oa, &holesa[..ra], &partsa[..ra]) * excitation_phase(gdet.ob, &holesb[..rb], &partsb[..rb]);

    if ra == 0 && rb == 0 {
        let mut h = ao.enuc;

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
                h += 0.5 * cache.eri_aa_asym[(i, i, j, j)];
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
                h += 0.5 * cache.eri_bb_asym[(i, i, j, j)];
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
            h += cache.eri_aa_asym[(j, j, a, i)];
        }

        let mut bits = ldet.ob & gdet.ob;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.eri_ab_coul[(i, a, j, j)];
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
            h += cache.eri_bb_asym[(j, j, a, i)];
        }

        let mut bits = ldet.oa & gdet.oa;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.eri_ab_coul[(j, j, a, i)];
        }

        return (phase * h, s);
    }

    if ra == 2 && rb == 0 {
        let i = holesa[0];
        let j = holesa[1];
        let a = partsa[0];
        let b = partsa[1];
        return (phase * cache.eri_aa_asym[(i, a, b, j)], s);
    }

    if ra == 0 && rb == 2 {
        let i = holesb[0];
        let j = holesb[1];
        let a = partsb[0];
        let b = partsb[1];
        return (phase * cache.eri_bb_asym[(i, a, b, j)], s);
    }

    if ra == 1 && rb == 1 {
        let i = holesa[0];
        let j = holesb[0];
        let a = partsa[0];
        let b = partsb[0];
        return (phase * cache.eri_ab_coul[(i, a, b, j)], s);
    }

    (0.0, s)
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants \Lambda and \Gamma 
/// using generalised Slater-Condon rules.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data. 
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// # Returns:
/// - `(f64, f64)`: Hamiltonian and overlap matrix elements between `ldet` and `gdet`.
pub(crate) fn calculate_hs_pair_naive(ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64) -> (f64, f64) {

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
        (0, 0) => ao.enuc * s,
        _ => 0.0,
    };

    let h1a = one_electron(&ao.h, &pa);
    let h1b = one_electron(&ao.h, &pb);
    let h1 = pb.s * h1a + pa.s * h1b;

    let h2aa = pb.s * two_electron_same(&ao.eri_asym, &pa); 
    let h2bb = pa.s * two_electron_same(&ao.eri_asym, &pb); 
    let h2ab = two_electron_diff(&ao.eri_coul, &pa, &pb);
    let h2 = h2aa + h2bb + h2ab;

    (hnuc + h1 + h2, s)
}

/// Calculate both the Hamiltonian and overlap matrix elements between
/// determinants \Lambda and \Gamma using extended non-orthogonal Wick's theorem.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `ldet`: State \Lambda.
/// - `gdet`: State \Gamma.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `wicks`: Precomputed Wick's intermediates.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(f64, f64)`: Hamiltonian and overlap matrix elements for the pair.
pub(crate) fn calculate_hs_pair_wicks(ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64, wicks: &WicksView, scratch: &mut WickScratchSpin) -> (f64, f64) {
    let lp = ldet.parent;
    let gp = gdet.parent;

    let w = wicks.pair(lp, gp);
    let ex_la = &ldet.excitation.alpha;
    let ex_ga = &gdet.excitation.alpha;
    let ex_lb = &ldet.excitation.beta;
    let ex_gb = &gdet.excitation.beta;

    let la = ex_la.holes.len() + ex_ga.holes.len();
    let lb = ex_lb.holes.len() + ex_gb.holes.len();

    let dosa = w.aa.m <= la;
    let doh1a = w.aa.m <= la + 1;
    let doh2aa = w.aa.m <= la + 2;
    let dosb = w.bb.m <= lb;
    let doh1b = w.bb.m <= lb + 1;
    let doh2bb = w.bb.m <= lb + 2;
    let doh2ab = (w.aa.m <= la + 1) && (w.bb.m <= lb + 1);

    let mut sa = 0.0;
    let mut h1a = 0.0;
    let mut h2aa = 0.0;
    let pha = ldet.pha * gdet.pha;

    if dosa || doh1a || doh2aa {
        prepare_same(&w.aa, ex_la, ex_ga, &mut scratch.aa);
        if dosa {
            sa = pha * lg_overlap(&w.aa, ex_la, ex_ga, &mut scratch.aa);
        }
        if doh1a {
            h1a = lg_h1(&w.aa, ex_la, ex_ga, &mut scratch.aa, tol);
        }
        if doh2aa {
            h2aa = lg_h2_same(&w.aa, ex_la, ex_ga, &mut scratch.aa, tol);
        }
    }

    let mut sb = 0.0;
    let mut h1b = 0.0;
    let mut h2bb = 0.0;
    let phb = ldet.phb * gdet.phb;

    if dosb || doh1b || doh2bb {
        prepare_same(&w.bb, ex_lb, ex_gb, &mut scratch.bb);

        if dosb {
            sb = phb * lg_overlap(&w.bb, ex_lb, ex_gb, &mut scratch.bb);
        }
        if doh1b {
            h1b = lg_h1(&w.bb, ex_lb, ex_gb, &mut scratch.bb, tol);
        }
        if doh2bb {
            h2bb = lg_h2_same(&w.bb, ex_lb, ex_gb, &mut scratch.bb, tol);
        }
    }

    let mut h2ab = 0.0;
    if doh2ab {
        h2ab = lg_h2_diff(&w, ex_la, ex_ga, ex_lb, ex_gb, &mut scratch.diff, &scratch.aa, &scratch.bb, tol);
    }

    let s = sa * sb;
    let mut hnuc = 0.0;
    if dosa && dosb && w.aa.m == 0 && w.bb.m == 0 {
        hnuc = ao.enuc * s
    }

    let h1 = pha * h1a * sb + phb * h1b * sa;
    let h2 = (0.5 * pha * sb * h2aa) + (0.5 * phb * sa * h2bb) + (pha * phb * h2ab);

    (hnuc + h1 + h2, s)
}

