// nonorthogonalwicks/eval/rdm.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

// External crate imports.
use ndarray::Array2;
#[cfg(target_arch = "x86_64")]
use num_complex::Complex64;

// Crate-root imports.
use crate::maths::{det, det_const, mix_columns};
use crate::noci::NOCIScalar;
use crate::time_call;
use crate::{Excitation, ExcitationSpin};

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::{SameSpinView, WicksPairView};
use super::dispatch::dispatch_rdm_ranks;
use super::helpers::{extend_rdm_d, for_each_m_combination};
use super::overlap::xw_overlap_prepared;
use super::prepare::construct_determinant_indices;
#[cfg(target_arch = "x86_64")]
use super::simd::{C64x4, C64x8, F64x4, F64x8};

/// Evaluate one unnormalised same-spin rank-`K` transition-density element:
/// `{}^{xw}\Gamma_\sigma{}^{p_1\cdots p_K}_{q_1\cdots q_K}`
/// ` = \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat a^\dagger_{p_1\sigma}\cdots`
/// `\hat a^\dagger_{p_K\sigma}\hat a_{q_K\sigma}\cdots\hat a_{q_1\sigma}`
/// `|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// ` = {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+K}\\m_1+\cdots+m_{L+K} = m}}`
/// `\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(m_1,\ldots,m_{L+K}).`
/// The first `K` contraction columns belong to the external creation-annihilation pairs and the
/// remaining `L = RX + RW` columns belong to the bra and ket excitations. Expanding the determinant
/// generates every fully contracted term with its fermionic sign, while the constrained sum
/// distributes the `m` zero-overlap orbital pairs among the contraction columns.
/// For `K = 0`, the empty external operator string reduces exactly to the prepared overlap.
/// The element is zero when `K > N_\sigma`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `coeff`: Bra- and ket-reference orbital coefficients in the external RDM basis.
/// - `indices`: Const-sized creation indices `\mathbf p` and annihilation indices `\mathbf q`.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// # Returns
/// - `T`: Unnormalised same-spin rank-`K` transition-density element.
#[inline(always)]
pub(crate) fn xw_rdmk_same_prepared<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    indices: (&[usize; K], &[usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_prepared,
        {
            if K > w.nocc {
                return <T as From<f64>>::from(0.0);
            }
            if K == 0 {
                return xw_overlap_prepared(w, ex.0, ex.1, scratch);
            }

            let (l_c, g_c) = coeff;
            let nrdm = l_c.nrows();
            let ext_n = w.nmo + nrdm;
            let x0 = w.x(0);
            let y0 = w.y(0);
            let x0rdm = w.xrdm(0, nrdm);
            let y0rdm = w.yrdm(0, nrdm);
            let x0p = extend_rdm_d(w, &x0, &x0rdm, l_c, g_c).into_raw_vec();
            let y0p = extend_rdm_d(w, &y0, &y0rdm, l_c, g_c).into_raw_vec();
            let one = if w.m == 0 {
                None
            } else {
                let x1 = w.x(1);
                let y1 = w.y(1);
                let x1rdm = w.xrdm(1, nrdm);
                let y1rdm = w.yrdm(1, nrdm);
                Some((
                    extend_rdm_d(w, &x1, &x1rdm, l_c, g_c).into_raw_vec(),
                    extend_rdm_d(w, &y1, &y1rdm, l_c, g_c).into_raw_vec(),
                ))
            };
            let fundamental = (
                x0p.as_slice(),
                y0p.as_slice(),
                one.as_ref()
                    .map(|(x1p, y1p)| (x1p.as_slice(), y1p.as_slice())),
                ext_n,
            );
            let request = (*indices.0, *indices.1);
            xw_rdmk_same_prepared_scalar_value(w, ex, fundamental, &request, scratch, tol)
        }
    )
}

/// Evaluate a batch of unnormalised same-spin rank-`K` transition-density elements.
/// Every request evaluates
/// `{}^{xw}\tilde S\sum_{\sum_i m_i = m}`
/// `\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(m_1,\ldots,m_{L+K})`.
/// For `m = 0`, supported scalar types and ranks use the widest available fixed-rank SIMD kernel;
/// other requests use the scalar const-generic or arbitrary-rank path. The fundamental
/// contractions are transformed to the external RDM basis once for the complete batch.
/// A rank-zero batch is filled by the prepared overlap evaluator without an external-basis transform.
/// A batch is zero when `K > N_\sigma`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `coeff`: Bra- and ket-reference orbital coefficients in the external RDM basis.
/// - `requests`: Creation and annihilation index arrays in output order.
/// - `scratch`: Reusable determinant storage for scalar evaluation.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// - `out`: Same-spin RDM elements in request order.
/// # Returns
/// - `()`: Writes the evaluated requests into `out`.
pub(crate) fn xw_rdmk_same_prepared_batched<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    coeff: (&Array2<T>, &Array2<T>),
    requests: &[([usize; K], [usize; K])],
    scratch: &mut WickScratch<T>,
    tol: f64,
    out: &mut [T],
) {
    if requests.is_empty() || out.is_empty() {
        return;
    }
    if K > w.nocc {
        for value in out.iter_mut().take(requests.len()) {
            *value = <T as From<f64>>::from(0.0);
        }
        return;
    }
    if K == 0 {
        let overlap = xw_overlap_prepared(w, ex.0, ex.1, scratch);
        for value in out.iter_mut().take(requests.len()) {
            *value = overlap;
        }
        return;
    }

    let (l_c, g_c) = coeff;
    let nrdm = l_c.nrows();
    let ext_n = w.nmo + nrdm;
    let x0 = w.x(0);
    let y0 = w.y(0);
    let x0rdm = w.xrdm(0, nrdm);
    let y0rdm = w.yrdm(0, nrdm);
    let x0p = extend_rdm_d(w, &x0, &x0rdm, l_c, g_c).into_raw_vec();
    let y0p = extend_rdm_d(w, &y0, &y0rdm, l_c, g_c).into_raw_vec();

    #[cfg(target_arch = "x86_64")]
    if w.m == 0 && TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            // SAFETY: The `TypeId` check proves `T = f64`, so `out` has the layout required by
            // the real SIMD dispatcher. The dispatcher checks the required CPU features.
            let out_f64 = std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len());
            if try_xw_rdmk_same_prepared_f64_simd(
                w,
                ex,
                (&x0p, &y0p, ext_n),
                requests,
                tol,
                out_f64,
            ) {
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    if w.m == 0 && TypeId::of::<T>() == TypeId::of::<Complex64>() {
        unsafe {
            // SAFETY: The `TypeId` check proves `T = Complex64`, so `out` has the layout required
            // by the complex SIMD dispatcher. The dispatcher checks the required CPU features.
            let out_c64 =
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<Complex64>(), out.len());
            if try_xw_rdmk_same_prepared_c64_simd(
                w,
                ex,
                (&x0p, &y0p, ext_n),
                requests,
                tol,
                out_c64,
            ) {
                return;
            }
        }
    }

    let one = if w.m == 0 {
        None
    } else {
        let x1 = w.x(1);
        let y1 = w.y(1);
        let x1rdm = w.xrdm(1, nrdm);
        let y1rdm = w.yrdm(1, nrdm);
        Some((
            extend_rdm_d(w, &x1, &x1rdm, l_c, g_c).into_raw_vec(),
            extend_rdm_d(w, &y1, &y1rdm, l_c, g_c).into_raw_vec(),
        ))
    };
    let fundamental = (
        x0p.as_slice(),
        y0p.as_slice(),
        one.as_ref()
            .map(|(x1p, y1p)| (x1p.as_slice(), y1p.as_slice())),
        ext_n,
    );
    xw_rdmk_same_prepared_scalar_batch(w, ex, fundamental, requests, scratch, tol, out);
}

/// Try to evaluate a real same-spin rank-`K` RDM batch with fixed-rank SIMD kernels.
/// Each lane evaluates one augmented `m = 0` determinant of dimension `D = K + RX + RW`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, and their row dimension.
/// - `requests`: Creation and annihilation index arrays in output order.
/// - `tol`: Numerical threshold applied to each determinant contribution.
/// - `out`: Real same-spin RDM elements in request order.
/// # Returns
/// - `bool`: Whether a supported SIMD path evaluated the complete batch.
/// # Safety
/// - The caller must prove `T = f64` before `fundamental` is reinterpreted as real storage.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_rdmk_same_prepared_f64_simd<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K])],
    tol: f64,
    out: &mut [f64],
) -> bool {
    let rx = ex.0.holes.count_ones() as usize;
    let rw = ex.1.holes.count_ones() as usize;
    let supported =
        K <= 4 && rx <= 4 && rw <= 4 && (rx == 0 && rw == 0 || (1..=6).contains(&(rx + rw)));
    if !supported {
        return false;
    }

    unsafe {
        if is_x86_feature_detected!("avx512f") {
            let mut start = 0usize;
            let count = requests.len().min(out.len());
            while start < count {
                let lanes = (count - start).min(8);
                let mut packet = [*requests.get_unchecked(start); 8];
                for lane in 1..lanes {
                    packet[lane] = *requests.get_unchecked(start + lane);
                }
                let mut values = [0.0f64; 8];
                dispatch_rdm_ranks!(
                    K,
                    (rx, rw),
                    |K, RX, RW, L, D| xw_rdmk_same_m0_prepared_f64x8_const::<T, K, RX, RW, L, D>(
                        w,
                        ex,
                        fundamental,
                        &*std::ptr::from_ref(&packet).cast::<[([usize; K], [usize; K]); 8]>(),
                        tol,
                        &mut values,
                    ),
                    unreachable!(),
                );
                for lane in 0..lanes {
                    *out.get_unchecked_mut(start + lane) = values[lane];
                }
                start += lanes;
            }
            return true;
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut start = 0usize;
            let count = requests.len().min(out.len());
            while start < count {
                let lanes = (count - start).min(4);
                let mut packet = [*requests.get_unchecked(start); 4];
                for lane in 1..lanes {
                    packet[lane] = *requests.get_unchecked(start + lane);
                }
                let mut values = [0.0f64; 4];
                dispatch_rdm_ranks!(
                    K,
                    (rx, rw),
                    |K, RX, RW, L, D| xw_rdmk_same_m0_prepared_f64x4_const::<T, K, RX, RW, L, D>(
                        w,
                        ex,
                        fundamental,
                        &*std::ptr::from_ref(&packet).cast::<[([usize; K], [usize; K]); 4]>(),
                        tol,
                        &mut values,
                    ),
                    unreachable!(),
                );
                for lane in 0..lanes {
                    *out.get_unchecked_mut(start + lane) = values[lane];
                }
                start += lanes;
            }
            return true;
        }
    }

    false
}

/// Try to evaluate a complex same-spin rank-`K` RDM batch with fixed-rank SIMD kernels.
/// Each lane evaluates one augmented `m = 0` determinant of dimension `D = K + RX + RW`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, and their row dimension.
/// - `requests`: Creation and annihilation index arrays in output order.
/// - `tol`: Numerical threshold applied to each determinant contribution.
/// - `out`: Complex same-spin RDM elements in request order.
/// # Returns
/// - `bool`: Whether a supported SIMD path evaluated the complete batch.
/// # Safety
/// - The caller must prove `T = Complex64` before `fundamental` is reinterpreted as complex storage.
#[cfg(target_arch = "x86_64")]
unsafe fn try_xw_rdmk_same_prepared_c64_simd<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K])],
    tol: f64,
    out: &mut [Complex64],
) -> bool {
    let rx = ex.0.holes.count_ones() as usize;
    let rw = ex.1.holes.count_ones() as usize;
    let supported =
        K <= 4 && rx <= 4 && rw <= 4 && (rx == 0 && rw == 0 || (1..=6).contains(&(rx + rw)));
    if !supported {
        return false;
    }

    unsafe {
        if is_x86_feature_detected!("avx512f") {
            let mut start = 0usize;
            let count = requests.len().min(out.len());
            while start < count {
                let lanes = (count - start).min(8);
                let mut packet = [*requests.get_unchecked(start); 8];
                for lane in 1..lanes {
                    packet[lane] = *requests.get_unchecked(start + lane);
                }
                let mut values = [Complex64::new(0.0, 0.0); 8];
                dispatch_rdm_ranks!(
                    K,
                    (rx, rw),
                    |K, RX, RW, L, D| xw_rdmk_same_m0_prepared_c64x8_const::<T, K, RX, RW, L, D>(
                        w,
                        ex,
                        fundamental,
                        &*std::ptr::from_ref(&packet).cast::<[([usize; K], [usize; K]); 8]>(),
                        tol,
                        &mut values,
                    ),
                    unreachable!(),
                );
                for lane in 0..lanes {
                    *out.get_unchecked_mut(start + lane) = values[lane];
                }
                start += lanes;
            }
            return true;
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut start = 0usize;
            let count = requests.len().min(out.len());
            while start < count {
                let lanes = (count - start).min(4);
                let mut packet = [*requests.get_unchecked(start); 4];
                for lane in 1..lanes {
                    packet[lane] = *requests.get_unchecked(start + lane);
                }
                let mut values = [Complex64::new(0.0, 0.0); 4];
                dispatch_rdm_ranks!(
                    K,
                    (rx, rw),
                    |K, RX, RW, L, D| xw_rdmk_same_m0_prepared_c64x4_const::<T, K, RX, RW, L, D>(
                        w,
                        ex,
                        fundamental,
                        &*std::ptr::from_ref(&packet).cast::<[([usize; K], [usize; K]); 4]>(),
                        tol,
                        &mut values,
                    ),
                    unreachable!(),
                );
                for lane in 0..lanes {
                    *out.get_unchecked_mut(start + lane) = values[lane];
                }
                start += lanes;
            }
            return true;
        }
    }

    false
}

/// Evaluate four real fixed-rank same-spin rank-`K` RDM determinants for `m = 0`.
/// Every SIMD lane computes
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)`
/// with compile-time augmented dimension `D = K + RX + RW`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, and their row dimension.
/// - `requests`: Four creation-annihilation index tuples in SIMD-lane order.
/// - `tol`: Numerical threshold applied to each determinant contribution.
/// - `out`: Four real transition-density elements in SIMD-lane order.
/// # Returns
/// - `()`: Writes four rank-`K` RDM elements into `out`.
/// # Safety
/// - The caller must ensure `T = f64`, CPU support for `AVX2/FMA`, valid external and excitation
///   indices, and compile-time ranks satisfying `D = K + RX + RW` with `D <= 10`.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_rdmk_same_m0_prepared_f64x4_const<
    T: NOCIScalar,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K]); 4],
    tol: f64,
    out: &mut [f64; 4],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_prepared_const,
        {
            unsafe {
                // `pref = p\,{}^{xw}\tilde S` is the phase-weighted reduced overlap.
                let pref = *std::ptr::from_ref(&w.phase).cast::<f64>() * w.tilde_s_prod;
                // For `D = 0`, `\det\mathbf D_{\mathrm{RDM}} = \det\varnothing = 1`.
                if D == 0 {
                    out.fill(pref);
                    return;
                }

                // `x0` and `y0` store the extended `X^{(0)}` and `Y^{(0)}` contractions.
                let (x0, y0, ext_n) = fundamental;
                let x0 = x0.as_ptr().cast::<f64>();
                let y0 = y0.as_ptr().cast::<f64>();
                // Excitation rows are `V_x\cup O_w`; excitation columns are `O_x\cup V_w`.
                let mut excitation_rows = [0usize; L];
                let mut excitation_cols = [0usize; L];
                let nocc = w.nocc;
                let nvirt = w.nmo - nocc;
                let mut x_holes = ex.0.holes;
                let mut x_parts = ex.0.parts;
                for i in 0..RX {
                    excitation_cols[i] = x_holes.trailing_zeros() as usize;
                    excitation_rows[i] = x_parts.trailing_zeros() as usize - nocc;
                    x_holes &= x_holes - 1;
                    x_parts &= x_parts - 1;
                }
                let mut w_holes = ex.1.holes;
                let mut w_parts = ex.1.parts;
                for i in 0..RW {
                    excitation_rows[RX + i] = nvirt + w_holes.trailing_zeros() as usize;
                    excitation_cols[RX + i] = w_parts.trailing_zeros() as usize;
                    w_holes &= w_holes - 1;
                    w_parts &= w_parts - 1;
                }

                // Prepend the external creation labels `\mathbf p` to the excitation rows.
                let row_index = |position: usize, lane: usize| -> usize {
                    if position < K {
                        w.nmo + requests.get_unchecked(lane).0[position]
                    } else {
                        excitation_rows[position - K]
                    }
                };
                // Prepend the external annihilation labels `\mathbf q` to the excitation columns.
                let col_index = |position: usize, lane: usize| -> usize {
                    if position < K {
                        w.nmo + requests.get_unchecked(lane).1[position]
                    } else {
                        excitation_cols[position - K]
                    }
                };
                // `D^{\mathbf p\mathbf q}_{ij} = X^{(0)}_{r_i c_j}` for `i \geq j`, otherwise
                // `D^{\mathbf p\mathbf q}_{ij} = Y^{(0)}_{r_i c_j}`.
                let load_d = |i: usize, j: usize| -> F64x4 {
                    let matrix = if i >= j { x0 } else { y0 };
                    // Broadcast excitation-only entries and gather entries containing external labels.
                    if i >= K && j >= K {
                        F64x4::splat(*matrix.add(row_index(i, 0) * ext_n + col_index(j, 0)))
                    } else {
                        F64x4::from_values(
                            *matrix.add(row_index(i, 0) * ext_n + col_index(j, 0)),
                            *matrix.add(row_index(i, 1) * ext_n + col_index(j, 1)),
                            *matrix.add(row_index(i, 2) * ext_n + col_index(j, 2)),
                            *matrix.add(row_index(i, 3) * ext_n + col_index(j, 3)),
                        )
                    }
                };
                // Construct the packed augmented matrices `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
                let mut d = [F64x4::zero(); 100];
                for i in 0..D {
                    for j in 0..D {
                        d[i * D + j] = load_d(i, j);
                    }
                }

                // For each column subset `S`, evaluate the final-row Laplace recurrence
                // `M_S = \sum_{c \in S}(-1)^{\operatorname{pos}(c,S)}D_{D-|S|,c}M_{S\setminus\{c\}}`.
                // The complete subset gives `M_{\{0,\ldots,D-1\}} = \det\mathbf D_{\mathrm{RDM}}`.
                let full = (1usize << D) - 1;
                let mut minors = [F64x4::zero(); 1024];
                for c in 0..D {
                    minors[1usize << c] = d[(D - 1) * D + c];
                }
                let mut size = 2usize;
                while size <= D {
                    let row = D - size;
                    let mut next = [F64x4::zero(); 1024];
                    let mut mask = full;
                    loop {
                        if mask.count_ones() as usize == size {
                            let mut acc = F64x4::zero();
                            let mut position = 0usize;
                            for c in 0..D {
                                let bit = 1usize << c;
                                if mask & bit != 0 {
                                    let minor = minors[mask ^ bit];
                                    acc = if position & 1 == 0 {
                                        F64x4::madd(acc, d[row * D + c], minor)
                                    } else {
                                        F64x4::msub(acc, d[row * D + c], minor)
                                    };
                                    position += 1;
                                }
                            }
                            next[mask] = acc;
                        }
                        if mask == 0 {
                            break;
                        }
                        mask = (mask - 1) & full;
                    }
                    minors = next;
                    size += 1;
                }

                // Extract `\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}` for every lane.
                let mut determinant = [0.0f64; 4];
                minors[full].store(&mut determinant);
                for lane in 0..4 {
                    // `\Gamma^{\mathbf p}_{\mathbf q} = pref\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
                    out[lane] = if determinant[lane].abs() > tol {
                        pref * determinant[lane]
                    } else {
                        0.0
                    };
                }
            }
        }
    )
}

/// Evaluate four complex fixed-rank same-spin rank-`K` RDM determinants for `m = 0`.
/// Every SIMD lane computes
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)`
/// with compile-time augmented dimension `D = K + RX + RW`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, and their row dimension.
/// - `requests`: Four creation-annihilation index tuples in SIMD-lane order.
/// - `tol`: Numerical threshold applied to each determinant contribution.
/// - `out`: Four complex transition-density elements in SIMD-lane order.
/// # Returns
/// - `()`: Writes four rank-`K` RDM elements into `out`.
/// # Safety
/// - The caller must ensure `T = Complex64`, CPU support for `AVX2/FMA`, valid external and
///   excitation indices, and compile-time ranks satisfying `D = K + RX + RW` with `D <= 10`.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_rdmk_same_m0_prepared_c64x4_const<
    T: NOCIScalar,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K]); 4],
    tol: f64,
    out: &mut [Complex64; 4],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_prepared_const,
        {
            unsafe {
                // `pref = p\,{}^{xw}\tilde S` is the phase-weighted reduced overlap.
                let phase = *std::ptr::from_ref(&w.phase).cast::<Complex64>();
                let pref = phase * w.tilde_s_prod;
                // For `D = 0`, `\det\mathbf D_{\mathrm{RDM}} = \det\varnothing = 1`.
                if D == 0 {
                    out.fill(pref);
                    return;
                }

                // `x0` and `y0` store the extended `X^{(0)}` and `Y^{(0)}` contractions.
                let (x0, y0, ext_n) = fundamental;
                let x0 = x0.as_ptr().cast::<Complex64>();
                let y0 = y0.as_ptr().cast::<Complex64>();
                // Excitation rows are `V_x\cup O_w`; excitation columns are `O_x\cup V_w`.
                let mut excitation_rows = [0usize; L];
                let mut excitation_cols = [0usize; L];
                let nocc = w.nocc;
                let nvirt = w.nmo - nocc;
                let mut x_holes = ex.0.holes;
                let mut x_parts = ex.0.parts;
                for i in 0..RX {
                    excitation_cols[i] = x_holes.trailing_zeros() as usize;
                    excitation_rows[i] = x_parts.trailing_zeros() as usize - nocc;
                    x_holes &= x_holes - 1;
                    x_parts &= x_parts - 1;
                }
                let mut w_holes = ex.1.holes;
                let mut w_parts = ex.1.parts;
                for i in 0..RW {
                    excitation_rows[RX + i] = nvirt + w_holes.trailing_zeros() as usize;
                    excitation_cols[RX + i] = w_parts.trailing_zeros() as usize;
                    w_holes &= w_holes - 1;
                    w_parts &= w_parts - 1;
                }

                // Prepend the external creation labels `\mathbf p` to the excitation rows.
                let row_index = |position: usize, lane: usize| -> usize {
                    if position < K {
                        w.nmo + requests.get_unchecked(lane).0[position]
                    } else {
                        excitation_rows[position - K]
                    }
                };
                // Prepend the external annihilation labels `\mathbf q` to the excitation columns.
                let col_index = |position: usize, lane: usize| -> usize {
                    if position < K {
                        w.nmo + requests.get_unchecked(lane).1[position]
                    } else {
                        excitation_cols[position - K]
                    }
                };
                // `D^{\mathbf p\mathbf q}_{ij} = X^{(0)}_{r_i c_j}` for `i \geq j`, otherwise
                // `D^{\mathbf p\mathbf q}_{ij} = Y^{(0)}_{r_i c_j}`.
                let load_d = |i: usize, j: usize| -> C64x4 {
                    let matrix = if i >= j { x0 } else { y0 };
                    // Broadcast excitation-only entries and gather entries containing external labels.
                    if i >= K && j >= K {
                        let value = *matrix.add(row_index(i, 0) * ext_n + col_index(j, 0));
                        C64x4::splat(value.re, value.im)
                    } else {
                        C64x4::from_values(
                            *matrix.add(row_index(i, 0) * ext_n + col_index(j, 0)),
                            *matrix.add(row_index(i, 1) * ext_n + col_index(j, 1)),
                            *matrix.add(row_index(i, 2) * ext_n + col_index(j, 2)),
                            *matrix.add(row_index(i, 3) * ext_n + col_index(j, 3)),
                        )
                    }
                };
                // Construct the packed augmented matrices `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
                let mut d = [C64x4::zero(); 100];
                for i in 0..D {
                    for j in 0..D {
                        d[i * D + j] = load_d(i, j);
                    }
                }

                // For each column subset `S`, evaluate the final-row Laplace recurrence
                // `M_S = \sum_{c \in S}(-1)^{\operatorname{pos}(c,S)}D_{D-|S|,c}M_{S\setminus\{c\}}`.
                // The complete subset gives `M_{\{0,\ldots,D-1\}} = \det\mathbf D_{\mathrm{RDM}}`.
                let full = (1usize << D) - 1;
                let mut minors = [C64x4::zero(); 1024];
                for c in 0..D {
                    minors[1usize << c] = d[(D - 1) * D + c];
                }
                let mut size = 2usize;
                while size <= D {
                    let row = D - size;
                    let mut next = [C64x4::zero(); 1024];
                    let mut mask = full;
                    loop {
                        if mask.count_ones() as usize == size {
                            let mut acc = C64x4::zero();
                            let mut position = 0usize;
                            for c in 0..D {
                                let bit = 1usize << c;
                                if mask & bit != 0 {
                                    let minor = minors[mask ^ bit];
                                    acc = if position & 1 == 0 {
                                        C64x4::madd(acc, d[row * D + c], minor)
                                    } else {
                                        C64x4::msub(acc, d[row * D + c], minor)
                                    };
                                    position += 1;
                                }
                            }
                            next[mask] = acc;
                        }
                        if mask == 0 {
                            break;
                        }
                        mask = (mask - 1) & full;
                    }
                    minors = next;
                    size += 1;
                }

                // Extract `\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}` for every lane.
                let mut re = [0.0f64; 4];
                let mut im = [0.0f64; 4];
                minors[full].store(&mut re, &mut im);
                for lane in 0..4 {
                    let determinant = Complex64::new(re[lane], im[lane]);
                    // `\Gamma^{\mathbf p}_{\mathbf q} = pref\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
                    out[lane] = if determinant.norm() > tol {
                        pref * determinant
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                }
            }
        }
    )
}

/// Evaluate eight real fixed-rank same-spin rank-`K` RDM determinants for `m = 0`.
/// Every SIMD lane computes
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)`
/// with compile-time augmented dimension `D = K + RX + RW`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, and their row dimension.
/// - `requests`: Eight creation-annihilation index tuples in SIMD-lane order.
/// - `tol`: Numerical threshold applied to each determinant contribution.
/// - `out`: Eight real transition-density elements in SIMD-lane order.
/// # Returns
/// - `()`: Writes eight rank-`K` RDM elements into `out`.
/// # Safety
/// - The caller must ensure `T = f64`, CPU support for `AVX-512`, valid external and excitation
///   indices, and compile-time ranks satisfying `D = K + RX + RW` with `D <= 10`.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx512f")]
unsafe fn xw_rdmk_same_m0_prepared_f64x8_const<
    T: NOCIScalar,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K]); 8],
    tol: f64,
    out: &mut [f64; 8],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_prepared_const,
        {
            unsafe {
                // `pref = p\,{}^{xw}\tilde S` is the phase-weighted reduced overlap.
                let pref = *std::ptr::from_ref(&w.phase).cast::<f64>() * w.tilde_s_prod;
                // For `D = 0`, `\det\mathbf D_{\mathrm{RDM}} = \det\varnothing = 1`.
                if D == 0 {
                    out.fill(pref);
                    return;
                }

                // `x0` and `y0` store the extended `X^{(0)}` and `Y^{(0)}` contractions.
                let (x0, y0, ext_n) = fundamental;
                let x0 = x0.as_ptr().cast::<f64>();
                let y0 = y0.as_ptr().cast::<f64>();
                // Excitation rows are `V_x\cup O_w`; excitation columns are `O_x\cup V_w`.
                let mut excitation_rows = [0usize; L];
                let mut excitation_cols = [0usize; L];
                let nocc = w.nocc;
                let nvirt = w.nmo - nocc;
                let mut x_holes = ex.0.holes;
                let mut x_parts = ex.0.parts;
                for i in 0..RX {
                    excitation_cols[i] = x_holes.trailing_zeros() as usize;
                    excitation_rows[i] = x_parts.trailing_zeros() as usize - nocc;
                    x_holes &= x_holes - 1;
                    x_parts &= x_parts - 1;
                }
                let mut w_holes = ex.1.holes;
                let mut w_parts = ex.1.parts;
                for i in 0..RW {
                    excitation_rows[RX + i] = nvirt + w_holes.trailing_zeros() as usize;
                    excitation_cols[RX + i] = w_parts.trailing_zeros() as usize;
                    w_holes &= w_holes - 1;
                    w_parts &= w_parts - 1;
                }

                // Prepend the external creation labels `\mathbf p` to the excitation rows.
                let row_index = |position: usize, lane: usize| -> usize {
                    if position < K {
                        w.nmo + requests.get_unchecked(lane).0[position]
                    } else {
                        excitation_rows[position - K]
                    }
                };
                // Prepend the external annihilation labels `\mathbf q` to the excitation columns.
                let col_index = |position: usize, lane: usize| -> usize {
                    if position < K {
                        w.nmo + requests.get_unchecked(lane).1[position]
                    } else {
                        excitation_cols[position - K]
                    }
                };
                // `D^{\mathbf p\mathbf q}_{ij} = X^{(0)}_{r_i c_j}` for `i \geq j`, otherwise
                // `D^{\mathbf p\mathbf q}_{ij} = Y^{(0)}_{r_i c_j}`.
                let load_d = |i: usize, j: usize| -> F64x8 {
                    let matrix = if i >= j { x0 } else { y0 };
                    // Broadcast excitation-only entries and gather entries containing external labels.
                    if i >= K && j >= K {
                        F64x8::splat(*matrix.add(row_index(i, 0) * ext_n + col_index(j, 0)))
                    } else {
                        F64x8::from_values([
                            *matrix.add(row_index(i, 0) * ext_n + col_index(j, 0)),
                            *matrix.add(row_index(i, 1) * ext_n + col_index(j, 1)),
                            *matrix.add(row_index(i, 2) * ext_n + col_index(j, 2)),
                            *matrix.add(row_index(i, 3) * ext_n + col_index(j, 3)),
                            *matrix.add(row_index(i, 4) * ext_n + col_index(j, 4)),
                            *matrix.add(row_index(i, 5) * ext_n + col_index(j, 5)),
                            *matrix.add(row_index(i, 6) * ext_n + col_index(j, 6)),
                            *matrix.add(row_index(i, 7) * ext_n + col_index(j, 7)),
                        ])
                    }
                };
                // Construct the packed augmented matrices `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
                let mut d = [F64x8::zero(); 100];
                for i in 0..D {
                    for j in 0..D {
                        d[i * D + j] = load_d(i, j);
                    }
                }

                // For each column subset `S`, evaluate the final-row Laplace recurrence
                // `M_S = \sum_{c \in S}(-1)^{\operatorname{pos}(c,S)}D_{D-|S|,c}M_{S\setminus\{c\}}`.
                // The complete subset gives `M_{\{0,\ldots,D-1\}} = \det\mathbf D_{\mathrm{RDM}}`.
                let full = (1usize << D) - 1;
                let mut minors = [F64x8::zero(); 1024];
                for c in 0..D {
                    minors[1usize << c] = d[(D - 1) * D + c];
                }
                let mut size = 2usize;
                while size <= D {
                    let row = D - size;
                    let mut next = [F64x8::zero(); 1024];
                    let mut mask = full;
                    loop {
                        if mask.count_ones() as usize == size {
                            let mut acc = F64x8::zero();
                            let mut position = 0usize;
                            for c in 0..D {
                                let bit = 1usize << c;
                                if mask & bit != 0 {
                                    let minor = minors[mask ^ bit];
                                    acc = if position & 1 == 0 {
                                        F64x8::madd(acc, d[row * D + c], minor)
                                    } else {
                                        F64x8::msub(acc, d[row * D + c], minor)
                                    };
                                    position += 1;
                                }
                            }
                            next[mask] = acc;
                        }
                        if mask == 0 {
                            break;
                        }
                        mask = (mask - 1) & full;
                    }
                    minors = next;
                    size += 1;
                }

                // Extract `\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}` for every lane.
                let mut determinant = [0.0f64; 8];
                minors[full].store(&mut determinant);
                for lane in 0..8 {
                    // `\Gamma^{\mathbf p}_{\mathbf q} = pref\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
                    out[lane] = if determinant[lane].abs() > tol {
                        pref * determinant[lane]
                    } else {
                        0.0
                    };
                }
            }
        }
    )
}

/// Evaluate eight complex fixed-rank same-spin rank-`K` RDM determinants for `m = 0`.
/// Every SIMD lane computes
/// `{}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)`
/// with compile-time augmented dimension `D = K + RX + RW`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = Complex64` and `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, and their row dimension.
/// - `requests`: Eight creation-annihilation index tuples in SIMD-lane order.
/// - `tol`: Numerical threshold applied to each determinant contribution.
/// - `out`: Eight complex transition-density elements in SIMD-lane order.
/// # Returns
/// - `()`: Writes eight rank-`K` RDM elements into `out`.
/// # Safety
/// - The caller must ensure `T = Complex64`, CPU support for `AVX-512`, valid external and
///   excitation indices, and compile-time ranks satisfying `D = K + RX + RW` with `D <= 10`.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx512f")]
unsafe fn xw_rdmk_same_m0_prepared_c64x8_const<
    T: NOCIScalar,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K]); 8],
    tol: f64,
    out: &mut [Complex64; 8],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_prepared_const,
        {
            unsafe {
                // `pref = p\,{}^{xw}\tilde S` is the phase-weighted reduced overlap.
                let phase = *std::ptr::from_ref(&w.phase).cast::<Complex64>();
                let pref = phase * w.tilde_s_prod;
                // For `D = 0`, `\det\mathbf D_{\mathrm{RDM}} = \det\varnothing = 1`.
                if D == 0 {
                    out.fill(pref);
                    return;
                }

                // `x0` and `y0` store the extended `X^{(0)}` and `Y^{(0)}` contractions.
                let (x0, y0, ext_n) = fundamental;
                let x0 = x0.as_ptr().cast::<Complex64>();
                let y0 = y0.as_ptr().cast::<Complex64>();
                // Excitation rows are `V_x\cup O_w`; excitation columns are `O_x\cup V_w`.
                let mut excitation_rows = [0usize; L];
                let mut excitation_cols = [0usize; L];
                let nocc = w.nocc;
                let nvirt = w.nmo - nocc;
                let mut x_holes = ex.0.holes;
                let mut x_parts = ex.0.parts;
                for i in 0..RX {
                    excitation_cols[i] = x_holes.trailing_zeros() as usize;
                    excitation_rows[i] = x_parts.trailing_zeros() as usize - nocc;
                    x_holes &= x_holes - 1;
                    x_parts &= x_parts - 1;
                }
                let mut w_holes = ex.1.holes;
                let mut w_parts = ex.1.parts;
                for i in 0..RW {
                    excitation_rows[RX + i] = nvirt + w_holes.trailing_zeros() as usize;
                    excitation_cols[RX + i] = w_parts.trailing_zeros() as usize;
                    w_holes &= w_holes - 1;
                    w_parts &= w_parts - 1;
                }

                // Prepend the external creation labels `\mathbf p` to the excitation rows.
                let row_index = |position: usize, lane: usize| -> usize {
                    if position < K {
                        w.nmo + requests.get_unchecked(lane).0[position]
                    } else {
                        excitation_rows[position - K]
                    }
                };
                // Prepend the external annihilation labels `\mathbf q` to the excitation columns.
                let col_index = |position: usize, lane: usize| -> usize {
                    if position < K {
                        w.nmo + requests.get_unchecked(lane).1[position]
                    } else {
                        excitation_cols[position - K]
                    }
                };
                // `D^{\mathbf p\mathbf q}_{ij} = X^{(0)}_{r_i c_j}` for `i \geq j`, otherwise
                // `D^{\mathbf p\mathbf q}_{ij} = Y^{(0)}_{r_i c_j}`.
                let load_d = |i: usize, j: usize| -> C64x8 {
                    let matrix = if i >= j { x0 } else { y0 };
                    // Broadcast excitation-only entries and gather entries containing external labels.
                    if i >= K && j >= K {
                        let value = *matrix.add(row_index(i, 0) * ext_n + col_index(j, 0));
                        C64x8::splat(value.re, value.im)
                    } else {
                        C64x8::from_values([
                            *matrix.add(row_index(i, 0) * ext_n + col_index(j, 0)),
                            *matrix.add(row_index(i, 1) * ext_n + col_index(j, 1)),
                            *matrix.add(row_index(i, 2) * ext_n + col_index(j, 2)),
                            *matrix.add(row_index(i, 3) * ext_n + col_index(j, 3)),
                            *matrix.add(row_index(i, 4) * ext_n + col_index(j, 4)),
                            *matrix.add(row_index(i, 5) * ext_n + col_index(j, 5)),
                            *matrix.add(row_index(i, 6) * ext_n + col_index(j, 6)),
                            *matrix.add(row_index(i, 7) * ext_n + col_index(j, 7)),
                        ])
                    }
                };
                // Construct the packed augmented matrices `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
                let mut d = [C64x8::zero(); 100];
                for i in 0..D {
                    for j in 0..D {
                        d[i * D + j] = load_d(i, j);
                    }
                }

                // For each column subset `S`, evaluate the final-row Laplace recurrence
                // `M_S = \sum_{c \in S}(-1)^{\operatorname{pos}(c,S)}D_{D-|S|,c}M_{S\setminus\{c\}}`.
                // The complete subset gives `M_{\{0,\ldots,D-1\}} = \det\mathbf D_{\mathrm{RDM}}`.
                let full = (1usize << D) - 1;
                let mut minors = [C64x8::zero(); 1024];
                for c in 0..D {
                    minors[1usize << c] = d[(D - 1) * D + c];
                }
                let mut size = 2usize;
                while size <= D {
                    let row = D - size;
                    let mut next = [C64x8::zero(); 1024];
                    let mut mask = full;
                    loop {
                        if mask.count_ones() as usize == size {
                            let mut acc = C64x8::zero();
                            let mut position = 0usize;
                            for c in 0..D {
                                let bit = 1usize << c;
                                if mask & bit != 0 {
                                    let minor = minors[mask ^ bit];
                                    acc = if position & 1 == 0 {
                                        C64x8::madd(acc, d[row * D + c], minor)
                                    } else {
                                        C64x8::msub(acc, d[row * D + c], minor)
                                    };
                                    position += 1;
                                }
                            }
                            next[mask] = acc;
                        }
                        if mask == 0 {
                            break;
                        }
                        mask = (mask - 1) & full;
                    }
                    minors = next;
                    size += 1;
                }

                // Extract `\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}` for every lane.
                let mut re = [0.0f64; 8];
                let mut im = [0.0f64; 8];
                minors[full].store(&mut re, &mut im);
                for lane in 0..8 {
                    let determinant = Complex64::new(re[lane], im[lane]);
                    // `\Gamma^{\mathbf p}_{\mathbf q} = pref\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
                    out[lane] = if determinant.norm() > tol {
                        pref * determinant
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                }
            }
        }
    )
}

/// Evaluate a same-spin rank-`K` RDM request batch through the scalar prepared path.
/// Each request is the constrained determinant sum
/// `{}^{xw}\tilde S\sum_{\sum_i m_i = m}\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, optional `m_i = 1` branches, and matrix rank.
/// - `requests`: Creation and annihilation index arrays in output order.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// - `out`: Same-spin RDM elements in request order.
/// # Returns
/// - `()`: Writes the evaluated requests into `out`.
#[allow(clippy::type_complexity)]
fn xw_rdmk_same_prepared_scalar_batch<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], Option<(&[T], &[T])>, usize),
    requests: &[([usize; K], [usize; K])],
    scratch: &mut WickScratch<T>,
    tol: f64,
    out: &mut [T],
) {
    for (request, value) in requests.iter().zip(out.iter_mut()) {
        *value = xw_rdmk_same_prepared_scalar_value(w, ex, fundamental, request, scratch, tol);
    }
}

/// Evaluate one same-spin rank-`K` RDM request through the scalar prepared path.
/// The `m = 0` branch dispatches `(K,RX,RW,L,D)` to a const-generic determinant; the general
/// branch sums all binary contraction-column assignments satisfying `\sum_i m_i = m`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, optional `m_i = 1` branches, and matrix rank.
/// - `request`: Const-sized creation and annihilation index arrays.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// # Returns
/// - `T`: Unnormalised same-spin rank-`K` transition-density element.
#[allow(clippy::type_complexity)]
#[inline(always)]
fn xw_rdmk_same_prepared_scalar_value<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], Option<(&[T], &[T])>, usize),
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    let (x_ex, w_ex) = ex;
    let rx = x_ex.holes.count_ones() as usize;
    let rw = w_ex.holes.count_ones() as usize;
    let l = rx + rw;
    if w.m > l + K {
        return <T as From<f64>>::from(0.0);
    }
    if w.m == 0 {
        xw_rdmk_same_m0_prepared(w, ex, fundamental, request, scratch, tol)
    } else {
        xw_rdmk_same_gen_prepared(w, ex, fundamental, request, scratch, tol)
    }
}

/// Evaluate one same-spin rank-`K` RDM element when every contraction carries `m_i = 0`:
/// `{}^{xw}\Gamma_\sigma{}^{\mathbf p}_{\mathbf q}`
/// ` = {}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)`.
/// Supported `(K,RX,RW)` tuples dispatch to the const-generic determinant of dimension
/// `D = K + RX + RW`; arbitrary ranks use the scalar generic fallback.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, unused branch marker, and matrix rank.
/// - `request`: Const-sized creation and annihilation index arrays.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to the determinant contribution.
/// # Returns
/// - `T`: Unnormalised same-spin rank-`K` transition-density element for `m = 0`.
#[allow(clippy::type_complexity)]
#[inline(always)]
fn xw_rdmk_same_m0_prepared<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], Option<(&[T], &[T])>, usize),
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_prepared,
        {
            let rx = ex.0.holes.count_ones() as usize;
            let rw = ex.1.holes.count_ones() as usize;
            dispatch_rdm_ranks!(
                K,
                (rx, rw),
                |K, RX, RW, L, D| xw_rdmk_same_m0_prepared_const::<T, K, RX, RW, L, D>(
                    w,
                    ex,
                    (fundamental.0, fundamental.1, fundamental.3),
                    // SAFETY: `dispatch_rdm_ranks!` selects this arm only when the caller's const
                    // `K` equals the arm-local literal `K`, so the two array-reference layouts are
                    // identical.
                    unsafe { &*std::ptr::from_ref(request).cast::<([usize; K], [usize; K])>() },
                    scratch,
                    tol,
                ),
                xw_rdmk_same_m0_gen_prepared(
                    w,
                    ex,
                    (fundamental.0, fundamental.1, fundamental.3),
                    request,
                    scratch,
                    tol,
                ),
            )
        }
    )
}

/// Evaluate one fixed-rank same-spin rank-`K` RDM determinant for `m = 0`.
/// The augmented matrix has dimension `D = K + L`, with `X^{(0)}` on and below the diagonal and
/// `Y^{(0)}` above it. Its first `K` labels are external RDM indices; its remaining labels use the
/// contraction-space ordering `V_x \cup O_w` by `O_x \cup V_w` for ranks `(RX,RW)`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, and their row dimension.
/// - `request`: Const-sized creation and annihilation index arrays.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to the determinant contribution.
/// # Returns
/// - `T`: `{}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
#[inline(always)]
fn xw_rdmk_same_m0_prepared_const<
    T: NOCIScalar,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_prepared_const,
        {
            scratch.ensure_same(D);
            let rows = scratch.rows.as_mut_slice();
            let cols = scratch.cols.as_mut_slice();
            for i in 0..K {
                rows[i] = w.nmo + request.0[i];
                cols[i] = w.nmo + request.1[i];
            }

            let (x_ex, w_ex) = ex;
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;
            let mut x_holes = x_ex.holes;
            let mut x_parts = x_ex.parts;
            for i in 0..RX {
                cols[K + i] = x_holes.trailing_zeros() as usize;
                rows[K + i] = x_parts.trailing_zeros() as usize - nocc;
                x_holes &= x_holes - 1;
                x_parts &= x_parts - 1;
            }
            let mut w_holes = w_ex.holes;
            let mut w_parts = w_ex.parts;
            for i in 0..RW {
                rows[K + RX + i] = nvirt + w_holes.trailing_zeros() as usize;
                cols[K + RX + i] = w_parts.trailing_zeros() as usize;
                w_holes &= w_holes - 1;
                w_parts &= w_parts - 1;
            }

            let (x0, y0, ext_n) = fundamental;
            let d = scratch.det0.as_mut_slice();
            for i in 0..D {
                let row = rows[i] * ext_n;
                for j in 0..D {
                    d[i * D + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            let zero = <T as From<f64>>::from(0.0);
            if let Some(value) = det_const::<T, D>(d)
                && value.abs() > tol
            {
                w.phase * <T as From<f64>>::from(w.tilde_s_prod) * value
            } else {
                zero
            }
        }
    )
}

/// Evaluate one same-spin rank-`K` `m = 0` element outside the const-dispatch table.
/// This computes `{}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}` with runtime
/// augmented dimension `D = K + RX + RW` and the same `X`-lower/`Y`-upper convention.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended `X^{(0)}`, `Y^{(0)}`, and their row dimension.
/// - `request`: Const-sized creation and annihilation index arrays.
/// - `scratch`: Reusable determinant storage.
/// - `tol`: Numerical threshold applied to the determinant contribution.
/// # Returns
/// - `T`: Unnormalised same-spin rank-`K` transition-density element for `m = 0`.
#[inline(always)]
fn xw_rdmk_same_m0_gen_prepared<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_gen_prepared,
        {
            let l = ex.0.holes.count_ones() as usize + ex.1.holes.count_ones() as usize;
            let d_rank = K + l;
            scratch.ensure_same(d_rank);
            let rows = scratch.rows.as_mut_slice();
            let cols = scratch.cols.as_mut_slice();
            for i in 0..K {
                rows[i] = w.nmo + request.0[i];
                cols[i] = w.nmo + request.1[i];
            }
            construct_determinant_indices(ex.0, ex.1, w, &mut rows[K..], &mut cols[K..]);

            let (x0, y0, ext_n) = fundamental;
            let d = scratch.det0.as_mut_slice();
            for i in 0..d_rank {
                let row = rows[i] * ext_n;
                for j in 0..d_rank {
                    d[i * d_rank + j] = if i >= j {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            let zero = <T as From<f64>>::from(0.0);
            if let Some(value) = det(d, d_rank)
                && value.abs() > tol
            {
                w.phase * <T as From<f64>>::from(w.tilde_s_prod) * value
            } else {
                zero
            }
        }
    )
}

/// Evaluate one same-spin rank-`K` RDM element for `m > 0`.
/// For augmented rank `D = K + L`, this sums every determinant obtained by selecting column `i`
/// from the `m_i = 0` or `m_i = 1` endpoint according to each binary assignment satisfying
/// `m_1 + \cdots + m_D = m`, then multiplies by `{}^{xw}\tilde S`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates containing zero-overlap orbital pairs.
/// - `ex`: Excitations defining the bra and ket determinants respectively.
/// - `fundamental`: Extended endpoint contractions and their row dimension.
/// - `request`: Const-sized creation and annihilation index arrays.
/// - `scratch`: Reusable endpoint and mixed-determinant storage.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// # Returns
/// - `T`: Constrained determinant sum multiplied by the reduced reference overlap.
#[allow(clippy::type_complexity)]
#[inline(always)]
fn xw_rdmk_same_gen_prepared<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], Option<(&[T], &[T])>, usize),
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_gen_prepared,
        {
            let l = ex.0.holes.count_ones() as usize + ex.1.holes.count_ones() as usize;
            let d_rank = K + l;
            scratch.ensure_same(d_rank);
            let rows = scratch.rows.as_mut_slice();
            let cols = scratch.cols.as_mut_slice();
            for i in 0..K {
                rows[i] = w.nmo + request.0[i];
                cols[i] = w.nmo + request.1[i];
            }
            construct_determinant_indices(ex.0, ex.1, w, &mut rows[K..], &mut cols[K..]);

            let (x0, y0, one, ext_n) = fundamental;
            let (x1, y1) = one.unwrap_or((x0, y0));
            for i in 0..d_rank {
                let row = rows[i] * ext_n;
                for j in 0..d_rank {
                    let index = i * d_rank + j;
                    if i >= j {
                        scratch.det0.as_mut_slice()[index] = x0[row + cols[j]];
                        scratch.det1.as_mut_slice()[index] = x1[row + cols[j]];
                    } else {
                        scratch.det0.as_mut_slice()[index] = y0[row + cols[j]];
                        scratch.det1.as_mut_slice()[index] = y1[row + cols[j]];
                    }
                }
            }

            let zero = <T as From<f64>>::from(0.0);
            let mut acc = zero;
            for_each_m_combination(d_rank, w.m, |bits| {
                mix_columns(
                    scratch.det_mix.as_mut_slice(),
                    scratch.det0.as_slice(),
                    scratch.det1.as_slice(),
                    d_rank,
                    bits,
                );
                if let Some(value) = det(scratch.det_mix.as_slice(), d_rank)
                    && value.abs() > tol
                {
                    acc += value;
                }
            });
            w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
        }
    )
}

/// Evaluate one different-spin rank-`(KA,KB)` transition-density contribution.
/// Operators of different spin commute after an even fermionic permutation, and the determinant
/// product state separates into spin sectors, giving
/// `{}^{xw}\Gamma_{\alpha\beta}^{\mathbf p_\alpha\mathbf p_\beta}`
/// `{}_{\mathbf q_\alpha\mathbf q_\beta}`
/// ` = {}^{xw}\Gamma_\alpha^{\mathbf p_\alpha}{}_{\mathbf q_\alpha}`
/// `{}^{xw}\Gamma_\beta^{\mathbf p_\beta}{}_{\mathbf q_\beta}`.
/// Both factors use the same rank-`K` same-spin determinant evaluator.
/// The contribution is zero when `KA > N_\alpha` or `KB > N_\beta`.
/// # Arguments:
/// - `w`: Alpha-, beta-, and different-spin intermediates for one reference pair.
/// - `ex`: Bra and ket excitations containing both spin sectors.
/// - `coeff`: Alpha and beta pairs of bra- and ket-reference orbital coefficients.
/// - `indices`: Alpha and beta creation-annihilation index pairs.
/// - `scratch`: Reusable alpha- and beta-spin determinant storage.
/// - `tol`: Numerical threshold applied to individual determinant contributions.
/// # Returns
/// - `T`: Product of the unnormalised alpha- and beta-spin transition-density elements.
#[allow(clippy::type_complexity)]
pub(crate) fn xw_rdmk_diff_prepared<T: NOCIScalar, const KA: usize, const KB: usize>(
    w: &WicksPairView<'_, T>,
    ex: (&Excitation, &Excitation),
    coeff: ((&Array2<T>, &Array2<T>), (&Array2<T>, &Array2<T>)),
    indices: ((&[usize; KA], &[usize; KA]), (&[usize; KB], &[usize; KB])),
    scratch: (&mut WickScratch<T>, &mut WickScratch<T>),
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_diff_prepared,
        {
            if KA > w.aa.nocc || KB > w.bb.nocc {
                return <T as From<f64>>::from(0.0);
            }

            let alpha = xw_rdmk_same_prepared(
                &w.aa,
                (&ex.0.alpha, &ex.1.alpha),
                coeff.0,
                indices.0,
                scratch.0,
                tol,
            );
            let beta = xw_rdmk_same_prepared(
                &w.bb,
                (&ex.0.beta, &ex.1.beta),
                coeff.1,
                indices.1,
                scratch.1,
                tol,
            );
            alpha * beta
        }
    )
}
