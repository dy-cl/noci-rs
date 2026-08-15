// nonorthogonalwicks/eval/prepareonebodyoverlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_sd, _mm_cvtsd_f64, _mm256_add_pd, _mm256_castpd256_pd128, _mm256_extractf128_pd,
    _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_hadd_pd, _mm256_loadu_pd, _mm256_mul_pd,
    _mm256_set_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
    _mm512_add_pd, _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_loadu_pd, _mm512_mul_pd,
    _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd,
};

// Crate-root imports.
use crate::ExcitationSpin;
use crate::maths::{adjugate_transpose, build_d, det};
use crate::noci::NOCIScalar;
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, mix_dets_same,
};
use super::prepare::construct_determinant_indices;

/// Prepare and evaluate the same-spin overlap and generalised-Fock matrix element between excited
/// determinants generated from the reference pair `\langle{}^x\Psi| and |{}^w\Psi\rangle:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L=m}}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L),`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat F|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+1}\\m_1+\cdots+m_{L+1}=m}}`
/// `[{}^x F_0^{(m_1)}\det\mathbf D_{\mathrm{ov}}(m_2,\ldots,m_{L+1})`
/// `- \sum_{z=1}^{L}\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}`
/// `(m_1,\ldots,m_{L+1})].`
/// `For m = 0 and L = 1,\ldots,4, the contraction determinant is consumed directly while it is`
/// `constructed, avoiding the separate prepare-then-evaluate traversal. For arbitrary L and m > 0,`
/// `the required contraction determinants are constructed once before the constrained distribution sum.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: `Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.`
/// - `g_ex`: `Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.`
/// - `scratch`: Scratch storage for determinant labels, generic contraction determinants and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)`.
#[inline(always)]
pub(crate) fn xw_f_overlap_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap, {
        if w.m == 0 {
            xw_f_overlap_m0_prepared(w, l_ex, g_ex, scratch, tol)
        } else {
            xw_f_overlap_gen_prepared(w, l_ex, g_ex, scratch, tol)
        }
    })
}

/// Borrowed excitation pair consumed by the prepared batched Wick evaluator.
/// The total excitation rank and determinant phase are supplied by the caller so immutable
/// determinant metadata is not recomputed or copied into whole-row temporary buffers.
#[derive(Clone, Copy)]
pub(crate) struct WickBatchPair<'a> {
    l_ex: &'a ExcitationSpin,
    g_ex: &'a ExcitationSpin,
    rank: usize,
    phase: f64,
    output: usize,
}

impl<'a> WickBatchPair<'a> {
    /// Construct one borrowed prepared-Wick batch item.
    /// # Arguments:
    /// - `l_ex`: Bra excitation relative to its parent reference.
    /// - `g_ex`: Ket excitation relative to its parent reference.
    /// - `rank`: Total excitation rank `L = L_x + L_w`.
    /// - `phase`: Product of the bra and ket determinant excitation phases.
    /// - `output`: Output-row position receiving the matrix element.
    /// # Returns
    /// - `WickBatchPair<'a>`: Borrowed batch item.
    #[inline(always)]
    pub(crate) fn new(
        l_ex: &'a ExcitationSpin,
        g_ex: &'a ExcitationSpin,
        rank: usize,
        phase: f64,
        output: usize,
    ) -> Self {
        Self {
            l_ex,
            g_ex,
            rank,
            phase,
            output,
        }
    }
}

/// Prepare and evaluate a stream of same-spin overlap and generalised-Fock matrix elements.
/// Real `m = 0` pairs are accumulated only in fixed-size per-rank SIMD bins; no arrays scaling
/// with the number of source determinants are constructed. Unsupported ranks, complex arithmetic,
/// singular-reference cases with `m > 0`, and incomplete SIMD bins use the existing scalar
/// prepared evaluator without changing the matrix-element definition.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates shared by every pair in the stream.
/// - `pairs`: Borrowed excitation pairs with cached total ranks, phases and output positions.
/// - `scratch`: Scratch storage used by scalar-tail and unsupported-rank fallback evaluations.
/// - `tol`: Numerical tolerance used by determinant and adjugate-transpose fallbacks.
/// - `overlap`: Final same-spin overlap output row.
/// - `fock`: Final same-spin generalised-Fock output row.
/// # Returns
/// - `()`: Writes phased matrix elements directly to `overlap` and `fock`.
#[inline(always)]
pub(crate) fn xw_f_overlap_prepared_batch<'a, T, I>(
    w: &SameSpinView<'_, T>,
    pairs: I,
    scratch: &mut WickScratch<T>,
    tol: f64,
    overlap: &mut [T],
    fock: &mut [T],
) where
    T: NOCIScalar,
    I: IntoIterator<Item = WickBatchPair<'a>>,
{
    let mut pairs = pairs.into_iter();

    #[cfg(target_arch = "x86_64")]
    if TypeId::of::<T>() == TypeId::of::<f64>() && w.m == 0 {
        unsafe {
            let overlap_f64 =
                std::slice::from_raw_parts_mut(overlap.as_mut_ptr().cast::<f64>(), overlap.len());
            let fock_f64 =
                std::slice::from_raw_parts_mut(fock.as_mut_ptr().cast::<f64>(), fock.len());

            if is_x86_feature_detected!("avx512f") {
                xw_f_overlap_m0_prepared_f64x8(w, &mut pairs, scratch, tol, overlap_f64, fock_f64);
                return;
            }

            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                xw_f_overlap_m0_prepared_f64x4(w, &mut pairs, scratch, tol, overlap_f64, fock_f64);
                return;
            }
        }
    }

    for pair in pairs {
        let (s, f) = xw_f_overlap_prepared(w, pair.l_ex, pair.g_ex, scratch, tol);
        let phase = T::from_real(pair.phase);
        overlap[pair.output] = phase * s;
        fock[pair.output] = phase * f;
    }
}

/// Prepare and evaluate the overlap and generalised-Fock matrix element together when `m = 0`.
/// `Every contraction uses m_i = 0, so the total excitation rank L = L_x + L_w determines one`
/// `contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0). Fixed-rank prepared kernels are`
/// `used for L = 1,\ldots,4; arbitrary ranks construct the generic determinant once and then apply`
/// `the general cofactor form. For L = 0, the overlap is {}^{xw}\tilde S and only`
/// `{}^x F_0^{(0)} contributes to F.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for determinant labels and generic work arrays.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0, {
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;

        match l {
            0 => {
                let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
                (pref, pref * w.f0f[0])
            }
            1 => xw_f_overlap_m0_l1_prepared(w, l_ex, g_ex, scratch),
            2 => xw_f_overlap_m0_l2_prepared(w, l_ex, g_ex, scratch),
            3 => xw_f_overlap_m0_l3_prepared(w, l_ex, g_ex, scratch, tol),
            4 => xw_f_overlap_m0_l4_prepared(w, l_ex, g_ex, scratch, tol),
            _ => xw_f_overlap_m0_gen_prepared(w, l_ex, g_ex, scratch, l, tol),
        }
    })
}

/// Evaluate a stream of real `m = 0` prepared matrix elements in four-pair AVX2/FMA batches.
/// Only four excitation descriptors per supported rank are retained at once; complete bins use
/// the corresponding fixed-rank kernel and write phased values directly to the final output row.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `pairs`: Stream of borrowed excitation pairs with cached rank, phase and output position.
/// - `scratch`: Scratch storage used by scalar-tail and unsupported-rank fallbacks.
/// - `tol`: Numerical tolerance used by fallback determinant evaluation.
/// - `overlap`: Final real overlap output row.
/// - `fock`: Final real generalised-Fock output row.
/// # Returns
/// - `()`: Fills `overlap` and `fock`.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for both `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_prepared_f64x4<'a, T, I>(
    w: &SameSpinView<'_, T>,
    pairs: &mut I,
    scratch: &mut WickScratch<T>,
    tol: f64,
    overlap: &mut [f64],
    fock: &mut [f64],
) where
    T: NOCIScalar,
    I: Iterator<Item = WickBatchPair<'a>>,
{
    unsafe {
        let empty_ex = ExcitationSpin { holes: 0, parts: 0 };
        let mut l_bins = [[empty_ex; 4]; 5];
        let mut g_bins = [[empty_ex; 4]; 5];
        let mut phases = [[1.0f64; 4]; 5];
        let mut outputs = [[0usize; 4]; 5];
        let mut counts = [0usize; 5];
        let indices = [0usize, 1, 2, 3];

        for pair in pairs {
            let l = pair.rank;
            if (1..=4).contains(&l) {
                let count = counts[l];
                l_bins[l][count] = *pair.l_ex;
                g_bins[l][count] = *pair.g_ex;
                phases[l][count] = pair.phase;
                outputs[l][count] = pair.output;
                counts[l] += 1;

                if counts[l] == 4 {
                    let mut s = [0.0f64; 4];
                    let mut f = [0.0f64; 4];
                    match l {
                        1 => xw_f_overlap_m0_l1_prepared_f64x4(
                            w, &l_bins[l], &g_bins[l], indices, &mut s, &mut f,
                        ),
                        2 => xw_f_overlap_m0_l2_prepared_f64x4(
                            w, &l_bins[l], &g_bins[l], indices, &mut s, &mut f,
                        ),
                        3 => xw_f_overlap_m0_l3_prepared_f64x4(
                            w, &l_bins[l], &g_bins[l], indices, &mut s, &mut f,
                        ),
                        4 => xw_f_overlap_m0_l4_prepared_f64x4(
                            w, &l_bins[l], &g_bins[l], indices, &mut s, &mut f,
                        ),
                        _ => unreachable!(),
                    }

                    for lane in 0..4 {
                        let output = outputs[l][lane];
                        let phase = phases[l][lane];
                        overlap[output] = phase * s[lane];
                        fock[output] = phase * f[lane];
                    }
                    counts[l] = 0;
                }
            } else {
                let (s, f) = xw_f_overlap_prepared(w, pair.l_ex, pair.g_ex, scratch, tol);
                overlap[pair.output] = pair.phase * *std::ptr::from_ref(&s).cast::<f64>();
                fock[pair.output] = pair.phase * *std::ptr::from_ref(&f).cast::<f64>();
            }
        }

        for l in 1..=4 {
            for pos in 0..counts[l] {
                let (s, f) =
                    xw_f_overlap_prepared(w, &l_bins[l][pos], &g_bins[l][pos], scratch, tol);
                let output = outputs[l][pos];
                let phase = phases[l][pos];
                overlap[output] = phase * *std::ptr::from_ref(&s).cast::<f64>();
                fock[output] = phase * *std::ptr::from_ref(&f).cast::<f64>();
            }
        }
    }
}

/// Evaluate a stream of real `m = 0` prepared matrix elements in eight-pair AVX-512 batches.
/// Only eight excitation descriptors per supported rank are retained at once; complete bins use
/// the corresponding fixed-rank kernel and write phased values directly to the final output row.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `pairs`: Stream of borrowed excitation pairs with cached rank, phase and output position.
/// - `scratch`: Scratch storage used by scalar-tail and unsupported-rank fallbacks.
/// - `tol`: Numerical tolerance used by fallback determinant evaluation.
/// - `overlap`: Final real overlap output row.
/// - `fock`: Final real generalised-Fock output row.
/// # Returns
/// - `()`: Fills `overlap` and `fock`.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_prepared_f64x8<'a, T, I>(
    w: &SameSpinView<'_, T>,
    pairs: &mut I,
    scratch: &mut WickScratch<T>,
    tol: f64,
    overlap: &mut [f64],
    fock: &mut [f64],
) where
    T: NOCIScalar,
    I: Iterator<Item = WickBatchPair<'a>>,
{
    unsafe {
        let empty_ex = ExcitationSpin { holes: 0, parts: 0 };
        let mut l_bins = [[empty_ex; 8]; 5];
        let mut g_bins = [[empty_ex; 8]; 5];
        let mut phases = [[1.0f64; 8]; 5];
        let mut outputs = [[0usize; 8]; 5];
        let mut counts = [0usize; 5];
        let indices = [0usize, 1, 2, 3, 4, 5, 6, 7];

        for pair in pairs {
            let l = pair.rank;
            if (1..=4).contains(&l) {
                let count = counts[l];
                l_bins[l][count] = *pair.l_ex;
                g_bins[l][count] = *pair.g_ex;
                phases[l][count] = pair.phase;
                outputs[l][count] = pair.output;
                counts[l] += 1;

                if counts[l] == 8 {
                    let mut s = [0.0f64; 8];
                    let mut f = [0.0f64; 8];
                    match l {
                        1 => xw_f_overlap_m0_l1_prepared_f64x8(
                            w, &l_bins[l], &g_bins[l], indices, &mut s, &mut f,
                        ),
                        2 => xw_f_overlap_m0_l2_prepared_f64x8(
                            w, &l_bins[l], &g_bins[l], indices, &mut s, &mut f,
                        ),
                        3 => xw_f_overlap_m0_l3_prepared_f64x8(
                            w, &l_bins[l], &g_bins[l], indices, &mut s, &mut f,
                        ),
                        4 => xw_f_overlap_m0_l4_prepared_f64x8(
                            w, &l_bins[l], &g_bins[l], indices, &mut s, &mut f,
                        ),
                        _ => unreachable!(),
                    }

                    for lane in 0..8 {
                        let output = outputs[l][lane];
                        let phase = phases[l][lane];
                        overlap[output] = phase * s[lane];
                        fock[output] = phase * f[lane];
                    }
                    counts[l] = 0;
                }
            } else {
                let (s, f) = xw_f_overlap_prepared(w, pair.l_ex, pair.g_ex, scratch, tol);
                overlap[pair.output] = pair.phase * *std::ptr::from_ref(&s).cast::<f64>();
                fock[pair.output] = pair.phase * *std::ptr::from_ref(&f).cast::<f64>();
            }
        }

        for l in 1..=4 {
            for pos in 0..counts[l] {
                let (s, f) =
                    xw_f_overlap_prepared(w, &l_bins[l][pos], &g_bins[l][pos], scratch, tol);
                let output = outputs[l][pos];
                let phase = phases[l][pos];
                overlap[output] = phase * *std::ptr::from_ref(&s).cast::<f64>();
                fock[output] = phase * *std::ptr::from_ref(&f).cast::<f64>();
            }
        }
    }
}

/// Prepare and evaluate the fixed-rank `L = 1` overlap and generalised-Fock matrix element for `m = 0`.
/// `For \mathbf D_{\mathrm{ov}} = [D_{00}], S = {}^{xw}\tilde S D_{00}` and
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}D_{00} - \mathcal F_{r_0c_0}^{(0,0)}].`
/// The determinant labels are constructed and the single contraction is consumed directly without
/// materialising `scratch.det0`.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the rank-one row and column labels.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 1` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l1_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l1, {
        scratch.ensure_same_m0(1);
        scratch.rows.ensure(1);
        scratch.cols.ensure(1);

        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let n = w.n();
        let r0 = scratch.rows[0];
        let c0 = scratch.cols[0];
        let det = w.x_slice(0)[r0 * n + c0];
        let repl = w.ff_t_slice(0, 0)[c0 * n + r0];
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let overlap = pref * det;
        let fock = pref * (det * w.f0f[0] - repl);

        (overlap, fock)
    })
}

/// Prepare and evaluate 4 independent real fixed-rank `L = 1` matrix elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX2/FMA` arithmetic evaluates the same
/// determinant, cofactor and generalised-Fock algebra for 4 independent excitation pairs without
/// horizontal reductions between pairs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l_ex`: Bra excitations for the enclosing batch.
/// - `g_ex`: Ket excitations for the enclosing batch.
/// - `indices`: Positions of the 4 equal-rank pairs within the enclosing batch.
/// - `overlap`: Real overlap output slice.
/// - `fock`: Real generalised-Fock output slice.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements at `indices`.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_l1_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &[ExcitationSpin],
    g_ex: &[ExcitationSpin],
    indices: [usize; 4],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        let n = w.n();
        let x0_t = w.x_slice(0);
        let fsl_t = w.ff_t_slice(0, 0);
        let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
        let fsl = std::slice::from_raw_parts(fsl_t.as_ptr().cast::<f64>(), fsl_t.len());
        let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
        let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
        let pref = phase * w.tilde_s_prod;
        let mut d = [[0.0f64; 4]; 1];
        let mut ff = [[0.0f64; 4]; 1];
        for lane in 0..4 {
            let i = indices[lane];
            let mut rows = [0usize; 1];
            let mut cols = [0usize; 1];
            construct_determinant_indices(&l_ex[i], &g_ex[i], w, &mut rows, &mut cols);
            d[0][lane] = x0[rows[0] * n + cols[0]];
            ff[0][lane] = fsl[cols[0] * n + rows[0]];
        }
        let det = _mm256_loadu_pd(d[0].as_ptr());
        let repl = _mm256_loadu_pd(ff[0].as_ptr());
        let pref_v = _mm256_set1_pd(pref);
        let f0_v = _mm256_set1_pd(f0);
        let overlap_v = _mm256_mul_pd(det, pref_v);
        let contrib = _mm256_fmsub_pd(det, f0_v, repl);
        let fock_v = _mm256_mul_pd(contrib, pref_v);

        let mut overlap_lane = [0.0f64; 4];
        let mut fock_lane = [0.0f64; 4];
        _mm256_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
        _mm256_storeu_pd(fock_lane.as_mut_ptr(), fock_v);
        for lane in 0..4 {
            overlap[indices[lane]] = overlap_lane[lane];
            fock[indices[lane]] = fock_lane[lane];
        }
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 1` matrix elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates the same
/// determinant, cofactor and generalised-Fock algebra for 8 independent excitation pairs without
/// horizontal reductions between pairs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l_ex`: Bra excitations for the enclosing batch.
/// - `g_ex`: Ket excitations for the enclosing batch.
/// - `indices`: Positions of the 8 equal-rank pairs within the enclosing batch.
/// - `overlap`: Real overlap output slice.
/// - `fock`: Real generalised-Fock output slice.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements at `indices`.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_l1_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &[ExcitationSpin],
    g_ex: &[ExcitationSpin],
    indices: [usize; 8],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        let n = w.n();
        let x0_t = w.x_slice(0);
        let fsl_t = w.ff_t_slice(0, 0);
        let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
        let fsl = std::slice::from_raw_parts(fsl_t.as_ptr().cast::<f64>(), fsl_t.len());
        let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
        let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
        let pref = phase * w.tilde_s_prod;
        let mut d = [[0.0f64; 8]; 1];
        let mut ff = [[0.0f64; 8]; 1];
        for lane in 0..8 {
            let i = indices[lane];
            let mut rows = [0usize; 1];
            let mut cols = [0usize; 1];
            construct_determinant_indices(&l_ex[i], &g_ex[i], w, &mut rows, &mut cols);
            d[0][lane] = x0[rows[0] * n + cols[0]];
            ff[0][lane] = fsl[cols[0] * n + rows[0]];
        }
        let det = _mm512_loadu_pd(d[0].as_ptr());
        let repl = _mm512_loadu_pd(ff[0].as_ptr());
        let pref_v = _mm512_set1_pd(pref);
        let f0_v = _mm512_set1_pd(f0);
        let overlap_v = _mm512_mul_pd(det, pref_v);
        let contrib = _mm512_fmsub_pd(det, f0_v, repl);
        let fock_v = _mm512_mul_pd(contrib, pref_v);

        let mut overlap_lane = [0.0f64; 8];
        let mut fock_lane = [0.0f64; 8];
        _mm512_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
        _mm512_storeu_pd(fock_lane.as_mut_ptr(), fock_v);
        for lane in 0..8 {
            overlap[indices[lane]] = overlap_lane[lane];
            fock[indices[lane]] = fock_lane[lane];
        }
    }
}

/// Prepare and evaluate the fixed-rank `L = 2` overlap and generalised-Fock matrix element for `m = 0`:
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}},`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0}`
/// `- \det\mathbf D_{\mathrm{ov}}^{1\rightarrow\mathcal F_1}].`
/// The determinant labels are constructed first and the real-valued path loads the four contraction
/// entries directly from `X^{(0)}` and `Y^{(0)}` before evaluating the overlap determinant and both
/// replacement determinants simultaneously with packed AVX FMA operations.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the rank-two row and column labels.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 2` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l2_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l2, {
        scratch.ensure_same_m0(2);
        scratch.rows.ensure(2);
        scratch.cols.ensure(2);

        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let n = w.n();
        let r0 = scratch.rows[0];
        let r1 = scratch.rows[1];
        let c0 = scratch.cols[0];
        let c1 = scratch.cols[1];
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let fsl = w.ff_t_slice(0, 0);
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            unsafe {
                let x0 = std::slice::from_raw_parts(x0.as_ptr().cast::<f64>(), x0.len());
                let y0 = std::slice::from_raw_parts(y0.as_ptr().cast::<f64>(), y0.len());
                let fsl = std::slice::from_raw_parts(fsl.as_ptr().cast::<f64>(), fsl.len());

                let a00 = x0[r0 * n + c0];
                let a01 = y0[r0 * n + c1];
                let a10 = x0[r1 * n + c0];
                let a11 = x0[r1 * n + c1];

                let u0 = fsl[c0 * n + r0];
                let u1 = fsl[c0 * n + r1];
                let v0 = fsl[c1 * n + r0];
                let v1 = fsl[c1 * n + r1];

                let lhs0 = _mm256_set_pd(0.0, a00, u0, a00);
                let rhs0 = _mm256_set_pd(0.0, v1, a11, a11);
                let lhs1 = _mm256_set_pd(0.0, v0, a01, a01);
                let rhs1 = _mm256_set_pd(0.0, a10, u1, a10);
                let values = _mm256_fmsub_pd(lhs0, rhs0, _mm256_mul_pd(lhs1, rhs1));

                let mut packed = [0.0; 4];
                _mm256_storeu_pd(packed.as_mut_ptr(), values);

                let det = T::from_real(packed[0]);
                let det_c0 = T::from_real(packed[1]);
                let det_c1 = T::from_real(packed[2]);
                let overlap = pref * det;
                let fock = pref * (det * w.f0f[0] - det_c0 - det_c1);

                return (overlap, fock);
            }
        }

        let a00 = x0[r0 * n + c0];
        let a01 = y0[r0 * n + c1];
        let a10 = x0[r1 * n + c0];
        let a11 = x0[r1 * n + c1];
        let det = a00 * a11 - a01 * a10;

        let u0 = fsl[c0 * n + r0];
        let u1 = fsl[c0 * n + r1];
        let v0 = fsl[c1 * n + r0];
        let v1 = fsl[c1 * n + r1];
        let det_c0 = u0 * a11 - a01 * u1;
        let det_c1 = a00 * v1 - v0 * a10;

        let overlap = pref * det;
        let fock = pref * (det * w.f0f[0] - det_c0 - det_c1);

        (overlap, fock)
    })
}

/// Prepare and evaluate 4 independent real fixed-rank `L = 2` matrix elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX2/FMA` arithmetic evaluates the same
/// determinant, cofactor and generalised-Fock algebra for 4 independent excitation pairs without
/// horizontal reductions between pairs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l_ex`: Bra excitations for the enclosing batch.
/// - `g_ex`: Ket excitations for the enclosing batch.
/// - `indices`: Positions of the 4 equal-rank pairs within the enclosing batch.
/// - `overlap`: Real overlap output slice.
/// - `fock`: Real generalised-Fock output slice.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements at `indices`.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_l2_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &[ExcitationSpin],
    g_ex: &[ExcitationSpin],
    indices: [usize; 4],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        let n = w.n();
        let x0_t = w.x_slice(0);
        let y0_t = w.y_slice(0);
        let fsl_t = w.ff_t_slice(0, 0);
        let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
        let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
        let fsl = std::slice::from_raw_parts(fsl_t.as_ptr().cast::<f64>(), fsl_t.len());
        let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
        let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
        let pref = phase * w.tilde_s_prod;
        let mut d = [[0.0f64; 4]; 4];
        let mut ff = [[0.0f64; 4]; 4];
        for lane in 0..4 {
            let i = indices[lane];
            let mut rows = [0usize; 2];
            let mut cols = [0usize; 2];
            construct_determinant_indices(&l_ex[i], &g_ex[i], w, &mut rows, &mut cols);
            d[0][lane] = x0[rows[0] * n + cols[0]];
            ff[0][lane] = fsl[cols[0] * n + rows[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            ff[1][lane] = fsl[cols[1] * n + rows[0]];
            d[2][lane] = x0[rows[1] * n + cols[0]];
            ff[2][lane] = fsl[cols[0] * n + rows[1]];
            d[3][lane] = x0[rows[1] * n + cols[1]];
            ff[3][lane] = fsl[cols[1] * n + rows[1]];
        }
        let a00 = _mm256_loadu_pd(d[0].as_ptr());
        let a01 = _mm256_loadu_pd(d[1].as_ptr());
        let a10 = _mm256_loadu_pd(d[2].as_ptr());
        let a11 = _mm256_loadu_pd(d[3].as_ptr());
        let f00 = _mm256_loadu_pd(ff[0].as_ptr());
        let f01 = _mm256_loadu_pd(ff[1].as_ptr());
        let f10 = _mm256_loadu_pd(ff[2].as_ptr());
        let f11 = _mm256_loadu_pd(ff[3].as_ptr());

        let det = _mm256_fmsub_pd(a00, a11, _mm256_mul_pd(a01, a10));
        let cof00 = a11;
        let cof01 = _mm256_sub_pd(_mm256_setzero_pd(), a10);
        let cof10 = _mm256_sub_pd(_mm256_setzero_pd(), a01);
        let cof11 = a00;

        let repl0 = _mm256_fmadd_pd(cof01, f01, _mm256_mul_pd(cof00, f00));
        let repl1 = _mm256_fmadd_pd(cof11, f11, _mm256_mul_pd(cof10, f10));
        let repl = _mm256_add_pd(repl0, repl1);

        let pref_v = _mm256_set1_pd(pref);
        let f0_v = _mm256_set1_pd(f0);
        let overlap_v = _mm256_mul_pd(det, pref_v);
        let contrib = _mm256_fmsub_pd(det, f0_v, repl);
        let fock_v = _mm256_mul_pd(contrib, pref_v);

        let mut overlap_lane = [0.0f64; 4];
        let mut fock_lane = [0.0f64; 4];
        _mm256_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
        _mm256_storeu_pd(fock_lane.as_mut_ptr(), fock_v);
        for lane in 0..4 {
            overlap[indices[lane]] = overlap_lane[lane];
            fock[indices[lane]] = fock_lane[lane];
        }
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 2` matrix elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates the same
/// determinant, cofactor and generalised-Fock algebra for 8 independent excitation pairs without
/// horizontal reductions between pairs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l_ex`: Bra excitations for the enclosing batch.
/// - `g_ex`: Ket excitations for the enclosing batch.
/// - `indices`: Positions of the 8 equal-rank pairs within the enclosing batch.
/// - `overlap`: Real overlap output slice.
/// - `fock`: Real generalised-Fock output slice.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements at `indices`.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_l2_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &[ExcitationSpin],
    g_ex: &[ExcitationSpin],
    indices: [usize; 8],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        let n = w.n();
        let x0_t = w.x_slice(0);
        let y0_t = w.y_slice(0);
        let fsl_t = w.ff_t_slice(0, 0);
        let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
        let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
        let fsl = std::slice::from_raw_parts(fsl_t.as_ptr().cast::<f64>(), fsl_t.len());
        let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
        let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
        let pref = phase * w.tilde_s_prod;
        let mut d = [[0.0f64; 8]; 4];
        let mut ff = [[0.0f64; 8]; 4];
        for lane in 0..8 {
            let i = indices[lane];
            let mut rows = [0usize; 2];
            let mut cols = [0usize; 2];
            construct_determinant_indices(&l_ex[i], &g_ex[i], w, &mut rows, &mut cols);
            d[0][lane] = x0[rows[0] * n + cols[0]];
            ff[0][lane] = fsl[cols[0] * n + rows[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            ff[1][lane] = fsl[cols[1] * n + rows[0]];
            d[2][lane] = x0[rows[1] * n + cols[0]];
            ff[2][lane] = fsl[cols[0] * n + rows[1]];
            d[3][lane] = x0[rows[1] * n + cols[1]];
            ff[3][lane] = fsl[cols[1] * n + rows[1]];
        }
        let a00 = _mm512_loadu_pd(d[0].as_ptr());
        let a01 = _mm512_loadu_pd(d[1].as_ptr());
        let a10 = _mm512_loadu_pd(d[2].as_ptr());
        let a11 = _mm512_loadu_pd(d[3].as_ptr());
        let f00 = _mm512_loadu_pd(ff[0].as_ptr());
        let f01 = _mm512_loadu_pd(ff[1].as_ptr());
        let f10 = _mm512_loadu_pd(ff[2].as_ptr());
        let f11 = _mm512_loadu_pd(ff[3].as_ptr());

        let det = _mm512_fmsub_pd(a00, a11, _mm512_mul_pd(a01, a10));
        let cof00 = a11;
        let cof01 = _mm512_sub_pd(_mm512_setzero_pd(), a10);
        let cof10 = _mm512_sub_pd(_mm512_setzero_pd(), a01);
        let cof11 = a00;

        let repl0 = _mm512_fmadd_pd(cof01, f01, _mm512_mul_pd(cof00, f00));
        let repl1 = _mm512_fmadd_pd(cof11, f11, _mm512_mul_pd(cof10, f10));
        let repl = _mm512_add_pd(repl0, repl1);

        let pref_v = _mm512_set1_pd(pref);
        let f0_v = _mm512_set1_pd(f0);
        let overlap_v = _mm512_mul_pd(det, pref_v);
        let contrib = _mm512_fmsub_pd(det, f0_v, repl);
        let fock_v = _mm512_mul_pd(contrib, pref_v);

        let mut overlap_lane = [0.0f64; 8];
        let mut fock_lane = [0.0f64; 8];
        _mm512_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
        _mm512_storeu_pd(fock_lane.as_mut_ptr(), fock_v);
        for lane in 0..8 {
            overlap[indices[lane]] = overlap_lane[lane];
            fock[indices[lane]] = fock_lane[lane];
        }
    }
}

/// Prepare and evaluate the fixed-rank `L = 3` overlap and generalised-Fock matrix element for `m = 0`:
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}},`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}`
/// `\mathcal F_{\eta z}^{(0,0)}].`
/// The determinant labels are constructed first. The real-valued path loads the nine contraction
/// entries directly from `X^{(0)}` and `Y^{(0)}`, forms the cofactor rows with packed AVX FMA
/// operations and immediately contracts them with the generalised-Fock entries. The generic scalar
/// path materialises the local `3 x 3` contraction determinant before applying the shared adjugate routine.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for determinant labels and generic adjugate work arrays.
/// - `tol`: Numerical tolerance used when evaluating the generic adjugate-transpose matrix.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 3` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l3_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l3, {
        scratch.ensure_same(3);

        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let n = w.n();
        let r0 = scratch.rows[0];
        let r1 = scratch.rows[1];
        let r2 = scratch.rows[2];
        let c0 = scratch.cols[0];
        let c1 = scratch.cols[1];
        let c2 = scratch.cols[2];
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let fsl = w.ff_t_slice(0, 0);
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);

        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            unsafe {
                let x0 = std::slice::from_raw_parts(x0.as_ptr().cast::<f64>(), x0.len());
                let y0 = std::slice::from_raw_parts(y0.as_ptr().cast::<f64>(), y0.len());
                let fsl = std::slice::from_raw_parts(fsl.as_ptr().cast::<f64>(), fsl.len());

                let a00 = x0[r0 * n + c0];
                let a01 = y0[r0 * n + c1];
                let a02 = y0[r0 * n + c2];
                let a10 = x0[r1 * n + c0];
                let a11 = x0[r1 * n + c1];
                let a12 = y0[r1 * n + c2];
                let a20 = x0[r2 * n + c0];
                let a21 = x0[r2 * n + c1];
                let a22 = x0[r2 * n + c2];

                let p00 = _mm256_set_pd(0.0, a10, a12, a11);
                let q00 = _mm256_set_pd(0.0, a21, a20, a22);
                let p01 = _mm256_set_pd(0.0, a11, a10, a12);
                let q01 = _mm256_set_pd(0.0, a20, a22, a21);
                let cof0 = _mm256_fmsub_pd(p00, q00, _mm256_mul_pd(p01, q01));

                let p10 = _mm256_set_pd(0.0, a00, a02, a01);
                let q10 = _mm256_set_pd(0.0, a21, a20, a22);
                let p11 = _mm256_set_pd(0.0, a01, a00, a02);
                let q11 = _mm256_set_pd(0.0, a20, a22, a21);
                let cof1 = _mm256_fmsub_pd(p11, q11, _mm256_mul_pd(p10, q10));

                let p20 = _mm256_set_pd(0.0, a00, a02, a01);
                let q20 = _mm256_set_pd(0.0, a11, a10, a12);
                let p21 = _mm256_set_pd(0.0, a01, a00, a02);
                let q21 = _mm256_set_pd(0.0, a10, a12, a11);
                let cof2 = _mm256_fmsub_pd(p20, q20, _mm256_mul_pd(p21, q21));

                let row0 = _mm256_set_pd(0.0, a02, a01, a00);
                let det_products = _mm256_mul_pd(row0, cof0);
                let det_sums = _mm256_hadd_pd(det_products, det_products);
                let det_low = _mm256_castpd256_pd128(det_sums);
                let det_high = _mm256_extractf128_pd(det_sums, 1);
                let det = _mm_cvtsd_f64(_mm_add_sd(det_low, det_high));

                if !det.is_finite() {
                    return (zero, zero);
                }

                let frow0 =
                    _mm256_set_pd(0.0, fsl[c2 * n + r0], fsl[c1 * n + r0], fsl[c0 * n + r0]);
                let frow1 =
                    _mm256_set_pd(0.0, fsl[c2 * n + r1], fsl[c1 * n + r1], fsl[c0 * n + r1]);
                let frow2 =
                    _mm256_set_pd(0.0, fsl[c2 * n + r2], fsl[c1 * n + r2], fsl[c0 * n + r2]);

                let repl01 = _mm256_fmadd_pd(cof1, frow1, _mm256_mul_pd(cof0, frow0));
                let repl_v = _mm256_fmadd_pd(cof2, frow2, repl01);
                let repl_sums = _mm256_hadd_pd(repl_v, repl_v);
                let repl_low = _mm256_castpd256_pd128(repl_sums);
                let repl_high = _mm256_extractf128_pd(repl_sums, 1);
                let repl = _mm_cvtsd_f64(_mm_add_sd(repl_low, repl_high));

                let det_t = T::from_real(det);
                let overlap = pref * det_t;
                let fock = pref * (det_t * w.f0f[0] - T::from_real(repl));

                return (overlap, fock);
            }
        }

        let det0 = [
            x0[r0 * n + c0],
            y0[r0 * n + c1],
            y0[r0 * n + c2],
            x0[r1 * n + c0],
            x0[r1 * n + c1],
            y0[r1 * n + c2],
            x0[r2 * n + c0],
            x0[r2 * n + c1],
            x0[r2 * n + c2],
        ];

        if let Some(det_det) = adjugate_transpose(
            scratch.adjt_det.as_mut_slice(),
            scratch.invs.as_mut_slice(),
            scratch.lu.as_mut_slice(),
            &det0,
            3,
            tol,
        ) {
            let cof = scratch.adjt_det.as_slice();
            let f00 = fsl[c0 * n + r0];
            let f10 = fsl[c0 * n + r1];
            let f20 = fsl[c0 * n + r2];
            let f01 = fsl[c1 * n + r0];
            let f11 = fsl[c1 * n + r1];
            let f21 = fsl[c1 * n + r2];
            let f02 = fsl[c2 * n + r0];
            let f12 = fsl[c2 * n + r1];
            let f22 = fsl[c2 * n + r2];
            let repl = cof[0] * f00
                + cof[3] * f10
                + cof[6] * f20
                + cof[1] * f01
                + cof[4] * f11
                + cof[7] * f21
                + cof[2] * f02
                + cof[5] * f12
                + cof[8] * f22;

            let overlap = pref * det_det;
            let fock = pref * (det_det * w.f0f[0] - repl);

            (overlap, fock)
        } else {
            let overlap = pref * det(&det0, 3).unwrap_or(zero);
            (overlap, zero)
        }
    })
}

/// Prepare and evaluate 4 independent real fixed-rank `L = 3` matrix elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX2/FMA` arithmetic evaluates the same
/// determinant, cofactor and generalised-Fock algebra for 4 independent excitation pairs without
/// horizontal reductions between pairs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l_ex`: Bra excitations for the enclosing batch.
/// - `g_ex`: Ket excitations for the enclosing batch.
/// - `indices`: Positions of the 4 equal-rank pairs within the enclosing batch.
/// - `overlap`: Real overlap output slice.
/// - `fock`: Real generalised-Fock output slice.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements at `indices`.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_l3_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &[ExcitationSpin],
    g_ex: &[ExcitationSpin],
    indices: [usize; 4],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        let n = w.n();
        let x0_t = w.x_slice(0);
        let y0_t = w.y_slice(0);
        let fsl_t = w.ff_t_slice(0, 0);
        let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
        let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
        let fsl = std::slice::from_raw_parts(fsl_t.as_ptr().cast::<f64>(), fsl_t.len());
        let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
        let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
        let pref = phase * w.tilde_s_prod;
        let mut d = [[0.0f64; 4]; 9];
        let mut ff = [[0.0f64; 4]; 9];
        for lane in 0..4 {
            let i = indices[lane];
            let mut rows = [0usize; 3];
            let mut cols = [0usize; 3];
            construct_determinant_indices(&l_ex[i], &g_ex[i], w, &mut rows, &mut cols);
            d[0][lane] = x0[rows[0] * n + cols[0]];
            ff[0][lane] = fsl[cols[0] * n + rows[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            ff[1][lane] = fsl[cols[1] * n + rows[0]];
            d[2][lane] = y0[rows[0] * n + cols[2]];
            ff[2][lane] = fsl[cols[2] * n + rows[0]];
            d[3][lane] = x0[rows[1] * n + cols[0]];
            ff[3][lane] = fsl[cols[0] * n + rows[1]];
            d[4][lane] = x0[rows[1] * n + cols[1]];
            ff[4][lane] = fsl[cols[1] * n + rows[1]];
            d[5][lane] = y0[rows[1] * n + cols[2]];
            ff[5][lane] = fsl[cols[2] * n + rows[1]];
            d[6][lane] = x0[rows[2] * n + cols[0]];
            ff[6][lane] = fsl[cols[0] * n + rows[2]];
            d[7][lane] = x0[rows[2] * n + cols[1]];
            ff[7][lane] = fsl[cols[1] * n + rows[2]];
            d[8][lane] = x0[rows[2] * n + cols[2]];
            ff[8][lane] = fsl[cols[2] * n + rows[2]];
        }
        let mut det_v = _mm256_setzero_pd();
        let mut repl0 = _mm256_setzero_pd();
        let mut repl1 = _mm256_setzero_pd();
        let mut repl2 = _mm256_setzero_pd();

        let cof00 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[4].as_ptr()),
            _mm256_loadu_pd(d[8].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[7].as_ptr()),
            ),
        );
        det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[0].as_ptr()), cof00, det_v);
        repl0 = _mm256_fmadd_pd(_mm256_loadu_pd(ff[0].as_ptr()), cof00, repl0);
        let cof01 = _mm256_sub_pd(
            _mm256_setzero_pd(),
            _mm256_fmsub_pd(
                _mm256_loadu_pd(d[3].as_ptr()),
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[5].as_ptr()),
                    _mm256_loadu_pd(d[6].as_ptr()),
                ),
            ),
        );
        det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[1].as_ptr()), cof01, det_v);
        repl0 = _mm256_fmadd_pd(_mm256_loadu_pd(ff[1].as_ptr()), cof01, repl0);
        let cof02 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[3].as_ptr()),
            _mm256_loadu_pd(d[7].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[6].as_ptr()),
            ),
        );
        det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[2].as_ptr()), cof02, det_v);
        repl0 = _mm256_fmadd_pd(_mm256_loadu_pd(ff[2].as_ptr()), cof02, repl0);
        let cof10 = _mm256_sub_pd(
            _mm256_setzero_pd(),
            _mm256_fmsub_pd(
                _mm256_loadu_pd(d[1].as_ptr()),
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[2].as_ptr()),
                    _mm256_loadu_pd(d[7].as_ptr()),
                ),
            ),
        );
        repl1 = _mm256_fmadd_pd(_mm256_loadu_pd(ff[3].as_ptr()), cof10, repl1);
        let cof11 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[0].as_ptr()),
            _mm256_loadu_pd(d[8].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[2].as_ptr()),
                _mm256_loadu_pd(d[6].as_ptr()),
            ),
        );
        repl1 = _mm256_fmadd_pd(_mm256_loadu_pd(ff[4].as_ptr()), cof11, repl1);
        let cof12 = _mm256_sub_pd(
            _mm256_setzero_pd(),
            _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                _mm256_loadu_pd(d[7].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[1].as_ptr()),
                    _mm256_loadu_pd(d[6].as_ptr()),
                ),
            ),
        );
        repl1 = _mm256_fmadd_pd(_mm256_loadu_pd(ff[5].as_ptr()), cof12, repl1);
        let cof20 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[1].as_ptr()),
            _mm256_loadu_pd(d[5].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[2].as_ptr()),
                _mm256_loadu_pd(d[4].as_ptr()),
            ),
        );
        repl2 = _mm256_fmadd_pd(_mm256_loadu_pd(ff[6].as_ptr()), cof20, repl2);
        let cof21 = _mm256_sub_pd(
            _mm256_setzero_pd(),
            _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[2].as_ptr()),
                    _mm256_loadu_pd(d[3].as_ptr()),
                ),
            ),
        );
        repl2 = _mm256_fmadd_pd(_mm256_loadu_pd(ff[7].as_ptr()), cof21, repl2);
        let cof22 = _mm256_fmsub_pd(
            _mm256_loadu_pd(d[0].as_ptr()),
            _mm256_loadu_pd(d[4].as_ptr()),
            _mm256_mul_pd(
                _mm256_loadu_pd(d[1].as_ptr()),
                _mm256_loadu_pd(d[3].as_ptr()),
            ),
        );
        repl2 = _mm256_fmadd_pd(_mm256_loadu_pd(ff[8].as_ptr()), cof22, repl2);
        let repl01 = _mm256_add_pd(repl0, repl1);
        let repl_v = _mm256_add_pd(repl01, repl2);
        let pref_v = _mm256_set1_pd(pref);
        let f0_v = _mm256_set1_pd(f0);
        let overlap_v = _mm256_mul_pd(det_v, pref_v);
        let contrib = _mm256_fmsub_pd(det_v, f0_v, repl_v);
        let fock_v = _mm256_mul_pd(contrib, pref_v);

        let mut det_lane = [0.0f64; 4];
        let mut overlap_lane = [0.0f64; 4];
        let mut fock_lane = [0.0f64; 4];
        _mm256_storeu_pd(det_lane.as_mut_ptr(), det_v);
        _mm256_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
        _mm256_storeu_pd(fock_lane.as_mut_ptr(), fock_v);
        for lane in 0..4 {
            if det_lane[lane].is_finite() {
                overlap[indices[lane]] = overlap_lane[lane];
                fock[indices[lane]] = fock_lane[lane];
            } else {
                overlap[indices[lane]] = 0.0;
                fock[indices[lane]] = 0.0;
            }
        }
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 3` matrix elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates the same
/// determinant, cofactor and generalised-Fock algebra for 8 independent excitation pairs without
/// horizontal reductions between pairs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l_ex`: Bra excitations for the enclosing batch.
/// - `g_ex`: Ket excitations for the enclosing batch.
/// - `indices`: Positions of the 8 equal-rank pairs within the enclosing batch.
/// - `overlap`: Real overlap output slice.
/// - `fock`: Real generalised-Fock output slice.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements at `indices`.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_l3_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &[ExcitationSpin],
    g_ex: &[ExcitationSpin],
    indices: [usize; 8],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        let n = w.n();
        let x0_t = w.x_slice(0);
        let y0_t = w.y_slice(0);
        let fsl_t = w.ff_t_slice(0, 0);
        let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
        let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
        let fsl = std::slice::from_raw_parts(fsl_t.as_ptr().cast::<f64>(), fsl_t.len());
        let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
        let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
        let pref = phase * w.tilde_s_prod;
        let mut d = [[0.0f64; 8]; 9];
        let mut ff = [[0.0f64; 8]; 9];
        for lane in 0..8 {
            let i = indices[lane];
            let mut rows = [0usize; 3];
            let mut cols = [0usize; 3];
            construct_determinant_indices(&l_ex[i], &g_ex[i], w, &mut rows, &mut cols);
            d[0][lane] = x0[rows[0] * n + cols[0]];
            ff[0][lane] = fsl[cols[0] * n + rows[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            ff[1][lane] = fsl[cols[1] * n + rows[0]];
            d[2][lane] = y0[rows[0] * n + cols[2]];
            ff[2][lane] = fsl[cols[2] * n + rows[0]];
            d[3][lane] = x0[rows[1] * n + cols[0]];
            ff[3][lane] = fsl[cols[0] * n + rows[1]];
            d[4][lane] = x0[rows[1] * n + cols[1]];
            ff[4][lane] = fsl[cols[1] * n + rows[1]];
            d[5][lane] = y0[rows[1] * n + cols[2]];
            ff[5][lane] = fsl[cols[2] * n + rows[1]];
            d[6][lane] = x0[rows[2] * n + cols[0]];
            ff[6][lane] = fsl[cols[0] * n + rows[2]];
            d[7][lane] = x0[rows[2] * n + cols[1]];
            ff[7][lane] = fsl[cols[1] * n + rows[2]];
            d[8][lane] = x0[rows[2] * n + cols[2]];
            ff[8][lane] = fsl[cols[2] * n + rows[2]];
        }
        let mut det_v = _mm512_setzero_pd();
        let mut repl0 = _mm512_setzero_pd();
        let mut repl1 = _mm512_setzero_pd();
        let mut repl2 = _mm512_setzero_pd();

        let cof00 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[4].as_ptr()),
            _mm512_loadu_pd(d[8].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[7].as_ptr()),
            ),
        );
        det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[0].as_ptr()), cof00, det_v);
        repl0 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[0].as_ptr()), cof00, repl0);
        let cof01 = _mm512_sub_pd(
            _mm512_setzero_pd(),
            _mm512_fmsub_pd(
                _mm512_loadu_pd(d[3].as_ptr()),
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[5].as_ptr()),
                    _mm512_loadu_pd(d[6].as_ptr()),
                ),
            ),
        );
        det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[1].as_ptr()), cof01, det_v);
        repl0 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[1].as_ptr()), cof01, repl0);
        let cof02 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[3].as_ptr()),
            _mm512_loadu_pd(d[7].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[6].as_ptr()),
            ),
        );
        det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[2].as_ptr()), cof02, det_v);
        repl0 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[2].as_ptr()), cof02, repl0);
        let cof10 = _mm512_sub_pd(
            _mm512_setzero_pd(),
            _mm512_fmsub_pd(
                _mm512_loadu_pd(d[1].as_ptr()),
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[2].as_ptr()),
                    _mm512_loadu_pd(d[7].as_ptr()),
                ),
            ),
        );
        repl1 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[3].as_ptr()), cof10, repl1);
        let cof11 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[0].as_ptr()),
            _mm512_loadu_pd(d[8].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[2].as_ptr()),
                _mm512_loadu_pd(d[6].as_ptr()),
            ),
        );
        repl1 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[4].as_ptr()), cof11, repl1);
        let cof12 = _mm512_sub_pd(
            _mm512_setzero_pd(),
            _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                _mm512_loadu_pd(d[7].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[1].as_ptr()),
                    _mm512_loadu_pd(d[6].as_ptr()),
                ),
            ),
        );
        repl1 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[5].as_ptr()), cof12, repl1);
        let cof20 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[1].as_ptr()),
            _mm512_loadu_pd(d[5].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[2].as_ptr()),
                _mm512_loadu_pd(d[4].as_ptr()),
            ),
        );
        repl2 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[6].as_ptr()), cof20, repl2);
        let cof21 = _mm512_sub_pd(
            _mm512_setzero_pd(),
            _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[2].as_ptr()),
                    _mm512_loadu_pd(d[3].as_ptr()),
                ),
            ),
        );
        repl2 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[7].as_ptr()), cof21, repl2);
        let cof22 = _mm512_fmsub_pd(
            _mm512_loadu_pd(d[0].as_ptr()),
            _mm512_loadu_pd(d[4].as_ptr()),
            _mm512_mul_pd(
                _mm512_loadu_pd(d[1].as_ptr()),
                _mm512_loadu_pd(d[3].as_ptr()),
            ),
        );
        repl2 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[8].as_ptr()), cof22, repl2);
        let repl01 = _mm512_add_pd(repl0, repl1);
        let repl_v = _mm512_add_pd(repl01, repl2);
        let pref_v = _mm512_set1_pd(pref);
        let f0_v = _mm512_set1_pd(f0);
        let overlap_v = _mm512_mul_pd(det_v, pref_v);
        let contrib = _mm512_fmsub_pd(det_v, f0_v, repl_v);
        let fock_v = _mm512_mul_pd(contrib, pref_v);

        let mut det_lane = [0.0f64; 8];
        let mut overlap_lane = [0.0f64; 8];
        let mut fock_lane = [0.0f64; 8];
        _mm512_storeu_pd(det_lane.as_mut_ptr(), det_v);
        _mm512_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
        _mm512_storeu_pd(fock_lane.as_mut_ptr(), fock_v);
        for lane in 0..8 {
            if det_lane[lane].is_finite() {
                overlap[indices[lane]] = overlap_lane[lane];
                fock[indices[lane]] = fock_lane[lane];
            } else {
                overlap[indices[lane]] = 0.0;
                fock[indices[lane]] = 0.0;
            }
        }
    }
}

/// Prepare and evaluate the fixed-rank `L = 4` overlap and generalised-Fock matrix element for `m = 0`:
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}},`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}`
/// `\mathcal F_{\eta z}^{(0,0)}].`
/// The determinant labels are constructed first. The real-valued path loads the sixteen contraction
/// entries directly from `X^{(0)}` and `Y^{(0)}`, forms four cofactor rows with packed AVX FMA
/// operations and immediately contracts them with the generalised-Fock entries. The generic scalar
/// path materialises the local `4 x 4` contraction determinant before applying the shared adjugate routine.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for determinant labels and generic adjugate work arrays.
/// - `tol`: Numerical tolerance used when evaluating the generic adjugate-transpose matrix.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 4` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l4_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l4, {
        scratch.ensure_same(4);

        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let n = w.n();
        let r0 = scratch.rows[0];
        let r1 = scratch.rows[1];
        let r2 = scratch.rows[2];
        let r3 = scratch.rows[3];
        let c0 = scratch.cols[0];
        let c1 = scratch.cols[1];
        let c2 = scratch.cols[2];
        let c3 = scratch.cols[3];
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let fsl = w.ff_t_slice(0, 0);
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);

        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            unsafe {
                let x0 = std::slice::from_raw_parts(x0.as_ptr().cast::<f64>(), x0.len());
                let y0 = std::slice::from_raw_parts(y0.as_ptr().cast::<f64>(), y0.len());
                let fsl = std::slice::from_raw_parts(fsl.as_ptr().cast::<f64>(), fsl.len());

                let a00 = x0[r0 * n + c0];
                let a01 = y0[r0 * n + c1];
                let a02 = y0[r0 * n + c2];
                let a03 = y0[r0 * n + c3];
                let a10 = x0[r1 * n + c0];
                let a11 = x0[r1 * n + c1];
                let a12 = y0[r1 * n + c2];
                let a13 = y0[r1 * n + c3];
                let a20 = x0[r2 * n + c0];
                let a21 = x0[r2 * n + c1];
                let a22 = x0[r2 * n + c2];
                let a23 = y0[r2 * n + c3];
                let a30 = x0[r3 * n + c0];
                let a31 = x0[r3 * n + c1];
                let a32 = x0[r3 * n + c2];
                let a33 = x0[r3 * n + c3];

                let r0v = [a00, a01, a02, a03];
                let r1v = [a10, a11, a12, a13];
                let r2v = [a20, a21, a22, a23];
                let r3v = [a30, a31, a32, a33];

                let x0v = _mm256_set_pd(r1v[0], r1v[0], r1v[0], r1v[1]);
                let x1v = _mm256_set_pd(r1v[1], r1v[1], r1v[2], r1v[2]);
                let x2v = _mm256_set_pd(r1v[2], r1v[3], r1v[3], r1v[3]);
                let y0v = _mm256_set_pd(r2v[0], r2v[0], r2v[0], r2v[1]);
                let y1v = _mm256_set_pd(r2v[1], r2v[1], r2v[2], r2v[2]);
                let y2v = _mm256_set_pd(r2v[2], r2v[3], r2v[3], r2v[3]);
                let z0v = _mm256_set_pd(r3v[0], r3v[0], r3v[0], r3v[1]);
                let z1v = _mm256_set_pd(r3v[1], r3v[1], r3v[2], r3v[2]);
                let z2v = _mm256_set_pd(r3v[2], r3v[3], r3v[3], r3v[3]);
                let m0 = _mm256_fmsub_pd(y1v, z2v, _mm256_mul_pd(y2v, z1v));
                let m1 = _mm256_fmsub_pd(y0v, z2v, _mm256_mul_pd(y2v, z0v));
                let m2 = _mm256_fmsub_pd(y0v, z1v, _mm256_mul_pd(y1v, z0v));
                let minors01 = _mm256_fmsub_pd(x0v, m0, _mm256_mul_pd(x1v, m1));
                let minors = _mm256_fmadd_pd(x2v, m2, minors01);
                let cof0 = _mm256_mul_pd(minors, _mm256_set_pd(-1.0, 1.0, -1.0, 1.0));

                let x0v = _mm256_set_pd(r0v[0], r0v[0], r0v[0], r0v[1]);
                let x1v = _mm256_set_pd(r0v[1], r0v[1], r0v[2], r0v[2]);
                let x2v = _mm256_set_pd(r0v[2], r0v[3], r0v[3], r0v[3]);
                let y0v = _mm256_set_pd(r2v[0], r2v[0], r2v[0], r2v[1]);
                let y1v = _mm256_set_pd(r2v[1], r2v[1], r2v[2], r2v[2]);
                let y2v = _mm256_set_pd(r2v[2], r2v[3], r2v[3], r2v[3]);
                let z0v = _mm256_set_pd(r3v[0], r3v[0], r3v[0], r3v[1]);
                let z1v = _mm256_set_pd(r3v[1], r3v[1], r3v[2], r3v[2]);
                let z2v = _mm256_set_pd(r3v[2], r3v[3], r3v[3], r3v[3]);
                let m0 = _mm256_fmsub_pd(y1v, z2v, _mm256_mul_pd(y2v, z1v));
                let m1 = _mm256_fmsub_pd(y0v, z2v, _mm256_mul_pd(y2v, z0v));
                let m2 = _mm256_fmsub_pd(y0v, z1v, _mm256_mul_pd(y1v, z0v));
                let minors01 = _mm256_fmsub_pd(x0v, m0, _mm256_mul_pd(x1v, m1));
                let minors = _mm256_fmadd_pd(x2v, m2, minors01);
                let cof1 = _mm256_mul_pd(minors, _mm256_set_pd(1.0, -1.0, 1.0, -1.0));

                let x0v = _mm256_set_pd(r0v[0], r0v[0], r0v[0], r0v[1]);
                let x1v = _mm256_set_pd(r0v[1], r0v[1], r0v[2], r0v[2]);
                let x2v = _mm256_set_pd(r0v[2], r0v[3], r0v[3], r0v[3]);
                let y0v = _mm256_set_pd(r1v[0], r1v[0], r1v[0], r1v[1]);
                let y1v = _mm256_set_pd(r1v[1], r1v[1], r1v[2], r1v[2]);
                let y2v = _mm256_set_pd(r1v[2], r1v[3], r1v[3], r1v[3]);
                let z0v = _mm256_set_pd(r3v[0], r3v[0], r3v[0], r3v[1]);
                let z1v = _mm256_set_pd(r3v[1], r3v[1], r3v[2], r3v[2]);
                let z2v = _mm256_set_pd(r3v[2], r3v[3], r3v[3], r3v[3]);
                let m0 = _mm256_fmsub_pd(y1v, z2v, _mm256_mul_pd(y2v, z1v));
                let m1 = _mm256_fmsub_pd(y0v, z2v, _mm256_mul_pd(y2v, z0v));
                let m2 = _mm256_fmsub_pd(y0v, z1v, _mm256_mul_pd(y1v, z0v));
                let minors01 = _mm256_fmsub_pd(x0v, m0, _mm256_mul_pd(x1v, m1));
                let minors = _mm256_fmadd_pd(x2v, m2, minors01);
                let cof2 = _mm256_mul_pd(minors, _mm256_set_pd(-1.0, 1.0, -1.0, 1.0));

                let x0v = _mm256_set_pd(r0v[0], r0v[0], r0v[0], r0v[1]);
                let x1v = _mm256_set_pd(r0v[1], r0v[1], r0v[2], r0v[2]);
                let x2v = _mm256_set_pd(r0v[2], r0v[3], r0v[3], r0v[3]);
                let y0v = _mm256_set_pd(r1v[0], r1v[0], r1v[0], r1v[1]);
                let y1v = _mm256_set_pd(r1v[1], r1v[1], r1v[2], r1v[2]);
                let y2v = _mm256_set_pd(r1v[2], r1v[3], r1v[3], r1v[3]);
                let z0v = _mm256_set_pd(r2v[0], r2v[0], r2v[0], r2v[1]);
                let z1v = _mm256_set_pd(r2v[1], r2v[1], r2v[2], r2v[2]);
                let z2v = _mm256_set_pd(r2v[2], r2v[3], r2v[3], r2v[3]);
                let m0 = _mm256_fmsub_pd(y1v, z2v, _mm256_mul_pd(y2v, z1v));
                let m1 = _mm256_fmsub_pd(y0v, z2v, _mm256_mul_pd(y2v, z0v));
                let m2 = _mm256_fmsub_pd(y0v, z1v, _mm256_mul_pd(y1v, z0v));
                let minors01 = _mm256_fmsub_pd(x0v, m0, _mm256_mul_pd(x1v, m1));
                let minors = _mm256_fmadd_pd(x2v, m2, minors01);
                let cof3 = _mm256_mul_pd(minors, _mm256_set_pd(1.0, -1.0, 1.0, -1.0));

                let det_row = _mm256_set_pd(a03, a02, a01, a00);
                let det_products = _mm256_mul_pd(det_row, cof0);
                let det_sums = _mm256_hadd_pd(det_products, det_products);
                let det_low = _mm256_castpd256_pd128(det_sums);
                let det_high = _mm256_extractf128_pd(det_sums, 1);
                let det = _mm_cvtsd_f64(_mm_add_sd(det_low, det_high));

                if !det.is_finite() {
                    return (zero, zero);
                }

                let frow0 = _mm256_set_pd(
                    fsl[c3 * n + r0],
                    fsl[c2 * n + r0],
                    fsl[c1 * n + r0],
                    fsl[c0 * n + r0],
                );
                let frow1 = _mm256_set_pd(
                    fsl[c3 * n + r1],
                    fsl[c2 * n + r1],
                    fsl[c1 * n + r1],
                    fsl[c0 * n + r1],
                );
                let frow2 = _mm256_set_pd(
                    fsl[c3 * n + r2],
                    fsl[c2 * n + r2],
                    fsl[c1 * n + r2],
                    fsl[c0 * n + r2],
                );
                let frow3 = _mm256_set_pd(
                    fsl[c3 * n + r3],
                    fsl[c2 * n + r3],
                    fsl[c1 * n + r3],
                    fsl[c0 * n + r3],
                );
                let repl01 = _mm256_fmadd_pd(cof1, frow1, _mm256_mul_pd(cof0, frow0));
                let repl23 = _mm256_fmadd_pd(cof3, frow3, _mm256_mul_pd(cof2, frow2));
                let repl_v = _mm256_add_pd(repl01, repl23);
                let repl_sums = _mm256_hadd_pd(repl_v, repl_v);
                let repl_low = _mm256_castpd256_pd128(repl_sums);
                let repl_high = _mm256_extractf128_pd(repl_sums, 1);
                let repl = _mm_cvtsd_f64(_mm_add_sd(repl_low, repl_high));

                let det_t = T::from_real(det);
                let overlap = pref * det_t;
                let fock = pref * (det_t * w.f0f[0] - T::from_real(repl));

                return (overlap, fock);
            }
        }

        let det0 = [
            x0[r0 * n + c0],
            y0[r0 * n + c1],
            y0[r0 * n + c2],
            y0[r0 * n + c3],
            x0[r1 * n + c0],
            x0[r1 * n + c1],
            y0[r1 * n + c2],
            y0[r1 * n + c3],
            x0[r2 * n + c0],
            x0[r2 * n + c1],
            x0[r2 * n + c2],
            y0[r2 * n + c3],
            x0[r3 * n + c0],
            x0[r3 * n + c1],
            x0[r3 * n + c2],
            x0[r3 * n + c3],
        ];

        if let Some(det_det) = adjugate_transpose(
            scratch.adjt_det.as_mut_slice(),
            scratch.invs.as_mut_slice(),
            scratch.lu.as_mut_slice(),
            &det0,
            4,
            tol,
        ) {
            let cof = scratch.adjt_det.as_slice();
            let f00 = fsl[c0 * n + r0];
            let f10 = fsl[c0 * n + r1];
            let f20 = fsl[c0 * n + r2];
            let f30 = fsl[c0 * n + r3];
            let f01 = fsl[c1 * n + r0];
            let f11 = fsl[c1 * n + r1];
            let f21 = fsl[c1 * n + r2];
            let f31 = fsl[c1 * n + r3];
            let f02 = fsl[c2 * n + r0];
            let f12 = fsl[c2 * n + r1];
            let f22 = fsl[c2 * n + r2];
            let f32 = fsl[c2 * n + r3];
            let f03 = fsl[c3 * n + r0];
            let f13 = fsl[c3 * n + r1];
            let f23 = fsl[c3 * n + r2];
            let f33 = fsl[c3 * n + r3];
            let repl = cof[0] * f00
                + cof[4] * f10
                + cof[8] * f20
                + cof[12] * f30
                + cof[1] * f01
                + cof[5] * f11
                + cof[9] * f21
                + cof[13] * f31
                + cof[2] * f02
                + cof[6] * f12
                + cof[10] * f22
                + cof[14] * f32
                + cof[3] * f03
                + cof[7] * f13
                + cof[11] * f23
                + cof[15] * f33;

            let overlap = pref * det_det;
            let fock = pref * (det_det * w.f0f[0] - repl);

            (overlap, fock)
        } else {
            let overlap = pref * det(&det0, 4).unwrap_or(zero);
            (overlap, zero)
        }
    })
}

/// Prepare and evaluate 4 independent real fixed-rank `L = 4` matrix elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX2/FMA` arithmetic evaluates the same
/// determinant, cofactor and generalised-Fock algebra for 4 independent excitation pairs without
/// horizontal reductions between pairs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l_ex`: Bra excitations for the enclosing batch.
/// - `g_ex`: Ket excitations for the enclosing batch.
/// - `indices`: Positions of the 4 equal-rank pairs within the enclosing batch.
/// - `overlap`: Real overlap output slice.
/// - `fock`: Real generalised-Fock output slice.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements at `indices`.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
/// - Every entry in `indices` must address valid elements of `l_ex`, `g_ex`, `overlap`, and `fock`.
/// - Excitation labels must define valid contraction indices for the dimensions stored in `w`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_l4_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &[ExcitationSpin],
    g_ex: &[ExcitationSpin],
    indices: [usize; 4],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        let n = w.n();
        let x0_t = w.x_slice(0);
        let y0_t = w.y_slice(0);
        let fsl_t = w.ff_t_slice(0, 0);
        let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
        let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
        let fsl = std::slice::from_raw_parts(fsl_t.as_ptr().cast::<f64>(), fsl_t.len());
        let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
        let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
        let pref = phase * w.tilde_s_prod;

        // Preserve the four contraction-label sets because the generalised-Fock matrix
        // element is consumed once per cofactor and therefore does not need its own
        // 16 x 4 scalar staging buffer.
        let mut rows = [[0usize; 4]; 4];
        let mut cols = [[0usize; 4]; 4];
        let mut d = [[0.0f64; 4]; 16];

        for lane in 0..4 {
            let i = *indices.get_unchecked(lane);

            construct_determinant_indices(
                l_ex.get_unchecked(i),
                g_ex.get_unchecked(i),
                w,
                &mut rows[lane],
                &mut cols[lane],
            );

            let r0 = rows[lane][0];
            let r1 = rows[lane][1];
            let r2 = rows[lane][2];
            let r3 = rows[lane][3];
            let c0 = cols[lane][0];
            let c1 = cols[lane][1];
            let c2 = cols[lane][2];
            let c3 = cols[lane][3];

            d[0][lane] = *x0.get_unchecked(r0 * n + c0);
            d[1][lane] = *y0.get_unchecked(r0 * n + c1);
            d[2][lane] = *y0.get_unchecked(r0 * n + c2);
            d[3][lane] = *y0.get_unchecked(r0 * n + c3);

            d[4][lane] = *x0.get_unchecked(r1 * n + c0);
            d[5][lane] = *x0.get_unchecked(r1 * n + c1);
            d[6][lane] = *y0.get_unchecked(r1 * n + c2);
            d[7][lane] = *y0.get_unchecked(r1 * n + c3);

            d[8][lane] = *x0.get_unchecked(r2 * n + c0);
            d[9][lane] = *x0.get_unchecked(r2 * n + c1);
            d[10][lane] = *x0.get_unchecked(r2 * n + c2);
            d[11][lane] = *y0.get_unchecked(r2 * n + c3);

            d[12][lane] = *x0.get_unchecked(r3 * n + c0);
            d[13][lane] = *x0.get_unchecked(r3 * n + c1);
            d[14][lane] = *x0.get_unchecked(r3 * n + c2);
            d[15][lane] = *x0.get_unchecked(r3 * n + c3);
        }

        // Each Fock element is used exactly once. Build the four-lane vector directly
        // from the four pair-specific contraction labels instead of staging 64 scalars
        // to the stack and loading them back into a YMM register.
        macro_rules! fvec {
            ($row:expr, $col:expr) => {{
                _mm256_set_pd(
                    *fsl.get_unchecked(cols[3][$col] * n + rows[3][$row]),
                    *fsl.get_unchecked(cols[2][$col] * n + rows[2][$row]),
                    *fsl.get_unchecked(cols[1][$col] * n + rows[1][$row]),
                    *fsl.get_unchecked(cols[0][$col] * n + rows[0][$row]),
                )
            }};
        }

        let mut det_v = _mm256_setzero_pd();
        let mut repl0 = _mm256_setzero_pd();
        let mut repl1 = _mm256_setzero_pd();
        let mut repl2 = _mm256_setzero_pd();
        let mut repl3 = _mm256_setzero_pd();

        let cof00 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[10].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[10].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[6].as_ptr()), m1),
            );
            _mm256_fmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m2, t)
        };
        det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[0].as_ptr()), cof00, det_v);
        repl0 = _mm256_fmadd_pd(fvec!(0, 0), cof00, repl0);

        let cof01 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[10].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[10].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[6].as_ptr()), m1),
            );
            let minor = _mm256_fmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m2, t);
            _mm256_sub_pd(_mm256_setzero_pd(), minor)
        };
        det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[1].as_ptr()), cof01, det_v);
        repl0 = _mm256_fmadd_pd(fvec!(0, 1), cof01, repl0);

        let cof02 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[13].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[9].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[5].as_ptr()), m1),
            );
            _mm256_fmadd_pd(_mm256_loadu_pd(d[7].as_ptr()), m2, t)
        };
        det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[2].as_ptr()), cof02, det_v);
        repl0 = _mm256_fmadd_pd(fvec!(0, 2), cof02, repl0);

        let cof03 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[10].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[10].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[13].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[9].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[5].as_ptr()), m1),
            );
            let minor = _mm256_fmadd_pd(_mm256_loadu_pd(d[6].as_ptr()), m2, t);
            _mm256_sub_pd(_mm256_setzero_pd(), minor)
        };
        det_v = _mm256_fmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), cof03, det_v);
        repl0 = _mm256_fmadd_pd(fvec!(0, 3), cof03, repl0);

        let cof10 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[10].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[10].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[1].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[2].as_ptr()), m1),
            );
            let minor = _mm256_fmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m2, t);
            _mm256_sub_pd(_mm256_setzero_pd(), minor)
        };
        repl1 = _mm256_fmadd_pd(fvec!(1, 0), cof10, repl1);

        let cof11 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[10].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[10].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[2].as_ptr()), m1),
            );
            _mm256_fmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m2, t)
        };
        repl1 = _mm256_fmadd_pd(fvec!(1, 1), cof11, repl1);

        let cof12 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[11].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[13].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[9].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[1].as_ptr()), m1),
            );
            let minor = _mm256_fmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m2, t);
            _mm256_sub_pd(_mm256_setzero_pd(), minor)
        };
        repl1 = _mm256_fmadd_pd(fvec!(1, 2), cof12, repl1);

        let cof13 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[10].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[10].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[8].as_ptr()),
                _mm256_loadu_pd(d[13].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[9].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[1].as_ptr()), m1),
            );
            _mm256_fmadd_pd(_mm256_loadu_pd(d[2].as_ptr()), m2, t)
        };
        repl1 = _mm256_fmadd_pd(fvec!(1, 3), cof13, repl1);

        let cof20 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[6].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[6].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[1].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[2].as_ptr()), m1),
            );
            _mm256_fmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m2, t)
        };
        repl2 = _mm256_fmadd_pd(fvec!(2, 0), cof20, repl2);

        let cof21 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[6].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[6].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[2].as_ptr()), m1),
            );
            let minor = _mm256_fmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m2, t);
            _mm256_sub_pd(_mm256_setzero_pd(), minor)
        };
        repl2 = _mm256_fmadd_pd(fvec!(2, 1), cof21, repl2);

        let cof22 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[15].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[13].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[5].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[1].as_ptr()), m1),
            );
            _mm256_fmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m2, t)
        };
        repl2 = _mm256_fmadd_pd(fvec!(2, 2), cof22, repl2);

        let cof23 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[6].as_ptr()),
                    _mm256_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[14].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[6].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[13].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[5].as_ptr()),
                    _mm256_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[1].as_ptr()), m1),
            );
            let minor = _mm256_fmadd_pd(_mm256_loadu_pd(d[2].as_ptr()), m2, t);
            _mm256_sub_pd(_mm256_setzero_pd(), minor)
        };
        repl2 = _mm256_fmadd_pd(fvec!(2, 3), cof23, repl2);

        let cof30 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[6].as_ptr()),
                _mm256_loadu_pd(d[11].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[10].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[11].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[9].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[10].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[6].as_ptr()),
                    _mm256_loadu_pd(d[9].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[1].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[2].as_ptr()), m1),
            );
            let minor = _mm256_fmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m2, t);
            _mm256_sub_pd(_mm256_setzero_pd(), minor)
        };
        repl3 = _mm256_fmadd_pd(fvec!(3, 0), cof30, repl3);

        let cof31 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[6].as_ptr()),
                _mm256_loadu_pd(d[11].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[10].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[11].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[8].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[10].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[6].as_ptr()),
                    _mm256_loadu_pd(d[8].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[2].as_ptr()), m1),
            );
            _mm256_fmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m2, t)
        };
        repl3 = _mm256_fmadd_pd(fvec!(3, 1), cof31, repl3);

        let cof32 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[11].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[9].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[11].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[7].as_ptr()),
                    _mm256_loadu_pd(d[8].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[5].as_ptr()),
                    _mm256_loadu_pd(d[8].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[1].as_ptr()), m1),
            );
            let minor = _mm256_fmadd_pd(_mm256_loadu_pd(d[3].as_ptr()), m2, t);
            _mm256_sub_pd(_mm256_setzero_pd(), minor)
        };
        repl3 = _mm256_fmadd_pd(fvec!(3, 2), cof32, repl3);

        let cof33 = {
            let m0 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[5].as_ptr()),
                _mm256_loadu_pd(d[10].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[6].as_ptr()),
                    _mm256_loadu_pd(d[9].as_ptr()),
                ),
            );
            let m1 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[10].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[6].as_ptr()),
                    _mm256_loadu_pd(d[8].as_ptr()),
                ),
            );
            let m2 = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[4].as_ptr()),
                _mm256_loadu_pd(d[9].as_ptr()),
                _mm256_mul_pd(
                    _mm256_loadu_pd(d[5].as_ptr()),
                    _mm256_loadu_pd(d[8].as_ptr()),
                ),
            );
            let t = _mm256_fmsub_pd(
                _mm256_loadu_pd(d[0].as_ptr()),
                m0,
                _mm256_mul_pd(_mm256_loadu_pd(d[1].as_ptr()), m1),
            );
            _mm256_fmadd_pd(_mm256_loadu_pd(d[2].as_ptr()), m2, t)
        };
        repl3 = _mm256_fmadd_pd(fvec!(3, 3), cof33, repl3);

        let repl01 = _mm256_add_pd(repl0, repl1);
        let repl23 = _mm256_add_pd(repl2, repl3);
        let repl_v = _mm256_add_pd(repl01, repl23);
        let pref_v = _mm256_set1_pd(pref);
        let f0_v = _mm256_set1_pd(f0);
        let overlap_v = _mm256_mul_pd(det_v, pref_v);
        let contrib = _mm256_fmsub_pd(det_v, f0_v, repl_v);
        let fock_v = _mm256_mul_pd(contrib, pref_v);

        let mut det_lane = [0.0f64; 4];
        let mut overlap_lane = [0.0f64; 4];
        let mut fock_lane = [0.0f64; 4];

        _mm256_storeu_pd(det_lane.as_mut_ptr(), det_v);
        _mm256_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
        _mm256_storeu_pd(fock_lane.as_mut_ptr(), fock_v);

        for lane in 0..4 {
            let i = *indices.get_unchecked(lane);

            if det_lane[lane].is_finite() {
                *overlap.get_unchecked_mut(i) = overlap_lane[lane];
                *fock.get_unchecked_mut(i) = fock_lane[lane];
            } else {
                *overlap.get_unchecked_mut(i) = 0.0;
                *fock.get_unchecked_mut(i) = 0.0;
            }
        }
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 4` matrix elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates the same
/// determinant, cofactor and generalised-Fock algebra for 8 independent excitation pairs without
/// horizontal reductions between pairs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l_ex`: Bra excitations for the enclosing batch.
/// - `g_ex`: Ket excitations for the enclosing batch.
/// - `indices`: Positions of the 8 equal-rank pairs within the enclosing batch.
/// - `overlap`: Real overlap output slice.
/// - `fock`: Real generalised-Fock output slice.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements at `indices`.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_l4_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &[ExcitationSpin],
    g_ex: &[ExcitationSpin],
    indices: [usize; 8],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        let n = w.n();
        let x0_t = w.x_slice(0);
        let y0_t = w.y_slice(0);
        let fsl_t = w.ff_t_slice(0, 0);
        let x0 = std::slice::from_raw_parts(x0_t.as_ptr().cast::<f64>(), x0_t.len());
        let y0 = std::slice::from_raw_parts(y0_t.as_ptr().cast::<f64>(), y0_t.len());
        let fsl = std::slice::from_raw_parts(fsl_t.as_ptr().cast::<f64>(), fsl_t.len());
        let phase = *std::ptr::from_ref(&w.phase).cast::<f64>();
        let f0 = *std::ptr::from_ref(&w.f0f[0]).cast::<f64>();
        let pref = phase * w.tilde_s_prod;
        let mut d = [[0.0f64; 8]; 16];
        let mut ff = [[0.0f64; 8]; 16];
        for lane in 0..8 {
            let i = indices[lane];
            let mut rows = [0usize; 4];
            let mut cols = [0usize; 4];
            construct_determinant_indices(&l_ex[i], &g_ex[i], w, &mut rows, &mut cols);
            d[0][lane] = x0[rows[0] * n + cols[0]];
            ff[0][lane] = fsl[cols[0] * n + rows[0]];
            d[1][lane] = y0[rows[0] * n + cols[1]];
            ff[1][lane] = fsl[cols[1] * n + rows[0]];
            d[2][lane] = y0[rows[0] * n + cols[2]];
            ff[2][lane] = fsl[cols[2] * n + rows[0]];
            d[3][lane] = y0[rows[0] * n + cols[3]];
            ff[3][lane] = fsl[cols[3] * n + rows[0]];
            d[4][lane] = x0[rows[1] * n + cols[0]];
            ff[4][lane] = fsl[cols[0] * n + rows[1]];
            d[5][lane] = x0[rows[1] * n + cols[1]];
            ff[5][lane] = fsl[cols[1] * n + rows[1]];
            d[6][lane] = y0[rows[1] * n + cols[2]];
            ff[6][lane] = fsl[cols[2] * n + rows[1]];
            d[7][lane] = y0[rows[1] * n + cols[3]];
            ff[7][lane] = fsl[cols[3] * n + rows[1]];
            d[8][lane] = x0[rows[2] * n + cols[0]];
            ff[8][lane] = fsl[cols[0] * n + rows[2]];
            d[9][lane] = x0[rows[2] * n + cols[1]];
            ff[9][lane] = fsl[cols[1] * n + rows[2]];
            d[10][lane] = x0[rows[2] * n + cols[2]];
            ff[10][lane] = fsl[cols[2] * n + rows[2]];
            d[11][lane] = y0[rows[2] * n + cols[3]];
            ff[11][lane] = fsl[cols[3] * n + rows[2]];
            d[12][lane] = x0[rows[3] * n + cols[0]];
            ff[12][lane] = fsl[cols[0] * n + rows[3]];
            d[13][lane] = x0[rows[3] * n + cols[1]];
            ff[13][lane] = fsl[cols[1] * n + rows[3]];
            d[14][lane] = x0[rows[3] * n + cols[2]];
            ff[14][lane] = fsl[cols[2] * n + rows[3]];
            d[15][lane] = x0[rows[3] * n + cols[3]];
            ff[15][lane] = fsl[cols[3] * n + rows[3]];
        }
        let mut det_v = _mm512_setzero_pd();
        let mut repl0 = _mm512_setzero_pd();
        let mut repl1 = _mm512_setzero_pd();
        let mut repl2 = _mm512_setzero_pd();
        let mut repl3 = _mm512_setzero_pd();

        let cof00 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[10].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[10].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[6].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m2, t);
            minor
        };
        det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[0].as_ptr()), cof00, det_v);
        repl0 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[0].as_ptr()), cof00, repl0);
        let cof01 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[10].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[10].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[6].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m2, t);
            _mm512_sub_pd(_mm512_setzero_pd(), minor)
        };
        det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[1].as_ptr()), cof01, det_v);
        repl0 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[1].as_ptr()), cof01, repl0);
        let cof02 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[13].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[9].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[5].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[7].as_ptr()), m2, t);
            minor
        };
        det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[2].as_ptr()), cof02, det_v);
        repl0 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[2].as_ptr()), cof02, repl0);
        let cof03 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[10].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[10].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[13].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[9].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[5].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[6].as_ptr()), m2, t);
            _mm512_sub_pd(_mm512_setzero_pd(), minor)
        };
        det_v = _mm512_fmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), cof03, det_v);
        repl0 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[3].as_ptr()), cof03, repl0);
        let cof10 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[10].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[10].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[1].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[2].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m2, t);
            _mm512_sub_pd(_mm512_setzero_pd(), minor)
        };
        repl1 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[4].as_ptr()), cof10, repl1);
        let cof11 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[10].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[10].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[2].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m2, t);
            minor
        };
        repl1 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[5].as_ptr()), cof11, repl1);
        let cof12 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[11].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[13].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[9].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[1].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m2, t);
            _mm512_sub_pd(_mm512_setzero_pd(), minor)
        };
        repl1 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[6].as_ptr()), cof12, repl1);
        let cof13 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[10].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[10].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[8].as_ptr()),
                _mm512_loadu_pd(d[13].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[9].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[1].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[2].as_ptr()), m2, t);
            minor
        };
        repl1 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[7].as_ptr()), cof13, repl1);
        let cof20 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[6].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[6].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[1].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[2].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m2, t);
            minor
        };
        repl2 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[8].as_ptr()), cof20, repl2);
        let cof21 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[6].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[14].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[6].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[2].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m2, t);
            _mm512_sub_pd(_mm512_setzero_pd(), minor)
        };
        repl2 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[9].as_ptr()), cof21, repl2);
        let cof22 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[15].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[13].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[5].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[1].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m2, t);
            minor
        };
        repl2 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[10].as_ptr()), cof22, repl2);
        let cof23 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[6].as_ptr()),
                    _mm512_loadu_pd(d[13].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[14].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[6].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[13].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[5].as_ptr()),
                    _mm512_loadu_pd(d[12].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[1].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[2].as_ptr()), m2, t);
            _mm512_sub_pd(_mm512_setzero_pd(), minor)
        };
        repl2 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[11].as_ptr()), cof23, repl2);
        let cof30 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[6].as_ptr()),
                _mm512_loadu_pd(d[11].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[10].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[11].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[9].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[10].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[6].as_ptr()),
                    _mm512_loadu_pd(d[9].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[1].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[2].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m2, t);
            _mm512_sub_pd(_mm512_setzero_pd(), minor)
        };
        repl3 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[12].as_ptr()), cof30, repl3);
        let cof31 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[6].as_ptr()),
                _mm512_loadu_pd(d[11].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[10].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[11].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[8].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[10].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[6].as_ptr()),
                    _mm512_loadu_pd(d[8].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[2].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m2, t);
            minor
        };
        repl3 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[13].as_ptr()), cof31, repl3);
        let cof32 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[11].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[9].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[11].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[7].as_ptr()),
                    _mm512_loadu_pd(d[8].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[5].as_ptr()),
                    _mm512_loadu_pd(d[8].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[1].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[3].as_ptr()), m2, t);
            _mm512_sub_pd(_mm512_setzero_pd(), minor)
        };
        repl3 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[14].as_ptr()), cof32, repl3);
        let cof33 = {
            let m0 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[5].as_ptr()),
                _mm512_loadu_pd(d[10].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[6].as_ptr()),
                    _mm512_loadu_pd(d[9].as_ptr()),
                ),
            );
            let m1 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[10].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[6].as_ptr()),
                    _mm512_loadu_pd(d[8].as_ptr()),
                ),
            );
            let m2 = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[4].as_ptr()),
                _mm512_loadu_pd(d[9].as_ptr()),
                _mm512_mul_pd(
                    _mm512_loadu_pd(d[5].as_ptr()),
                    _mm512_loadu_pd(d[8].as_ptr()),
                ),
            );
            let t = _mm512_fmsub_pd(
                _mm512_loadu_pd(d[0].as_ptr()),
                m0,
                _mm512_mul_pd(_mm512_loadu_pd(d[1].as_ptr()), m1),
            );
            let minor = _mm512_fmadd_pd(_mm512_loadu_pd(d[2].as_ptr()), m2, t);
            minor
        };
        repl3 = _mm512_fmadd_pd(_mm512_loadu_pd(ff[15].as_ptr()), cof33, repl3);
        let repl01 = _mm512_add_pd(repl0, repl1);
        let repl23 = _mm512_add_pd(repl2, repl3);
        let repl_v = _mm512_add_pd(repl01, repl23);
        let pref_v = _mm512_set1_pd(pref);
        let f0_v = _mm512_set1_pd(f0);
        let overlap_v = _mm512_mul_pd(det_v, pref_v);
        let contrib = _mm512_fmsub_pd(det_v, f0_v, repl_v);
        let fock_v = _mm512_mul_pd(contrib, pref_v);

        let mut det_lane = [0.0f64; 8];
        let mut overlap_lane = [0.0f64; 8];
        let mut fock_lane = [0.0f64; 8];
        _mm512_storeu_pd(det_lane.as_mut_ptr(), det_v);
        _mm512_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
        _mm512_storeu_pd(fock_lane.as_mut_ptr(), fock_v);
        for lane in 0..8 {
            if det_lane[lane].is_finite() {
                overlap[indices[lane]] = overlap_lane[lane];
                fock[indices[lane]] = fock_lane[lane];
            } else {
                overlap[indices[lane]] = 0.0;
                fock[indices[lane]] = 0.0;
            }
        }
    }
}

/// Prepare and evaluate the overlap and generalised-Fock matrix element for arbitrary `L` when `m = 0`:
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}},`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{z=1}^{L}\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}].`
/// `The determinant labels and \mathbf D_{\mathrm{ov}}(0,\ldots,0) are prepared once before the`
/// `cofactor evaluation. If the one-body adjugate path rejects the determinant, only the overlap`
/// `determinant is evaluated separately, preserving the numerical convention of the existing evaluator.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the contraction determinant and its cofactors.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate-transpose matrix.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for arbitrary `L` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_gen_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    l: usize,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_gen,
        {
            scratch.ensure_same(l);

            construct_determinant_indices(
                l_ex,
                g_ex,
                w,
                scratch.rows.as_mut_slice(),
                scratch.cols.as_mut_slice(),
            );

            let x0 = w.x(0);
            let y0 = w.y(0);
            build_d(
                scratch.det0.as_mut_slice(),
                l,
                &x0,
                &y0,
                scratch.rows.as_slice(),
                scratch.cols.as_slice(),
            );

            let zero = <T as From<f64>>::from(0.0);
            let n = w.n();
            let det0 = &scratch.det0.as_slice()[..l * l];
            let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

            if let Some(det_det) = adjugate_transpose(
                scratch.adjt_det.as_mut_slice(),
                scratch.invs.as_mut_slice(),
                scratch.lu.as_mut_slice(),
                det0,
                l,
                tol,
            ) {
                let mut contrib = det_det * w.f0f[0];
                let fsl = w.ff_t_slice(0, 0);

                for b in 0..l {
                    let cb = scratch.cols[b];
                    let base = cb * n;
                    let corr = column_replacement_correction(
                        l,
                        det0,
                        scratch.adjt_det.as_slice(),
                        b,
                        |r| fsl[base + scratch.rows[r]],
                    );
                    contrib -= det_det + corr;
                }

                (pref * det_det, pref * contrib)
            } else {
                let overlap = pref * det(det0, l).unwrap_or(zero);
                (overlap, zero)
            }
        }
    )
}

/// Prepare and evaluate the overlap and generalised-Fock matrix element when `m > 0` by summing
/// the allowed one-body distributions
/// `m_1 + \cdots + m_{L+1} = m, \qquad m_i \in \{0,1\}.`
/// `The determinant labels and endpoint contraction determinants`
/// `\mathbf D_{\mathrm{ov}}(0,\ldots,0)` and `\mathbf D_{\mathrm{ov}}(1,\ldots,1)` are built once.
/// `The first assignment selects {}^x F_0^{(m_1)} and the operator side of each`
/// `\mathcal F^{(m_1,m_j)} column; the remaining assignments select the columns of`
/// `\mathbf D_{\mathrm{ov}}. Terms with m_1 = 0 also satisfy the overlap constraint`
/// `m_2+\cdots+m_{L+1}=m and are accumulated into the overlap without a second distribution loop.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for endpoint and mixed contraction determinants, cofactors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` summed over all distributions.
#[inline(always)]
fn xw_f_overlap_gen_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_gen, {
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
        scratch.ensure_same(l);

        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let x0 = w.x(0);
        let y0 = w.y(0);
        build_d(
            scratch.det0.as_mut_slice(),
            l,
            &x0,
            &y0,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

        let x1 = w.x(1);
        let y1 = w.y(1);
        build_d(
            scratch.det1.as_mut_slice(),
            l,
            &x1,
            &y1,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

        let zero = <T as From<f64>>::from(0.0);
        let n = w.n();
        let mut overlap_acc = zero;
        let mut fock_acc = zero;

        mix_dets_same(w, l, 1, scratch, |bits, scratch| {
            let mi = bit(bits, 0);

            if let Some(det_det) = adjugate_transpose_generic(
                scratch.adjt_det.as_mut_slice(),
                scratch.det_mix.as_slice(),
                l,
                tol,
            ) {
                if mi == 0 {
                    overlap_acc += det_det;
                }

                let mut contrib = det_det * w.f0f[mi];
                let f0 = w.ff_t_slice(mi, 0);
                let f1 = w.ff_t_slice(mi, 1);

                for b in 0..l {
                    let mj = bit(bits, b + 1);
                    let cb = scratch.cols[b];
                    let fsl = if mj == 0 { f0 } else { f1 };
                    let base = cb * n;
                    let corr = column_replacement_correction(
                        l,
                        scratch.det_mix.as_slice(),
                        scratch.adjt_det.as_slice(),
                        b,
                        |r| fsl[base + scratch.rows[r]],
                    );
                    contrib -= det_det + corr;
                }

                fock_acc += contrib;
            } else if mi == 0 {
                overlap_acc += det(scratch.det_mix.as_slice(), l).unwrap_or(zero);
            }
        });

        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        (pref * overlap_acc, pref * fock_acc)
    })
}
