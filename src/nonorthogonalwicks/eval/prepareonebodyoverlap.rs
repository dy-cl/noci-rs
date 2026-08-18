// nonorthogonalwicks/eval/prepareonebodyoverlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_sd, _mm_cvtsd_f64, _mm256_add_pd, _mm256_castpd256_pd128, _mm256_extractf128_pd,
    _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_hadd_pd, _mm256_loadu_pd, _mm256_mul_pd,
    _mm256_set_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
    _mm512_add_pd, _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_set_pd,
    _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd,
};

// Crate-root imports.
use crate::maths::{adjugate_transpose, build_d, det};
use crate::noci::NOCIScalar;
use crate::time_call;
use crate::{ExcitationSpin, ExcitationSpinCache};

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, mix_dets_same,
};
use super::prepare::{
    construct_determinant_indices, construct_determinant_indices_l1,
    construct_determinant_indices_l2, construct_determinant_indices_l3,
};

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
/// - `x_ex`: `Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.`
/// - `w_ex`: `Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.`
/// - `scratch`: Scratch storage for determinant labels, generic contraction determinants and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)`.
#[inline(always)]
pub(crate) fn xw_f_overlap_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap, {
        if w.m == 0 {
            xw_f_overlap_m0_prepared(w, x_ex, w_ex, scratch, tol)
        } else {
            xw_f_overlap_gen_prepared(w, x_ex, w_ex, scratch, tol)
        }
    })
}

/// Prepare and evaluate the overlap and generalised-Fock matrix element together when `m = 0`.
/// `Every contraction uses m_i = 0, so the total excitation rank L = L_x + L_w determines one`
/// `contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0). Fixed-rank prepared kernels are`
/// `used for L = 1,\ldots,4; arbitrary ranks construct the generic determinant once and then apply`
/// `the general cofactor form. For L = 0, the overlap is {}^{xw}\tilde S and only`
/// `{}^x F_0^{(0)} contributes to F.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for determinant labels and generic work arrays.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0, {
        let l = x_ex.holes.count_ones() as usize + w_ex.holes.count_ones() as usize;

        match l {
            0 => {
                let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
                (pref, pref * w.f0f[0])
            }
            1 => xw_f_overlap_m0_l1_prepared(w, x_ex, w_ex, scratch),
            2 => xw_f_overlap_m0_l2_prepared(w, x_ex, w_ex, scratch),
            3 => xw_f_overlap_m0_l3_prepared(w, x_ex, w_ex, scratch, tol),
            4 => xw_f_overlap_m0_l4_prepared(w, x_ex, w_ex, scratch, tol),
            _ => xw_f_overlap_m0_gen_prepared(w, x_ex, w_ex, scratch, l, tol),
        }
    })
}

/// Dispatch 4 real `m = 0` matrix elements to the fixed-rank AVX2/FMA kernel.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `x_ex`: 4 x-reference excitation caches.
/// - `w_ex`: 4 w-reference excitation caches.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn xw_f_overlap_m0_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        match l {
            1 => xw_f_overlap_m0_l1_prepared_f64x4(w, x_ex, w_ex, overlap, fock),
            2 => xw_f_overlap_m0_l2_prepared_f64x4(w, x_ex, w_ex, overlap, fock),
            3 => xw_f_overlap_m0_l3_prepared_f64x4(w, x_ex, w_ex, overlap, fock),
            4 => xw_f_overlap_m0_l4_prepared_f64x4(w, x_ex, w_ex, overlap, fock),
            _ => unreachable!(),
        }
    }
}

/// Dispatch 8 real `m = 0` matrix elements to the fixed-rank AVX-512 kernel.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `x_ex`: 8 x-reference excitation caches.
/// - `w_ex`: 8 w-reference excitation caches.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn xw_f_overlap_m0_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    unsafe {
        match l {
            1 => xw_f_overlap_m0_l1_prepared_f64x8(w, x_ex, w_ex, overlap, fock),
            2 => xw_f_overlap_m0_l2_prepared_f64x8(w, x_ex, w_ex, overlap, fock),
            3 => xw_f_overlap_m0_l3_prepared_f64x8(w, x_ex, w_ex, overlap, fock),
            4 => xw_f_overlap_m0_l4_prepared_f64x8(w, x_ex, w_ex, overlap, fock),
            _ => unreachable!(),
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
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the rank-one row and column labels.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 1` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l1_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l1, {
        scratch.ensure_same_m0(1);
        scratch.rows.ensure(1);
        scratch.cols.ensure(1);

        construct_determinant_indices(
            x_ex,
            w_ex,
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
/// - `x_ex`: 4 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 4 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_l1_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
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
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);

            let mut rows = [0usize; 1];
            let mut cols = [0usize; 1];
            construct_determinant_indices_l1(
                x_data.rank,
                &x_data.indices,
                &w_data.indices,
                w,
                &mut rows,
                &mut cols,
            );

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
        overlap[..4].copy_from_slice(&overlap_lane);
        fock[..4].copy_from_slice(&fock_lane);
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 1` matrix elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates the same
/// determinant, cofactor and generalised-Fock algebra for 8 independent excitation pairs without
/// horizontal reductions between pairs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `x_ex`: 8 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 8 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_l1_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
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
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);

            let mut rows = [0usize; 1];
            let mut cols = [0usize; 1];
            construct_determinant_indices_l1(
                x_data.rank,
                &x_data.indices,
                &w_data.indices,
                w,
                &mut rows,
                &mut cols,
            );

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
        overlap[..8].copy_from_slice(&overlap_lane);
        fock[..8].copy_from_slice(&fock_lane);
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
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the rank-two row and column labels.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 2` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l2_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l2, {
        scratch.ensure_same_m0(2);
        scratch.rows.ensure(2);
        scratch.cols.ensure(2);

        construct_determinant_indices(
            x_ex,
            w_ex,
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
/// - `x_ex`: 4 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 4 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_l2_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
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
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);

            let mut rows = [0usize; 2];
            let mut cols = [0usize; 2];
            construct_determinant_indices_l2(
                x_data.rank,
                &x_data.indices,
                &w_data.indices,
                w,
                &mut rows,
                &mut cols,
            );

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
        overlap[..4].copy_from_slice(&overlap_lane);
        fock[..4].copy_from_slice(&fock_lane);
    }
}

/// Prepare and evaluate 8 independent real fixed-rank `L = 2` matrix elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair, so the `AVX-512` arithmetic evaluates the same
/// determinant, cofactor and generalised-Fock algebra for 8 independent excitation pairs without
/// horizontal reductions between pairs.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `x_ex`: 8 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 8 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_l2_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
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
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);

            let mut rows = [0usize; 2];
            let mut cols = [0usize; 2];
            construct_determinant_indices_l2(
                x_data.rank,
                &x_data.indices,
                &w_data.indices,
                w,
                &mut rows,
                &mut cols,
            );

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
        overlap[..8].copy_from_slice(&overlap_lane);
        fock[..8].copy_from_slice(&fock_lane);
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
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for determinant labels and generic adjugate work arrays.
/// - `tol`: Numerical tolerance used when evaluating the generic adjugate-transpose matrix.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 3` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l3_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l3, {
        scratch.ensure_same(3);

        construct_determinant_indices(
            x_ex,
            w_ex,
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
/// - `x_ex`: 4 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 4 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 4 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX2/FMA`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_l3_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
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
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);

            let mut rows = [0usize; 3];
            let mut cols = [0usize; 3];
            construct_determinant_indices_l3(
                x_data.rank,
                &x_data.indices,
                &w_data.indices,
                w,
                &mut rows,
                &mut cols,
            );

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
                overlap[lane] = overlap_lane[lane];
                fock[lane] = fock_lane[lane];
            } else {
                overlap[lane] = 0.0;
                fock[lane] = 0.0;
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
/// - `x_ex`: 8 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 8 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_l3_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
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
            let x_data = x_ex.get_unchecked(lane);
            let w_data = w_ex.get_unchecked(lane);

            let mut rows = [0usize; 3];
            let mut cols = [0usize; 3];
            construct_determinant_indices_l3(
                x_data.rank,
                &x_data.indices,
                &w_data.indices,
                w,
                &mut rows,
                &mut cols,
            );

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
                overlap[lane] = overlap_lane[lane];
                fock[lane] = fock_lane[lane];
            } else {
                overlap[lane] = 0.0;
                fock[lane] = 0.0;
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
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for determinant labels and generic adjugate work arrays.
/// - `tol`: Numerical tolerance used when evaluating the generic adjugate-transpose matrix.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 4` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l4_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l4, {
        scratch.ensure_same(4);

        construct_determinant_indices(
            x_ex,
            w_ex,
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

/// Prepare and evaluate 4 real `m = 0`, `L = 4` overlap and generalised-Fock matrix elements.
/// Each AVX2 lane represents one independent Wick pair.
/// # Safety
/// The caller must ensure `T = f64`, AVX2/FMA support and valid cached excitation labels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_f_overlap_m0_l4_prepared_f64x4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
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

        // Construct the integer labels defining each lane's `4 x 4` contraction matrix `\mathbf D`.
        let mut rows = [[0usize; 4]; 4];
        let mut cols = [[0usize; 4]; 4];

        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;
        let x_rank = x_ex.get_unchecked(0).rank;

        match x_rank {
            0 => {
                for lane in 0..4 {
                    let wi = &w_ex.get_unchecked(lane).indices;

                    rows[lane][0] = nvirt + usize::from(wi[0]);
                    cols[lane][0] = usize::from(wi[4]);
                    rows[lane][1] = nvirt + usize::from(wi[1]);
                    cols[lane][1] = usize::from(wi[5]);
                    rows[lane][2] = nvirt + usize::from(wi[2]);
                    cols[lane][2] = usize::from(wi[6]);
                    rows[lane][3] = nvirt + usize::from(wi[3]);
                    cols[lane][3] = usize::from(wi[7]);
                }
            }
            1 => {
                for lane in 0..4 {
                    let xi = &x_ex.get_unchecked(lane).indices;
                    let wi = &w_ex.get_unchecked(lane).indices;

                    rows[lane][0] = usize::from(xi[4]) - nocc;
                    cols[lane][0] = usize::from(xi[0]);
                    rows[lane][1] = nvirt + usize::from(wi[0]);
                    cols[lane][1] = usize::from(wi[4]);
                    rows[lane][2] = nvirt + usize::from(wi[1]);
                    cols[lane][2] = usize::from(wi[5]);
                    rows[lane][3] = nvirt + usize::from(wi[2]);
                    cols[lane][3] = usize::from(wi[6]);
                }
            }
            2 => {
                for lane in 0..4 {
                    let xi = &x_ex.get_unchecked(lane).indices;
                    let wi = &w_ex.get_unchecked(lane).indices;

                    rows[lane][0] = usize::from(xi[4]) - nocc;
                    cols[lane][0] = usize::from(xi[0]);
                    rows[lane][1] = usize::from(xi[5]) - nocc;
                    cols[lane][1] = usize::from(xi[1]);
                    rows[lane][2] = nvirt + usize::from(wi[0]);
                    cols[lane][2] = usize::from(wi[4]);
                    rows[lane][3] = nvirt + usize::from(wi[1]);
                    cols[lane][3] = usize::from(wi[5]);
                }
            }
            3 => {
                for lane in 0..4 {
                    let xi = &x_ex.get_unchecked(lane).indices;
                    let wi = &w_ex.get_unchecked(lane).indices;

                    rows[lane][0] = usize::from(xi[4]) - nocc;
                    cols[lane][0] = usize::from(xi[0]);
                    rows[lane][1] = usize::from(xi[5]) - nocc;
                    cols[lane][1] = usize::from(xi[1]);
                    rows[lane][2] = usize::from(xi[6]) - nocc;
                    cols[lane][2] = usize::from(xi[2]);
                    rows[lane][3] = nvirt + usize::from(wi[0]);
                    cols[lane][3] = usize::from(wi[4]);
                }
            }
            4 => {
                for lane in 0..4 {
                    let xi = &x_ex.get_unchecked(lane).indices;

                    rows[lane][0] = usize::from(xi[4]) - nocc;
                    cols[lane][0] = usize::from(xi[0]);
                    rows[lane][1] = usize::from(xi[5]) - nocc;
                    cols[lane][1] = usize::from(xi[1]);
                    rows[lane][2] = usize::from(xi[6]) - nocc;
                    cols[lane][2] = usize::from(xi[2]);
                    rows[lane][3] = usize::from(xi[7]) - nocc;
                    cols[lane][3] = usize::from(xi[3]);
                }
            }
            _ => unreachable!(),
        }

        // Gather one `D_{ij}` across the four independent matrix elements.
        macro_rules! dvec {
            ($slice:expr, $row:expr, $col:expr) => {{
                _mm256_set_pd(
                    *$slice.get_unchecked(rows[3][$row] * n + cols[3][$col]),
                    *$slice.get_unchecked(rows[2][$row] * n + cols[2][$col]),
                    *$slice.get_unchecked(rows[1][$row] * n + cols[1][$col]),
                    *$slice.get_unchecked(rows[0][$row] * n + cols[0][$col]),
                )
            }};
        }

        // Gather `\mathcal F_{ij}` only when its cofactor `C_{ij}` is ready for contraction.
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

        // Evaluate the `2 x 2` minor `ab - cd` with the same operation order as the previous kernel.
        macro_rules! minor {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {{ _mm256_fmsub_pd($a, $b, _mm256_mul_pd($c, $d)) }};
        }

        // Evaluate `C = a M_0 - b M_1 + c M_2` with the original `fmsub` then `fmadd` ordering.
        macro_rules! cof_pos {
            ($a:expr, $m0:expr, $b:expr, $m1:expr, $c:expr, $m2:expr) => {{
                let t = _mm256_fmsub_pd($a, $m0, _mm256_mul_pd($b, $m1));
                _mm256_fmadd_pd($c, $m2, t)
            }};
        }

        // Evaluate `C = -(a M_0 - b M_1 + c M_2)` without reassociating the `3 x 3` determinant.
        macro_rules! cof_neg {
            ($a:expr, $m0:expr, $b:expr, $m1:expr, $c:expr, $m2:expr) => {{
                let t = _mm256_fmsub_pd($a, $m0, _mm256_mul_pd($b, $m1));
                let value = _mm256_fmadd_pd($c, $m2, t);
                _mm256_sub_pd(_mm256_setzero_pd(), value)
            }};
        }

        // The old 16 cofactors contain `16 x 3 = 48` minor occurrences. Preserving their exact
        // expansions leaves `6 + 6 + 6 = 18` distinct minors, so 18 is the lower bound for this DAG.
        // AVX2 cannot keep all 16 `D_{ij}` plus these intermediates live, so `D_{ij}` is reloaded by group.

        // `B_{ab} = D_{2a}D_{3b} - D_{2b}D_{3a}` supplies cofactor rows 0 and 1.
        let (b01, b02, b03, b12, b13, b23) = {
            let d20 = dvec!(x0, 2, 0);
            let d21 = dvec!(x0, 2, 1);
            let d22 = dvec!(x0, 2, 2);
            let d23 = dvec!(y0, 2, 3);

            let d30 = dvec!(x0, 3, 0);
            let d31 = dvec!(x0, 3, 1);

            let b01 = minor!(d20, d31, d21, d30);

            let d32 = dvec!(x0, 3, 2);

            let b02 = minor!(d20, d32, d22, d30);
            let b12 = minor!(d21, d32, d22, d31);

            let d33 = dvec!(x0, 3, 3);

            let b03 = minor!(d20, d33, d23, d30);
            let b13 = minor!(d21, d33, d23, d31);
            let b23 = minor!(d22, d33, d23, d32);

            (b01, b02, b03, b12, b13, b23)
        };

        // Form `det(\mathbf D) = \sum_j D_{0j}C_{0j}` and row 0 of `C:\mathcal F`.
        let mut det_v = _mm256_setzero_pd();
        let mut repl0 = _mm256_setzero_pd();

        {
            let d10 = dvec!(x0, 1, 0);
            let d11 = dvec!(x0, 1, 1);
            let d12 = dvec!(y0, 1, 2);
            let d13 = dvec!(y0, 1, 3);

            let cof00 = cof_pos!(d11, b23, d12, b13, d13, b12);
            det_v = _mm256_fmadd_pd(dvec!(x0, 0, 0), cof00, det_v);
            repl0 = _mm256_fmadd_pd(fvec!(0, 0), cof00, repl0);

            let cof01 = cof_neg!(d10, b23, d12, b03, d13, b02);
            det_v = _mm256_fmadd_pd(dvec!(y0, 0, 1), cof01, det_v);
            repl0 = _mm256_fmadd_pd(fvec!(0, 1), cof01, repl0);

            let cof02 = cof_pos!(d10, b13, d11, b03, d13, b01);
            det_v = _mm256_fmadd_pd(dvec!(y0, 0, 2), cof02, det_v);
            repl0 = _mm256_fmadd_pd(fvec!(0, 2), cof02, repl0);

            let cof03 = cof_neg!(d10, b12, d11, b02, d12, b01);
            det_v = _mm256_fmadd_pd(dvec!(y0, 0, 3), cof03, det_v);
            repl0 = _mm256_fmadd_pd(fvec!(0, 3), cof03, repl0);
        }

        // Store `det(\mathbf D)` at its first natural endpoint so it does not remain live across all cofactors.
        let mut det_lane = [0.0f64; 4];
        let mut overlap_lane = [0.0f64; 4];
        let mut fock_lane = [0.0f64; 4];

        _mm256_storeu_pd(det_lane.as_mut_ptr(), det_v);

        // Reuse the same six `B_{ab}` values for row 1 of `C:\mathcal F`.
        let mut repl1 = _mm256_setzero_pd();

        {
            let d00 = dvec!(x0, 0, 0);
            let d01 = dvec!(y0, 0, 1);
            let d02 = dvec!(y0, 0, 2);
            let d03 = dvec!(y0, 0, 3);

            let cof10 = cof_neg!(d01, b23, d02, b13, d03, b12);
            repl1 = _mm256_fmadd_pd(fvec!(1, 0), cof10, repl1);

            let cof11 = cof_pos!(d00, b23, d02, b03, d03, b02);
            repl1 = _mm256_fmadd_pd(fvec!(1, 1), cof11, repl1);

            let cof12 = cof_neg!(d00, b13, d01, b03, d03, b01);
            repl1 = _mm256_fmadd_pd(fvec!(1, 2), cof12, repl1);

            let cof13 = cof_pos!(d00, b12, d01, b02, d02, b01);
            repl1 = _mm256_fmadd_pd(fvec!(1, 3), cof13, repl1);
        }

        let repl01 = _mm256_add_pd(repl0, repl1);

        // `Q_{ab} = D_{1a}D_{3b} - D_{1b}D_{3a}` supplies cofactor row 2.
        let (q01, q02, q03, q12, q13, q23) = {
            let d10 = dvec!(x0, 1, 0);
            let d11 = dvec!(x0, 1, 1);
            let d12 = dvec!(y0, 1, 2);
            let d13 = dvec!(y0, 1, 3);

            let d30 = dvec!(x0, 3, 0);
            let d31 = dvec!(x0, 3, 1);

            let q01 = minor!(d10, d31, d11, d30);

            let d32 = dvec!(x0, 3, 2);

            let q02 = minor!(d10, d32, d12, d30);
            let q12 = minor!(d11, d32, d12, d31);

            let d33 = dvec!(x0, 3, 3);

            let q03 = minor!(d10, d33, d13, d30);
            let q13 = minor!(d11, d33, d13, d31);
            let q23 = minor!(d12, d33, d13, d32);

            (q01, q02, q03, q12, q13, q23)
        };

        let mut repl2 = _mm256_setzero_pd();

        {
            let d00 = dvec!(x0, 0, 0);
            let d01 = dvec!(y0, 0, 1);
            let d02 = dvec!(y0, 0, 2);
            let d03 = dvec!(y0, 0, 3);

            let cof20 = cof_pos!(d01, q23, d02, q13, d03, q12);
            repl2 = _mm256_fmadd_pd(fvec!(2, 0), cof20, repl2);

            let cof21 = cof_neg!(d00, q23, d02, q03, d03, q02);
            repl2 = _mm256_fmadd_pd(fvec!(2, 1), cof21, repl2);

            let cof22 = cof_pos!(d00, q13, d01, q03, d03, q01);
            repl2 = _mm256_fmadd_pd(fvec!(2, 2), cof22, repl2);

            let cof23 = cof_neg!(d00, q12, d01, q02, d02, q01);
            repl2 = _mm256_fmadd_pd(fvec!(2, 3), cof23, repl2);
        }

        // `R_{ab} = D_{1a}D_{2b} - D_{1b}D_{2a}` supplies cofactor row 3.
        let (r01, r02, r03, r12, r13, r23) = {
            let d10 = dvec!(x0, 1, 0);
            let d11 = dvec!(x0, 1, 1);
            let d12 = dvec!(y0, 1, 2);
            let d13 = dvec!(y0, 1, 3);

            let d20 = dvec!(x0, 2, 0);
            let d21 = dvec!(x0, 2, 1);

            let r01 = minor!(d10, d21, d11, d20);

            let d22 = dvec!(x0, 2, 2);

            let r02 = minor!(d10, d22, d12, d20);
            let r12 = minor!(d11, d22, d12, d21);

            let d23 = dvec!(y0, 2, 3);

            let r03 = minor!(d10, d23, d13, d20);
            let r13 = minor!(d11, d23, d13, d21);
            let r23 = minor!(d12, d23, d13, d22);

            (r01, r02, r03, r12, r13, r23)
        };

        let mut repl3 = _mm256_setzero_pd();

        {
            let d00 = dvec!(x0, 0, 0);
            let d01 = dvec!(y0, 0, 1);
            let d02 = dvec!(y0, 0, 2);
            let d03 = dvec!(y0, 0, 3);

            let cof30 = cof_neg!(d01, r23, d02, r13, d03, r12);
            repl3 = _mm256_fmadd_pd(fvec!(3, 0), cof30, repl3);

            let cof31 = cof_pos!(d00, r23, d02, r03, d03, r02);
            repl3 = _mm256_fmadd_pd(fvec!(3, 1), cof31, repl3);

            let cof32 = cof_neg!(d00, r13, d01, r03, d03, r01);
            repl3 = _mm256_fmadd_pd(fvec!(3, 2), cof32, repl3);

            let cof33 = cof_pos!(d00, r12, d01, r02, d02, r01);
            repl3 = _mm256_fmadd_pd(fvec!(3, 3), cof33, repl3);
        }

        // Preserve the previous contraction tree `((repl0 + repl1) + (repl2 + repl3))`.
        let repl23 = _mm256_add_pd(repl2, repl3);
        let repl_v = _mm256_add_pd(repl01, repl23);

        let det_v = _mm256_loadu_pd(det_lane.as_ptr());
        let pref_v = _mm256_set1_pd(pref);
        let f0_v = _mm256_set1_pd(f0);

        // `S = P det(\mathbf D)` and `F = P[F_0 det(\mathbf D) - C:\mathcal F]`.
        let overlap_v = _mm256_mul_pd(det_v, pref_v);
        let contrib = _mm256_fmsub_pd(det_v, f0_v, repl_v);
        let fock_v = _mm256_mul_pd(contrib, pref_v);

        _mm256_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
        _mm256_storeu_pd(fock_lane.as_mut_ptr(), fock_v);

        for lane in 0..4 {
            if det_lane[lane].is_finite() {
                overlap[lane] = overlap_lane[lane];
                fock[lane] = fock_lane[lane];
            } else {
                overlap[lane] = 0.0;
                fock[lane] = 0.0;
            }
        }
    }
}

/// Prepare and evaluate 8 real `m = 0`, `L = 4` overlap and generalised-Fock matrix elements.
/// Each AVX-512 lane represents one independent Wick pair.
/// # Safety
/// The caller must ensure `T = f64`, AVX-512 support and valid cached excitation labels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_l4_prepared_f64x8<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
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

        // Construct the integer labels defining each lane's `4 x 4` contraction matrix `\mathbf D`.
        let mut rows = [[0usize; 4]; 8];
        let mut cols = [[0usize; 4]; 8];

        let nocc = w.nocc;
        let nvirt = w.nmo - nocc;
        let x_rank = x_ex.get_unchecked(0).rank;

        match x_rank {
            0 => {
                for lane in 0..8 {
                    let wi = &w_ex.get_unchecked(lane).indices;

                    rows[lane][0] = nvirt + usize::from(wi[0]);
                    cols[lane][0] = usize::from(wi[4]);
                    rows[lane][1] = nvirt + usize::from(wi[1]);
                    cols[lane][1] = usize::from(wi[5]);
                    rows[lane][2] = nvirt + usize::from(wi[2]);
                    cols[lane][2] = usize::from(wi[6]);
                    rows[lane][3] = nvirt + usize::from(wi[3]);
                    cols[lane][3] = usize::from(wi[7]);
                }
            }
            1 => {
                for lane in 0..8 {
                    let xi = &x_ex.get_unchecked(lane).indices;
                    let wi = &w_ex.get_unchecked(lane).indices;

                    rows[lane][0] = usize::from(xi[4]) - nocc;
                    cols[lane][0] = usize::from(xi[0]);
                    rows[lane][1] = nvirt + usize::from(wi[0]);
                    cols[lane][1] = usize::from(wi[4]);
                    rows[lane][2] = nvirt + usize::from(wi[1]);
                    cols[lane][2] = usize::from(wi[5]);
                    rows[lane][3] = nvirt + usize::from(wi[2]);
                    cols[lane][3] = usize::from(wi[6]);
                }
            }
            2 => {
                for lane in 0..8 {
                    let xi = &x_ex.get_unchecked(lane).indices;
                    let wi = &w_ex.get_unchecked(lane).indices;

                    rows[lane][0] = usize::from(xi[4]) - nocc;
                    cols[lane][0] = usize::from(xi[0]);
                    rows[lane][1] = usize::from(xi[5]) - nocc;
                    cols[lane][1] = usize::from(xi[1]);
                    rows[lane][2] = nvirt + usize::from(wi[0]);
                    cols[lane][2] = usize::from(wi[4]);
                    rows[lane][3] = nvirt + usize::from(wi[1]);
                    cols[lane][3] = usize::from(wi[5]);
                }
            }
            3 => {
                for lane in 0..8 {
                    let xi = &x_ex.get_unchecked(lane).indices;
                    let wi = &w_ex.get_unchecked(lane).indices;

                    rows[lane][0] = usize::from(xi[4]) - nocc;
                    cols[lane][0] = usize::from(xi[0]);
                    rows[lane][1] = usize::from(xi[5]) - nocc;
                    cols[lane][1] = usize::from(xi[1]);
                    rows[lane][2] = usize::from(xi[6]) - nocc;
                    cols[lane][2] = usize::from(xi[2]);
                    rows[lane][3] = nvirt + usize::from(wi[0]);
                    cols[lane][3] = usize::from(wi[4]);
                }
            }
            4 => {
                for lane in 0..8 {
                    let xi = &x_ex.get_unchecked(lane).indices;

                    rows[lane][0] = usize::from(xi[4]) - nocc;
                    cols[lane][0] = usize::from(xi[0]);
                    rows[lane][1] = usize::from(xi[5]) - nocc;
                    cols[lane][1] = usize::from(xi[1]);
                    rows[lane][2] = usize::from(xi[6]) - nocc;
                    cols[lane][2] = usize::from(xi[2]);
                    rows[lane][3] = usize::from(xi[7]) - nocc;
                    cols[lane][3] = usize::from(xi[3]);
                }
            }
            _ => unreachable!(),
        }

        macro_rules! dvec {
            ($slice:expr, $row:expr, $col:expr) => {{
                _mm512_set_pd(
                    *$slice.get_unchecked(rows[7][$row] * n + cols[7][$col]),
                    *$slice.get_unchecked(rows[6][$row] * n + cols[6][$col]),
                    *$slice.get_unchecked(rows[5][$row] * n + cols[5][$col]),
                    *$slice.get_unchecked(rows[4][$row] * n + cols[4][$col]),
                    *$slice.get_unchecked(rows[3][$row] * n + cols[3][$col]),
                    *$slice.get_unchecked(rows[2][$row] * n + cols[2][$col]),
                    *$slice.get_unchecked(rows[1][$row] * n + cols[1][$col]),
                    *$slice.get_unchecked(rows[0][$row] * n + cols[0][$col]),
                )
            }};
        }

        // Each `\mathcal F_{ij}` occurs once in `C:\mathcal F`, so consume it directly with `C_{ij}`.
        macro_rules! fvec {
            ($row:expr, $col:expr) => {{
                _mm512_set_pd(
                    *fsl.get_unchecked(cols[7][$col] * n + rows[7][$row]),
                    *fsl.get_unchecked(cols[6][$col] * n + rows[6][$row]),
                    *fsl.get_unchecked(cols[5][$col] * n + rows[5][$row]),
                    *fsl.get_unchecked(cols[4][$col] * n + rows[4][$row]),
                    *fsl.get_unchecked(cols[3][$col] * n + rows[3][$row]),
                    *fsl.get_unchecked(cols[2][$col] * n + rows[2][$row]),
                    *fsl.get_unchecked(cols[1][$col] * n + rows[1][$row]),
                    *fsl.get_unchecked(cols[0][$col] * n + rows[0][$row]),
                )
            }};
        }

        // Keep the existing `ab - cd` and `a M_0 - b M_1 + c M_2` floating-point expressions.
        macro_rules! minor {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {{ _mm512_fmsub_pd($a, $b, _mm512_mul_pd($c, $d)) }};
        }

        macro_rules! cof_pos {
            ($a:expr, $m0:expr, $b:expr, $m1:expr, $c:expr, $m2:expr) => {{
                let t = _mm512_fmsub_pd($a, $m0, _mm512_mul_pd($b, $m1));
                _mm512_fmadd_pd($c, $m2, t)
            }};
        }

        macro_rules! cof_neg {
            ($a:expr, $m0:expr, $b:expr, $m1:expr, $c:expr, $m2:expr) => {{
                let t = _mm512_fmsub_pd($a, $m0, _mm512_mul_pd($b, $m1));
                let value = _mm512_fmadd_pd($c, $m2, t);
                _mm512_sub_pd(_mm512_setzero_pd(), value)
            }};
        }

        // Preserving the explicit `L = 4` cofactor DAG reduces 48 minor occurrences to exactly
        // `3 binom(4,2) = 18` distinct minors. Each is required, so 18 is the lower bound for this DAG.
        // AVX-512 has 32 ZMM registers, allowing the 16 distinct `D_{ij}` inputs to remain resident.

        let d00 = dvec!(x0, 0, 0);
        let d01 = dvec!(y0, 0, 1);
        let d02 = dvec!(y0, 0, 2);
        let d03 = dvec!(y0, 0, 3);

        let d10 = dvec!(x0, 1, 0);
        let d11 = dvec!(x0, 1, 1);
        let d12 = dvec!(y0, 1, 2);
        let d13 = dvec!(y0, 1, 3);

        let d20 = dvec!(x0, 2, 0);
        let d21 = dvec!(x0, 2, 1);
        let d22 = dvec!(x0, 2, 2);
        let d23 = dvec!(y0, 2, 3);

        let d30 = dvec!(x0, 3, 0);
        let d31 = dvec!(x0, 3, 1);
        let d32 = dvec!(x0, 3, 2);
        let d33 = dvec!(x0, 3, 3);

        // `B_{ab}` contains the six row-pair `(2,3)` minors used by cofactor rows 0 and 1.
        let b01 = minor!(d20, d31, d21, d30);
        let b02 = minor!(d20, d32, d22, d30);
        let b03 = minor!(d20, d33, d23, d30);
        let b12 = minor!(d21, d32, d22, d31);
        let b13 = minor!(d21, d33, d23, d31);
        let b23 = minor!(d22, d33, d23, d32);

        let mut det_v = _mm512_setzero_pd();
        let mut repl0 = _mm512_setzero_pd();
        let mut repl1 = _mm512_setzero_pd();

        // Form `det(\mathbf D)` through row 0 while contracting cofactor row 0 with `\mathcal F`.
        let cof00 = cof_pos!(d11, b23, d12, b13, d13, b12);
        det_v = _mm512_fmadd_pd(d00, cof00, det_v);
        repl0 = _mm512_fmadd_pd(fvec!(0, 0), cof00, repl0);

        let cof01 = cof_neg!(d10, b23, d12, b03, d13, b02);
        det_v = _mm512_fmadd_pd(d01, cof01, det_v);
        repl0 = _mm512_fmadd_pd(fvec!(0, 1), cof01, repl0);

        let cof02 = cof_pos!(d10, b13, d11, b03, d13, b01);
        det_v = _mm512_fmadd_pd(d02, cof02, det_v);
        repl0 = _mm512_fmadd_pd(fvec!(0, 2), cof02, repl0);

        let cof03 = cof_neg!(d10, b12, d11, b02, d12, b01);
        det_v = _mm512_fmadd_pd(d03, cof03, det_v);
        repl0 = _mm512_fmadd_pd(fvec!(0, 3), cof03, repl0);

        // The same `B_{ab}` values give cofactor row 1 without any further minor evaluation.
        let cof10 = cof_neg!(d01, b23, d02, b13, d03, b12);
        repl1 = _mm512_fmadd_pd(fvec!(1, 0), cof10, repl1);

        let cof11 = cof_pos!(d00, b23, d02, b03, d03, b02);
        repl1 = _mm512_fmadd_pd(fvec!(1, 1), cof11, repl1);

        let cof12 = cof_neg!(d00, b13, d01, b03, d03, b01);
        repl1 = _mm512_fmadd_pd(fvec!(1, 2), cof12, repl1);

        let cof13 = cof_pos!(d00, b12, d01, b02, d02, b01);
        repl1 = _mm512_fmadd_pd(fvec!(1, 3), cof13, repl1);

        let repl01 = _mm512_add_pd(repl0, repl1);

        // `Q_{ab}` contains the six row-pair `(1,3)` minors required by cofactor row 2.
        let q01 = minor!(d10, d31, d11, d30);
        let q02 = minor!(d10, d32, d12, d30);
        let q03 = minor!(d10, d33, d13, d30);
        let q12 = minor!(d11, d32, d12, d31);
        let q13 = minor!(d11, d33, d13, d31);
        let q23 = minor!(d12, d33, d13, d32);

        let mut repl2 = _mm512_setzero_pd();

        let cof20 = cof_pos!(d01, q23, d02, q13, d03, q12);
        repl2 = _mm512_fmadd_pd(fvec!(2, 0), cof20, repl2);

        let cof21 = cof_neg!(d00, q23, d02, q03, d03, q02);
        repl2 = _mm512_fmadd_pd(fvec!(2, 1), cof21, repl2);

        let cof22 = cof_pos!(d00, q13, d01, q03, d03, q01);
        repl2 = _mm512_fmadd_pd(fvec!(2, 2), cof22, repl2);

        let cof23 = cof_neg!(d00, q12, d01, q02, d02, q01);
        repl2 = _mm512_fmadd_pd(fvec!(2, 3), cof23, repl2);

        // `R_{ab}` contains the final six row-pair `(1,2)` minors required by cofactor row 3.
        let r01 = minor!(d10, d21, d11, d20);
        let r02 = minor!(d10, d22, d12, d20);
        let r03 = minor!(d10, d23, d13, d20);
        let r12 = minor!(d11, d22, d12, d21);
        let r13 = minor!(d11, d23, d13, d21);
        let r23 = minor!(d12, d23, d13, d22);

        let mut repl3 = _mm512_setzero_pd();

        let cof30 = cof_neg!(d01, r23, d02, r13, d03, r12);
        repl3 = _mm512_fmadd_pd(fvec!(3, 0), cof30, repl3);

        let cof31 = cof_pos!(d00, r23, d02, r03, d03, r02);
        repl3 = _mm512_fmadd_pd(fvec!(3, 1), cof31, repl3);

        let cof32 = cof_neg!(d00, r13, d01, r03, d03, r01);
        repl3 = _mm512_fmadd_pd(fvec!(3, 2), cof32, repl3);

        let cof33 = cof_pos!(d00, r12, d01, r02, d02, r01);
        repl3 = _mm512_fmadd_pd(fvec!(3, 3), cof33, repl3);

        // Preserve `C:\mathcal F = (repl0 + repl1) + (repl2 + repl3)` from the old kernel.
        let repl23 = _mm512_add_pd(repl2, repl3);
        let repl_v = _mm512_add_pd(repl01, repl23);

        let pref_v = _mm512_set1_pd(pref);
        let f0_v = _mm512_set1_pd(f0);

        // `S = P det(\mathbf D)` and `F = P[F_0 det(\mathbf D) - C:\mathcal F]`.
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
                overlap[lane] = overlap_lane[lane];
                fock[lane] = fock_lane[lane];
            } else {
                overlap[lane] = 0.0;
                fock[lane] = 0.0;
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
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the contraction determinant and its cofactors.
/// - `l`: Total excitation rank `L = L_x + L_w`.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate-transpose matrix.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for arbitrary `L` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_gen_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    l: usize,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_gen,
        {
            scratch.ensure_same(l);

            construct_determinant_indices(
                x_ex,
                w_ex,
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
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for endpoint and mixed contraction determinants, cofactors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` summed over all distributions.
#[inline(always)]
fn xw_f_overlap_gen_prepared<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_gen, {
        let l = x_ex.holes.count_ones() as usize + w_ex.holes.count_ones() as usize;
        scratch.ensure_same(l);

        construct_determinant_indices(
            x_ex,
            w_ex,
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
