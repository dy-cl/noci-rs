// nonorthogonalwicks/eval/prepareonebodyoverlap.rs

// Standard library imports.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx",
    target_feature = "fma"
))]
use std::any::TypeId;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx",
    target_feature = "fma"
))]
use std::arch::x86_64::{
    _mm_add_sd, _mm_cvtsd_f64, _mm256_add_pd, _mm256_castpd256_pd128,
    _mm256_extractf128_pd, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_hadd_pd,
    _mm256_mul_pd, _mm256_set_pd, _mm256_storeu_pd,
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

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx",
            target_feature = "fma"
        ))]
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
                let values =
                    _mm256_fmsub_pd(lhs0, rhs0, _mm256_mul_pd(lhs1, rhs1));

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

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx",
            target_feature = "fma"
        ))]
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

                let frow0 = _mm256_set_pd(
                    0.0,
                    fsl[c2 * n + r0],
                    fsl[c1 * n + r0],
                    fsl[c0 * n + r0],
                );
                let frow1 = _mm256_set_pd(
                    0.0,
                    fsl[c2 * n + r1],
                    fsl[c1 * n + r1],
                    fsl[c0 * n + r1],
                );
                let frow2 = _mm256_set_pd(
                    0.0,
                    fsl[c2 * n + r2],
                    fsl[c1 * n + r2],
                    fsl[c0 * n + r2],
                );

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

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx",
            target_feature = "fma"
        ))]
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
                let cof0 = _mm256_mul_pd(
                    minors,
                    _mm256_set_pd(-1.0, 1.0, -1.0, 1.0),
                );

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
                let cof1 = _mm256_mul_pd(
                    minors,
                    _mm256_set_pd(1.0, -1.0, 1.0, -1.0),
                );

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
                let cof2 = _mm256_mul_pd(
                    minors,
                    _mm256_set_pd(-1.0, 1.0, -1.0, 1.0),
                );

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
                let cof3 = _mm256_mul_pd(
                    minors,
                    _mm256_set_pd(1.0, -1.0, 1.0, -1.0),
                );

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
