// nonorthogonalwicks/eval/onebodyoverlap.rs

// Standard library imports.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use std::any::TypeId;
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use std::arch::x86_64::{
    _mm_add_sd, _mm_cvtsd_f64, _mm256_add_pd, _mm256_castpd256_pd128,
    _mm256_extractf128_pd, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_hadd_pd,
    _mm256_loadu_pd, _mm256_maskload_pd, _mm256_mul_pd, _mm256_set_epi64x,
    _mm256_set_pd, _mm256_storeu_pd,
};

// Crate-root imports.
use crate::ExcitationSpin;
use crate::maths::{adjugate_transpose, det};
use crate::noci::NOCIScalar;
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, mix_dets_same,
};

/// Evaluate the same-spin overlap and generalised-Fock matrix element between excited determinants
/// generated from the reference pair `\langle{}^x\Psi| and |{}^w\Psi\rangle:`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\m_1+\cdots+m_L=m}}`
/// `\det\mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L),`
/// `\langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat F|{}^w\Psi_{j\cdots}^{b\cdots}\rangle`
/// `= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+1}\\m_1+\cdots+m_{L+1}=m}}`
/// `[{}^x F_0^{(m_1)}\det\mathbf D_{\mathrm{ov}}(m_2,\ldots,m_{L+1})`
/// `- \sum_{z=1}^{L}\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}`
/// `(m_1,\ldots,m_{L+1})].`
/// `For m = 0 the two quantities share the same contraction determinant and its determinant value.`
/// `For m > 0 the one-body distributions are traversed once; terms with m_1 = 0 also form the`
/// `overlap constrained sum because the remaining L assignments then satisfy`
/// `m_2+\cdots+m_{L+1}=m.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: `Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.`
/// - `g_ex`: `Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.`
/// - `scratch`: Scratch storage for contraction determinants, cofactors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)`.
#[inline(always)]
pub(crate) fn xw_f_overlap<T: NOCIScalar>(
    w: &SameSpinView<T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap, {
        // For m = 0 only the all-m_i = 0 contraction determinant contributes. Otherwise,
        // sum the distributions satisfying \sum_{i=1}^{L+1}m_i = m.
        if w.m == 0 {
            xw_f_overlap_m0(w, l_ex, g_ex, scratch, tol)
        } else {
            xw_f_overlap_gen(w, l_ex, g_ex, scratch, tol)
        }
    })
}

/// `Evaluate the overlap and generalised-Fock matrix element together when m = 0, so every`
/// `contraction uses m_i = 0. Specialised kernels are used for L = 1,2,3,4, while all other`
/// `excitation ranks use the general cofactor form. For L = 0, the overlap is`
/// `{}^{xw}\tilde S and only the scalar one-body intermediate {}^x F_0^{(0)} contributes to F.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: `Prepared m_i = 0 contraction determinant and scratch work arrays.`
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0, {
        // Determine the total excitation rank L = L_x + L_w.
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;

        // Dispatch to direct fixed-rank forms of the shared overlap determinant and
        // {}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}
        // - \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}.
        match l {
            0 => {
                // For L = 0, \det\mathbf D_{\mathrm{ov}} = 1 and there are no replacement columns.
                let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
                (pref, pref * w.f0f[0])
            }
            1 => xw_f_overlap_m0_l1(w, scratch),
            2 => xw_f_overlap_m0_l2(w, scratch),
            3 => xw_f_overlap_m0_l3(w, scratch, tol),
            4 => xw_f_overlap_m0_l4(w, scratch, tol),
            _ => xw_f_overlap_m0_gen(w, l_ex, g_ex, scratch, tol),
        }
    })
}

/// `Evaluate the fixed-rank L = 1 overlap and generalised-Fock matrix element for m = 0.`
/// `The contraction determinant and sole one-column replacement are evaluated directly.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-one contraction determinant and its row and column labels.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 1` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l1, {
        // For L = 1, \mathbf D_{\mathrm{ov}} = [D_{00}] and
        // \det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0}
        // = \mathcal F_{r_0c_0}^{(0,0)}.
        let n = w.n();
        let det0 = scratch.det0.as_slice();
        let det = det0[0];
        let r0 = scratch.rows[0];
        let c0 = scratch.cols[0];

        // Select {}^{\chi_{r_0}\chi_{c_0}}\mathcal F_{r_0c_0}^{(0,0)}
        // constructed from the current generalised-Fock operator.
        let fsl = w.ff_t_slice(0, 0);
        let repl = fsl[c0 * n + r0];

        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

        // \langle{}^x\Psi_{i\cdots}^{a\cdots}|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
        // = {}^{xw}\tilde S D_{00}.
        let overlap = pref * det;

        // \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat F|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
        // = {}^{xw}\tilde S[{}^x F_0^{(0)}D_{00}
        // - \mathcal F_{r_0c_0}^{(0,0)}].
        let fock = pref * (det * w.f0f[0] - repl);

        (overlap, fock)
    })
}

/// Evaluate the fixed-rank `L = 2` overlap and generalised-Fock matrix element for `m = 0`:
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}},`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0}`
/// `- \det\mathbf D_{\mathrm{ov}}^{1\rightarrow\mathcal F_1}].`
/// The real-valued path evaluates the overlap determinant and both replacement determinants
/// simultaneously with packed AVX FMA operations.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-two contraction determinant and its row and column labels.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 2` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l2, {
        let n = w.n();
        let d = scratch.det0.as_slice();
        let r0 = scratch.rows[0];
        let r1 = scratch.rows[1];
        let c0 = scratch.cols[0];
        let c1 = scratch.cols[1];
        let fsl = w.ff_t_slice(0, 0);
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            unsafe {
                let d = std::slice::from_raw_parts(d.as_ptr().cast::<f64>(), 4);
                let fsl = std::slice::from_raw_parts(fsl.as_ptr().cast::<f64>(), fsl.len());

                let a00 = d[0];
                let a01 = d[1];
                let a10 = d[2];
                let a11 = d[3];
                let u0 = fsl[c0 * n + r0];
                let u1 = fsl[c0 * n + r1];
                let v0 = fsl[c1 * n + r0];
                let v1 = fsl[c1 * n + r1];

                // Pack [det(D), det(D^{0->F_0}), det(D^{1->F_1}), 0].
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

        // Evaluate \det\mathbf D_{\mathrm{ov}} = D_{00}D_{11} - D_{01}D_{10}.
        let a00 = d[0];
        let a01 = d[1];
        let a10 = d[2];
        let a11 = d[3];
        let det = a00 * a11 - a01 * a10;

        // Form the two generalised-Fock replacement columns.
        let u0 = fsl[c0 * n + r0];
        let u1 = fsl[c0 * n + r1];
        let v0 = fsl[c1 * n + r0];
        let v1 = fsl[c1 * n + r1];

        // Evaluate \det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0} and
        // \det\mathbf D_{\mathrm{ov}}^{1\rightarrow\mathcal F_1}.
        let det_c0 = u0 * a11 - a01 * u1;
        let det_c1 = a00 * v1 - v0 * a10;

        let overlap = pref * det;
        let fock = pref * (det * w.f0f[0] - det_c0 - det_c1);

        (overlap, fock)
    })
}

/// Evaluate the fixed-rank `L = 3` overlap and generalised-Fock matrix element for `m = 0`:
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}},`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}`
/// `\mathcal F_{\eta z}^{(0,0)}].`
/// The determinant and cofactor matrix are shared between the overlap and Fock terms. The
/// real-valued cofactor-Fock contraction is evaluated three rows at a time with packed AVX FMA operations.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-three contraction determinant and scratch storage for its cofactors.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate-transpose matrix.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 3` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l3, {
        let n = w.n();
        let det0 = &scratch.det0.as_slice()[..9];
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);

        if let Some(det_det) = adjugate_transpose(
            scratch.adjt_det.as_mut_slice(),
            scratch.invs.as_mut_slice(),
            scratch.lu.as_mut_slice(),
            det0,
            3,
            tol,
        ) {
            let cof = scratch.adjt_det.as_slice();
            let rows = scratch.rows.as_slice();
            let cols = scratch.cols.as_slice();
            let r0 = rows[0];
            let r1 = rows[1];
            let r2 = rows[2];
            let c0 = cols[0];
            let c1 = cols[1];
            let c2 = cols[2];
            let fsl = w.ff_t_slice(0, 0);

            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                unsafe {
                    let cof = std::slice::from_raw_parts(cof.as_ptr().cast::<f64>(), 9);
                    let fsl = std::slice::from_raw_parts(fsl.as_ptr().cast::<f64>(), fsl.len());
                    let mask = _mm256_set_epi64x(0, -1, -1, -1);

                    // Contract each contiguous cofactor row with the corresponding Fock row.
                    let c_row0 = _mm256_maskload_pd(cof.as_ptr(), mask);
                    let c_row1 = _mm256_maskload_pd(cof.as_ptr().add(3), mask);
                    let c_row2 = _mm256_maskload_pd(cof.as_ptr().add(6), mask);
                    let f_row0 = _mm256_set_pd(
                        0.0,
                        fsl[c2 * n + r0],
                        fsl[c1 * n + r0],
                        fsl[c0 * n + r0],
                    );
                    let f_row1 = _mm256_set_pd(
                        0.0,
                        fsl[c2 * n + r1],
                        fsl[c1 * n + r1],
                        fsl[c0 * n + r1],
                    );
                    let f_row2 = _mm256_set_pd(
                        0.0,
                        fsl[c2 * n + r2],
                        fsl[c1 * n + r2],
                        fsl[c0 * n + r2],
                    );
                    let repl01 =
                        _mm256_fmadd_pd(c_row1, f_row1, _mm256_mul_pd(c_row0, f_row0));
                    let repl_v = _mm256_fmadd_pd(c_row2, f_row2, repl01);
                    let sums = _mm256_hadd_pd(repl_v, repl_v);
                    let low = _mm256_castpd256_pd128(sums);
                    let high = _mm256_extractf128_pd(sums, 1);
                    let repl = _mm_cvtsd_f64(_mm_add_sd(low, high));

                    let overlap = pref * det_det;
                    let fock = pref * (det_det * w.f0f[0] - T::from_real(repl));

                    return (overlap, fock);
                }
            }

            // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}
            // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}
            // \mathcal F_{\eta z}^{(0,0)}.
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
            // Preserve the standalone overlap convention if the one-body adjugate path rejects
            // the determinant under the numerical threshold.
            let overlap = pref * det(det0, 3).unwrap_or(zero);
            (overlap, zero)
        }
    })
}

/// Evaluate the fixed-rank `L = 4` overlap and generalised-Fock matrix element for `m = 0`:
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}},`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}`
/// `\mathcal F_{\eta z}^{(0,0)}].`
/// The determinant and cofactor matrix are shared between the overlap and Fock terms. The
/// real-valued cofactor-Fock contraction is evaluated four rows at a time with packed AVX FMA operations.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared rank-four contraction determinant and scratch storage for its cofactors.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate-transpose matrix.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for `L = 4` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_l4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_l4, {
        let n = w.n();
        let det0 = &scratch.det0.as_slice()[..16];
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        let zero = <T as From<f64>>::from(0.0);

        if let Some(det_det) = adjugate_transpose(
            scratch.adjt_det.as_mut_slice(),
            scratch.invs.as_mut_slice(),
            scratch.lu.as_mut_slice(),
            det0,
            4,
            tol,
        ) {
            let cof = scratch.adjt_det.as_slice();
            let rows = scratch.rows.as_slice();
            let cols = scratch.cols.as_slice();
            let r0 = rows[0];
            let r1 = rows[1];
            let r2 = rows[2];
            let r3 = rows[3];
            let c0 = cols[0];
            let c1 = cols[1];
            let c2 = cols[2];
            let c3 = cols[3];
            let fsl = w.ff_t_slice(0, 0);

            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                unsafe {
                    let cof = std::slice::from_raw_parts(cof.as_ptr().cast::<f64>(), 16);
                    let fsl = std::slice::from_raw_parts(fsl.as_ptr().cast::<f64>(), fsl.len());

                    // Contract four complete cofactor rows with the four generalised-Fock rows.
                    let c_row0 = _mm256_loadu_pd(cof.as_ptr());
                    let c_row1 = _mm256_loadu_pd(cof.as_ptr().add(4));
                    let c_row2 = _mm256_loadu_pd(cof.as_ptr().add(8));
                    let c_row3 = _mm256_loadu_pd(cof.as_ptr().add(12));
                    let f_row0 = _mm256_set_pd(
                        fsl[c3 * n + r0],
                        fsl[c2 * n + r0],
                        fsl[c1 * n + r0],
                        fsl[c0 * n + r0],
                    );
                    let f_row1 = _mm256_set_pd(
                        fsl[c3 * n + r1],
                        fsl[c2 * n + r1],
                        fsl[c1 * n + r1],
                        fsl[c0 * n + r1],
                    );
                    let f_row2 = _mm256_set_pd(
                        fsl[c3 * n + r2],
                        fsl[c2 * n + r2],
                        fsl[c1 * n + r2],
                        fsl[c0 * n + r2],
                    );
                    let f_row3 = _mm256_set_pd(
                        fsl[c3 * n + r3],
                        fsl[c2 * n + r3],
                        fsl[c1 * n + r3],
                        fsl[c0 * n + r3],
                    );
                    let repl01 =
                        _mm256_fmadd_pd(c_row1, f_row1, _mm256_mul_pd(c_row0, f_row0));
                    let repl23 =
                        _mm256_fmadd_pd(c_row3, f_row3, _mm256_mul_pd(c_row2, f_row2));
                    let repl_v = _mm256_add_pd(repl01, repl23);
                    let sums = _mm256_hadd_pd(repl_v, repl_v);
                    let low = _mm256_castpd256_pd128(sums);
                    let high = _mm256_extractf128_pd(sums, 1);
                    let repl = _mm_cvtsd_f64(_mm_add_sd(low, high));

                    let overlap = pref * det_det;
                    let fock = pref * (det_det * w.f0f[0] - T::from_real(repl));

                    return (overlap, fock);
                }
            }

            // \sum_z\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}
            // = \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}
            // \mathcal F_{\eta z}^{(0,0)}.
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
            // Preserve the standalone overlap convention if the one-body adjugate path rejects
            // the determinant under the numerical threshold.
            let overlap = pref * det(det0, 4).unwrap_or(zero);
            (overlap, zero)
        }
    })
}

/// Evaluate the overlap and generalised-Fock matrix element for arbitrary L when m = 0:
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}},`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{z=1}^{L}\det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}].`
/// `The determinant returned while constructing the one-body cofactor matrix is reused for the`
/// `overlap. If that path rejects the determinant under tol, only the overlap determinant is`
/// `evaluated separately, preserving the numerical convention of the standalone evaluators.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Prepared contraction determinant and scratch storage for its cofactors.
/// - `tol`: Numerical tolerance used when evaluating the determinant and adjugate-transpose matrix.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for arbitrary `L` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_gen,
        {
            // Determine L = L_x + L_w and select \mathbf D_{\mathrm{ov}}(0,\ldots,0).
            let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
            let zero = <T as From<f64>>::from(0.0);
            let n = w.n();
            let det0 = &scratch.det0.as_slice()[..l * l];
            let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

            // Evaluate \det\mathbf D_{\mathrm{ov}} and its cofactor matrix.
            if let Some(det_det) = adjugate_transpose(
                scratch.adjt_det.as_mut_slice(),
                scratch.invs.as_mut_slice(),
                scratch.lu.as_mut_slice(),
                det0,
                l,
                tol,
            ) {
                // Start with {}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}.
                let mut contrib = det_det * w.f0f[0];
                let fsl = w.ff_t_slice(0, 0);

                // Subtract \det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z} for each column z.
                for b in 0..l {
                    let cb = scratch.cols[b];
                    let base = cb * n;

                    // `corr` is the determinant correction, so `det_det + corr` is the determinant
                    // obtained by replacing column b with \mathcal F_b^{(0,0)}.
                    let corr = column_replacement_correction(
                        l,
                        det0,
                        scratch.adjt_det.as_slice(),
                        b,
                        |r| fsl[base + scratch.rows[r]],
                    );
                    contrib -= det_det + corr;
                }

                // Apply the orbital-pairing phase to the shared overlap determinant and one-body sum.
                (pref * det_det, pref * contrib)
            } else {
                // Preserve the overlap path when the one-body determinant is rejected under tol.
                let overlap = pref * det(det0, l).unwrap_or(zero);
                (overlap, zero)
            }
        }
    )
}

/// Evaluate the overlap and generalised-Fock matrix element when m > 0 by summing the allowed
/// one-body distributions:
/// `m_1 + \cdots + m_{L+1} = m, \qquad m_i \in \{0,1\}.`
/// `The first assignment selects {}^x F_0^{(m_1)} and the operator side of each`
/// `\mathcal F^{(m_1,m_j)} column; the remaining assignments select the columns of`
/// `\mathbf D_{\mathrm{ov}}. Terms with m_1 = 0 also satisfy the overlap constraint`
/// `m_2+\cdots+m_{L+1}=m and are accumulated into the overlap without a second distribution loop.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage for mixed contraction determinants, cofactors and work buffers.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` summed over all distributions.
#[inline(always)]
fn xw_f_overlap_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(crate::timers::nonorthogonalwicks::add_xw_f_overlap_gen, {
        // Determine L = L_x + L_w.
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
        let zero = <T as From<f64>>::from(0.0);
        let n = w.n();
        let mut overlap_acc = zero;
        let mut fock_acc = zero;

        // Enumerate all distributions over the operator contraction and L determinant columns,
        // constructing each mixed \mathbf D_{\mathrm{ov}} only once.
        mix_dets_same(w, l, 1, scratch, |bits, scratch| {
            // Bit zero is m_1, the assignment of the operator contraction.
            let mi = bit(bits, 0);

            // Evaluate \det\mathbf D_{\mathrm{ov}} and its cofactor matrix using the same routine
            // used by the existing general one-body evaluator.
            if let Some(det_det) = adjugate_transpose_generic(
                scratch.adjt_det.as_mut_slice(),
                scratch.det_mix.as_slice(),
                l,
                tol,
            ) {
                // For m_1 = 0, the remaining L assignments satisfy the standalone overlap
                // constraint and this determinant contributes to the overlap sum.
                if mi == 0 {
                    overlap_acc += det_det;
                }

                // Start with {}^x F_0^{(m_1)}\det\mathbf D_{\mathrm{ov}}.
                let mut contrib = det_det * w.f0f[mi];

                // Select \mathcal F^{(m_1,0)} and \mathcal F^{(m_1,1)}. The assignment
                // of each replaced determinant column chooses between these two slices.
                let f0 = w.ff_t_slice(mi, 0);
                let f1 = w.ff_t_slice(mi, 1);

                // Subtract every \det\mathbf D_{\mathrm{ov}}^{z\rightarrow\mathcal F_z}.
                for b in 0..l {
                    // Bit b + 1 is the zero-overlap assignment of determinant column b.
                    let mj = bit(bits, b + 1);
                    let cb = scratch.cols[b];
                    let fsl = if mj == 0 { f0 } else { f1 };
                    let base = cb * n;

                    // `det_det + corr` is the mixed contraction determinant with column b
                    // replaced by \mathcal F_b^{(m_1,m_{b+2})}.
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
                // The standalone overlap evaluator does not apply the one-body determinant
                // threshold, so evaluate only the rejected overlap term separately.
                overlap_acc += det(scratch.det_mix.as_slice(), l).unwrap_or(zero);
            }
        });

        // Apply the orbital-pairing phase to the product of non-zero singular values and both
        // constrained sums.
        let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
        (pref * overlap_acc, pref * fock_acc)
    })
}