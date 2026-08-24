// nonorthogonalwicks/eval/prepareonebodyoverlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_pd, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_mul_pd,
    _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd, _mm512_add_pd,
    _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_set1_pd,
    _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd,
};

// Crate-root imports.
use crate::maths::{adjugate_transpose, adjugate_transpose_const, build_d, build_d_const, det};
use crate::noci::NOCIScalar;
use crate::time_call;
use crate::{ExcitationSpin, ExcitationSpinCache};

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, mix_dets_same,
};
use super::prepare::{construct_determinant_indices, construct_determinant_indices_const};

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
        // The zero-overlap count decides whether the prepared GNME expression is the single
        // all-zero determinant or a constrained sum over mixed `m_i` distributions.
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
        // For m = 0, the rank L determines the one prepared D_ov determinant and its
        // one-body replacement determinants.
        let l = x_ex.holes.count_ones() as usize + w_ex.holes.count_ones() as usize;

        match l {
            0 => {
                let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);
                (pref, pref * w.f0f[0])
            }
            1 => xw_f_overlap_m0_prepared_const::<T, 1>(w, x_ex, w_ex, scratch, tol),
            2 => xw_f_overlap_m0_prepared_const::<T, 2>(w, x_ex, w_ex, scratch, tol),
            3 => xw_f_overlap_m0_prepared_const::<T, 3>(w, x_ex, w_ex, scratch, tol),
            4 => xw_f_overlap_m0_prepared_const::<T, 4>(w, x_ex, w_ex, scratch, tol),
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
        // Select the AVX2 const-rank kernel for four same-rank one-body/overlap pairs.
        match l {
            1 => xw_f_overlap_m0_prepared_f64x4_const::<T, 1>(w, x_ex, w_ex, overlap, fock),
            2 => xw_f_overlap_m0_prepared_f64x4_const::<T, 2>(w, x_ex, w_ex, overlap, fock),
            3 => xw_f_overlap_m0_prepared_f64x4_const::<T, 3>(w, x_ex, w_ex, overlap, fock),
            4 => xw_f_overlap_m0_prepared_f64x4_const::<T, 4>(w, x_ex, w_ex, overlap, fock),
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
        // Select the AVX-512 const-rank kernel for eight same-rank one-body/overlap pairs.
        match l {
            1 => xw_f_overlap_m0_prepared_f64x8_const::<T, 1>(w, x_ex, w_ex, overlap, fock),
            2 => xw_f_overlap_m0_prepared_f64x8_const::<T, 2>(w, x_ex, w_ex, overlap, fock),
            3 => xw_f_overlap_m0_prepared_f64x8_const::<T, 3>(w, x_ex, w_ex, overlap, fock),
            4 => xw_f_overlap_m0_prepared_f64x8_const::<T, 4>(w, x_ex, w_ex, overlap, fock),
            _ => unreachable!(),
        }
    }
}

/// Prepare and evaluate the fixed-rank `L` overlap and generalised-Fock matrix element for `m = 0`.
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}.`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}\mathcal F_{\eta z}^{(0,0)}].`
/// `For L = 1 and \mathbf D_{\mathrm{ov}} = [D_{00}], this reduces to`
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}D_{00} - \mathcal F_{r_0c_0}^{(0,0)}].`
/// `For L = 2, the replacement contribution is`
/// `\det\mathbf D_{\mathrm{ov}}^{0\rightarrow\mathcal F_0}`
/// `+ \det\mathbf D_{\mathrm{ov}}^{1\rightarrow\mathcal F_1}.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `x_ex`: Excitation defining the bra determinant.
/// - `w_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving determinant labels, determinant entries and cofactors.
/// - `tol`: Numerical tolerance used when evaluating determinants and adjugate-transpose matrices.
/// # Returns
/// - `(T, T)`: Same-spin `(overlap, generalised-Fock matrix element)` for fixed `L` and `m = 0`.
#[inline(always)]
fn xw_f_overlap_m0_prepared_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
            scratch.ensure_same(L);

            // Construct the rows r_eta and columns c_z of D_ov from the x- and w-excitation
            // operators in the order used by the GNME determinant.
            construct_determinant_indices(
                x_ex,
                w_ex,
                w,
                scratch.rows.as_mut_slice(),
                scratch.cols.as_mut_slice(),
            );

            // For m = 0, every column uses the m_i = 0 fundamental contractions:
            // D_ov[eta,z] = X^{(0)}_{r_eta c_z} for eta >= z, otherwise Y^{(0)}_{r_eta c_z}.
            let x0 = w.x(0);
            let y0 = w.y(0);
            build_d_const::<T, L>(
                scratch.det0.as_mut_slice(),
                &x0,
                &y0,
                scratch.rows.as_slice(),
                scratch.cols.as_slice(),
            );

            let det0 = &scratch.det0.as_slice()[..L * L];
            let pref = w.phase * <T as From<f64>>::from(w.tilde_s_prod);

            // Evaluate det D_ov and cof[D_ov]_{eta z}=(-1)^{eta+z} det D_ov[eta|z]
            // for the overlap part and the one-body column replacements.
            if let Some(det) = adjugate_transpose_const::<T, L>(
                scratch.adjt_det.as_mut_slice(),
                scratch.invs.as_mut_slice(),
                scratch.lu.as_mut_slice(),
                det0,
                tol,
            ) {
                let n = w.n();
                let fsl = w.ff_t_slice(0, 0);
                let rows = scratch.rows.as_slice();
                let cols = scratch.cols.as_slice();
                let cof = scratch.adjt_det.as_slice();
                let mut repl = <T as From<f64>>::from(0.0);

                // Laplace expansion of the inserted one-body row gives
                // sum_z det D_ov^{z -> F_z} = sum_{eta,z} cof[D_ov]_{eta z} F_{eta z}.
                for z in 0..L {
                    let base = cols[z] * n;

                    for eta in 0..L {
                        repl += cof[eta * L + z] * fsl[base + rows[eta]];
                    }
                }

                // Return S_tilde det D_ov and S_tilde [F_0 det D_ov - C:F].
                (pref * det, pref * (det * w.f0f[0] - repl))
            } else {
                (<T as From<f64>>::from(0.0), <T as From<f64>>::from(0.0))
            }
        }
    )
}

/// Prepare and evaluate 4 independent real fixed-rank `L` overlap and generalised-Fock matrix
/// elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair with
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}`.
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}`
/// `\mathcal F_{\eta z}^{(0,0)}]`.
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
unsafe fn xw_f_overlap_m0_prepared_f64x4_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 4],
    w_ex: &[ExcitationSpinCache; 4],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
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
                let mut d_lanes = [[0.0f64; 4]; 16];
                let mut f_lanes = [[0.0f64; 4]; 16];

                for lane in 0..4 {
                    let x_data = x_ex.get_unchecked(lane);
                    let w_data = w_ex.get_unchecked(lane);
                    let mut rows = [0usize; L];
                    let mut cols = [0usize; L];

                    construct_determinant_indices_const::<T, L>(
                        x_data.rank,
                        &x_data.indices,
                        &w_data.indices,
                        w,
                        &mut rows,
                        &mut cols,
                    );

                    // Pack D_ov and the one-body replacement column entries for the
                    // same four Wick pairs before evaluating cof[D_ov] lane-wise.
                    for eta in 0..L {
                        let d_base = eta * L;

                        for z in 0..L {
                            let src = rows[eta] * n + cols[z];
                            d_lanes[d_base + z][lane] = if eta >= z { x0[src] } else { y0[src] };
                            f_lanes[d_base + z][lane] = fsl[cols[z] * n + rows[eta]];
                        }
                    }
                }

                let zero_v = _mm256_setzero_pd();
                let one_v = _mm256_set1_pd(1.0);

                let (det, repl) = if L == 4 {
                    let dvec =
                        |row: usize, col: usize| _mm256_loadu_pd(d_lanes[row * L + col].as_ptr());
                    let fvec =
                        |row: usize, col: usize| _mm256_loadu_pd(f_lanes[row * L + col].as_ptr());

                    // Evaluate the `2 x 2` minor `ab - cd` with the same operation order as the previous kernel.
                    let minor = |a, b, c, d| _mm256_fmsub_pd(a, b, _mm256_mul_pd(c, d));

                    // Evaluate `C = a M_0 - b M_1 + c M_2` with the original `fmsub` then `fmadd` ordering.
                    let cof_pos = |a, m0, b, m1, c, m2| {
                        let t = _mm256_fmsub_pd(a, m0, _mm256_mul_pd(b, m1));
                        _mm256_fmadd_pd(c, m2, t)
                    };

                    // Evaluate `C = -(a M_0 - b M_1 + c M_2)` without reassociating the `3 x 3` determinant.
                    let cof_neg = |a, m0, b, m1, c, m2| {
                        let t = _mm256_fmsub_pd(a, m0, _mm256_mul_pd(b, m1));
                        let value = _mm256_fmadd_pd(c, m2, t);
                        _mm256_sub_pd(_mm256_setzero_pd(), value)
                    };

                    // The old 16 cofactors contain `16 x 3 = 48` minor occurrences. Preserving their exact
                    // expansions leaves `6 + 6 + 6 = 18` distinct minors, so 18 is the lower bound for this DAG.
                    // AVX2 cannot keep all 16 `D_{ij}` plus these intermediates live, so `D_{ij}` is reloaded by group.

                    // `B_{ab} = D_{2a}D_{3b} - D_{2b}D_{3a}` supplies cofactor rows 0 and 1.
                    let (b01, b02, b03, b12, b13, b23) = {
                        let d20 = dvec(2, 0);
                        let d21 = dvec(2, 1);
                        let d22 = dvec(2, 2);
                        let d23 = dvec(2, 3);

                        let d30 = dvec(3, 0);
                        let d31 = dvec(3, 1);

                        let b01 = minor(d20, d31, d21, d30);

                        let d32 = dvec(3, 2);

                        let b02 = minor(d20, d32, d22, d30);
                        let b12 = minor(d21, d32, d22, d31);

                        let d33 = dvec(3, 3);

                        let b03 = minor(d20, d33, d23, d30);
                        let b13 = minor(d21, d33, d23, d31);
                        let b23 = minor(d22, d33, d23, d32);

                        (b01, b02, b03, b12, b13, b23)
                    };

                    // Form `det(\mathbf D) = \sum_j D_{0j}C_{0j}` and row 0 of `C:\mathcal F`.
                    let mut det_v = _mm256_setzero_pd();
                    let mut repl0 = _mm256_setzero_pd();

                    {
                        let d10 = dvec(1, 0);
                        let d11 = dvec(1, 1);
                        let d12 = dvec(1, 2);
                        let d13 = dvec(1, 3);

                        let cof00 = cof_pos(d11, b23, d12, b13, d13, b12);
                        det_v = _mm256_fmadd_pd(dvec(0, 0), cof00, det_v);
                        repl0 = _mm256_fmadd_pd(fvec(0, 0), cof00, repl0);

                        let cof01 = cof_neg(d10, b23, d12, b03, d13, b02);
                        det_v = _mm256_fmadd_pd(dvec(0, 1), cof01, det_v);
                        repl0 = _mm256_fmadd_pd(fvec(0, 1), cof01, repl0);

                        let cof02 = cof_pos(d10, b13, d11, b03, d13, b01);
                        det_v = _mm256_fmadd_pd(dvec(0, 2), cof02, det_v);
                        repl0 = _mm256_fmadd_pd(fvec(0, 2), cof02, repl0);

                        let cof03 = cof_neg(d10, b12, d11, b02, d12, b01);
                        det_v = _mm256_fmadd_pd(dvec(0, 3), cof03, det_v);
                        repl0 = _mm256_fmadd_pd(fvec(0, 3), cof03, repl0);
                    }

                    // Store `det(\mathbf D)` at its first natural endpoint so it does not remain live across all cofactors.
                    let mut det_lane = [0.0f64; 4];
                    _mm256_storeu_pd(det_lane.as_mut_ptr(), det_v);

                    // Reuse the same six `B_{ab}` values for row 1 of `C:\mathcal F`.
                    let mut repl1 = _mm256_setzero_pd();

                    {
                        let d00 = dvec(0, 0);
                        let d01 = dvec(0, 1);
                        let d02 = dvec(0, 2);
                        let d03 = dvec(0, 3);

                        let cof10 = cof_neg(d01, b23, d02, b13, d03, b12);
                        repl1 = _mm256_fmadd_pd(fvec(1, 0), cof10, repl1);

                        let cof11 = cof_pos(d00, b23, d02, b03, d03, b02);
                        repl1 = _mm256_fmadd_pd(fvec(1, 1), cof11, repl1);

                        let cof12 = cof_neg(d00, b13, d01, b03, d03, b01);
                        repl1 = _mm256_fmadd_pd(fvec(1, 2), cof12, repl1);

                        let cof13 = cof_pos(d00, b12, d01, b02, d02, b01);
                        repl1 = _mm256_fmadd_pd(fvec(1, 3), cof13, repl1);
                    }

                    let repl01 = _mm256_add_pd(repl0, repl1);

                    // `Q_{ab} = D_{1a}D_{3b} - D_{1b}D_{3a}` supplies cofactor row 2.
                    let (q01, q02, q03, q12, q13, q23) = {
                        let d10 = dvec(1, 0);
                        let d11 = dvec(1, 1);
                        let d12 = dvec(1, 2);
                        let d13 = dvec(1, 3);

                        let d30 = dvec(3, 0);
                        let d31 = dvec(3, 1);

                        let q01 = minor(d10, d31, d11, d30);

                        let d32 = dvec(3, 2);

                        let q02 = minor(d10, d32, d12, d30);
                        let q12 = minor(d11, d32, d12, d31);

                        let d33 = dvec(3, 3);

                        let q03 = minor(d10, d33, d13, d30);
                        let q13 = minor(d11, d33, d13, d31);
                        let q23 = minor(d12, d33, d13, d32);

                        (q01, q02, q03, q12, q13, q23)
                    };

                    let mut repl2 = _mm256_setzero_pd();

                    {
                        let d00 = dvec(0, 0);
                        let d01 = dvec(0, 1);
                        let d02 = dvec(0, 2);
                        let d03 = dvec(0, 3);

                        let cof20 = cof_pos(d01, q23, d02, q13, d03, q12);
                        repl2 = _mm256_fmadd_pd(fvec(2, 0), cof20, repl2);

                        let cof21 = cof_neg(d00, q23, d02, q03, d03, q02);
                        repl2 = _mm256_fmadd_pd(fvec(2, 1), cof21, repl2);

                        let cof22 = cof_pos(d00, q13, d01, q03, d03, q01);
                        repl2 = _mm256_fmadd_pd(fvec(2, 2), cof22, repl2);

                        let cof23 = cof_neg(d00, q12, d01, q02, d02, q01);
                        repl2 = _mm256_fmadd_pd(fvec(2, 3), cof23, repl2);
                    }

                    // `R_{ab} = D_{1a}D_{2b} - D_{1b}D_{2a}` supplies cofactor row 3.
                    let (r01, r02, r03, r12, r13, r23) = {
                        let d10 = dvec(1, 0);
                        let d11 = dvec(1, 1);
                        let d12 = dvec(1, 2);
                        let d13 = dvec(1, 3);

                        let d20 = dvec(2, 0);
                        let d21 = dvec(2, 1);

                        let r01 = minor(d10, d21, d11, d20);

                        let d22 = dvec(2, 2);

                        let r02 = minor(d10, d22, d12, d20);
                        let r12 = minor(d11, d22, d12, d21);

                        let d23 = dvec(2, 3);

                        let r03 = minor(d10, d23, d13, d20);
                        let r13 = minor(d11, d23, d13, d21);
                        let r23 = minor(d12, d23, d13, d22);

                        (r01, r02, r03, r12, r13, r23)
                    };

                    let mut repl3 = _mm256_setzero_pd();

                    {
                        let d00 = dvec(0, 0);
                        let d01 = dvec(0, 1);
                        let d02 = dvec(0, 2);
                        let d03 = dvec(0, 3);

                        let cof30 = cof_neg(d01, r23, d02, r13, d03, r12);
                        repl3 = _mm256_fmadd_pd(fvec(3, 0), cof30, repl3);

                        let cof31 = cof_pos(d00, r23, d02, r03, d03, r02);
                        repl3 = _mm256_fmadd_pd(fvec(3, 1), cof31, repl3);

                        let cof32 = cof_neg(d00, r13, d01, r03, d03, r01);
                        repl3 = _mm256_fmadd_pd(fvec(3, 2), cof32, repl3);

                        let cof33 = cof_pos(d00, r12, d01, r02, d02, r01);
                        repl3 = _mm256_fmadd_pd(fvec(3, 3), cof33, repl3);
                    }

                    // Preserve the previous contraction tree `((repl0 + repl1) + (repl2 + repl3))`.
                    let repl23 = _mm256_add_pd(repl2, repl3);
                    let repl_v = _mm256_add_pd(repl01, repl23);
                    let det_v = _mm256_loadu_pd(det_lane.as_ptr());

                    (det_v, repl_v)
                } else {
                    let mut d = [_mm256_setzero_pd(); 16];
                    let mut ff = [_mm256_setzero_pd(); 16];
                    for idx in 0..L * L {
                        d[idx] = _mm256_loadu_pd(d_lanes[idx].as_ptr());
                        ff[idx] = _mm256_loadu_pd(f_lanes[idx].as_ptr());
                    }

                    let mut cof = [zero_v; 16];
                    let det;

                    if L == 1 {
                        cof[0] = one_v;
                        det = d[0];
                    } else if L == 2 {
                        cof[0] = d[3];
                        cof[1] = _mm256_sub_pd(zero_v, d[2]);
                        cof[2] = _mm256_sub_pd(zero_v, d[1]);
                        cof[3] = d[0];
                        det = _mm256_fmsub_pd(d[0], d[3], _mm256_mul_pd(d[1], d[2]));
                    } else {
                        for eta in 0..L {
                            let mut rows_keep = [0usize; 2];
                            let mut ri = 0usize;
                            for r in 0..L {
                                if r != eta {
                                    rows_keep[ri] = r;
                                    ri += 1;
                                }
                            }

                            for z in 0..L {
                                let mut cols_keep = [0usize; 2];
                                let mut ci = 0usize;
                                for c in 0..L {
                                    if c != z {
                                        cols_keep[ci] = c;
                                        ci += 1;
                                    }
                                }

                                let value = _mm256_fmsub_pd(
                                    d[rows_keep[0] * L + cols_keep[0]],
                                    d[rows_keep[1] * L + cols_keep[1]],
                                    _mm256_mul_pd(
                                        d[rows_keep[0] * L + cols_keep[1]],
                                        d[rows_keep[1] * L + cols_keep[0]],
                                    ),
                                );
                                cof[eta * L + z] = if ((eta + z) & 1) == 0 {
                                    value
                                } else {
                                    _mm256_sub_pd(zero_v, value)
                                };
                            }
                        }

                        let mut det_acc = _mm256_mul_pd(d[0], cof[0]);
                        for z in 1..L {
                            det_acc = _mm256_fmadd_pd(d[z], cof[z], det_acc);
                        }
                        det = det_acc;
                    }

                    let mut repl = _mm256_setzero_pd();
                    for eta in 0..L {
                        for z in 0..L {
                            repl = _mm256_fmadd_pd(cof[eta * L + z], ff[eta * L + z], repl);
                        }
                    }

                    (det, repl)
                };

                let overlap_v = _mm256_mul_pd(det, _mm256_set1_pd(pref));
                let fock_v = _mm256_mul_pd(
                    _mm256_fmsub_pd(det, _mm256_set1_pd(f0), repl),
                    _mm256_set1_pd(pref),
                );

                let mut det_lane = [0.0f64; 4];
                let mut overlap_lane = [0.0f64; 4];
                let mut fock_lane = [0.0f64; 4];
                _mm256_storeu_pd(det_lane.as_mut_ptr(), det);
                _mm256_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
                _mm256_storeu_pd(fock_lane.as_mut_ptr(), fock_v);

                // Store four packed overlap and one-body lanes, zeroing non-finite determinant lanes.
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
    )
}

/// Prepare and evaluate 8 independent real fixed-rank `L` overlap and generalised-Fock matrix
/// elements for `m = 0`.
/// Each SIMD lane is one complete Wick pair with
/// `S = {}^{xw}\tilde S\det\mathbf D_{\mathrm{ov}}`.
/// `F = {}^{xw}\tilde S[{}^x F_0^{(0)}\det\mathbf D_{\mathrm{ov}}`
/// `- \sum_{\eta,z}\operatorname{cof}[\mathbf D_{\mathrm{ov}}]_{\eta z}`
/// `\mathcal F_{\eta z}^{(0,0)}]`.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates with `T = f64` and `m = 0`.
/// - `x_ex`: 8 x-reference excitations with cached ranks and decoded orbital indices.
/// - `w_ex`: 8 w-reference excitations with cached ranks and decoded orbital indices.
/// - `overlap`: Real overlap output slice in SIMD-lane order.
/// - `fock`: Real generalised-Fock output slice in SIMD-lane order.
/// # Returns
/// - `()`: Writes 8 overlaps and generalised-Fock matrix elements in SIMD-lane order.
/// # Safety
/// - The caller must ensure `T = f64` and CPU support for `AVX-512F`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_f_overlap_m0_prepared_f64x8_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    x_ex: &[ExcitationSpinCache; 8],
    w_ex: &[ExcitationSpinCache; 8],
    overlap: &mut [f64],
    fock: &mut [f64],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_f_overlap_m0_const,
        {
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
                let mut d_lanes = [[0.0f64; 8]; 16];
                let mut f_lanes = [[0.0f64; 8]; 16];

                for lane in 0..8 {
                    let x_data = x_ex.get_unchecked(lane);
                    let w_data = w_ex.get_unchecked(lane);
                    let mut rows = [0usize; L];
                    let mut cols = [0usize; L];

                    construct_determinant_indices_const::<T, L>(
                        x_data.rank,
                        &x_data.indices,
                        &w_data.indices,
                        w,
                        &mut rows,
                        &mut cols,
                    );

                    // Pack D_ov and the one-body replacement column entries for the
                    // same eight Wick pairs before evaluating cof[D_ov] lane-wise.
                    for eta in 0..L {
                        let d_base = eta * L;

                        for z in 0..L {
                            let src = rows[eta] * n + cols[z];
                            d_lanes[d_base + z][lane] = if eta >= z { x0[src] } else { y0[src] };
                            f_lanes[d_base + z][lane] = fsl[cols[z] * n + rows[eta]];
                        }
                    }
                }

                let zero_v = _mm512_setzero_pd();
                let one_v = _mm512_set1_pd(1.0);
                let mut d = [_mm512_setzero_pd(); 16];
                let mut ff = [_mm512_setzero_pd(); 16];
                for idx in 0..L * L {
                    d[idx] = _mm512_loadu_pd(d_lanes[idx].as_ptr());
                    ff[idx] = _mm512_loadu_pd(f_lanes[idx].as_ptr());
                }

                let (det, repl) = if L == 4 {
                    let fvec = |row: usize, col: usize| ff[row * L + col];

                    // Evaluate the `2 x 2` minor `ab - cd` with the same operation order as the previous kernel.
                    let minor = |a, b, c, d| _mm512_fmsub_pd(a, b, _mm512_mul_pd(c, d));

                    // Evaluate `C = a M_0 - b M_1 + c M_2` with the original `fmsub` then `fmadd` ordering.
                    let cof_pos = |a, m0, b, m1, c, m2| {
                        let t = _mm512_fmsub_pd(a, m0, _mm512_mul_pd(b, m1));
                        _mm512_fmadd_pd(c, m2, t)
                    };

                    // Evaluate `C = -(a M_0 - b M_1 + c M_2)` without reassociating the `3 x 3` determinant.
                    let cof_neg = |a, m0, b, m1, c, m2| {
                        let t = _mm512_fmsub_pd(a, m0, _mm512_mul_pd(b, m1));
                        let value = _mm512_fmadd_pd(c, m2, t);
                        _mm512_sub_pd(_mm512_setzero_pd(), value)
                    };

                    // Preserving the explicit `L = 4` cofactor DAG reduces 48 minor occurrences to exactly
                    // `3 binom(4,2) = 18` distinct minors. Each is required, so 18 is the lower bound for this DAG.
                    // AVX-512 has 32 ZMM registers, allowing the 16 distinct `D_{ij}` inputs to remain resident.

                    let d00 = d[0];
                    let d01 = d[1];
                    let d02 = d[2];
                    let d03 = d[3];

                    let d10 = d[4];
                    let d11 = d[5];
                    let d12 = d[6];
                    let d13 = d[7];

                    let d20 = d[8];
                    let d21 = d[9];
                    let d22 = d[10];
                    let d23 = d[11];

                    let d30 = d[12];
                    let d31 = d[13];
                    let d32 = d[14];
                    let d33 = d[15];

                    // `B_{ab}` contains the six row-pair `(2,3)` minors used by cofactor rows 0 and 1.
                    let b01 = minor(d20, d31, d21, d30);
                    let b02 = minor(d20, d32, d22, d30);
                    let b03 = minor(d20, d33, d23, d30);
                    let b12 = minor(d21, d32, d22, d31);
                    let b13 = minor(d21, d33, d23, d31);
                    let b23 = minor(d22, d33, d23, d32);

                    let mut det_v = _mm512_setzero_pd();
                    let mut repl0 = _mm512_setzero_pd();
                    let mut repl1 = _mm512_setzero_pd();

                    // Form `det(\mathbf D)` through row 0 while contracting cofactor row 0 with `\mathcal F`.
                    let cof00 = cof_pos(d11, b23, d12, b13, d13, b12);
                    det_v = _mm512_fmadd_pd(d00, cof00, det_v);
                    repl0 = _mm512_fmadd_pd(fvec(0, 0), cof00, repl0);

                    let cof01 = cof_neg(d10, b23, d12, b03, d13, b02);
                    det_v = _mm512_fmadd_pd(d01, cof01, det_v);
                    repl0 = _mm512_fmadd_pd(fvec(0, 1), cof01, repl0);

                    let cof02 = cof_pos(d10, b13, d11, b03, d13, b01);
                    det_v = _mm512_fmadd_pd(d02, cof02, det_v);
                    repl0 = _mm512_fmadd_pd(fvec(0, 2), cof02, repl0);

                    let cof03 = cof_neg(d10, b12, d11, b02, d12, b01);
                    det_v = _mm512_fmadd_pd(d03, cof03, det_v);
                    repl0 = _mm512_fmadd_pd(fvec(0, 3), cof03, repl0);

                    // The same `B_{ab}` values give cofactor row 1 without any further minor evaluation.
                    let cof10 = cof_neg(d01, b23, d02, b13, d03, b12);
                    repl1 = _mm512_fmadd_pd(fvec(1, 0), cof10, repl1);

                    let cof11 = cof_pos(d00, b23, d02, b03, d03, b02);
                    repl1 = _mm512_fmadd_pd(fvec(1, 1), cof11, repl1);

                    let cof12 = cof_neg(d00, b13, d01, b03, d03, b01);
                    repl1 = _mm512_fmadd_pd(fvec(1, 2), cof12, repl1);

                    let cof13 = cof_pos(d00, b12, d01, b02, d02, b01);
                    repl1 = _mm512_fmadd_pd(fvec(1, 3), cof13, repl1);

                    let repl01 = _mm512_add_pd(repl0, repl1);

                    // `Q_{ab}` contains the six row-pair `(1,3)` minors required by cofactor row 2.
                    let q01 = minor(d10, d31, d11, d30);
                    let q02 = minor(d10, d32, d12, d30);
                    let q03 = minor(d10, d33, d13, d30);
                    let q12 = minor(d11, d32, d12, d31);
                    let q13 = minor(d11, d33, d13, d31);
                    let q23 = minor(d12, d33, d13, d32);

                    let mut repl2 = _mm512_setzero_pd();

                    let cof20 = cof_pos(d01, q23, d02, q13, d03, q12);
                    repl2 = _mm512_fmadd_pd(fvec(2, 0), cof20, repl2);

                    let cof21 = cof_neg(d00, q23, d02, q03, d03, q02);
                    repl2 = _mm512_fmadd_pd(fvec(2, 1), cof21, repl2);

                    let cof22 = cof_pos(d00, q13, d01, q03, d03, q01);
                    repl2 = _mm512_fmadd_pd(fvec(2, 2), cof22, repl2);

                    let cof23 = cof_neg(d00, q12, d01, q02, d02, q01);
                    repl2 = _mm512_fmadd_pd(fvec(2, 3), cof23, repl2);

                    // `R_{ab}` contains the final six row-pair `(1,2)` minors required by cofactor row 3.
                    let r01 = minor(d10, d21, d11, d20);
                    let r02 = minor(d10, d22, d12, d20);
                    let r03 = minor(d10, d23, d13, d20);
                    let r12 = minor(d11, d22, d12, d21);
                    let r13 = minor(d11, d23, d13, d21);
                    let r23 = minor(d12, d23, d13, d22);

                    let mut repl3 = _mm512_setzero_pd();

                    let cof30 = cof_neg(d01, r23, d02, r13, d03, r12);
                    repl3 = _mm512_fmadd_pd(fvec(3, 0), cof30, repl3);

                    let cof31 = cof_pos(d00, r23, d02, r03, d03, r02);
                    repl3 = _mm512_fmadd_pd(fvec(3, 1), cof31, repl3);

                    let cof32 = cof_neg(d00, r13, d01, r03, d03, r01);
                    repl3 = _mm512_fmadd_pd(fvec(3, 2), cof32, repl3);

                    let cof33 = cof_pos(d00, r12, d01, r02, d02, r01);
                    repl3 = _mm512_fmadd_pd(fvec(3, 3), cof33, repl3);

                    // Preserve `C:\mathcal F = (repl0 + repl1) + (repl2 + repl3)` from the old kernel.
                    let repl23 = _mm512_add_pd(repl2, repl3);
                    let repl_v = _mm512_add_pd(repl01, repl23);

                    (det_v, repl_v)
                } else {
                    let mut cof = [zero_v; 16];
                    let det;

                    if L == 1 {
                        cof[0] = one_v;
                        det = d[0];
                    } else if L == 2 {
                        cof[0] = d[3];
                        cof[1] = _mm512_sub_pd(zero_v, d[2]);
                        cof[2] = _mm512_sub_pd(zero_v, d[1]);
                        cof[3] = d[0];
                        det = _mm512_fmsub_pd(d[0], d[3], _mm512_mul_pd(d[1], d[2]));
                    } else {
                        for eta in 0..L {
                            let mut rows_keep = [0usize; 2];
                            let mut ri = 0usize;
                            for r in 0..L {
                                if r != eta {
                                    rows_keep[ri] = r;
                                    ri += 1;
                                }
                            }

                            for z in 0..L {
                                let mut cols_keep = [0usize; 2];
                                let mut ci = 0usize;
                                for c in 0..L {
                                    if c != z {
                                        cols_keep[ci] = c;
                                        ci += 1;
                                    }
                                }

                                let value = _mm512_fmsub_pd(
                                    d[rows_keep[0] * L + cols_keep[0]],
                                    d[rows_keep[1] * L + cols_keep[1]],
                                    _mm512_mul_pd(
                                        d[rows_keep[0] * L + cols_keep[1]],
                                        d[rows_keep[1] * L + cols_keep[0]],
                                    ),
                                );
                                cof[eta * L + z] = if ((eta + z) & 1) == 0 {
                                    value
                                } else {
                                    _mm512_sub_pd(zero_v, value)
                                };
                            }
                        }

                        let mut det_acc = _mm512_mul_pd(d[0], cof[0]);
                        for z in 1..L {
                            det_acc = _mm512_fmadd_pd(d[z], cof[z], det_acc);
                        }
                        det = det_acc;
                    }

                    let mut repl = _mm512_setzero_pd();
                    for eta in 0..L {
                        for z in 0..L {
                            repl = _mm512_fmadd_pd(cof[eta * L + z], ff[eta * L + z], repl);
                        }
                    }

                    (det, repl)
                };

                let overlap_v = _mm512_mul_pd(det, _mm512_set1_pd(pref));
                let fock_v = _mm512_mul_pd(
                    _mm512_fmsub_pd(det, _mm512_set1_pd(f0), repl),
                    _mm512_set1_pd(pref),
                );

                let mut det_lane = [0.0f64; 8];
                let mut overlap_lane = [0.0f64; 8];
                let mut fock_lane = [0.0f64; 8];
                _mm512_storeu_pd(det_lane.as_mut_ptr(), det);
                _mm512_storeu_pd(overlap_lane.as_mut_ptr(), overlap_v);
                _mm512_storeu_pd(fock_lane.as_mut_ptr(), fock_v);

                // Store eight packed overlap and one-body lanes, zeroing non-finite determinant lanes.
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
    )
}

/// Prepare and evaluate the generic-rank overlap and generalised-Fock matrix element for `m = 0`.
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
            // Generic m = 0 path: build D_ov once, then use cof[D_ov] for every
            // one-column F replacement determinant.
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
        // General zero-overlap path: prepare the all-zero and all-one endpoint determinants,
        // then sum the GNME one-body expression over every allowed mixed distribution.
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
