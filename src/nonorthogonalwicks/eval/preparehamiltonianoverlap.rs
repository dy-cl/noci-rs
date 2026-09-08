// nonorthogonalwicks/eval/preparehamiltonianoverlap.rs

// Standard library imports.
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

// External crate imports.
use num_complex::Complex64;

// Crate-root imports.
#[cfg(target_arch = "x86_64")]
use crate::maths::{
    C64x4, C64x8, F64x4, F64x8, Simd, adjugate_transpose_simd_const, det_simd_const,
};
use crate::maths::{
    adjugate_transpose_const, adjugate_transpose_dynamic, det_const, det_dynamic,
    second_minor_const, second_minor_dynamic,
};
use crate::noci::NOCIScalar;
use crate::time_call;
use crate::{DetState, Excitation, ExcitationCache, ExcitationSpinCache, ReducedTwoSpinDetState};

// Parent/sibling imports.
use super::super::scratch::WickScratchSpin;
use super::super::view::WicksPairView;
use super::dispatch::{
    HAMNRANKS, HAMRADIX, HAMRANKS, HAMSPACE, dispatch_hamiltonian_ranks,
    dispatch_hamiltonian_ranks_inner, dispatch_hamiltonian_scalar_ranks,
};
use super::helpers::{DetBranches, DetIndex, Minor, ReplacementLayout};
use super::helpers::{
    adjugate_transpose_generic, bit, column_replacement_correction, column_replacement_det,
    get_det_adjt_diff, ii_replacement, j_replacement, jslot, minor_adjt, mix_dets_same,
};
use super::prepare::prepare_same;

/// Evaluate the Hamiltonian and overlap matrix elements between two determinants generated from
/// one ordered pair of nonorthogonal references.
/// For `m_\alpha = m_\beta = 0`, the scalar fixed path is used when each determinant's total
/// alpha-plus-beta excitation rank is no larger than build-time `MAXEXCIT`.
/// Other `m = 0` cases use the generic fused cofactor kernel. Cases with
/// `m_\alpha > 0` or `m_\beta > 0` use the generic fused distribution kernel, so this evaluator
/// imposes no rank cutoff beyond the underlying excitation representation.
/// The fixed path evaluates `S`, `H_1`, `H_{2,\alpha\alpha}`, `H_{2,\beta\beta}` and
/// `H_{2,\alpha\beta}` together so determinants, cofactors and second minors are reused.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Full bra excitation used by the generic fallback.
/// - `w_ex`: Full ket excitation used by the generic fallback.
/// - `x_cache`: Cached bra excitation ranks and orbital labels per spin.
/// - `w_cache`: Cached ket excitation ranks and orbital labels per spin.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `scratch`: Reusable Wick workspace for generic-rank and nonzero-`m` evaluation.
/// - `tol`: Numerical tolerance used by generic determinant and adjugate evaluation.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
pub(crate) fn xw_hamiltonian_overlap_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    ex: (&Excitation, &Excitation),
    cache: (&ExcitationCache, &ExcitationCache),
    excitation_phase: f64,
    enuc: f64,
    scratch: &mut WickScratchSpin<T>,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_prepared,
        {
            let (x_ex, w_ex) = ex;
            let (x_cache, w_cache) = cache;

            if w.aa.m == 0 && w.bb.m == 0 {
                let ranks = (
                    usize::from(x_cache.alpha.rank),
                    usize::from(w_cache.alpha.rank),
                    usize::from(x_cache.beta.rank),
                    usize::from(w_cache.beta.rank),
                );

                if let Some(value) = xw_hamiltonian_overlap_m0_prepared(
                    w,
                    ranks,
                    x_cache,
                    w_cache,
                    excitation_phase,
                    enuc,
                ) {
                    return value;
                }

                return xw_hamiltonian_overlap_m0_gen_prepared(
                    w,
                    x_ex,
                    w_ex,
                    excitation_phase,
                    enuc,
                    scratch,
                    tol,
                );
            }

            xw_hamiltonian_overlap_gen_prepared(w, x_ex, w_ex, excitation_phase, enuc, scratch, tol)
        }
    )
}

/// Evaluate batched Hamiltonian and overlap matrix elements for one ordered reference pair.
/// Every request supplied to this routine already belongs to that reference pair. Requests are
/// streamed through generated `(R_{x,\alpha},R_{w,\alpha},R_{x,\beta},R_{w,\beta})` bins when
/// `m_\alpha = m_\beta = 0`. The widest matching real or complex SIMD kernel is selected internally,
/// incomplete bins are padded with one valid request, and unsupported requests use the existing
/// prepared scalar evaluator.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `basis`: Determinant basis used only by generic fallback evaluation.
/// - `reduced_basis`: Compact two-spin metadata keyed by global determinant index.
/// - `requests`: Tuples `(output, a, b)` containing output position and determinant indices.
/// - `enuc`: Nuclear repulsion energy.
/// - `scratch`: Reusable Wick workspace for scalar generic-rank evaluation.
/// - `tol`: Numerical tolerance used by generic determinant and adjugate evaluation.
/// - `out`: Hamiltonian and overlap results aligned with the original request order.
/// # Returns:
/// - `()`: Writes every matrix element in `requests` into `out`.
pub(crate) fn xw_hamiltonian_overlap_prepared_batched<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    basis: (&[DetState<T>], &[ReducedTwoSpinDetState]),
    requests: &[(usize, usize, usize)],
    enuc: f64,
    scratch: &mut WickScratchSpin<T>,
    tol: f64,
    out: &mut [(T, T)],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_prepared_batched,
        {
            #[cfg(target_arch = "x86_64")]
            if w.aa.m == 0 && w.bb.m == 0 {
                unsafe {
                    if TypeId::of::<T>() == TypeId::of::<f64>() {
                        let w_f64 = &*std::ptr::from_ref(w).cast::<WicksPairView<'_, f64>>();
                        let basis_f64 = std::slice::from_raw_parts(
                            basis.0.as_ptr().cast::<DetState<f64>>(),
                            basis.0.len(),
                        );
                        let scratch_f64 =
                            &mut *std::ptr::from_mut(scratch).cast::<WickScratchSpin<f64>>();
                        let out_f64 = std::slice::from_raw_parts_mut(
                            out.as_mut_ptr().cast::<(f64, f64)>(),
                            out.len(),
                        );

                        if is_x86_feature_detected!("avx512f") {
                            xw_hamiltonian_overlap_prepared_simd(
                                w_f64,
                                (basis_f64, basis.1),
                                requests,
                                (enuc, tol),
                                scratch_f64,
                                out_f64,
                                xw_hamiltonian_overlap_m0_prepared_f64x8,
                            );
                            return;
                        }

                        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                            xw_hamiltonian_overlap_prepared_simd(
                                w_f64,
                                (basis_f64, basis.1),
                                requests,
                                (enuc, tol),
                                scratch_f64,
                                out_f64,
                                xw_hamiltonian_overlap_m0_prepared_f64x4,
                            );
                            return;
                        }
                    }

                    if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                        let w_c64 = &*std::ptr::from_ref(w).cast::<WicksPairView<'_, Complex64>>();
                        let basis_c64 = std::slice::from_raw_parts(
                            basis.0.as_ptr().cast::<DetState<Complex64>>(),
                            basis.0.len(),
                        );
                        let scratch_c64 =
                            &mut *std::ptr::from_mut(scratch).cast::<WickScratchSpin<Complex64>>();
                        let out_c64 = std::slice::from_raw_parts_mut(
                            out.as_mut_ptr().cast::<(Complex64, Complex64)>(),
                            out.len(),
                        );

                        if is_x86_feature_detected!("avx512f") {
                            xw_hamiltonian_overlap_prepared_simd(
                                w_c64,
                                (basis_c64, basis.1),
                                requests,
                                (enuc, tol),
                                scratch_c64,
                                out_c64,
                                xw_hamiltonian_overlap_m0_prepared_c64x8,
                            );
                            return;
                        }

                        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                            xw_hamiltonian_overlap_prepared_simd(
                                w_c64,
                                (basis_c64, basis.1),
                                requests,
                                (enuc, tol),
                                scratch_c64,
                                out_c64,
                                xw_hamiltonian_overlap_m0_prepared_c64x4,
                            );
                            return;
                        }
                    }
                }
            }

            let (basis, reduced_basis) = basis;
            for &(output, a, b) in requests {
                let x_det = &reduced_basis[a];
                let w_det = &reduced_basis[b];
                let x_state = &basis[a];
                let w_state = &basis[b];

                out[output] = xw_hamiltonian_overlap_prepared(
                    w,
                    (&x_state.excitation, &w_state.excitation),
                    (&x_det.excitation_cache, &w_det.excitation_cache),
                    x_det.phase * w_det.phase,
                    enuc,
                    scratch,
                    tol,
                );
            }
        }
    )
}

/// Evaluate one real or complex SIMD Hamiltonian/overlap request group.
/// Requests are separated by all four reference-resolved spin ranks, so every packet uses one
/// fixed-rank determinant, cofactor, same-spin second-minor and mixed-spin cofactor expression.
/// Incomplete packets are padded with a valid lane and unsupported ranks use the scalar evaluator.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `basis`: Full determinant data and compact determinant metadata.
/// - `requests`: Output positions and determinant pairs belonging to this reference pair.
/// - `parameters`: Nuclear repulsion energy and scalar-fallback numerical tolerance.
/// - `scratch`: Reusable scalar fallback workspace.
/// - `out`: Real or complex Hamiltonian and overlap outputs.
/// - `kernel`: Fixed-width real or complex SIMD rank dispatcher.
/// # Returns:
/// - `()`: Writes every request into `out`.
/// # Safety:
/// - The caller must provide a kernel supported by the current CPU.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::type_complexity)]
unsafe fn xw_hamiltonian_overlap_prepared_simd<T: NOCIScalar, const N: usize>(
    w: &WicksPairView<'_, T>,
    basis: (&[DetState<T>], &[ReducedTwoSpinDetState]),
    requests: &[(usize, usize, usize)],
    parameters: (f64, f64),
    scratch: &mut WickScratchSpin<T>,
    out: &mut [(T, T)],
    kernel: for<'a> unsafe fn(
        &WicksPairView<'a, T>,
        (usize, usize, usize, usize),
        (&[ExcitationCache; N], &[ExcitationCache; N]),
        &[f64; N],
        f64,
        (&mut [T; N], &mut [T; N]),
    ),
) {
    let (enuc, tol) = parameters;
    let (basis, reduced_basis) = basis;

    let mut rank_bins = [usize::MAX; HAMSPACE];

    for (bin, &(rxa, rwa, rxb, rwb)) in HAMRANKS.iter().enumerate() {
        let key = ((rxa * HAMRADIX + rwa) * HAMRADIX + rxb) * HAMRADIX + rwb;
        rank_bins[key] = bin;
    }

    let mut x_bins = [[ExcitationCache::default(); N]; HAMNRANKS];
    let mut w_bins = [[ExcitationCache::default(); N]; HAMNRANKS];
    let mut phases = [[1.0f64; N]; HAMNRANKS];
    let mut outputs = [[0usize; N]; HAMNRANKS];
    let mut counts = [0usize; HAMNRANKS];

    unsafe {
        for &(output, a, b) in requests {
            let x_det = &reduced_basis[a];
            let w_det = &reduced_basis[b];
            let x_cache = x_det.excitation_cache;
            let w_cache = w_det.excitation_cache;
            let ranks = (
                usize::from(x_cache.alpha.rank),
                usize::from(w_cache.alpha.rank),
                usize::from(x_cache.beta.rank),
                usize::from(w_cache.beta.rank),
            );
            let bin = if ranks.0 < HAMRADIX
                && ranks.1 < HAMRADIX
                && ranks.2 < HAMRADIX
                && ranks.3 < HAMRADIX
            {
                let key =
                    ((ranks.0 * HAMRADIX + ranks.1) * HAMRADIX + ranks.2) * HAMRADIX + ranks.3;
                rank_bins[key]
            } else {
                usize::MAX
            };

            if bin != usize::MAX {
                let count = counts[bin];
                x_bins[bin][count] = x_cache;
                w_bins[bin][count] = w_cache;
                phases[bin][count] = x_det.phase * w_det.phase;
                outputs[bin][count] = output;
                counts[bin] += 1;

                if counts[bin] == N {
                    let mut h = [T::from_real(0.0); N];
                    let mut s = [T::from_real(0.0); N];
                    kernel(
                        w,
                        HAMRANKS[bin],
                        (&x_bins[bin], &w_bins[bin]),
                        &phases[bin],
                        enuc,
                        (&mut h, &mut s),
                    );

                    for lane in 0..N {
                        out[outputs[bin][lane]] = (h[lane], s[lane]);
                    }
                    counts[bin] = 0;
                }
            } else {
                let x_state = &basis[a];
                let w_state = &basis[b];
                let value = xw_hamiltonian_overlap_prepared(
                    w,
                    (&x_state.excitation, &w_state.excitation),
                    (&x_cache, &w_cache),
                    x_det.phase * w_det.phase,
                    enuc,
                    scratch,
                    tol,
                );
                out[output] = value;
            }
        }

        for bin in 0..counts.len() {
            let count = counts[bin];
            if count == 0 {
                continue;
            }

            let fill_x = x_bins[bin][0];
            let fill_w = w_bins[bin][0];
            let fill_phase = phases[bin][0];
            for lane in count..N {
                x_bins[bin][lane] = fill_x;
                w_bins[bin][lane] = fill_w;
                phases[bin][lane] = fill_phase;
            }

            let mut h = [T::from_real(0.0); N];
            let mut s = [T::from_real(0.0); N];
            kernel(
                w,
                HAMRANKS[bin],
                (&x_bins[bin], &w_bins[bin]),
                &phases[bin],
                enuc,
                (&mut h, &mut s),
            );

            for lane in 0..count {
                out[outputs[bin][lane]] = (h[lane], s[lane]);
            }
        }
    }
}

/// Dispatch an `m_\alpha = m_\beta = 0` Hamiltonian and overlap matrix element to a generated
/// fixed-rank scalar kernel.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `ranks`: Bra/ket alpha and beta ranks `(R_{x,\alpha},R_{w,\alpha},R_{x,\beta},R_{w,\beta})`.
/// - `x_ex`: Cached bra excitation ranks and orbital labels.
/// - `w_ex`: Cached ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `Option<(T, T)>`: Fixed-rank `(H,S)` result or `None` outside the generated region.
#[inline(never)]
fn xw_hamiltonian_overlap_m0_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    ranks: (usize, usize, usize, usize),
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> Option<(T, T)> {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_m0_prepared,
        {
            dispatch_hamiltonian_scalar_ranks!(
                ranks,
                |RXA, RWA, LA, DA, MA, MDA, RXB, RWB, LB, DB, MB, MDB| {
                    Some(xw_hamiltonian_overlap_m0_prepared_const::<
                        T,
                        RXA,
                        RWA,
                        LA,
                        DA,
                        MA,
                        MDA,
                        RXB,
                        RWB,
                        LB,
                        DB,
                        MB,
                        MDB,
                    >(w, x_ex, w_ex, excitation_phase, enuc))
                },
                None,
            )
        }
    )
}

/// Construct fixed-rank Hamiltonian contraction labels from cached excitation data.
/// # Arguments:
/// - `x_ex`: Cached bra same-spin excitation.
/// - `w_ex`: Cached ket same-spin excitation.
/// - `nocc`: Number of occupied orbitals.
/// - `nvirt`: Number of virtual orbitals.
/// - `rows`: Output contraction-row labels.
/// - `cols`: Output contraction-column labels.
/// # Returns:
/// - `()`: Writes `RX + RW` row and column labels.
#[inline(always)]
fn construct_hamiltonian_indices<const RX: usize, const RW: usize, const L: usize>(
    x_ex: &ExcitationSpinCache,
    w_ex: &ExcitationSpinCache,
    nocc: usize,
    nvirt: usize,
    rows: &mut [usize; L],
    cols: &mut [usize; L],
) {
    for i in 0..RX {
        rows[i] = usize::from(x_ex.particles[i]) - nocc;
        cols[i] = usize::from(x_ex.holes[i]);
    }

    for i in 0..RW {
        rows[RX + i] = nvirt + usize::from(w_ex.holes[i]);
        cols[RX + i] = usize::from(w_ex.particles[i]);
    }
}
/// Evaluate the fixed-rank `(L_\alpha, L_\beta)` Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// The contraction determinants, cofactors and required second minors are evaluated
/// directly for this rank pair and reused by all Hamiltonian contributions.
/// For each spin sector `\sigma`, the overlap determinant is
/// `D^\sigma_{ij} = X^{(0)}_{r_i c_j}` for `i >= j` and
/// `D^\sigma_{ij} = Y^{(0)}_{r_i c_j}` for `i < j`.
/// The overlap contribution is
/// `S = p\det\mathbf D_{\alpha,\mathrm{ov}}\det\mathbf D_{\beta,\mathrm{ov}}`, with
/// `p` the excitation phase times the two reduced reference-overlap factors.
/// The one-body contribution in each spin sector is the `m = 0` form
/// `F^\sigma = F^\sigma_0\det\mathbf D_{\sigma,\mathrm{ov}}`
/// `- \sum_z\det\mathbf D_{\sigma,\mathrm{ov}}^{z\rightarrow\boldsymbol{\mathcal F}_z}`.
/// The same-spin two-body contribution uses the three Laplace classes:
/// `V^\sigma_0\det\mathbf D_{\sigma,\mathrm{ov}}`;
/// `-2\sum_z\det\mathbf D_{\sigma,\mathrm{ov}}^{z\rightarrow\boldsymbol{\mathcal V}_z}`,
/// and the second-minor contraction
/// `\sum_{z<y}\sum_{\eta<\xi}\phi_{\eta\xi}^{zy}`
/// `\mathcal J^\sigma_{\eta z,\xi y}\det\mathbf D_{\sigma,\mathrm{ov}}[\eta,\xi|z,y]`.
/// The mixed-spin contribution is
/// `V^{\alpha\beta}_0\det\mathbf D_{\alpha,\mathrm{ov}}\det\mathbf D_{\beta,\mathrm{ov}}`
/// minus the alpha and beta one-column replacements, plus the cofactor contraction
/// `\sum_{z,y}\sum_{\eta,\xi}\operatorname{cof}[\mathbf D_{\alpha,\mathrm{ov}}]_{\eta z}`
/// `\mathcal{II}_{\eta z,\xi y}`
/// `\operatorname{cof}[\mathbf D_{\beta,\mathrm{ov}}]_{\xi y}`.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Predecoded bra excitation ranks and orbital labels.
/// - `w_ex`: Predecoded ket excitation ranks and orbital labels.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(never)]
fn xw_hamiltonian_overlap_m0_prepared_const<
    T: NOCIScalar,
    const RXA: usize,
    const RWA: usize,
    const LA: usize,
    const DA: usize,
    const MA: usize,
    const MDA: usize,
    const RXB: usize,
    const RWB: usize,
    const LB: usize,
    const DB: usize,
    const MB: usize,
    const MDB: usize,
>(
    w: &WicksPairView<'_, T>,
    x_ex: &ExcitationCache,
    w_ex: &ExcitationCache,
    excitation_phase: f64,
    enuc: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_m0_prepared_const,
        {
            let zero = T::from_real(0.0);
            let half = T::from_real(0.5);
            let mut rows_a = [0usize; LA];
            let mut cols_a = [0usize; LA];
            let mut d_a = [zero; DA];
            let mut cof_a = [zero; DA];
            let n_a = w.aa.n();
            let nocc_a = w.aa.nocc;
            let nvirt_a = w.aa.nmo - nocc_a;
            construct_hamiltonian_indices::<RXA, RWA, LA>(
                &x_ex.alpha,
                &w_ex.alpha,
                nocc_a,
                nvirt_a,
                &mut rows_a,
                &mut cols_a,
            );

            let x0_a = w.aa.x_slice(0);
            let y0_a = w.aa.y_slice(0);
            for i in 0..LA {
                let row = rows_a[i] * n_a;
                for j in 0..LA {
                    d_a[i * LA + j] = if i >= j {
                        x0_a[row + cols_a[j]]
                    } else {
                        y0_a[row + cols_a[j]]
                    };
                }
            }

            let det_a = adjugate_transpose_const::<T, LA, DA>(&mut cof_a, &d_a);
            let mut j_a = zero;
            let jsl_a = w.aa.j_slice(0);
            let n2_a = n_a * n_a;
            let n3_a = n2_a * n_a;
            for eta in 0..LA {
                for xi in (eta + 1)..LA {
                    for z in 0..LA {
                        for y in (z + 1)..LA {
                            let mut minor = [zero; MDA];
                            second_minor_const::<T, LA>(&mut minor, &d_a, eta, xi, z, y);
                            let second = det_const::<T, MA, MDA>(&minor);
                            let direct_base =
                                rows_a[eta] * n3_a + cols_a[z] * n2_a + rows_a[xi] * n_a;
                            let exchange_base =
                                rows_a[eta] * n3_a + cols_a[y] * n2_a + rows_a[xi] * n_a;
                            let term = second
                                * (jsl_a[direct_base + cols_a[y]]
                                    - jsl_a[exchange_base + cols_a[z]]);
                            if ((eta + xi + z + y) & 1) == 0 {
                                j_a += term;
                            } else {
                                j_a -= term;
                            }
                        }
                    }
                }
            }

            let mut replacement_a = zero;
            let hcol0_a = w.aa.hcol0_t_slice();
            for z in 0..LA {
                let base = cols_a[z] * n_a;
                for eta in 0..LA {
                    replacement_a += cof_a[eta * LA + z] * hcol0_a[base + rows_a[eta]];
                }
            }

            let mut rows_b = [0usize; LB];
            let mut cols_b = [0usize; LB];
            let mut d_b = [zero; DB];
            let mut cof_b = [zero; DB];
            let n_b = w.bb.n();
            let nocc_b = w.bb.nocc;
            let nvirt_b = w.bb.nmo - nocc_b;
            construct_hamiltonian_indices::<RXB, RWB, LB>(
                &x_ex.beta,
                &w_ex.beta,
                nocc_b,
                nvirt_b,
                &mut rows_b,
                &mut cols_b,
            );

            let x0_b = w.bb.x_slice(0);
            let y0_b = w.bb.y_slice(0);
            for i in 0..LB {
                let row = rows_b[i] * n_b;
                for j in 0..LB {
                    d_b[i * LB + j] = if i >= j {
                        x0_b[row + cols_b[j]]
                    } else {
                        y0_b[row + cols_b[j]]
                    };
                }
            }

            let det_b = adjugate_transpose_const::<T, LB, DB>(&mut cof_b, &d_b);
            let mut j_b = zero;
            let jsl_b = w.bb.j_slice(0);
            let n2_b = n_b * n_b;
            let n3_b = n2_b * n_b;
            for eta in 0..LB {
                for xi in (eta + 1)..LB {
                    for z in 0..LB {
                        for y in (z + 1)..LB {
                            let mut minor = [zero; MDB];
                            second_minor_const::<T, LB>(&mut minor, &d_b, eta, xi, z, y);
                            let second = det_const::<T, MB, MDB>(&minor);
                            let direct_base =
                                rows_b[eta] * n3_b + cols_b[z] * n2_b + rows_b[xi] * n_b;
                            let exchange_base =
                                rows_b[eta] * n3_b + cols_b[y] * n2_b + rows_b[xi] * n_b;
                            let term = second
                                * (jsl_b[direct_base + cols_b[y]]
                                    - jsl_b[exchange_base + cols_b[z]]);
                            if ((eta + xi + z + y) & 1) == 0 {
                                j_b += term;
                            } else {
                                j_b -= term;
                            }
                        }
                    }
                }
            }

            let mut replacement_b = zero;
            let hcol0_b = w.bb.hcol0_t_slice();
            for z in 0..LB {
                let base = cols_b[z] * n_b;
                for eta in 0..LB {
                    replacement_b += cof_b[eta * LB + z] * hcol0_b[base + rows_b[eta]];
                }
            }

            let mut ii_term = zero;
            let iisl = w.ab.iiab_slice(0, 0, 0, 0);
            let n = w.ab.n();
            let n2 = n * n;
            let n3 = n2 * n;
            if LA <= LB {
                for z in 0..LA {
                    for eta in 0..LA {
                        let base_a = rows_a[eta] * n3 + cols_a[z] * n2;
                        let mut inner = zero;
                        for y in 0..LB {
                            for xi in 0..LB {
                                inner +=
                                    cof_b[xi * LB + y] * iisl[base_a + rows_b[xi] * n + cols_b[y]];
                            }
                        }
                        ii_term += cof_a[eta * LA + z] * inner;
                    }
                }
            } else {
                for y in 0..LB {
                    for xi in 0..LB {
                        let suffix_b = rows_b[xi] * n + cols_b[y];
                        let mut inner = zero;
                        for z in 0..LA {
                            for eta in 0..LA {
                                let base_a = rows_a[eta] * n3 + cols_a[z] * n2;
                                inner += cof_a[eta * LA + z] * iisl[base_a + suffix_b];
                            }
                        }
                        ii_term += cof_b[xi * LB + y] * inner;
                    }
                }
            }

            let det_ab = det_a * det_b;
            let g0 = T::from_real(enuc)
                + w.aa.f0h[0]
                + half * w.aa.v0[0]
                + w.bb.f0h[0]
                + half * w.bb.v0[0]
                + w.ab.vab0[0][0];
            let mut core = g0 * det_ab;
            core -= det_b * replacement_a;
            core -= det_a * replacement_b;
            core += j_a * det_b;
            core += j_b * det_a;
            core += ii_term;
            let pref = T::from_real(excitation_phase)
                * w.aa.phase
                * T::from_real(w.aa.tilde_s_prod)
                * w.bb.phase
                * T::from_real(w.bb.tilde_s_prod);
            (pref * core, pref * det_ab)
        }
    )
}

/// Evaluate packed fixed-rank Hamiltonian and overlap matrix elements.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Cached bra excitations in lane order.
/// - `w_ex`: Cached ket excitations in lane order.
/// - `excitation_phase`: Excitation phases in lane order.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian outputs.
/// - `s`: Overlap outputs.
/// # Returns
/// - `()`: Writes `LANES` Hamiltonian and overlap values.
/// # Safety
/// - Cached labels must match fixed ranks; caller must establish `V` CPU support.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xw_hamiltonian_overlap_m0_prepared_simd_const<
    T: NOCIScalar,
    V: Simd<LANES, Scalar = T>,
    const LANES: usize,
    const RXA: usize,
    const RWA: usize,
    const LA: usize,
    const DA: usize,
    const MA: usize,
    const MDA: usize,
    const RXB: usize,
    const RWB: usize,
    const LB: usize,
    const DB: usize,
    const MB: usize,
    const MDB: usize,
>(
    w: &WicksPairView<'_, T>,
    x_ex: &[ExcitationCache; LANES],
    w_ex: &[ExcitationCache; LANES],
    excitation_phase: &[f64; LANES],
    enuc: f64,
    h: &mut [T; LANES],
    s: &mut [T; LANES],
) {
    let zero = V::zero();
    let mut rows_a = [[0usize; LA]; LANES];
    let mut cols_a = [[0usize; LA]; LANES];
    let nocc_a = w.aa.nocc;
    let nvirt_a = w.aa.nmo - nocc_a;
    for lane in 0..LANES {
        let x_cache = unsafe { &x_ex.get_unchecked(lane).alpha };
        let w_cache = unsafe { &w_ex.get_unchecked(lane).alpha };
        for i in 0..RXA {
            rows_a[lane][i] = usize::from(unsafe { *x_cache.particles.get_unchecked(i) }) - nocc_a;
            cols_a[lane][i] = usize::from(unsafe { *x_cache.holes.get_unchecked(i) });
        }
        for i in RXA..LA {
            let k = i - RXA;
            rows_a[lane][i] = nvirt_a + usize::from(unsafe { *w_cache.holes.get_unchecked(k) });
            cols_a[lane][i] = usize::from(unsafe { *w_cache.particles.get_unchecked(k) });
        }
    }

    let n_a = w.aa.n();
    let x0_a = w.aa.x_slice(0);
    let y0_a = w.aa.y_slice(0);
    let mut d_a = [zero; DA];
    let mut cof_a = [zero; DA];
    for i in 0..LA {
        for j in 0..LA {
            let matrix = if i >= j { x0_a } else { y0_a };
            let mut values = [T::from_real(0.0); LANES];
            for lane in 0..LANES {
                let index = rows_a[lane][i] * n_a + cols_a[lane][j];
                values[lane] = unsafe { *matrix.get_unchecked(index) };
            }
            d_a[i * LA + j] = V::load(&values);
        }
    }
    let det_a = adjugate_transpose_simd_const::<V, LANES, LA, DA>(&mut cof_a, &d_a);

    let mut j_a = zero;
    let jsl_a = w.aa.j_slice(0);
    let n2_a = n_a * n_a;
    let n3_a = n2_a * n_a;
    for eta in 0..LA {
        for xi in (eta + 1)..LA {
            for z in 0..LA {
                for y in (z + 1)..LA {
                    let mut minor = [zero; MDA];
                    second_minor_const::<V, LA>(&mut minor, &d_a, eta, xi, z, y);
                    let second = det_simd_const::<V, LANES, MA, MDA>(&minor);
                    let mut direct = [T::from_real(0.0); LANES];
                    let mut exchange = [T::from_real(0.0); LANES];
                    for lane in 0..LANES {
                        let direct_index = rows_a[lane][eta] * n3_a
                            + cols_a[lane][z] * n2_a
                            + rows_a[lane][xi] * n_a
                            + cols_a[lane][y];
                        let exchange_index = rows_a[lane][eta] * n3_a
                            + cols_a[lane][y] * n2_a
                            + rows_a[lane][xi] * n_a
                            + cols_a[lane][z];
                        direct[lane] = unsafe { *jsl_a.get_unchecked(direct_index) };
                        exchange[lane] = unsafe { *jsl_a.get_unchecked(exchange_index) };
                    }
                    let difference = V::sub(V::load(&direct), V::load(&exchange));
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_a = V::madd(j_a, second, difference);
                    } else {
                        j_a = V::msub(j_a, second, difference);
                    }
                }
            }
        }
    }

    let hcol0_a = w.aa.hcol0_t_slice();
    let mut replacement_a = zero;
    for z in 0..LA {
        for eta in 0..LA {
            let mut values = [T::from_real(0.0); LANES];
            for lane in 0..LANES {
                let index = cols_a[lane][z] * n_a + rows_a[lane][eta];
                values[lane] = unsafe { *hcol0_a.get_unchecked(index) };
            }
            replacement_a = V::madd(replacement_a, cof_a[eta * LA + z], V::load(&values));
        }
    }

    let mut rows_b = [[0usize; LB]; LANES];
    let mut cols_b = [[0usize; LB]; LANES];
    let nocc_b = w.bb.nocc;
    let nvirt_b = w.bb.nmo - nocc_b;
    for lane in 0..LANES {
        let x_cache = unsafe { &x_ex.get_unchecked(lane).beta };
        let w_cache = unsafe { &w_ex.get_unchecked(lane).beta };
        for i in 0..RXB {
            rows_b[lane][i] = usize::from(unsafe { *x_cache.particles.get_unchecked(i) }) - nocc_b;
            cols_b[lane][i] = usize::from(unsafe { *x_cache.holes.get_unchecked(i) });
        }
        for i in RXB..LB {
            let k = i - RXB;
            rows_b[lane][i] = nvirt_b + usize::from(unsafe { *w_cache.holes.get_unchecked(k) });
            cols_b[lane][i] = usize::from(unsafe { *w_cache.particles.get_unchecked(k) });
        }
    }

    let n_b = w.bb.n();
    let x0_b = w.bb.x_slice(0);
    let y0_b = w.bb.y_slice(0);
    let mut d_b = [zero; DB];
    let mut cof_b = [zero; DB];
    for i in 0..LB {
        for j in 0..LB {
            let matrix = if i >= j { x0_b } else { y0_b };
            let mut values = [T::from_real(0.0); LANES];
            for lane in 0..LANES {
                let index = rows_b[lane][i] * n_b + cols_b[lane][j];
                values[lane] = unsafe { *matrix.get_unchecked(index) };
            }
            d_b[i * LB + j] = V::load(&values);
        }
    }
    let det_b = adjugate_transpose_simd_const::<V, LANES, LB, DB>(&mut cof_b, &d_b);

    let mut j_b = zero;
    let jsl_b = w.bb.j_slice(0);
    let n2_b = n_b * n_b;
    let n3_b = n2_b * n_b;
    for eta in 0..LB {
        for xi in (eta + 1)..LB {
            for z in 0..LB {
                for y in (z + 1)..LB {
                    let mut minor = [zero; MDB];
                    second_minor_const::<V, LB>(&mut minor, &d_b, eta, xi, z, y);
                    let second = det_simd_const::<V, LANES, MB, MDB>(&minor);
                    let mut direct = [T::from_real(0.0); LANES];
                    let mut exchange = [T::from_real(0.0); LANES];
                    for lane in 0..LANES {
                        let direct_index = rows_b[lane][eta] * n3_b
                            + cols_b[lane][z] * n2_b
                            + rows_b[lane][xi] * n_b
                            + cols_b[lane][y];
                        let exchange_index = rows_b[lane][eta] * n3_b
                            + cols_b[lane][y] * n2_b
                            + rows_b[lane][xi] * n_b
                            + cols_b[lane][z];
                        direct[lane] = unsafe { *jsl_b.get_unchecked(direct_index) };
                        exchange[lane] = unsafe { *jsl_b.get_unchecked(exchange_index) };
                    }
                    let difference = V::sub(V::load(&direct), V::load(&exchange));
                    if ((eta + xi + z + y) & 1) == 0 {
                        j_b = V::madd(j_b, second, difference);
                    } else {
                        j_b = V::msub(j_b, second, difference);
                    }
                }
            }
        }
    }

    let hcol0_b = w.bb.hcol0_t_slice();
    let mut replacement_b = zero;
    for z in 0..LB {
        for eta in 0..LB {
            let mut values = [T::from_real(0.0); LANES];
            for lane in 0..LANES {
                let index = cols_b[lane][z] * n_b + rows_b[lane][eta];
                values[lane] = unsafe { *hcol0_b.get_unchecked(index) };
            }
            replacement_b = V::madd(replacement_b, cof_b[eta * LB + z], V::load(&values));
        }
    }

    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
    let n = w.ab.n();
    let n2 = n * n;
    let n3 = n2 * n;
    let mut ii_term = zero;
    if LA <= LB {
        for z in 0..LA {
            for eta in 0..LA {
                let mut inner = zero;
                for y in 0..LB {
                    for xi in 0..LB {
                        let mut values = [T::from_real(0.0); LANES];
                        for lane in 0..LANES {
                            let index = rows_a[lane][eta] * n3
                                + cols_a[lane][z] * n2
                                + rows_b[lane][xi] * n
                                + cols_b[lane][y];
                            values[lane] = unsafe { *iisl.get_unchecked(index) };
                        }
                        inner = V::madd(inner, cof_b[xi * LB + y], V::load(&values));
                    }
                }
                ii_term = V::madd(ii_term, cof_a[eta * LA + z], inner);
            }
        }
    } else {
        for y in 0..LB {
            for xi in 0..LB {
                let mut inner = zero;
                for z in 0..LA {
                    for eta in 0..LA {
                        let mut values = [T::from_real(0.0); LANES];
                        for lane in 0..LANES {
                            let index = rows_a[lane][eta] * n3
                                + cols_a[lane][z] * n2
                                + rows_b[lane][xi] * n
                                + cols_b[lane][y];
                            values[lane] = unsafe { *iisl.get_unchecked(index) };
                        }
                        inner = V::madd(inner, cof_a[eta * LA + z], V::load(&values));
                    }
                }
                ii_term = V::madd(ii_term, cof_b[xi * LB + y], inner);
            }
        }
    }

    let det_ab = V::mul(det_a, det_b);
    let half = T::from_real(0.5);
    let g0 = T::from_real(enuc)
        + w.aa.f0h[0]
        + half * w.aa.v0[0]
        + w.bb.f0h[0]
        + half * w.bb.v0[0]
        + w.ab.vab0[0][0];
    let mut core = V::mul(V::splat(g0), det_ab);
    core = V::msub(core, det_b, replacement_a);
    core = V::msub(core, det_a, replacement_b);
    core = V::madd(core, j_a, det_b);
    core = V::madd(core, j_b, det_a);
    core = V::add(core, ii_term);
    let reference_pref =
        w.aa.phase * T::from_real(w.aa.tilde_s_prod) * w.bb.phase * T::from_real(w.bb.tilde_s_prod);
    let mut pref = [T::from_real(0.0); LANES];
    for lane in 0..LANES {
        pref[lane] = T::from_real(excitation_phase[lane]) * reference_pref;
    }
    let pref = V::load(&pref);
    V::store(V::mul(pref, core), h);
    V::store(V::mul(pref, det_ab), s);
}

/// Dispatch four real Hamiltonian/overlap values to one fixed-rank AVX2/FMA kernel.
/// # Arguments:
/// - `w`: Real Wick intermediates.
/// - `ranks`: Reference-resolved excitation ranks.
/// - `ex`: Bra and ket excitation caches.
/// - `excitation_phase`: Excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `out`: Hamiltonian and overlap outputs.
/// # Returns
/// - `()`: Writes four matrix-element pairs.
/// # Safety
/// - The current CPU must support AVX2 and FMA; cached ranks must match `ranks`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn xw_hamiltonian_overlap_m0_prepared_f64x4(
    w: &WicksPairView<'_, f64>,
    ranks: (usize, usize, usize, usize),
    ex: (&[ExcitationCache; 4], &[ExcitationCache; 4]),
    excitation_phase: &[f64; 4],
    enuc: f64,
    out: (&mut [f64; 4], &mut [f64; 4]),
) {
    let (x_ex, w_ex) = ex;
    let (h, s) = out;
    dispatch_hamiltonian_ranks!(
        ranks,
        |RXA, RWA, LA, DA, MA, MDA, RXB, RWB, LB, DB, MB, MDB| unsafe {
            xw_hamiltonian_overlap_m0_prepared_f64x4_const::<
                RXA,
                RWA,
                LA,
                DA,
                MA,
                MDA,
                RXB,
                RWB,
                LB,
                DB,
                MB,
                MDB,
            >(w, x_ex, w_ex, excitation_phase, enuc, h, s)
        },
        (),
    )
}

/// Dispatch four complex Hamiltonian/overlap values to one fixed-rank AVX2/FMA kernel.
/// # Arguments:
/// - `w`: Complex Wick intermediates.
/// - `ranks`: Reference-resolved excitation ranks.
/// - `ex`: Bra and ket excitation caches.
/// - `excitation_phase`: Excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `out`: Hamiltonian and overlap outputs.
/// # Returns
/// - `()`: Writes four matrix-element pairs.
/// # Safety
/// - The current CPU must support AVX2 and FMA; cached ranks must match `ranks`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn xw_hamiltonian_overlap_m0_prepared_c64x4(
    w: &WicksPairView<'_, Complex64>,
    ranks: (usize, usize, usize, usize),
    ex: (&[ExcitationCache; 4], &[ExcitationCache; 4]),
    excitation_phase: &[f64; 4],
    enuc: f64,
    out: (&mut [Complex64; 4], &mut [Complex64; 4]),
) {
    let (x_ex, w_ex) = ex;
    let (h, s) = out;
    dispatch_hamiltonian_ranks!(
        ranks,
        |RXA, RWA, LA, DA, MA, MDA, RXB, RWB, LB, DB, MB, MDB| unsafe {
            xw_hamiltonian_overlap_m0_prepared_c64x4_const::<
                RXA,
                RWA,
                LA,
                DA,
                MA,
                MDA,
                RXB,
                RWB,
                LB,
                DB,
                MB,
                MDB,
            >(w, x_ex, w_ex, excitation_phase, enuc, h, s)
        },
        (),
    )
}

/// Dispatch eight real Hamiltonian/overlap values to one fixed-rank AVX-512F kernel.
/// # Arguments:
/// - `w`: Real Wick intermediates.
/// - `ranks`: Reference-resolved excitation ranks.
/// - `ex`: Bra and ket excitation caches.
/// - `excitation_phase`: Excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `out`: Hamiltonian and overlap outputs.
/// # Returns
/// - `()`: Writes eight matrix-element pairs.
/// # Safety
/// - The current CPU must support AVX-512F; cached ranks must match `ranks`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn xw_hamiltonian_overlap_m0_prepared_f64x8(
    w: &WicksPairView<'_, f64>,
    ranks: (usize, usize, usize, usize),
    ex: (&[ExcitationCache; 8], &[ExcitationCache; 8]),
    excitation_phase: &[f64; 8],
    enuc: f64,
    out: (&mut [f64; 8], &mut [f64; 8]),
) {
    let (x_ex, w_ex) = ex;
    let (h, s) = out;
    dispatch_hamiltonian_ranks!(
        ranks,
        |RXA, RWA, LA, DA, MA, MDA, RXB, RWB, LB, DB, MB, MDB| unsafe {
            xw_hamiltonian_overlap_m0_prepared_f64x8_const::<
                RXA,
                RWA,
                LA,
                DA,
                MA,
                MDA,
                RXB,
                RWB,
                LB,
                DB,
                MB,
                MDB,
            >(w, x_ex, w_ex, excitation_phase, enuc, h, s)
        },
        (),
    )
}

/// Dispatch eight complex Hamiltonian/overlap values to one fixed-rank AVX-512F kernel.
/// # Arguments:
/// - `w`: Complex Wick intermediates.
/// - `ranks`: Reference-resolved excitation ranks.
/// - `ex`: Bra and ket excitation caches.
/// - `excitation_phase`: Excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `out`: Hamiltonian and overlap outputs.
/// # Returns
/// - `()`: Writes eight matrix-element pairs.
/// # Safety
/// - The current CPU must support AVX-512F; cached ranks must match `ranks`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn xw_hamiltonian_overlap_m0_prepared_c64x8(
    w: &WicksPairView<'_, Complex64>,
    ranks: (usize, usize, usize, usize),
    ex: (&[ExcitationCache; 8], &[ExcitationCache; 8]),
    excitation_phase: &[f64; 8],
    enuc: f64,
    out: (&mut [Complex64; 8], &mut [Complex64; 8]),
) {
    let (x_ex, w_ex) = ex;
    let (h, s) = out;
    dispatch_hamiltonian_ranks!(
        ranks,
        |RXA, RWA, LA, DA, MA, MDA, RXB, RWB, LB, DB, MB, MDB| unsafe {
            xw_hamiltonian_overlap_m0_prepared_c64x8_const::<
                RXA,
                RWA,
                LA,
                DA,
                MA,
                MDA,
                RXB,
                RWB,
                LB,
                DB,
                MB,
                MDB,
            >(w, x_ex, w_ex, excitation_phase, enuc, h, s)
        },
        (),
    )
}

/// Evaluate four real fixed-rank Hamiltonian/overlap values with AVX2/FMA.
/// # Arguments:
/// - `w`: Real Wick intermediates.
/// - `x_ex`: Bra excitation caches.
/// - `w_ex`: Ket excitation caches.
/// - `excitation_phase`: Excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian outputs.
/// - `s`: Overlap outputs.
/// # Returns
/// - `()`: Writes four matrix-element pairs.
/// # Safety
/// - The current CPU must support AVX2 and FMA; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_prepared_f64x4_const<
    const RXA: usize,
    const RWA: usize,
    const LA: usize,
    const DA: usize,
    const MA: usize,
    const MDA: usize,
    const RXB: usize,
    const RWB: usize,
    const LB: usize,
    const DB: usize,
    const MB: usize,
    const MDB: usize,
>(
    w: &WicksPairView<'_, f64>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [f64; 4],
    s: &mut [f64; 4],
) {
    unsafe {
        xw_hamiltonian_overlap_m0_prepared_simd_const::<
            f64,
            F64x4,
            4,
            RXA,
            RWA,
            LA,
            DA,
            MA,
            MDA,
            RXB,
            RWB,
            LB,
            DB,
            MB,
            MDB,
        >(w, x_ex, w_ex, excitation_phase, enuc, h, s);
    }
}

/// Evaluate four complex fixed-rank Hamiltonian/overlap values with AVX2/FMA.
/// # Arguments:
/// - `w`: Complex Wick intermediates.
/// - `x_ex`: Bra excitation caches.
/// - `w_ex`: Ket excitation caches.
/// - `excitation_phase`: Excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian outputs.
/// - `s`: Overlap outputs.
/// # Returns
/// - `()`: Writes four matrix-element pairs.
/// # Safety
/// - The current CPU must support AVX2 and FMA; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_hamiltonian_overlap_m0_prepared_c64x4_const<
    const RXA: usize,
    const RWA: usize,
    const LA: usize,
    const DA: usize,
    const MA: usize,
    const MDA: usize,
    const RXB: usize,
    const RWB: usize,
    const LB: usize,
    const DB: usize,
    const MB: usize,
    const MDB: usize,
>(
    w: &WicksPairView<'_, Complex64>,
    x_ex: &[ExcitationCache; 4],
    w_ex: &[ExcitationCache; 4],
    excitation_phase: &[f64; 4],
    enuc: f64,
    h: &mut [Complex64; 4],
    s: &mut [Complex64; 4],
) {
    unsafe {
        xw_hamiltonian_overlap_m0_prepared_simd_const::<
            Complex64,
            C64x4,
            4,
            RXA,
            RWA,
            LA,
            DA,
            MA,
            MDA,
            RXB,
            RWB,
            LB,
            DB,
            MB,
            MDB,
        >(w, x_ex, w_ex, excitation_phase, enuc, h, s);
    }
}

/// Evaluate eight real fixed-rank Hamiltonian/overlap values with AVX-512F.
/// # Arguments:
/// - `w`: Real Wick intermediates.
/// - `x_ex`: Bra excitation caches.
/// - `w_ex`: Ket excitation caches.
/// - `excitation_phase`: Excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian outputs.
/// - `s`: Overlap outputs.
/// # Returns
/// - `()`: Writes eight matrix-element pairs.
/// # Safety
/// - The current CPU must support AVX-512F; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_prepared_f64x8_const<
    const RXA: usize,
    const RWA: usize,
    const LA: usize,
    const DA: usize,
    const MA: usize,
    const MDA: usize,
    const RXB: usize,
    const RWB: usize,
    const LB: usize,
    const DB: usize,
    const MB: usize,
    const MDB: usize,
>(
    w: &WicksPairView<'_, f64>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [f64; 8],
    s: &mut [f64; 8],
) {
    unsafe {
        xw_hamiltonian_overlap_m0_prepared_simd_const::<
            f64,
            F64x8,
            8,
            RXA,
            RWA,
            LA,
            DA,
            MA,
            MDA,
            RXB,
            RWB,
            LB,
            DB,
            MB,
            MDB,
        >(w, x_ex, w_ex, excitation_phase, enuc, h, s);
    }
}

/// Evaluate eight complex fixed-rank Hamiltonian/overlap values with AVX-512F.
/// # Arguments:
/// - `w`: Complex Wick intermediates.
/// - `x_ex`: Bra excitation caches.
/// - `w_ex`: Ket excitation caches.
/// - `excitation_phase`: Excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `h`: Hamiltonian outputs.
/// - `s`: Overlap outputs.
/// # Returns
/// - `()`: Writes eight matrix-element pairs.
/// # Safety
/// - The current CPU must support AVX-512F; cached labels must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_hamiltonian_overlap_m0_prepared_c64x8_const<
    const RXA: usize,
    const RWA: usize,
    const LA: usize,
    const DA: usize,
    const MA: usize,
    const MDA: usize,
    const RXB: usize,
    const RWB: usize,
    const LB: usize,
    const DB: usize,
    const MB: usize,
    const MDB: usize,
>(
    w: &WicksPairView<'_, Complex64>,
    x_ex: &[ExcitationCache; 8],
    w_ex: &[ExcitationCache; 8],
    excitation_phase: &[f64; 8],
    enuc: f64,
    h: &mut [Complex64; 8],
    s: &mut [Complex64; 8],
) {
    unsafe {
        xw_hamiltonian_overlap_m0_prepared_simd_const::<
            Complex64,
            C64x8,
            8,
            RXA,
            RWA,
            LA,
            DA,
            MA,
            MDA,
            RXB,
            RWB,
            LB,
            DB,
            MB,
            MDB,
        >(w, x_ex, w_ex, excitation_phase, enuc, h, s);
    }
}

/// Evaluate an arbitrary contraction-rank Hamiltonian and overlap for
/// `m_\alpha = m_\beta = 0`.
/// This is the generic fused fallback when the predecoded four-label cache is insufficient or
/// `L_\alpha + L_\beta > 6`. Each spin contraction determinant and adjugate is evaluated once
/// and its cofactors are reused by all operator contributions.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Full bra excitation.
/// - `w_ex`: Full ket excitation.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy.
/// - `scratch`: Reusable spin-resolved Wick workspace.
/// - `tol`: Numerical tolerance used when evaluating determinant adjugates.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_m0_gen_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &Excitation,
    w_ex: &Excitation,
    excitation_phase: f64,
    enuc: f64,
    scratch: &mut WickScratchSpin<T>,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_m0_gen_prepared,
        {
            // Generic `m = 0` path: build spin-sector `\mathbf D_{\mathrm{ov}}` determinants
            // and cofactors once,
            // then reuse them for scalar, one-column, same-spin J and mixed-spin II terms.
            let la =
                x_ex.alpha.holes.count_ones() as usize + w_ex.alpha.holes.count_ones() as usize;
            let lb = x_ex.beta.holes.count_ones() as usize + w_ex.beta.holes.count_ones() as usize;
            let zero = <T as From<f64>>::from(0.0);
            let half = <T as From<f64>>::from(0.5);

            prepare_same(&w.aa, &x_ex.alpha, &w_ex.alpha, &mut scratch.aa);
            prepare_same(&w.bb, &x_ex.beta, &w_ex.beta, &mut scratch.bb);

            let (det_a, have_a) = if la == 0 {
                (<T as From<f64>>::from(1.0), true)
            } else if let Some(value) = adjugate_transpose_dynamic(
                scratch.aa.adjt_det.as_mut_slice(),
                scratch.aa.invs.as_mut_slice(),
                scratch.aa.lu.as_mut_slice(),
                scratch.aa.det0.as_slice(),
                la,
                tol,
            ) {
                (value, true)
            } else {
                (
                    det_dynamic(scratch.aa.det0.as_slice(), la).unwrap_or(zero),
                    false,
                )
            };

            let (det_b, have_b) = if lb == 0 {
                (<T as From<f64>>::from(1.0), true)
            } else if let Some(value) = adjugate_transpose_dynamic(
                scratch.bb.adjt_det.as_mut_slice(),
                scratch.bb.invs.as_mut_slice(),
                scratch.bb.lu.as_mut_slice(),
                scratch.bb.det0.as_slice(),
                lb,
                tol,
            ) {
                (value, true)
            } else {
                (
                    det_dynamic(scratch.bb.det0.as_slice(), lb).unwrap_or(zero),
                    false,
                )
            };
            let mut same_a = zero;
            let mut same_b = zero;
            let mut h2ab = zero;

            if have_a {
                same_a = (w.aa.f0h[0] + half * w.aa.v0[0]) * det_a;
                if la > 0 {
                    let cof = scratch.aa.adjt_det.as_slice();
                    let n = w.aa.n();
                    if have_b {
                        let hcol0 = w.aa.hcol0_t_slice();
                        for z in 0..la {
                            let base = scratch.aa.cols[z] * n;
                            for eta in 0..la {
                                same_a -= cof[eta * la + z] * hcol0[base + scratch.aa.rows[eta]];
                            }
                        }
                    } else {
                        let fh = w.aa.fh_t_slice(0, 0);
                        let vv = w.aa.v_t_slice(0, 0, 0);
                        for z in 0..la {
                            let base = scratch.aa.cols[z] * n;
                            for eta in 0..la {
                                same_a -= cof[eta * la + z]
                                    * (fh[base + scratch.aa.rows[eta]]
                                        + vv[base + scratch.aa.rows[eta]]);
                            }
                        }
                    }
                }
                if la >= 2 {
                    let d = scratch.aa.det0.as_slice();
                    let rows = scratch.aa.rows.as_slice();
                    let cols = scratch.aa.cols.as_slice();
                    let jsl = w.aa.j_slice(0);
                    let n = w.aa.n();
                    let mut minor = vec![zero; (la - 2) * (la - 2)];
                    for eta in 0..la {
                        for xi in (eta + 1)..la {
                            for z in 0..la {
                                for y in (z + 1)..la {
                                    second_minor_dynamic(&mut minor, d, la, eta, xi, z, y);
                                    let second = det_dynamic(&minor, la - 2).unwrap_or(zero);
                                    let n2 = n * n;
                                    let n3 = n2 * n;
                                    let row_eta_n3 = rows[eta] * n3;
                                    let row_xi_n = rows[xi] * n;
                                    let direct_base = row_eta_n3 + cols[z] * n2 + row_xi_n;
                                    let exchange_base = row_eta_n3 + cols[y] * n2 + row_xi_n;
                                    let direct = jsl[direct_base + cols[y]];
                                    let exchange = jsl[exchange_base + cols[z]];
                                    let term = second * (direct - exchange);
                                    if ((eta + xi + z + y) & 1) == 0 {
                                        same_a += term;
                                    } else {
                                        same_a -= term;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if have_b {
                same_b = (w.bb.f0h[0] + half * w.bb.v0[0]) * det_b;
                if lb > 0 {
                    let cof = scratch.bb.adjt_det.as_slice();
                    let n = w.bb.n();
                    if have_a {
                        let hcol0 = w.bb.hcol0_t_slice();
                        for z in 0..lb {
                            let base = scratch.bb.cols[z] * n;
                            for eta in 0..lb {
                                same_b -= cof[eta * lb + z] * hcol0[base + scratch.bb.rows[eta]];
                            }
                        }
                    } else {
                        let fh = w.bb.fh_t_slice(0, 0);
                        let vv = w.bb.v_t_slice(0, 0, 0);
                        for z in 0..lb {
                            let base = scratch.bb.cols[z] * n;
                            for eta in 0..lb {
                                same_b -= cof[eta * lb + z]
                                    * (fh[base + scratch.bb.rows[eta]]
                                        + vv[base + scratch.bb.rows[eta]]);
                            }
                        }
                    }
                }
                if lb >= 2 {
                    let d = scratch.bb.det0.as_slice();
                    let rows = scratch.bb.rows.as_slice();
                    let cols = scratch.bb.cols.as_slice();
                    let jsl = w.bb.j_slice(0);
                    let n = w.bb.n();
                    let mut minor = vec![zero; (lb - 2) * (lb - 2)];
                    for eta in 0..lb {
                        for xi in (eta + 1)..lb {
                            for z in 0..lb {
                                for y in (z + 1)..lb {
                                    second_minor_dynamic(&mut minor, d, lb, eta, xi, z, y);
                                    let second = det_dynamic(&minor, lb - 2).unwrap_or(zero);
                                    let n2 = n * n;
                                    let n3 = n2 * n;
                                    let row_eta_n3 = rows[eta] * n3;
                                    let row_xi_n = rows[xi] * n;
                                    let direct_base = row_eta_n3 + cols[z] * n2 + row_xi_n;
                                    let exchange_base = row_eta_n3 + cols[y] * n2 + row_xi_n;
                                    let direct = jsl[direct_base + cols[y]];
                                    let exchange = jsl[exchange_base + cols[z]];
                                    let term = second * (direct - exchange);
                                    if ((eta + xi + z + y) & 1) == 0 {
                                        same_b += term;
                                    } else {
                                        same_b -= term;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if have_a && have_b {
                h2ab = w.ab.vab0[0][0] * det_a * det_b;
                let n = w.ab.n();
                if la > 0 && lb > 0 {
                    let iisl = w.ab.iiab_slice(0, 0, 0, 0);
                    let cofa = scratch.aa.adjt_det.as_slice();
                    let cofb = scratch.bb.adjt_det.as_slice();
                    for z in 0..la {
                        for eta in 0..la {
                            let n2 = n * n;
                            let n3 = n2 * n;
                            let base_a = scratch.aa.rows[eta] * n3 + scratch.aa.cols[z] * n2;
                            let mut inner = zero;
                            for y in 0..lb {
                                for xi in 0..lb {
                                    inner += cofb[xi * lb + y]
                                        * iisl
                                            [base_a + scratch.bb.rows[xi] * n + scratch.bb.cols[y]];
                                }
                            }
                            h2ab += cofa[eta * la + z] * inner;
                        }
                    }
                }
            }

            let sa_pref = w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod);
            let sb_pref = w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod);
            let excitation = <T as From<f64>>::from(excitation_phase);
            let s = excitation * sa_pref * sb_pref * det_a * det_b;
            let mut h = <T as From<f64>>::from(enuc) * s;
            h += excitation * sa_pref * sb_pref * (same_a * det_b + same_b * det_a + h2ab);
            (h, s)
        }
    )
}

/// Evaluate the fused Hamiltonian and overlap for nonzero reference-pair nullity.
/// The alpha- and beta-spin same-spin distribution sums are traversed once each and reuse every
/// mixed contraction determinant and adjugate across overlap, one-electron and same-spin
/// two-electron terms. The different-spin term is then evaluated in its own natural distribution
/// space. This generic path supports all excitation ranks represented by `Excitation`.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `x_ex`: Full bra excitation.
/// - `w_ex`: Full ket excitation.
/// - `excitation_phase`: Product of the alpha- and beta-spin excitation phases.
/// - `enuc`: Nuclear repulsion energy; retained for the common interface.
/// - `scratch`: Reusable spin-resolved Wick workspace.
/// - `tol`: Numerical tolerance used by generic determinant and adjugate evaluation.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
fn xw_hamiltonian_overlap_gen_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    x_ex: &Excitation,
    w_ex: &Excitation,
    excitation_phase: f64,
    _enuc: f64,
    scratch: &mut WickScratchSpin<T>,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_gen_prepared,
        {
            let la =
                x_ex.alpha.holes.count_ones() as usize + w_ex.alpha.holes.count_ones() as usize;
            let lb = x_ex.beta.holes.count_ones() as usize + w_ex.beta.holes.count_ones() as usize;

            let zero = <T as From<f64>>::from(0.0);
            let one = <T as From<f64>>::from(1.0);
            let two = <T as From<f64>>::from(2.0);
            let half = <T as From<f64>>::from(0.5);

            // Construct the all-zero and all-one endpoint contraction determinants once per spin.
            prepare_same(&w.aa, &x_ex.alpha, &w_ex.alpha, &mut scratch.aa);
            prepare_same(&w.bb, &x_ex.beta, &w_ex.beta, &mut scratch.bb);

            let mut sa = zero;
            let mut h1a = zero;
            let mut h2aa = zero;
            let na = w.aa.n();

            // Embed overlap and one-body distributions in the two-operator same-spin distribution.
            // `m_2 = 0` identifies one-body terms and `m_1 = m_2 = 0` identifies overlap terms.
            mix_dets_same(&w.aa, la, 2, &mut scratch.aa, |bits, scratch| {
                let m1 = bit(bits, 0);
                let m2 = bit(bits, 1);

                let det_det = if la == 0 {
                    Some(one)
                } else {
                    adjugate_transpose_generic(
                        scratch.adjt_det.as_mut_slice(),
                        scratch.det_mix.as_slice(),
                        la,
                        tol,
                    )
                };

                if let Some(det_det) = det_det {
                    // Accumulate the overlap only for the embedded distribution with both operator bits
                    // zero.
                    if m1 == 0 && m2 == 0 {
                        sa += det_det;
                    }

                    // Accumulate the one-electron term when the second operator assignment is zero.
                    if m2 == 0 {
                        let mut contrib = det_det * w.aa.f0h[m1];
                        let f0 = w.aa.fh_t_slice(m1, 0);
                        let f1 = w.aa.fh_t_slice(m1, 1);

                        for k in 0..la {
                            let mk = bit(bits, k + 2);
                            let ck = scratch.cols[k];
                            let fsl = if mk == 0 { f0 } else { f1 };
                            let base = ck * na;

                            let corr = column_replacement_correction(
                                la,
                                scratch.det_mix.as_slice(),
                                scratch.adjt_det.as_slice(),
                                k,
                                |r| fsl[base + scratch.rows[r]],
                            );

                            contrib -= det_det + corr;
                        }

                        h1a += contrib;
                    }

                    // Start the same-spin two-electron term from its scalar intermediate.
                    let mut contrib = w.aa.v0[m1 + m2] * det_det;
                    let v0 = w.aa.v_t_slice(m1, m2, 0);
                    let v1 = w.aa.v_t_slice(m1, m2, 1);

                    // Reuse the same cofactor matrix for every one-column two-electron replacement.
                    for k in 0..la {
                        let mk = bit(bits, k + 2);
                        let ck = scratch.cols[k];
                        let vsl = if mk == 0 { v0 } else { v1 };
                        let base = ck * na;

                        let corr = column_replacement_correction(
                            la,
                            scratch.det_mix.as_slice(),
                            scratch.adjt_det.as_slice(),
                            k,
                            |r| vsl[base + scratch.rows[r]],
                        );

                        contrib -= two * (det_det + corr);
                    }

                    if la >= 2 {
                        let layout = ReplacementLayout {
                            n: na,
                            rows: scratch.rows.as_slice(),
                            cols: scratch.cols.as_slice(),
                        };

                        // Evaluate the two-column `\mathcal J` contribution from first minors of the
                        // same mixed determinant. The existing replacement helpers preserve the stored
                        // pair-exchange symmetry and assignment ordering of the original evaluator.
                        let mut i = 0usize;
                        while i < la {
                            let mut j = 0usize;
                            while j < la {
                                let phase = if ((i + j) & 1) == 0 { one } else { -one };
                                let ri = scratch.rows[i];
                                let cj = scratch.cols[j];
                                let mj = bit(bits, j + 2);

                                minor_adjt(
                                    scratch.det_mix.as_slice(),
                                    Minor {
                                        l: la,
                                        row: i,
                                        col: j,
                                    },
                                    &mut scratch.det_mix2,
                                    &mut scratch.adjt_det2,
                                    tol,
                                    |lm1, _det_minor, cof_minor, _det_det2| {
                                        for k2 in 0..lm1 {
                                            let k_full = if k2 < j { k2 } else { k2 + 1 };
                                            let mk = bit(bits, k_full + 2);
                                            let (slot, swap) = jslot(m1, m2, mk, mj);
                                            let jsl = w.aa.j_slice(slot);

                                            let det_repl =
                                                column_replacement_det(lm1, cof_minor, k2, |r| {
                                                    j_replacement(
                                                        jsl,
                                                        layout,
                                                        DetIndex { row: i, col: j },
                                                        DetIndex { row: r, col: k2 },
                                                        DetIndex { row: ri, col: cj },
                                                        swap,
                                                    )
                                                });

                                            contrib += phase * det_repl;
                                        }
                                    },
                                );
                                j += 1;
                            }
                            i += 1;
                        }
                    }

                    h2aa += contrib;
                } else if m1 == 0 && m2 == 0 {
                    // Preserve overlap evaluation when the adjugate path rejects a singular mixed
                    // determinant.
                    sa += det_dynamic(scratch.det_mix.as_slice(), la).unwrap_or(zero);
                }
            });

            let mut sb = zero;
            let mut h1b = zero;
            let mut h2bb = zero;
            let nb = w.bb.n();

            // Repeat the same fused distribution traversal for the beta-spin contraction determinant.
            mix_dets_same(&w.bb, lb, 2, &mut scratch.bb, |bits, scratch| {
                let m1 = bit(bits, 0);
                let m2 = bit(bits, 1);

                let det_det = if lb == 0 {
                    Some(one)
                } else {
                    adjugate_transpose_generic(
                        scratch.adjt_det.as_mut_slice(),
                        scratch.det_mix.as_slice(),
                        lb,
                        tol,
                    )
                };

                if let Some(det_det) = det_det {
                    if m1 == 0 && m2 == 0 {
                        sb += det_det;
                    }

                    if m2 == 0 {
                        let mut contrib = det_det * w.bb.f0h[m1];
                        let f0 = w.bb.fh_t_slice(m1, 0);
                        let f1 = w.bb.fh_t_slice(m1, 1);

                        for k in 0..lb {
                            let mk = bit(bits, k + 2);
                            let ck = scratch.cols[k];
                            let fsl = if mk == 0 { f0 } else { f1 };
                            let base = ck * nb;

                            let corr = column_replacement_correction(
                                lb,
                                scratch.det_mix.as_slice(),
                                scratch.adjt_det.as_slice(),
                                k,
                                |r| fsl[base + scratch.rows[r]],
                            );

                            contrib -= det_det + corr;
                        }

                        h1b += contrib;
                    }

                    let mut contrib = w.bb.v0[m1 + m2] * det_det;
                    let v0 = w.bb.v_t_slice(m1, m2, 0);
                    let v1 = w.bb.v_t_slice(m1, m2, 1);

                    for k in 0..lb {
                        let mk = bit(bits, k + 2);
                        let ck = scratch.cols[k];
                        let vsl = if mk == 0 { v0 } else { v1 };
                        let base = ck * nb;

                        let corr = column_replacement_correction(
                            lb,
                            scratch.det_mix.as_slice(),
                            scratch.adjt_det.as_slice(),
                            k,
                            |r| vsl[base + scratch.rows[r]],
                        );

                        contrib -= two * (det_det + corr);
                    }

                    if lb >= 2 {
                        let layout = ReplacementLayout {
                            n: nb,
                            rows: scratch.rows.as_slice(),
                            cols: scratch.cols.as_slice(),
                        };

                        let mut i = 0usize;
                        while i < lb {
                            let mut j = 0usize;
                            while j < lb {
                                let phase = if ((i + j) & 1) == 0 { one } else { -one };
                                let ri = scratch.rows[i];
                                let cj = scratch.cols[j];
                                let mj = bit(bits, j + 2);

                                minor_adjt(
                                    scratch.det_mix.as_slice(),
                                    Minor {
                                        l: lb,
                                        row: i,
                                        col: j,
                                    },
                                    &mut scratch.det_mix2,
                                    &mut scratch.adjt_det2,
                                    tol,
                                    |lm1, _det_minor, cof_minor, _det_det2| {
                                        for k2 in 0..lm1 {
                                            let k_full = if k2 < j { k2 } else { k2 + 1 };
                                            let mk = bit(bits, k_full + 2);
                                            let (slot, swap) = jslot(m1, m2, mk, mj);
                                            let jsl = w.bb.j_slice(slot);

                                            let det_repl =
                                                column_replacement_det(lm1, cof_minor, k2, |r| {
                                                    j_replacement(
                                                        jsl,
                                                        layout,
                                                        DetIndex { row: i, col: j },
                                                        DetIndex { row: r, col: k2 },
                                                        DetIndex { row: ri, col: cj },
                                                        swap,
                                                    )
                                                });

                                            contrib += phase * det_repl;
                                        }
                                    },
                                );
                                j += 1;
                            }
                            i += 1;
                        }
                    }

                    h2bb += contrib;
                } else if m1 == 0 && m2 == 0 {
                    sb += det_dynamic(scratch.det_mix.as_slice(), lb).unwrap_or(zero);
                }
            });

            let mut h2ab = zero;

            // The mixed-spin operator has one operator assignment per spin, so its distribution space is
            // different from the same-spin `L + 2` traversal above and is evaluated once in its natural
            // form.
            scratch.diff.ensure_diff(la, lb);

            let rows_a = scratch.aa.rows.as_slice();
            let cols_a = scratch.aa.cols.as_slice();
            let rows_b = scratch.bb.rows.as_slice();
            let cols_b = scratch.bb.cols.as_slice();
            let deta0 = scratch.aa.det0.as_slice();
            let deta1 = scratch.aa.det1.as_slice();
            let detb0 = scratch.bb.det0.as_slice();
            let detb1 = scratch.bb.det1.as_slice();

            let layout_b = ReplacementLayout {
                n: w.ab.n(),
                rows: rows_b,
                cols: cols_b,
            };

            get_det_adjt_diff(
                w,
                (la, lb),
                &mut scratch.diff,
                DetBranches {
                    zero: deta0,
                    one: deta1,
                },
                DetBranches {
                    zero: detb0,
                    one: detb1,
                },
                tol,
                |bits_a, bits_b, scratch, det_deta, det_detb| {
                    let ma0 = bit(bits_a, 0);
                    let mb0 = bit(bits_b, 0);
                    let mut contrib = w.ab.vab0[ma0][mb0] * det_deta * det_detb;
                    let n = w.ab.n();

                    let vab0 = w.ab.vab_t_slice(ma0, mb0, 0);
                    let vab1 = w.ab.vab_t_slice(ma0, mb0, 1);

                    for (k, &ck) in cols_a.iter().enumerate().take(la) {
                        let mak = bit(bits_a, k + 1);
                        let vsl = if mak == 0 { vab0 } else { vab1 };
                        let base = ck * n;

                        let det_repl =
                            column_replacement_det(la, scratch.adjt_deta.as_slice(), k, |r| {
                                vsl[base + rows_a[r]]
                            });

                        contrib -= det_repl * det_detb;
                    }

                    let vba0 = w.ab.vba_t_slice(mb0, ma0, 0);
                    let vba1 = w.ab.vba_t_slice(mb0, ma0, 1);

                    for (k, &ck) in cols_b.iter().enumerate().take(lb) {
                        let mbk = bit(bits_b, k + 1);
                        let vsl = if mbk == 0 { vba0 } else { vba1 };
                        let base = ck * n;

                        let det_repl =
                            column_replacement_det(lb, scratch.adjt_detb.as_slice(), k, |r| {
                                vsl[base + rows_b[r]]
                            });

                        contrib -= det_repl * det_deta;
                    }

                    // Contract the alpha cofactor with a beta-column replacement of `\mathcal{II}`.
                    // This reproduces the existing assignment ordering while avoiding a separate tensor
                    // pass.
                    let mut i = 0usize;
                    while i < la {
                        let ra = rows_a[i];

                        let mut j = 0usize;
                        while j < la {
                            let ca = cols_a[j];
                            let cofa = scratch.adjt_deta.as_slice()[i * la + j];
                            let ma1 = bit(bits_a, j + 1);

                            for k in 0..lb {
                                let mbk = bit(bits_b, k + 1);
                                let iisl = w.ab.iiab_slice(ma0, ma1, mb0, mbk);

                                let det_repl = column_replacement_det(
                                    lb,
                                    scratch.adjt_detb.as_slice(),
                                    k,
                                    |r| {
                                        ii_replacement(
                                            iisl,
                                            layout_b,
                                            DetIndex { row: r, col: k },
                                            DetIndex { row: ra, col: ca },
                                            true,
                                        )
                                    },
                                );

                                contrib += cofa * det_repl;
                            }
                            j += 1;
                        }
                        i += 1;
                    }

                    h2ab += contrib;
                },
            );

            let ref_pref = (w.aa.phase * <T as From<f64>>::from(w.aa.tilde_s_prod))
                * (w.bb.phase * <T as From<f64>>::from(w.bb.tilde_s_prod));
            let pref = <T as From<f64>>::from(excitation_phase) * ref_pref;

            let s = pref * sa * sb;
            let h = pref * (h1a * sb + h1b * sa + half * h2aa * sb + half * h2bb * sa + h2ab);

            (h, s)
        }
    )
}
