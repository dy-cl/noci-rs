// elements/nonorthogonalwicks/eval/hamiltonian.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::DeterminantData;
#[cfg(target_arch = "x86_64")]
use crate::elements::LaneKernel;
use crate::elements::{LaneBatch, group_lanes, xw_lanes};
use crate::maths::{
    Simd, adjugate_transpose_const, adjugate_transpose_dynamic, det_const, det_dynamic,
    second_minor_const, second_minor_dynamic,
};
use crate::time_call;
use crate::{Excitation, ExcitationCache};

// Parent/sibling imports.
use super::super::scratch::WickScratchSpin;
use super::super::view::WicksPairView;
use super::dispatch::{HAMRADIX, dispatch_hamiltonian_ranks, dispatch_hamiltonian_ranks_inner};
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
/// - `states`: Bra and ket determinants relative to their references.
/// - `enuc`: Nuclear repulsion energy.
/// - `scratch`: Reusable Wick workspace for generic-rank and nonzero-`m` evaluation.
/// - `tol`: Numerical tolerance used by generic determinant and adjugate evaluation.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
#[inline(always)]
pub(crate) fn xw_hamiltonian_overlap_prepared<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    states: (&DeterminantData, &DeterminantData),
    enuc: f64,
    scratch: &mut WickScratchSpin<T>,
    tol: f64,
) -> (T, T) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_prepared,
        {
            // A reference pair with near-zero orbital pairs is the sum over its combinations of
            // alpha- and beta-spin branches.
            if w.ncombinations() > 1 {
                let zero = <T as From<f64>>::from(0.0);
                return (0..w.ncombinations()).fold((zero, zero), |(h, s), k| {
                    let (hk, sk) = xw_hamiltonian_overlap_prepared(
                        &w.combination(k),
                        states,
                        enuc,
                        scratch,
                        tol,
                    );
                    (h + hk, s + sk)
                });
            }

            // The kernels read both spins' masks and caches of each determinant together.
            let (x, wdet) = states;
            let spins = |d: &DeterminantData| {
                (
                    Excitation {
                        alpha: d.alpha.excitation,
                        beta: d.beta.excitation,
                    },
                    ExcitationCache {
                        alpha: d.alpha.cache,
                        beta: d.beta.cache,
                    },
                )
            };
            let ((x_ex, x_cache), (w_ex, w_cache)) = (spins(x), spins(wdet));
            let (x_ex, w_ex, x_cache, w_cache) = (&x_ex, &w_ex, &x_cache, &w_cache);
            let excitation_phase = x.phase() * wdet.phase();

            // For `m_\alpha=m_\beta=0`, try fixed-rank determinant/cofactor formulas before the
            // generic evaluator. Any other zero-overlap distribution requires the general path.
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
                    // Fixed-rank path evaluates `(H,S)` from stored `X^{(m_i)}` and `Y^{(m_i)}`
                    // contractions.
                    return value;
                }

                // Unsupported excitation ranks fall back to the generic augmented determinant.
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

            // Nonzero `m_\alpha` or `m_\beta` requires the constrained sum over binary
            // contraction assignments, evaluated by the general prepared path.
            xw_hamiltonian_overlap_gen_prepared(w, x_ex, w_ex, excitation_phase, enuc, scratch, tol)
        }
    )
}

/// Evaluate batched Hamiltonian and overlap matrix elements for one ordered reference pair.
/// Every request supplied to this routine already belongs to that reference pair. A single
/// request has no other requests to fill SIMD lanes and is evaluated directly. Otherwise requests
/// are streamed through generated `(R_{x,\alpha},R_{w,\alpha},R_{x,\beta},R_{w,\beta})` bins when
/// `m_\alpha = m_\beta = 0`. The widest matching real or complex SIMD kernel is selected internally,
/// incomplete bins are padded with one valid request, and unsupported requests use the existing
/// prepared scalar evaluator.
/// # Arguments:
/// - `w`: Wick intermediates for one ordered nonorthogonal reference pair.
/// - `requests`: Output positions with their bra and ket determinant data.
/// - `enuc`: Nuclear repulsion energy.
/// - `scratch`: Reusable Wick workspace for scalar generic-rank evaluation.
/// - `tol`: Numerical tolerance used by generic determinant and adjugate evaluation.
/// - `out`: Hamiltonian and overlap results aligned with the original request order.
/// # Returns:
/// - `()`: Writes every matrix element in `requests` into `out`.
pub(crate) fn xw_hamiltonian_overlap_prepared_batched<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    requests: &[(usize, DeterminantData, DeterminantData)],
    enuc: f64,
    scratch: &mut WickScratchSpin<T>,
    tol: f64,
    out: &mut [(T, T)],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_hamiltonian_overlap_prepared_batched,
        {
            if let [(output, x, wdet)] = requests {
                out[*output] = xw_hamiltonian_overlap_prepared(w, (x, wdet), enuc, scratch, tol);
                return;
            }

            // A reference pair with near-zero orbital pairs is the sum over its combinations of
            // alpha- and beta-spin branches; accumulate the requested outputs of each one.
            if w.ncombinations() > 1 {
                let zero = <T as From<f64>>::from(0.0);
                for &(output, _, _) in requests {
                    out[output] = (zero, zero);
                }
                let mut part = vec![(zero, zero); out.len()];
                for k in 0..w.ncombinations() {
                    xw_hamiltonian_overlap_prepared_batched(
                        &w.combination(k),
                        requests,
                        enuc,
                        scratch,
                        tol,
                        &mut part,
                    );
                    for &(output, _, _) in requests {
                        out[output].0 += part[output].0;
                        out[output].1 += part[output].1;
                    }
                }
                return;
            }

            // Requests are grouped by their four excitation ranks so every packet shares one
            // fixed-rank determinant, cofactor, same-spin second-minor and mixed-spin cofactor
            // expression; ranks beyond the generated table share one scalar group.
            let keys: Vec<usize> = requests
                .iter()
                .map(|(_, x, wdet)| {
                    let (rxa, rwa) = (
                        usize::from(x.alpha.cache.rank),
                        usize::from(wdet.alpha.cache.rank),
                    );
                    let (rxb, rwb) = (
                        usize::from(x.beta.cache.rank),
                        usize::from(wdet.beta.cache.rank),
                    );
                    if rxa.max(rwa).max(rxb).max(rwb) < HAMRADIX {
                        ((rxa * HAMRADIX + rwa) * HAMRADIX + rxb) * HAMRADIX + rwb
                    } else {
                        usize::MAX
                    }
                })
                .collect();
            let (order, groups) = group_lanes(&keys);
            let lanes: Vec<(DeterminantData, DeterminantData)> = order
                .iter()
                .map(|&request| (requests[request].1, requests[request].2))
                .collect();

            let mut batch = HamiltonianLanes {
                w,
                enuc,
                tol,
                scratch,
                requests,
                order: &order,
                out,
            };
            xw_lanes(&mut batch, &lanes, &groups);
        }
    )
}

/// Batch of Hamiltonian and overlap requests of one ordered reference pair, grouped by their four
/// excitation ranks.
struct HamiltonianLanes<'a, T: NOCIScalar> {
    /// Wick intermediates of the ordered reference pair.
    w: &'a WicksPairView<'a, T>,
    /// Nuclear repulsion energy.
    enuc: f64,
    /// Numerical tolerance of the scalar evaluator.
    tol: f64,
    /// Reusable Wick workspace for scalar evaluation.
    scratch: &'a mut WickScratchSpin<T>,
    /// Output positions with their bra and ket determinants.
    requests: &'a [(usize, DeterminantData, DeterminantData)],
    /// Request of every lane.
    order: &'a [usize],
    /// Hamiltonian and overlap results in output order.
    out: &'a mut [(T, T)],
}

impl<T: NOCIScalar> LaneBatch for HamiltonianLanes<'_, T> {
    type Lane = (DeterminantData, DeterminantData);
    type Out = (T, T);

    /// Evaluate the Hamiltonian and overlap of one determinant pair through the prepared scalar
    /// path.
    /// # Arguments:
    /// - `self`: Batch of one ordered reference pair.
    /// - `_`: Position of the lane, unused by the scalar path.
    /// - `lane`: Bra and ket determinants relative to their references.
    /// # Returns
    /// - `(T, T)`: Hamiltonian and overlap matrix elements `(H, S)`.
    fn scalar(
        &mut self,
        _: usize,
        lane: &(DeterminantData, DeterminantData),
    ) -> (T, T) {
        let (x, wdet) = lane;
        xw_hamiltonian_overlap_prepared(self.w, (x, wdet), self.enuc, self.scratch, self.tol)
    }

    /// Write the Hamiltonian and overlap of one lane to the output of its request.
    /// # Arguments:
    /// - `self`: Batch of one ordered reference pair.
    /// - `position`: Position of the lane in rank order.
    /// - `value`: Hamiltonian and overlap matrix elements `(H, S)`.
    /// # Returns
    /// - `()`: Writes one entry of `out`.
    fn write(
        &mut self,
        position: usize,
        value: (T, T),
    ) {
        self.out[self.requests[self.order[position]].0] = value;
    }

    /// Select the four-lane kernel of a group of equal excitation ranks when
    /// `m_\alpha = m_\beta = 0`.
    /// # Arguments:
    /// - `self`: Batch of one ordered reference pair.
    /// - `lane`: First determinant pair of the group.
    /// # Returns
    /// - `Option<LaneKernel<Self, 4>>`: Kernel of the group, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn kernel_x4(
        &self,
        lane: &(DeterminantData, DeterminantData),
    ) -> Option<LaneKernel<Self, 4>> {
        if self.w.aa.m != 0 || self.w.bb.m != 0 {
            return None;
        }
        let (x, w) = lane;
        let ranks = (
            usize::from(x.alpha.cache.rank),
            usize::from(w.alpha.cache.rank),
            usize::from(x.beta.cache.rank),
            usize::from(w.beta.cache.rank),
        );
        dispatch_hamiltonian_ranks!(
            ranks,
            |RXA, RWA, LA, DA, MA, MDA, RXB, RWB, LB, DB, MB, MDB| Some(
                packet_x4::<T, RXA, RWA, LA, DA, MA, MDA, RXB, RWB, LB, DB, MB, MDB>
                    as LaneKernel<Self, 4>
            ),
            None,
        )
    }

    /// Select the eight-lane kernel of a group of equal excitation ranks when
    /// `m_\alpha = m_\beta = 0`.
    /// # Arguments:
    /// - `self`: Batch of one ordered reference pair.
    /// - `lane`: First determinant pair of the group.
    /// # Returns
    /// - `Option<LaneKernel<Self, 8>>`: Kernel of the group, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn kernel_x8(
        &self,
        lane: &(DeterminantData, DeterminantData),
    ) -> Option<LaneKernel<Self, 8>> {
        if self.w.aa.m != 0 || self.w.bb.m != 0 {
            return None;
        }
        let (x, w) = lane;
        let ranks = (
            usize::from(x.alpha.cache.rank),
            usize::from(w.alpha.cache.rank),
            usize::from(x.beta.cache.rank),
            usize::from(w.beta.cache.rank),
        );
        dispatch_hamiltonian_ranks!(
            ranks,
            |RXA, RWA, LA, DA, MA, MDA, RXB, RWB, LB, DB, MB, MDB| Some(
                packet_x8::<T, RXA, RWA, LA, DA, MA, MDA, RXB, RWB, LB, DB, MB, MDB>
                    as LaneKernel<Self, 8>
            ),
            None,
        )
    }
}

/// Dispatch an `m_\alpha = m_\beta = 0` Hamiltonian and overlap matrix element to the generated
/// fixed-rank packed kernel with one lane.
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
            dispatch_hamiltonian_ranks!(
                ranks,
                |RXA, RWA, LA, DA, MA, MDA, RXB, RWB, LB, DB, MB, MDB| {
                    let (mut h, mut s) = ([T::from_real(0.0)], [T::from_real(0.0)]);
                    unsafe {
                        xw_hamiltonian_overlap_m0_const::<
                            T,
                            T::Packed1,
                            1,
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
                        >(
                            w,
                            &[*x_ex],
                            &[*w_ex],
                            &[excitation_phase],
                            enuc,
                            &mut h,
                            &mut s,
                        )
                    };
                    Some((h[0], s[0]))
                },
                None,
            )
        }
    )
}

/// Evaluate packed fixed-rank Hamiltonian and overlap matrix elements.
/// Every lane evaluates the `m_\alpha=m_\beta=0` cofactor form: overlap is
/// `S=p\det\mathbf D_\alpha\det\mathbf D_\beta`, same-spin terms use scalar, one-column, and
/// `\mathcal J` second-minor classes, and different spin uses
/// `\sum\operatorname{cof}(\mathbf D_\alpha)\mathcal{II}\operatorname{cof}(\mathbf D_\beta)`.
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
#[inline(always)]
unsafe fn xw_hamiltonian_overlap_m0_const<
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

    // Map every SIMD lane onto the alpha-spin rows and columns of
    // `D_{alpha,ov}`; lanes share ranks but retain their own excitation labels.
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

    // Form each packed alpha determinant from `X` on/below the diagonal and
    // `Y` above it, retaining `cof(D_alpha)` for column replacements.
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
    let det_a = adjugate_transpose_const::<V, LANES, LA, DA>(&mut cof_a, &d_a);

    let mut j_a = zero;
    let jsl_a = w.aa.j_slice(0);
    let n2_a = n_a * n_a;
    let n3_a = n2_a * n_a;

    // Accumulate the same-spin two-column class
    // `sum phi J_{eta z,xi y} det D_alpha[eta,xi|z,y]`.
    for eta in 0..LA {
        for xi in (eta + 1)..LA {
            for z in 0..LA {
                for y in (z + 1)..LA {
                    let mut minor = [zero; MDA];
                    second_minor_const::<V, LA>(&mut minor, &d_a, eta, xi, z, y);
                    let second = det_const::<V, LANES, MA, MDA>(&minor);
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

    // Contract the alpha one-column intermediate with `cof(D_alpha)`.
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

    // Repeat the determinant, two-column, and one-column construction for beta spin.
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

    // `D_beta` obeys the same lower-`X`/upper-`Y` convention as `D_alpha`.
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
    let det_b = adjugate_transpose_const::<V, LANES, LB, DB>(&mut cof_b, &d_b);

    let mut j_b = zero;
    let jsl_b = w.bb.j_slice(0);
    let n2_b = n_b * n_b;
    let n3_b = n2_b * n_b;

    // Accumulate the beta same-spin two-column replacement class.
    for eta in 0..LB {
        for xi in (eta + 1)..LB {
            for z in 0..LB {
                for y in (z + 1)..LB {
                    let mut minor = [zero; MDB];
                    second_minor_const::<V, LB>(&mut minor, &d_b, eta, xi, z, y);
                    let second = det_const::<V, LANES, MB, MDB>(&minor);
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

    // Contract the beta one-column intermediate with `cof(D_beta)`.
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

    // Mixed spin factorises into one cofactor from each spin sector:
    // `sum cof(D_alpha) II^{alpha beta} cof(D_beta)`.
    // Put the smaller determinant outside to reduce repeated packed work.
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

    // Combine scalar, one-column, same-spin two-column, and mixed-spin classes.
    let det_ab = V::mul(det_a, det_b);
    let half = T::from_real(0.5);

    // `g0` is the unexcited nuclear, one-body, same-spin, and mixed-spin scalar.
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

    // Apply excitation phases and reduced overlaps after forming unscaled `(H,S)`.
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

/// Evaluate one four-lane packet of fixed-rank Hamiltonian and overlap values with AVX2/FMA.
/// # Arguments:
/// - `batch`: Batch holding the Wick intermediates and the nuclear repulsion energy.
/// - `lanes`: Bra and ket determinants of every lane.
/// # Returns
/// - `[(T, T); 4]`: Hamiltonian and overlap of every lane.
/// # Safety
/// - The current CPU must support AVX2/FMA; cached labels must match the fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn packet_x4<
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
    batch: &HamiltonianLanes<'_, T>,
    lanes: &[(DeterminantData, DeterminantData); 4],
) -> [(T, T); 4] {
    // Gather the excitation caches and the phases `p_x p_w` of every lane.
    let mut x_ex = [ExcitationCache::default(); 4];
    let mut w_ex = [ExcitationCache::default(); 4];
    let mut phases = [0.0; 4];
    for (lane, (x, w)) in lanes.iter().enumerate() {
        x_ex[lane] = ExcitationCache {
            alpha: x.alpha.cache,
            beta: x.beta.cache,
        };
        w_ex[lane] = ExcitationCache {
            alpha: w.alpha.cache,
            beta: w.beta.cache,
        };
        phases[lane] = x.phase() * w.phase();
    }

    // Evaluate `H` and `S` of every lane with one fixed-rank expression.
    let mut h = [T::from_real(0.0); 4];
    let mut s = [T::from_real(0.0); 4];
    unsafe {
        xw_hamiltonian_overlap_m0_const::<
            T,
            T::Packed4,
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
        >(batch.w, &x_ex, &w_ex, &phases, batch.enuc, &mut h, &mut s);
    }

    // Pair the Hamiltonian and overlap of every lane.
    let mut out = [(T::from_real(0.0), T::from_real(0.0)); 4];
    for (lane, value) in out.iter_mut().enumerate() {
        *value = (h[lane], s[lane]);
    }
    out
}

/// Evaluate one eight-lane packet of fixed-rank Hamiltonian and overlap values with AVX-512F.
/// # Arguments:
/// - `batch`: Batch holding the Wick intermediates and the nuclear repulsion energy.
/// - `lanes`: Bra and ket determinants of every lane.
/// # Returns
/// - `[(T, T); 8]`: Hamiltonian and overlap of every lane.
/// # Safety
/// - The current CPU must support AVX-512F; cached labels must match the fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn packet_x8<
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
    batch: &HamiltonianLanes<'_, T>,
    lanes: &[(DeterminantData, DeterminantData); 8],
) -> [(T, T); 8] {
    // Gather the excitation caches and the phases `p_x p_w` of every lane.
    let mut x_ex = [ExcitationCache::default(); 8];
    let mut w_ex = [ExcitationCache::default(); 8];
    let mut phases = [0.0; 8];
    for (lane, (x, w)) in lanes.iter().enumerate() {
        x_ex[lane] = ExcitationCache {
            alpha: x.alpha.cache,
            beta: x.beta.cache,
        };
        w_ex[lane] = ExcitationCache {
            alpha: w.alpha.cache,
            beta: w.beta.cache,
        };
        phases[lane] = x.phase() * w.phase();
    }

    // Evaluate `H` and `S` of every lane with one fixed-rank expression.
    let mut h = [T::from_real(0.0); 8];
    let mut s = [T::from_real(0.0); 8];
    unsafe {
        xw_hamiltonian_overlap_m0_const::<
            T,
            T::Packed8,
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
        >(batch.w, &x_ex, &w_ex, &phases, batch.enuc, &mut h, &mut s);
    }

    // Pair the Hamiltonian and overlap of every lane.
    let mut out = [(T::from_real(0.0), T::from_real(0.0)); 8];
    for (lane, value) in out.iter_mut().enumerate() {
        *value = (h[lane], s[lane]);
    }
    out
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

            // Evaluate `D_{\alpha,\mathrm{ov}}` and its cofactors; retain the determinant even
            // when the tolerance rejects cofactor-based operator replacements.
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

            // Evaluate `D_{\beta,\mathrm{ov}}` and its cofactors under the same tolerance rule.
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

            // Alpha spin: combine its scalar intermediate, one-column replacements, and
            // same-spin `J` second-minor class.
            if have_a {
                same_a = (w.aa.f0h[0] + half * w.aa.v0[0]) * det_a;

                // The alpha one-column class is a cofactor contraction with `H_{eta z}`.
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

                // The alpha two-column class contracts `J` with signed second minors.
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

            // Beta spin is the spin-swapped alpha construction.
            if have_b {
                same_b = (w.bb.f0h[0] + half * w.bb.v0[0]) * det_b;

                // Contract beta one-column intermediates against `cof(D_beta)`.
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

                // Contract beta `J` intermediates against signed second minors.
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

            // Different spin: contract `II` between one cofactor from each spin determinant.
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

            // Restore excitation phases and both reduced reference overlaps only after the
            // unscaled overlap and Hamiltonian classes have been assembled.
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
/// - `enuc`: Nuclear repulsion energy, added as `E_{\mathrm{nuc}} S`.
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
    enuc: f64,
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
                    // Preserve the overlap determinant if the adjugate evaluation fails.
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

            // As in the `m = 0` paths, the nuclear repulsion contributes `E_{\mathrm{nuc}} S`.
            let s = pref * sa * sb;
            let h = pref * (h1a * sb + h1b * sa + half * h2aa * sb + half * h2bb * sa + h2ab)
                + <T as From<f64>>::from(enuc) * s;

            (h, s)
        }
    )
}
