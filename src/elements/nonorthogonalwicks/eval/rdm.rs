// elements/nonorthogonalwicks/eval/rdm.rs

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::ExcitationSpin;
use crate::NOCIScalar;
use crate::config::MAXEXCIT;
#[cfg(target_arch = "x86_64")]
use crate::elements::LaneKernel;
use crate::elements::{LaneBatch, xw_lanes};
use crate::maths::{Simd, det_const, det_dynamic, mix_columns_dynamic};
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::dispatch::{dispatch_pair_ranks, dispatch_rdm_ranks, dispatch_spin_ranks};
use super::helpers::{extend_rdm_d, for_each_m_combination};
use super::overlap::xw_overlap_prepared;
use super::prepare::construct_determinant_indices;

/// Extended `X^{(0)}` and `Y^{(0)}` contractions in the external RDM basis, the optional `m_i = 1`
/// contractions and the row dimension.
type RdmFundamental<'a, T> = (&'a [T], &'a [T], Option<(&'a [T], &'a [T])>, usize);

/// Batch of same-spin rank-`K` RDM requests of one determinant pair.
struct RdmLanes<'a, T: NOCIScalar, const K: usize> {
    /// Same-spin reference-pair Wick intermediates.
    w: &'a SameSpinView<'a, T>,
    /// Bra and ket excitations.
    ex: (&'a ExcitationSpin, &'a ExcitationSpin),
    /// Extended `X^{(0)}`, `Y^{(0)}`, optional `m_i = 1` branches, and matrix rank.
    fundamental: RdmFundamental<'a, T>,
    /// Reusable determinant storage for scalar evaluation.
    scratch: &'a mut WickScratch<T>,
    /// Numerical threshold applied to individual determinant contributions.
    tol: f64,
    /// Same-spin RDM elements in request order.
    out: &'a mut [T],
}

impl<T: NOCIScalar, const K: usize> LaneBatch for RdmLanes<'_, T, K> {
    // LaneBatch requires the index request and scalar result types for its shared packet loop.
    type Lane = ([usize; K], [usize; K]);
    type Out = T;

    /// Evaluate one RDM request through the scalar prepared path.
    /// # Arguments:
    /// - `self`: Batch of one determinant pair.
    /// - `_`: Position of the request, unused by the scalar path.
    /// - `request`: Creation and annihilation index arrays.
    /// # Returns
    /// - `T`: Unnormalised same-spin rank-`K` transition-density element.
    fn scalar(
        &mut self,
        _: usize,
        request: &([usize; K], [usize; K]),
    ) -> T {
        xw_rdmk_same_prepared_scalar_value(
            self.w,
            self.ex,
            self.fundamental,
            request,
            self.scratch,
            self.tol,
        )
    }

    /// Write the RDM element of one request.
    /// # Arguments:
    /// - `self`: Batch of one determinant pair.
    /// - `position`: Position of the request.
    /// - `value`: Transition-density element of the request.
    /// # Returns
    /// - `()`: Writes one entry of `out`.
    fn write(
        &mut self,
        position: usize,
        value: T,
    ) {
        self.out[position] = value;
    }

    /// Select the four-lane kernel of the batch, which dispatches the fixed ranks of the pair.
    /// # Arguments:
    /// - `self`: Batch of one determinant pair.
    /// - `_`: First request, unused since every request shares the ranks of the pair.
    /// # Returns
    /// - `Option<LaneKernel<Self, 4>>`: Kernel of the batch, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn kernel_x4(
        &self,
        _: &([usize; K], [usize; K]),
    ) -> Option<LaneKernel<Self, 4>> {
        self.packed()
            .then_some(packet_x4::<T, K> as LaneKernel<Self, 4>)
    }

    /// Select the eight-lane kernel of the batch, which dispatches the fixed ranks of the pair.
    /// # Arguments:
    /// - `self`: Batch of one determinant pair.
    /// - `_`: First request, unused since every request shares the ranks of the pair.
    /// # Returns
    /// - `Option<LaneKernel<Self, 8>>`: Kernel of the batch, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn kernel_x8(
        &self,
        _: &([usize; K], [usize; K]),
    ) -> Option<LaneKernel<Self, 8>> {
        self.packed()
            .then_some(packet_x8::<T, K> as LaneKernel<Self, 8>)
    }
}

impl<T: NOCIScalar, const K: usize> RdmLanes<'_, T, K> {
    /// Return whether the generated packets cover the batch: `m = 0`, external rank `K <= 4` and
    /// excitation ranks up to `MAXEXCIT`.
    /// # Arguments:
    /// - `self`: Batch of RDM requests.
    /// # Returns
    /// - `bool`: Whether the batch is evaluated in packets.
    #[cfg(target_arch = "x86_64")]
    fn packed(&self) -> bool {
        self.w.m == 0
            && K <= 4
            && self.ex.0.holes.count_ones() as usize <= MAXEXCIT
            && self.ex.1.holes.count_ones() as usize <= MAXEXCIT
    }
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

    // A spin sector with near-zero orbital pairs is the sum of its branches.
    if w.nbranches() > 1 {
        let zero = <T as From<f64>>::from(0.0);
        out.fill(zero);
        let mut part = vec![zero; out.len()];
        for k in 0..w.nbranches() {
            xw_rdmk_same_prepared_batched::<T, K>(
                &w.branch(k),
                ex,
                coeff,
                requests,
                scratch,
                tol,
                &mut part,
            );
            for (value, branch) in out.iter_mut().zip(&part) {
                *value += *branch;
            }
        }
        return;
    }

    // A rank-`K` density annihilates `K` electrons, so `\Gamma^{(K)}=0` for `K>N_\sigma`.
    if K > w.nocc {
        for value in out.iter_mut().take(requests.len()) {
            *value = <T as From<f64>>::from(0.0);
        }
        return;
    }

    // With no external operators, `\Gamma^{(0)}=S` and no basis transform is needed.
    if K == 0 {
        let overlap = xw_overlap_prepared(w, ex.0, ex.1, scratch);
        for value in out.iter_mut().take(requests.len()) {
            *value = overlap;
        }
        return;
    }

    // Extend each fundamental contraction once for the whole batch:
    // `D_{\text{ext}} = [[D,C_{\text{row}}^\dagger D_{\text{RDM}}],`
    // `[D_{\text{RDM}}C_{\text{col}},D_{\text{RDM}}]]`, for both `X^{(0)}` and `Y^{(0)}`.
    // This matrix supplies compact excitation labels and external RDM labels together.
    let (l_c, g_c) = coeff;
    let nrdm = l_c.nrows();
    let ext_n = w.nmo + nrdm;
    let x0 = w.x(0);
    let y0 = w.y(0);
    let x0rdm = w.xrdm(0, nrdm);
    let y0rdm = w.yrdm(0, nrdm);
    let x0p = extend_rdm_d(w, &x0, &x0rdm, l_c, g_c).into_raw_vec();
    let y0p = extend_rdm_d(w, &y0, &y0rdm, l_c, g_c).into_raw_vec();

    // Nonzero `m` also needs the `m_i = 1` contractions in the external RDM basis.
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

    // Every request shares the excitations of the pair, so the batch is one group.
    let mut batch = RdmLanes {
        w,
        ex,
        fundamental,
        scratch,
        tol,
        out,
    };
    xw_lanes(&mut batch, requests, &[0, requests.len()]);
}

/// Evaluate packed fixed-rank same-spin RDM determinants.
/// Each lane evaluates
/// `\Gamma^{\mathbf p}_{\mathbf q} = {}^{xw}\tilde S`
/// `\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)`, where the external RDM
/// contractions precede the excitation contractions in the augmented determinant.
/// # Arguments:
/// - `w`: Same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions and row dimension.
/// - `requests`: External RDM indices in lane order.
/// - `tol`: Numerical threshold.
/// - `out`: RDM outputs in lane order.
/// # Returns
/// - `()`: Writes `LANES` transition-density values.
/// # Safety
/// - Indices must be valid and caller must establish `V` CPU support.
#[inline(always)]
unsafe fn xw_rdmk_same_m0_const<
    T: NOCIScalar,
    V: Simd<LANES, Scalar = T>,
    const LANES: usize,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const DD: usize,
>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K]); LANES],
    tol: f64,
    out: &mut [T; LANES],
) {
    // The augmented determinant has rank `D=K+R_x+R_w`. Its first `K` label pairs are
    // `(r_i,c_i)=(p_i,q_i)` in the external basis, offset by `n_{\text{mo}}` in the slabs.
    // Excitation labels occupy the remaining positions and are shared by every lane.
    let (x0, y0, ext_n) = fundamental;
    let mut rows = [[0usize; D]; LANES];
    let mut cols = [[0usize; D]; LANES];

    for lane in 0..LANES {
        for i in 0..K {
            rows[lane][i] = w.nmo + requests[lane].0[i];
            cols[lane][i] = w.nmo + requests[lane].1[i];
        }
    }

    // Append bra `(a,i)` pairs in the compact `V_x \times O_x` orbital spaces.
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    let mut x_holes = ex.0.holes;
    let mut x_parts = ex.0.parts;

    for i in 0..RX {
        let col = x_holes.trailing_zeros() as usize;
        let row = x_parts.trailing_zeros() as usize - nocc;

        for lane in 0..LANES {
            cols[lane][K + i] = col;
            rows[lane][K + i] = row;
        }
        x_holes &= x_holes - 1;
        x_parts &= x_parts - 1;
    }

    // Append ket `(j,b)` pairs in `O_w \times V_w`, after all bra excitation labels.
    let mut w_holes = ex.1.holes;
    let mut w_parts = ex.1.parts;
    for i in 0..(L - RX) {
        let row = nvirt + w_holes.trailing_zeros() as usize;
        let col = w_parts.trailing_zeros() as usize;

        for lane in 0..LANES {
            rows[lane][K + RX + i] = row;
            cols[lane][K + RX + i] = col;
        }
        w_holes &= w_holes - 1;
        w_parts &= w_parts - 1;
    }

    let zero = V::zero();
    let mut determinant = [zero; DD];

    // Build packed augmented determinant `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}`.
    // Entries use `X^{(0)}` where the column creator stands left of the row annihilator in the
    // operator string, and `Y^{(0)}` otherwise. The external string is creators first,
    // `a^\dagger_{p_1}\cdots a^\dagger_{p_K} a_{q_K}\cdots a_{q_1}`, so its whole `K x K` block uses
    // `X^{(0)}`; the remaining entries take `X^{(0)}` on and below the diagonal.
    // Thus `D_{ij}=X^{(0)}_{r_i c_j}` if `i\geq j` or `j<K`, and `Y^{(0)}_{r_i c_j}` otherwise.
    for i in 0..D {
        for j in 0..D {
            let matrix = if i >= j || j < K { x0 } else { y0 };
            let mut values = [T::from_real(0.0); LANES];

            for lane in 0..LANES {
                let index = rows[lane][i] * ext_n + cols[lane][j];
                values[lane] = unsafe { *matrix.get_unchecked(index) };
            }
            determinant[i * D + j] = V::load(&values);
        }
    }

    // `\Gamma^{\mathbf p}_{\mathbf q}={}^{xw}\tilde S\det\mathbf D_{\text{RDM}}^{\mathbf p\mathbf q}`.
    // The orbital-pairing phase is included here; the caller supplies excitation phases.
    let value = det_const::<V, LANES, D, DD>(&determinant);
    let pref = V::splat(w.phase * T::from_real(w.tilde_s_prod));
    let mut lanes = [T::from_real(0.0); LANES];
    V::store(V::mul(pref, value), &mut lanes);

    // The packed path retains the scaled element only when `|\Gamma^{\mathbf p}_{\mathbf q}|>tol`.
    for lane in 0..LANES {
        out[lane] = if lanes[lane].abs() > tol {
            lanes[lane]
        } else {
            T::from_real(0.0)
        };
    }
}

/// Dispatch four RDM values to a fixed-rank AVX2/FMA kernel.
/// # Arguments:
/// - `batch`: Batch holding the Wick intermediates, excitations and extended contractions.
/// - `requests`: External RDM indices of every lane.
/// # Returns
/// - `[T; 4]`: RDM value of every lane.
/// # Safety
/// - The current CPU must support AVX2/FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn packet_x4<T: NOCIScalar, const K: usize>(
    batch: &RdmLanes<'_, T, K>,
    requests: &[([usize; K], [usize; K]); 4],
) -> [T; 4] {
    // Packets evaluate `m = 0`, which needs only the `m_i = 0` contractions.
    let (w, ex, tol) = (batch.w, batch.ex, batch.tol);
    let (x0p, y0p, _, ext_n) = batch.fundamental;
    let fundamental = (x0p, y0p, ext_n);

    // Dispatch the fixed external and excitation ranks of the pair.
    let mut out = [T::from_real(0.0); 4];
    let rx = ex.0.holes.count_ones() as usize;
    let rw = ex.1.holes.count_ones() as usize;
    dispatch_rdm_ranks!(
        K,
        (rx, rw),
        |K, RX, RW, L, D, DD| unsafe {
            xw_rdmk_same_m0_prepared_x4_const::<T, K, RX, RW, L, D, DD>(
                w,
                ex,
                fundamental,
                &*std::ptr::from_ref(requests).cast::<[([usize; K], [usize; K]); 4]>(),
                tol,
                &mut out,
            )
        },
        (),
    );
    out
}

/// Dispatch eight RDM values to a fixed-rank AVX-512F kernel.
/// # Arguments:
/// - `batch`: Batch holding the Wick intermediates, excitations and extended contractions.
/// - `requests`: External RDM indices of every lane.
/// # Returns
/// - `[T; 8]`: RDM value of every lane.
/// # Safety
/// - The current CPU must support AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn packet_x8<T: NOCIScalar, const K: usize>(
    batch: &RdmLanes<'_, T, K>,
    requests: &[([usize; K], [usize; K]); 8],
) -> [T; 8] {
    // Packets evaluate `m = 0`, which needs only the `m_i = 0` contractions.
    let (w, ex, tol) = (batch.w, batch.ex, batch.tol);
    let (x0p, y0p, _, ext_n) = batch.fundamental;
    let fundamental = (x0p, y0p, ext_n);

    // Dispatch the fixed external and excitation ranks of the pair.
    let mut out = [T::from_real(0.0); 8];
    let rx = ex.0.holes.count_ones() as usize;
    let rw = ex.1.holes.count_ones() as usize;
    dispatch_rdm_ranks!(
        K,
        (rx, rw),
        |K, RX, RW, L, D, DD| unsafe {
            xw_rdmk_same_m0_prepared_x8_const::<T, K, RX, RW, L, D, DD>(
                w,
                ex,
                fundamental,
                &*std::ptr::from_ref(requests).cast::<[([usize; K], [usize; K]); 8]>(),
                tol,
                &mut out,
            )
        },
        (),
    );
    out
}

/// Evaluate four fixed-rank RDM values with AVX2/FMA.
/// # Arguments:
/// - `w`: Same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: RDM outputs.
/// # Returns
/// - `()`: Writes four values.
/// # Safety
/// - The current CPU must support AVX2/FMA; indices must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xw_rdmk_same_m0_prepared_x4_const<
    T: NOCIScalar,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const DD: usize,
>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K]); 4],
    tol: f64,
    out: &mut [T; 4],
) {
    unsafe {
        xw_rdmk_same_m0_const::<T, T::Packed4, 4, K, RX, RW, L, D, DD>(
            w,
            ex,
            fundamental,
            requests,
            tol,
            out,
        );
    }
}

/// Evaluate eight fixed-rank RDM values with AVX-512F.
/// # Arguments:
/// - `w`: Same-spin Wick intermediates.
/// - `ex`: Bra and ket excitations.
/// - `fundamental`: Extended contractions.
/// - `requests`: External RDM indices.
/// - `tol`: Numerical threshold.
/// - `out`: RDM outputs.
/// # Returns
/// - `()`: Writes eight values.
/// # Safety
/// - The current CPU must support AVX-512F; indices must match fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xw_rdmk_same_m0_prepared_x8_const<
    T: NOCIScalar,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const DD: usize,
>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: (&[T], &[T], usize),
    requests: &[([usize; K], [usize; K]); 8],
    tol: f64,
    out: &mut [T; 8],
) {
    unsafe {
        xw_rdmk_same_m0_const::<T, T::Packed8, 8, K, RX, RW, L, D, DD>(
            w,
            ex,
            fundamental,
            requests,
            tol,
            out,
        );
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
#[inline(always)]
fn xw_rdmk_same_prepared_scalar_value<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: RdmFundamental<'_, T>,
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    let (x_ex, w_ex) = ex;
    let rx = x_ex.holes.count_ones() as usize;
    let rw = w_ex.holes.count_ones() as usize;
    let l = rx + rw;

    // Augmented determinant has `K + L` contractions, so no assignment can absorb `m > K + L`
    // zero-overlap pairs.
    if w.m > l + K {
        return <T as From<f64>>::from(0.0);
    }

    // `m = 0` needs one determinant; `m > 0` sums all binary column assignments with
    // `\sum_i m_i = m`.
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
#[inline(always)]
fn xw_rdmk_same_m0_prepared<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: RdmFundamental<'_, T>,
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_m0_prepared,
        {
            // Dispatch by `(K,R_x,R_w)` so generated dimensions use stack-allocated determinants;
            // arbitrary ranks preserve the same augmented determinant in runtime storage.
            let rx = ex.0.holes.count_ones() as usize;
            let rw = ex.1.holes.count_ones() as usize;
            dispatch_rdm_ranks!(
                K,
                (rx, rw),
                |K, RX, RW, L, D, DD| {
                    // SAFETY: `dispatch_rdm_ranks!` selects this arm only when the caller's const
                    // `K` equals the arm-local literal `K`, so the two array-reference layouts are
                    // identical.
                    let request =
                        unsafe { &*std::ptr::from_ref(request).cast::<([usize; K], [usize; K])>() };

                    // Generated ranks evaluate the packed kernel with one lane.
                    let mut value = [T::from_real(0.0)];
                    unsafe {
                        xw_rdmk_same_m0_const::<T, T::Packed1, 1, K, RX, RW, L, D, DD>(
                            w,
                            ex,
                            (fundamental.0, fundamental.1, fundamental.3),
                            std::array::from_ref(request),
                            tol,
                            &mut value,
                        )
                    };
                    value[0]
                },
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

/// Evaluate one same-spin rank-`K` `m = 0` element outside the const-dispatch table.
/// This computes `{}^{xw}\tilde S\det\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}` with runtime
/// augmented dimension `D = K + RX + RW` and the same fill convention: `X` on and below the
/// diagonal and in the creators-first external block, `Y` elsewhere above it.
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
            // `D=K+R_x+R_w`: put external `(p,q)` labels first, followed by the compact
            // bra `(a,i)` and ket `(j,b)` excitation labels.
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

            // Construct `\mathbf D_{\mathrm{RDM}}^{\mathbf p\mathbf q}(0,\ldots,0)` at runtime, with
            // the creators-first external block entirely in `X^{(0)}`:
            // `D_{ij}=X^{(0)}_{r_i c_j}` for `i\geq j` or `j<K`, and `Y^{(0)}_{r_i c_j}` otherwise.
            for (i, &r) in rows.iter().enumerate().take(d_rank) {
                let row = r * ext_n;
                for j in 0..d_rank {
                    d[i * d_rank + j] = if i >= j || j < K {
                        x0[row + cols[j]]
                    } else {
                        y0[row + cols[j]]
                    };
                }
            }

            // The runtime path thresholds the unscaled determinant, `|\det\mathbf D|>tol`,
            // then restores `\Gamma^{\mathbf p}_{\mathbf q}={}^{xw}\tilde S\det\mathbf D`.
            let zero = <T as From<f64>>::from(0.0);
            if let Some(value) = det_dynamic(d, d_rank)
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
#[inline(always)]
fn xw_rdmk_same_gen_prepared<T: NOCIScalar, const K: usize>(
    w: &SameSpinView<'_, T>,
    ex: (&ExcitationSpin, &ExcitationSpin),
    fundamental: RdmFundamental<'_, T>,
    request: &([usize; K], [usize; K]),
    scratch: &mut WickScratch<T>,
    tol: f64,
) -> T {
    time_call!(
        crate::timers::nonorthogonalwicks::add_xw_rdmk_same_gen_prepared,
        {
            // Build external-first labels for augmented rank `D = K + L`:
            // `(p,q)` external pairs, then `(a,i)` bra pairs and `(j,b)` ket pairs.
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

            // Prepare all-`m_i=0` and all-`m_i=1` endpoint determinants. Each allowed distribution
            // later selects whole columns from these endpoints. The creators-first external
            // block lies entirely in `X`.
            // `D^{(m)}_{ij}=X^{(m)}_{r_i c_j}` for `i\geq j` or `j<K`, and `Y^{(m)}_{r_i c_j}` otherwise.
            for (i, &r) in rows.iter().enumerate().take(d_rank) {
                let row = r * ext_n;
                for j in 0..d_rank {
                    let index = i * d_rank + j;
                    if i >= j || j < K {
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

            // Sum `\det\mathbf D_{\mathrm{RDM}}(m_1,\ldots,m_D)` over every binary assignment
            // satisfying `\sum_i m_i = m`.
            for_each_m_combination(d_rank, w.m, |bits| {
                mix_columns_dynamic(
                    scratch.det_mix.as_mut_slice(),
                    scratch.det0.as_slice(),
                    scratch.det1.as_slice(),
                    d_rank,
                    bits,
                );

                // Retain each term with `|\det\mathbf D(\{m_i\})|>tol` before summing.
                if let Some(value) = det_dynamic(scratch.det_mix.as_slice(), d_rank)
                    && value.abs() > tol
                {
                    acc += value;
                }
            });

            // `\Gamma^{\mathbf p}_{\mathbf q}={}^{xw}\tilde S`
            // `\sum_{\sum_i m_i=m}\det\mathbf D_{\text{RDM}}^{\mathbf p\mathbf q}(\{m_i\})`.
            w.phase * <T as From<f64>>::from(w.tilde_s_prod) * acc
        }
    )
}
