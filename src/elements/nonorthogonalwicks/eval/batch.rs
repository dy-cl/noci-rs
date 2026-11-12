// elements/nonorthogonalwicks/eval/batch.rs
//! Rows of same-spin Wick factors for one ordered reference pair.
//!
//! A row pairs one target spin determinant with many source spin determinants. Every factor family,
//! the overlap and the one-body factors, supplies only its scalar evaluator and its packed
//! fixed-rank `m = 0` evaluator; this module sums near-zero branches, evaluates the groups of equal
//! rank and holes of the sources through the shared lane batching and restores the excitation
//! phases.

// Crate-root imports.
use crate::ExcitationSpinCache;
use crate::NOCIScalar;
#[cfg(target_arch = "x86_64")]
use crate::config::MAXEXCIT;
use crate::determinant::SpinStringData;
#[cfg(target_arch = "x86_64")]
use crate::elements::LaneKernel;
use crate::elements::{LaneBatch, xw_lanes};
use crate::maths::Simd;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
#[cfg(target_arch = "x86_64")]
use super::dispatch::{dispatch_pair_ranks, dispatch_spin_ranks};

/// A same-spin Wick factor family evaluating `K` factors for every determinant pair, such as the
/// overlap alone or the overlap together with a one-body operator.
///
/// For excited spin strings built on references `x` and `w`, `OverlapFactor` returns
/// `S^\sigma_{xw} = \langle{}^x\Psi^\sigma_{i\cdots}^{a\cdots}|{}^w\Psi^\sigma_{j\cdots}^{b\cdots}\rangle`
/// and `OneBodyFactor` returns both `S^\sigma_{xw}` and
/// `F^\sigma_{xw} = \langle{}^x\Psi^\sigma_{i\cdots}^{a\cdots}|\hat f_\sigma|{}^w\Psi^\sigma_{j\cdots}^{b\cdots}\rangle`.
/// With `L = R_x + R_w`, their phase-free Wick expressions are
/// `S = {}^{xw}\tilde S\sum_{m_1+\cdots+m_L=m}\det\mathbf D_{\text{ov}}(m_1,\ldots,m_L)` and
/// `F = {}^{xw}\tilde S\sum_{m_1+\cdots+m_{L+1}=m}`
/// `[{}^xF_0^{(m_1)}\det\mathbf D_{\text{ov}}(m_2,\ldots,m_{L+1})`
/// `-\sum_z\det\mathbf D_{\text{ov}}^{z\rightarrow\boldsymbol{\mathcal F}_z}]`, with `m_i\in\{0,1\}`.
/// The lower triangle of `\mathbf D_{\text{ov}}` uses `X^{(m_i)}`, and its upper triangle uses
/// `Y^{(m_i)}`. `packed` evaluates the single all-zero assignment when `m = 0`; `values` also
/// handles the constrained distribution sum. Both include the reduced reference overlap, with
/// its orbital-pairing phase; the row writer supplies the excitation phases `p_x p_w`.
pub(crate) trait SpinFactor<const K: usize> {
    /// Evaluate the factors of one determinant pair through the scalar path, excluding the
    /// excitation phases.
    /// # Arguments:
    /// - `self`: Factor family and its evaluation parameters.
    /// - `w`: Same-spin reference-pair Wick intermediates.
    /// - `pair`: Bra and ket determinants, oriented as the reference pair `(x,w)`.
    /// - `scratch`: Reusable same-spin Wick workspace.
    /// # Returns
    /// - `[T; K]`: Factors of the determinant pair.
    fn values<T: NOCIScalar>(
        &self,
        w: &SameSpinView<'_, T>,
        pair: (&SpinStringData, &SpinStringData),
        scratch: &mut WickScratch<T>,
    ) -> [T; K];

    /// Evaluate the `m = 0` factors of one packet of fixed excitation ranks, excluding the
    /// excitation phases.
    /// # Arguments:
    /// - `w`: Same-spin reference-pair Wick intermediates with `m = 0`.
    /// - `packet`: Contraction labels of every lane.
    /// # Returns
    /// - `[[T; LANES]; K]`: Factors of every lane.
    /// # Safety
    /// - Cached labels must match `RX` and `RW`, and the caller must establish CPU support for `V`.
    unsafe fn packed<
        T: NOCIScalar,
        V: Simd<LANES, Scalar = T>,
        const LANES: usize,
        const RX: usize,
        const RW: usize,
        const L: usize,
        const D: usize,
        const XFIX: bool,
    >(
        w: &SameSpinView<'_, T>,
        packet: &Packet<'_, LANES>,
    ) -> [[T; LANES]; K];
}

/// Source spin determinants of factor rows in packed evaluation order. Sources are sorted by
/// excitation rank, holes and particles, so every group of equal rank and holes fills packets of
/// one fixed-rank kernel whose hole contractions are shared by every lane.
#[derive(Default)]
pub(crate) struct SpinStringSources {
    /// Source spin determinants in evaluation order.
    states: Vec<SpinStringData>,
    /// Excitation cache of every source in evaluation order, contiguous so that a packet of
    /// sources is one array.
    caches: Vec<ExcitationSpinCache>,
    /// Output column of every source in evaluation order.
    columns: Vec<usize>,
    /// Boundaries of the equal-rank, common-hole groups in `states`.
    groups: Vec<usize>,
}

/// One row of same-spin factors, pairing a target spin determinant with every source.
pub(crate) struct SpinFactorRow<'a, T, const K: usize> {
    /// Target spin determinant shared by the row.
    pub(crate) target: SpinStringData,
    /// Source spin determinants in packed evaluation order.
    pub(crate) sources: &'a SpinStringSources,
    /// Whether the target belongs to the left reference of the Wick intermediates.
    pub(crate) target_left: bool,
    /// Output factors in source-column order, one row per factor.
    pub(crate) out: [&'a mut [T]; K],
}

/// Contraction labels of one packet. The fixed determinant is shared by every lane and the varying
/// determinants, one per lane, share their holes, so an entry of the contraction determinant is
/// shared by every lane whenever its labels come from the fixed determinant or from the holes.
pub(crate) struct Packet<'a, const LANES: usize> {
    /// Excitation cache shared by every lane.
    fixed: &'a ExcitationSpinCache,
    /// Excitation cache of every lane.
    varying: &'a [ExcitationSpinCache; LANES],
    /// Number of occupied orbitals of the spin sector.
    nocc: usize,
    /// Number of virtual orbitals of the spin sector.
    nvirt: usize,
    /// Row length of the contraction matrices.
    n: usize,
}

/// One row of same-spin factors, with its sources in packed evaluation order.
struct SpinFactorLanes<'a, 'r, F, T: NOCIScalar, const K: usize> {
    /// Factor family and its evaluation parameters.
    factor: &'a F,
    /// Same-spin reference-pair Wick intermediates.
    w: &'a SameSpinView<'a, T>,
    /// Target, sources and outputs of the row.
    row: &'a mut SpinFactorRow<'r, T, K>,
    /// Reusable same-spin Wick workspace for scalar evaluation.
    scratch: &'a mut WickScratch<T>,
}

impl SpinStringSources {
    /// Sort source spin determinants into packed evaluation order.
    /// # Arguments:
    /// - `sources`: Source spin determinants in output-column order.
    /// # Returns
    /// - `Self`: Sources in evaluation order with their output columns and group boundaries.
    pub(crate) fn new(sources: &[SpinStringData]) -> Self {
        // Order sources by rank, holes and particles so equal-rank, common-hole sources are
        // adjacent; ties keep source order.
        let mut columns: Vec<usize> = (0..sources.len()).collect();
        columns.sort_unstable_by(|&i, &j| {
            let ic = sources[i].cache;
            let jc = sources[j].cache;
            ic.rank
                .cmp(&jc.rank)
                .then_with(|| ic.holes.cmp(&jc.holes))
                .then_with(|| ic.particles.cmp(&jc.particles))
                .then_with(|| i.cmp(&j))
        });
        let states: Vec<SpinStringData> = columns.iter().map(|&col| sources[col]).collect();
        let caches = states.iter().map(|source| source.cache).collect();

        // A new group starts wherever the rank or the holes change.
        let mut groups = vec![0];
        for position in 1..states.len() {
            let previous = states[position - 1].cache;
            let current = states[position].cache;
            if current.rank != previous.rank || current.holes != previous.holes {
                groups.push(position);
            }
        }
        if !states.is_empty() {
            groups.push(states.len());
        }

        Self {
            states,
            caches,
            columns,
            groups,
        }
    }

    /// Return the number of source spin determinants.
    /// # Arguments:
    /// - `self`: Sources in evaluation order.
    /// # Returns
    /// - `usize`: Number of sources, the length of every output row.
    pub(crate) fn len(&self) -> usize {
        self.states.len()
    }
}

impl<'a, const LANES: usize> Packet<'a, LANES> {
    /// Collect the contraction labels of one packet.
    /// # Arguments:
    /// - `w`: Same-spin reference-pair Wick intermediates.
    /// - `fixed`: Excitation cache shared by every lane.
    /// - `varying`: Excitation cache of every lane.
    /// # Returns
    /// - `Self`: Contraction labels of the packet.
    pub(crate) fn new<T: NOCIScalar>(
        w: &SameSpinView<'_, T>,
        fixed: &'a ExcitationSpinCache,
        varying: &'a [ExcitationSpinCache; LANES],
    ) -> Self {
        Self {
            fixed,
            varying,
            nocc: w.nocc,
            nvirt: w.nmo - w.nocc,
            n: w.n(),
        }
    }

    /// Load entry `(\eta, z)` of a contraction matrix for every lane, `M_{r_\eta c_z}`, or
    /// `M_{c_z r_\eta}` when `transpose` is set. Rows `r_\eta` run over the bra particles and then
    /// the ket holes, columns `c_z` over the bra holes and then the ket particles. Entries shared
    /// by every lane are broadcast instead of gathered.
    /// # Arguments:
    /// - `values`: Row-major contraction matrix.
    /// - `eta`: Row of the contraction determinant.
    /// - `z`: Column of the contraction determinant.
    /// - `transpose`: Whether to read the transposed entry.
    /// # Returns
    /// - `V`: Entry of every lane.
    /// # Safety
    /// - Cached labels must match `RX` and index within `values`.
    #[inline(always)]
    pub(crate) unsafe fn load<
        T: Copy,
        V: Simd<LANES, Scalar = T>,
        const RX: usize,
        const XFIX: bool,
    >(
        &self,
        values: &[T],
        eta: usize,
        z: usize,
        transpose: bool,
    ) -> V {
        // The bra is fixed when `XFIX`; otherwise the ket is.
        let bra = |lane: usize| -> &ExcitationSpinCache {
            if XFIX {
                self.fixed
            } else {
                unsafe { self.varying.get_unchecked(lane) }
            }
        };

        let ket = |lane: usize| -> &ExcitationSpinCache {
            if XFIX {
                unsafe { self.varying.get_unchecked(lane) }
            } else {
                self.fixed
            }
        };

        let index = |lane: usize| -> usize {
            let row = if eta < RX {
                usize::from(unsafe { *bra(lane).particles.get_unchecked(eta) }) - self.nocc
            } else {
                self.nvirt + usize::from(unsafe { *ket(lane).holes.get_unchecked(eta - RX) })
            };
            let col = if z < RX {
                usize::from(unsafe { *bra(lane).holes.get_unchecked(z) })
            } else {
                usize::from(unsafe { *ket(lane).particles.get_unchecked(z - RX) })
            };

            if transpose {
                col * self.n + row
            } else {
                row * self.n + col
            }
        };

        // With a fixed bra only the ket particles vary, in columns `z \geq R_x`; with a fixed ket
        // only the bra particles vary, in rows `\eta < R_x`.
        let shared = if XFIX { z < RX } else { eta >= RX };

        if shared {
            V::splat(unsafe { *values.get_unchecked(index(0)) })
        } else {
            let mut lanes = [unsafe { *values.get_unchecked(index(0)) }; LANES];
            for (lane, value) in lanes.iter_mut().enumerate().skip(1) {
                *value = unsafe { *values.get_unchecked(index(lane)) };
            }

            V::load(&lanes)
        }
    }
}

impl<F: SpinFactor<K>, T: NOCIScalar, const K: usize> LaneBatch
    for SpinFactorLanes<'_, '_, F, T, K>
{
    // Required by LaneBatch: each source contributes its contraction labels and K spin factors.
    type Lane = ExcitationSpinCache;
    type Out = [T; K];

    /// Evaluate the factors of the row target and one source through the scalar path, excluding
    /// the excitation phases.
    /// # Arguments:
    /// - `self`: Row of same-spin factors.
    /// - `position`: Position of the source in evaluation order.
    /// - `_`: Excitation cache of the source, read from the row instead.
    /// # Returns
    /// - `[T; K]`: Factors of the pair.
    fn scalar(
        &mut self,
        position: usize,
        _: &ExcitationSpinCache,
    ) -> [T; K] {
        // Orient the pair as the ordered reference pair `(x,w)`.
        let source = &self.row.sources.states[position];
        let pair = if self.row.target_left {
            (&self.row.target, source)
        } else {
            (source, &self.row.target)
        };

        self.factor.values(self.w, pair, self.scratch)
    }

    /// Select the four-lane kernel of a group of equal rank and holes.
    /// # Arguments:
    /// - `self`: Row of same-spin factors.
    /// - `source`: First source of the group.
    /// # Returns
    /// - `Option<LaneKernel<Self, 4>>`: Kernel of the group, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn kernel_x4(
        &self,
        source: &ExcitationSpinCache,
    ) -> Option<LaneKernel<Self, 4>> {
        let ranks = self.packed_ranks(source)?;

        if self.row.target_left {
            dispatch_spin_ranks!(
                ranks,
                |RX, RW, L, D| Some(
                    packet_x4::<F, T, K, RX, RW, L, D, true> as LaneKernel<Self, 4>
                ),
                None,
            )
        } else {
            dispatch_spin_ranks!(
                ranks,
                |RX, RW, L, D| Some(
                    packet_x4::<F, T, K, RX, RW, L, D, false> as LaneKernel<Self, 4>
                ),
                None,
            )
        }
    }

    /// Select the eight-lane kernel of a group of equal rank and holes.
    /// # Arguments:
    /// - `self`: Row of same-spin factors.
    /// - `source`: First source of the group.
    /// # Returns
    /// - `Option<LaneKernel<Self, 8>>`: Kernel of the group, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn kernel_x8(
        &self,
        source: &ExcitationSpinCache,
    ) -> Option<LaneKernel<Self, 8>> {
        let ranks = self.packed_ranks(source)?;

        if self.row.target_left {
            dispatch_spin_ranks!(
                ranks,
                |RX, RW, L, D| Some(
                    packet_x8::<F, T, K, RX, RW, L, D, true> as LaneKernel<Self, 8>
                ),
                None,
            )
        } else {
            dispatch_spin_ranks!(
                ranks,
                |RX, RW, L, D| Some(
                    packet_x8::<F, T, K, RX, RW, L, D, false> as LaneKernel<Self, 8>
                ),
                None,
            )
        }
    }

    /// Write the factors of one source into its output column with the excitation phases.
    /// # Arguments:
    /// - `self`: Row of same-spin factors.
    /// - `position`: Position of the source in evaluation order.
    /// - `values`: Factors of the pair, excluding the excitation phases.
    /// # Returns
    /// - `()`: Writes one column of the row outputs.
    fn write(
        &mut self,
        position: usize,
        values: [T; K],
    ) {
        // Restore the excitation phases and scatter from evaluation order to source order.
        let column = self.row.sources.columns[position];
        let phase = T::from_real(self.row.target.phase * self.row.sources.states[position].phase);

        for (out, value) in self.row.out.iter_mut().zip(values) {
            out[column] = phase * value;
        }
    }
}

impl<F, T: NOCIScalar, const K: usize> SpinFactorLanes<'_, '_, F, T, K> {
    /// Return the bra and ket excitation ranks of a group when it has a packed kernel: `m = 0`,
    /// both ranks within the generated table and not both determinants unexcited.
    /// # Arguments:
    /// - `self`: Row of same-spin factors.
    /// - `source`: First source of the group.
    /// # Returns
    /// - `Option<(usize, usize)>`: Bra and ket ranks, or `None` for scalar evaluation.
    #[cfg(target_arch = "x86_64")]
    fn packed_ranks(
        &self,
        source: &ExcitationSpinCache,
    ) -> Option<(usize, usize)> {
        let target = usize::from(self.row.target.cache.rank);
        let source = usize::from(source.rank);
        if self.w.m != 0 || target > MAXEXCIT || source > MAXEXCIT || target + source == 0 {
            return None;
        }

        Some(if self.row.target_left {
            (target, source)
        } else {
            (source, target)
        })
    }
}

/// Evaluate one row of same-spin factors for one ordered reference pair.
/// For each source j, the overlap family writes `out[0][j] = p_x p_w S` and the one-body family
/// also writes `out[1][j] = p_x p_w F`, using the constrained determinant expressions in
/// `SpinFactor`. Thus the same row supplies both distinct factors of pair requests and the
/// same-spin matrices used by factorised overlap and Fock applications.
/// Groups with `m = 0` and generated ranks are evaluated in packets of the widest available SIMD
/// kernel; every other source uses the scalar evaluator.
/// # Arguments:
/// - `factor`: Factor family and its evaluation parameters.
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `row`: Target, sources and outputs of the row.
/// - `scratch`: Reusable same-spin Wick workspace for scalar evaluation.
/// # Returns
/// - `()`: Writes every factor of the row into `row.out`.
pub(crate) fn xw_spin_factor_row<F: SpinFactor<K>, T: NOCIScalar, const K: usize>(
    factor: &F,
    w: &SameSpinView<'_, T>,
    mut row: SpinFactorRow<'_, T, K>,
    scratch: &mut WickScratch<T>,
) {
    // Resolve the stored zero-overlap branches before batching. Their sum gives the complete
    // constrained contraction expression for S or F; each branch already carries its weight.
    if w.nbranches() > 1 {
        let zero = T::from_real(0.0);
        let ncol = row.out[0].len();
        let mut parts: [Vec<T>; K] = std::array::from_fn(|_| vec![zero; ncol]);
        for out in row.out.iter_mut() {
            out.fill(zero);
        }

        for k in 0..w.nbranches() {
            let branch = SpinFactorRow {
                target: row.target,
                sources: row.sources,
                target_left: row.target_left,
                out: parts.each_mut().map(|part| part.as_mut_slice()),
            };
            xw_spin_factor_row(factor, &w.branch(k), branch, scratch);

            // Accumulate this branch's S, or its (S,F), in the original source-column order.
            for (out, part) in row.out.iter_mut().zip(&parts) {
                for (value, branch) in out.iter_mut().zip(part) {
                    *value += *branch;
                }
            }
        }

        return;
    }

    // Groups of equal rank and holes share one fixed-rank kernel.
    let sources = row.sources;
    let mut batch = SpinFactorLanes {
        factor,
        w,
        row: &mut row,
        scratch,
    };

    xw_lanes(&mut batch, &sources.caches, &sources.groups);
}

/// Evaluate four independent same-spin Wick matrix elements with AVX2/FMA.
/// Each lane forms `\det\mathbf D_{\text{ov}}(0,\ldots,0)` for S and, for the one-body family,
/// `{}^xF_0^{(0)}\det\mathbf D_{\text{ov}}`
/// `-\sum_z\det\mathbf D_{\text{ov}}^{z\rightarrow\boldsymbol{\mathcal F}_z}` for F.
/// `F::packed` includes `{}^{xw}\tilde S`; the row writer applies `p_x p_w` afterwards.
/// # Arguments:
/// - `batch`: Row holding the target and the Wick intermediates with `m = 0`.
/// - `varying`: Excitation cache of every lane.
/// # Returns
/// - `[[T; K]; 4]`: Factors of every lane, excluding the excitation phases.
/// # Safety
/// - The current CPU must support AVX2 and FMA, and cached labels must match the fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn packet_x4<
    F: SpinFactor<K>,
    T: NOCIScalar,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const XFIX: bool,
>(
    batch: &SpinFactorLanes<'_, '_, F, T, K>,
    varying: &[ExcitationSpinCache; 4],
) -> [[T; K]; 4] {
    let packet = Packet::new(batch.w, &batch.row.target.cache, varying);
    let values = unsafe { F::packed::<T, T::Packed4, 4, RX, RW, L, D, XFIX>(batch.w, &packet) };

    // Transpose `values[k][lane]` into `lanes[lane][k]` for LaneBatch::write. These are
    // independent determinant pairs, each with its own S or (S,F) result.
    let mut lanes = [[T::from_real(0.0); K]; 4];
    for (k, factor) in values.iter().enumerate() {
        for (lane, &value) in factor.iter().enumerate() {
            lanes[lane][k] = value;
        }
    }
    lanes
}

/// Evaluate eight independent same-spin Wick matrix elements with AVX-512F.
/// Each lane evaluates the same S or (S,F) determinant expressions as `packet_x4`, with the
/// reduced reference overlap included and the excitation phases left to the row writer.
/// # Arguments:
/// - `batch`: Row holding the target and the Wick intermediates with `m = 0`.
/// - `varying`: Excitation cache of every lane.
/// # Returns
/// - `[[T; K]; 8]`: Factors of every lane, excluding the excitation phases.
/// # Safety
/// - The current CPU must support AVX-512F, and cached labels must match the fixed ranks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn packet_x8<
    F: SpinFactor<K>,
    T: NOCIScalar,
    const K: usize,
    const RX: usize,
    const RW: usize,
    const L: usize,
    const D: usize,
    const XFIX: bool,
>(
    batch: &SpinFactorLanes<'_, '_, F, T, K>,
    varying: &[ExcitationSpinCache; 8],
) -> [[T; K]; 8] {
    let packet = Packet::new(batch.w, &batch.row.target.cache, varying);
    let values = unsafe { F::packed::<T, T::Packed8, 8, RX, RW, L, D, XFIX>(batch.w, &packet) };

    // Transpose `values[k][lane]` into `lanes[lane][k]` for LaneBatch::write.
    let mut lanes = [[T::from_real(0.0); K]; 8];
    for (k, factor) in values.iter().enumerate() {
        for (lane, &value) in factor.iter().enumerate() {
            lanes[lane][k] = value;
        }
    }
    lanes
}
