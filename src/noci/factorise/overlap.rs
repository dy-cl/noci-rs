// noci/factorise/overlap.rs

// Standard library imports.
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::determinant::ParentComponents;
use crate::input::SNOCIStorage;
use crate::maths::dot_f64;
use crate::noci::overlap::{calculate_s_pair, calculate_s_pair_naive};
use crate::noci::types::{DetPair, NOCIData};
use crate::noci::{
    AuxiliaryDeterminantState, AuxiliaryIndex, AuxiliarySpace, AuxiliarySpinIndex, NOCIIndex,
    ReducedOneSpinNOCIDeterminantState,
};
use crate::nonorthogonalwicks::{
    SameSpinOrthogonalOverlapBatch, SameSpinOverlapBatch, WickScratchSpin, WicksPairView,
    xw_overlap_orthogonal_prepared_batched, xw_overlap_prepared_batched,
};
use crate::{ExcitationSpin, ReducedOneSpinState};

// Parent/sibling imports.
use super::storage::{OverlapFactorStorage, OverlapStoragePlan};
use super::{SpinFactorisation, ordered_parent_pair};

#[derive(Clone, Copy)]
struct SpinUpdate {
    /// Active source a position for this sparse entry.
    apos: usize,
    /// Active source b position for this sparse entry.
    bpos: usize,
    /// `Sparse pre-overlap update value \Delta_\Omega.`
    dn: f64,
}

struct ParentUpdates {
    /// `Source parent P for all sparse D^P_{ab} entries.`
    parent: usize,
    /// `Sparse non-zero entries of D^P_{ab}.`
    entries: Vec<SpinUpdate>,
    /// Retained determinant identities aligned one-to-one with `entries`.
    dets: Vec<usize>,
    /// Active source a component IDs for this application.
    aids: Vec<usize>,
    /// Active source b component IDs for this application.
    bids: Vec<usize>,
    /// Source-parent a ID to active position map.
    apos: Vec<usize>,
    /// Source-parent b ID to active position map.
    bpos: Vec<usize>,
}

struct OrthogonalParentUpdates {
    /// Parent `P` whose MO basis defines every transient determinant.
    parent: usize,
    /// Sparse `\chi^P_{ab}` entries using active source-component positions.
    entries: Vec<SpinUpdate>,
    /// Active alpha numerical payloads resolved from auxiliary components.
    areps: Vec<ReducedOneSpinState>,
    /// Active beta numerical payloads resolved from auxiliary components.
    breps: Vec<ReducedOneSpinState>,
    /// Full alpha excitations used by scalar Wick fallback.
    aex: Vec<ExcitationSpin>,
    /// Full beta excitations used by scalar Wick fallback.
    bex: Vec<ExcitationSpin>,
    /// Same-parent component-pair lookup containing source phase times `\chi_D^P`.
    exact: HashMap<(usize, usize), f64>,
    /// Canonical parent-local alpha component IDs for active sources.
    aids: Vec<usize>,
    /// Canonical parent-local beta component IDs for active sources.
    bids: Vec<usize>,
    /// Canonical alpha component ID to active source position.
    apos: Vec<usize>,
    /// Canonical beta component ID to active source position.
    bpos: Vec<usize>,
}

/// Borrowed sparse numerical source used by blocked overlap contractions.
#[derive(Clone, Copy)]
struct FactorisedSource<'a> {
    /// Sparse amplitudes and active spin-component positions.
    entries: &'a [SpinUpdate],
    /// Number of active alpha source components.
    nalpha: usize,
    /// Number of active beta source components.
    nbeta: usize,
}

#[derive(Clone, Copy)]
struct FactorisedFactors<'a> {
    /// Row-major alpha overlap factors.
    alpha: &'a [f64],
    /// Row-major beta overlap factors.
    beta: &'a [f64],
    /// Number of source columns in the alpha factor table.
    alpha_stride: usize,
    /// Number of source columns in the beta factor table.
    beta_stride: usize,
    /// Optional persistent target alpha component IDs.
    target_alpha_ids: Option<&'a [usize]>,
    /// Optional persistent target beta component IDs.
    target_beta_ids: Option<&'a [usize]>,
    /// Optional persistent source alpha component IDs.
    source_alpha_ids: Option<&'a [usize]>,
    /// Optional persistent source beta component IDs.
    source_beta_ids: Option<&'a [usize]>,
}

impl FactorisedFactors<'_> {
    /// Return `A^{QP}_{\bar a a}` for active target and source positions.
    /// # Arguments:
    /// - `self`: Factor-table view.
    /// - `target`: Active target alpha position.
    /// - `source`: Active source alpha position.
    /// # Returns
    /// - `f64`: Selected alpha overlap factor.
    #[inline(always)]
    fn alpha(
        &self,
        target: usize,
        source: usize,
    ) -> f64 {
        let target = self.target_alpha_ids.map_or(target, |ids| ids[target]);
        let source = self.source_alpha_ids.map_or(source, |ids| ids[source]);
        self.alpha[target * self.alpha_stride + source]
    }

    /// Return `B^{QP}_{\bar b b}` for active target and source positions.
    /// # Arguments:
    /// - `self`: Factor-table view.
    /// - `target`: Active target beta position.
    /// - `source`: Active source beta position.
    /// # Returns
    /// - `f64`: Selected beta overlap factor.
    #[inline(always)]
    fn beta(
        &self,
        target: usize,
        source: usize,
    ) -> f64 {
        let target = self.target_beta_ids.map_or(target, |ids| ids[target]);
        let source = self.source_beta_ids.map_or(source, |ids| ids[source]);
        self.beta[target * self.beta_stride + source]
    }

    /// Contract one active beta row with a contiguous intermediate vector.
    /// # Arguments:
    /// - `self`: Factor-table view.
    /// - `target`: Active target beta position.
    /// - `values`: Intermediate vector indexed by active source position.
    /// # Returns
    /// - `f64`: Dot product `sum_b B^{QP}_{\bar b b}U_b`.
    #[inline(always)]
    fn dot_beta(
        &self,
        target: usize,
        values: &[f64],
    ) -> f64 {
        if self.source_beta_ids.is_none() && self.target_beta_ids.is_none() {
            return dot_f64(
                values,
                &self.beta[target * self.beta_stride..(target + 1) * self.beta_stride],
            );
        }
        (0..values.len())
            .map(|source| values[source] * self.beta(target, source))
            .sum()
    }

    /// Contract one active alpha row with a contiguous intermediate vector.
    /// # Arguments:
    /// - `self`: Factor-table view.
    /// - `target`: Active target alpha position.
    /// - `values`: Intermediate vector indexed by active source position.
    /// # Returns
    /// - `f64`: Dot product `sum_a A^{QP}_{\bar a a}T_a`.
    #[inline(always)]
    fn dot_alpha(
        &self,
        target: usize,
        values: &[f64],
    ) -> f64 {
        if self.source_alpha_ids.is_none() && self.target_alpha_ids.is_none() {
            return dot_f64(
                &self.alpha[target * self.alpha_stride..(target + 1) * self.alpha_stride],
                values,
            );
        }
        (0..values.len())
            .map(|source| self.alpha(target, source) * values[source])
            .sum()
    }
}

#[derive(Clone, Copy)]
struct LocalTarget {
    /// `Rank-local population row receiving \delta N_w.`
    local: usize,
    /// Global determinant index w.
    det: usize,
    /// `Target-parent local a component ID a_w.`
    a: usize,
    /// `Target-parent local b component ID b_w.`
    b: usize,
}

#[derive(Clone, Copy)]
struct OrthogonalTarget {
    /// Rank-local population row for an orthogonal same-parent target.
    local: usize,
    /// Product of target determinant spin phases.
    phase: f64,
}

struct OrthogonalTargetGroup {
    /// Targets sharing this occupation pair.
    targets: Vec<OrthogonalTarget>,
}

struct LocalParentBlock {
    /// Target parent Q for all local rows in this block.
    parent: usize,
    /// Rank-local target rows in this parent block.
    targets: Vec<LocalTarget>,
    /// First rank-local row when target rows are contiguous.
    first_local: usize,
    /// Whether target local rows are consecutive in `populations`.
    contiguous_locals: bool,
    /// Active target a component IDs.
    aids: Vec<usize>,
    /// Active target b component IDs.
    bids: Vec<usize>,
    /// Target-parent a ID to active position map.
    apos: Vec<usize>,
    /// Target-parent b ID to active position map.
    bpos: Vec<usize>,
    /// Same-parent orthogonal occupation groups.
    orthogonal: Vec<OrthogonalTargetGroup>,
    /// Parent-local component pair to target same-parent group position.
    opos: HashMap<(usize, usize), usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OverlapContraction {
    /// Factorise each target row before looping over sparse source updates.
    FactorisedRows,
    /// `Form T_{\bar a b} before applying B^{QP}_{\bar b_w b}.`
    AFirst,
    /// `Form U_{a\bar b} before applying A^{QP}_{\bar a_w a}.`
    BFirst,
}

/// Persistent same-spin overlap factors for one ordered cross-parent block `QP`.
pub(crate) struct OverlapFactorBlock {
    /// Number of target alpha-spin components.
    pub(super) nta: usize,
    /// Number of target beta-spin components.
    pub(super) ntb: usize,
    /// Number of source alpha-spin components.
    pub(super) nsa: usize,
    /// Number of source beta-spin components.
    pub(super) nsb: usize,
    /// Initial retained alpha factor-column count.
    retained_nsa: usize,
    /// Initial retained beta factor-column count.
    retained_nsb: usize,
    /// Raw overlap factor and optional proposal-CDF backing.
    factors: OverlapFactorStorage,
    /// Persistent alpha factor-column storage indexed by physical auxiliary component ID.
    alpha_columns: Vec<usize>,
    /// Persistent beta factor-column storage indexed by physical auxiliary component ID.
    beta_columns: Vec<usize>,
}

/// Persistent cross-parent overlap factors indexed by ordered parent pair `QP`.
pub(crate) struct OverlapFactors {
    /// Cross-parent factor blocks indexed as `Q * nparent + P`.
    blocks: Vec<Option<OverlapFactorBlock>>,
}

/// `Reusable storage for one application of S\Delta.`
pub(crate) struct OverlapScratch {
    /// Sparse updates grouped by source parent.
    updates: Vec<ParentUpdates>,
    /// Source parents touched by the current update list.
    active_parents: Vec<usize>,
    /// `Temporary A^{QP}_{\bar a a} factor table.`
    afac: Vec<f64>,
    /// `Temporary B^{QP}_{\bar b b} factor table.`
    bfac: Vec<f64>,
    /// Temporary blocked contraction table T or U.
    intermediate: Vec<f64>,
    /// Temporary per-target output values for one parent block.
    values: Vec<f64>,
    /// Cached target slice pointer for validating reusable target blocks.
    cached_targets_ptr: *const usize,
    /// Cached target slice length for validating reusable target blocks.
    cached_targets_len: usize,
    /// Reusable target parent blocks for a fixed rank-local target list.
    target_blocks: Vec<LocalParentBlock>,
}

/// Reusable BApply storage for exact sparse `B^\dagger` application.
pub(crate) struct AuxiliaryOverlapScratch {
    /// Transient orthogonal sources grouped by defining parent.
    updates: Vec<OrthogonalParentUpdates>,
    /// Existing retained overlap contraction scratch reused after source-factor construction.
    overlap: OverlapScratch,
}

impl OverlapFactors {
    /// Count additional persistent factor columns materialised for BApply sources.
    /// # Arguments:
    /// - `self`: Persistent cross-parent overlap factors.
    /// # Returns:
    /// - `(usize, usize)`: Used additional alpha and beta columns across factor blocks.
    pub(crate) fn added_components(&self) -> (usize, usize) {
        self.blocks
            .iter()
            .flatten()
            .fold((0, 0), |(alpha, beta), block| {
                (
                    alpha + block.nsa - block.retained_nsa,
                    beta + block.nsb - block.retained_nsb,
                )
            })
    }
    /// Measure the actual overlap-factor and proposal-CDF backing.
    /// # Arguments:
    /// - `self`: Persistent cross-parent factor blocks.
    /// # Returns:
    /// - `(usize, usize)`: Factor and CDF backing bytes, counting each block once.
    pub(crate) fn storage_bytes(&self) -> (usize, usize) {
        self.blocks
            .iter()
            .flatten()
            .fold((0, 0), |(tables, cdfs), block| {
                let (alpha, beta, acdf, bcdf) = block.factors.factors();
                (
                    tables + (alpha.len() + beta.len()) * std::mem::size_of::<f64>(),
                    cdfs + (acdf.len() + bcdf.len()) * std::mem::size_of::<f64>(),
                )
            })
    }

    /// Return the persistent factor block for ordered parent pair `QP`.
    /// # Arguments:
    /// - `self`: Persistent overlap factors.
    /// - `nparent`: Number of parent references in the spin factorisation.
    /// - `target_parent`: Target parent `Q`.
    /// - `source_parent`: Source parent `P`.
    /// # Returns:
    /// - `Option<&OverlapFactorBlock>`: Cross-parent factor block when present.
    pub(crate) fn block(
        &self,
        nparent: usize,
        target_parent: usize,
        source_parent: usize,
    ) -> Option<&OverlapFactorBlock> {
        self.blocks[target_parent * nparent + source_parent].as_ref()
    }

    /// Return the mutable persistent factor block for ordered parent pair `QP`.
    /// # Arguments:
    /// - `self`: Persistent overlap factors.
    /// - `nparent`: Number of parent references in the spin factorisation.
    /// - `target_parent`: Target parent `Q`.
    /// - `source_parent`: Source parent `P`.
    /// # Returns
    /// - `Option<&mut OverlapFactorBlock>`: Mutable factor block when present.
    pub(crate) fn block_mut(
        &mut self,
        nparent: usize,
        target_parent: usize,
        source_parent: usize,
    ) -> Option<&mut OverlapFactorBlock> {
        self.blocks[target_parent * nparent + source_parent].as_mut()
    }
}

impl OverlapFactorBlock {
    /// Return `Z_A^{QP}(a_x)` from the source-major alpha CDF.
    /// # Arguments:
    /// - `self`: Ordered parent-pair factor block.
    /// - `source_a`: Source alpha component `a_x`.
    /// # Returns:
    /// - `f64`: Absolute alpha-column sum.
    pub(crate) fn alpha_total(
        &self,
        source_a: usize,
    ) -> f64 {
        let (_, _, acdf, _) = self.factors.factors();
        if acdf.is_empty() {
            0.0
        } else {
            acdf[source_a * self.nta + self.nta - 1]
        }
    }

    /// Return `Z_B^{QP}(b_x)` from the source-major beta CDF.
    /// # Arguments:
    /// - `self`: Ordered parent-pair factor block.
    /// - `source_b`: Source beta component `b_x`.
    /// # Returns:
    /// - `f64`: Absolute beta-column sum.
    pub(crate) fn beta_total(
        &self,
        source_b: usize,
    ) -> f64 {
        let (_, _, _, bcdf) = self.factors.factors();
        if bcdf.is_empty() {
            0.0
        } else {
            bcdf[source_b * self.ntb + self.ntb - 1]
        }
    }

    /// Return `|A^{QP}_{a_w a_x}B^{QP}_{b_w b_x}|`.
    /// # Arguments:
    /// - `self`: Ordered parent-pair factor block.
    /// - `target_a`: Target alpha component `a_w`.
    /// - `target_b`: Target beta component `b_w`.
    /// - `source_a`: Source alpha component `a_x`.
    /// - `source_b`: Source beta component `b_x`.
    /// # Returns:
    /// - `f64`: Absolute factorised determinant overlap.
    pub(crate) fn factor_abs(
        &self,
        target_a: usize,
        target_b: usize,
        source_a: usize,
        source_b: usize,
    ) -> f64 {
        let (afac, bfac, _, _) = self.factors.factors();
        (afac[target_a * self.nsa + source_a] * bfac[target_b * self.nsb + source_b]).abs()
    }

    /// Sample a target alpha component from one source-major CDF row.
    /// # Arguments:
    /// - `self`: Ordered parent-pair factor block.
    /// - `source_a`: Source alpha component `a_x`.
    /// - `draw`: Uniform draw in `[0,Z_A)`.
    /// # Returns:
    /// - `usize`: Sampled target alpha component.
    pub(crate) fn sample_alpha(
        &self,
        source_a: usize,
        draw: f64,
    ) -> usize {
        let (_, _, acdf, _) = self.factors.factors();
        let row = &acdf[source_a * self.nta..(source_a + 1) * self.nta];
        row.partition_point(|&value| value <= draw)
            .min(self.nta - 1)
    }

    /// Sample a target beta component from one source-major CDF row.
    /// # Arguments:
    /// - `self`: Ordered parent-pair factor block.
    /// - `source_b`: Source beta component `b_x`.
    /// - `draw`: Uniform draw in `[0,Z_B)`.
    /// # Returns:
    /// - `usize`: Sampled target beta component.
    pub(crate) fn sample_beta(
        &self,
        source_b: usize,
        draw: f64,
    ) -> usize {
        let (_, _, _, bcdf) = self.factors.factors();
        let row = &bcdf[source_b * self.ntb..(source_b + 1) * self.ntb];
        row.partition_point(|&value| value <= draw)
            .min(self.ntb - 1)
    }
}

impl SpinFactorisation {
    /// Construct persistent cross-parent factors for factorised overlap application.
    /// The factor tables store `S_{wx} = A^{QP}_{a_w a_x}B^{QP}_{b_w b_x}` inputs once for
    /// every ordered cross-parent block, and optionally build source-major CDFs for excitation
    /// generation.
    /// # Arguments:
    /// - `self`: Immutable sparse overlap action plan.
    /// - `data`: Shared NOCI data containing fixed Wick intermediates.
    /// - `cache`: Directory for persistent file-backed factor blocks.
    /// - `rank`: MPI rank used in factor-cache filenames.
    /// - `storage`: Requested persistent factor-table storage backend.
    /// - `build_cdfs`: Whether to build overlap-weighted proposal CDFs.
    /// # Returns:
    /// - `OverlapFactors`: Persistent cross-parent factor tables, or transient markers for `none`.
    pub(crate) fn build_overlap_factors(
        &self,
        data: &NOCIData<'_, f64>,
        cache: &Path,
        rank: i32,
        storage: SNOCIStorage,
        build_cdfs: bool,
    ) -> OverlapFactors {
        let nparent = self.parents.len();
        let mut factor_blocks = (0..nparent * nparent).map(|_| None).collect::<Vec<_>>();
        let mut storage_plan = OverlapStoragePlan::new(cache, rank, storage);
        if !matches!(storage, SNOCIStorage::None)
            && data.input.wicks.enabled
            && let Some(wicks) = data.wicks
        {
            for target_parent in 0..nparent {
                let target = &self.parents[target_parent];
                if target.entries.is_empty() {
                    continue;
                }

                for source_parent in 0..nparent {
                    if target_parent == source_parent {
                        continue;
                    }

                    let source = &self.parents[source_parent];
                    if source.entries.is_empty() {
                        continue;
                    }

                    let nta = target.areps.len();
                    let ntb = target.breps.len();
                    let nsa = source.areps.len();
                    let nsb = source.breps.len();
                    let (lp, gp, target_left) =
                        ordered_parent_pair(self, target_parent, source_parent);
                    let pair = wicks.pair(lp, gp);
                    let na = nta
                        .checked_mul(nsa)
                        .expect("alpha overlap factor length overflow");
                    let nb = ntb
                        .checked_mul(nsb)
                        .expect("beta overlap factor length overflow");
                    let mut factors =
                        storage_plan.allocate(target_parent, source_parent, na, nb, build_cdfs);

                    {
                        let (afac, _, _, _) = factors.factors_mut();
                        build_spin_overlap_factors(
                            &pair,
                            data,
                            (target.areps.as_slice(), source.areps.as_slice()),
                            0..nta,
                            target_left,
                            true,
                            afac,
                        );
                    }
                    factors.flush();

                    {
                        let (_, bfac, _, _) = factors.factors_mut();
                        build_spin_overlap_factors(
                            &pair,
                            data,
                            (target.breps.as_slice(), source.breps.as_slice()),
                            0..ntb,
                            target_left,
                            false,
                            bfac,
                        );
                    }
                    factors.flush();

                    if build_cdfs {
                        let (afac, bfac, acdf, bcdf) = factors.factors_mut();
                        for sa in 0..nsa {
                            let mut sum = 0.0;
                            for ta in 0..nta {
                                sum += afac[ta * nsa + sa].abs();
                                acdf[sa * nta + ta] = sum;
                            }
                        }

                        for sb in 0..nsb {
                            let mut sum = 0.0;
                            for tb in 0..ntb {
                                sum += bfac[tb * nsb + sb].abs();
                                bcdf[sb * ntb + tb] = sum;
                            }
                        }
                    }
                    factors.flush();

                    factor_blocks[target_parent * nparent + source_parent] =
                        Some(OverlapFactorBlock {
                            nta,
                            ntb,
                            nsa,
                            nsb,
                            retained_nsa: nsa,
                            retained_nsb: nsb,
                            factors,
                            alpha_columns: Vec::new(),
                            beta_columns: Vec::new(),
                        });
                }
            }
        }

        OverlapFactors {
            blocks: factor_blocks,
        }
    }

    /// Intern physical source-spin occupations and append their `B^{\dagger}` factor columns.
    /// For persistent `ram` or `disk` storage, each new occupation is evaluated once and then
    /// addressed by its stable source component ID; `none` leaves construction to the transient
    /// active-factor path.
    /// # Arguments:
    /// - `self`: Shared retained determinant spin factorisation.
    /// - `factors`: Persistent overlap-factor blocks selected by `SNOCIStorage`.
    /// - `scratch`: BApply occupation interning and transient source metadata.
    /// - `target_parent`: Target parent `Q`.
    /// - `source`: Orthogonal source block belonging to parent `P`.
    /// - `context`: Shared NOCI data, ordered Wick pair, and left-reference flag.
    /// # Returns
    /// - `Option<&OverlapFactorBlock>`: Persistent factor block after interning, when available.
    fn intern_orthogonal_source_columns<'a>(
        &self,
        factors: &'a mut OverlapFactors,
        auxiliary: &AuxiliarySpace,
        target_parent: usize,
        source: &OrthogonalParentUpdates,
        context: (&NOCIData<'_, f64>, &WicksPairView<'_, f64>, bool),
    ) -> Option<&'a OverlapFactorBlock> {
        let (data, pair, target_left) = context;
        let source_parent = source.parent;
        let nparent = self.parents.len();
        let block = factors.block_mut(nparent, target_parent, source_parent)?;
        let target_areps = self.parents[target_parent].areps.as_slice();
        let target_breps = self.parents[target_parent].breps.as_slice();
        let components = auxiliary.parent_components(source_parent);

        if block.alpha_columns.is_empty() {
            block.alpha_columns.resize(components.na(), usize::MAX);
            for id in 0..block.retained_nsa {
                block.alpha_columns[id] = id;
            }
        }
        if block.beta_columns.is_empty() {
            block.beta_columns.resize(components.nb(), usize::MAX);
            for id in 0..block.retained_nsb {
                block.beta_columns[id] = id;
            }
        }

        let mut missing_alpha = source
            .aids
            .iter()
            .copied()
            .filter(|&id| block.alpha_columns[id] == usize::MAX)
            .collect::<Vec<_>>();
        missing_alpha.sort_unstable();
        let mut alpha_columns = Vec::with_capacity(missing_alpha.len());
        if !missing_alpha.is_empty() {
            let source_reps = missing_alpha
                .iter()
                .map(|&id| components.alpha(AuxiliarySpinIndex(id)).reduced)
                .collect::<Vec<_>>();
            let source_excitations = missing_alpha
                .iter()
                .map(|&id| components.alpha(AuxiliarySpinIndex(id)).excitation)
                .collect::<Vec<_>>();
            let mut table = vec![0.0; target_areps.len() * source_reps.len()];
            table
                .par_chunks_mut(source_reps.len())
                .zip(target_areps.par_iter())
                .for_each_init(WickScratchSpin::new, |wick, (row, &target_rep)| {
                    xw_overlap_orthogonal_prepared_batched(
                        &pair.aa,
                        SameSpinOrthogonalOverlapBatch {
                            basis: data.space,
                            target: target_rep,
                            sources: &source_reps,
                            source_excitations: &source_excitations,
                            target_left,
                            alpha: true,
                            out: row,
                        },
                        &mut wick.aa,
                    );
                });
            for column in 0..source_reps.len() {
                alpha_columns.push(
                    (0..target_areps.len())
                        .map(|row| table[row * source_reps.len() + column])
                        .collect(),
                );
            }
        }

        let mut missing_beta = source
            .bids
            .iter()
            .copied()
            .filter(|&id| block.beta_columns[id] == usize::MAX)
            .collect::<Vec<_>>();
        missing_beta.sort_unstable();
        let mut beta_columns = Vec::with_capacity(missing_beta.len());
        if !missing_beta.is_empty() {
            let source_reps = missing_beta
                .iter()
                .map(|&id| components.beta(AuxiliarySpinIndex(id)).reduced)
                .collect::<Vec<_>>();
            let source_excitations = missing_beta
                .iter()
                .map(|&id| components.beta(AuxiliarySpinIndex(id)).excitation)
                .collect::<Vec<_>>();
            let mut table = vec![0.0; target_breps.len() * source_reps.len()];
            table
                .par_chunks_mut(source_reps.len())
                .zip(target_breps.par_iter())
                .for_each_init(WickScratchSpin::new, |wick, (row, &target_rep)| {
                    xw_overlap_orthogonal_prepared_batched(
                        &pair.bb,
                        SameSpinOrthogonalOverlapBatch {
                            basis: data.space,
                            target: target_rep,
                            sources: &source_reps,
                            source_excitations: &source_excitations,
                            target_left,
                            alpha: false,
                            out: row,
                        },
                        &mut wick.bb,
                    );
                });
            for column in 0..source_reps.len() {
                beta_columns.push(
                    (0..target_breps.len())
                        .map(|row| table[row * source_reps.len() + column])
                        .collect(),
                );
            }
        }

        if !alpha_columns.is_empty() || !beta_columns.is_empty() {
            block.factors.append_source_columns(
                block.nta,
                block.ntb,
                block.nsa,
                block.nsb,
                &alpha_columns,
                &beta_columns,
            );
            block.factors.flush();
            for (column, &id) in missing_alpha.iter().enumerate() {
                block.alpha_columns[id] = block.nsa + column;
            }
            for (column, &id) in missing_beta.iter().enumerate() {
                block.beta_columns[id] = block.nsb + column;
            }
            block.nsa += alpha_columns.len();
            block.nsb += beta_columns.len();
        }

        Some(block)
    }

    /// `Construct reusable storage for one full application of S\Delta.`
    /// The scratch contains only mutable grouping and contraction workspaces; persistent
    /// factor tables live in `OverlapFactors`.
    /// # Arguments:
    /// - `self`: Immutable sparse overlap action plan.
    /// # Returns:
    /// - `OverlapScratch`: Reusable grouped-update and contraction storage.
    pub(crate) fn overlap_scratch(&self) -> OverlapScratch {
        let nparent = self.parents.len();
        let mut updates = Vec::with_capacity(nparent);
        for parent in 0..nparent {
            updates.push(ParentUpdates::new(parent, self.ma, self.mb));
        }

        OverlapScratch {
            updates,
            active_parents: Vec::new(),
            afac: Vec::new(),
            bfac: Vec::new(),
            intermediate: Vec::new(),
            values: Vec::new(),
            cached_targets_ptr: std::ptr::null(),
            cached_targets_len: 0,
            target_blocks: Vec::new(),
        }
    }

    /// Construct reusable storage for exact sparse `B^\dagger` application.
    /// Parent occupations are reconstructed once and retained with BApply-only transient metadata.
    /// # Arguments:
    /// - `self`: Immutable sparse overlap action plan.
    /// - `data`: Shared NOCI data defining parent-relative excitations.
    /// # Returns:
    /// - `OrthogonalOverlapScratch`: Empty reusable orthogonal-source contraction storage.
    pub(crate) fn auxiliary_overlap_scratch(&self) -> AuxiliaryOverlapScratch {
        let updates = (0..self.parents.len())
            .map(OrthogonalParentUpdates::new)
            .collect();
        AuxiliaryOverlapScratch {
            updates,
            overlap: self.overlap_scratch(),
        }
    }

    /// Apply an exact sparse physical-to-NOCI overlap transformation.
    /// `\delta N_w = \sum_{P,D}\langle\Phi_w|D^P\rangle\chi_D^P = (B^\dagger\chi)_w`.
    /// # Arguments:
    /// - `populations`: Rank-local persistent range populations receiving the update.
    /// - `targets`: Global retained-NOCI determinant index for each local population row.
    /// - `updates`: Complete realised orthogonal-space residual after report-level FRI.
    /// - `spaces`: Retained matrix data and physical auxiliary topology.
    /// - `factors`: Persistent overlap storage selected by `SNOCIStorage`.
    /// - `scratch`: Reusable BApply source metadata and overlap contraction storage.
    /// # Returns:
    /// - `()`: Applies the complete `B^\dagger\chi` update without target-row sampling.
    pub(crate) fn apply_orthogonal_overlap_sparse<I>(
        &self,
        populations: &mut [f64],
        targets: &[usize],
        updates: I,
        spaces: (&NOCIData<'_, f64>, &AuxiliarySpace),
        factors: &mut OverlapFactors,
        scratch: &mut AuxiliaryOverlapScratch,
    ) where
        I: IntoIterator<Item = (AuxiliaryIndex, f64)>,
    {
        let (data, auxiliary) = spaces;

        if populations.is_empty() {
            return;
        }

        for source in &mut scratch.updates {
            source.clear();
        }
        for (det, dn) in updates {
            if dn != 0.0 {
                let state = auxiliary.state(det);
                let components = auxiliary.parent_components(state.parent);

                scratch.updates[state.parent].push(state, dn, components);
            }
        }

        let target_blocks = self.take_overlap_target_blocks(targets, data, &mut scratch.overlap);
        // Apply the complete realised orthogonal-space residual:
        // `\delta N_w = \sum_{P,D}<\Phi_w|D^P>\chi_D^P = (B^\dagger\chi)_w`.
        // Hence every realised update obeys
        // `\delta N \in range(B^\dagger) = range(S)`.
        for parent in 0..scratch.updates.len() {
            let mut source = std::mem::replace(
                &mut scratch.updates[parent],
                OrthogonalParentUpdates::new(parent),
            );
            if source.entries.is_empty() {
                scratch.updates[parent] = source;
                continue;
            }

            for target in &target_blocks {
                if target.parent == source.parent {
                    Self::apply_orthogonal_source_exact(populations, target, &source, data);
                    continue;
                }

                let wicks = data
                    .wicks
                    .expect("BApply cross-parent overlap requires Wick intermediates");
                let (lp, gp, target_left) = ordered_parent_pair(self, target.parent, source.parent);
                let pair = wicks.pair(lp, gp);

                let persistent = self.intern_orthogonal_source_columns(
                    factors,
                    auxiliary,
                    target.parent,
                    &source,
                    (data, &pair, target_left),
                );
                if let Some(block) = persistent {
                    let (alpha, beta, _, _) = block.factors.factors();
                    let factorised = source.factorised_source();
                    let source_alpha_columns = source
                        .aids
                        .iter()
                        .map(|&id| block.alpha_columns[id])
                        .collect::<Vec<_>>();
                    let source_beta_columns = source
                        .bids
                        .iter()
                        .map(|&id| block.beta_columns[id])
                        .collect::<Vec<_>>();
                    let factor_tables = FactorisedFactors {
                        alpha,
                        beta,
                        alpha_stride: alpha.len() / block.nta,
                        beta_stride: beta.len() / block.ntb,
                        target_alpha_ids: Some(target.aids.as_slice()),
                        target_beta_ids: Some(target.bids.as_slice()),
                        source_alpha_ids: Some(source_alpha_columns.as_slice()),
                        source_beta_ids: Some(source_beta_columns.as_slice()),
                    };
                    // For `|Phi_w^Q> = |Phi^Q_{\\bar a\\bar b}>` and one transient orthogonal source
                    // `|D^P_{ab}>`, factorise
                    // `<Phi_w^Q|D^P_{ab}> = A^{QP}_{\\bar a a} B^{QP}_{\\bar b b}`.
                    // Only source-factor construction differs from retained SApply; the blocked
                    // contraction is shared.
                    match self.select_overlap_contraction(target, factorised) {
                        OverlapContraction::FactorisedRows => {
                            Self::apply_overlap_factorised_rows_tables(
                                populations,
                                target,
                                factorised,
                                factor_tables,
                                &mut scratch.overlap.values,
                            );
                        }
                        OverlapContraction::AFirst => {
                            Self::gather_overlap_factor_tables(
                                target,
                                &source_alpha_columns,
                                &source_beta_columns,
                                block,
                                &mut scratch.overlap.afac,
                                &mut scratch.overlap.bfac,
                            );
                            self.apply_overlap_a_first_scratch(
                                populations,
                                target,
                                factorised,
                                &mut scratch.overlap,
                            );
                        }
                        OverlapContraction::BFirst => {
                            Self::gather_overlap_factor_tables(
                                target,
                                &source_alpha_columns,
                                &source_beta_columns,
                                block,
                                &mut scratch.overlap.afac,
                                &mut scratch.overlap.bfac,
                            );
                            self.apply_overlap_b_first_scratch(
                                populations,
                                target,
                                factorised,
                                &mut scratch.overlap,
                            );
                        }
                    }
                } else {
                    let factorised = source.factorised_source();
                    self.build_orthogonal_overlap_factor_tables(
                        target,
                        &source,
                        data,
                        &pair,
                        target_left,
                        &mut scratch.overlap,
                    );
                    match self.select_overlap_contraction(target, factorised) {
                        OverlapContraction::FactorisedRows => {
                            Self::apply_overlap_factorised_rows_scratch(
                                populations,
                                target,
                                factorised,
                                &mut scratch.overlap,
                            );
                        }
                        OverlapContraction::AFirst => {
                            self.apply_overlap_a_first_scratch(
                                populations,
                                target,
                                factorised,
                                &mut scratch.overlap,
                            );
                        }
                        OverlapContraction::BFirst => {
                            self.apply_overlap_b_first_scratch(
                                populations,
                                target,
                                factorised,
                                &mut scratch.overlap,
                            );
                        }
                    }
                }
            }
            source.clear();
            scratch.updates[parent] = source;
        }

        scratch.overlap.target_blocks = target_blocks;
        scratch.overlap.afac.clear();
        scratch.overlap.bfac.clear();
        scratch.overlap.intermediate.clear();
        scratch.overlap.values.clear();
    }

    /// `Apply \delta N_w = \sum_\Omega S_{w\Omega}\Delta_\Omega.`
    /// Orthogonal same-parent blocks are applied directly, while cross-parent blocks use
    /// `S_{w\Omega} = A^{QP}_{\bar a_w a_\Omega}B^{QP}_{\bar b_w b_\Omega}.`
    /// Cross-parent same-spin factors are cached for `ram` and `disk`, and regenerated from
    /// active spin components for `none`.
    /// # Arguments:
    /// - `populations`: `Rank-local persistent populations N_w.`
    /// - `targets`: Global determinant index for each rank-local row in `populations`.
    /// - `updates`: `Sparse pre-overlap changes \Omega, \Delta_\Omega.`
    /// - `data`: Shared NOCI data.
    /// - `factors`: Persistent cross-parent same-spin overlap factors.
    /// - `scratch`: `Reusable allocation storage for one application of S\Delta.`
    /// # Returns:
    /// - `()`: `Applies N_w \leftarrow N_w + \delta N_w.`
    pub(crate) fn apply_overlap_sparse<I>(
        &self,
        populations: &mut [f64],
        targets: &[usize],
        updates: I,
        data: &NOCIData<'_, f64>,
        factors: &OverlapFactors,
        scratch: &mut OverlapScratch,
    ) where
        I: IntoIterator<Item = (usize, f64)>,
    {
        if populations.is_empty() {
            return;
        }

        self.group_overlap_updates(updates, data, scratch);
        if scratch.active_parents.is_empty() {
            return;
        }

        let target_blocks = self.take_overlap_target_blocks(targets, data, scratch);

        for source_parent in scratch.active_parents.clone() {
            let mut source = std::mem::replace(
                &mut scratch.updates[source_parent],
                ParentUpdates::empty(source_parent),
            );
            if source.entries.is_empty() {
                scratch.updates[source_parent] = source;
                continue;
            }
            for target in &target_blocks {
                self.apply_overlap_parent_pair(
                    populations,
                    target,
                    &source,
                    data,
                    factors,
                    scratch,
                );
            }
            source.clear();
            scratch.updates[source_parent] = source;
        }

        scratch.target_blocks = target_blocks;

        scratch.active_parents.clear();
        scratch.afac.clear();
        scratch.bfac.clear();
        scratch.intermediate.clear();
        scratch.values.clear();
    }

    /// Take reusable target blocks for the current rank-local rows.
    /// The blocks contain only determinant IDs and spin-component topology, not overlap factors,
    /// so reusing them avoids rebuilding fixed QMC target metadata without caching matrix elements.
    /// # Arguments:
    /// - `targets`: Global determinant index for each rank-local population row.
    /// - `data`: Shared NOCI data used when a rebuild is required.
    /// - `scratch`: Reusable overlap storage owning the cached blocks.
    /// # Returns:
    /// - `Vec<LocalParentBlock>`: Target blocks moved out of scratch for this application.
    fn take_overlap_target_blocks(
        &self,
        targets: &[usize],
        data: &NOCIData<'_, f64>,
        scratch: &mut OverlapScratch,
    ) -> Vec<LocalParentBlock> {
        if scratch.target_blocks.is_empty()
            || scratch.cached_targets_ptr != targets.as_ptr()
            || scratch.cached_targets_len != targets.len()
        {
            scratch.cached_targets_ptr = targets.as_ptr();
            scratch.cached_targets_len = targets.len();
            self.build_overlap_target_blocks(targets, data)
        } else {
            std::mem::take(&mut scratch.target_blocks)
        }
    }

    /// Group sparse updates by source parent and active spin components.
    /// `This constructs D^P_{ab} in sparse form for the current S\Delta application.`
    /// # Arguments:
    /// - `updates`: `Sparse determinant changes \Omega, \Delta_\Omega.`
    /// - `data`: Shared NOCI data used to map determinants to parents.
    /// - `scratch`: Reusable grouped-update storage cleared and refilled for this application.
    /// # Returns:
    /// - `()`: Fills `scratch.updates` and `scratch.active_parents`.
    fn group_overlap_updates<I>(
        &self,
        updates: I,
        data: &NOCIData<'_, f64>,
        scratch: &mut OverlapScratch,
    ) where
        I: IntoIterator<Item = (usize, f64)>,
    {
        for &parent in &scratch.active_parents {
            scratch.updates[parent].clear();
        }
        scratch.active_parents.clear();

        for (det, dn) in updates {
            if dn == 0.0 {
                continue;
            }

            let state = data.space.state(NOCIIndex(det));
            let parent = state.parent;
            if scratch.updates[parent].entries.is_empty() {
                scratch.active_parents.push(parent);
            }
            scratch.updates[parent].push(det, state.aid.0, state.bid.0, dn);
        }
    }

    /// `Build target parent blocks for the rank-local rows receiving S\Delta.`
    /// Each block records active target spin components and same-parent occupation groups.
    /// # Arguments:
    /// - `targets`: Global determinant index for each rank-local population row.
    /// - `data`: Shared NOCI data used to read determinant parents and occupations.
    /// # Returns:
    /// - `Vec<LocalParentBlock>`: Non-empty target blocks grouped by parent Q.
    fn build_overlap_target_blocks(
        &self,
        targets: &[usize],
        data: &NOCIData<'_, f64>,
    ) -> Vec<LocalParentBlock> {
        let mut blocks = (0..self.parents.len())
            .map(|parent| LocalParentBlock {
                parent,
                targets: Vec::new(),
                first_local: 0,
                contiguous_locals: true,
                aids: Vec::new(),
                bids: Vec::new(),
                apos: vec![usize::MAX; self.parents[parent].areps.len()],
                bpos: vec![usize::MAX; self.parents[parent].breps.len()],
                orthogonal: Vec::new(),
                opos: HashMap::new(),
            })
            .collect::<Vec<_>>();

        for (local, &det) in targets.iter().enumerate() {
            let state = data.space.state(NOCIIndex(det));
            let parent = state.parent;
            let a = state.aid.0;
            let b = state.bid.0;
            let block = &mut blocks[parent];

            if block.targets.is_empty() {
                block.first_local = local;
            } else if block.contiguous_locals && local != block.first_local + block.targets.len() {
                block.contiguous_locals = false;
            }

            // Add the a component to the active set on its first occurrence.
            if block.apos[a] == usize::MAX {
                block.apos[a] = block.aids.len();
                block.aids.push(a);
            }
            // Add the b component to the active set on its first occurrence.
            if block.bpos[b] == usize::MAX {
                block.bpos[b] = block.bids.len();
                block.bids.push(b);
            }

            block.targets.push(LocalTarget { local, det, a, b });
        }

        for block in &mut blocks {
            self.build_overlap_orthogonal_groups(block, data);
        }

        blocks.retain(|block| !block.targets.is_empty());
        blocks
    }

    /// Group same-parent orthogonal targets by occupation bitstrings.
    /// `Direct same-parent overlap then matches D^P entries by (o_a,o_b) and determinant phases.`
    /// # Arguments:
    /// - `block`: Target parent block whose orthogonal groups are rebuilt.
    /// - `data`: Shared NOCI data used to read occupation bitstrings and phases.
    /// # Returns:
    /// - `()`: Fills `block.orthogonal` without storing numerical overlap factors.
    fn build_overlap_orthogonal_groups(
        &self,
        block: &mut LocalParentBlock,
        data: &NOCIData<'_, f64>,
    ) {
        block.orthogonal.clear();
        for target in &block.targets {
            let key = (target.a, target.b);
            let phase = data.space.phase(NOCIIndex(target.det));

            if let Some(&position) = block.opos.get(&key) {
                let group = &mut block.orthogonal[position];
                group.targets.push(OrthogonalTarget {
                    local: target.local,
                    phase,
                });
            } else {
                block.opos.insert(key, block.orthogonal.len());
                block.orthogonal.push(OrthogonalTargetGroup {
                    targets: vec![OrthogonalTarget {
                        local: target.local,
                        phase,
                    }],
                });
            }
        }
    }

    /// Apply one source-parent to target-parent contribution.
    /// The method chooses direct orthogonal matching, sparse rows, or a blocked spin factorisation.
    /// # Arguments:
    /// - `target`: Rank-local target block for parent Q.
    /// - `source`: `Source parent P grouped D^P updates.`
    /// - `data`: Shared NOCI data and Wick intermediates.
    /// - `factors`: Persistent cross-parent same-spin overlap factors.
    /// - `scratch`: Reusable storage for factors, contractions, and output increments.
    /// # Returns:
    /// - `()`: Adds the QP contribution to `scratch.increments`.
    fn apply_overlap_parent_pair(
        &self,
        output: &mut [f64],
        target: &LocalParentBlock,
        source: &ParentUpdates,
        data: &NOCIData<'_, f64>,
        factors: &OverlapFactors,
        scratch: &mut OverlapScratch,
    ) {
        if target.parent == source.parent
            && let Some(mocache) = data.mocache
            && mocache[target.parent].orthogonal_slater_condon
        {
            self.apply_overlap_orthogonal(output, target, source, data, scratch);
            return;
        }
        if target.parent == source.parent {
            self.apply_overlap_direct(output, target, source, data, scratch);
            return;
        }

        if !data.input.wicks.enabled {
            self.apply_overlap_direct(output, target, source, data, scratch);
            return;
        }

        let Some(wicks) = data.wicks else {
            self.apply_overlap_direct(output, target, source, data, scratch);
            return;
        };

        let factorised = source.factorised_source();
        let contraction = self.select_overlap_contraction(target, factorised);
        let factors = factors.blocks[target.parent * self.parents.len() + source.parent].as_ref();
        let Some(factors) = factors else {
            let (lp, gp, target_left) = ordered_parent_pair(self, target.parent, source.parent);
            let pair = wicks.pair(lp, gp);

            match contraction {
                OverlapContraction::FactorisedRows => {
                    self.apply_overlap_factorised_rows_transient(
                        output,
                        (target, source),
                        data,
                        &pair,
                        target_left,
                        scratch,
                    );
                }
                OverlapContraction::AFirst => {
                    self.build_overlap_factor_tables(
                        target,
                        source,
                        data,
                        &pair,
                        target_left,
                        scratch,
                    );
                    self.apply_overlap_a_first_scratch(output, target, factorised, scratch);
                }
                OverlapContraction::BFirst => {
                    self.build_overlap_factor_tables(
                        target,
                        source,
                        data,
                        &pair,
                        target_left,
                        scratch,
                    );
                    self.apply_overlap_b_first_scratch(output, target, factorised, scratch);
                }
            }
            return;
        };

        match contraction {
            OverlapContraction::FactorisedRows => {
                Self::apply_overlap_factorised_rows(
                    output,
                    target,
                    source,
                    factors,
                    &mut scratch.values,
                );
            }
            OverlapContraction::AFirst => {
                Self::gather_overlap_factor_tables(
                    target,
                    &source.aids,
                    &source.bids,
                    factors,
                    &mut scratch.afac,
                    &mut scratch.bfac,
                );
                self.apply_overlap_a_first_scratch(output, target, factorised, scratch);
            }
            OverlapContraction::BFirst => {
                Self::gather_overlap_factor_tables(
                    target,
                    &source.aids,
                    &source.bids,
                    factors,
                    &mut scratch.afac,
                    &mut scratch.bfac,
                );
                self.apply_overlap_b_first_scratch(output, target, factorised, scratch);
            }
        }
    }

    /// Apply one cross-parent block with target-local sparse-row factor reuse.
    /// Precomputed same-spin factors are indexed directly by target and source component IDs before
    /// contracting `\delta N_w^{QP} = \sum_{(a,b)} A^{QP}_{\bar a a} B^{QP}_{\bar b b}D^P_{ab}`.
    /// # Arguments:
    /// - `output`: Rank-local persistent population increment.
    /// - `target`: `Rank-local target parent block Q defining w = (\bar a,\bar b).`
    /// - `source`: `Source parent P sparse D^P_{ab} entries and active positions.`
    /// - `factors`: Persistent full same-spin overlap factors for this ordered parent pair.
    /// - `values`: Reusable value storage receiving one output per target row.
    /// # Returns:
    /// - `()`: Adds factorised sparse-row `S\Delta` values to `output`.
    fn apply_overlap_factorised_rows(
        output: &mut [f64],
        target: &LocalParentBlock,
        source: &ParentUpdates,
        factors: &OverlapFactorBlock,
        values: &mut Vec<f64>,
    ) {
        let (afac, bfac, _, _) = factors.factors.factors();
        let factorised = source.factorised_source();
        Self::apply_overlap_factorised_rows_tables(
            output,
            target,
            factorised,
            FactorisedFactors {
                alpha: afac,
                beta: bfac,
                alpha_stride: afac.len() / factors.nta,
                beta_stride: bfac.len() / factors.ntb,
                target_alpha_ids: Some(target.aids.as_slice()),
                target_beta_ids: Some(target.bids.as_slice()),
                source_alpha_ids: Some(source.aids.as_slice()),
                source_beta_ids: Some(source.bids.as_slice()),
            },
            values,
        );
    }

    /// Apply one cross-parent block through shared source-factor tables and sparse rows.
    /// `\delta N_w^{QP} = \sum_{(a,b)}A^{QP}_{\bar a a}B^{QP}_{\bar b b}D^P_{ab}`.
    /// # Arguments:
    /// - `output`: Rank-local persistent population increment.
    /// - `target`: Rank-local target parent block.
    /// - `source`: Sparse source entries and active component IDs.
    /// - `factors`: Active alpha and beta factor tables.
    /// - `active_target_rows`: Whether factor rows use active target positions.
    /// - `source_position`: Map active source positions to factor-table columns.
    /// - `values`: Reusable target-row output storage.
    /// # Returns:
    /// - `()`: Adds the factorised sparse-row contribution to `output`.
    fn apply_overlap_factorised_rows_tables(
        output: &mut [f64],
        target: &LocalParentBlock,
        source: FactorisedSource<'_>,
        factors: FactorisedFactors<'_>,
        values: &mut Vec<f64>,
    ) {
        values.clear();
        values.resize(target.targets.len(), 0.0);

        values
            .par_iter_mut()
            .zip(target.targets.par_iter())
            .for_each(|(value, row)| {
                let mut dp = 0.0;

                for entry in source.entries {
                    dp += factors.alpha(target.apos[row.a], entry.apos)
                        * factors.beta(target.bpos[row.b], entry.bpos)
                        * entry.dn;
                }

                *value = dp;
            });

        for (value, target) in values.iter().zip(target.targets.iter()) {
            if *value != 0.0 {
                output[target.local] += value;
            }
        }
    }

    /// Apply one transient cross-parent block with target-local sparse-row factor reuse.
    /// Same-spin factors are generated only for source components active in the current
    /// `S\Delta` application and are discarded after the target row is contracted.
    /// # Arguments:
    /// - `output`: Rank-local persistent population increment.
    /// - `blocks`: Rank-local target block `Q` and source-parent sparse updates `P`.
    /// - `data`: Shared NOCI determinant data.
    /// - `pair`: Wick intermediates for the ordered parent pair.
    /// - `target_left`: Whether target determinants belong to the left Wick reference.
    /// - `scratch`: Reusable per-target output storage.
    /// # Returns:
    /// - `()`: Adds the transient factorised-row contribution to `output`.
    fn apply_overlap_factorised_rows_transient(
        &self,
        output: &mut [f64],
        blocks: (&LocalParentBlock, &ParentUpdates),
        data: &NOCIData<'_, f64>,
        pair: &WicksPairView<'_, f64>,
        target_left: bool,
        scratch: &mut OverlapScratch,
    ) {
        let (target, source) = blocks;
        let factorised = source.factorised_source();
        let nsa = source.aids.len();
        let nsb = source.bids.len();
        let source_areps = source
            .aids
            .iter()
            .map(|&a| self.parents[source.parent].areps[a])
            .collect::<Vec<_>>();
        let source_breps = source
            .bids
            .iter()
            .map(|&b| self.parents[source.parent].breps[b])
            .collect::<Vec<_>>();

        scratch.values.clear();
        scratch.values.resize(target.targets.len(), 0.0);

        scratch
            .values
            .par_iter_mut()
            .zip(target.targets.par_iter())
            .for_each_init(
                || (WickScratchSpin::new(), vec![0.0; nsa], vec![0.0; nsb]),
                |state, (value, t)| {
                    let (wick, afac, bfac) = state;
                    build_spin_overlap_factor_row(
                        pair,
                        data,
                        (
                            self.parents[target.parent].areps[t.a],
                            source_areps.as_slice(),
                        ),
                        target_left,
                        true,
                        wick,
                        afac.as_mut_slice(),
                    );
                    build_spin_overlap_factor_row(
                        pair,
                        data,
                        (
                            self.parents[target.parent].breps[t.b],
                            source_breps.as_slice(),
                        ),
                        target_left,
                        false,
                        wick,
                        bfac.as_mut_slice(),
                    );

                    let mut dp = 0.0;
                    for entry in factorised.entries {
                        dp += afac[entry.apos] * bfac[entry.bpos] * entry.dn;
                    }
                    *value = dp;
                },
            );

        for (value, target) in scratch.values.iter().zip(target.targets.iter()) {
            if *value != 0.0 {
                output[target.local] += value;
            }
        }
    }

    /// Apply factorised rows using temporary active factor tables owned by `scratch`.
    /// `\delta N_w = \sum_{(a,b)}A^{QP}_{\bar a a}B^{QP}_{\bar b b}D^P_{ab}` is evaluated
    /// through the shared row contraction after the tables are moved out of the workspace.
    /// # Arguments:
    /// - `output`: Rank-local persistent population increment.
    /// - `target`: Target parent block defining active target rows.
    /// - `source`: Borrowed sparse source description.
    /// - `scratch`: Reusable active factor and contraction storage.
    /// # Returns:
    /// - `()`: Adds the factorised row contribution to `output`.
    fn apply_overlap_factorised_rows_scratch(
        output: &mut [f64],
        target: &LocalParentBlock,
        source: FactorisedSource<'_>,
        scratch: &mut OverlapScratch,
    ) {
        let afac = std::mem::take(&mut scratch.afac);
        let bfac = std::mem::take(&mut scratch.bfac);
        let factors = FactorisedFactors {
            alpha: &afac,
            beta: &bfac,
            alpha_stride: source.nalpha,
            beta_stride: source.nbeta,
            target_alpha_ids: None,
            target_beta_ids: None,
            source_alpha_ids: None,
            source_beta_ids: None,
        };
        Self::apply_overlap_factorised_rows_tables(
            output,
            target,
            source,
            factors,
            &mut scratch.values,
        );
        scratch.afac = afac;
        scratch.bfac = bfac;
    }

    /// Apply A-first blocked contractions using temporary active factor tables in `scratch`.
    /// `T_{\bar a b} = \sum_a A^{QP}_{\bar a a}D^P_{ab}` is formed before the shared final
    /// beta contraction, with the same mathematics as persistent factor storage.
    /// # Arguments:
    /// - `self`: Shared determinant-space factorisation.
    /// - `output`: Rank-local persistent population increment.
    /// - `target`: Target parent block defining active target rows.
    /// - `source`: Borrowed sparse source description.
    /// - `scratch`: Reusable active factor and contraction storage.
    /// # Returns:
    /// - `()`: Adds the A-first contribution to `output`.
    fn apply_overlap_a_first_scratch(
        &self,
        output: &mut [f64],
        target: &LocalParentBlock,
        source: FactorisedSource<'_>,
        scratch: &mut OverlapScratch,
    ) {
        let afac = std::mem::take(&mut scratch.afac);
        let bfac = std::mem::take(&mut scratch.bfac);
        let factors = FactorisedFactors {
            alpha: &afac,
            beta: &bfac,
            alpha_stride: source.nalpha,
            beta_stride: source.nbeta,
            target_alpha_ids: None,
            target_beta_ids: None,
            source_alpha_ids: None,
            source_beta_ids: None,
        };
        self.apply_overlap_a_first(output, target, source, factors, scratch);
        scratch.afac = afac;
        scratch.bfac = bfac;
    }

    /// Apply B-first blocked contractions using temporary active factor tables in `scratch`.
    /// `U_{a\bar b} = \sum_b D^P_{ab}B^{QP}_{\bar b b}` is formed before the shared final
    /// alpha contraction, with the same mathematics as persistent factor storage.
    /// # Arguments:
    /// - `self`: Shared determinant-space factorisation.
    /// - `output`: Rank-local persistent population increment.
    /// - `target`: Target parent block defining active target rows.
    /// - `source`: Borrowed sparse source description.
    /// - `scratch`: Reusable active factor and contraction storage.
    /// # Returns:
    /// - `()`: Adds the B-first contribution to `output`.
    fn apply_overlap_b_first_scratch(
        &self,
        output: &mut [f64],
        target: &LocalParentBlock,
        source: FactorisedSource<'_>,
        scratch: &mut OverlapScratch,
    ) {
        let afac = std::mem::take(&mut scratch.afac);
        let bfac = std::mem::take(&mut scratch.bfac);
        let factors = FactorisedFactors {
            alpha: &afac,
            beta: &bfac,
            alpha_stride: source.nalpha,
            beta_stride: source.nbeta,
            target_alpha_ids: None,
            target_beta_ids: None,
            source_alpha_ids: None,
            source_beta_ids: None,
        };
        self.apply_overlap_b_first(output, target, source, factors, scratch);
        scratch.afac = afac;
        scratch.bfac = bfac;
    }

    /// `Select how to apply one cross-parent block of S\Delta.`
    /// The row path factorises each target as
    /// `\delta N_w^{QP} = \sum_{(a,b)} A^{QP}_{\bar a a} B^{QP}_{\bar b b} D^P_{ab}.`
    /// The direct determinant-pair Wick loop is avoided because the weighted model accounts for
    /// same-spin factor reuse instead of charging every sparse product as a full overlap.
    /// `Scores are C = 32\,F + M, where F is the number of same-spin Wick factors and M is the`
    /// number of scalar sparse products; the factor weight reflects that one same-spin Wick
    /// evaluation is substantially more expensive than one multiply-add.
    /// # Arguments:
    /// - `target`: Rank-local target parent block.
    /// - `source`: `Sparse source-parent D^P entries and active spin IDs.`
    /// # Returns:
    /// - `OverlapContraction`: `FactorisedRows`, `AFirst`, or `BFirst` selected by weighted score.
    fn select_overlap_contraction(
        &self,
        target: &LocalParentBlock,
        source: FactorisedSource<'_>,
    ) -> OverlapContraction {
        let nt = target.targets.len();
        let ne = source.entries.len();
        let nta = target.aids.len();
        let ntb = target.bids.len();
        let nsa = source.nalpha;
        let nsb = source.nbeta;

        let row_factors = nt.saturating_mul(nsa.saturating_add(nsb));
        let row_products = nt.saturating_mul(ne);
        let a_factors = nta
            .saturating_mul(nsa)
            .saturating_add(ntb.saturating_mul(nsb));
        let a_products = nta
            .saturating_mul(ne)
            .saturating_add(nt.saturating_mul(nsb));
        let b_factors = a_factors;
        let b_products = ntb
            .saturating_mul(ne)
            .saturating_add(nt.saturating_mul(nsa));

        let wick_factor_cost = 32usize;
        let row_score = row_factors
            .saturating_mul(wick_factor_cost)
            .saturating_add(row_products);
        let a_score = a_factors
            .saturating_mul(wick_factor_cost)
            .saturating_add(a_products);
        let b_score = b_factors
            .saturating_mul(wick_factor_cost)
            .saturating_add(b_products);

        if row_score <= a_score && row_score <= b_score {
            OverlapContraction::FactorisedRows
        } else if a_score <= b_score {
            OverlapContraction::AFirst
        } else {
            OverlapContraction::BFirst
        }
    }

    /// Apply same-parent orthogonal contributions by occupation matching.
    /// This avoids Wick evaluation and reproduces the determinant phase product.
    /// # Arguments:
    /// - `target`: Rank-local target block with occupation groups.
    /// - `source`: `Same-parent source updates D^P_{ab}.`
    /// - `data`: Shared NOCI data used to read source occupations and phases.
    /// - `scratch`: Reusable values and increment storage.
    /// # Returns:
    /// - `()`: Adds the same-parent orthogonal contribution to `output`.
    fn apply_overlap_orthogonal(
        &self,
        output: &mut [f64],
        target: &LocalParentBlock,
        source: &ParentUpdates,
        data: &NOCIData<'_, f64>,
        scratch: &mut OverlapScratch,
    ) {
        scratch.values.clear();
        // The component pair is the canonical same-parent occupation identity. Accumulate
        // source phases before applying the target phase to every retained duplicate.
        for (&det, entry) in source.dets.iter().zip(source.entries.iter()) {
            let state = data.space.state(NOCIIndex(det));
            let key = (state.aid.0, state.bid.0);

            if let Some(&position) = target.opos.get(&key) {
                let phase = data.space.phase(NOCIIndex(det));
                for t in &target.orthogonal[position].targets {
                    output[t.local] += t.phase * phase * entry.dn;
                }
            }
        }
    }

    /// Apply one parent block by direct sparse rows.
    /// This fallback is selected for sparse updates and non-Wick overlap evaluation.
    /// # Arguments:
    /// - `target`: Rank-local target parent block.
    /// - `source`: `Sparse source-parent D^P entries.`
    /// - `data`: Shared NOCI data used by the general overlap evaluator.
    /// - `scratch`: Reusable per-target value and increment storage.
    /// # Returns:
    /// - `()`: `Adds sparse-row S\Delta values to output.`
    fn apply_overlap_direct(
        &self,
        output: &mut [f64],
        target: &LocalParentBlock,
        source: &ParentUpdates,
        data: &NOCIData<'_, f64>,
        scratch: &mut OverlapScratch,
    ) {
        scratch.values.clear();
        scratch.values.resize(target.targets.len(), 0.0);

        scratch
            .values
            .par_iter_mut()
            .zip(target.targets.par_iter())
            .for_each_init(WickScratchSpin::new, |wick_scratch, (value, target)| {
                let mut dp = 0.0;
                for (&det, entry) in source.dets.iter().zip(source.entries.iter()) {
                    let (a, b) = if target.det <= det {
                        (target.det, det)
                    } else {
                        (det, target.det)
                    };
                    let s = if data.input.wicks.enabled && data.wicks.is_none() {
                        calculate_s_pair_naive(data, NOCIIndex(a), NOCIIndex(b))
                    } else {
                        calculate_s_pair(
                            data,
                            DetPair::new(NOCIIndex(a), NOCIIndex(b)),
                            Some(wick_scratch),
                        )
                    };
                    dp += s * entry.dn;
                }
                *value = dp;
            });

        for (value, target) in scratch.values.iter().zip(target.targets.iter()) {
            if *value != 0.0 {
                output[target.local] += value;
            }
        }
    }

    /// Apply same-parent BApply overlap by exact occupation matching.
    /// `\langle\Phi_w^P|D^P\rangle` is zero unless both spin occupations agree, in which case
    /// the retained and transient parent-relative determinant phases multiply.
    /// # Arguments:
    /// - `output`: Rank-local persistent population increment.
    /// - `target`: Retained target block belonging to parent `P`.
    /// - `source`: Transient orthogonal sources belonging to the same parent.
    /// - `data`: Shared retained determinant basis.
    /// # Returns:
    /// - `()`: Adds exact same-parent `B^\dagger\chi` contributions.
    fn apply_orthogonal_source_exact(
        output: &mut [f64],
        target: &LocalParentBlock,
        source: &OrthogonalParentUpdates,
        data: &NOCIData<'_, f64>,
    ) {
        for t in &target.targets {
            let state = data.space.state(NOCIIndex(t.det));
            if let Some(value) = source.exact.get(&(state.aid.0, state.bid.0)) {
                output[t.local] += data.space.phase(NOCIIndex(t.det)) * value;
            }
        }
    }

    /// Construct active cross-parent factors for transient orthogonal BApply sources.
    /// # Arguments:
    /// - `target`: Retained target block defining active target spin components.
    /// - `source`: Transient orthogonal sources and parent-relative excitation side arrays.
    /// - `data`: Shared retained NOCI basis.
    /// - `pair`: Wick intermediates for ordered reference pair `QP`.
    /// - `target_left`: Whether target parent `Q` is the left Wick reference.
    /// - `scratch`: Shared factor and blocked-contraction workspace.
    /// # Returns:
    /// - `()`: Fills active `A^{QP}` and `B^{QP}` factor tables.
    fn build_orthogonal_overlap_factor_tables(
        &self,
        target: &LocalParentBlock,
        source: &OrthogonalParentUpdates,
        data: &NOCIData<'_, f64>,
        pair: &WicksPairView<'_, f64>,
        target_left: bool,
        scratch: &mut OverlapScratch,
    ) {
        let target_areps = target
            .aids
            .iter()
            .map(|&a| self.parents[target.parent].areps[a])
            .collect::<Vec<_>>();
        let target_breps = target
            .bids
            .iter()
            .map(|&b| self.parents[target.parent].breps[b])
            .collect::<Vec<_>>();
        let nsa = source.areps.len();
        let nsb = source.breps.len();

        scratch.afac.clear();
        scratch.bfac.clear();
        scratch.afac.resize(target_areps.len() * nsa, 0.0);
        scratch.bfac.resize(target_breps.len() * nsb, 0.0);

        scratch
            .afac
            .par_chunks_mut(nsa)
            .zip(target_areps.par_iter())
            .for_each_init(WickScratchSpin::new, |wick, (row, &target_rep)| {
                xw_overlap_orthogonal_prepared_batched(
                    &pair.aa,
                    SameSpinOrthogonalOverlapBatch {
                        basis: data.space,
                        target: target_rep,
                        sources: &source.areps,
                        source_excitations: &source.aex,
                        target_left,
                        alpha: true,
                        out: row,
                    },
                    &mut wick.aa,
                );
            });
        scratch
            .bfac
            .par_chunks_mut(nsb)
            .zip(target_breps.par_iter())
            .for_each_init(WickScratchSpin::new, |wick, (row, &target_rep)| {
                xw_overlap_orthogonal_prepared_batched(
                    &pair.bb,
                    SameSpinOrthogonalOverlapBatch {
                        basis: data.space,
                        target: target_rep,
                        sources: &source.breps,
                        source_excitations: &source.bex,
                        target_left,
                        alpha: false,
                        out: row,
                    },
                    &mut wick.bb,
                );
            });
    }

    /// Build active `A^{QP}` and `B^{QP}` factor tables for one transient parent-pair application.
    /// Only target and source spin components active in the current sparse `S\Delta` are
    /// materialised in `scratch`.
    /// # Arguments:
    /// - `target`: Target parent block defining active target component IDs.
    /// - `source`: Source parent updates defining active source component IDs.
    /// - `data`: Shared NOCI determinant data.
    /// - `pair`: Wick intermediates for the ordered parent pair.
    /// - `target_left`: Whether target determinants belong to the left Wick reference.
    /// - `scratch`: Reusable active factor-table storage.
    /// # Returns:
    /// - `()`: Fills `scratch.afac` and `scratch.bfac` for the active parent-pair block.
    fn build_overlap_factor_tables(
        &self,
        target: &LocalParentBlock,
        source: &ParentUpdates,
        data: &NOCIData<'_, f64>,
        pair: &WicksPairView<'_, f64>,
        target_left: bool,
        scratch: &mut OverlapScratch,
    ) {
        let target_areps = target
            .aids
            .iter()
            .map(|&a| self.parents[target.parent].areps[a])
            .collect::<Vec<_>>();
        let target_breps = target
            .bids
            .iter()
            .map(|&b| self.parents[target.parent].breps[b])
            .collect::<Vec<_>>();
        let source_areps = source
            .aids
            .iter()
            .map(|&a| self.parents[source.parent].areps[a])
            .collect::<Vec<_>>();
        let source_breps = source
            .bids
            .iter()
            .map(|&b| self.parents[source.parent].breps[b])
            .collect::<Vec<_>>();

        let nta = target_areps.len();
        let ntb = target_breps.len();
        let nsa = source_areps.len();
        let nsb = source_breps.len();

        scratch.afac.clear();
        scratch.bfac.clear();
        scratch.afac.resize(nta * nsa, 0.0);
        scratch.bfac.resize(ntb * nsb, 0.0);

        build_spin_overlap_factors(
            pair,
            data,
            (target_areps.as_slice(), source_areps.as_slice()),
            0..nta,
            target_left,
            true,
            scratch.afac.as_mut_slice(),
        );
        build_spin_overlap_factors(
            pair,
            data,
            (target_breps.as_slice(), source_breps.as_slice()),
            0..ntb,
            target_left,
            false,
            scratch.bfac.as_mut_slice(),
        );
    }

    /// Gather active same-spin factor submatrices from the persistent full parent-pair tables.
    /// `A^{QP}_{\bar a a}` and `B^{QP}_{\bar b b}` are selected only for the target and source
    /// component IDs active in the current sparse `S\Delta` application.
    /// # Arguments:
    /// - `target`: Target parent block defining active target component IDs.
    /// - `source`: Source parent updates defining active source component IDs.
    /// - `factors`: Persistent full same-spin overlap factors for this ordered parent pair.
    /// - `afac`: Reusable active alpha-factor submatrix storage.
    /// - `bfac`: Reusable active beta-factor submatrix storage.
    /// # Returns:
    /// - `()`: Fills the active alpha- and beta-factor submatrices without Wick evaluation.
    fn gather_overlap_factor_tables(
        target: &LocalParentBlock,
        aids: &[usize],
        bids: &[usize],
        factors: &OverlapFactorBlock,
        afac: &mut Vec<f64>,
        bfac: &mut Vec<f64>,
    ) {
        let nta = target.aids.len();
        let ntb = target.bids.len();
        let nsa = aids.len();
        let nsb = bids.len();
        let na = nta * nsa;
        let nb = ntb * nsb;

        if afac.len() != na {
            afac.resize(na, 0.0);
        }
        if bfac.len() != nb {
            bfac.resize(nb, 0.0);
        }
        let (full_afac, full_bfac, _, _) = factors.factors.factors();

        afac.par_chunks_mut(nsa)
            .zip(target.aids.par_iter())
            .for_each(|(row, &ta)| {
                let stride = full_afac.len() / factors.nta;
                let full = &full_afac[ta * stride..(ta + 1) * stride];
                for (col, &sa) in aids.iter().enumerate() {
                    row[col] = full[sa];
                }
            });

        bfac.par_chunks_mut(nsb)
            .zip(target.bids.par_iter())
            .for_each(|(row, &tb)| {
                let stride = full_bfac.len() / factors.ntb;
                let full = &full_bfac[tb * stride..(tb + 1) * stride];
                for (col, &sb) in bids.iter().enumerate() {
                    row[col] = full[sb];
                }
            });
    }

    /// `Apply T_{\bar a b} = \sum_a A^{QP}_{\bar a a}D^P_{ab}.`
    /// `The final target rows multiply T by B^{QP}_{\bar b_w b}.`
    /// # Arguments:
    /// - `output`: Rank-local persistent population increment.
    /// - `target`: `Target parent block defining \bar a_w and \bar b_w rows.`
    /// - `source`: `Sparse source-parent D^P entries and active positions.`
    /// - `scratch`: Reusable factor, intermediate, value, and increment storage.
    /// # Returns:
    /// - `()`: Adds the A-first blocked contribution to `output`.
    fn apply_overlap_a_first(
        &self,
        output: &mut [f64],
        target: &LocalParentBlock,
        source: FactorisedSource<'_>,
        factors: FactorisedFactors<'_>,
        scratch: &mut OverlapScratch,
    ) {
        let nta = target.aids.len();
        let nsb = source.nbeta;

        scratch.intermediate.clear();
        scratch.intermediate.resize(nta * nsb, 0.0);

        // Form `T_{\bar a b} = \sum_a A^{QP}_{\bar a a}D^P_{ab}`.
        scratch
            .intermediate
            .par_chunks_mut(nsb)
            .enumerate()
            .for_each(|(ta_pos, row)| {
                for entry in source.entries {
                    row[entry.bpos] += factors.alpha(ta_pos, entry.apos) * entry.dn;
                }
            });

        scratch.values.clear();

        if target.contiguous_locals {
            let first = target.first_local;
            let increments = &mut output[first..first + target.targets.len()];

            // Finish `\delta N_w^{QP} = \sum_b T_{\bar a_w b}B^{QP}_{\bar b_w b}`.
            increments
                .par_iter_mut()
                .zip(target.targets.par_iter())
                .for_each(|(increment, t)| {
                    let ta_pos = target.apos[t.a];
                    let tb_pos = target.bpos[t.b];

                    let trow = &scratch.intermediate[ta_pos * nsb..(ta_pos + 1) * nsb];
                    *increment += factors.dot_beta(tb_pos, trow);
                });
        } else {
            scratch.values.resize(target.targets.len(), 0.0);

            scratch
                .values
                .par_iter_mut()
                .zip(target.targets.par_iter())
                .for_each(|(value, t)| {
                    let ta_pos = target.apos[t.a];
                    let tb_pos = target.bpos[t.b];

                    let trow = &scratch.intermediate[ta_pos * nsb..(ta_pos + 1) * nsb];
                    *value = factors.dot_beta(tb_pos, trow);
                });

            for (value, target) in scratch.values.iter().zip(target.targets.iter()) {
                if *value != 0.0 {
                    output[target.local] += value;
                }
            }
        }
    }

    /// `Apply U_{a\bar b} = \sum_b D^P_{ab}B^{QP}_{\bar b b}.`
    /// `The final target rows multiply U by A^{QP}_{\bar a_w a}.`
    /// # Arguments:
    /// - `output`: Rank-local persistent population increment.
    /// - `target`: `Target parent block defining \bar a_w and \bar b_w rows.`
    /// - `source`: `Sparse source-parent D^P entries and active positions.`
    /// - `scratch`: Reusable factor, intermediate, value, and increment storage.
    /// # Returns:
    /// - `()`: Adds the B-first blocked contribution to `output`.
    fn apply_overlap_b_first(
        &self,
        output: &mut [f64],
        target: &LocalParentBlock,
        source: FactorisedSource<'_>,
        factors: FactorisedFactors<'_>,
        scratch: &mut OverlapScratch,
    ) {
        let ntb = target.bids.len();
        let nsa = source.nalpha;

        scratch.intermediate.clear();
        scratch.intermediate.resize(ntb * nsa, 0.0);

        // Form `U_{a\bar b} = \sum_b D^P_{ab}B^{QP}_{\bar b b}`.
        scratch
            .intermediate
            .par_chunks_mut(nsa)
            .enumerate()
            .for_each(|(tb_pos, row)| {
                for entry in source.entries {
                    row[entry.apos] += entry.dn * factors.beta(tb_pos, entry.bpos);
                }
            });

        scratch.values.clear();

        if target.contiguous_locals {
            let first = target.first_local;
            let increments = &mut output[first..first + target.targets.len()];

            // Finish `\delta N_w^{QP} = \sum_a A^{QP}_{\bar a_w a}U_{a\bar b_w}`.
            increments
                .par_iter_mut()
                .zip(target.targets.par_iter())
                .for_each(|(increment, t)| {
                    let ta_pos = target.apos[t.a];
                    let tb_pos = target.bpos[t.b];

                    let urow = &scratch.intermediate[tb_pos * nsa..(tb_pos + 1) * nsa];
                    *increment += factors.dot_alpha(ta_pos, urow);
                });
        } else {
            scratch.values.resize(target.targets.len(), 0.0);

            scratch
                .values
                .par_iter_mut()
                .zip(target.targets.par_iter())
                .for_each(|(value, t)| {
                    let ta_pos = target.apos[t.a];
                    let tb_pos = target.bpos[t.b];

                    let urow = &scratch.intermediate[tb_pos * nsa..(tb_pos + 1) * nsa];

                    *value = factors.dot_alpha(ta_pos, urow);
                });

            for (value, target) in scratch.values.iter().zip(target.targets.iter()) {
                if *value != 0.0 {
                    output[target.local] += value;
                }
            }
        }
    }
}

/// Build same-spin overlap factor rows from the common overlap-factor dispatcher.
/// Independent source components are evaluated together so the widest available fixed-rank SIMD
/// kernel is used when applicable, with the scalar overlap evaluator as the fallback.
/// # Arguments:
/// - `pair`: Wick intermediates for the ordered parent pair.
/// - `data`: Shared NOCI determinant data.
/// - `reps`: Reduced target and source spin representatives.
/// - `rows`: Target-row range to fill.
/// - `target_left`: Whether target determinants belong to the left Wick reference.
/// - `alpha`: Whether to build alpha or beta overlap factors.
/// - `out`: Mutable row-major overlap factor table.
/// # Returns:
/// - `()`: Fills `out` overlap factor rows.
fn build_spin_overlap_factors(
    pair: &WicksPairView<'_, f64>,
    data: &NOCIData<'_, f64>,
    reps: (
        &[ReducedOneSpinNOCIDeterminantState],
        &[ReducedOneSpinNOCIDeterminantState],
    ),
    rows: Range<usize>,
    target_left: bool,
    alpha: bool,
    out: &mut [f64],
) {
    let (target_reps, source_reps) = reps;
    let nsource = source_reps.len();
    let row0 = rows.start;
    let row1 = rows.end;

    out[row0 * nsource..row1 * nsource]
        .par_chunks_mut(nsource)
        .zip(target_reps[row0..row1].par_iter())
        .for_each_init(WickScratchSpin::new, |scratch, (row, &target_rep)| {
            build_spin_overlap_factor_row(
                pair,
                data,
                (target_rep, source_reps),
                target_left,
                alpha,
                scratch,
                row,
            );
        });
}

/// Build one same-spin overlap factor row through the common overlap-factor dispatcher.
/// The dispatcher selects AVX-512, AVX2/FMA or scalar evaluation according to the Wick pair,
/// excitation ranks and available CPU features.
/// # Arguments:
/// - `pair`: Wick intermediates for the ordered parent pair.
/// - `data`: Shared NOCI determinant data used by scalar fallback evaluation.
/// - `reps`: Reduced target representative and source representatives.
/// - `target_left`: Whether the target determinant belongs to the left Wick reference.
/// - `alpha`: Whether to evaluate alpha-alpha or beta-beta overlap factors.
/// - `scratch`: Reusable spin-resolved Wick evaluator workspace.
/// - `out`: Output same-spin overlap factor row.
/// # Returns:
/// - `()`: Fills `out` for the target spin component.
fn build_spin_overlap_factor_row(
    pair: &WicksPairView<'_, f64>,
    data: &NOCIData<'_, f64>,
    reps: (
        ReducedOneSpinNOCIDeterminantState,
        &[ReducedOneSpinNOCIDeterminantState],
    ),
    target_left: bool,
    alpha: bool,
    scratch: &mut WickScratchSpin<f64>,
    out: &mut [f64],
) {
    let (target, sources) = reps;
    let (w, scratch) = if alpha {
        (&pair.aa, &mut scratch.aa)
    } else {
        (&pair.bb, &mut scratch.bb)
    };

    xw_overlap_prepared_batched(
        w,
        SameSpinOverlapBatch {
            basis: data.space,
            target,
            sources,
            target_left,
            alpha,
            out,
        },
        scratch,
    );
}

impl ParentUpdates {
    /// Construct a temporary empty placeholder while a source block is moved out of scratch.
    /// # Arguments:
    /// - `parent`: Source parent P represented by the placeholder.
    /// # Returns:
    /// - `ParentUpdates`: Empty block without allocated position maps.
    fn empty(parent: usize) -> Self {
        Self {
            parent,
            entries: Vec::new(),
            dets: Vec::new(),
            aids: Vec::new(),
            bids: Vec::new(),
            apos: Vec::new(),
            bpos: Vec::new(),
        }
    }

    /// Construct empty grouped storage for one source parent.
    /// # Arguments:
    /// - `parent`: Source parent P represented by this update block.
    /// - `na`: Maximum number of parent-local a components.
    /// - `nb`: Maximum number of parent-local b components.
    /// # Returns:
    /// - `ParentUpdates`: `Empty D^P storage with inactive position maps.`
    fn new(
        parent: usize,
        na: usize,
        nb: usize,
    ) -> Self {
        Self {
            parent,
            entries: Vec::new(),
            dets: Vec::new(),
            aids: Vec::new(),
            bids: Vec::new(),
            apos: vec![usize::MAX; na],
            bpos: vec![usize::MAX; nb],
        }
    }

    /// Borrow the minimal sparse source required by blocked contractions.
    /// # Arguments:
    /// - `self`: Retained parent-local update storage.
    /// # Returns
    /// - `FactorisedSource`: Sparse amplitudes and active spin dimensions without identity data.
    fn factorised_source(&self) -> FactorisedSource<'_> {
        FactorisedSource {
            entries: &self.entries,
            nalpha: self.aids.len(),
            nbeta: self.bids.len(),
        }
    }

    /// `Add one sparse D^P_{ab} entry and record active spin IDs on first occurrence.`
    /// # Arguments:
    /// - `det`: `Source determinant \Omega.`
    /// - `a`: `Source-parent local a component ID a_\Omega.`
    /// - `b`: `Source-parent local b component ID b_\Omega.`
    /// - `dn`: `Sparse pre-overlap update \Delta_\Omega.`
    /// # Returns:
    /// - `()`: Appends one sparse entry and updates active ID maps.
    fn push(
        &mut self,
        det: usize,
        a: usize,
        b: usize,
        dn: f64,
    ) {
        if self.apos[a] == usize::MAX {
            self.apos[a] = self.aids.len();
            self.aids.push(a);
        }

        if self.bpos[b] == usize::MAX {
            self.bpos[b] = self.bids.len();
            self.bids.push(b);
        }

        self.entries.push(SpinUpdate {
            apos: self.apos[a],
            bpos: self.bpos[b],
            dn,
        });
        self.dets.push(det);
    }

    /// `Clear D^P_{ab} while invalidating only IDs active in the last application.`
    /// # Arguments:
    /// - `self`: Grouped source-parent updates to clear.
    /// # Returns:
    /// - `()`: Clears entries and active IDs while retaining allocation capacity.
    fn clear(&mut self) {
        for &a in &self.aids {
            self.apos[a] = usize::MAX;
        }

        for &b in &self.bids {
            self.bpos[b] = usize::MAX;
        }

        self.entries.clear();
        self.dets.clear();
        self.aids.clear();
        self.bids.clear();
    }
}

impl OrthogonalParentUpdates {
    /// Construct empty BApply source storage for one parent reference.
    /// # Arguments:
    /// - `parent`: Parent reference whose orthonormal MOs define the sources.
    /// # Returns:
    /// - `Self`: Empty report-local orthogonal source block.
    fn new(parent: usize) -> Self {
        Self {
            parent,
            entries: Vec::new(),
            areps: Vec::new(),
            breps: Vec::new(),
            aex: Vec::new(),
            bex: Vec::new(),
            exact: HashMap::new(),
            aids: Vec::new(),
            bids: Vec::new(),
            apos: Vec::new(),
            bpos: Vec::new(),
        }
    }

    /// Add one coalesced orthogonal-space residual entry and its transient Wick metadata.
    /// # Arguments:
    /// - `self`: Parent-local transient source block.
    /// - `det`: Orthogonal determinant receiving `\chi_D^P`.
    /// - `dn`: Coalesced residual amplitude.
    /// - `components`: Permanent auxiliary spin metadata for parent `P`.
    /// # Returns:
    /// - `()`: Adds one sparse entry and any newly active spin components.
    fn push(
        &mut self,
        det: AuxiliaryDeterminantState,
        dn: f64,
        components: &ParentComponents<AuxiliarySpinIndex>,
    ) {
        let aid = det.aid.0;
        let bid = det.bid.0;

        if self.apos.len() <= aid {
            self.apos.resize(aid + 1, usize::MAX);
        }
        if self.bpos.len() <= bid {
            self.bpos.resize(bid + 1, usize::MAX);
        }

        let apos = if self.apos[aid] == usize::MAX {
            let position = self.aids.len();
            self.aids.push(aid);
            self.apos[aid] = position;
            let alpha = components.alpha(det.aid);
            self.aex.push(alpha.excitation);
            self.areps.push(alpha.reduced);
            self.apos[aid]
        } else {
            self.apos[aid]
        };

        let bpos = if self.bpos[bid] == usize::MAX {
            let position = self.bids.len();
            self.bids.push(bid);
            self.bpos[bid] = position;
            let beta = components.beta(det.bid);
            self.bex.push(beta.excitation);
            self.breps.push(beta.reduced);
            self.bpos[bid]
        } else {
            self.bpos[bid]
        };

        let phase =
            components.alpha(det.aid).reduced.phase * components.beta(det.bid).reduced.phase;

        self.entries.push(SpinUpdate { apos, bpos, dn });
        *self.exact.entry((aid, bid)).or_insert(0.0) += phase * dn;
    }

    /// Borrow the minimal transient source required by blocked contractions.
    /// # Arguments:
    /// - `self`: Parent-orthogonal source storage.
    /// # Returns
    /// - `FactorisedSource`: Sparse amplitudes and transient spin dimensions without fake IDs.
    fn factorised_source(&self) -> FactorisedSource<'_> {
        FactorisedSource {
            entries: &self.entries,
            nalpha: self.aids.len(),
            nbeta: self.bids.len(),
        }
    }

    /// Clear report-local orthogonal source metadata while retaining allocations.
    /// # Arguments:
    /// - `self`: Parent-local transient source block.
    /// # Returns:
    /// - `()`: Removes all source entries and metadata.
    fn clear(&mut self) {
        for &aid in &self.aids {
            self.apos[aid] = usize::MAX;
        }
        for &bid in &self.bids {
            self.bpos[bid] = usize::MAX;
        }
        self.entries.clear();
        self.areps.clear();
        self.breps.clear();
        self.aex.clear();
        self.bex.clear();
        self.exact.clear();
        self.aids.clear();
        self.bids.clear();
    }
}
