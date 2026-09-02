// noci/factorise/overlap.rs

// Standard library imports.
use std::ops::Range;
use std::path::Path;

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::ReducedOneSpinDetState;
use crate::input::SNOCIStorage;
use crate::maths::dot_f64;
use crate::noci::overlap::{calculate_s_pair, calculate_s_pair_naive};
use crate::noci::types::{DetPair, NOCIData};
use crate::nonorthogonalwicks::{
    SameSpinOverlapBatch, WickScratchSpin, WicksPairView, xw_overlap_prepared_batched,
};

// Parent/sibling imports.
use super::storage::{OverlapFactorStorage, OverlapStoragePlan};
use super::{SpinFactorisation, ordered_parent_pair};

#[derive(Clone, Copy)]
struct SpinUpdate {
    /// `Global determinant index \Omega receiving the pre-overlap update.`
    det: usize,
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
    /// Active source a component IDs for this application.
    aids: Vec<usize>,
    /// Active source b component IDs for this application.
    bids: Vec<usize>,
    /// Source-parent a ID to active position map.
    apos: Vec<usize>,
    /// Source-parent b ID to active position map.
    bpos: Vec<usize>,
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
    /// Parent-local occupation-pair ID to target occupation-group position.
    opos: Vec<usize>,
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
    /// Raw overlap factor and optional proposal-CDF backing.
    factors: OverlapFactorStorage,
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
    /// Temporary active occupation IDs for sparse orthogonal same-parent application.
    active_oids: Vec<usize>,
    /// Cached target slice pointer for validating reusable target blocks.
    cached_targets_ptr: *const usize,
    /// Cached target slice length for validating reusable target blocks.
    cached_targets_len: usize,
    /// Reusable target parent blocks for a fixed rank-local target list.
    target_blocks: Vec<LocalParentBlock>,
}

impl OverlapFactors {
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
                            factors,
                        });
                }
            }
        }

        OverlapFactors {
            blocks: factor_blocks,
        }
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
            active_oids: Vec::new(),
            cached_targets_ptr: std::ptr::null(),
            cached_targets_len: 0,
            target_blocks: Vec::new(),
        }
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
        scratch.active_oids.clear();
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

            let parent = data.basis[det].parent;
            if scratch.updates[parent].entries.is_empty() {
                scratch.active_parents.push(parent);
            }
            scratch.updates[parent].push(det, self.aids[det], self.bids[det], dn);
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
                opos: vec![usize::MAX; self.parents[parent].oreps.len()],
            })
            .collect::<Vec<_>>();

        for (local, &det) in targets.iter().enumerate() {
            let parent = data.basis[det].parent;
            let a = self.aids[det];
            let b = self.bids[det];
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
            let det = &data.basis[target.det];
            let oid =
                self.parents[block.parent].oids[target.det - self.parents[block.parent].first_det];
            let phase = det.pha * det.phb;
            if block.opos[oid] != usize::MAX {
                let group = &mut block.orthogonal[block.opos[oid]];
                group.targets.push(OrthogonalTarget {
                    local: target.local,
                    phase,
                });
            } else {
                block.opos[oid] = block.orthogonal.len();
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

        let contraction = self.select_overlap_contraction(target, source);
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
                    self.apply_overlap_a_first(output, target, source, scratch);
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
                    self.apply_overlap_b_first(output, target, source, scratch);
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
                    source,
                    factors,
                    &mut scratch.afac,
                    &mut scratch.bfac,
                );
                self.apply_overlap_a_first(output, target, source, scratch);
            }
            OverlapContraction::BFirst => {
                Self::gather_overlap_factor_tables(
                    target,
                    source,
                    factors,
                    &mut scratch.afac,
                    &mut scratch.bfac,
                );
                self.apply_overlap_b_first(output, target, source, scratch);
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
        values.clear();
        values.resize(target.targets.len(), 0.0);

        values
            .par_iter_mut()
            .zip(target.targets.par_iter())
            .for_each(|(value, target)| {
                let arow = &afac[target.a * factors.nsa..(target.a + 1) * factors.nsa];
                let brow = &bfac[target.b * factors.nsb..(target.b + 1) * factors.nsb];
                let mut dp = 0.0;

                for entry in &source.entries {
                    let a = source.aids[entry.apos];
                    let b = source.bids[entry.bpos];
                    dp += arow[a] * brow[b] * entry.dn;
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
                    for entry in &source.entries {
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
        source: &ParentUpdates,
    ) -> OverlapContraction {
        let nt = target.targets.len();
        let ne = source.entries.len();
        let nta = target.aids.len();
        let ntb = target.bids.len();
        let nsa = source.aids.len();
        let nsb = source.bids.len();

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
        scratch
            .values
            .resize(self.parents[source.parent].oreps.len(), 0.0);
        scratch.active_oids.clear();

        // Accumulate source D^P entries by occupation ID.
        for entry in &source.entries {
            let sdet = &data.basis[entry.det];
            let sphase = sdet.pha * sdet.phb;
            let oid =
                self.parents[source.parent].oids[entry.det - self.parents[source.parent].first_det];

            scratch.active_oids.push(oid);
            scratch.values[oid] += sphase * entry.dn;
        }

        scratch.active_oids.sort_unstable();
        scratch.active_oids.dedup();

        // Apply target phases only for occupation groups touched by sparse source entries.
        for &oid in &scratch.active_oids {
            let value = scratch.values[oid];
            if value == 0.0 {
                continue;
            }

            let opos = target.opos[oid];
            if opos == usize::MAX {
                continue;
            }

            let group = &target.orthogonal[opos];
            for t in &group.targets {
                output[t.local] += t.phase * value;
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
                for entry in &source.entries {
                    let (a, b) = if target.det <= entry.det {
                        (target.det, entry.det)
                    } else {
                        (entry.det, target.det)
                    };
                    let ldet = &data.basis[a];
                    let gdet = &data.basis[b];
                    let s = if data.input.wicks.enabled && data.wicks.is_none() {
                        calculate_s_pair_naive(data, ldet, gdet)
                    } else {
                        calculate_s_pair(data, DetPair::new(ldet, gdet), Some(wick_scratch))
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
        source: &ParentUpdates,
        factors: &OverlapFactorBlock,
        afac: &mut Vec<f64>,
        bfac: &mut Vec<f64>,
    ) {
        let nta = target.aids.len();
        let ntb = target.bids.len();
        let nsa = source.aids.len();
        let nsb = source.bids.len();
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
                let full = &full_afac[ta * factors.nsa..(ta + 1) * factors.nsa];
                for (col, &sa) in source.aids.iter().enumerate() {
                    row[col] = full[sa];
                }
            });

        bfac.par_chunks_mut(nsb)
            .zip(target.bids.par_iter())
            .for_each(|(row, &tb)| {
                let full = &full_bfac[tb * factors.nsb..(tb + 1) * factors.nsb];
                for (col, &sb) in source.bids.iter().enumerate() {
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
        source: &ParentUpdates,
        scratch: &mut OverlapScratch,
    ) {
        let nta = target.aids.len();
        let nsa = source.aids.len();
        let nsb = source.bids.len();

        scratch.intermediate.clear();
        scratch.intermediate.resize(nta * nsb, 0.0);

        // Form T_{\bar a b} = \sum_a A^{QP}_{\bar a a}D^P_{ab}.
        scratch
            .intermediate
            .par_chunks_mut(nsb)
            .enumerate()
            .for_each(|(ta_pos, row)| {
                let arow = &scratch.afac[ta_pos * nsa..(ta_pos + 1) * nsa];

                for entry in &source.entries {
                    row[entry.bpos] += arow[entry.apos] * entry.dn;
                }
            });

        scratch.values.clear();

        if target.contiguous_locals {
            let first = target.first_local;
            let increments = &mut output[first..first + target.targets.len()];

            // Finish \delta N_w^{QP} = \sum_b T_{\bar a_w b}B^{QP}_{\bar b_w b}.
            increments
                .par_iter_mut()
                .zip(target.targets.par_iter())
                .for_each(|(increment, t)| {
                    let ta_pos = target.apos[t.a];
                    let tb_pos = target.bpos[t.b];

                    let trow = &scratch.intermediate[ta_pos * nsb..(ta_pos + 1) * nsb];
                    let brow = &scratch.bfac[tb_pos * nsb..(tb_pos + 1) * nsb];

                    *increment += dot_f64(trow, brow);
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
                    let brow = &scratch.bfac[tb_pos * nsb..(tb_pos + 1) * nsb];

                    *value = dot_f64(trow, brow);
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
        source: &ParentUpdates,
        scratch: &mut OverlapScratch,
    ) {
        let ntb = target.bids.len();
        let nsa = source.aids.len();
        let nsb = source.bids.len();

        scratch.intermediate.clear();
        scratch.intermediate.resize(ntb * nsa, 0.0);

        // Form U_{a\bar b} = \sum_b D^P_{ab}B^{QP}_{\bar b b}.
        scratch
            .intermediate
            .par_chunks_mut(nsa)
            .enumerate()
            .for_each(|(tb_pos, row)| {
                let brow = &scratch.bfac[tb_pos * nsb..(tb_pos + 1) * nsb];

                for entry in &source.entries {
                    row[entry.apos] += entry.dn * brow[entry.bpos];
                }
            });

        scratch.values.clear();

        if target.contiguous_locals {
            let first = target.first_local;
            let increments = &mut output[first..first + target.targets.len()];

            // Finish \delta N_w^{QP} = \sum_a A^{QP}_{\bar a_w a}U_{a\bar b_w}.
            increments
                .par_iter_mut()
                .zip(target.targets.par_iter())
                .for_each(|(increment, t)| {
                    let ta_pos = target.apos[t.a];
                    let tb_pos = target.bpos[t.b];

                    let arow = &scratch.afac[ta_pos * nsa..(ta_pos + 1) * nsa];
                    let urow = &scratch.intermediate[tb_pos * nsa..(tb_pos + 1) * nsa];

                    *increment += dot_f64(arow, urow);
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

                    let arow = &scratch.afac[ta_pos * nsa..(ta_pos + 1) * nsa];
                    let urow = &scratch.intermediate[tb_pos * nsa..(tb_pos + 1) * nsa];

                    *value = dot_f64(arow, urow);
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
    reps: (&[ReducedOneSpinDetState], &[ReducedOneSpinDetState]),
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
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
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
            basis: data.basis,
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
            aids: Vec::new(),
            bids: Vec::new(),
            apos: vec![usize::MAX; na],
            bpos: vec![usize::MAX; nb],
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
            det,
            apos: self.apos[a],
            bpos: self.bpos[b],
            dn,
        });
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
        self.aids.clear();
        self.bids.clear();
    }
}
