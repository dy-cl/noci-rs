// noci/overlaps.rs

use std::collections::HashMap;

use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};
use rayon::prelude::*;

use super::overlap::{
    calculate_s_alpha_pair_wicks, calculate_s_beta_pair_wicks, calculate_s_pair,
    calculate_s_pair_naive,
};
use super::types::{DetPair, NOCIData};

#[derive(Clone, Copy)]
struct SpinUpdate {
    /// Global determinant index \Omega receiving the pre-overlap update.
    det: usize,
    /// Active source a position for this sparse entry.
    apos: usize,
    /// Active source b position for this sparse entry.
    bpos: usize,
    /// Sparse pre-overlap update value \Delta_\Omega.
    dn: f64,
}

struct ParentUpdates {
    /// Source parent P for all sparse D^P_{ab} entries.
    parent: usize,
    /// Sparse non-zero entries of D^P_{ab}.
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

#[derive(Default)]
struct ParentSpinSpace {
    /// Representative determinant for each parent-local a component.
    areps: Vec<usize>,
    /// Representative determinant for each parent-local b component.
    breps: Vec<usize>,
    /// Representative determinant for each parent-local occupation pair.
    oreps: Vec<usize>,
    /// Occupation-pair ID keyed by determinant offset from `first_det`.
    oids: Vec<usize>,
    /// First determinant index belonging to this parent.
    first_det: usize,
    /// One-past-last determinant index belonging to this parent when parent blocks are contiguous.
    last_det: usize,
}

#[derive(Clone, Copy)]
struct LocalTarget {
    /// Rank-local population row receiving \delta N_\Gamma.
    local: usize,
    /// Global determinant index \Gamma.
    det: usize,
    /// Target-parent local a component ID a_\Gamma.
    a: usize,
    /// Target-parent local b component ID b_\Gamma.
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
    /// Parent-local occupation-pair ID.
    oid: usize,
    /// Targets sharing this occupation pair.
    targets: Vec<OrthogonalTarget>,
}

struct LocalParentBlock {
    /// Target parent Q for all local rows in this block.
    parent: usize,
    /// Rank-local target rows in this parent block.
    targets: Vec<LocalTarget>,
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
    FactorizedRows,
    /// Form T_{\bar a b} before applying B^{QP}_{\bar b_\Gamma b}.
    AFirst,
    /// Form U_{a\bar b} before applying A^{QP}_{\bar a_\Gamma a}.
    BFirst,
}

/// Reusable storage for one application of S\Delta.
pub(crate) struct OverlapFactorScratch {
    /// Sparse updates grouped by source parent.
    updates: Vec<ParentUpdates>,
    /// Source parents touched by the current update list.
    active_parents: Vec<usize>,
    /// Rank-local accumulated \delta N_\Gamma values.
    increments: Vec<f64>,
    /// Temporary A^{QP}_{\bar a a} factor table.
    afac: Vec<f64>,
    /// Temporary B^{QP}_{\bar b b} factor table.
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

/// Precomputed determinant to spin mappings for reusing alpha and beta overlap factors.
pub(crate) struct OverlapFactor {
    /// Alpha compact IDs keyed by determinant index and local to the determinant parent.
    aids: Vec<usize>,
    /// Beta compact IDs keyed by determinant index and local to the determinant parent.
    bids: Vec<usize>,
    /// Largest number of unique alpha spin components in one parent reference.
    ma: usize,
    /// Largest number of unique beta spin components in one parent reference.
    mb: usize,
    /// Parent-local determinant ranges and spin representatives.
    parents: Vec<ParentSpinSpace>,
}

impl OverlapFactor {
    /// Construct parent-local alpha and beta component IDs for every determinant.
    /// # Arguments:
    /// - `data`: Shared NOCI data defining the determinant basis and parent references.
    /// # Returns:
    /// - `OverlapAction`: Immutable sparse overlap action plan.
    pub(crate) fn new(data: &NOCIData<'_, f64>) -> Self {
        // Allocate determinant to ID mappings.
        let mut aids = vec![0usize; data.basis.len()];
        let mut bids = vec![0usize; data.basis.len()];
        // Generate determinant IDs.
        let ma = assign_aids(data, &mut aids);
        let mb = assign_bids(data, &mut bids);
        let parents = build_parent_spin_spaces(data, &aids, &bids);

        Self {
            aids,
            bids,
            ma,
            mb,
            parents,
        }
    }

    /// Construct reusable storage for one full application of S\Delta.
    /// Temporary factor tables and contraction buffers are allocated once, but their numerical
    /// values are cleared or overwritten on every application and are never reused as overlap data.
    /// # Arguments:
    /// - `self`: Immutable sparse overlap action plan.
    /// # Returns:
    /// - `OverlapFactorScratch`: Empty allocation storage for grouped updates and factor tables.
    pub(crate) fn scratch(&self) -> OverlapFactorScratch {
        let mut updates = Vec::with_capacity(self.parents.len());
        for parent in 0..self.parents.len() {
            updates.push(ParentUpdates::new(parent, self.ma, self.mb));
        }
        OverlapFactorScratch {
            updates,
            active_parents: Vec::new(),
            increments: Vec::new(),
            afac: Vec::new(),
            bfac: Vec::new(),
            intermediate: Vec::new(),
            values: Vec::new(),
            cached_targets_ptr: std::ptr::null(),
            cached_targets_len: 0,
            target_blocks: Vec::new(),
        }
    }

    /// Apply \delta N_\Gamma = \sum_\Omega S_{\Gamma\Omega}\Delta_\Omega.
    /// Orthogonal same-parent blocks are applied directly, while cross-parent blocks use
    /// S_{\Gamma\Omega} = A^{QP}_{\bar a_\Gamma a_\Omega}B^{QP}_{\bar b_\Gamma b_\Omega}.
    /// Temporary factor tables are rebuilt for each overlap application and are not cached across iterations.
    /// # Arguments:
    /// - `populations`: Rank-local persistent populations N_\Gamma.
    /// - `targets`: Global determinant index for each rank-local row in `populations`.
    /// - `updates`: Sparse pre-overlap changes \Omega, \Delta_\Omega.
    /// - `data`: Shared NOCI data.
    /// - `scratch`: Reusable allocation storage for one application of S\Delta.
    /// # Returns:
    /// - `()`: Applies N_\Gamma \leftarrow N_\Gamma + \delta N_\Gamma.
    pub(crate) fn apply<I>(
        &self,
        populations: &mut [f64],
        targets: &[usize],
        updates: I,
        data: &NOCIData<'_, f64>,
        scratch: &mut OverlapFactorScratch,
    ) where
        I: IntoIterator<Item = (usize, f64)>,
    {
        if populations.is_empty() {
            return;
        }

        self.group_updates(updates, data, scratch);
        if scratch.active_parents.is_empty() {
            return;
        }

        scratch.increments.clear();
        scratch.increments.resize(populations.len(), 0.0);

        let target_blocks = self.take_target_blocks(targets, data, scratch);

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
                self.apply_parent_pair(target, &source, data, scratch);
            }
            source.clear();
            scratch.updates[source_parent] = source;
        }

        scratch.target_blocks = target_blocks;

        // Apply the completed S\Delta contribution to the persistent populations.
        populations
            .iter_mut()
            .zip(scratch.increments.iter())
            .for_each(|(n, dn)| *n += dn);

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
    fn take_target_blocks(
        &self,
        targets: &[usize],
        data: &NOCIData<'_, f64>,
        scratch: &mut OverlapFactorScratch,
    ) -> Vec<LocalParentBlock> {
        if scratch.target_blocks.is_empty()
            || scratch.cached_targets_ptr != targets.as_ptr()
            || scratch.cached_targets_len != targets.len()
        {
            scratch.cached_targets_ptr = targets.as_ptr();
            scratch.cached_targets_len = targets.len();
            self.build_target_blocks(targets, data)
        } else {
            std::mem::take(&mut scratch.target_blocks)
        }
    }

    /// Group sparse updates by source parent and active spin components.
    /// This constructs D^P_{ab} in sparse form for the current S\Delta application.
    /// # Arguments:
    /// - `updates`: Sparse determinant changes \Omega, \Delta_\Omega.
    /// - `data`: Shared NOCI data used to map determinants to parents.
    /// - `scratch`: Reusable grouped-update storage cleared and refilled for this application.
    /// # Returns:
    /// - `()`: Fills `scratch.updates` and `scratch.active_parents`.
    fn group_updates<I>(
        &self,
        updates: I,
        data: &NOCIData<'_, f64>,
        scratch: &mut OverlapFactorScratch,
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

    /// Build target parent blocks for the rank-local rows receiving S\Delta.
    /// Each block records active target spin components and same-parent occupation groups.
    /// # Arguments:
    /// - `targets`: Global determinant index for each rank-local population row.
    /// - `data`: Shared NOCI data used to read determinant parents and occupations.
    /// # Returns:
    /// - `Vec<LocalParentBlock>`: Non-empty target blocks grouped by parent Q.
    fn build_target_blocks(
        &self,
        targets: &[usize],
        data: &NOCIData<'_, f64>,
    ) -> Vec<LocalParentBlock> {
        let mut blocks = (0..self.parents.len())
            .map(|parent| LocalParentBlock {
                parent,
                targets: Vec::new(),
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
            self.build_orthogonal_groups(block, data);
        }

        blocks.retain(|block| !block.targets.is_empty());
        blocks
    }

    /// Group same-parent orthogonal targets by occupation bitstrings.
    /// Direct same-parent overlap then matches D^P entries by (o_a,o_b) and determinant phases.
    /// # Arguments:
    /// - `block`: Target parent block whose orthogonal groups are rebuilt.
    /// - `data`: Shared NOCI data used to read occupation bitstrings and phases.
    /// # Returns:
    /// - `()`: Fills `block.orthogonal` without storing numerical overlap factors.
    fn build_orthogonal_groups(
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
                    oid,
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
    /// - `source`: Source parent P grouped D^P updates.
    /// - `data`: Shared NOCI data and Wick intermediates.
    /// - `scratch`: Reusable storage for factors, contractions, and output increments.
    /// # Returns:
    /// - `()`: Adds the QP contribution to `scratch.increments`.
    fn apply_parent_pair(
        &self,
        target: &LocalParentBlock,
        source: &ParentUpdates,
        data: &NOCIData<'_, f64>,
        scratch: &mut OverlapFactorScratch,
    ) {
        if target.parent == source.parent
            && let Some(mocache) = data.mocache
            && mocache[target.parent].orthogonal_slater_condon
        {
            self.apply_orthogonal_parent_pair(target, source, data, scratch);
            return;
        }
        if target.parent == source.parent {
            self.apply_sparse_parent_pair(target, source, data, scratch);
            return;
        }

        if !data.input.wicks.enabled {
            self.apply_sparse_parent_pair(target, source, data, scratch);
            return;
        }

        let Some(wicks) = data.wicks else {
            self.apply_sparse_parent_pair(target, source, data, scratch);
            return;
        };

        let contraction = self.select_contraction(target, source);
        match contraction {
            OverlapContraction::FactorizedRows => {
                self.apply_factorized_parent_pair(target, source, data, wicks, scratch)
            }
            OverlapContraction::AFirst | OverlapContraction::BFirst => {
                self.build_factor_tables(target, source, data, wicks, scratch);
                match contraction {
                    OverlapContraction::AFirst => self.apply_a_first(target, source, scratch),
                    OverlapContraction::BFirst => self.apply_b_first(target, source, scratch),
                    OverlapContraction::FactorizedRows => {}
                }
            }
        }
    }

    /// Apply one cross-parent block with target-local sparse-row factor reuse.
    /// For each target determinant, this builds A and B factor vectors once and contracts
    /// \delta N_\Gamma^{QP} = \sum_{(a,b)} A^{QP}_{\bar a a} B^{QP}_{\bar b b} D^P_{ab}.
    /// The direct determinant-pair Wick loop is avoided because it would recompute the same
    /// same-spin factors for every sparse entry sharing a source a or b component.
    /// # Arguments:
    /// - `target`: Rank-local target parent block Q defining \Gamma = (\bar a,\bar b).
    /// - `source`: Source parent P sparse D^P_{ab} entries and active positions.
    /// - `data`: Shared NOCI determinant data.
    /// - `wicks`: Shared Wick intermediates for parent-pair factor evaluation.
    /// - `scratch`: Reusable value storage receiving one output per target row.
    /// # Returns:
    /// - `()`: Adds factorized sparse-row S\Delta values to `scratch.increments`.
    fn apply_factorized_parent_pair(
        &self,
        target: &LocalParentBlock,
        source: &ParentUpdates,
        data: &NOCIData<'_, f64>,
        wicks: &WicksView<f64>,
        scratch: &mut OverlapFactorScratch,
    ) {
        let nsa = source.aids.len();
        let nsb = source.bids.len();

        scratch.values.clear();
        scratch.values.resize(target.targets.len(), 0.0);

        let (lp, gp, target_left) =
            if self.parents[target.parent].first_det <= self.parents[source.parent].first_det {
                (target.parent, source.parent, true)
            } else {
                (source.parent, target.parent, false)
            };

        scratch
            .values
            .par_iter_mut()
            .zip(target.targets.par_iter())
            .for_each_init(
                || {
                    (
                        WickScratchSpin::new(),
                        Vec::<f64>::new(),
                        Vec::<f64>::new(),
                    )
                },
                |(wick_scratch, afac, bfac), (value, t)| {
                    let pair = wicks.pair(lp, gp);
                    afac.resize(nsa, 0.0);
                    bfac.resize(nsb, 0.0);

                    let tadet = self.parents[target.parent].areps[t.a];
                    let tbdet = self.parents[target.parent].breps[t.b];

                    // Construct A^{QP}_{\bar a a} and B^{QP}_{\bar b b} once for this target row.
                    for (pos, &sa) in source.aids.iter().enumerate() {
                        let sdet = self.parents[source.parent].areps[sa];
                        let (ldet, gdet) = if target_left {
                            (&data.basis[tadet], &data.basis[sdet])
                        } else {
                            (&data.basis[sdet], &data.basis[tadet])
                        };

                        afac[pos] =
                            calculate_s_alpha_pair_wicks(ldet, gdet, &pair, wick_scratch);
                    }

                    for (pos, &sb) in source.bids.iter().enumerate() {
                        let sdet = self.parents[source.parent].breps[sb];
                        let (ldet, gdet) = if target_left {
                            (&data.basis[tbdet], &data.basis[sdet])
                        } else {
                            (&data.basis[sdet], &data.basis[tbdet])
                        };

                        bfac[pos] = calculate_s_beta_pair_wicks(ldet, gdet, &pair, wick_scratch);
                    }

                    // Accumulate sparse D^P_{ab} with factors indexed through active positions.
                    let mut dp = 0.0;

                    for entry in &source.entries {
                        let sa_pos = entry.apos;
                        let sb_pos = entry.bpos;

                        dp += afac[sa_pos] * bfac[sb_pos] * entry.dn;
                    }

                    *value = dp;
                },
            );

        for (value, target) in scratch.values.iter().zip(target.targets.iter()) {
            if *value != 0.0 {
                scratch.increments[target.local] += value;
            }
        }
    }

    /// Select how to apply one cross-parent block of S\Delta.
    /// The row path factorises each target as
    /// \delta N_\Gamma^{QP} = \sum_{(a,b)} A^{QP}_{\bar a a} B^{QP}_{\bar b b} D^P_{ab}.
    /// The direct determinant-pair Wick loop is avoided because the weighted model accounts for
    /// same-spin factor reuse instead of charging every sparse product as a full overlap.
    /// Scores are C = 32\,F + M, where F is the number of same-spin Wick factors and M is the
    /// number of scalar sparse products; the factor weight reflects that one same-spin Wick
    /// evaluation is substantially more expensive than one multiply-add.
    /// # Arguments:
    /// - `target`: Rank-local target parent block.
    /// - `source`: Sparse source-parent D^P entries and active spin IDs.
    /// # Returns:
    /// - `OverlapContraction`: `FactorizedRows`, `AFirst`, or `BFirst` selected by weighted score.
    fn select_contraction(
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
            OverlapContraction::FactorizedRows
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
    /// - `source`: Same-parent source updates D^P_{ab}.
    /// - `data`: Shared NOCI data used to read source occupations and phases.
    /// - `scratch`: Reusable values and increment storage.
    /// # Returns:
    /// - `()`: Adds the same-parent orthogonal contribution to `scratch.increments`.
    fn apply_orthogonal_parent_pair(
        &self,
        target: &LocalParentBlock,
        source: &ParentUpdates,
        data: &NOCIData<'_, f64>,
        scratch: &mut OverlapFactorScratch,
    ) {
        scratch.values.clear();
        scratch
            .values
            .resize(self.parents[source.parent].oreps.len(), 0.0);

        // Accumulate source D^P entries by occupation ID.
        for entry in &source.entries {
            let sdet = &data.basis[entry.det];
            let sphase = sdet.pha * sdet.phb;
            let oid =
                self.parents[source.parent].oids[entry.det - self.parents[source.parent].first_det];
            scratch.values[oid] += sphase * entry.dn;
        }

        // Apply target phases to the occupation-group contribution.
        for group in &target.orthogonal {
            let value = scratch.values[group.oid];
            if value == 0.0 {
                continue;
            }
            for t in &group.targets {
                scratch.increments[t.local] += t.phase * value;
            }
        }
    }

    /// Apply one parent block by direct sparse rows.
    /// This fallback is selected for sparse updates and non-Wick overlap evaluation.
    /// # Arguments:
    /// - `target`: Rank-local target parent block.
    /// - `source`: Sparse source-parent D^P entries.
    /// - `data`: Shared NOCI data used by the general overlap evaluator.
    /// - `scratch`: Reusable per-target value and increment storage.
    /// # Returns:
    /// - `()`: Adds sparse-row S\Delta values to `scratch.increments`.
    fn apply_sparse_parent_pair(
        &self,
        target: &LocalParentBlock,
        source: &ParentUpdates,
        data: &NOCIData<'_, f64>,
        scratch: &mut OverlapFactorScratch,
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
                scratch.increments[target.local] += value;
            }
        }
    }

    /// Build A^{QP}_{\bar a a} and B^{QP}_{\bar b b} for active spin components.
    /// Each factor value is recomputed for this S\Delta application using representative determinants.
    /// # Arguments:
    /// - `target`: Target parent block defining active \bar a and \bar b rows.
    /// - `source`: Source parent updates defining active a and b columns.
    /// - `data`: Shared NOCI determinant data.
    /// - `wicks`: Shared Wick intermediates for parent-pair factor evaluation.
    /// - `scratch`: Reusable `afac` and `bfac` storage overwritten for this parent pair.
    /// # Returns:
    /// - `()`: Fills `scratch.afac` and `scratch.bfac` for the current QP block.
    fn build_factor_tables(
        &self,
        target: &LocalParentBlock,
        source: &ParentUpdates,
        data: &NOCIData<'_, f64>,
        wicks: &WicksView<f64>,
        scratch: &mut OverlapFactorScratch,
    ) {
        let nta = target.aids.len();
        let ntb = target.bids.len();
        let nsa = source.aids.len();
        let nsb = source.bids.len();

        scratch.afac.clear();
        scratch.bfac.clear();
        scratch.afac.resize(nta * nsa, 0.0);
        scratch.bfac.resize(ntb * nsb, 0.0);

        let (lp, gp, target_left) =
            if self.parents[target.parent].first_det <= self.parents[source.parent].first_det {
                (target.parent, source.parent, true)
            } else {
                (source.parent, target.parent, false)
            };

        // Build A rows independently; each task owns one output row.
        scratch
            .afac
            .par_chunks_mut(nsa)
            .zip(target.aids.par_iter())
            .for_each_init(WickScratchSpin::new, |wick_scratch, (row, &ta)| {
                let pair = wicks.pair(lp, gp);
                let tdet = self.parents[target.parent].areps[ta];
                for (col, &sa) in source.aids.iter().enumerate() {
                    let sdet = self.parents[source.parent].areps[sa];
                    let (ldet, gdet) = if target_left {
                        (&data.basis[tdet], &data.basis[sdet])
                    } else {
                        (&data.basis[sdet], &data.basis[tdet])
                    };
                    row[col] = calculate_s_alpha_pair_wicks(ldet, gdet, &pair, wick_scratch);
                }
            });

        // Build B rows independently; each task owns one output row.
        scratch
            .bfac
            .par_chunks_mut(nsb)
            .zip(target.bids.par_iter())
            .for_each_init(WickScratchSpin::new, |wick_scratch, (row, &tb)| {
                let pair = wicks.pair(lp, gp);
                let tdet = self.parents[target.parent].breps[tb];
                for (col, &sb) in source.bids.iter().enumerate() {
                    let sdet = self.parents[source.parent].breps[sb];
                    let (ldet, gdet) = if target_left {
                        (&data.basis[tdet], &data.basis[sdet])
                    } else {
                        (&data.basis[sdet], &data.basis[tdet])
                    };
                    row[col] = calculate_s_beta_pair_wicks(ldet, gdet, &pair, wick_scratch);
                }
            });
    }

    /// Apply T_{\bar a b} = \sum_a A^{QP}_{\bar a a}D^P_{ab}.
    /// The final target rows multiply T by B^{QP}_{\bar b_\Gamma b}.
    /// # Arguments:
    /// - `target`: Target parent block defining \bar a_\Gamma and \bar b_\Gamma rows.
    /// - `source`: Sparse source-parent D^P entries and active positions.
    /// - `scratch`: Reusable factor, intermediate, value, and increment storage.
    /// # Returns:
    /// - `()`: Adds the A-first blocked contribution to `scratch.increments`.
    fn apply_a_first(
        &self,
        target: &LocalParentBlock,
        source: &ParentUpdates,
        scratch: &mut OverlapFactorScratch,
    ) {
        let nta = target.aids.len();
        let nsb = source.bids.len();
        let nsa = source.aids.len();

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
                    let sa_pos = entry.apos;
                    let sb_pos = entry.bpos;

                    row[sb_pos] += arow[sa_pos] * entry.dn;
                }
            });

        scratch.values.clear();
        scratch.values.resize(target.targets.len(), 0.0);

        // Finish \delta N_\Gamma^{QP} = \sum_b T_{\bar a_\Gamma b}B^{QP}_{\bar b_\Gamma b}.
        scratch
            .values
            .par_iter_mut()
            .zip(target.targets.par_iter())
            .for_each(|(value, t)| {
                let ta_pos = target.apos[t.a];
                let tb_pos = target.bpos[t.b];
                let trow = &scratch.intermediate[ta_pos * nsb..(ta_pos + 1) * nsb];
                let brow = &scratch.bfac[tb_pos * nsb..(tb_pos + 1) * nsb];
                *value = trow.iter().zip(brow.iter()).map(|(x, y)| x * y).sum();
            });

        for (value, target) in scratch.values.iter().zip(target.targets.iter()) {
            if *value != 0.0 {
                scratch.increments[target.local] += value;
            }
        }
    }

    /// Apply U_{a\bar b} = \sum_b D^P_{ab}B^{QP}_{\bar b b}.
    /// The final target rows multiply U by A^{QP}_{\bar a_\Gamma a}.
    /// # Arguments:
    /// - `target`: Target parent block defining \bar a_\Gamma and \bar b_\Gamma rows.
    /// - `source`: Sparse source-parent D^P entries and active positions.
    /// - `scratch`: Reusable factor, intermediate, value, and increment storage.
    /// # Returns:
    /// - `()`: Adds the B-first blocked contribution to `scratch.increments`.
    fn apply_b_first(
        &self,
        target: &LocalParentBlock,
        source: &ParentUpdates,
        scratch: &mut OverlapFactorScratch,
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
                    let sa_pos = entry.apos;
                    let sb_pos = entry.bpos;

                    row[sa_pos] += entry.dn * brow[sb_pos];
                }
            });

        scratch.values.clear();
        scratch.values.resize(target.targets.len(), 0.0);

        // Finish \delta N_\Gamma^{QP} = \sum_a A^{QP}_{\bar a_\Gamma a}U_{a\bar b_\Gamma}.
        scratch
            .values
            .par_iter_mut()
            .zip(target.targets.par_iter())
            .for_each(|(value, t)| {
                let ta_pos = target.apos[t.a];
                let tb_pos = target.bpos[t.b];
                let arow = &scratch.afac[ta_pos * nsa..(ta_pos + 1) * nsa];
                let urow = &scratch.intermediate[tb_pos * nsa..(tb_pos + 1) * nsa];
                *value = arow.iter().zip(urow.iter()).map(|(x, y)| x * y).sum();
            });

        for (value, target) in scratch.values.iter().zip(target.targets.iter()) {
            if *value != 0.0 {
                scratch.increments[target.local] += value;
            }
        }
    }
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
    /// - `ParentUpdates`: Empty D^P storage with inactive position maps.
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

    /// Add one sparse D^P_{ab} entry and record active spin IDs on first occurrence.
    /// # Arguments:
    /// - `det`: Source determinant \Omega.
    /// - `a`: Source-parent local a component ID a_\Omega.
    /// - `b`: Source-parent local b component ID b_\Omega.
    /// - `dn`: Sparse pre-overlap update \Delta_\Omega.
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

    /// Clear D^P_{ab} while invalidating only IDs active in the last application.
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

/// Assign compact alpha IDs by sorting determinant indices and deduplicating consecutive identities.
/// # Arguments:
/// - `data`: Shared NOCI data defining the determinant basis.
/// - `aids`: Output alpha compact IDs keyed by determinant index.
/// # Returns:
/// - `usize`: Largest number of unique alpha components in any parent.
fn assign_aids(
    data: &NOCIData<'_, f64>,
    aids: &mut [usize],
) -> usize {
    let mut indices = (0..data.basis.len()).collect::<Vec<_>>();

    // Sort by the alpha ID key such that equivalent determinants are consecutive.
    indices.sort_unstable_by(|&i, &j| {
        let id = &data.basis[i];
        let jd = &data.basis[j];

        // Sort by the ID key:
        // Parent reference, alpha occupation bitstring,
        // alpha holes, alpha particles, and alpha phase.
        id.parent
            .cmp(&jd.parent)
            .then_with(|| id.oa.cmp(&jd.oa))
            .then_with(|| id.excitation.alpha.holes.cmp(&jd.excitation.alpha.holes))
            .then_with(|| id.excitation.alpha.parts.cmp(&jd.excitation.alpha.parts))
            .then_with(|| id.pha.to_bits().cmp(&jd.pha.to_bits()))
    });

    // Previous determinant in sorted alpha key order.
    let mut last = usize::MAX;
    // Next unused alpha ID and number of unique components for a given parent.
    let mut next = 0usize;
    // Largest number of unique alpha components for any parent.
    let mut maxu = 0usize;

    // Enumerate sorted determinants.
    for (pos, &det) in indices.iter().enumerate() {
        let parent = data.basis[det].parent;
        // Detect first determinant or a new parent.
        if pos == 0 || parent != data.basis[last].parent {
            // If not the first determinant check if a new maximum number
            // of unique alpha components has been found for the previous parent.
            if pos != 0 {
                maxu = maxu.max(next);
            }
            // Reset IDs for the new parent.
            next = 0;
            aids[det] = next;
            next += 1;
        // Otherwise we have current determinant same parent as previous.
        } else {
            let prev = &data.basis[last];
            let curr = &data.basis[det];

            // If all keys match they get the same alpha ID.
            if prev.oa == curr.oa
                && prev.excitation.alpha.holes == curr.excitation.alpha.holes
                && prev.excitation.alpha.parts == curr.excitation.alpha.parts
                && prev.pha.to_bits() == curr.pha.to_bits()
            {
                aids[det] = aids[last];
            // Assign a new ID otherwise.
            } else {
                aids[det] = next;
                next += 1;
            }
        }
        last = det;
    }

    maxu.max(next)
}

/// Assign compact beta IDs by sorting determinant indices and deduplicating consecutive identities.
/// # Arguments:
/// - `data`: Shared NOCI data defining the determinant basis.
/// - `bids`: Output beta compact IDs keyed by determinant index.
/// # Returns:
/// - `usize`: Largest number of unique beta components in any parent.
fn assign_bids(
    data: &NOCIData<'_, f64>,
    bids: &mut [usize],
) -> usize {
    let mut indices = (0..data.basis.len()).collect::<Vec<_>>();

    // Sort by the beta ID key such that equivalent determinants are consecutive.
    indices.sort_unstable_by(|&i, &j| {
        let id = &data.basis[i];
        let jd = &data.basis[j];

        // Sort by the ID key:
        // Parent reference, beta occupation bitstring,
        // beta holes, beta particles, and beta phase.
        id.parent
            .cmp(&jd.parent)
            .then_with(|| id.ob.cmp(&jd.ob))
            .then_with(|| id.excitation.beta.holes.cmp(&jd.excitation.beta.holes))
            .then_with(|| id.excitation.beta.parts.cmp(&jd.excitation.beta.parts))
            .then_with(|| id.phb.to_bits().cmp(&jd.phb.to_bits()))
    });

    // Previous determinant in sorted beta key order.
    let mut last = usize::MAX;
    // Next unused beta ID and number of unique components for a given parent.
    let mut next = 0usize;
    // Largest number of unique beta components for any parent.
    let mut maxu = 0usize;

    // Enumerate sorted determinants.
    for (pos, &det) in indices.iter().enumerate() {
        let parent = data.basis[det].parent;

        // Detect first determinant or a new parent.
        if pos == 0 || parent != data.basis[last].parent {
            // If not the first determinant check if a new maximum number
            // of unique beta components has been found for the previous parent.
            if pos != 0 {
                maxu = maxu.max(next);
            }

            // Reset IDs for the new parent.
            next = 0;
            bids[det] = next;
            next += 1;
        // Otherwise we have current determinant same parent as previous.
        } else {
            let prev = &data.basis[last];
            let curr = &data.basis[det];

            // If all keys match they get the same beta ID.
            if prev.ob == curr.ob
                && prev.excitation.beta.holes == curr.excitation.beta.holes
                && prev.excitation.beta.parts == curr.excitation.beta.parts
                && prev.phb.to_bits() == curr.phb.to_bits()
            {
                bids[det] = bids[last];
            // Assign a new ID otherwise.
            } else {
                bids[det] = next;
                next += 1;
            }
        }

        last = det;
    }

    maxu.max(next)
}

/// Build parent-local spin and occupation representative tables.
/// The a and b representatives provide determinant rows for A^{QP}_{\bar a a} and
/// B^{QP}_{\bar b b}; occupation IDs provide direct same-parent orthogonal matching by existing
/// determinant `oa` and `ob` bitstrings. These tables contain only determinant IDs and parent-local
/// IDs, not overlap factors, so no numerical S values persist across QMC iterations.
/// # Arguments:
/// - `data`: Shared NOCI data defining determinant parents.
/// - `aids`: Parent-local a component IDs keyed by determinant.
/// - `bids`: Parent-local b component IDs keyed by determinant.
/// # Returns:
/// - `Vec<ParentSpinSpace>`: Per-parent determinant ranges, spin representatives, occupation representatives, and occupation IDs.
fn build_parent_spin_spaces(
    data: &NOCIData<'_, f64>,
    aids: &[usize],
    bids: &[usize],
) -> Vec<ParentSpinSpace> {
    let nparents = data
        .basis
        .iter()
        .map(|det| det.parent)
        .max()
        .map(|parent| parent + 1)
        .unwrap_or(0);
    let mut parents = (0..nparents)
        .map(|_| ParentSpinSpace {
            areps: Vec::new(),
            breps: Vec::new(),
            oreps: Vec::new(),
            oids: Vec::new(),
            first_det: usize::MAX,
            last_det: 0,
        })
        .collect::<Vec<_>>();

    for (det, state) in data.basis.iter().enumerate() {
        let parent = &mut parents[state.parent];
        parent.first_det = parent.first_det.min(det);
        parent.last_det = parent.last_det.max(det + 1);

        // Store the first determinant representative for this a component.
        if parent.areps.len() <= aids[det] {
            parent.areps.resize(aids[det] + 1, usize::MAX);
        }
        if parent.areps[aids[det]] == usize::MAX {
            parent.areps[aids[det]] = det;
        }

        // Store the first determinant representative for this b component.
        if parent.breps.len() <= bids[det] {
            parent.breps.resize(bids[det] + 1, usize::MAX);
        }
        if parent.breps[bids[det]] == usize::MAX {
            parent.breps[bids[det]] = det;
        }
    }

    for parent in &mut parents {
        if parent.first_det != usize::MAX {
            parent
                .oids
                .resize(parent.last_det - parent.first_det, usize::MAX);
        }
    }

    let mut occupation_ids = (0..nparents)
        .map(|_| HashMap::new())
        .collect::<Vec<HashMap<(u128, u128), usize>>>();

    for (det, state) in data.basis.iter().enumerate() {
        let parent = &mut parents[state.parent];
        let oid = *occupation_ids[state.parent]
            .entry((state.oa, state.ob))
            .or_insert_with(|| {
                parent.oreps.push(det);
                parent.oreps.len() - 1
            });

        // Store the occupation ID in determinant order for direct same-parent matching.
        parent.oids[det - parent.first_det] = oid;
    }

    parents
}
