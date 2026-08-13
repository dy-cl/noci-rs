// noci/factorise/onebody/gpu/backend.rs
//! GPU backend for spin-factorised one-body NOCI operator contractions.

// Standard library imports.
use std::any::TypeId;
use std::marker::PhantomData;
use std::path::Path;

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::gpu::{GpuBuffer, GpuContext};
use crate::input::SNOCIStorage;
use crate::noci::types::{FockData, NOCIData, NOCIScalar};
use crate::nonorthogonalwicks::gpu::DeviceWicksShared;
use crate::nonorthogonalwicks::{WicksRequirements, gpu};

// Parent/sibling imports.
use super::super::super::{SpinFactorisation, ordered_parent_pair};
use super::super::plan::{OneBodyBlockPlan, OneBodyContraction, OneBodyPlan};
use super::consts::{
    GEMM_TILE_DIM, GROUPED_FINAL_BLOCK_FIELDS, GROUPED_FINAL_CONTRIBUTION_FIELDS,
    GROUPED_STAGE_BLOCK_FIELDS, GROUPED_STAGE_CONTRIBUTION_FIELDS,
};
use super::contract::{
    AFirstFinalLaunch, AFirstStageLaunch, BFirstFinalLaunch, BFirstStageLaunch, GroupedRankLaunch,
    launch_a_first_final, launch_a_first_stage, launch_b_first_final, launch_b_first_stage,
    launch_gather_rank_block, launch_grouped_rank_final, launch_grouped_rank_stage,
    launch_zero_f64,
};
use super::data::{DeviceOneBodyData, GpuOneBodyData};
use super::diagonals::{DiagonalBlockLaunch, launch_fill_one_body_diagonal_block};
use super::factors::{
    DiagonalFactorOutput, DiagonalFactorRequest, FactorOutput, FactorRequest,
    build_spin_one_body_diagonal_factors, build_spin_one_body_factors,
};
use super::orthogonal::{
    GpuOrthogonalBlocks, launch_apply_orthogonal_block, launch_fill_orthogonal_diagonal,
};

/// CubeCL factorised one-body backend for the current generalised Fock.
pub(crate) struct GpuOneBodyBackend<T: NOCIScalar> {
    /// Common CubeCL context descriptor.
    context: GpuContext,
    /// Shared determinant-space factorisation `I <-> (P,a_I,b_I)`.
    spin: SpinFactorisation,
    /// Shared one-body topology and contraction plan.
    plan: OneBodyPlan,
    /// Host-known zero-overlap counts using the device Wick slot ordering.
    wick_m: Vec<usize>,
    /// Device-resident real Wick data.
    device_wicks: DeviceWicksShared,
    /// Factorised-operator GPU topology data.
    data: GpuOneBodyData,
    /// Sparse same-parent orthogonal Slater-Condon blocks.
    orthogonal: GpuOrthogonalBlocks,
    /// Device-resident determinant topology and decoded excitations.
    device_data: DeviceOneBodyData,
    /// Reusable device buffers for factor panels, intermediates and vectors.
    scratch: GpuOneBodyScratch,
    /// Backend-neutral CubeCL allocation layout for each contraction region.
    workspace: WorkspaceLayout,
    /// Marker preserving the scalar type checked at construction.
    marker: PhantomData<T>,
}

impl<T: NOCIScalar + 'static> GpuOneBodyBackend<T> {
    /// Build GPU-resident topology and Wick data for the current generalised Fock operator.
    /// Factor tables remain transient and `storage` must therefore be `SNOCIStorage::None`.
    /// # Arguments:
    /// - `data`: Shared NOCI data with Wick intermediates for the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data already reflected in Wick intermediates.
    /// - `_cache`: Unused persistent-cache directory.
    /// - `_rank`: Unused MPI cache rank.
    /// - `_iteration`: Unused SNOCI cache iteration.
    /// - `storage`: Requested factor-table storage strategy.
    /// # Returns
    /// - `GpuOneBodyBackend<T>`: GPU one-body backend.
    pub(crate) fn new(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        _cache: &Path,
        _rank: i32,
        _iteration: usize,
        storage: SNOCIStorage,
    ) -> Self {
        if TypeId::of::<T>() != TypeId::of::<f64>() {
            eprintln!("snoci.backend = \"gpu\" currently supports real f64 NOCI-PT2 data only");
            std::process::exit(1);
        }

        if !matches!(storage, SNOCIStorage::None) {
            eprintln!("snoci.backend = \"gpu\" requires snoci.gmres.factor_tables = \"none\"");
            std::process::exit(1);
        }

        let Some(wicks) = data.wicks else {
            eprintln!("snoci.backend = \"gpu\" requires Wick intermediates");
            std::process::exit(1);
        };

        let spin = SpinFactorisation::new(data);
        let plan = OneBodyPlan::new(&spin, fock);
        let requirements = WicksRequirements::one_body();
        let wicks = gpu::pack_wicks(wicks, requirements);
        let wick_m = collect_wick_m(&wicks, plan.nparent);
        let gpu_data = GpuOneBodyData::new(&spin, data);
        let context = GpuContext::new();
        let workspace = WorkspaceLayout::from_context(&context);
        let scratch =
            GpuOneBodyScratch::with_contraction_regions(&context, workspace.region_elements);
        let orthogonal = GpuOrthogonalBlocks::new(&context, &spin, data, fock);
        let device_wicks = upload_real_wicks(&wicks, &context);
        let device_data = gpu_data.upload(&context);

        Self {
            context,
            spin,
            plan,
            wick_m,
            device_wicks,
            data: gpu_data,
            orthogonal,
            device_data,
            scratch,
            workspace,
            marker: PhantomData,
        }
    }

    /// Print the resolved CubeCL workspace and representative grouped-contraction plan.
    /// # Returns
    /// - `()`: Writes one human-readable GPU memory summary to standard output.
    pub(crate) fn report_memory_configuration(&self) {
        println!("{}", "=".repeat(100));
        println!("GPU memory configuration");
        println!("{}", "-".repeat(100));

        for index in 0..self.plan.blocks.len() {
            if matches!(
                self.plan.blocks[index],
                OneBodyBlockPlan::NonOrthogonal { .. }
            ) {
                let block = self.gpu_nonorthogonal_block(index);
                if self.rank_block_dense(block.target_parent)
                    && self.rank_block_dense(block.source_parent)
                {
                    self.report_grouped_plan(block);
                    return;
                }
            }
        }

        let region_mib = self.workspace.region_bytes as f64 / (1024.0 * 1024.0);
        let backing_mib = self.workspace.backing_bytes as f64 / (1024.0 * 1024.0);
        let page_mib = self.workspace.page_bytes as f64 / (1024.0 * 1024.0);
        println!("  Grouped GPU contraction: unavailable");
        println!("  CubeCL workspace region (MiB): {region_mib:.3}");
        println!("  CubeCL sliced-page size (MiB): {page_mib:.3}");
        println!("  CubeCL backing pages: {}", self.workspace.backing_pages);
        println!("  CubeCL backing demand (MiB): {backing_mib:.3}");
    }

    /// Apply `Y = (F + lambda S)x` using transient GPU factor generation and dense contractions.
    /// No complete candidate-candidate matrix or persistent factor table is materialised.
    /// # Arguments:
    /// - `x`: Source vector over actual candidate determinants.
    /// - `_data`: Shared NOCI data already represented by the GPU topology.
    /// - `_fock`: Current generalised-Fock data already represented by GPU Wick data.
    /// - `lambda`: Scalar multiplying the overlap contribution.
    /// - `partition`: Worker index and worker count for distributed target rows.
    /// # Returns
    /// - `Array1<T>`: Partial or complete determinant-space action.
    pub(crate) fn apply_one_body(
        &mut self,
        x: &Array1<T>,
        _data: &NOCIData<'_, T>,
        _fock: &FockData<'_, T>,
        lambda: T,
        partition: (usize, usize),
    ) -> Array1<T> {
        let lambda = real_scalar(lambda);
        let xs = x
            .as_slice_memory_order()
            .expect("NOCI-PT2 vector must be contiguous.");
        let xs = real_slice(xs);

        self.scratch.x = Some(GpuBuffer::from_slice(&self.context, xs));
        self.scratch.ensure_y(&self.context, x.len());
        self.scratch
            .ensure_packed(&self.context, self.data.dense_rank_dets.len());

        if !self.data.dense_rank_dets.is_empty() {
            launch_gather_rank_block(
                &self.context,
                &self.device_data,
                self.scratch.x.as_ref().expect("GPU x buffer must exist"),
                self.scratch
                    .packed
                    .as_ref()
                    .expect("GPU packed coefficient buffer must exist"),
                0,
                self.data.dense_rank_dets.len(),
            );
        }

        let y = self.scratch.y.as_ref().expect("GPU y buffer must exist");
        launch_zero_f64(&self.context, y, x.len());

        for index in 0..self.plan.blocks.len() {
            match &self.plan.blocks[index] {
                OneBodyBlockPlan::Orthogonal { parent } => {
                    launch_apply_orthogonal_block(
                        &self.context,
                        self.orthogonal.block(*parent),
                        self.scratch.x.as_ref().expect("GPU x buffer must exist"),
                        self.scratch.y.as_ref().expect("GPU y buffer must exist"),
                        lambda,
                        partition,
                    );
                }
                OneBodyBlockPlan::NonOrthogonal { .. } => {
                    let block = self.gpu_nonorthogonal_block(index);
                    self.apply_factorised_block(block, lambda, partition);
                }
            }
        }

        let values = self
            .scratch
            .y
            .as_ref()
            .expect("GPU y buffer must exist")
            .read(&self.context);

        Array1::from_vec(values.into_iter().map(T::from_real).collect())
    }

    /// Build the diagonal of `F + lambda S` and the diagonal of `S`.
    /// Nonorthogonal same-parent blocks construct only same-component spin factors.
    /// # Arguments:
    /// - `_data`: Shared NOCI data already represented by GPU topology.
    /// - `_fock`: Current generalised-Fock data already represented by GPU Wick data.
    /// - `lambda`: Scalar multiplying the overlap operator.
    /// # Returns
    /// - `(Array1<T>, Array1<T>)`: Shifted-Fock and overlap diagonals.
    pub(crate) fn one_body_diagonals(
        &mut self,
        _data: &NOCIData<'_, T>,
        _fock: &FockData<'_, T>,
        lambda: T,
    ) -> (Array1<T>, Array1<T>) {
        let lambda = real_scalar(lambda);
        let ndet = self.data.entry_det.len();

        self.scratch.ensure_m_diag(&self.context, ndet);
        self.scratch.ensure_s_diag(&self.context, ndet);

        let m_diag = self
            .scratch
            .m_diag
            .as_ref()
            .expect("GPU diagonal buffer must exist");
        let s_diag = self
            .scratch
            .s_diag
            .as_ref()
            .expect("GPU diagonal buffer must exist");

        launch_zero_f64(&self.context, m_diag, ndet);
        launch_zero_f64(&self.context, s_diag, ndet);

        for parent in 0..self.spin.parents.len() {
            let index = parent * self.plan.nparent + parent;

            match &self.plan.blocks[index] {
                OneBodyBlockPlan::Orthogonal { .. } => {
                    launch_fill_orthogonal_diagonal(
                        &self.context,
                        self.orthogonal.block(parent),
                        self.scratch
                            .m_diag
                            .as_ref()
                            .expect("GPU diagonal buffer must exist"),
                        self.scratch
                            .s_diag
                            .as_ref()
                            .expect("GPU diagonal buffer must exist"),
                        lambda,
                    );
                }
                OneBodyBlockPlan::NonOrthogonal { .. } => {
                    self.fill_parent_diagonal(parent, lambda);
                }
            }
        }

        let m = self
            .scratch
            .m_diag
            .as_ref()
            .expect("GPU diagonal buffer must exist")
            .read(&self.context)
            .into_iter()
            .map(T::from_real)
            .collect();

        let s = self
            .scratch
            .s_diag
            .as_ref()
            .expect("GPU diagonal buffer must exist")
            .read(&self.context)
            .into_iter()
            .map(T::from_real)
            .collect();

        (Array1::from_vec(m), Array1::from_vec(s))
    }

    /// Convert one shared plan entry into a GPU block and select its transient contraction order.
    /// The GPU ordering minimises repeated transient factor generation under the current panel budget.
    /// # Arguments:
    /// - `index`: Shared plan index `Q * nparent + P`.
    /// # Returns
    /// - `GpuFactorisedBlock`: GPU parent-pair block with transient contraction order.
    fn gpu_nonorthogonal_block(
        &self,
        index: usize,
    ) -> GpuFactorisedBlock {
        match self.plan.blocks[index] {
            OneBodyBlockPlan::NonOrthogonal {
                target_parent,
                source_parent,
                lp,
                gp,
                target_left,
                nta,
                ntb,
                nsa,
                nsb,
                contraction,
            } => {
                let mut block = GpuFactorisedBlock {
                    target_parent,
                    source_parent,
                    lp,
                    gp,
                    target_left,
                    nta,
                    ntb,
                    nsa,
                    nsb,
                    contraction,
                };

                block.contraction = select_gpu_contraction(
                    block,
                    self.workspace.region_elements,
                    self.workspace.region_elements,
                );
                block
            }
            OneBodyBlockPlan::Orthogonal { .. } => {
                unreachable!("orthogonal GPU blocks must use Slater-Condon")
            }
        }
    }

    /// Apply one nonorthogonal parent-pair block in the selected dense contraction order.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: Worker index and worker count.
    /// # Returns
    /// - `()`: Accumulates the parent-pair action into device `y`.
    fn apply_factorised_block(
        &mut self,
        block: GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
    ) {
        if self.rank_block_dense(block.target_parent) && self.rank_block_dense(block.source_parent)
        {
            match block.contraction {
                OneBodyContraction::AFirst => {
                    self.apply_a_first_grouped_rank_blocks(block, lambda, partition)
                }
                OneBodyContraction::BFirst => {
                    self.apply_b_first_grouped_rank_blocks(block, lambda, partition)
                }
            }
            return;
        }
        match block.contraction {
            OneBodyContraction::AFirst => self.apply_a_first_block(block, lambda, partition),
            OneBodyContraction::BFirst => self.apply_b_first_block(block, lambda, partition),
        }
    }

    /// Test whether every populated joint-rank sector of one parent is exactly Cartesian dense.
    /// # Arguments:
    /// - `parent`: Parent reference.
    /// # Returns
    /// - `bool`: Whether rank-block GEMM can represent the complete actual determinant space.
    fn rank_block_dense(
        &self,
        parent: usize,
    ) -> bool {
        self.data.rank_blocks[parent]
            .iter()
            .all(|block| block.dense)
    }

    /// Print the pooled-memory plan and its estimated Wick factor-generation work once.
    /// The estimate counts every generated same-spin factor entry, including regeneration
    /// of the inner-spin factor table for each outer target panel.
    /// # Arguments:
    /// - `block`: Dense parent-pair dimensions and selected contraction order.
    /// # Returns
    /// - `()`: Writes one human-readable grouped-plan summary to standard output.
    fn report_grouped_plan(
        &self,
        block: GpuFactorisedBlock,
    ) {
        let (label, panels) = match block.contraction {
            OneBodyContraction::AFirst => (
                "alpha-first",
                grouped_plan(
                    block.nta,
                    block.nsa,
                    block.nsb,
                    block.ntb,
                    block.nsb,
                    self.workspace.region_elements,
                    self.workspace.region_elements,
                ),
            ),
            OneBodyContraction::BFirst => (
                "beta-first",
                grouped_plan(
                    block.ntb,
                    block.nsb,
                    block.nsa,
                    block.nta,
                    block.nsa,
                    self.workspace.region_elements,
                    self.workspace.region_elements,
                ),
            ),
        };
        let outer_target = match block.contraction {
            OneBodyContraction::AFirst => block.nta,
            OneBodyContraction::BFirst => block.ntb,
        };
        let inner_target = match block.contraction {
            OneBodyContraction::AFirst => block.ntb,
            OneBodyContraction::BFirst => block.nta,
        };
        let outer_panels = outer_target.div_ceil(panels.outer_rows);
        let inner_panels = inner_target.div_ceil(panels.inner_rows);
        let source_columns = match block.contraction {
            OneBodyContraction::AFirst => block.nsb,
            OneBodyContraction::BFirst => block.nsa,
        };
        let reduction_columns = match block.contraction {
            OneBodyContraction::AFirst => block.nsa,
            OneBodyContraction::BFirst => block.nsb,
        };
        let source_panels = source_columns.div_ceil(panels.intermediate_columns);
        let reduction_panels = reduction_columns.div_ceil(panels.reduction_columns);
        let contraction_groups = outer_panels
            .saturating_mul(source_panels)
            .saturating_mul(reduction_panels.saturating_add(inner_panels));
        let alpha_table = block.nta.saturating_mul(block.nsa);
        let beta_table = block.ntb.saturating_mul(block.nsb);
        let (alpha_work, beta_work) = match block.contraction {
            OneBodyContraction::AFirst => (
                source_panels.saturating_mul(alpha_table),
                outer_panels.saturating_mul(beta_table),
            ),
            OneBodyContraction::BFirst => (
                outer_panels.saturating_mul(alpha_table),
                source_panels.saturating_mul(beta_table),
            ),
        };
        let work = alpha_work.saturating_add(beta_work);
        let table_scale = alpha_table.min(beta_table).max(1);
        let equivalents = work as f64 / table_scale as f64;
        let region_mib = self.workspace.region_bytes as f64 / (1024.0 * 1024.0);
        let backing_mib = self.workspace.backing_bytes as f64 / (1024.0 * 1024.0);
        let page_mib = self.workspace.page_bytes as f64 / (1024.0 * 1024.0);

        println!("  Grouped GPU contraction: streamed");
        println!("  Contraction order: {label}");
        println!("  CubeCL workspace region (MiB): {region_mib:.3}");
        println!("  CubeCL sliced-page size (MiB): {page_mib:.3}");
        println!("  CubeCL backing pages: {}", self.workspace.backing_pages);
        println!("  CubeCL backing demand (MiB): {backing_mib:.3}");
        println!("  Outer factor rows: {}", panels.outer_rows);
        println!("  Inner factor rows: {}", panels.inner_rows);
        println!("  Reduction columns: {}", panels.reduction_columns);
        println!("  Intermediate columns: {}", panels.intermediate_columns);
        println!("  Outer panels: {outer_panels}");
        println!("  Inner panels: {inner_panels}");
        println!("  Source panels: {source_panels}");
        println!("  Reduction panels: {reduction_panels}");
        println!("  Sub-contraction groups: {contraction_groups}");
        println!("  Estimated alpha factor entries: {alpha_work}");
        println!("  Estimated beta factor entries: {beta_work}");
        println!("  Factor-table equivalents: {equivalents:.3}");
    }

    /// Build heterogeneous grouped-GEMM descriptors for one parent-pair target panel.
    /// Rank sums are represented as contribution ranges inside each owned output block,
    /// while clipped component maps retain the dense block's original determinant stride.
    /// # Arguments:
    /// - `block`: Parent-pair dimensions and selected contraction order.
    /// - `alpha_begin`: First target alpha component represented by the panel.
    /// - `alpha_end`: One-past-last target alpha component represented by the panel.
    /// - `beta_begin`: First target beta component represented by the panel.
    /// - `beta_end`: One-past-last target beta component represented by the panel.
    /// - `source_alpha_begin`: First source alpha component represented by the panel.
    /// - `source_alpha_end`: One-past-last source alpha component represented by the panel.
    /// - `source_beta_begin`: First source beta component represented by the panel.
    /// - `source_beta_end`: One-past-last source beta component represented by the panel.
    /// # Returns
    /// - `GroupedRankPlan`: Device descriptor buffers and flattened tile counts.
    fn build_grouped_rank_plan(
        &self,
        block: GpuFactorisedBlock,
        alpha_begin: usize,
        alpha_end: usize,
        beta_begin: usize,
        beta_end: usize,
        source_alpha_begin: usize,
        source_alpha_end: usize,
        source_beta_begin: usize,
        source_beta_end: usize,
    ) -> GroupedRankPlan {
        let full_target_alpha = self.rank_groups(block.target_parent, true);
        let full_target_beta = self.rank_groups(block.target_parent, false);
        let target_alpha =
            self.rank_groups_in_range(block.target_parent, true, alpha_begin, alpha_end);
        let target_beta =
            self.rank_groups_in_range(block.target_parent, false, beta_begin, beta_end);
        let full_source_alpha = self.rank_groups(block.source_parent, true);
        let full_source_beta = self.rank_groups(block.source_parent, false);
        let source_alpha = self.rank_groups_in_range(
            block.source_parent,
            true,
            source_alpha_begin,
            source_alpha_end,
        );
        let source_beta = self.rank_groups_in_range(
            block.source_parent,
            false,
            source_beta_begin,
            source_beta_end,
        );
        let source_blocks = &self.data.rank_blocks[block.source_parent];
        let target_blocks = &self.data.rank_blocks[block.target_parent];
        let mut stage_blocks = Vec::new();
        let mut stage_contributions = Vec::new();
        let mut stage_tiles = 0usize;

        match block.contraction {
            OneBodyContraction::AFirst => {
                for &(_, row_offset, m) in target_alpha.iter().filter(|group| group.2 > 0) {
                    for &(source_rank, col_offset, n) in
                        source_beta.iter().filter(|group| group.2 > 0)
                    {
                        let contribution_begin =
                            stage_contributions.len() / GROUPED_STAGE_CONTRIBUTION_FIELDS;
                        for source in source_blocks
                            .iter()
                            .filter(|source| source.beta_rank == source_rank)
                        {
                            let (_, source_offset, k) = source_alpha[source.alpha_rank];
                            if k == 0 {
                                continue;
                            }
                            let alpha_start =
                                source_offset - full_source_alpha[source.alpha_rank].1;
                            let beta_start = col_offset - full_source_beta[source.beta_rank].1;
                            extend_u32(
                                &mut stage_contributions,
                                &[
                                    k,
                                    source_offset,
                                    source.det_offset + alpha_start * source.nbeta + beta_start,
                                    source.nbeta,
                                ],
                            );
                        }
                        let contribution_end =
                            stage_contributions.len() / GROUPED_STAGE_CONTRIBUTION_FIELDS;
                        if contribution_begin == contribution_end {
                            continue;
                        }
                        let tile_begin = stage_tiles;
                        stage_tiles += grouped_tile_count(m, n);
                        extend_u32(
                            &mut stage_blocks,
                            &[
                                tile_begin,
                                stage_tiles,
                                m,
                                n,
                                row_offset,
                                col_offset,
                                contribution_begin,
                                contribution_end,
                            ],
                        );
                    }
                }
            }
            OneBodyContraction::BFirst => {
                for &(source_rank, row_offset, m) in source_alpha.iter().filter(|group| group.2 > 0)
                {
                    for &(_, col_offset, n) in target_beta.iter().filter(|group| group.2 > 0) {
                        let contribution_begin =
                            stage_contributions.len() / GROUPED_STAGE_CONTRIBUTION_FIELDS;
                        for source in source_blocks
                            .iter()
                            .filter(|source| source.alpha_rank == source_rank)
                        {
                            let (_, source_offset, k) = source_beta[source.beta_rank];
                            if k == 0 {
                                continue;
                            }
                            let alpha_start = row_offset - full_source_alpha[source.alpha_rank].1;
                            let beta_start = source_offset - full_source_beta[source.beta_rank].1;
                            extend_u32(
                                &mut stage_contributions,
                                &[
                                    k,
                                    source_offset,
                                    source.det_offset + alpha_start * source.nbeta + beta_start,
                                    source.nbeta,
                                ],
                            );
                        }
                        let contribution_end =
                            stage_contributions.len() / GROUPED_STAGE_CONTRIBUTION_FIELDS;
                        if contribution_begin == contribution_end {
                            continue;
                        }
                        let tile_begin = stage_tiles;
                        stage_tiles += grouped_tile_count(m, n);
                        extend_u32(
                            &mut stage_blocks,
                            &[
                                tile_begin,
                                stage_tiles,
                                m,
                                n,
                                row_offset,
                                col_offset,
                                contribution_begin,
                                contribution_end,
                            ],
                        );
                    }
                }
            }
        }

        let mut final_blocks = Vec::new();
        let mut final_contributions = Vec::new();
        let mut final_tiles = 0usize;
        for target in target_blocks {
            let (_, alpha_offset, m) = target_alpha[target.alpha_rank];
            let (_, beta_offset, n) = target_beta[target.beta_rank];
            if m == 0 || n == 0 {
                continue;
            }
            let alpha_start = alpha_offset - full_target_alpha[target.alpha_rank].1;
            let beta_start = beta_offset - full_target_beta[target.beta_rank].1;
            let contribution_begin = final_contributions.len() / GROUPED_FINAL_CONTRIBUTION_FIELDS;
            let source_groups = match block.contraction {
                OneBodyContraction::AFirst => &source_beta,
                OneBodyContraction::BFirst => &source_alpha,
            };
            for &(rank, source_offset, k) in source_groups.iter().filter(|group| group.2 > 0) {
                let populated = source_blocks.iter().any(|source| match block.contraction {
                    OneBodyContraction::AFirst => source.beta_rank == rank,
                    OneBodyContraction::BFirst => source.alpha_rank == rank,
                });
                if populated {
                    extend_u32(&mut final_contributions, &[k, source_offset]);
                }
            }
            let contribution_end = final_contributions.len() / GROUPED_FINAL_CONTRIBUTION_FIELDS;
            let tile_begin = final_tiles;
            final_tiles += grouped_tile_count(m, n);
            extend_u32(
                &mut final_blocks,
                &[
                    tile_begin,
                    final_tiles,
                    m,
                    n,
                    alpha_offset,
                    beta_offset,
                    target.det_offset + alpha_start * target.nbeta + beta_start,
                    contribution_begin,
                    contribution_end,
                    target.nbeta,
                ],
            );
        }

        let stage_blocks_len = stage_blocks.len();
        let stage_contributions_len = stage_contributions.len();
        let final_blocks_len = final_blocks.len();
        let final_contributions_len = final_contributions.len();
        let mut descriptors = Vec::with_capacity(
            stage_blocks_len + stage_contributions_len + final_blocks_len + final_contributions_len,
        );
        descriptors.extend_from_slice(&stage_blocks);
        descriptors.extend_from_slice(&stage_contributions);
        descriptors.extend_from_slice(&final_blocks);
        descriptors.extend_from_slice(&final_contributions);
        let storage = GpuBuffer::from_slice(&self.context, &descriptors);
        let stage_contributions_offset = stage_blocks_len;
        let final_blocks_offset = stage_contributions_offset + stage_contributions_len;
        let final_contributions_offset = final_blocks_offset + final_blocks_len;

        GroupedRankPlan {
            stage_blocks: storage.slice(0, stage_blocks_len),
            stage_contributions: storage.slice(stage_contributions_offset, stage_contributions_len),
            final_blocks: storage.slice(final_blocks_offset, final_blocks_len),
            final_contributions: storage.slice(final_contributions_offset, final_contributions_len),
            stage_tiles,
            final_tiles,
        }
    }

    /// Apply one dense alpha-first grouped contraction with planner-selected ranges `I,K,J,L`.
    /// The same loop evaluates `T_F[I,J] += F^alpha[I,K] D[K,J]` and
    /// `T_S[I,J] += S^alpha[I,K] D[K,J]`, then accumulates
    /// `Y[I,L] += T_F[I,J] (S^beta[L,J])^T
    /// + T_S[I,J] (F^beta[L,J] + lambda S^beta[L,J])^T`.
    /// Any range may equal its full logical dimension, so residency and streaming are plan values
    /// of this algorithm rather than separate contraction paths.
    /// # Arguments:
    /// - `block`: Dense parent-pair dimensions and ordered Wick metadata.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: MPI worker id and count using target alpha ownership.
    /// # Returns
    /// - `()`: Accumulates actual target determinants into device `y` with bounded scratch.
    fn apply_a_first_grouped_rank_blocks(
        &mut self,
        block: GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
    ) {
        let panels = grouped_plan(
            block.nta,
            block.nsa,
            block.nsb,
            block.ntb,
            block.nsb,
            self.workspace.region_elements,
            self.workspace.region_elements,
        );
        let outer_factor_len = checked_mul(
            panels.outer_rows,
            panels.reduction_columns,
            "GPU alpha panel factor length",
        );
        let inner_factor_len = checked_mul(
            panels.inner_rows,
            panels.intermediate_columns,
            "GPU beta panel factor length",
        );
        let first_len = checked_mul(
            panels.outer_rows,
            panels.intermediate_columns,
            "GPU alpha-first intermediate length",
        );
        self.scratch
            .prepare_contraction(outer_factor_len.max(inner_factor_len), first_len);

        for a0 in (0..block.nta).step_by(panels.outer_rows) {
            let a1 = (a0 + panels.outer_rows).min(block.nta);
            for sb0 in (0..block.nsb).step_by(panels.intermediate_columns) {
                let sb1 = (sb0 + panels.intermediate_columns).min(block.nsb);
                let accumulate_stage = panels.reduction_columns < block.nsa;
                if accumulate_stage {
                    let (tf, ts) = self.scratch.first_buffers();
                    let panel_first_len = (a1 - a0) * (sb1 - sb0);
                    launch_zero_f64(&self.context, tf, panel_first_len);
                    launch_zero_f64(&self.context, ts, panel_first_len);
                }
                for sa0 in (0..block.nsa).step_by(panels.reduction_columns) {
                    let sa1 = (sa0 + panels.reduction_columns).min(block.nsa);
                    self.build_factors(&block, true, a0, a1, sa0, sa1);
                    let stage = self
                        .build_grouped_rank_plan(block, a0, a1, 0, block.ntb, sa0, sa1, sb0, sb1);
                    let packed = self
                        .scratch
                        .packed
                        .as_ref()
                        .expect("GPU packed coefficient buffer must exist");
                    let (sa, fa) = self.scratch.factor_buffers();
                    let (tf, ts) = self.scratch.first_buffers();
                    launch_grouped_rank_stage(
                        &self.context,
                        sa,
                        fa,
                        packed,
                        &self.device_data.alpha_rank_component,
                        &self.device_data.beta_rank_component,
                        &self.device_data.alpha_rank_component,
                        &stage.stage_blocks,
                        &stage.stage_contributions,
                        tf,
                        ts,
                        GroupedRankLaunch {
                            tiles: stage.stage_tiles,
                            blocks: stage.stage_blocks.len() / GROUPED_STAGE_BLOCK_FIELDS,
                            factor_stride: sa1 - sa0,
                            intermediate_stride: sb1 - sb0,
                            factor_target_base: a0,
                            factor_source_base: sa0,
                            intermediate_target_base: a0,
                            intermediate_source_base: sb0,
                            lambda,
                            worker: partition.0,
                            nworker: partition.1,
                        },
                        true,
                        accumulate_stage,
                    );
                }

                for b0 in (0..block.ntb).step_by(panels.inner_rows) {
                    let b1 = (b0 + panels.inner_rows).min(block.ntb);
                    self.build_factors(&block, false, b0, b1, sb0, sb1);
                    let final_plan =
                        self.build_grouped_rank_plan(block, a0, a1, b0, b1, 0, block.nsa, sb0, sb1);
                    let (sb, fb) = self.scratch.factor_buffers();
                    let (tf, ts) = self.scratch.first_buffers();
                    launch_grouped_rank_final(
                        &self.context,
                        sb,
                        fb,
                        tf,
                        ts,
                        &self.device_data.alpha_rank_component,
                        &self.device_data.beta_rank_component,
                        &self.device_data.beta_rank_component,
                        &self.device_data.dense_rank_dets,
                        &final_plan.final_blocks,
                        &final_plan.final_contributions,
                        self.scratch.y.as_ref().expect("GPU y buffer must exist"),
                        GroupedRankLaunch {
                            tiles: final_plan.final_tiles,
                            blocks: final_plan.final_blocks.len() / GROUPED_FINAL_BLOCK_FIELDS,
                            factor_stride: sb1 - sb0,
                            intermediate_stride: sb1 - sb0,
                            factor_target_base: b0,
                            factor_source_base: sb0,
                            intermediate_target_base: a0,
                            intermediate_source_base: sb0,
                            lambda,
                            worker: partition.0,
                            nworker: partition.1,
                        },
                        true,
                    );
                }
            }
        }
    }

    /// Apply one dense beta-first grouped contraction with planner-selected ranges `I,K,J,L`.
    /// The same loop evaluates `U_F[J,L] += D[J,K] (F^beta[L,K])^T` and
    /// `U_S[J,L] += D[J,K] (S^beta[L,K])^T`, then accumulates
    /// `Y[I,L] += S^alpha[I,J] U_F[J,L]
    /// + (F^alpha[I,J] + lambda S^alpha[I,J]) U_S[J,L]`.
    /// Any range may equal its full logical dimension, preserving spin symmetry with alpha-first.
    /// # Arguments:
    /// - `block`: Dense parent-pair dimensions and ordered Wick metadata.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: MPI worker id and count using target beta ownership.
    /// # Returns
    /// - `()`: Accumulates actual target determinants into device `y` with bounded scratch.
    fn apply_b_first_grouped_rank_blocks(
        &mut self,
        block: GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
    ) {
        let panels = grouped_plan(
            block.ntb,
            block.nsb,
            block.nsa,
            block.nta,
            block.nsa,
            self.workspace.region_elements,
            self.workspace.region_elements,
        );
        let outer_factor_len = checked_mul(
            panels.outer_rows,
            panels.reduction_columns,
            "GPU beta panel factor length",
        );
        let inner_factor_len = checked_mul(
            panels.inner_rows,
            panels.intermediate_columns,
            "GPU alpha panel factor length",
        );
        let first_len = checked_mul(
            panels.outer_rows,
            panels.intermediate_columns,
            "GPU beta-first intermediate length",
        );
        self.scratch
            .prepare_contraction(outer_factor_len.max(inner_factor_len), first_len);

        for b0 in (0..block.ntb).step_by(panels.outer_rows) {
            let b1 = (b0 + panels.outer_rows).min(block.ntb);
            for sa0 in (0..block.nsa).step_by(panels.intermediate_columns) {
                let sa1 = (sa0 + panels.intermediate_columns).min(block.nsa);
                let accumulate_stage = panels.reduction_columns < block.nsb;
                if accumulate_stage {
                    let (uf, us) = self.scratch.first_buffers();
                    let panel_first_len = (sa1 - sa0) * (b1 - b0);
                    launch_zero_f64(&self.context, uf, panel_first_len);
                    launch_zero_f64(&self.context, us, panel_first_len);
                }
                for sb0 in (0..block.nsb).step_by(panels.reduction_columns) {
                    let sb1 = (sb0 + panels.reduction_columns).min(block.nsb);
                    self.build_factors(&block, false, b0, b1, sb0, sb1);
                    let stage = self
                        .build_grouped_rank_plan(block, 0, block.nta, b0, b1, sa0, sa1, sb0, sb1);
                    let packed = self
                        .scratch
                        .packed
                        .as_ref()
                        .expect("GPU packed coefficient buffer must exist");
                    let (sb, fb) = self.scratch.factor_buffers();
                    let (uf, us) = self.scratch.first_buffers();
                    launch_grouped_rank_stage(
                        &self.context,
                        sb,
                        fb,
                        packed,
                        &self.device_data.alpha_rank_component,
                        &self.device_data.beta_rank_component,
                        &self.device_data.beta_rank_component,
                        &stage.stage_blocks,
                        &stage.stage_contributions,
                        uf,
                        us,
                        GroupedRankLaunch {
                            tiles: stage.stage_tiles,
                            blocks: stage.stage_blocks.len() / GROUPED_STAGE_BLOCK_FIELDS,
                            factor_stride: sb1 - sb0,
                            intermediate_stride: b1 - b0,
                            factor_target_base: b0,
                            factor_source_base: sb0,
                            intermediate_target_base: b0,
                            intermediate_source_base: sa0,
                            lambda,
                            worker: partition.0,
                            nworker: partition.1,
                        },
                        false,
                        accumulate_stage,
                    );
                }

                for a0 in (0..block.nta).step_by(panels.inner_rows) {
                    let a1 = (a0 + panels.inner_rows).min(block.nta);
                    self.build_factors(&block, true, a0, a1, sa0, sa1);
                    let final_plan =
                        self.build_grouped_rank_plan(block, a0, a1, b0, b1, sa0, sa1, 0, block.nsb);
                    let (sa, fa) = self.scratch.factor_buffers();
                    let (uf, us) = self.scratch.first_buffers();
                    launch_grouped_rank_final(
                        &self.context,
                        sa,
                        fa,
                        uf,
                        us,
                        &self.device_data.alpha_rank_component,
                        &self.device_data.beta_rank_component,
                        &self.device_data.alpha_rank_component,
                        &self.device_data.dense_rank_dets,
                        &final_plan.final_blocks,
                        &final_plan.final_contributions,
                        self.scratch.y.as_ref().expect("GPU y buffer must exist"),
                        GroupedRankLaunch {
                            tiles: final_plan.final_tiles,
                            blocks: final_plan.final_blocks.len() / GROUPED_FINAL_BLOCK_FIELDS,
                            factor_stride: sa1 - sa0,
                            intermediate_stride: b1 - b0,
                            factor_target_base: a0,
                            factor_source_base: sa0,
                            intermediate_target_base: b0,
                            intermediate_source_base: sa0,
                            lambda,
                            worker: partition.0,
                            nworker: partition.1,
                        },
                        false,
                    );
                }
            }
        }
    }

    /// Return rank-group component-map ranges for one parent and spin.
    /// Empty ranks are retained so excitation rank remains a direct vector index.
    /// # Arguments:
    /// - `parent`: Parent reference.
    /// - `alpha`: Whether to select alpha rather than beta groups.
    /// # Returns
    /// - `Vec<(usize,usize,usize)>`: Rank, map offset and group length.
    fn rank_groups(
        &self,
        parent: usize,
        alpha: bool,
    ) -> Vec<(usize, usize, usize)> {
        let max_rank = if alpha {
            self.data.max_alpha_rank
        } else {
            self.data.max_beta_rank
        };
        let offsets = if alpha {
            &self.data.parent_alpha_rank_offsets
        } else {
            &self.data.parent_beta_rank_offsets
        };
        let stride = max_rank + 2;
        (0..=max_rank)
            .map(|rank| {
                let slot = parent * stride + rank;
                (rank, offsets[slot], offsets[slot + 1] - offsets[slot])
            })
            .collect()
    }

    /// Return rank-group map ranges clipped to one parent-local component interval.
    /// Empty ranks are retained so excitation rank remains a direct vector index.
    /// # Arguments:
    /// - `parent`: Parent reference.
    /// - `alpha`: Whether to select alpha rather than beta groups.
    /// - `component_begin`: First requested parent-local component.
    /// - `component_end`: One-past-last requested parent-local component.
    /// # Returns
    /// - `Vec<(usize,usize,usize)>`: Rank, clipped map offset and clipped length.
    fn rank_groups_in_range(
        &self,
        parent: usize,
        alpha: bool,
        component_begin: usize,
        component_end: usize,
    ) -> Vec<(usize, usize, usize)> {
        let groups = self.rank_groups(parent, alpha);
        let components = if alpha {
            &self.data.alpha_rank_component
        } else {
            &self.data.beta_rank_component
        };
        groups
            .into_iter()
            .map(|(rank, offset, len)| {
                let values = &components[offset..offset + len];
                let begin = values.partition_point(|&value| value < component_begin);
                let end = values.partition_point(|&value| value < component_end);
                (rank, offset + begin, end - begin)
            })
            .collect()
    }

    /// Apply `Y^Q += F^alpha D (S^beta)^T + S^alpha D (F^beta + lambda S^beta)^T`.
    /// Panel widths are chosen from bounded factor and first-intermediate memory budgets.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: Worker index and worker count.
    /// # Returns
    /// - `()`: Accumulates the alpha-first action into device `y`.
    fn apply_a_first_block(
        &mut self,
        block: GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
    ) {
        let panels = grouped_plan(
            block.nta,
            block.nsa,
            block.nsb,
            block.ntb,
            block.nsb,
            self.workspace.region_elements,
            self.workspace.region_elements,
        );

        let outer_factor_len = checked_mul(
            panels.outer_rows,
            block.nsa,
            "GPU alpha panel factor length",
        );
        let inner_factor_len =
            checked_mul(panels.inner_rows, block.nsb, "GPU beta panel factor length");
        let first_len = checked_mul(
            panels.outer_rows,
            block.nsb,
            "GPU alpha-first intermediate length",
        );
        self.scratch
            .prepare_contraction(outer_factor_len.max(inner_factor_len), first_len);

        for a0 in (0..block.nta).step_by(panels.outer_rows) {
            let a1 = (a0 + panels.outer_rows).min(block.nta);

            self.build_factors(&block, true, a0, a1, 0, block.nsa);
            self.launch_a_first_stage_panel(&block, partition, a0, a1);

            for b0 in (0..block.ntb).step_by(panels.inner_rows) {
                let b1 = (b0 + panels.inner_rows).min(block.ntb);

                self.build_factors(&block, false, b0, b1, 0, block.nsb);
                self.launch_a_first_final_panel(&block, lambda, partition, a0, a1, b0, b1);
            }
        }
    }

    /// Apply `Y^Q += S^alpha D (F^beta)^T + (F^alpha + lambda S^alpha) D (S^beta)^T`.
    /// Panel widths are chosen from bounded factor and first-intermediate memory budgets.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: Worker index and worker count.
    /// # Returns
    /// - `()`: Accumulates the beta-first action into device `y`.
    fn apply_b_first_block(
        &mut self,
        block: GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
    ) {
        let panels = grouped_plan(
            block.ntb,
            block.nsb,
            block.nsa,
            block.nta,
            block.nsa,
            self.workspace.region_elements,
            self.workspace.region_elements,
        );

        let outer_factor_len =
            checked_mul(panels.outer_rows, block.nsb, "GPU beta panel factor length");
        let inner_factor_len = checked_mul(
            panels.inner_rows,
            block.nsa,
            "GPU alpha panel factor length",
        );
        let first_len = checked_mul(
            panels.outer_rows,
            block.nsa,
            "GPU beta-first intermediate length",
        );
        self.scratch
            .prepare_contraction(outer_factor_len.max(inner_factor_len), first_len);

        for b0 in (0..block.ntb).step_by(panels.outer_rows) {
            let b1 = (b0 + panels.outer_rows).min(block.ntb);

            self.build_factors(&block, false, b0, b1, 0, block.nsb);
            self.launch_b_first_stage_panel(&block, partition, b0, b1);

            for a0 in (0..block.nta).step_by(panels.inner_rows) {
                let a1 = (a0 + panels.inner_rows).min(block.nta);

                self.build_factors(&block, true, a0, a1, 0, block.nsa);
                self.launch_b_first_final_panel(&block, lambda, partition, a0, a1, b0, b1);
            }
        }
    }

    /// Build one transient same-spin factor panel for a parent pair.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `alpha`: Whether to construct alpha-spin factors.
    /// - `row0`: First target component represented by panel row zero.
    /// - `row1`: One-past-last target component represented by the panel.
    /// - `source0`: First source component represented by panel column zero.
    /// - `source1`: One-past-last source component represented by the panel.
    /// # Returns
    /// - `()`: Writes the requested overlap and Fock factor panel.
    fn build_factors(
        &self,
        block: &GpuFactorisedBlock,
        alpha: bool,
        row0: usize,
        row1: usize,
        source0: usize,
        source1: usize,
    ) {
        let (s, f) = self.scratch.factor_buffers();
        let wslot = self.wick_slot(block.lp, block.gp, alpha);

        build_spin_one_body_factors(
            &self.context,
            &self.device_wicks,
            &self.device_data,
            &self.data,
            FactorRequest {
                target_parent: block.target_parent,
                source_parent: block.source_parent,
                wslot,
                target_left: block.target_left,
                alpha,
                m: self.wick_m[wslot],
                target_component_base: row0,
                target_component_end: row1,
                source_component_base: source0,
                source_component_end: source1,
            },
            FactorOutput {
                s,
                f,
                target_component_base: row0,
                source_component_base: source0,
                source_stride: source1 - source0,
            },
        );
    }

    /// Form `T^F_{abar,b} = sum_a F^alpha_{abar,a} D_{a,b}` and
    /// `T^S_{abar,b} = sum_a S^alpha_{abar,a} D_{a,b}` for one alpha panel.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `partition`: Worker index and worker count.
    /// - `a0`: First target alpha component.
    /// - `a1`: One-past-last target alpha component.
    /// # Returns
    /// - `()`: Writes the first-stage alpha intermediates.
    fn launch_a_first_stage_panel(
        &self,
        block: &GpuFactorisedBlock,
        partition: (usize, usize),
        a0: usize,
        a1: usize,
    ) {
        let (worker, nworker) = partition;
        let (sa, fa) = self.scratch.factor_buffers();
        let (tf, ts) = self.scratch.first_buffers();

        launch_a_first_stage(
            &self.context,
            &self.device_data,
            sa,
            fa,
            self.scratch.x.as_ref().expect("GPU x buffer must exist"),
            tf,
            ts,
            AFirstStageLaunch {
                nrow: a1 - a0,
                nsb: block.nsb,
                nsa: block.nsa,
                csr_base: self.data.by_beta_parent_offsets[block.source_parent],
                target_component_base: a0,
                worker,
                nworker,
            },
        );
    }

    /// Contract one alpha-first intermediate panel with one beta factor panel.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: Worker index and worker count.
    /// - `a0`: First target alpha component.
    /// - `a1`: One-past-last target alpha component.
    /// - `b0`: First target beta component.
    /// - `b1`: One-past-last target beta component.
    /// # Returns
    /// - `()`: Accumulates one two-dimensional target panel into device `y`.
    fn launch_a_first_final_panel(
        &self,
        block: &GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
        a0: usize,
        a1: usize,
        b0: usize,
        b1: usize,
    ) {
        let (worker, nworker) = partition;
        let (sb, fb) = self.scratch.factor_buffers();
        let (tf, ts) = self.scratch.first_buffers();
        let entry_base = self.data.parent_entry_offsets[block.target_parent];
        let entry_end = self.data.parent_entry_offsets[block.target_parent + 1];

        launch_a_first_final(
            &self.context,
            &self.device_data,
            sb,
            fb,
            tf,
            ts,
            self.scratch.y.as_ref().expect("GPU y buffer must exist"),
            AFirstFinalLaunch {
                entry_base,
                nentry: entry_end - entry_base,
                nsb: block.nsb,
                target_alpha_component_base: a0,
                target_alpha_component_end: a1,
                target_beta_component_base: b0,
                target_beta_component_end: b1,
                lambda,
                worker,
                nworker,
            },
        );
    }

    /// Form `U^F_{a,bbar} = sum_b D_{a,b} F^beta_{bbar,b}` and
    /// `U^S_{a,bbar} = sum_b D_{a,b} S^beta_{bbar,b}` for one beta panel.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `partition`: Worker index and worker count.
    /// - `b0`: First target beta component.
    /// - `b1`: One-past-last target beta component.
    /// # Returns
    /// - `()`: Writes the first-stage beta intermediates.
    fn launch_b_first_stage_panel(
        &self,
        block: &GpuFactorisedBlock,
        partition: (usize, usize),
        b0: usize,
        b1: usize,
    ) {
        let (worker, nworker) = partition;
        let (sb, fb) = self.scratch.factor_buffers();
        let (uf, us) = self.scratch.first_buffers();

        launch_b_first_stage(
            &self.context,
            &self.device_data,
            sb,
            fb,
            self.scratch.x.as_ref().expect("GPU x buffer must exist"),
            uf,
            us,
            BFirstStageLaunch {
                nrow: b1 - b0,
                nsa: block.nsa,
                nsb: block.nsb,
                csr_base: self.data.by_alpha_parent_offsets[block.source_parent],
                target_component_base: b0,
                worker,
                nworker,
            },
        );
    }

    /// Contract one beta-first intermediate panel with one alpha factor panel.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: Worker index and worker count.
    /// - `a0`: First target alpha component.
    /// - `a1`: One-past-last target alpha component.
    /// - `b0`: First target beta component.
    /// - `b1`: One-past-last target beta component.
    /// # Returns
    /// - `()`: Accumulates one two-dimensional target panel into device `y`.
    fn launch_b_first_final_panel(
        &self,
        block: &GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
        a0: usize,
        a1: usize,
        b0: usize,
        b1: usize,
    ) {
        let (worker, nworker) = partition;
        let (sa, fa) = self.scratch.factor_buffers();
        let (uf, us) = self.scratch.first_buffers();
        let entry_base = self.data.parent_entry_offsets[block.target_parent];
        let entry_end = self.data.parent_entry_offsets[block.target_parent + 1];

        launch_b_first_final(
            &self.context,
            &self.device_data,
            sa,
            fa,
            uf,
            us,
            self.scratch.y.as_ref().expect("GPU y buffer must exist"),
            BFirstFinalLaunch {
                entry_base,
                nentry: entry_end - entry_base,
                nsa: block.nsa,
                nrow: b1 - b0,
                target_alpha_component_base: a0,
                target_alpha_component_end: a1,
                target_beta_component_base: b0,
                target_beta_component_end: b1,
                lambda,
                worker,
                nworker,
            },
        );
    }

    /// Build one nonorthogonal same-parent determinant diagonal from same-component spin factors.
    /// # Arguments:
    /// - `parent`: Parent reference index.
    /// - `lambda`: Scalar overlap shift.
    /// # Returns
    /// - `()`: Writes this parent's determinant diagonals to device buffers.
    fn fill_parent_diagonal(
        &mut self,
        parent: usize,
        lambda: f64,
    ) {
        let na = self.spin.parents[parent].areps.len();
        let nb = self.spin.parents[parent].breps.len();
        let (lp, gp, _) = ordered_parent_pair(&self.spin, parent, parent);
        let alpha_slot = self.wick_slot(lp, gp, true);
        let beta_slot = self.wick_slot(lp, gp, false);

        self.scratch.ensure_alpha_factors(&self.context, na);
        self.scratch.ensure_beta_factors(&self.context, nb);

        let (sa, fa) = self.scratch.alpha_factors();

        build_spin_one_body_diagonal_factors(
            &self.context,
            &self.device_wicks,
            &self.device_data,
            &self.data,
            DiagonalFactorRequest {
                parent,
                wslot: alpha_slot,
                alpha: true,
                m: self.wick_m[alpha_slot],
                ncomponent: na,
            },
            DiagonalFactorOutput { s: sa, f: fa },
        );

        let (sb, fb) = self.scratch.beta_factors();

        build_spin_one_body_diagonal_factors(
            &self.context,
            &self.device_wicks,
            &self.device_data,
            &self.data,
            DiagonalFactorRequest {
                parent,
                wslot: beta_slot,
                alpha: false,
                m: self.wick_m[beta_slot],
                ncomponent: nb,
            },
            DiagonalFactorOutput { s: sb, f: fb },
        );

        let entry_base = self.data.parent_entry_offsets[parent];
        let entry_end = self.data.parent_entry_offsets[parent + 1];

        launch_fill_one_body_diagonal_block(
            &self.context,
            &self.device_data,
            sa,
            fa,
            sb,
            fb,
            self.scratch
                .m_diag
                .as_ref()
                .expect("GPU diagonal buffer must exist"),
            self.scratch
                .s_diag
                .as_ref()
                .expect("GPU diagonal buffer must exist"),
            DiagonalBlockLaunch {
                entry_base,
                nentry: entry_end - entry_base,
                lambda,
            },
        );
    }

    /// Convert an ordered reference pair and spin to the flattened device Wick slot.
    /// # Arguments:
    /// - `lp`: Left reference index.
    /// - `gp`: Greater reference index.
    /// - `alpha`: Whether to select the alpha-spin Wick sector.
    /// # Returns
    /// - `usize`: Flattened same-spin Wick slot.
    fn wick_slot(
        &self,
        lp: usize,
        gp: usize,
        alpha: bool,
    ) -> usize {
        (lp * self.plan.nparent + gp) * 2 + if alpha { 0 } else { 1 }
    }
}

/// Largest efficient CubeCL subslice class used by the four persistent contraction regions.
#[derive(Clone, Copy)]
struct WorkspaceLayout {
    /// Logical bytes available in each factor or intermediate region.
    region_bytes: usize,
    /// Logical `f64` entries available in each region.
    region_elements: usize,
    /// Backing-page bytes for the selected `SubSlices` class.
    page_bytes: usize,
    /// Backing pages required when all four regions are live.
    backing_pages: usize,
    /// Total backing bytes required by the four regions.
    backing_bytes: usize,
}

impl WorkspaceLayout {
    /// Reproduce CubeCL 0.10's largest efficient `SubSlices` class from device properties.
    /// For ordinary GPU page limits CubeCL constructs this class with
    /// `page = align(max_page / 4)` and `max_slice = page / 2`, so two maximum slices share
    /// one backing page. Devices below CubeCL's 32 MiB subdivision threshold use the final
    /// full-page class instead.
    /// # Arguments:
    /// - `context`: CubeCL context exposing backend-neutral memory properties.
    /// # Returns
    /// - `WorkspaceLayout`: Logical region capacity and exact worst-case page demand.
    fn from_context(context: &GpuContext) -> Self {
        let memory = &context.client().properties().memory;
        let alignment =
            usize::try_from(memory.alignment).expect("CubeCL memory alignment exceeds usize");
        let max_page =
            usize::try_from(memory.max_page_size).expect("CubeCL maximum page size exceeds usize");
        let subdivision_threshold = 32usize * 1024usize * 1024usize;
        let (page_bytes, max_slice_bytes) = if max_page >= subdivision_threshold {
            let page_bytes = max_page
                .checked_div(4)
                .expect("CubeCL maximum page size cannot form a sliced-page class")
                .next_multiple_of(alignment);
            (page_bytes, page_bytes / 2)
        } else {
            let page_bytes = max_page / alignment * alignment;
            (page_bytes, page_bytes)
        };
        let element_bytes = std::mem::size_of::<f64>();
        let region_elements = max_slice_bytes / element_bytes;
        let region_bytes = region_elements
            .checked_mul(element_bytes)
            .expect("GPU workspace region byte count overflow");
        let aligned_region_bytes = region_bytes.next_multiple_of(alignment);
        let backing_pages = aligned_region_bytes
            .checked_mul(4)
            .expect("GPU workspace backing demand overflow")
            .div_ceil(page_bytes);
        let backing_bytes = backing_pages
            .checked_mul(page_bytes)
            .expect("GPU workspace backing byte count overflow");

        Self {
            region_bytes,
            region_elements,
            page_bytes,
            backing_pages,
            backing_bytes,
        }
    }
}

/// Generic nonorthogonal GPU parent-pair block.
#[derive(Clone, Copy)]
struct GpuFactorisedBlock {
    /// Target parent `Q`.
    target_parent: usize,
    /// Source parent `P`.
    source_parent: usize,
    /// Ordered left reference.
    lp: usize,
    /// Ordered greater reference.
    gp: usize,
    /// Whether the target determinant is the left ordered determinant.
    target_left: bool,
    /// Target alpha-component count.
    nta: usize,
    /// Target beta-component count.
    ntb: usize,
    /// Source alpha-component count.
    nsa: usize,
    /// Source beta-component count.
    nsb: usize,
    /// Dense contraction order.
    contraction: OneBodyContraction,
}

/// Target, reduction and intermediate streaming widths.
#[derive(Clone, Copy)]
struct GroupedPlan {
    /// First-stage target-spin panel width.
    outer_rows: usize,
    /// Final-stage target-spin factor-panel width.
    inner_rows: usize,
    /// First-stage source-spin reduction width.
    reduction_columns: usize,
    /// First-stage other-spin coefficient and intermediate width.
    intermediate_columns: usize,
}

/// Persistent device descriptors for one heterogeneous grouped rank contraction.
struct GroupedRankPlan {
    /// Grouped first-stage output block descriptors.
    stage_blocks: GpuBuffer<u32>,
    /// Grouped first-stage inner-rank contribution descriptors.
    stage_contributions: GpuBuffer<u32>,
    /// Grouped final target block descriptors.
    final_blocks: GpuBuffer<u32>,
    /// Grouped final source-rank contribution descriptors.
    final_contributions: GpuBuffer<u32>,
    /// Flattened first-stage tile count.
    stage_tiles: usize,
    /// Flattened final tile count.
    final_tiles: usize,
}

/// Append checked device-width values to an interleaved descriptor table.
/// # Arguments:
/// - `out`: Descriptor table receiving the values.
/// - `values`: Host-width descriptor values.
/// # Returns
/// - `()`: Appends every value after checked conversion to `u32`.
fn extend_u32(
    out: &mut Vec<u32>,
    values: &[usize],
) {
    out.extend(
        values
            .iter()
            .map(|&value| u32::try_from(value).expect("GPU grouped descriptor exceeds u32")),
    );
}

/// Count square output tiles for one heterogeneous grouped matrix block.
/// # Arguments:
/// - `m`: Output row count.
/// - `n`: Output column count.
/// # Returns
/// - `usize`: Number of `GEMM_TILE_DIM` square tiles.
fn grouped_tile_count(
    m: usize,
    n: usize,
) -> usize {
    m.div_ceil(GEMM_TILE_DIM)
        .saturating_mul(n.div_ceil(GEMM_TILE_DIM))
}

/// Choose factor and first-intermediate panels from their independent persistent capacities.
/// The selected widths balance repeated Wick factor evaluations and contraction fragmentation under
/// `p_o p_k <= C_F`, `p_i p_j <= C_F` and `p_o p_j <= C_T`, where `C_F` and `C_T`
/// are the element capacities of one factor and one intermediate region. Candidate widths cover
/// every distinct panel count, including the full logical dimensions.
/// # Arguments:
/// - `outer_target`: Full first-contracted target-spin component count.
/// - `outer_factor_columns`: Source-spin columns in each outer factor row.
/// - `intermediate_columns`: Other-spin columns in each first-stage intermediate row.
/// - `inner_target`: Full final-contracted target-spin component count.
/// - `inner_factor_columns`: Source-spin columns in each inner factor row.
/// - `factor_capacity`: Entries available in each overlap/Fock factor region.
/// - `first_capacity`: Entries available in each first-stage intermediate region.
/// # Returns
/// - `GroupedPlan`: Target, reduction and intermediate panel widths.
fn grouped_plan(
    outer_target: usize,
    outer_factor_columns: usize,
    intermediate_columns: usize,
    inner_target: usize,
    inner_factor_columns: usize,
    factor_capacity: usize,
    first_capacity: usize,
) -> GroupedPlan {
    let outer_target = outer_target.max(1);
    let outer_factor_columns = outer_factor_columns.max(1);
    let intermediate_columns = intermediate_columns.max(1);
    let inner_target = inner_target.max(1);
    let inner_factor_columns = inner_factor_columns.max(1);
    let alpha_entries = outer_target.saturating_mul(outer_factor_columns);
    let beta_entries = inner_target.saturating_mul(inner_factor_columns);
    let mut best = GroupedPlan {
        outer_rows: 1,
        inner_rows: 1,
        reduction_columns: 1,
        intermediate_columns: 1,
    };
    let mut best_score = u128::MAX;
    let mut best_work = usize::MAX;
    let mut best_groups = usize::MAX;

    for shared_columns in panel_width_candidates(intermediate_columns) {
        for reduction_columns in panel_width_candidates(outer_factor_columns) {
            let outer_rows = (factor_capacity / reduction_columns)
                .min(first_capacity / shared_columns)
                .max(1)
                .min(outer_target);
            let first_entries = outer_rows.saturating_mul(shared_columns);
            let inner_rows = (factor_capacity / shared_columns).max(1).min(inner_target);
            let factor_entries = outer_rows
                .saturating_mul(reduction_columns)
                .max(inner_rows.saturating_mul(shared_columns));
            if first_entries > first_capacity || factor_entries > factor_capacity {
                continue;
            }

            let outer_panels = outer_target.div_ceil(outer_rows);
            let inner_panels = inner_target.div_ceil(inner_rows);
            let reduction_panels = outer_factor_columns.div_ceil(reduction_columns);
            let source_panels = intermediate_columns.div_ceil(shared_columns);
            let work = source_panels
                .saturating_mul(alpha_entries)
                .saturating_add(outer_panels.saturating_mul(beta_entries));
            let groups = outer_panels
                .saturating_mul(source_panels)
                .saturating_mul(reduction_panels.saturating_add(inner_panels));
            let score = (work as u128).saturating_mul(groups as u128);
            if score < best_score
                || (score == best_score
                    && (work < best_work || (work == best_work && groups < best_groups)))
            {
                best = GroupedPlan {
                    outer_rows,
                    inner_rows,
                    reduction_columns,
                    intermediate_columns: shared_columns,
                };
                best_score = score;
                best_work = work;
                best_groups = groups;
            }
        }
    }

    best
}

/// Enumerate one representative width for every distinct panel count of a logical dimension.
/// # Arguments:
/// - `dimension`: Full logical matrix dimension.
/// # Returns
/// - `Vec<usize>`: Sorted candidate panel widths including one and the full dimension.
fn panel_width_candidates(dimension: usize) -> Vec<usize> {
    let dimension = dimension.max(1);
    let mut widths = (1..=dimension)
        .map(|panels| dimension.div_ceil(panels))
        .collect::<Vec<_>>();
    widths.sort_unstable();
    widths.dedup();
    widths
}

/// Select the transient GPU contraction order from factor-generation work.
/// For alpha-first, `W_A = ceil(nsb / p_b) nta nsa + ceil(nta / p_a) ntb nsb`,
/// where `p_b` is the intermediate source panel and `p_a` is the outer target panel;
/// beta-first is the spin-swapped expression.
/// # Arguments:
/// - `block`: Parent-pair dimensions and shared contraction order.
/// # Returns
/// - `OneBodyContraction`: Lower transient factor-generation cost, using the shared order on ties.
fn select_gpu_contraction(
    block: GpuFactorisedBlock,
    factor_capacity: usize,
    first_capacity: usize,
) -> OneBodyContraction {
    let a_panels = grouped_plan(
        block.nta,
        block.nsa,
        block.nsb,
        block.ntb,
        block.nsb,
        factor_capacity,
        first_capacity,
    );
    let b_panels = grouped_plan(
        block.ntb,
        block.nsb,
        block.nsa,
        block.nta,
        block.nsa,
        factor_capacity,
        first_capacity,
    );
    let alpha_entries = block.nta.saturating_mul(block.nsa);
    let beta_entries = block.ntb.saturating_mul(block.nsb);
    let na_panels = block.nta.div_ceil(a_panels.outer_rows);
    let nb_panels = block.ntb.div_ceil(b_panels.outer_rows);
    let a_source_panels = block.nsb.div_ceil(a_panels.intermediate_columns);
    let b_source_panels = block.nsa.div_ceil(b_panels.intermediate_columns);

    let a_work = a_source_panels
        .saturating_mul(alpha_entries)
        .saturating_add(na_panels.saturating_mul(beta_entries));
    let b_work = b_source_panels
        .saturating_mul(beta_entries)
        .saturating_add(nb_panels.saturating_mul(alpha_entries));

    if a_work < b_work {
        OneBodyContraction::AFirst
    } else if b_work < a_work {
        OneBodyContraction::BFirst
    } else {
        block.contraction
    }
}

/// Reusable GPU one-body scratch buffers.
#[derive(Default)]
struct GpuOneBodyScratch {
    /// Persistent overlap-factor region reused by both spin stages.
    factor_s: Option<GpuBuffer<f64>>,
    /// Persistent Fock-factor region reused by both spin stages.
    factor_f: Option<GpuBuffer<f64>>,
    /// Alpha target-panel overlap factors.
    alpha_s: Option<GpuBuffer<f64>>,
    /// Alpha target-panel Fock factors.
    alpha_f: Option<GpuBuffer<f64>>,
    /// Beta target-panel overlap factors.
    beta_s: Option<GpuBuffer<f64>>,
    /// Beta target-panel Fock factors.
    beta_f: Option<GpuBuffer<f64>>,
    /// First-stage Fock intermediate.
    first_f: Option<GpuBuffer<f64>>,
    /// First-stage overlap intermediate.
    first_s: Option<GpuBuffer<f64>>,
    /// Packed source coefficient rank block.
    packed: Option<GpuBuffer<f64>>,
    /// Uploaded source vector.
    x: Option<GpuBuffer<f64>>,
    /// Device output vector.
    y: Option<GpuBuffer<f64>>,
    /// Diagonal of `F + lambda S`.
    m_diag: Option<GpuBuffer<f64>>,
    /// Diagonal of `S`.
    s_diag: Option<GpuBuffer<f64>>,
}

impl GpuOneBodyScratch {
    /// Reserve four allocator-compatible contraction regions before persistent Wick uploads.
    /// # Arguments:
    /// - `context`: CubeCL context used for allocation.
    /// - `region_elements`: Logical `f64` capacity of each persistent region.
    /// # Returns
    /// - `GpuOneBodyScratch`: Scratch storage owning the reserved allocation.
    fn with_contraction_regions(
        context: &GpuContext,
        region_elements: usize,
    ) -> Self {
        Self {
            factor_s: Some(GpuBuffer::empty(context, region_elements)),
            factor_f: Some(GpuBuffer::empty(context, region_elements)),
            first_f: Some(GpuBuffer::empty(context, region_elements)),
            first_s: Some(GpuBuffer::empty(context, region_elements)),
            ..Self::default()
        }
    }

    /// Verify one planned factor panel and first-stage intermediate fit their persistent regions.
    /// # Arguments:
    /// - `factor_len`: Maximum entries in either paired factor panel.
    /// - `first_len`: Entries in either paired first-stage intermediate.
    /// # Returns
    /// - `()`: Confirms the planner respected both physical region capacities.
    fn prepare_contraction(
        &mut self,
        factor_len: usize,
        first_len: usize,
    ) {
        let factor_capacity = self
            .factor_s
            .as_ref()
            .expect("GPU factor S buffer missing")
            .len();
        let first_capacity = self
            .first_f
            .as_ref()
            .expect("GPU first F buffer missing")
            .len();
        assert!(
            factor_len <= factor_capacity,
            "GPU factor panel exceeds persistent region"
        );
        assert!(
            first_len <= first_capacity,
            "GPU intermediate exceeds persistent region"
        );
    }

    /// Ensure packed coefficient storage can hold one dense rank block.
    /// # Arguments:
    /// - `context`: CubeCL context used for allocation.
    /// - `len`: Required packed block length.
    /// # Returns
    /// - `()`: Enlarges the buffer only when required.
    fn ensure_packed(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.packed, len);
    }
    /// Ensure the output-vector buffer can hold `len` entries.
    /// # Arguments:
    /// - `context`: CubeCL context used for allocation.
    /// - `len`: Required determinant-vector length.
    /// # Returns
    /// - `()`: Enlarges the buffer only when required.
    fn ensure_y(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.y, len);
    }

    /// Ensure the shifted-matrix diagonal buffer can hold `len` entries.
    /// # Arguments:
    /// - `context`: CubeCL context used for allocation.
    /// - `len`: Required diagonal length.
    /// # Returns
    /// - `()`: Enlarges the buffer only when required.
    fn ensure_m_diag(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.m_diag, len);
    }

    /// Ensure the overlap diagonal buffer can hold `len` entries.
    /// # Arguments:
    /// - `context`: CubeCL context used for allocation.
    /// - `len`: Required diagonal length.
    /// # Returns
    /// - `()`: Enlarges the buffer only when required.
    fn ensure_s_diag(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.s_diag, len);
    }

    /// Ensure alpha factor-panel buffers can hold `len` entries.
    /// # Arguments:
    /// - `context`: CubeCL context used for allocation.
    /// - `len`: Required alpha factor-panel length.
    /// # Returns
    /// - `()`: Enlarges the buffers only when required.
    fn ensure_alpha_factors(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.alpha_s, len);
        ensure_buffer(context, &mut self.alpha_f, len);
    }

    /// Ensure beta factor-panel buffers can hold `len` entries.
    /// # Arguments:
    /// - `context`: CubeCL context used for allocation.
    /// - `len`: Required beta factor-panel length.
    /// # Returns
    /// - `()`: Enlarges the buffers only when required.
    fn ensure_beta_factors(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.beta_s, len);
        ensure_buffer(context, &mut self.beta_f, len);
    }

    /// Borrow alpha overlap and Fock factor buffers.
    /// # Arguments:
    /// - `self`: Scratch storage with allocated alpha buffers.
    /// # Returns
    /// - `(&GpuBuffer<f64>, &GpuBuffer<f64>)`: Alpha overlap and Fock buffers.
    fn alpha_factors(&self) -> (&GpuBuffer<f64>, &GpuBuffer<f64>) {
        (
            self.alpha_s.as_ref().expect("GPU alpha S buffer missing"),
            self.alpha_f.as_ref().expect("GPU alpha F buffer missing"),
        )
    }

    /// Borrow beta overlap and Fock factor buffers.
    /// # Arguments:
    /// - `self`: Scratch storage with allocated beta buffers.
    /// # Returns
    /// - `(&GpuBuffer<f64>, &GpuBuffer<f64>)`: Beta overlap and Fock buffers.
    fn beta_factors(&self) -> (&GpuBuffer<f64>, &GpuBuffer<f64>) {
        (
            self.beta_s.as_ref().expect("GPU beta S buffer missing"),
            self.beta_f.as_ref().expect("GPU beta F buffer missing"),
        )
    }

    /// Borrow the lifetime-reused overlap and Fock factor views.
    /// # Arguments:
    /// - `self`: Scratch storage with a prepared contraction workspace.
    /// # Returns
    /// - `(&GpuBuffer<f64>, &GpuBuffer<f64>)`: Overlap and Fock factor views.
    fn factor_buffers(&self) -> (&GpuBuffer<f64>, &GpuBuffer<f64>) {
        (
            self.factor_s.as_ref().expect("GPU factor S view missing"),
            self.factor_f.as_ref().expect("GPU factor F view missing"),
        )
    }

    /// Borrow first-stage Fock and overlap intermediate buffers.
    /// # Arguments:
    /// - `self`: Scratch storage with allocated first-stage buffers.
    /// # Returns
    /// - `(&GpuBuffer<f64>, &GpuBuffer<f64>)`: Fock and overlap intermediates.
    fn first_buffers(&self) -> (&GpuBuffer<f64>, &GpuBuffer<f64>) {
        (
            self.first_f.as_ref().expect("GPU first F buffer missing"),
            self.first_s.as_ref().expect("GPU first S buffer missing"),
        )
    }
}

/// Ensure a device buffer has at least `len` elements.
/// # Arguments:
/// - `context`: CubeCL context used for allocation.
/// - `buffer`: Optional reusable buffer.
/// - `len`: Required logical element count.
/// # Returns
/// - `()`: Replaces the buffer only when its capacity is insufficient.
fn ensure_buffer(
    context: &GpuContext,
    buffer: &mut Option<GpuBuffer<f64>>,
    len: usize,
) {
    if buffer.as_ref().map_or(true, |buf| buf.len() < len) {
        *buffer = Some(GpuBuffer::empty(context, len));
    }
}

/// Multiply two launch dimensions with overflow checking.
/// # Arguments:
/// - `lhs`: Left factor.
/// - `rhs`: Right factor.
/// - `context`: Panic message used on overflow.
/// # Returns
/// - `usize`: Checked product.
fn checked_mul(
    lhs: usize,
    rhs: usize,
    context: &str,
) -> usize {
    lhs.checked_mul(rhs).expect(context)
}

/// Collect host-known zero-overlap counts in the same pair/spin ordering as device Wick metadata.
/// # Arguments:
/// - `wicks`: Host-packed GPU Wick storage.
/// - `nref`: Number of NOCI reference parents.
/// # Returns
/// - `Vec<usize>`: Flattened `m` values ordered as `(lp,gp,alpha/beta)`.
fn collect_wick_m<T: NOCIScalar>(
    wicks: &gpu::WicksShared<T>,
    nref: usize,
) -> Vec<usize> {
    let view = wicks.view();
    let mut out = Vec::with_capacity(
        nref.checked_mul(nref)
            .and_then(|x| x.checked_mul(2))
            .expect("GPU Wick metadata length overflow"),
    );

    for lp in 0..nref {
        for gp in 0..nref {
            let pair = view.pair(lp, gp);
            out.push(pair.aa.m);
            out.push(pair.bb.m);
        }
    }

    out
}

/// Reinterpret a scalar after construction has enforced `T = f64`.
/// # Arguments:
/// - `value`: Generic scalar known to have `f64` layout.
/// # Returns
/// - `f64`: Reinterpreted real scalar.
/// # Safety
/// The pointer cast is valid because every GPU backend instance is created only after
/// `TypeId::<T>() == TypeId::<f64>()`.
fn real_scalar<T: NOCIScalar + 'static>(value: T) -> f64 {
    if TypeId::of::<T>() != TypeId::of::<f64>() {
        eprintln!("snoci.backend = \"gpu\" currently supports real f64 NOCI-PT2 data only");
        std::process::exit(1);
    }

    let ptr = &value as *const T as *const f64;

    unsafe { *ptr }
}

/// Reinterpret a contiguous scalar slice after construction has enforced `T = f64`.
/// # Arguments:
/// - `values`: Contiguous generic scalar slice known to contain `f64`.
/// # Returns
/// - `&[f64]`: Real slice over the same storage.
/// # Safety
/// The pointer cast is valid because every GPU backend instance is created only after
/// `TypeId::<T>() == TypeId::<f64>()`.
fn real_slice<T: NOCIScalar + 'static>(values: &[T]) -> &[f64] {
    if TypeId::of::<T>() != TypeId::of::<f64>() {
        eprintln!("snoci.backend = \"gpu\" currently supports real f64 NOCI-PT2 data only");
        std::process::exit(1);
    }

    let ptr = values.as_ptr() as *const f64;

    unsafe { std::slice::from_raw_parts(ptr, values.len()) }
}

/// Upload host-packed Wick storage after construction has enforced `T = f64`.
/// # Arguments:
/// - `wicks`: Host-packed generic Wick storage known to contain `f64`.
/// - `context`: CubeCL context owning the target device.
/// # Returns
/// - `DeviceWicksShared`: Device-resident real Wick buffers.
/// # Safety
/// The pointer cast is valid because the caller constructs this backend only after
/// `TypeId::<T>() == TypeId::<f64>()`.
fn upload_real_wicks<T: NOCIScalar + 'static>(
    wicks: &gpu::WicksShared<T>,
    context: &GpuContext,
) -> DeviceWicksShared {
    let real = wicks as *const gpu::WicksShared<T> as *const gpu::WicksShared<f64>;
    let real = unsafe { &*real };

    real.upload_f64(context)
}
