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
use crate::input::{SNOCIGpuOptions, SNOCIStorage};
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
    /// Persistent heterogeneous grouped-GEMM descriptors keyed by parent-pair plan index.
    grouped_plans: Vec<Option<GroupedRankPlan>>,
    /// Maximum bytes for one paired same-spin factor panel.
    factor_panel_bytes: usize,
    /// Maximum bytes for one paired first-stage intermediate panel.
    first_panel_bytes: usize,
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
    /// - `gpu`: GPU transient-memory options.
    /// # Returns
    /// - `GpuOneBodyBackend<T>`: GPU one-body backend.
    pub(crate) fn new(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        _cache: &Path,
        _rank: i32,
        _iteration: usize,
        storage: SNOCIStorage,
        gpu: SNOCIGpuOptions,
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
        let orthogonal = GpuOrthogonalBlocks::new(&context, &spin, data, fock);
        let device_wicks = upload_real_wicks(&wicks, &context);
        let device_data = gpu_data.upload(&context);
        let factor_panel_bytes = mib_to_bytes(gpu.factor_panel_mib, "GPU factor panel budget");
        let first_panel_bytes = mib_to_bytes(gpu.first_panel_mib, "GPU first-stage panel budget");

        let mut backend = Self {
            context,
            spin,
            plan,
            wick_m,
            device_wicks,
            data: gpu_data,
            orthogonal,
            device_data,
            scratch: GpuOneBodyScratch::default(),
            grouped_plans: Vec::new(),
            factor_panel_bytes,
            first_panel_bytes,
            marker: PhantomData,
        };
        for index in 0..backend.plan.blocks.len() {
            let grouped = match backend.plan.blocks[index] {
                OneBodyBlockPlan::NonOrthogonal { .. } => {
                    let block = backend.gpu_nonorthogonal_block(index);
                    Some(backend.build_grouped_rank_plan(block))
                }
                OneBodyBlockPlan::Orthogonal { .. } => None,
            };
            backend.grouped_plans.push(grouped);
        }
        backend
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

                block.contraction =
                    select_gpu_contraction(block, self.factor_panel_bytes, self.first_panel_bytes);
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
        if self.rank_block_dense(block.target_parent)
            && self.rank_block_dense(block.source_parent)
            && self.dense_rank_scratch_fits(block)
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

    /// Test whether unstreamed rank-block factor and intermediate tables fit configured budgets.
    /// # Arguments:
    /// - `block`: Parent-pair dimensions and selected contraction order.
    /// # Returns
    /// - `bool`: Whether dense tiled routing respects both transient byte budgets.
    fn dense_rank_scratch_fits(
        &self,
        block: GpuFactorisedBlock,
    ) -> bool {
        let alpha_bytes = paired_f64_bytes(block.nta, block.nsa);
        let beta_bytes = paired_f64_bytes(block.ntb, block.nsb);
        let first_bytes = match block.contraction {
            OneBodyContraction::AFirst => paired_f64_bytes(block.nta, block.nsb),
            OneBodyContraction::BFirst => paired_f64_bytes(block.nsa, block.ntb),
        };
        alpha_bytes <= self.factor_panel_bytes
            && beta_bytes <= self.factor_panel_bytes
            && first_bytes <= self.first_panel_bytes
    }

    /// Build persistent heterogeneous grouped-GEMM descriptors for one parent pair.
    /// Rank sums are represented as contribution ranges inside each owned output block.
    /// # Arguments:
    /// - `block`: Parent-pair dimensions and selected contraction order.
    /// # Returns
    /// - `GroupedRankPlan`: Device descriptor buffers and flattened tile counts.
    fn build_grouped_rank_plan(
        &self,
        block: GpuFactorisedBlock,
    ) -> GroupedRankPlan {
        let target_alpha = self.rank_groups(block.target_parent, true);
        let target_beta = self.rank_groups(block.target_parent, false);
        let source_alpha = self.rank_groups(block.source_parent, true);
        let source_beta = self.rank_groups(block.source_parent, false);
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
                            extend_u32(
                                &mut stage_contributions,
                                &[k, source_offset, source.det_offset, source.nbeta],
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
                            extend_u32(
                                &mut stage_contributions,
                                &[k, source_offset, source.det_offset, source.nbeta],
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
                    target.det_offset,
                    contribution_begin,
                    contribution_end,
                ],
            );
        }

        GroupedRankPlan {
            stage_blocks: GpuBuffer::from_slice(&self.context, &stage_blocks),
            stage_contributions: GpuBuffer::from_slice(&self.context, &stage_contributions),
            final_blocks: GpuBuffer::from_slice(&self.context, &final_blocks),
            final_contributions: GpuBuffer::from_slice(&self.context, &final_contributions),
            stage_tiles,
            final_tiles,
        }
    }

    /// Apply one alpha-first parent pair with one grouped stage and one grouped final launch.
    /// # Arguments:
    /// - `block`: Dense parent-pair dimensions and ordered Wick metadata.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: MPI worker id and count using target alpha ownership.
    /// # Returns
    /// - `()`: Accumulates actual target determinants into device `y`.
    fn apply_a_first_grouped_rank_blocks(
        &mut self,
        block: GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
    ) {
        self.scratch
            .ensure_alpha_factors(&self.context, block.nta * block.nsa);
        self.scratch
            .ensure_beta_factors(&self.context, block.ntb * block.nsb);
        self.scratch
            .ensure_first(&self.context, block.nta * block.nsb);
        self.build_factors(&block, true, 0, block.nta, block.nsa);
        self.build_factors(&block, false, 0, block.ntb, block.nsb);
        let plan_index = block.target_parent * self.plan.nparent + block.source_parent;
        let grouped = self.grouped_plans[plan_index]
            .as_ref()
            .expect("dense parent pair must have grouped descriptors");
        let packed = self
            .scratch
            .packed
            .as_ref()
            .expect("GPU packed coefficient buffer must exist");
        let (sa, fa) = self.scratch.alpha_factors();
        let (tf, ts) = self.scratch.first_buffers();
        launch_grouped_rank_stage(
            &self.context,
            sa,
            fa,
            packed,
            &self.device_data.alpha_rank_component,
            &self.device_data.beta_rank_component,
            &self.device_data.alpha_rank_component,
            &grouped.stage_blocks,
            &grouped.stage_contributions,
            tf,
            ts,
            GroupedRankLaunch {
                tiles: grouped.stage_tiles,
                blocks: grouped.stage_blocks.len() / GROUPED_STAGE_BLOCK_FIELDS,
                factor_stride: block.nsa,
                intermediate_stride: block.nsb,
                lambda,
                worker: partition.0,
                nworker: partition.1,
            },
            true,
        );
        let (sb, fb) = self.scratch.beta_factors();
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
            &grouped.final_blocks,
            &grouped.final_contributions,
            self.scratch.y.as_ref().expect("GPU y buffer must exist"),
            GroupedRankLaunch {
                tiles: grouped.final_tiles,
                blocks: grouped.final_blocks.len() / GROUPED_FINAL_BLOCK_FIELDS,
                factor_stride: block.nsb,
                intermediate_stride: block.nsb,
                lambda,
                worker: partition.0,
                nworker: partition.1,
            },
            true,
        );
    }

    /// Apply one beta-first parent pair with one grouped stage and one grouped final launch.
    /// # Arguments:
    /// - `block`: Dense parent-pair dimensions and ordered Wick metadata.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: MPI worker id and count using target beta ownership.
    /// # Returns
    /// - `()`: Accumulates actual target determinants into device `y`.
    fn apply_b_first_grouped_rank_blocks(
        &mut self,
        block: GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
    ) {
        self.scratch
            .ensure_alpha_factors(&self.context, block.nta * block.nsa);
        self.scratch
            .ensure_beta_factors(&self.context, block.ntb * block.nsb);
        self.scratch
            .ensure_first(&self.context, block.nsa * block.ntb);
        self.build_factors(&block, true, 0, block.nta, block.nsa);
        self.build_factors(&block, false, 0, block.ntb, block.nsb);
        let plan_index = block.target_parent * self.plan.nparent + block.source_parent;
        let grouped = self.grouped_plans[plan_index]
            .as_ref()
            .expect("dense parent pair must have grouped descriptors");
        let packed = self
            .scratch
            .packed
            .as_ref()
            .expect("GPU packed coefficient buffer must exist");
        let (sb, fb) = self.scratch.beta_factors();
        let (uf, us) = self.scratch.first_buffers();
        launch_grouped_rank_stage(
            &self.context,
            sb,
            fb,
            packed,
            &self.device_data.alpha_rank_component,
            &self.device_data.beta_rank_component,
            &self.device_data.beta_rank_component,
            &grouped.stage_blocks,
            &grouped.stage_contributions,
            uf,
            us,
            GroupedRankLaunch {
                tiles: grouped.stage_tiles,
                blocks: grouped.stage_blocks.len() / GROUPED_STAGE_BLOCK_FIELDS,
                factor_stride: block.nsb,
                intermediate_stride: block.ntb,
                lambda,
                worker: partition.0,
                nworker: partition.1,
            },
            false,
        );
        let (sa, fa) = self.scratch.alpha_factors();
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
            &grouped.final_blocks,
            &grouped.final_contributions,
            self.scratch.y.as_ref().expect("GPU y buffer must exist"),
            GroupedRankLaunch {
                tiles: grouped.final_tiles,
                blocks: grouped.final_blocks.len() / GROUPED_FINAL_BLOCK_FIELDS,
                factor_stride: block.nsa,
                intermediate_stride: block.ntb,
                lambda,
                worker: partition.0,
                nworker: partition.1,
            },
            false,
        );
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
        let panels = a_first_panels(block, self.factor_panel_bytes, self.first_panel_bytes);

        self.scratch.ensure_alpha_factors(
            &self.context,
            checked_mul(
                panels.outer_rows,
                block.nsa,
                "GPU alpha panel factor length",
            ),
        );
        self.scratch.ensure_beta_factors(
            &self.context,
            checked_mul(panels.inner_rows, block.nsb, "GPU beta panel factor length"),
        );
        self.scratch.ensure_first(
            &self.context,
            checked_mul(
                panels.outer_rows,
                block.nsb,
                "GPU alpha-first intermediate length",
            ),
        );

        for a0 in (0..block.nta).step_by(panels.outer_rows) {
            let a1 = (a0 + panels.outer_rows).min(block.nta);

            self.build_factors(&block, true, a0, a1, block.nsa);
            self.launch_a_first_stage_panel(&block, partition, a0, a1);

            for b0 in (0..block.ntb).step_by(panels.inner_rows) {
                let b1 = (b0 + panels.inner_rows).min(block.ntb);

                self.build_factors(&block, false, b0, b1, block.nsb);
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
        let panels = b_first_panels(block, self.factor_panel_bytes, self.first_panel_bytes);

        self.scratch.ensure_beta_factors(
            &self.context,
            checked_mul(panels.outer_rows, block.nsb, "GPU beta panel factor length"),
        );
        self.scratch.ensure_alpha_factors(
            &self.context,
            checked_mul(
                panels.inner_rows,
                block.nsa,
                "GPU alpha panel factor length",
            ),
        );
        self.scratch.ensure_first(
            &self.context,
            checked_mul(
                panels.outer_rows,
                block.nsa,
                "GPU beta-first intermediate length",
            ),
        );

        for b0 in (0..block.ntb).step_by(panels.outer_rows) {
            let b1 = (b0 + panels.outer_rows).min(block.ntb);

            self.build_factors(&block, false, b0, b1, block.nsb);
            self.launch_b_first_stage_panel(&block, partition, b0, b1);

            for a0 in (0..block.nta).step_by(panels.inner_rows) {
                let a1 = (a0 + panels.inner_rows).min(block.nta);

                self.build_factors(&block, true, a0, a1, block.nsa);
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
    /// - `nsource`: Full source-component count.
    /// # Returns
    /// - `()`: Writes the requested overlap and Fock factor panel.
    fn build_factors(
        &self,
        block: &GpuFactorisedBlock,
        alpha: bool,
        row0: usize,
        row1: usize,
        nsource: usize,
    ) {
        let (s, f) = if alpha {
            self.scratch.alpha_factors()
        } else {
            self.scratch.beta_factors()
        };
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
                nsource,
            },
            FactorOutput {
                s,
                f,
                target_component_base: row0,
                nsource,
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
        let (sa, fa) = self.scratch.alpha_factors();
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
        let (sb, fb) = self.scratch.beta_factors();
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
        let (sb, fb) = self.scratch.beta_factors();
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
        let (sa, fa) = self.scratch.alpha_factors();
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

/// Outer and inner target-spin streaming widths.
#[derive(Clone, Copy)]
struct StreamingPanels {
    /// First-stage target-spin panel width.
    outer_rows: usize,
    /// Final-stage target-spin factor-panel width.
    inner_rows: usize,
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

/// Select alpha-first streaming widths from bounded factor and intermediate memory.
/// # Arguments:
/// - `block`: Parent-pair dimensions.
/// # Returns
/// - `StreamingPanels`: Alpha and beta target-panel widths.
fn a_first_panels(
    block: GpuFactorisedBlock,
    factor_panel_bytes: usize,
    first_panel_bytes: usize,
) -> StreamingPanels {
    let factor_rows = panel_rows(block.nta, block.nsa, factor_panel_bytes);
    let first_rows = panel_rows(block.nta, block.nsb, first_panel_bytes);

    StreamingPanels {
        outer_rows: factor_rows.min(first_rows),
        inner_rows: panel_rows(block.ntb, block.nsb, factor_panel_bytes),
    }
}

/// Select beta-first streaming widths from bounded factor and intermediate memory.
/// # Arguments:
/// - `block`: Parent-pair dimensions.
/// # Returns
/// - `StreamingPanels`: Beta and alpha target-panel widths.
fn b_first_panels(
    block: GpuFactorisedBlock,
    factor_panel_bytes: usize,
    first_panel_bytes: usize,
) -> StreamingPanels {
    let factor_rows = panel_rows(block.ntb, block.nsb, factor_panel_bytes);
    let first_rows = panel_rows(block.ntb, block.nsa, first_panel_bytes);

    StreamingPanels {
        outer_rows: factor_rows.min(first_rows),
        inner_rows: panel_rows(block.nta, block.nsa, factor_panel_bytes),
    }
}

/// Select the transient GPU contraction order from factor-generation work.
/// For alpha-first, the beta factor table is regenerated once per alpha outer panel,
/// giving `W_A = nta nsa + ceil(nta / p_a) ntb nsb`; beta-first is analogous.
/// # Arguments:
/// - `block`: Parent-pair dimensions and shared contraction order.
/// # Returns
/// - `OneBodyContraction`: Lower transient factor-generation cost, using the shared order on ties.
fn select_gpu_contraction(
    block: GpuFactorisedBlock,
    factor_panel_bytes: usize,
    first_panel_bytes: usize,
) -> OneBodyContraction {
    let a_panels = a_first_panels(block, factor_panel_bytes, first_panel_bytes);
    let b_panels = b_first_panels(block, factor_panel_bytes, first_panel_bytes);
    let alpha_entries = block.nta.saturating_mul(block.nsa);
    let beta_entries = block.ntb.saturating_mul(block.nsb);
    let na_panels = block.nta.div_ceil(a_panels.outer_rows);
    let nb_panels = block.ntb.div_ceil(b_panels.outer_rows);

    let a_work = alpha_entries.saturating_add(na_panels.saturating_mul(beta_entries));
    let b_work = beta_entries.saturating_add(nb_panels.saturating_mul(alpha_entries));

    if a_work < b_work {
        OneBodyContraction::AFirst
    } else if b_work < a_work {
        OneBodyContraction::BFirst
    } else {
        block.contraction
    }
}

/// Return the largest target-component panel fitting one paired `f64` scratch allocation.
/// A minimum width of one is retained when one logical row itself exceeds the nominal budget.
/// # Arguments:
/// - `ntarget`: Full target-component count.
/// - `nsource`: Source-component count forming one factor/intermediate row.
/// - `byte_budget`: Maximum nominal bytes for the paired `S/F` buffers.
/// # Returns
/// - `usize`: Target rows per streaming panel.
fn panel_rows(
    ntarget: usize,
    nsource: usize,
    byte_budget: usize,
) -> usize {
    if ntarget == 0 {
        return 1;
    }

    if nsource == 0 {
        return ntarget;
    }

    let bytes_per_row = nsource
        .checked_mul(2 * std::mem::size_of::<f64>())
        .expect("GPU panel row byte count overflow");

    (byte_budget / bytes_per_row).max(1).min(ntarget)
}

/// Reusable GPU one-body scratch buffers.
#[derive(Default)]
struct GpuOneBodyScratch {
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
    /// Ensure first-stage buffers can hold `len` entries.
    /// # Arguments:
    /// - `context`: CubeCL context used for allocation.
    /// - `len`: Required first-stage length.
    /// # Returns
    /// - `()`: Enlarges the buffers only when required.
    fn ensure_first(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.first_f, len);
        ensure_buffer(context, &mut self.first_s, len);
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

/// Convert a user MiB budget to bytes with overflow checking.
/// # Arguments:
/// - `mib`: Binary mebibyte count.
/// - `context`: Panic message used on overflow.
/// # Returns
/// - `usize`: Byte budget.
fn mib_to_bytes(
    mib: usize,
    context: &str,
) -> usize {
    mib.checked_mul(1024)
        .and_then(|value| value.checked_mul(1024))
        .expect(context)
}

/// Count bytes occupied by two row-major `f64` matrices.
/// # Arguments:
/// - `rows`: Matrix row count.
/// - `cols`: Matrix column count.
/// # Returns
/// - `usize`: Saturating paired allocation size in bytes.
fn paired_f64_bytes(
    rows: usize,
    cols: usize,
) -> usize {
    rows.saturating_mul(cols)
        .saturating_mul(2 * std::mem::size_of::<f64>())
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
