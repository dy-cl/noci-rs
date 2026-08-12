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
use super::super::plan::{
    OneBodyBlockPlan, OneBodyContraction, OneBodyPlan, select_one_body_contraction,
};
use super::contract::{
    AFirstFinalLaunch, AFirstStageLaunch, BFirstFinalLaunch, BFirstStageLaunch,
    launch_a_first_final, launch_a_first_stage, launch_b_first_final, launch_b_first_stage,
    launch_zero_f64,
};
use super::data::{DeviceOneBodyData, GpuOneBodyData};
use super::diagonals::{DiagonalBlockLaunch, launch_fill_one_body_diagonal_block};
use super::factors::{FactorOutput, FactorRequest, build_spin_one_body_factors};
use super::orthogonal::{
    GpuOrthogonalBlocks, launch_apply_orthogonal_block, launch_fill_orthogonal_diagonal,
};

/// Target component panel width, matching the CPU transient contraction path initially.
const PANEL_ROWS: usize = 512;

/// CubeCL factorised one-body backend for the current generalised Fock.
pub(crate) struct GpuOneBodyBackend<T: NOCIScalar> {
    /// Common CubeCL context descriptor.
    context: GpuContext,
    /// Shared determinant-space factorisation `I <-> (P,a_I,b_I)`.
    spin: SpinFactorisation,
    /// Shared one-body topology and contraction plan.
    plan: OneBodyPlan,
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
    /// Marker preserving the scalar type checked at construction.
    marker: PhantomData<T>,
}

impl<T: NOCIScalar + 'static> GpuOneBodyBackend<T> {
    /// Build GPU-resident data for the current generalised Fock operator.
    /// # Arguments:
    /// - `data`: Shared NOCI data with Wick intermediates for the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data, already reflected in Wick intermediates.
    /// - `cache`: Directory for persistent file-backed factor blocks.
    /// - `rank`: MPI rank used in factor-cache filenames.
    /// - `iteration`: SNOCI iteration used in factor-cache filenames.
    /// - `storage`: Requested persistent factor-table storage backend.
    /// # Returns
    /// - `GpuOneBodyBackend<T>`: GPU one-body backend descriptor.
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
        let gpu_data = GpuOneBodyData::new(&spin, data);
        let context = GpuContext::new();
        let orthogonal = GpuOrthogonalBlocks::new(&context, &spin, data, fock);
        let device_wicks = upload_real_wicks(&wicks, &context);
        let device_data = gpu_data.upload(&context);
        Self {
            context,
            spin,
            plan,
            device_wicks,
            data: gpu_data,
            device_data,
            orthogonal,
            scratch: GpuOneBodyScratch::default(),
            marker: PhantomData,
        }
    }

    /// Apply `Y = (F + \lambda S)x` using GPU-resident factor generation and contractions.
    /// The intended arithmetic is `Y^Q += F^alpha D (S^beta)^T
    /// + S^alpha D (F^beta + \lambda S^beta)^T` for A-first blocks and the corresponding
    /// beta-first expression for B-first blocks.
    /// # Arguments:
    /// - `x`: Source vector over actual candidate determinants.
    /// - `data`: Shared NOCI data used by same-parent orthogonal blocks.
    /// - `fock`: Current generalised-Fock data used by same-parent orthogonal blocks.
    /// - `lambda`: Scalar shift multiplying the overlap operator.
    /// - `partition`: Worker index and worker count for target rows.
    /// # Returns
    /// - `Array1<T>`: Partial or complete determinant-space result vector.
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
        let y = self.scratch.y.as_ref().expect("GPU y buffer must exist");
        launch_zero_f64(&self.context, y, x.len());

        for index in 0..self.plan.blocks.len() {
            match &self.plan.blocks[index] {
                OneBodyBlockPlan::Orthogonal { parent } => {
                    let block = self.orthogonal.block(*parent);

                    launch_apply_orthogonal_block(
                        &self.context,
                        block,
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

    /// Build diagonal entries of `F + \lambda S` and `S` using GPU one-body arithmetic.
    /// # Arguments:
    /// - `data`: Shared NOCI data used by same-parent orthogonal blocks.
    /// - `fock`: Current generalised-Fock data used by same-parent orthogonal blocks.
    /// - `lambda`: Scalar overlap shift in `F + \lambda S`.
    /// # Returns
    /// - `(Array1<T>, Array1<T>)`: Diagonal of `F + \lambda S` and diagonal of `S`.
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

    /// Convert one plan entry into the generic GPU factorised block representation.
    /// # Arguments:
    /// - `index`: Plan index `Q * nparent + P`.
    /// # Returns
    /// - `GpuFactorisedBlock`: Parent-pair dimensions and contraction order.
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
            } => GpuFactorisedBlock {
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
            },
            OneBodyBlockPlan::Orthogonal { .. } => {
                unreachable!("orthogonal GPU blocks must use Slater-Condon")
            }
        }
    }

    /// Apply one generic factorised parent-pair block.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `lambda`: Overlap shift.
    /// - `partition`: Worker id and worker count.
    /// # Returns
    /// - `()`: Accumulates the block into device `y`.
    fn apply_factorised_block(
        &mut self,
        block: GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
    ) {
        match block.contraction {
            OneBodyContraction::AFirst => self.apply_a_first_block(block, lambda, partition),
            OneBodyContraction::BFirst => self.apply_b_first_block(block, lambda, partition),
        }
    }

    /// Apply one alpha-first GPU parent-pair block.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `lambda`: Overlap shift.
    /// - `partition`: Worker id and worker count.
    /// # Returns
    /// - `()`: Accumulates the block into device `y`.
    fn apply_a_first_block(
        &mut self,
        block: GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
    ) {
        let beta_len = checked_mul(block.ntb, block.nsb, "GPU beta factor length");
        self.scratch.ensure_factor(&self.context, beta_len);
        self.build_factors(&block, false, 0, block.ntb, block.nsb, FactorSlot::Full);
        for a0 in (0..block.nta).step_by(PANEL_ROWS) {
            let a1 = (a0 + PANEL_ROWS).min(block.nta);
            let nrow = a1 - a0;
            self.scratch.ensure_panel(
                &self.context,
                checked_mul(nrow, block.nsa, "GPU alpha panel factor length"),
            );
            self.scratch.ensure_first(
                &self.context,
                checked_mul(nrow, block.nsb, "GPU alpha-first intermediate length"),
            );
            self.build_factors(&block, true, a0, a1, block.nsa, FactorSlot::Panel);
            self.launch_a_first_panel(&block, lambda, partition, a0, a1);
        }
    }

    /// Apply one beta-first GPU parent-pair block.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `lambda`: Overlap shift.
    /// - `partition`: Worker id and worker count.
    /// # Returns
    /// - `()`: Accumulates the block into device `y`.
    fn apply_b_first_block(
        &mut self,
        block: GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
    ) {
        let alpha_len = checked_mul(block.nta, block.nsa, "GPU alpha factor length");
        self.scratch.ensure_factor(&self.context, alpha_len);
        self.build_factors(&block, true, 0, block.nta, block.nsa, FactorSlot::Full);
        for b0 in (0..block.ntb).step_by(PANEL_ROWS) {
            let b1 = (b0 + PANEL_ROWS).min(block.ntb);
            let nrow = b1 - b0;
            self.scratch.ensure_panel(
                &self.context,
                checked_mul(nrow, block.nsb, "GPU beta panel factor length"),
            );
            self.scratch.ensure_first(
                &self.context,
                checked_mul(nrow, block.nsa, "GPU beta-first intermediate length"),
            );
            self.build_factors(&block, false, b0, b1, block.nsb, FactorSlot::Panel);
            self.launch_b_first_panel(&block, lambda, partition, b0, b1);
        }
    }

    /// Build one spin factor table or panel into a selected scratch slot.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `alpha`: Whether to build alpha or beta factors.
    /// - `row0`: First target component represented by row zero.
    /// - `row1`: One-past-last target component.
    /// - `nsource`: Full source component count for this spin.
    /// - `slot`: Scratch factor slot.
    /// # Returns
    /// - `()`: Writes factors on device.
    fn build_factors(
        &self,
        block: &GpuFactorisedBlock,
        alpha: bool,
        row0: usize,
        row1: usize,
        nsource: usize,
        slot: FactorSlot,
    ) {
        let (s, f) = self.scratch.factor_slot(slot);
        build_spin_one_body_factors(
            &self.context,
            &self.device_wicks,
            &self.device_data,
            &self.data,
            FactorRequest {
                target_parent: block.target_parent,
                source_parent: block.source_parent,
                lp: block.lp,
                gp: block.gp,
                nref: self.plan.nparent,
                target_left: block.target_left,
                alpha,
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

    /// Launch alpha-first stage and final kernels for one alpha panel.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `lambda`: Overlap shift.
    /// - `partition`: Worker id and worker count.
    /// - `a0`: First target alpha component in the panel.
    /// - `a1`: One-past-last target alpha component in the panel.
    /// # Returns
    /// - `()`: Accumulates the panel into device `y`.
    fn launch_a_first_panel(
        &self,
        block: &GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
        a0: usize,
        a1: usize,
    ) {
        let (worker, nworker) = partition;
        let (sa, fa) = self.scratch.panel_factors();
        let (sb, fb) = self.scratch.full_factors();
        let (tf, ts) = self.scratch.first_buffers();
        launch_zero_f64(&self.context, tf, (a1 - a0) * block.nsb);
        launch_zero_f64(&self.context, ts, (a1 - a0) * block.nsb);
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
                target_component_base: a0,
                target_component_end: a1,
                lambda,
                worker,
                nworker,
            },
        );
    }

    /// Launch beta-first stage and final kernels for one beta panel.
    /// # Arguments:
    /// - `block`: Parent-pair block descriptor.
    /// - `lambda`: Overlap shift.
    /// - `partition`: Worker id and worker count.
    /// - `b0`: First target beta component in the panel.
    /// - `b1`: One-past-last target beta component in the panel.
    /// # Returns
    /// - `()`: Accumulates the panel into device `y`.
    fn launch_b_first_panel(
        &self,
        block: &GpuFactorisedBlock,
        lambda: f64,
        partition: (usize, usize),
        b0: usize,
        b1: usize,
    ) {
        let (worker, nworker) = partition;
        let (sa, fa) = self.scratch.full_factors();
        let (sb, fb) = self.scratch.panel_factors();
        let (uf, us) = self.scratch.first_buffers();
        launch_zero_f64(&self.context, uf, (b1 - b0) * block.nsa);
        launch_zero_f64(&self.context, us, (b1 - b0) * block.nsa);
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
                target_component_base: b0,
                target_component_end: b1,
                lambda,
                worker,
                nworker,
            },
        );
    }

    /// Fill diagonals for one same-parent block using generic Wick factors.
    /// # Arguments:
    /// - `parent`: Parent reference.
    /// - `lambda`: Overlap shift.
    /// # Returns
    /// - `()`: Writes parent determinant diagonals on device.
    fn fill_parent_diagonal(
        &mut self,
        parent: usize,
        lambda: f64,
    ) {
        let nta = self.spin.parents[parent].areps.len();
        let ntb = self.spin.parents[parent].breps.len();
        let (lp, gp, target_left) = ordered_parent_pair(&self.spin, parent, parent);
        let block = GpuFactorisedBlock {
            target_parent: parent,
            source_parent: parent,
            lp,
            gp,
            target_left,
            nta,
            ntb,
            nsa: nta,
            nsb: ntb,
            contraction: OneBodyContraction::AFirst,
        };
        self.scratch.ensure_factor(
            &self.context,
            checked_mul(nta, nta, "GPU alpha diagonal factors"),
        );
        self.build_factors(&block, true, 0, nta, nta, FactorSlot::Full);
        self.scratch.ensure_panel(
            &self.context,
            checked_mul(ntb, ntb, "GPU beta diagonal factors"),
        );
        self.build_factors(&block, false, 0, ntb, ntb, FactorSlot::Panel);
        let (sa, fa) = self.scratch.full_factors();
        let (sb, fb) = self.scratch.panel_factors();
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
                nsa: nta,
                nsb: ntb,
                lambda,
            },
        );
    }
}

/// Generic factorised GPU block, including same-parent blocks temporarily routed through Wick.
#[derive(Clone, Copy)]
struct GpuFactorisedBlock {
    /// Target parent `Q`.
    target_parent: usize,
    /// Source parent `P`.
    source_parent: usize,
    /// Ordered left parent.
    lp: usize,
    /// Ordered greater parent.
    gp: usize,
    /// Whether target parent is the left determinant.
    target_left: bool,
    /// Target alpha component count.
    nta: usize,
    /// Target beta component count.
    ntb: usize,
    /// Source alpha component count.
    nsa: usize,
    /// Source beta component count.
    nsb: usize,
    /// Dense contraction order.
    contraction: OneBodyContraction,
}

/// Scratch factor slot.
#[derive(Clone, Copy)]
enum FactorSlot {
    /// Full opposite-spin factor table.
    Full,
    /// Target-spin panel factor table.
    Panel,
}

/// Reusable GPU one-body buffers.
#[derive(Default)]
struct GpuOneBodyScratch {
    /// Full same-spin overlap factor table.
    factor_s: Option<GpuBuffer<f64>>,
    /// Full same-spin Fock factor table.
    factor_f: Option<GpuBuffer<f64>>,
    /// Target panel overlap factor table.
    panel_s: Option<GpuBuffer<f64>>,
    /// Target panel Fock factor table.
    panel_f: Option<GpuBuffer<f64>>,
    /// First-stage Fock intermediate.
    first_f: Option<GpuBuffer<f64>>,
    /// First-stage overlap intermediate.
    first_s: Option<GpuBuffer<f64>>,
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
    /// Ensure full factor buffers can hold `len` entries.
    /// # Arguments:
    /// - `self`: Reusable scratch buffers.
    /// - `context`: CubeCL context used for any allocation.
    /// - `len`: Required logical factor length.
    /// # Returns
    /// - `()`: Allocates larger buffers only when current capacity is insufficient.
    fn ensure_factor(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.factor_s, len);
        ensure_buffer(context, &mut self.factor_f, len);
    }

    /// Ensure panel factor buffers can hold `len` entries.
    /// # Arguments:
    /// - `self`: Reusable scratch buffers.
    /// - `context`: CubeCL context used for any allocation.
    /// - `len`: Required logical panel factor length.
    /// # Returns
    /// - `()`: Allocates larger buffers only when current capacity is insufficient.
    fn ensure_panel(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.panel_s, len);
        ensure_buffer(context, &mut self.panel_f, len);
    }

    /// Ensure first-stage buffers can hold `len` entries.
    /// # Arguments:
    /// - `self`: Reusable scratch buffers.
    /// - `context`: CubeCL context used for any allocation.
    /// - `len`: Required logical first-stage length.
    /// # Returns
    /// - `()`: Allocates larger buffers only when current capacity is insufficient.
    fn ensure_first(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.first_f, len);
        ensure_buffer(context, &mut self.first_s, len);
    }

    /// Ensure the output vector can hold `len` entries.
    /// # Arguments:
    /// - `self`: Reusable scratch buffers.
    /// - `context`: CubeCL context used for any allocation.
    /// - `len`: Required determinant-vector length.
    /// # Returns
    /// - `()`: Allocates a larger output buffer only when current capacity is insufficient.
    fn ensure_y(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.y, len);
    }

    /// Ensure the shifted-matrix diagonal buffer can hold `len` entries.
    /// # Arguments:
    /// - `self`: Reusable scratch buffers.
    /// - `context`: CubeCL context used for any allocation.
    /// - `len`: Required diagonal length.
    /// # Returns
    /// - `()`: Allocates a larger shifted-diagonal buffer only when current capacity is insufficient.
    fn ensure_m_diag(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.m_diag, len);
    }

    /// Ensure the overlap diagonal buffer can hold `len` entries.
    /// # Arguments:
    /// - `self`: Reusable scratch buffers.
    /// - `context`: CubeCL context used for any allocation.
    /// - `len`: Required diagonal length.
    /// # Returns
    /// - `()`: Allocates a larger overlap-diagonal buffer only when current capacity is insufficient.
    fn ensure_s_diag(
        &mut self,
        context: &GpuContext,
        len: usize,
    ) {
        ensure_buffer(context, &mut self.s_diag, len);
    }

    /// Borrow full factor buffers.
    /// # Arguments:
    /// - `self`: Reusable scratch buffers with full factor slots initialised.
    /// # Returns
    /// - `(&GpuBuffer<f64>, &GpuBuffer<f64>)`: Overlap and Fock factor buffers.
    fn full_factors(&self) -> (&GpuBuffer<f64>, &GpuBuffer<f64>) {
        (
            self.factor_s.as_ref().expect("GPU factor S buffer missing"),
            self.factor_f.as_ref().expect("GPU factor F buffer missing"),
        )
    }

    /// Borrow panel factor buffers.
    /// # Arguments:
    /// - `self`: Reusable scratch buffers with panel factor slots initialised.
    /// # Returns
    /// - `(&GpuBuffer<f64>, &GpuBuffer<f64>)`: Overlap and Fock panel buffers.
    fn panel_factors(&self) -> (&GpuBuffer<f64>, &GpuBuffer<f64>) {
        (
            self.panel_s.as_ref().expect("GPU panel S buffer missing"),
            self.panel_f.as_ref().expect("GPU panel F buffer missing"),
        )
    }

    /// Borrow first-stage buffers.
    /// # Arguments:
    /// - `self`: Reusable scratch buffers with first-stage slots initialised.
    /// # Returns
    /// - `(&GpuBuffer<f64>, &GpuBuffer<f64>)`: Fock and overlap first-stage buffers.
    fn first_buffers(&self) -> (&GpuBuffer<f64>, &GpuBuffer<f64>) {
        (
            self.first_f.as_ref().expect("GPU first F buffer missing"),
            self.first_s.as_ref().expect("GPU first S buffer missing"),
        )
    }

    /// Borrow selected factor slot.
    /// # Arguments:
    /// - `self`: Reusable scratch buffers.
    /// - `slot`: Full-table or panel-table selector.
    /// # Returns
    /// - `(&GpuBuffer<f64>, &GpuBuffer<f64>)`: Overlap and Fock buffers for the selected slot.
    fn factor_slot(
        &self,
        slot: FactorSlot,
    ) -> (&GpuBuffer<f64>, &GpuBuffer<f64>) {
        match slot {
            FactorSlot::Full => self.full_factors(),
            FactorSlot::Panel => self.panel_factors(),
        }
    }
}

/// Ensure a device buffer exists with at least `len` entries.
/// # Arguments:
/// - `context`: CubeCL context used for allocation.
/// - `buffer`: Optional buffer slot to resize.
/// - `len`: Required logical length.
/// # Returns
/// - `()`: Replaces `buffer` only when it is absent or too small.
fn ensure_buffer(
    context: &GpuContext,
    buffer: &mut Option<GpuBuffer<f64>>,
    len: usize,
) {
    if buffer.as_ref().map_or(true, |buf| buf.len() < len) {
        *buffer = Some(GpuBuffer::empty(context, len));
    }
}

/// Checked multiplication for launch sizes and buffer lengths.
/// # Arguments:
/// - `lhs`: Left factor.
/// - `rhs`: Right factor.
/// - `context`: Panic message context.
/// # Returns
/// - `usize`: Product `lhs * rhs`.
fn checked_mul(
    lhs: usize,
    rhs: usize,
    context: &str,
) -> usize {
    lhs.checked_mul(rhs).expect(context)
}

/// Reinterpret a scalar after the backend constructor has enforced `T = f64`.
/// # Arguments:
/// - `value`: Generic NOCI scalar known to be an `f64`.
/// # Returns
/// - `f64`: Reinterpreted real scalar.
fn real_scalar<T: NOCIScalar + 'static>(value: T) -> f64 {
    if TypeId::of::<T>() != TypeId::of::<f64>() {
        eprintln!("snoci.backend = \"gpu\" currently supports real f64 NOCI-PT2 data only");
        std::process::exit(1);
    }
    let ptr = &value as *const T as *const f64;
    // SAFETY: every GPU backend instance is constructed only after `TypeId::<T>() == TypeId::<f64>()`.
    unsafe { *ptr }
}

/// Reinterpret a contiguous scalar slice after the backend constructor has enforced `T = f64`.
/// # Arguments:
/// - `values`: Contiguous generic NOCI scalar slice known to contain `f64` elements.
/// # Returns
/// - `&[f64]`: Real slice over the same storage.
fn real_slice<T: NOCIScalar + 'static>(values: &[T]) -> &[f64] {
    if TypeId::of::<T>() != TypeId::of::<f64>() {
        eprintln!("snoci.backend = \"gpu\" currently supports real f64 NOCI-PT2 data only");
        std::process::exit(1);
    }
    let ptr = values.as_ptr() as *const f64;
    // SAFETY: every GPU backend instance is constructed only after `TypeId::<T>() == TypeId::<f64>()`,
    // so the element layout and slice length are exactly those of `[f64]`.
    unsafe { std::slice::from_raw_parts(ptr, values.len()) }
}

/// Upload host-packed Wick storage after the caller has proven `T = f64`.
/// # Arguments:
/// - `wicks`: Host-packed Wick storage with real scalar layout.
/// - `context`: CubeCL context owning the target device.
/// # Returns
/// - `DeviceWicksShared`: Device Wick buffers.
fn upload_real_wicks<T: NOCIScalar + 'static>(
    wicks: &gpu::WicksShared<T>,
    context: &GpuContext,
) -> DeviceWicksShared {
    let real = wicks as *const gpu::WicksShared<T> as *const gpu::WicksShared<f64>;
    // SAFETY: `GpuOneBodyBackend::new` calls this only after `TypeId::<T>() == TypeId::<f64>()`.
    // The cast does not assume compatibility between distinct scalar layouts at runtime; it only removes
    // generic typing from an already-real `WicksShared<f64>`.
    let real = unsafe { &*real };
    real.upload_f64(context)
}
