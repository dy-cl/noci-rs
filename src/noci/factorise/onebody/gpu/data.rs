// noci/factorise/onebody/gpu/data.rs
//! GPU-resident data layout for factorised one-body NOCI operator contractions.

// Crate-root imports.
use crate::gpu::{GpuBuffer, GpuContext};
use crate::noci::types::{NOCIData, NOCIScalar};

// Parent/sibling imports.
use super::super::super::SpinFactorisation;

/// Decoded spin excitation derived from the canonical CPU `u128` masks.
pub(crate) struct GpuDecodedExcitation {
    /// Excitation rank.
    pub(crate) rank: u16,
    /// Offset into the flattened hole-index array.
    pub(crate) holes_offset: usize,
    /// Offset into the flattened particle-index array.
    pub(crate) parts_offset: usize,
}

/// Persistent factorised-operator GPU topology.
pub(crate) struct GpuOneBodyData {
    /// Parent entry offsets.
    pub(crate) parent_entry_offsets: Vec<usize>,
    /// Parent alpha-representative offsets.
    pub(crate) parent_arep_offsets: Vec<usize>,
    /// Parent beta-representative offsets.
    pub(crate) parent_brep_offsets: Vec<usize>,
    /// Entry determinant indices.
    pub(crate) entry_det: Vec<usize>,
    /// Entry parent-local alpha IDs.
    pub(crate) entry_a: Vec<usize>,
    /// Entry parent-local beta IDs.
    pub(crate) entry_b: Vec<usize>,
    /// Flattened alpha representative determinant IDs.
    pub(crate) areps: Vec<usize>,
    /// Flattened beta representative determinant IDs.
    pub(crate) breps: Vec<usize>,
    /// Decoded alpha excitations keyed by determinant index.
    pub(crate) alpha: Vec<GpuDecodedExcitation>,
    /// Decoded beta excitations keyed by determinant index.
    pub(crate) beta: Vec<GpuDecodedExcitation>,
    /// Flattened alpha hole orbital indices.
    pub(crate) alpha_holes: Vec<u16>,
    /// Flattened alpha particle orbital indices.
    pub(crate) alpha_parts: Vec<u16>,
    /// Flattened beta hole orbital indices.
    pub(crate) beta_holes: Vec<u16>,
    /// Flattened beta particle orbital indices.
    pub(crate) beta_parts: Vec<u16>,
    /// Determinant alpha excitation phases.
    pub(crate) pha: Vec<f64>,
    /// Determinant beta excitation phases.
    pub(crate) phb: Vec<f64>,
    /// Maximum decoded alpha excitation rank.
    pub(crate) max_alpha_rank: usize,
    /// Maximum decoded beta excitation rank.
    pub(crate) max_beta_rank: usize,
    /// Per-parent alpha rank group offsets with stride `max_alpha_rank + 2`.
    pub(crate) parent_alpha_rank_offsets: Vec<usize>,
    /// Rank-sorted alpha representative determinant IDs.
    pub(crate) alpha_rank_rep_det: Vec<usize>,
    /// Original parent-local alpha component for each rank-sorted alpha representative.
    pub(crate) alpha_rank_component: Vec<usize>,
    /// Per-parent beta rank group offsets with stride `max_beta_rank + 2`.
    pub(crate) parent_beta_rank_offsets: Vec<usize>,
    /// Rank-sorted beta representative determinant IDs.
    pub(crate) beta_rank_rep_det: Vec<usize>,
    /// Original parent-local beta component for each rank-sorted beta representative.
    pub(crate) beta_rank_component: Vec<usize>,
    /// A-first CSR offsets keyed by source beta component for each parent.
    pub(crate) by_beta_offsets: Vec<usize>,
    /// Host CSR base into `by_beta_offsets` for each parent.
    pub(crate) by_beta_parent_offsets: Vec<usize>,
    /// A-first CSR determinant IDs.
    pub(crate) by_beta_det: Vec<usize>,
    /// A-first CSR source alpha component IDs.
    pub(crate) by_beta_alpha: Vec<usize>,
    /// B-first CSR offsets keyed by source alpha component for each parent.
    pub(crate) by_alpha_offsets: Vec<usize>,
    /// Host CSR base into `by_alpha_offsets` for each parent.
    pub(crate) by_alpha_parent_offsets: Vec<usize>,
    /// B-first CSR determinant IDs.
    pub(crate) by_alpha_det: Vec<usize>,
    /// B-first CSR source beta component IDs.
    pub(crate) by_alpha_beta: Vec<usize>,
}

/// Device-resident determinant topology and decoded excitation data.
pub(crate) struct DeviceOneBodyData {
    /// Parent entry offsets.
    pub(crate) parent_entry_offsets: GpuBuffer<u32>,
    /// Entry determinant indices.
    pub(crate) entry_det: GpuBuffer<u32>,
    /// Entry parent-local alpha IDs.
    pub(crate) entry_a: GpuBuffer<u32>,
    /// Entry parent-local beta IDs.
    pub(crate) entry_b: GpuBuffer<u32>,
    /// Alpha excitation rank keyed by determinant.
    pub(crate) alpha_rank: GpuBuffer<u32>,
    /// Alpha hole offset keyed by determinant.
    pub(crate) alpha_holes_offset: GpuBuffer<u32>,
    /// Alpha particle offset keyed by determinant.
    pub(crate) alpha_parts_offset: GpuBuffer<u32>,
    /// Flattened alpha hole orbital labels.
    pub(crate) alpha_holes: GpuBuffer<u32>,
    /// Flattened alpha particle orbital labels.
    pub(crate) alpha_parts: GpuBuffer<u32>,
    /// Beta excitation rank keyed by determinant.
    pub(crate) beta_rank: GpuBuffer<u32>,
    /// Beta hole offset keyed by determinant.
    pub(crate) beta_holes_offset: GpuBuffer<u32>,
    /// Beta particle offset keyed by determinant.
    pub(crate) beta_parts_offset: GpuBuffer<u32>,
    /// Flattened beta hole orbital labels.
    pub(crate) beta_holes: GpuBuffer<u32>,
    /// Flattened beta particle orbital labels.
    pub(crate) beta_parts: GpuBuffer<u32>,
    /// Determinant alpha excitation phases.
    pub(crate) pha: GpuBuffer<f64>,
    /// Determinant beta excitation phases.
    pub(crate) phb: GpuBuffer<f64>,
    /// Per-parent alpha rank group offsets.
    pub(crate) parent_alpha_rank_offsets: GpuBuffer<u32>,
    /// Rank-sorted alpha representative determinant IDs.
    pub(crate) alpha_rank_rep_det: GpuBuffer<u32>,
    /// Original parent-local alpha component for rank-sorted alpha representatives.
    pub(crate) alpha_rank_component: GpuBuffer<u32>,
    /// Per-parent beta rank group offsets.
    pub(crate) parent_beta_rank_offsets: GpuBuffer<u32>,
    /// Rank-sorted beta representative determinant IDs.
    pub(crate) beta_rank_rep_det: GpuBuffer<u32>,
    /// Original parent-local beta component for rank-sorted beta representatives.
    pub(crate) beta_rank_component: GpuBuffer<u32>,
    /// A-first CSR offsets keyed by source beta component for each parent.
    pub(crate) by_beta_offsets: GpuBuffer<u32>,
    /// A-first CSR determinant IDs.
    pub(crate) by_beta_det: GpuBuffer<u32>,
    /// A-first CSR source alpha component IDs.
    pub(crate) by_beta_alpha: GpuBuffer<u32>,
    /// B-first CSR offsets keyed by source alpha component for each parent.
    pub(crate) by_alpha_offsets: GpuBuffer<u32>,
    /// B-first CSR determinant IDs.
    pub(crate) by_alpha_det: GpuBuffer<u32>,
    /// B-first CSR source beta component IDs.
    pub(crate) by_alpha_beta: GpuBuffer<u32>,
}

impl GpuOneBodyData {
    /// Pack persistent determinant topology for CubeCL kernels.
    /// # Arguments:
    /// - `spin`: Shared determinant-space spin factorisation.
    /// - `data`: Shared NOCI data defining the candidate determinant basis and Wick views.
    /// # Returns
    /// - `GpuOneBodyData`: GPU factorised-operator data descriptor.
    pub(crate) fn new<T: NOCIScalar>(
        spin: &SpinFactorisation,
        data: &NOCIData<'_, T>,
    ) -> Self {
        let mut parent_entry_offsets = Vec::with_capacity(spin.parents.len() + 1);
        let mut parent_arep_offsets = Vec::with_capacity(spin.parents.len() + 1);
        let mut parent_brep_offsets = Vec::with_capacity(spin.parents.len() + 1);
        let mut entry_det = Vec::new();
        let mut entry_a = Vec::new();
        let mut entry_b = Vec::new();
        let mut areps = Vec::new();
        let mut breps = Vec::new();

        for parent in &spin.parents {
            parent_entry_offsets.push(entry_det.len());
            parent_arep_offsets.push(areps.len());
            parent_brep_offsets.push(breps.len());
            for entry in &parent.entries {
                entry_det.push(entry.det);
                entry_a.push(entry.a);
                entry_b.push(entry.b);
            }
            areps.extend_from_slice(&parent.areps);
            breps.extend_from_slice(&parent.breps);
        }
        parent_entry_offsets.push(entry_det.len());
        parent_arep_offsets.push(areps.len());
        parent_brep_offsets.push(breps.len());

        let mut alpha = Vec::with_capacity(data.basis.len());
        let mut beta = Vec::with_capacity(data.basis.len());
        let mut alpha_holes = Vec::new();
        let mut alpha_parts = Vec::new();
        let mut beta_holes = Vec::new();
        let mut beta_parts = Vec::new();
        let mut pha = Vec::with_capacity(data.basis.len());
        let mut phb = Vec::with_capacity(data.basis.len());

        for det in data.basis {
            pha.push(det.pha);
            phb.push(det.phb);
            alpha.push(decode_excitation(
                det.excitation.alpha.holes,
                det.excitation.alpha.parts,
                &mut alpha_holes,
                &mut alpha_parts,
            ));
            beta.push(decode_excitation(
                det.excitation.beta.holes,
                det.excitation.beta.parts,
                &mut beta_holes,
                &mut beta_parts,
            ));
        }

        let max_alpha_rank = alpha.iter().map(|ex| ex.rank as usize).max().unwrap_or(0);
        let max_beta_rank = beta.iter().map(|ex| ex.rank as usize).max().unwrap_or(0);
        let (parent_alpha_rank_offsets, alpha_rank_rep_det, alpha_rank_component) =
            build_rank_groups(&spin.parents, true, &alpha, max_alpha_rank);
        let (parent_beta_rank_offsets, beta_rank_rep_det, beta_rank_component) =
            build_rank_groups(&spin.parents, false, &beta, max_beta_rank);
        let (by_beta_offsets, by_beta_parent_offsets, by_beta_det, by_beta_alpha) =
            build_source_groups_by_beta(&spin.parents);
        let (by_alpha_offsets, by_alpha_parent_offsets, by_alpha_det, by_alpha_beta) =
            build_source_groups_by_alpha(&spin.parents);

        Self {
            parent_entry_offsets,
            parent_arep_offsets,
            parent_brep_offsets,
            entry_det,
            entry_a,
            entry_b,
            areps,
            breps,
            alpha,
            beta,
            alpha_holes,
            alpha_parts,
            beta_holes,
            beta_parts,
            pha,
            phb,
            max_alpha_rank,
            max_beta_rank,
            parent_alpha_rank_offsets,
            alpha_rank_rep_det,
            alpha_rank_component,
            parent_beta_rank_offsets,
            beta_rank_rep_det,
            beta_rank_component,
            by_beta_offsets,
            by_beta_parent_offsets,
            by_beta_det,
            by_beta_alpha,
            by_alpha_offsets,
            by_alpha_parent_offsets,
            by_alpha_det,
            by_alpha_beta,
        }
    }

    /// Upload determinant topology and decoded excitations to CubeCL device buffers.
    /// # Arguments:
    /// - `self`: Host topology data.
    /// - `context`: CubeCL context owning the target device.
    /// # Returns
    /// - `DeviceOneBodyData`: Device topology buffers.
    pub(crate) fn upload(
        &self,
        context: &GpuContext,
    ) -> DeviceOneBodyData {
        let alpha_rank = self
            .alpha
            .iter()
            .map(|ex| u32::from(ex.rank))
            .collect::<Vec<_>>();
        let alpha_holes_offset = self
            .alpha
            .iter()
            .map(|ex| checked_u32(ex.holes_offset))
            .collect::<Vec<_>>();
        let alpha_parts_offset = self
            .alpha
            .iter()
            .map(|ex| checked_u32(ex.parts_offset))
            .collect::<Vec<_>>();
        let beta_rank = self
            .beta
            .iter()
            .map(|ex| u32::from(ex.rank))
            .collect::<Vec<_>>();
        let beta_holes_offset = self
            .beta
            .iter()
            .map(|ex| checked_u32(ex.holes_offset))
            .collect::<Vec<_>>();
        let beta_parts_offset = self
            .beta
            .iter()
            .map(|ex| checked_u32(ex.parts_offset))
            .collect::<Vec<_>>();

        DeviceOneBodyData {
            parent_entry_offsets: upload_usize(context, &self.parent_entry_offsets),
            entry_det: upload_usize(context, &self.entry_det),
            entry_a: upload_usize(context, &self.entry_a),
            entry_b: upload_usize(context, &self.entry_b),
            alpha_rank: GpuBuffer::from_slice(context, &alpha_rank),
            alpha_holes_offset: GpuBuffer::from_slice(context, &alpha_holes_offset),
            alpha_parts_offset: GpuBuffer::from_slice(context, &alpha_parts_offset),
            alpha_holes: upload_u16(context, &self.alpha_holes),
            alpha_parts: upload_u16(context, &self.alpha_parts),
            beta_rank: GpuBuffer::from_slice(context, &beta_rank),
            beta_holes_offset: GpuBuffer::from_slice(context, &beta_holes_offset),
            beta_parts_offset: GpuBuffer::from_slice(context, &beta_parts_offset),
            beta_holes: upload_u16(context, &self.beta_holes),
            beta_parts: upload_u16(context, &self.beta_parts),
            pha: GpuBuffer::from_slice(context, &self.pha),
            phb: GpuBuffer::from_slice(context, &self.phb),
            parent_alpha_rank_offsets: upload_usize(context, &self.parent_alpha_rank_offsets),
            alpha_rank_rep_det: upload_usize(context, &self.alpha_rank_rep_det),
            alpha_rank_component: upload_usize(context, &self.alpha_rank_component),
            parent_beta_rank_offsets: upload_usize(context, &self.parent_beta_rank_offsets),
            beta_rank_rep_det: upload_usize(context, &self.beta_rank_rep_det),
            beta_rank_component: upload_usize(context, &self.beta_rank_component),
            by_beta_offsets: upload_usize(context, &self.by_beta_offsets),
            by_beta_det: upload_usize(context, &self.by_beta_det),
            by_beta_alpha: upload_usize(context, &self.by_beta_alpha),
            by_alpha_offsets: upload_usize(context, &self.by_alpha_offsets),
            by_alpha_det: upload_usize(context, &self.by_alpha_det),
            by_alpha_beta: upload_usize(context, &self.by_alpha_beta),
        }
    }
}

/// Decode canonical `u128` excitation masks into flattened orbital-index arrays.
/// # Arguments:
/// - `holes`: Canonical hole bit mask.
/// - `parts`: Canonical particle bit mask.
/// - `holes_out`: Flattened hole-index array to append to.
/// - `parts_out`: Flattened particle-index array to append to.
/// # Returns
/// - `GpuDecodedExcitation`: Offset and rank descriptor for this excitation.
fn decode_excitation(
    holes: u128,
    parts: u128,
    holes_out: &mut Vec<u16>,
    parts_out: &mut Vec<u16>,
) -> GpuDecodedExcitation {
    let holes_offset = holes_out.len();
    let parts_offset = parts_out.len();
    push_bits(holes, holes_out);
    push_bits(parts, parts_out);
    let rank = holes_out.len() - holes_offset;
    if rank != parts_out.len() - parts_offset {
        eprintln!("canonical excitation has different hole and particle ranks");
        std::process::exit(1);
    }
    GpuDecodedExcitation {
        rank: rank as u16,
        holes_offset,
        parts_offset,
    }
}

/// Append set-bit indices from one canonical `u128` mask.
/// # Arguments:
/// - `bits`: Canonical orbital bit mask.
/// - `out`: Output orbital-index array.
/// # Returns
/// - `()`: Appends one `u16` orbital index per set bit.
fn push_bits(
    mut bits: u128,
    out: &mut Vec<u16>,
) {
    while bits != 0 {
        let orbital = bits.trailing_zeros();
        if orbital > u16::MAX as u32 {
            eprintln!("GPU decoded excitation orbital index exceeds u16 storage");
            std::process::exit(1);
        }
        out.push(orbital as u16);
        bits &= bits - 1;
    }
}

/// Build rank-sorted representative groups for one spin sector.
/// # Arguments:
/// - `parents`: Parent-local spin spaces.
/// - `alpha`: Whether to group alpha or beta representatives.
/// - `decoded`: Decoded excitations keyed by determinant.
/// - `max_rank`: Largest rank in this spin sector.
/// # Returns
/// - `(Vec<usize>, Vec<usize>, Vec<usize>)`: Parent rank offsets, representative determinants and original components.
fn build_rank_groups(
    parents: &[super::super::super::ParentSpinSpace],
    alpha: bool,
    decoded: &[GpuDecodedExcitation],
    max_rank: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let stride = max_rank + 2;
    let mut offsets = Vec::with_capacity(parents.len() * stride);
    let mut rep_det = Vec::new();
    let mut component = Vec::new();

    for parent in parents {
        let reps = if alpha { &parent.areps } else { &parent.breps };
        for rank in 0..=max_rank {
            offsets.push(rep_det.len());
            for (comp, &det) in reps.iter().enumerate() {
                if decoded[det].rank as usize == rank {
                    rep_det.push(det);
                    component.push(comp);
                }
            }
        }
        offsets.push(rep_det.len());
    }

    (offsets, rep_det, component)
}

/// Build deterministic A-first source grouping by beta component for every parent.
/// # Arguments:
/// - `parents`: Parent-local spin spaces.
/// # Returns
/// - `(Vec<usize>, Vec<usize>, Vec<usize>)`: CSR offsets, determinant IDs and alpha components.
fn build_source_groups_by_beta(
    parents: &[super::super::super::ParentSpinSpace]
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut offsets = Vec::new();
    let mut parent_offsets = Vec::with_capacity(parents.len());
    let mut dets = Vec::new();
    let mut alphas = Vec::new();
    for parent in parents {
        parent_offsets.push(offsets.len());
        for b in 0..parent.breps.len() {
            offsets.push(dets.len());
            for entry in &parent.entries {
                if entry.b == b {
                    dets.push(entry.det);
                    alphas.push(entry.a);
                }
            }
        }
        offsets.push(dets.len());
    }
    (offsets, parent_offsets, dets, alphas)
}

/// Build deterministic B-first source grouping by alpha component for every parent.
/// # Arguments:
/// - `parents`: Parent-local spin spaces.
/// # Returns
/// - `(Vec<usize>, Vec<usize>, Vec<usize>)`: CSR offsets, determinant IDs and beta components.
fn build_source_groups_by_alpha(
    parents: &[super::super::super::ParentSpinSpace]
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut offsets = Vec::new();
    let mut parent_offsets = Vec::with_capacity(parents.len());
    let mut dets = Vec::new();
    let mut betas = Vec::new();
    for parent in parents {
        parent_offsets.push(offsets.len());
        for a in 0..parent.areps.len() {
            offsets.push(dets.len());
            for entry in &parent.entries {
                if entry.a == a {
                    dets.push(entry.det);
                    betas.push(entry.b);
                }
            }
        }
        offsets.push(dets.len());
    }
    (offsets, parent_offsets, dets, betas)
}

/// Upload checked `usize` metadata as `u32`.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `values`: Host values.
/// # Returns
/// - `GpuBuffer<u32>`: Device buffer.
fn upload_usize(
    context: &GpuContext,
    values: &[usize],
) -> GpuBuffer<u32> {
    let values = values.iter().copied().map(checked_u32).collect::<Vec<_>>();
    GpuBuffer::from_slice(context, &values)
}

/// Upload checked `u16` orbital labels as `u32`.
/// # Arguments:
/// - `context`: CubeCL context.
/// - `values`: Host orbital labels.
/// # Returns
/// - `GpuBuffer<u32>`: Device buffer.
fn upload_u16(
    context: &GpuContext,
    values: &[u16],
) -> GpuBuffer<u32> {
    let values = values.iter().copied().map(u32::from).collect::<Vec<_>>();
    GpuBuffer::from_slice(context, &values)
}

/// Convert `usize` metadata to `u32`, failing before any silent truncation.
/// # Arguments:
/// - `value`: Host metadata value.
/// # Returns
/// - `u32`: Checked device-width value.
fn checked_u32(value: usize) -> u32 {
    u32::try_from(value).expect("GPU one-body topology exceeds u32 device representation")
}
