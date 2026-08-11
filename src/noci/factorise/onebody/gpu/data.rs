// noci/factorise/onebody/gpu/data.rs
//! GPU-resident data layout for factorised one-body NOCI operator contractions.

// Crate-root imports.
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

        for det in data.basis {
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
        out.push(orbital as u16);
        bits &= bits - 1;
    }
}
